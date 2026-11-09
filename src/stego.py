"""LSB substitution in float32 weights"""

import zlib

import numpy as np
import torch
import torch.nn as nn

HEADER_BITS = 64
MANTISSA_BITS = 23


class ChecksumMismatch(Exception):
    pass


def get_carriers(model):
    return [m.weight.data for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def carrier_names(model):
    return [n + ".weight" for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


# Bits and header

def _bytes_to_bits(data):
    return torch.from_numpy(np.unpackbits(np.frombuffer(data, np.uint8)).copy())


def _bits_to_int(bits):
    return int.from_bytes(np.packbits(bits.numpy()).tobytes(), "big")


def _int_to_bits(value, width):
    return _bytes_to_bits(value.to_bytes(width // 8, "big"))


# [length][CRC32][payload] as a bit tensor
def build_bitstream(payload):
    length = _int_to_bits(len(payload), 32)
    crc = _int_to_bits(zlib.crc32(payload), 32)
    return torch.cat([length, crc, _bytes_to_bits(payload)])


def verify_crc(payload, crc):
    if zlib.crc32(payload) != crc:
        raise ChecksumMismatch
    return payload


# Payload bytes from a bitstream
def _payload(bits, length):
    return np.packbits(bits[HEADER_BITS:].numpy()).tobytes()[:length]


# Capacity

# Effective capacity: raw bits minus the header, in bytes
def _to_bytes(bits):
    return max(0, (bits - HEADER_BITS) // 8)


# Raw capacity with n_lsb bits in every weight
def capacity_bits(carriers, n_lsb):
    return sum(t.numel() for t in carriers) * n_lsb


def capacity_bytes(carriers, n_lsb):
    return _to_bytes(capacity_bits(carriers, n_lsb))


# Raw capacity of a bit map
def capacity_bits_map(bit_map):
    return int(sum(int(b.sum()) for b in bit_map))


def capacity_bytes_map(bit_map):
    return _to_bytes(capacity_bits_map(bit_map))


# Read / write

# Bit map
def _flat_map(carriers, bit_map):
    flat = torch.cat([b.reshape(-1).to(torch.int64) for b in bit_map])
    assert flat.numel() == sum(t.numel() for t in carriers)
    assert 0 <= flat.min() and flat.max() <= MANTISSA_BITS
    return flat


# Uniform bit map: n_lsb in every weight
def _const_map(carriers, n_lsb):
    assert 1 <= n_lsb <= MANTISSA_BITS
    return [torch.full(t.shape, n_lsb, dtype=torch.int64) for t in carriers]

def _flat_int_view(carriers):
    chunks = [t.detach().cpu().contiguous().view(torch.int32).reshape(-1) for t in carriers]
    flat = torch.cat([c.to(torch.int64) for c in chunks]) 

    def write_back(new):
        off = 0
        for t, c in zip(carriers, chunks):
            c[:] = new[off:off + c.numel()].to(torch.int32)
            off += c.numel()
            if c.device != t.device:
                t.copy_(c.view(t.dtype).reshape(t.shape))
    return flat, write_back


# Groups of weights with the same n_lsb
def _groups(flat_map, n_bits):
    start = torch.cumsum(flat_map, 0) - flat_map
    for n_lsb in flat_map.unique().tolist():
        if n_lsb == 0:
            continue
        idx = (flat_map == n_lsb).nonzero(as_tuple=True)[0]
        idx = idx[start[idx] < n_bits]  # only weights that hold bitstream bits
        if idx.numel():
            pos = start[idx].unsqueeze(1) + torch.arange(n_lsb)  # bitstream positions of each weight
            yield n_lsb, idx, pos


# Write the bitstream into the low bits of the carriers
def _write(carriers, bits, flat_map):
    flat, write_back = _flat_int_view(carriers)
    padded = torch.zeros(len(bits) + MANTISSA_BITS, dtype=torch.uint8)
    padded[:len(bits)] = bits
    start = torch.cumsum(flat_map, 0) - flat_map
    for n_lsb, idx, pos in _groups(flat_map, len(bits)):
        powers = 1 << torch.arange(n_lsb - 1, -1, -1)
        values = (padded[pos].to(torch.int64) * powers).sum(1)

        # Only overwrite the bits with the payload
        valid = (len(bits) - start[idx]).clamp_(0, n_lsb)
        mask = ((1 << valid) - 1) << (n_lsb - valid)
        flat[idx] = (flat[idx] & ~mask) | values
    write_back(flat)


# Read from bitstream
def _read(carriers, n_bits, flat_map):
    assert n_bits <= flat_map.sum()
    flat, _ = _flat_int_view(carriers)
    out = torch.zeros(n_bits + MANTISSA_BITS, dtype=torch.uint8)
    for n_lsb, idx, pos in _groups(flat_map, n_bits):
        shifts = torch.arange(n_lsb - 1, -1, -1)
        bits = (flat[idx].unsqueeze(1) >> shifts) & 1
        out[pos.reshape(-1)] = bits.to(torch.uint8).reshape(-1)
    return out[:n_bits]


# Embed / extract

# Embed
def embed_map(carriers, payload, bit_map):
    flat = _flat_map(carriers, bit_map)
    bits = build_bitstream(payload)
    cap = int(flat.sum())
    assert len(bits) <= cap, f"payload of {len(bits)} bits, capacity {cap}"
    _write(carriers, bits, flat)
    return {
        "payload_bytes": len(payload),
        "bits_used": len(bits),
        "capacity_bits": cap,
        "capacity_bytes": _to_bytes(cap),
        "utilization": len(bits) / cap,
        "bits_mean": float(flat.to(torch.float64).mean()),
    }


# Extract and verify CRC
def extract_map(carriers, bit_map):
    flat = _flat_map(carriers, bit_map)
    header = _read(carriers, HEADER_BITS, flat)
    length = _bits_to_int(header[:32])
    crc = _bits_to_int(header[32:])
    if length > _to_bytes(int(flat.sum())):
        raise ChecksumMismatch 
    bits = _read(carriers, HEADER_BITS + length * 8, flat)
    return verify_crc(_payload(bits, length), crc)


def extract_raw_map(carriers, length, bit_map):
    flat = _flat_map(carriers, bit_map)
    return _payload(_read(carriers, HEADER_BITS + length * 8, flat), length)

def embed(carriers, payload, n_lsb):
    info = embed_map(carriers, payload, _const_map(carriers, n_lsb))
    info["n_lsb"] = n_lsb
    return info


def extract(carriers, n_lsb):
    return extract_map(carriers, _const_map(carriers, n_lsb))


def extract_raw(carriers, length, n_lsb):
    return extract_raw_map(carriers, length, _const_map(carriers, n_lsb))



# Bit error rate
def ber(original, recovered):
    rec = recovered[:len(original)].ljust(len(original), b"\x00")
    a = np.unpackbits(np.frombuffer(original, np.uint8))
    b = np.unpackbits(np.frombuffer(rec, np.uint8))
    return float((a != b).mean()) if len(a) else 0.0


# Weight distortion
def weight_distortion(clean_state, stego_state, names=None):
    names = names or [n for n in clean_state if n in stego_state]
    d = torch.cat([(stego_state[n].float() - clean_state[n].float()).abs().reshape(-1) for n in names])
    return {
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "rmse": float((d ** 2).mean().sqrt()),
        "frac_changed": float((d > 0).float().mean()),
    }
