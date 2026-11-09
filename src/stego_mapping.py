"""Sign mapping and value mapping"""

import torch

import stego
from stego import HEADER_BITS, ChecksumMismatch, _bits_to_int, _payload, build_bitstream, verify_crc

MAX_SYMBOL_BITS = 16 
DEFAULT_DELTA = 0.01
SIGN_BITS_PER_WEIGHT = 1


def _n(carriers):
    return sum(t.numel() for t in carriers)


# Length and CRC from the header
def _read_header(bits, cap_bytes):
    length = _bits_to_int(bits[:32])
    crc = _bits_to_int(bits[32:64])
    if length > cap_bytes:
        raise ChecksumMismatch
    return length, crc


# All carriers as a float64 tensor, stored in carriers
def _flat_float_view(carriers):
    chunks = [t.detach().cpu().reshape(-1) for t in carriers]
    flat = torch.cat([c.to(torch.float64) for c in chunks]) 

    def write_back(new):
        off = 0
        for t, c in zip(carriers, chunks):
            c[:] = new[off:off + c.numel()].to(c.dtype)
            off += c.numel()
            if c.device != t.device:
                t.copy_(c.reshape(t.shape))
    return flat, write_back


# Sign mapping

def sign_capacity_bits(carriers):
    return _n(carriers)


def sign_capacity_bytes(carriers):
    return stego._to_bytes(sign_capacity_bits(carriers))


# Sign bits of the first weights
def _sign_read(carriers, n_bits):
    flat, _ = stego._flat_int_view(carriers)
    return ((flat[:n_bits] >> 31) & 1).to(torch.uint8)


def sign_embed(carriers, payload):
    bits = build_bitstream(payload)
    cap = sign_capacity_bits(carriers)
    assert len(bits) <= cap, f"payload of {len(bits)} bits, capacity {cap}"
    before = _sign_read(carriers, len(bits))
    flat, write_back = stego._flat_int_view(carriers)
    n = len(bits)

    # One bit per weight in bit 31
    flat[:n] = (flat[:n] & 0x7FFFFFFF) | (bits.to(torch.int64) << 31)
    write_back(flat)
    return {
        "payload_bytes": len(payload),
        "bits_used": len(bits),
        "capacity_bits": cap,
        "capacity_bytes": sign_capacity_bytes(carriers),
        "utilization": len(bits) / cap,
        "frac_signs_flipped": float((before != bits).float().mean()),
    }


def sign_extract(carriers):
    length, crc = _read_header(_sign_read(carriers, HEADER_BITS), sign_capacity_bytes(carriers))
    bits = _sign_read(carriers, HEADER_BITS + length * 8)
    return verify_crc(_payload(bits, length), crc)


def sign_extract_raw(carriers, length):
    return _payload(_sign_read(carriers, HEADER_BITS + length * 8), length)


# Value mapping

def value_capacity_bits(carriers, k):
    return _n(carriers) * k


def value_capacity_bytes(carriers, k):
    return stego._to_bytes(value_capacity_bits(carriers, k))


# Largest shift a weight tolerates before symbol changes
def decision_margin(k, delta):
    return delta / 2 ** (k + 1)


# Bitstream to k-bit symbols
def _bits_to_symbols(bits, k):
    bits = torch.cat([bits, torch.zeros((-len(bits)) % k, dtype=torch.uint8)])
    powers = 1 << torch.arange(k - 1, -1, -1)
    return (bits.reshape(-1, k).to(torch.int64) * powers).sum(1)


def _value_read_bits(carriers, n_bits, k, delta):
    n_sym = (n_bits + k - 1) // k
    flat, _ = _flat_float_view(carriers)
    frac = flat[:n_sym] / delta
    s = torch.floor((frac - frac.floor()) * 2 ** k).to(torch.int64).clamp_(0, 2 ** k - 1)
    shifts = torch.arange(k - 1, -1, -1)
    return ((s.unsqueeze(1) >> shifts) & 1).to(torch.uint8).reshape(-1)[:n_bits]


def value_embed(carriers, payload, k, delta=DEFAULT_DELTA):
    assert 1 <= k <= MAX_SYMBOL_BITS and delta > 0
    bits = build_bitstream(payload)
    cap = value_capacity_bits(carriers, k)
    assert len(bits) <= cap, f"payload of {len(bits)} bits, capacity {cap}"
    s = _bits_to_symbols(bits, k)
    flat, write_back = _flat_float_view(carriers)
    n = len(s)

    # Move each weight to the center of its subcell
    flat[:n] = (torch.floor(flat[:n] / delta) + (s.double() + 0.5) / 2 ** k) * delta
    write_back(flat)
    return {
        "payload_bytes": len(payload),
        "bits_used": len(bits),
        "capacity_bits": cap,
        "capacity_bytes": value_capacity_bytes(carriers, k),
        "utilization": len(bits) / cap,
        "k": k,
        "delta": delta,
        "decision_margin": decision_margin(k, delta),
    }


def value_extract(carriers, k, delta=DEFAULT_DELTA):
    header = _value_read_bits(carriers, HEADER_BITS, k, delta)
    length, crc = _read_header(header, value_capacity_bytes(carriers, k))
    bits = _value_read_bits(carriers, HEADER_BITS + length * 8, k, delta)
    return verify_crc(_payload(bits, length), crc)


def value_extract_raw(carriers, length, k, delta=DEFAULT_DELTA):
    return _payload(_value_read_bits(carriers, HEADER_BITS + length * 8, k, delta), length)


class Method:
    def __init__(self, name, label, embed, extract, extract_raw, capacity_bits, capacity_bytes):
        self.name = name
        self.label = label
        self.embed = embed
        self.extract = extract
        self.extract_raw = extract_raw
        self.capacity_bits = capacity_bits
        self.capacity_bytes = capacity_bytes


METHODS = {
    "lsb": Method("lsb", "LSB",
                  lambda c, p, n_lsb: stego.embed(c, p, n_lsb),
                  lambda c, n_lsb: stego.extract(c, n_lsb),
                  lambda c, length, n_lsb: stego.extract_raw(c, length, n_lsb),
                  lambda c, n_lsb: stego.capacity_bits(c, n_lsb),
                  lambda c, n_lsb: stego.capacity_bytes(c, n_lsb)),
    "sign": Method("sign", "Sign mapping",
                   sign_embed, sign_extract, sign_extract_raw,
                   sign_capacity_bits, sign_capacity_bytes),
    "value": Method("value", "Value mapping",
                    value_embed, value_extract, value_extract_raw,
                    lambda c, k, **_: value_capacity_bits(c, k),
                    lambda c, k, **_: value_capacity_bytes(c, k)),
}


def get_method(name):
    return METHODS[name]


def params_at_equal_capacity(carriers, bits_per_weight, delta=DEFAULT_DELTA):
    out = {}
    if bits_per_weight <= stego.MANTISSA_BITS:
        out["lsb"] = {"n_lsb": bits_per_weight}
    if bits_per_weight <= MAX_SYMBOL_BITS:
        out["value"] = {"k": bits_per_weight, "delta": delta}
    if bits_per_weight == SIGN_BITS_PER_WEIGHT:
        out["sign"] = {}
    return out
