"""
Compares LSB, value mapping and sign mapping: Capacity, fidelity at equal capacity, value mapping fidelity vs delta and fragility under noise.
python src/experiment_methods.py --datasets cifar10 --quick 500
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets as D 
import stego 
import stego_mapping as sm 
import train as T

OUTPUT = os.path.join(D.REPO_ROOT, "results", "results_methods.json")
PAYLOAD_SEED = 20250829
NOISE_SEED = 1234      
SIGMAS = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
DELTAS = [1e-4, 1e-3, 1e-2, 1e-1]
BITS = 1


def random_payload(n_bytes):
    return np.random.default_rng(PAYLOAD_SEED).bytes(n_bytes)


@torch.no_grad()
def copy_weights(dst, src):
    for a, b in zip(dst, src):
        a.copy_(b)


# Gaussian noise
@torch.no_grad()
def add_noise(carriers, sigma):
    g = torch.Generator().manual_seed(NOISE_SEED)
    for t in carriers:
        t.add_(torch.randn(t.shape, generator=g).to(t.device) * sigma)


def survives(method, carriers, params):
    try:
        method.extract(carriers, **params)
        return True
    except (stego.ChecksumMismatch, AssertionError):
        return False


def distortion(clean, carriers):
    a = {"w": torch.cat([t.cpu().reshape(-1) for t in clean])}
    b = {"w": torch.cat([t.cpu().reshape(-1) for t in carriers])}
    return stego.weight_distortion(a, b, ["w"])


def load_clean(name, batch_size, workers, quick=0, device=D.DEVICE):
    d = D.get_dataset(name)
    model = T.build_model(d.n_classes, device, pretrained=False)
    model.load_state_dict(torch.load(d.checkpoint_path(), map_location=device))
    train_loader, test_loader = T.build_loaders(d, batch_size, workers)
    if quick:
        subset = torch.utils.data.Subset(test_loader.dataset, range(quick))
        test_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, num_workers=workers)
    return d, model, train_loader, test_loader


def evaluate_dataset(name, batch_size=512, workers=4, quick=0):
    t0 = time.time()
    print(f"\n{name}", flush=True)
    d, model, _, loader = load_clean(name, batch_size, workers, quick)
    accuracy = lambda: T.evaluate(model, loader, d.label_offset)  # noqa: E731

    carriers = stego.get_carriers(model)
    clean = [t.clone() for t in carriers]
    abs_w = torch.cat([t.reshape(-1) for t in clean]).abs()
    baseline = accuracy()
    print(f"  baseline {baseline:.2f}%  |w| min {float(abs_w.min()):.1e}", flush=True)

    # Max capacity
    capacity = [
        {"method": "lsb", "param": f"n_lsb={stego.MANTISSA_BITS}", "bits_per_weight": stego.MANTISSA_BITS,
         "bytes": stego.capacity_bytes(carriers, stego.MANTISSA_BITS)},
        {"method": "value", "param": f"k={sm.MAX_SYMBOL_BITS}", "bits_per_weight": sm.MAX_SYMBOL_BITS,
         "bytes": sm.value_capacity_bytes(carriers, sm.MAX_SYMBOL_BITS)},
        {"method": "sign", "param": "1 bit/weight", "bits_per_weight": 1,
         "bytes": sm.sign_capacity_bytes(carriers)},
    ]
    for row in capacity:
        row["kb"] = row["bytes"] / 1024

    # Fidelity at equal capacity
    params = sm.params_at_equal_capacity(carriers, BITS)
    n_bytes = min(sm.get_method(m).capacity_bytes(carriers, **p) for m, p in params.items())
    payload = random_payload(n_bytes)
    fidelity = []
    for method_name, p in params.items():
        method = sm.get_method(method_name)
        copy_weights(carriers, clean)
        info = method.embed(carriers, payload, **p)
        acc = accuracy()
        assert method.extract(carriers, **p) == payload
        fidelity.append({"method": method_name, "params": p, "payload_kb": n_bytes / 1024,
                         "capacity_bits": info["capacity_bits"], "acc": acc,
                         "delta": acc - baseline, **distortion(clean, carriers)})
        print(f"  fidelity {method_name:<6} {acc:6.2f}% ({acc - baseline:+.2f})", flush=True)

    # Value mapping fidelity delta
    delta_sweep = []
    for delta in DELTAS:
        copy_weights(carriers, clean)
        full_payload = random_payload(sm.value_capacity_bytes(carriers, BITS))
        info = sm.value_embed(carriers, full_payload, BITS, delta)
        acc = accuracy()
        delta_sweep.append({"delta": delta, "k": BITS, "margin": info["decision_margin"], "acc": acc,
                            "delta_acc": acc - baseline, **distortion(clean, carriers)})
        print(f"  delta={delta:<6g} {acc:6.2f}% ({acc - baseline:+.2f})", flush=True)

    # Accuracy clean model under noise
    model_noise = []
    for sigma in SIGMAS:
        copy_weights(carriers, clean)
        add_noise(carriers, sigma)
        acc = accuracy()
        model_noise.append({"sigma": sigma, "acc": acc, "delta_acc": acc - baseline})

    # Payload fragility same noise
    variants = [(m, p, m) for m, p in params.items()]
    variants += [("value", {"k": BITS, "delta": dl}, f"value_d{dl:g}") for dl in (1e-4, 1e-3, 1e-1)]
    fragility = []
    for method_name, p, tag in variants:
        method = sm.get_method(method_name)
        for sigma in SIGMAS:
            copy_weights(carriers, clean)
            method.embed(carriers, payload, **p)
            add_noise(carriers, sigma)
            rate = stego.ber(payload, method.extract_raw(carriers, len(payload), **p))
            fragility.append({"method": method_name, "tag": tag, "params": p, "sigma": sigma,
                              "ber": rate, "bit_errors": round(rate * len(payload) * 8),
                              "crc": survives(method, carriers, p)})

        # Largest sigma where CRC pass
        ok = [f["sigma"] for f in fragility if f["tag"] == tag and f["crc"]]
        if ok:
            print(f"  {tag:<12} CRC survives up to sigma={max(ok):g}", flush=True)
        else:
            print(f"  {tag:<12} CRC never passes", flush=True)

    copy_weights(carriers, clean)
    print(f"  [{time.time() - t0:.0f}s]", flush=True)
    return {
        "baseline": baseline,
        "n_weights": int(abs_w.numel()),
        "w_min": float(abs_w.min()),
        "w_median": float(abs_w.median()),
        "n_classes": d.n_classes,
        "capacity": capacity,
        "fidelity": {"bits_per_weight": BITS, "payload_kb": n_bytes / 1024, "rows": fidelity},
        "delta_sweep": delta_sweep,
        "model_noise": model_noise,
        "fragility": fragility,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(D.DATASETS), choices=list(D.DATASETS))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--quick", type=int, default=0, help="only the first N test images (smoke test)")
    p.add_argument("--output", default=OUTPUT)
    a = p.parse_args()

    res = {}
    for name in a.datasets:
        res[name] = evaluate_dataset(name, a.batch_size, a.workers, a.quick)
        os.makedirs(os.path.dirname(a.output), exist_ok=True)
        json.dump(res, open(a.output, "w"), indent=1)
