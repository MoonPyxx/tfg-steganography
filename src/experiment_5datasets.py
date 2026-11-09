"""
LSB experiments on the 5 datasets: Capacity, Fragility, Allocations and magnitude_masked

python src/experiment_5datasets.py --datasets cifar10 --parts capacity --quick 500
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets as D 
import importance  
import stego  
import train as T  
from experiment_methods import SIGMAS, add_noise, copy_weights, load_clean  

OUTPUT = os.path.join(D.REPO_ROOT, "results", "results_5datasets.json")
N_LSBS = (1, 4, 8, 12, 16, 20, 23)
N_LSB_FRAGILITY = 16
ALLOCATIONS = ([16, 12, 8, 4], [20, 16, 12, 8, 4, 0], [20, 12, 4, 0])
PROFILE_ALLOCATION = [20, 16, 12, 8, 4, 0]
PROFILE_N_MAX = 20
CAPACITY_SEED = 0
FRAGILITY_SEED = 1
NONUNIFORM_SEED = 0
PARTS = ["capacity", "fragility", "nonuniform", "profile"]


# Accuracy vs n_lsb, random payload full capacity
def exp_capacity(carriers, clean, accuracy, baseline):
    rng = np.random.default_rng(CAPACITY_SEED)
    rows = []
    for n in N_LSBS:
        copy_weights(carriers, clean)
        cap = stego.capacity_bytes(carriers, n)
        payload = rng.bytes(cap)
        stego.embed(carriers, payload, n_lsb=n)
        assert stego.extract(carriers, n_lsb=n) == payload
        acc = accuracy()
        rows.append({"n_lsb": n, "kb": cap / 1024, "acc": acc, "delta": acc - baseline})
        print(f"  n_lsb={n:<2} {acc:6.2f}% ({acc - baseline:+.2f})", flush=True)
    return rows


# Payload at N_LSB_FRAGILITY under noise: accuracy, BER and CRC
def exp_fragility(carriers, clean, accuracy):
    copy_weights(carriers, clean)
    n_bytes = stego.capacity_bytes(carriers, N_LSB_FRAGILITY)
    payload = np.random.default_rng(FRAGILITY_SEED).bytes(n_bytes)
    stego.embed(carriers, payload, n_lsb=N_LSB_FRAGILITY)
    with_payload = [t.clone() for t in carriers]
    rows = []
    for sigma in SIGMAS:
        copy_weights(carriers, with_payload)
        add_noise(carriers, sigma)
        acc = accuracy()
        rate = stego.ber(payload, stego.extract_raw(carriers, len(payload), n_lsb=N_LSB_FRAGILITY))
        try:
            crc = "OK" if stego.extract(carriers, n_lsb=N_LSB_FRAGILITY) == payload else "corrupt"
        except stego.ChecksumMismatch:
            crc = "FAIL"
        rows.append({"sigma": sigma, "acc": acc, "ber": rate,
                     "bit_errors": round(rate * len(payload) * 8), "crc": crc})
        print(f"  sigma={sigma:<6g} {acc:6.2f}%  BER {rate:.4f}  {crc}", flush=True)
    return rows


# Non-uniform vs uniform allocation 
def exp_nonuniform(carriers, clean, accuracy, n_weights):
    rng = np.random.default_rng(NONUNIFORM_SEED)
    rows = []
    for allocation in ALLOCATIONS:
        n_max = max(allocation)
        copy_weights(carriers, clean)
        bm = importance.allocate_bits(importance.magnitude_masked(carriers, n_max), allocation,
                                      n_lsb_max=n_max)
        cap = stego.capacity_bytes_map(bm)
        mean_bits = stego.capacity_bits_map(bm) / n_weights
        payload = rng.bytes(cap)
        stego.embed_map(carriers, payload, bm)
        assert stego.extract_map(carriers, bm) == payload
        acc_nu = accuracy()

        n_unif = max(1, round(mean_bits))
        copy_weights(carriers, clean)
        payload_u = payload[:stego.capacity_bytes(carriers, n_unif)]
        stego.embed(carriers, payload_u, n_lsb=n_unif)
        assert stego.extract(carriers, n_lsb=n_unif) == payload_u
        acc_u = accuracy()

        rows.append({"allocation": allocation, "bits": mean_bits, "kb": cap / 1024,
                     "acc_nu": acc_nu, "acc_u": acc_u, "gain": acc_nu - acc_u})
        print(f"  {str(allocation):<22} non-unif {acc_nu:6.2f}%  unif {acc_u:6.2f}%", flush=True)
    return rows


def exp_profile(model, carriers, clean):
    copy_weights(carriers, clean)
    imp = importance.magnitude_masked(carriers, PROFILE_N_MAX)
    bm = importance.allocate_bits(imp, PROFILE_ALLOCATION, n_lsb_max=PROFILE_N_MAX)
    return importance.layer_summary(stego.carrier_names(model), imp, bm)


def evaluate_dataset(name, batch_size=512, workers=4, quick=0, parts=None):
    t0 = time.time()
    print(f"\n{name}", flush=True)
    d, model, _, loader = load_clean(name, batch_size, workers, quick)
    accuracy = lambda: T.evaluate(model, loader, d.label_offset)  # noqa: E731

    carriers = stego.get_carriers(model)
    clean = [t.clone() for t in carriers]
    n_weights = sum(t.numel() for t in carriers)
    baseline = accuracy()
    print(f"  baseline {baseline:.2f}%", flush=True)

    parts = parts or PARTS
    res = {"baseline": baseline, "n_weights": n_weights, "n_classes": d.n_classes}
    if "capacity" in parts:
        res["capacity"] = exp_capacity(carriers, clean, accuracy, baseline)
    if "fragility" in parts:
        res["fragility"] = exp_fragility(carriers, clean, accuracy)
    if "nonuniform" in parts:
        res["nonuniform"] = exp_nonuniform(carriers, clean, accuracy, n_weights)
    if "profile" in parts:
        res["profile"] = exp_profile(model, carriers, clean)

    copy_weights(carriers, clean)
    print(f"  [{time.time() - t0:.0f}s]", flush=True)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(D.DATASETS), choices=list(D.DATASETS))
    p.add_argument("--parts", nargs="+", choices=PARTS)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--quick", type=int, default=0, help="only the first N test images (smoke test)")
    p.add_argument("--output", default=OUTPUT)
    a = p.parse_args()

    res = {}
    for name in a.datasets:
        res[name] = evaluate_dataset(name, a.batch_size, a.workers, a.quick, a.parts)
        os.makedirs(os.path.dirname(a.output), exist_ok=True)
        json.dump(res, open(a.output, "w"), indent=1)
