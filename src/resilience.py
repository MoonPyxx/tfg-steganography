"""
Resilience training
python src/resilience.py cifar10 --n-lsb 23 --epochs 3
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D 
import stego  
import train as T 
from experiment_methods import copy_weights, load_clean, random_payload

OUTPUT = os.path.join(D.REPO_ROOT, "results", "results_resilience.json")


# Write payload into carriers
@torch.no_grad()
def project(carriers, bits, flat_map):
    stego._write(carriers, bits, flat_map)


def resilience_train(model, payload, n_lsb, train_loader, test_loader, epochs=3, lr=0.01,
                     offset=0, device=D.DEVICE, log=print):
    carriers = stego.get_carriers(model)
    control = payload is None

    # Uniform bit map and bitstream with header
    if not control:
        flat_map = stego._flat_map(carriers, stego._const_map(carriers, n_lsb))
        bits = stego.build_bitstream(payload)
        assert len(bits) <= int(flat_map.sum())

    # Buffer
    latents = [t.clone() for t in carriers]

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))

    history, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            if not control:
                copy_weights(latents, carriers)
                project(carriers, bits, flat_map)

            # forward and backward 
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.to(device)), (y - offset).to(device))
            loss.backward()

            if not control:
                copy_weights(carriers, latents)
            opt.step()
            sched.step()

        if not control:
            copy_weights(latents, carriers)
            project(carriers, bits, flat_map)
        acc = T.evaluate(model, test_loader, offset, device)
        assert control or stego.extract(carriers, n_lsb) == payload
        if not control:
            copy_weights(carriers, latents)

        history.append({"epoch": ep, "acc": acc, "t": time.time() - t0})
        log(f"  epoch {ep}/{epochs}  acc {acc:.2f}%  [{time.time() - t0:.0f}s]")

    if not control:
        project(carriers, bits, flat_map)
    return history


def control(name, epochs, lr, batch_size, workers, device=D.DEVICE):
    print(f"\n{name} CONTROL", flush=True)
    d, model, train_loader, test_loader = load_clean(name, batch_size, workers, device=device)
    offset = d.label_offset

    baseline = T.evaluate(model, test_loader, offset, device)
    history = resilience_train(model, None, None, train_loader, test_loader, epochs, lr, offset,
                               device, log=lambda s: print(s, flush=True))
    acc = history[-1]["acc"]
    print(f"  control {acc:.2f}% ({acc - baseline:+.2f})", flush=True)

    return {
        "dataset": name,
        "n_lsb": None,
        "control": True,
        "baseline": baseline,
        "acc_control": acc,
        "delta_control": acc - baseline,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "history": history,
    }


# LSB embedding, then resilience 
def experiment(name, n_lsb, epochs, lr, batch_size, workers, device=D.DEVICE):
    print(f"\n{name}  n_lsb={n_lsb} ", flush=True)
    d, model, train_loader, test_loader = load_clean(name, batch_size, workers, device=device)
    offset = d.label_offset
    carriers = stego.get_carriers(model)
    clean = [t.clone() for t in carriers]

    baseline = T.evaluate(model, test_loader, offset, device)
    payload = random_payload(stego.capacity_bytes(carriers, n_lsb))
    stego.embed(carriers, payload, n_lsb)
    acc_lsb = T.evaluate(model, test_loader, offset, device)
    print(f"  baseline {baseline:.2f}%  plain LSB {acc_lsb:.2f}% ({acc_lsb - baseline:+.2f})", flush=True)

    copy_weights(carriers, clean)

    history = resilience_train(model, payload, n_lsb, train_loader, test_loader, epochs, lr, offset,
                               device, log=lambda s: print(s, flush=True))
    acc_res = history[-1]["acc"]
    assert stego.extract(carriers, n_lsb) == payload

    # Save model
    os.makedirs(D.CKPT_ROOT, exist_ok=True)
    torch.save(model.state_dict(), d.checkpoint_path(f"resilience_n{n_lsb}"))
    print(f"  resilience {acc_res:.2f}% ({acc_res - baseline:+.2f})  "
          f"recovered {acc_res - acc_lsb:+.2f}", flush=True)

    return {
        "dataset": name,
        "n_lsb": n_lsb,
        "payload_kb": len(payload) / 1024,
        "baseline": baseline,
        "acc_lsb": acc_lsb,
        "delta_lsb": acc_lsb - baseline,
        "acc_resilience": acc_res,
        "delta_resilience": acc_res - baseline,
        "recovered": acc_res - acc_lsb,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "history": history,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="+", choices=list(D.DATASETS))
    p.add_argument("--n-lsb", type=int, nargs="+", default=[23])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--control", action="store_true", help="same recipe without payload (cost of retraining)")
    p.add_argument("--output", default=OUTPUT)
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.output), exist_ok=True)

    for name in a.datasets:
        for n in ([None] if a.control else a.n_lsb):
            if a.control:
                key = f"{name}_control"
                r = control(name, a.epochs, a.lr, a.batch_size, a.workers)
            else:
                key = f"{name}_n{n}"
                r = experiment(name, n, a.epochs, a.lr, a.batch_size, a.workers)

            res = json.load(open(a.output)) if os.path.exists(a.output) else {}
            res[key] = r
            json.dump(res, open(a.output, "w"), indent=1)
