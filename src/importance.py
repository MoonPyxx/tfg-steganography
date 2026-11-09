"""
Importance computed on weights with low n_lsb_max bits set to 0
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from stego import MANTISSA_BITS


# Weights with the low n_lsb_max bits set to 0
def mask_low_bits(tensor, n_lsb_max):
    assert 1 <= n_lsb_max <= MANTISSA_BITS
    raw = tensor.detach().cpu().contiguous().view(torch.int32).clone()
    raw &= ~((1 << n_lsb_max) - 1)
    return raw.view(torch.float32)


# |w| with the low bits masked
def magnitude_masked(carriers, n_lsb_max):
    return [mask_low_bits(t, n_lsb_max).abs() for t in carriers]


# Mean gradient over n batches
def _accumulate_grads(model, loader, device, n_batches, square):
    params = [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    acc = [torch.zeros_like(p) for p in params]
    model.eval()
    seen = 0
    for i, (x, y) in enumerate(loader):
        if n_batches is not None and i >= n_batches:
            break
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x.to(device)), y.to(device)).backward()
        for a, p in zip(acc, params):
            a += p.grad ** 2 if square else p.grad
        seen += 1
    model.zero_grad(set_to_none=True)
    return [a / seen for a in acc], params


# |w * dL/dw|
def grad_sensitivity(model, loader, device=None, n_batches=None):
    device = device or next(model.parameters()).device
    grads, params = _accumulate_grads(model, loader, device, n_batches, square=False)
    return [(p.detach() * g).abs().cpu() for p, g in zip(params, grads)]


# (dL/dw)^2
def fisher_diag(model, loader, device=None, n_batches=None):
    device = device or next(model.parameters()).device
    fisher, _ = _accumulate_grads(model, loader, device, n_batches, square=True)
    return [f.cpu() for f in fisher]


# Bit map by quantiles of importance
# goes from least important to most important
def allocate_bits(importances, bits_per_quantile, n_lsb_max=None):
    top = MANTISSA_BITS if n_lsb_max is None else n_lsb_max
    assert 1 <= top <= MANTISSA_BITS and all(0 <= b <= top for b in bits_per_quantile)
    flat = torch.cat([i.reshape(-1).float() for i in importances])
    n_groups = len(bits_per_quantile)

    rank = torch.empty(len(flat), dtype=torch.int64)
    rank[torch.argsort(flat, stable=True)] = torch.arange(len(flat))
    group = ((rank * n_groups) // len(flat)).clamp_(0, n_groups - 1)
    flat_bits = torch.tensor(bits_per_quantile)[group]

    out, off = [], 0
    for imp in importances:
        out.append(flat_bits[off:off + imp.numel()].reshape(imp.shape).clone())
        off += imp.numel()
    return out


def allocation_capacity_bits(bit_map):
    return int(sum(int(b.sum()) for b in bit_map))


# Importance per layer
def layer_summary(names, importances, bit_map=None):
    rows = []
    for i, (name, imp) in enumerate(zip(names, importances)):
        v = imp.reshape(-1).float()
        row = {
            "name": name,
            "n_weights": v.numel(),
            "imp_mean": float(v.mean()),
            "imp_median": float(v.median()),
            "imp_max": float(v.max()),
        }
        if bit_map is not None:
            b = bit_map[i].reshape(-1).double()
            row["bits_mean"] = float(b.mean())
            row["bits_total"] = int(b.sum())
        rows.append(row)
    return rows


def channel_importance(importance):
    return importance.reshape(importance.shape[0], -1).mean(1)


# Spearman correlation
def rank_correlation(a, b):
    from scipy.stats import spearmanr
    va = torch.cat([t.reshape(-1).double() for t in a]).numpy()
    vb = torch.cat([t.reshape(-1).double() for t in b]).numpy()
    return float(spearmanr(va, vb).statistic)
