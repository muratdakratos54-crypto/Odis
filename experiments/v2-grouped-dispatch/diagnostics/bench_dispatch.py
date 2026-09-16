"""Tek MoE katmanının expert kısmı için dispatch mikro-benchmark'ı (fwd+bwd, bf16), GPU.

Karşılaştırılan: loop (mevcut), grouped_mm, padded bmm. Ayrıca profiler ile kernel adları.
"""
import sys
import time

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
dev = "cuda"
torch.manual_seed(0)
N, k, E, C, H = 32 * 512, 3, 16, 512, 352
dt = torch.bfloat16
skew = sys.argv[1] if len(sys.argv) > 1 else "balanced"

Wg = torch.randn(E, H, C, device=dev) * 0.02
Wu = torch.randn(E, H, C, device=dev) * 0.02
Wd = torch.randn(E, C, H, device=dev) * 0.02
for w in (Wg, Wu, Wd):
    w.requires_grad_(True)

if skew == "balanced":
    logits = torch.randn(N, E, device=dev)
else:  # collapse-like: a few experts get most tokens
    bias = torch.zeros(E, device=dev)
    bias[:5] = 3.0
    bias[10:] = -6.0
    logits = torch.randn(N, E, device=dev) + bias
idx = logits.topk(k, dim=-1).indices
flat = idx.reshape(-1)
order = flat.argsort(stable=True)
token_of = order // k
counts = torch.bincount(flat, minlength=E)
print("counts:", counts.tolist())
h = torch.randn(N, C, device=dev, requires_grad=True)


def run_loop():
    xin = h[token_of]
    outs = []
    with torch.autocast("cuda", dtype=dt):
        for e, chunk in enumerate(xin.split(counts.tolist())):
            outs.append(F.linear(F.silu(F.linear(chunk, Wg[e])) * F.linear(chunk, Wu[e]), Wd[e]))
        out = torch.cat(outs)
    return out


def run_grouped():
    xin = h[token_of].to(dt)
    offs = counts.cumsum(0).to(torch.int32)
    g = torch._grouped_mm(xin, Wg.to(dt).transpose(-2, -1), offs=offs)
    u = torch._grouped_mm(xin, Wu.to(dt).transpose(-2, -1), offs=offs)
    return torch._grouped_mm(F.silu(g) * u, Wd.to(dt).transpose(-2, -1), offs=offs)


def run_bmm():
    cap = int(counts.max())
    xin = h[token_of].to(dt)
    starts = torch.cumsum(counts, 0) - counts
    expert_of = flat[order]
    pos = torch.arange(N * k, device=dev) - starts[expert_of]
    buf = xin.new_zeros(E, cap, C)
    buf = buf.index_put((expert_of, pos), xin)
    g = torch.bmm(buf, Wg.to(dt).transpose(1, 2))
    u = torch.bmm(buf, Wu.to(dt).transpose(1, 2))
    o = torch.bmm(F.silu(g) * u, Wd.to(dt).transpose(1, 2))
    return o[expert_of, pos]


def bench(fn, iters=30):
    for _ in range(5):
        out = fn()
        out.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        out = fn()
        out.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


ref = run_loop().float()
for name, fn in (("grouped", run_grouped), ("bmm", run_bmm)):
    o = fn().float()
    print(f"{name}: max|diff| vs loop {(o - ref).abs().max().item():.3e}, rel {((o - ref).abs().max() / ref.abs().max()).item():.3e}")

for name, fn in (("loop", run_loop), ("grouped", run_grouped), ("bmm", run_bmm)):
    print(f"{name:8s} fwd+bwd: {bench(fn):.2f} ms")

from torch.profiler import ProfilerActivity, profile

for name, fn in (("grouped", run_grouped), ("loop", run_loop)):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn()
        out.float().pow(2).mean().backward()
        torch.cuda.synchronize()
    print(f"--- {name} CUDA kernels (top 12 by count)")
    rows = sorted(prof.key_averages(), key=lambda r: -r.count)
    for r in rows[:12]:
        if getattr(r, "device_type", None) is not None and "CUDA" not in str(r.device_type):
            continue
        print(f"  {r.count:5d}  {r.key[:110]}")
