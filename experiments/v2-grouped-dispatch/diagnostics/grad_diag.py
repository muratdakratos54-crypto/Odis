"""bf16'da grouped vs loop gradyan farkı nereden geliyor? Parametre grubu başına fark + bf16 vs fp32 ölçeği."""
import sys
from collections import defaultdict

import torch

import os  # noqa: E402
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import os  # noqa: E402

os.chdir(ROOT)
from scripts.verify_dispatch import run_once  # noqa: E402
from src.checkpoint import load_model  # noqa: E402
from src.data import TokenDataset  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
ds = TokenDataset("data/sp8k/val.bin", 512, "cuda")
x, y = ds.get_batch(32, torch.Generator().manual_seed(12345))
model = load_model("checkpoints/moe/best.pt", "cuda")[0].train()


def group(n):
    if "experts" in n:
        return "experts." + n.split(".")[-2]
    if "shared" in n:
        return "shared." + n.split(".")[-2]
    for key in ("router", "attn", "norm", "tok_emb"):
        if key in n:
            return key
    return n


L = run_once(model, x, y, "bf16", "loop")
G = run_once(model, x, y, "bf16", "grouped")
F32 = run_once(model, x, y, "fp32", "loop")

agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0])  # ||g-l||^2, ||l||^2, ||l-f32||^2, n_diff, n
for n in L["grads"]:
    a, b, c = L["grads"][n].double(), G["grads"][n].double(), F32["grads"][n].double()
    s = agg[group(n)]
    s[0] += (a - b).pow(2).sum().item()
    s[1] += a.pow(2).sum().item()
    s[2] += (a - c).pow(2).sum().item()
    s[3] += int((a != b).sum())
    s[4] += a.numel()
tot = [0.0, 0.0, 0.0]
print(f"{'grup':18s} {'grouped-loop rel':>18s} {'bf16-fp32 rel':>14s} {'oran':>8s} {'farklı eleman':>14s}")
for k, (dgl, nl, dlf, nd, n) in sorted(agg.items()):
    tot[0] += dgl
    tot[1] += nl
    tot[2] += dlf
    print(f"{k:18s} {(dgl / nl) ** .5:18.3e} {(dlf / nl) ** .5:14.3e} {((dgl / dlf) ** .5 if dlf else float('nan')):8.3f} {nd / n:14.2%}")
print(f"{'TÜM GRADYAN':18s} {(tot[0] / tot[1]) ** .5:18.3e} {(tot[2] / tot[1]) ** .5:14.3e} {(tot[0] / tot[2]) ** .5:8.3f}")

# Tek bir expert ağırlık gradyanında ulp ölçeği: fark / bf16 ulp (grad bf16'da hesaplanıp float32'ye çevrilir)
n = "blocks.4.ffn.experts.3.w_up.weight"
a, b = L["grads"][n], G["grads"][n]
mask = a != b
if mask.any():
    ulp = a[mask].abs() * 2.0 ** -7  # bf16: 7 bit mantissa -> göreli aralık 2^-7..2^-8
    print(f"{n}: farklı eleman %{mask.float().mean() * 100:.2f}, |fark|/(|g|*2^-7) medyan "
          f"{((a[mask] - b[mask]).abs() / ulp).median():.3f}, maks {((a[mask] - b[mask]).abs() / ulp).max():.3f}")
