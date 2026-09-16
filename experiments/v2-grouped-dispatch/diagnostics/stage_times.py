"""MoE katmanının ve modelin diğer parçalarının aşama aşama fwd+bwd GPU süresi (senkronize, medyan).

    .venv\\Scripts\\python experiments/v2-grouped-dispatch/diagnostics/stage_times.py [moe|moe-noaux] > stage_times.log

Gerçek girdiler: eğitilmiş checkpoint, gerçek veri, katman 4'ün ffn_norm çıktısı (32 x 512 token). Her aşama yaprak
tensörlerle yalıtılmış ölçülür; parçaların toplamı tam katmanla karşılaştırılır. "MoE tam katman" iki dispatch yolu
için de ölçülür (src/moe.py). Sayılar README "Deney kaydı: v2-grouped-dispatch" bölümünde kullanılıyor.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
from src.checkpoint import load_model  # noqa: E402
from src.data import TokenDataset  # noqa: E402
from src.moe import load_balancing_loss, router_z_loss  # noqa: E402

torch.set_float32_matmul_precision("high")
ck = sys.argv[1] if len(sys.argv) > 1 else "moe"
dev, bf = "cuda", torch.bfloat16
model, _ = load_model(os.path.join(ROOT, "checkpoints", ck, "best.pt"), dev)
dense, _ = load_model(os.path.join(ROOT, "checkpoints", "dense", "best.pt"), dev)
ds = TokenDataset(os.path.join(ROOT, "data", "sp8k", "train.bin"), 512, dev)
x, y = ds.get_batch(32, torch.Generator().manual_seed(3))

cap = {}
blk = model.blocks[4]
h1 = blk.ffn_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o.detach()))
h2 = blk.attn_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("xa", i[0].detach()))
with torch.no_grad(), torch.autocast(dev, dtype=bf):
    model(x, y)
h1.remove(), h2.remove()
H = cap["h"].reshape(-1, 512)
moe = blk.ffn
E, k = moe.n_experts, moe.top_k
N, C = H.shape
print(f"checkpoint {ck} | girdi {tuple(H.shape)} {H.dtype} | E={E} k={k} h={moe.experts[0].w_gate.weight.shape[0]}")

Wg = torch.stack([e.w_gate.weight.detach() for e in moe.experts]).requires_grad_()
Wu = torch.stack([e.w_up.weight.detach() for e in moe.experts]).requires_grad_()
Wd = torch.stack([e.w_down.weight.detach() for e in moe.experts]).requires_grad_()


def timeit(fn, iters=25):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1e3


with torch.no_grad():
    logits0 = F.linear(H, moe.router.weight)
    probs0 = logits0.softmax(-1)
    tp0, idx0 = probs0.topk(k, dim=-1)
    gates0 = tp0 / tp0.sum(-1, keepdim=True)
    flat0 = idx0.reshape(-1)
    order0 = flat0.argsort(stable=True)
    tok0 = order0 // k
    counts0 = torch.bincount(flat0, minlength=E)
    offs0 = counts0.cumsum(0, dtype=torch.int32)
print("expert başına token:", counts0.tolist())

g_ein = torch.randn(N * k, C, device=dev, dtype=bf)
ein0 = H.to(bf).index_select(0, tok0)
g_eout = torch.randn(N * k, C, device=dev, dtype=bf)
eout0 = torch.randn(N * k, C, device=dev, dtype=bf)
g_y = torch.randn(N, C, device=dev, dtype=bf)
G_full = torch.randn(32, 512, C, device=dev, dtype=bf)
g_attn = torch.randn(32, 512, C, device=dev, dtype=torch.float32)
xa0 = cap["xa"]


def s_router():
    h = H.detach().requires_grad_()
    lg = F.linear(h, moe.router.weight)
    pr = lg.softmax(-1)
    tp, idx = pr.topk(k, dim=-1)
    gates = tp / tp.sum(-1, keepdim=True)
    ((gates * 0.1).sum() + 0.01 * load_balancing_loss(pr, torch.bincount(idx.reshape(-1), minlength=E))
     + 0.001 * router_z_loss(lg)).backward()


def s_sort():
    flat = idx0.reshape(-1)
    order = flat.argsort(stable=True)
    _ = order // k
    c = torch.bincount(flat, minlength=E)
    _ = c.cumsum(0, dtype=torch.int32)


def s_gather():
    """Şimdiki kod: önce cast, sonra gather."""
    h = H.detach().requires_grad_()
    h.to(bf).index_select(0, tok0).backward(g_ein)


def s_gather_fp32():
    """Alternatif: float32 satırları kopyala, sonra çevir (iki kat bellek trafiği)."""
    h = H.detach().requires_grad_()
    h.index_select(0, tok0).to(bf).backward(g_ein)


def s_experts_grouped():
    ein = ein0.detach().requires_grad_()
    gate = torch._grouped_mm(ein, Wg.to(bf).transpose(1, 2), offs=offs0)
    up = torch._grouped_mm(ein, Wu.to(bf).transpose(1, 2), offs=offs0)
    torch._grouped_mm(F.silu(gate) * up, Wd.to(bf).transpose(1, 2), offs=offs0).backward(g_eout)


def s_experts_loop():
    ein = ein0.detach().requires_grad_()
    with torch.autocast(dev, dtype=bf):
        torch.cat([ex(c) for ex, c in zip(moe.experts, ein.split(counts0.tolist()))]).backward(g_eout)


def s_combine():
    eo = eout0.detach().requires_grad_()
    gt = gates0.detach().requires_grad_()
    w = gt.reshape(-1)[order0].unsqueeze(1).to(bf)
    yy = torch.zeros(N, C, dtype=bf, device=dev)
    yy.index_add_(0, tok0, eo * w)
    yy.backward(g_y)


def s_shared():
    h = H.detach().requires_grad_()
    with torch.autocast(dev, dtype=bf):
        moe.shared(h).backward(g_y)


def make_full(impl):
    def fn():
        moe.impl = impl
        h = cap["h"].detach().requires_grad_()
        with torch.autocast(dev, dtype=bf):
            o, st = moe(h)
            reg = 0.01 * st["aux_loss"] + 0.001 * st["z_loss"]
        torch.autograd.backward([o, reg], [G_full, None])
    return fn


def s_dense_ffn():
    h = H.detach().requires_grad_()
    with torch.autocast(dev, dtype=bf):
        dense.blocks[4].ffn(h).backward(g_y)


def s_attn():
    xa = xa0.detach().requires_grad_()
    with torch.autocast(dev, dtype=bf):
        o = blk.attn(blk.attn_norm(xa), model.rope_cos, model.rope_sin)
    o.backward(g_attn.to(o.dtype))


def s_head():
    h = xa0.detach().requires_grad_()
    with torch.autocast(dev, dtype=bf):
        lg = model.lm_head(model.norm(h))
        F.cross_entropy(lg.float().view(-1, lg.size(-1)), y.reshape(-1)).backward()


stages = [("router + aux/z (float32)", s_router), ("argsort + bincount", s_sort),
          ("gather + cast (token -> expert)", s_gather), ("aynısı float32 gather ile", s_gather_fp32),
          ("expert'ler: _grouped_mm", s_experts_grouped),
          ("expert'ler: loop", s_experts_loop), ("combine: gate x çıktı + index_add", s_combine),
          ("shared expert", s_shared), ("MoE tam katman (grouped)", make_full("grouped")),
          ("MoE tam katman (loop)", make_full("loop")), ("dense FFN (h=1408)", s_dense_ffn),
          ("attention + norm", s_attn), ("son norm + lm_head + CE", s_head)]
res = {}
for name, fn in stages:
    res[name] = timeit(fn)
    print(f"  {name:34s} fwd+bwd {res[name]:7.2f} ms")
parts = ["router + aux/z (float32)", "argsort + bincount", "gather + cast (token -> expert)",
         "expert'ler: _grouped_mm", "combine: gate x çıktı + index_add", "shared expert"]
print(f"  parçalar toplamı (grouped): {sum(res[p] for p in parts):.2f} ms")
for nm, m in (("dense", dense), ("moe", model)):
    for p in m.parameters():
        p.grad = torch.randn_like(p)
    opt = m.configure_optimizer(1e-4, 0.1, (0.9, 0.95), "cuda")

    def s_opt():
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    print(f"  clip + AdamW ({nm}, {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M param): {timeit(s_opt):.2f} ms")
