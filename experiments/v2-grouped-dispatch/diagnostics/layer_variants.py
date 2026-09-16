"""Tam MoE katmanı (router + dispatch + experts + combine + shared + aux/z) fwd+bwd varyantları, dönüşümlü ölçüm.

    python layer_variants.py <moe|moe-noaux>
"""
import sys
import time

import torch
import torch.nn.functional as F

import os  # noqa: E402
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
ds = TokenDataset(os.path.join(ROOT, "data", "sp8k", "train.bin"), 512, dev)
xb, yb = ds.get_batch(32, torch.Generator().manual_seed(3))
cap = {}
blk = model.blocks[4]
hk = blk.ffn_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o.detach()))
with torch.no_grad(), torch.autocast(dev, dtype=bf):
    model(xb, yb)
hk.remove()
moe = blk.ffn
E, k = moe.n_experts, moe.top_k
X = cap["h"]  # (32, 512, 512) float32
N, C = 32 * 512, 512
G = torch.randn(32, 512, C, device=dev, dtype=bf)
Wg_p = torch.stack([e.w_gate.weight.detach() for e in moe.experts]).requires_grad_()
Wu_p = torch.stack([e.w_up.weight.detach() for e in moe.experts]).requires_grad_()
Wd_p = torch.stack([e.w_down.weight.detach() for e in moe.experts]).requires_grad_()
Wgu_p = torch.cat([Wg_p.detach(), Wu_p.detach()], 1).requires_grad_()


class CastGather(torch.autograd.Function):
    """h (float32) -> h.to(bf16)[idx]; backward float32'de biriktirir (orijinal cast-sonra-topla sırasıyla aynı dtype)."""

    @staticmethod
    def forward(ctx, h, idx, dtype):
        ctx.save_for_backward(idx)
        ctx.shape, ctx.hdtype = h.shape, h.dtype
        return h.to(dtype).index_select(0, idx)

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        out = torch.zeros(ctx.shape, dtype=ctx.hdtype, device=g.device)
        out.index_add_(0, idx, g.to(ctx.hdtype))
        return out, None, None


def route(h):
    with torch.autocast(device_type="cuda", enabled=False):
        logits = F.linear(h.float(), moe.router.weight.float())
    probs = logits.softmax(dim=-1)
    tp, idx = probs.topk(k, dim=-1)
    gates = tp / tp.sum(dim=-1, keepdim=True)
    flat = idx.reshape(-1)
    order = flat.argsort(stable=True)
    counts = torch.bincount(flat, minlength=E)
    return logits, probs, gates, order, order // k, counts


def finish(h, y, gates, order, token_of, eo, probs, logits, counts):
    w = gates.reshape(-1)[order].unsqueeze(1).to(eo.dtype)
    yy = torch.zeros(N, C, dtype=eo.dtype, device=h.device)
    yy.index_add_(0, token_of, eo * w)
    yy = yy + moe.shared(h).to(yy.dtype)
    aux = load_balancing_loss(probs, counts)
    z = router_z_loss(logits)
    return yy, 0.01 * aux + 0.001 * z


def make(variant):
    def fn(x):
        with torch.autocast("cuda", dtype=bf):
            h = x.reshape(-1, C)
            logits, probs, gates, order, token_of, counts = route(h)
            if variant == "L0_loop":
                ein = h[token_of]
                eo = torch.cat([ex(c) for ex, c in zip(moe.experts, ein.split(counts.tolist()))])
            else:
                if variant in ("L1_grp_stack",):
                    ein = h[token_of].to(bf)
                elif variant == "L2_grp_stack_isel":
                    ein = h.index_select(0, token_of).to(bf)
                else:
                    ein = CastGather.apply(h, token_of, bf)
                if variant == "L7_castgather_loop":
                    eo = torch.cat([ex(c) for ex, c in zip(moe.experts, ein.split(counts.tolist()))])
                else:
                    offs = counts.cumsum(0).to(torch.int32)
                    if variant in ("L4_grp_param", "L5_grp_param_fused", "L6_bmm_param"):
                        Wg, Wu, Wd = Wg_p, Wu_p, Wd_p
                    else:
                        Wg = torch.stack([e.w_gate.weight for e in moe.experts])
                        Wu = torch.stack([e.w_up.weight for e in moe.experts])
                        Wd = torch.stack([e.w_down.weight for e in moe.experts])
                    with torch.autocast("cuda", enabled=False):
                        if variant == "L5_grp_param_fused":
                            gg, uu = torch._grouped_mm(ein, Wgu_p.to(bf).transpose(-2, -1), offs=offs).chunk(2, -1)
                            eo = torch._grouped_mm(F.silu(gg) * uu, Wd.to(bf).transpose(-2, -1), offs=offs)
                        elif variant == "L6_bmm_param":
                            capn = int(counts.max())
                            starts = offs.long() - counts
                            eid = torch.repeat_interleave(torch.arange(E, device=dev), counts, output_size=N * k)
                            pos = torch.arange(N * k, device=dev) - starts[eid]
                            buf = ein.new_zeros(E, capn, C).index_put((eid, pos), ein)
                            gg = torch.bmm(buf, Wg.to(bf).transpose(1, 2))
                            uu = torch.bmm(buf, Wu.to(bf).transpose(1, 2))
                            eo = torch.bmm(F.silu(gg) * uu, Wd.to(bf).transpose(1, 2))[eid, pos]
                        else:
                            gg = torch._grouped_mm(ein, Wg.to(bf).transpose(-2, -1), offs=offs)
                            uu = torch._grouped_mm(ein, Wu.to(bf).transpose(-2, -1), offs=offs)
                            eo = torch._grouped_mm(F.silu(gg) * uu, Wd.to(bf).transpose(-2, -1), offs=offs)
            yy, reg = finish(h, None, gates, order, token_of, eo, probs, logits, counts)
        return yy.view(32, 512, C), reg
    return fn


variants = ["L0_loop", "L1_grp_stack", "L2_grp_stack_isel", "L3_grp_stack_castgather", "L4_grp_param",
            "L5_grp_param_fused", "L6_bmm_param", "L7_castgather_loop"]
fns = {v: make(v) for v in variants}
ref_out = None
for v in variants:
    x = X.detach().requires_grad_()
    o, reg = fns[v](x)
    if ref_out is None:
        ref_out = o.detach().float()
    print(f"{v:26s} max|out - L0| = {(o.detach().float() - ref_out).abs().max().item():.3e}")


def once(fn):
    x = X.detach().requires_grad_()
    o, reg = fn(x)
    torch.autograd.backward([o, reg], [G, None])


for v in variants:
    for _ in range(5):
        once(fns[v])
torch.cuda.synchronize()
times = {v: [] for v in variants}
for rnd in range(40):  # dönüşümlü: termal/arka plan kaymaları tüm varyantlara eşit dağılsın
    for v in variants:
        t = time.perf_counter()
        once(fns[v])
        torch.cuda.synchronize()
        times[v].append((time.perf_counter() - t) * 1e3)
for v in variants:
    ts = sorted(times[v])
    print(f"{v:26s} medyan {ts[len(ts) // 2]:6.2f} ms  (p10 {ts[len(ts) // 10]:6.2f}, p90 {ts[9 * len(ts) // 10]:6.2f})")
