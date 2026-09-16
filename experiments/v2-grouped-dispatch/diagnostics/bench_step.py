"""Tam eğitim adımı benchmark'ı: dense vs MoE dispatch varyantları (gerçek veri + eğitilmiş checkpoint).

    python bench_step.py <variant> <ckpt: moe|moe-noaux|dense>
variant: dense | loop | grouped_stack | grouped_stack_isel | grouped_param | grouped_param_fused
"""
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import os  # noqa: E402
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
from src.checkpoint import load_model  # noqa: E402
from src.data import TokenDataset  # noqa: E402
from src.moe import MoE, load_balancing_loss, router_z_loss  # noqa: E402

variant, ck = sys.argv[1], sys.argv[2]
torch.set_float32_matmul_precision("high")
torch.cuda.set_per_process_memory_fraction(0.92)
model, _ = load_model(os.path.join(ROOT, "checkpoints", ck, "best.pt"), "cuda")
model.train()


def route(self, h):
    with torch.autocast(device_type=h.device.type, enabled=False):
        logits = F.linear(h.float(), self.router.weight.float())
    probs = logits.softmax(dim=-1)
    topk_p, topk_idx = probs.topk(self.top_k, dim=-1)
    gates = topk_p / topk_p.sum(dim=-1, keepdim=True)
    return logits, probs, topk_idx, gates


def fwd_grouped(self, x):
    B, T, C = x.shape
    h = x.reshape(-1, C)
    N, E, k = h.size(0), self.n_experts, self.top_k
    logits, probs, topk_idx, gates = route(self, h)
    flat_expert = topk_idx.reshape(-1)
    order = flat_expert.argsort(stable=True)
    token_of = order // k
    counts = torch.bincount(flat_expert, minlength=E)
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else h.dtype
    if variant in ("grouped_stack_isel", "grouped_param", "grouped_param_fused"):
        expert_in = h.index_select(0, token_of).to(dt)
    else:
        expert_in = h[token_of].to(dt)
    offs = counts.cumsum(0).to(torch.int32)
    if variant.startswith("grouped_param"):
        Wg, Wu, Wd = self.Wg, self.Wu, self.Wd
    else:
        Wg = torch.stack([e.w_gate.weight for e in self.experts])
        Wu = torch.stack([e.w_up.weight for e in self.experts])
        Wd = torch.stack([e.w_down.weight for e in self.experts])
    with torch.autocast("cuda", enabled=False):
        if variant == "grouped_param_fused":
            gu = torch._grouped_mm(expert_in, self.Wgu.to(dt).transpose(-2, -1), offs=offs)
            g, u = gu.chunk(2, dim=-1)
        else:
            g = torch._grouped_mm(expert_in, Wg.to(dt).transpose(-2, -1), offs=offs)
            u = torch._grouped_mm(expert_in, Wu.to(dt).transpose(-2, -1), offs=offs)
        expert_out = torch._grouped_mm(F.silu(g) * u, Wd.to(dt).transpose(-2, -1), offs=offs)
    w = gates.reshape(-1)[order].unsqueeze(1).to(expert_out.dtype)
    y = torch.zeros(N, C, dtype=expert_out.dtype, device=h.device)
    y.index_add_(0, token_of, expert_out * w)
    y = y + self.shared(h).to(y.dtype)
    stats = {"aux_loss": load_balancing_loss(probs, counts), "z_loss": router_z_loss(logits),
             "counts": counts.detach(), "topk_idx": topk_idx.detach()}
    return y.view(B, T, C), stats


if variant != "loop" and variant != "dense":
    for b in model.blocks:
        if b.is_moe:
            m = b.ffn
            if variant.startswith("grouped_param"):
                m.Wg = nn.Parameter(torch.stack([e.w_gate.weight.detach() for e in m.experts]))
                m.Wu = nn.Parameter(torch.stack([e.w_up.weight.detach() for e in m.experts]))
                m.Wd = nn.Parameter(torch.stack([e.w_down.weight.detach() for e in m.experts]))
                if variant == "grouped_param_fused":
                    m.Wgu = nn.Parameter(torch.cat([m.Wg.detach(), m.Wu.detach()], dim=1))
                del m.experts
            m.forward = types.MethodType(fwd_grouped, m)

opt = model.configure_optimizer(1e-4, 0.1, (0.9, 0.95), "cuda")
ds = TokenDataset(os.path.join(ROOT, "data", "sp8k", "train.bin"), 512, "cuda")
g = torch.Generator().manual_seed(1)
batches = [ds.get_batch(32, g) for _ in range(2 * 30)]


def step(i):
    for j in range(2):
        x, y = batches[(2 * i + j) % len(batches)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, info = model(x, y)
        (loss / 2).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)


for i in range(6):
    step(i)
torch.cuda.synchronize()
times = []
for i in range(6, 30):
    t = time.perf_counter()
    step(i)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t)
times.sort()
med = times[len(times) // 2]
print(f"{variant:22s} ckpt={ck:9s} median {med * 1e3:7.1f} ms/step  -> {32768 / med / 1e3:6.1f}k tok/s  "
      f"(min {times[0] * 1e3:.1f}, max {times[-1] * 1e3:.1f})  peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
