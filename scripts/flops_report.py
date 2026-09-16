"""Analitik aktif FLOP tablosu (birincil) + profiler çapraz kontrolü (yalnızca kontrol).

    .venv\\Scripts\\python scripts/flops_report.py                 # tablo + GPU'da profiler kontrolü
    .venv\\Scripts\\python scripts/flops_report.py --no_profile    # yalnızca analitik tablo
    .venv\\Scripts\\python scripts/flops_report.py --json out.json

Türetim: docs/flops.md. Profiler (torch.utils.flop_counter.FlopCounterMode) çekirdek başına formül tablosuyla
sayar: formülü olmayan çekirdek 0 sayılır, SDPA causal maskeden bağımsız T x T sayılır. Bu yüzden profiler
sayısı hesap bütçesi olarak KULLANILMAZ; burada yalnızca analitik formülün matris çarpımı terimlerini
bağımsız bir sayaçla doğrulamak ve farkların nereden geldiğini açıkça göstermek için çalıştırılır.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PRESETS  # noqa: E402
from src.flops import forward_flops_per_token, secondary_forward_flops_per_token, train_flops_per_token  # noqa: E402
from src.model import GPT  # noqa: E402

RUNS = ["dense", "moe", "moe-noaux"]


def analytic(preset: str) -> dict:
    mcfg, tcfg = PRESETS[preset]
    fwd = forward_flops_per_token(mcfg)
    sec = secondary_forward_flops_per_token(mcfg)
    model = GPT(mcfg)
    return {
        "forward_terms": fwd, "forward": sum(fwd.values()), "train_per_token": train_flops_per_token(mcfg),
        "secondary_forward_terms": sec, "secondary_forward": sum(sec.values()),
        "active_params": model.num_params(active_only=True), "total_params": model.num_params(),
        "tokens_per_step": tcfg.batch_size * tcfg.grad_accum * mcfg.max_seq_len,
    }


def profile(preset: str, impl: str, batch: int = 4) -> dict:
    """Bir mikro-batch ileri + geri, bf16 autocast, gerçek eğitimdeki çekirdeklerle (CUDA)."""
    from torch.utils.flop_counter import FlopCounterMode

    mcfg, _ = PRESETS[preset]
    torch.manual_seed(0)
    model = GPT(replace(mcfg, moe_impl=impl)).cuda()
    T = mcfg.max_seq_len
    x = torch.randint(0, mcfg.vocab_size, (batch, T), device="cuda")
    out = {}
    for phase in ("forward", "forward+backward"):
        with FlopCounterMode(display=False) as counter:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, x)
            if phase != "forward":
                loss.backward()
        per_op = {str(op): n / (batch * T) for op, n in counter.get_flop_counts()["Global"].items()}
        out[phase] = {"total_per_token": counter.get_total_flops() / (batch * T), "per_op": per_op}
        model.zero_grad(set_to_none=True)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--no_profile", action="store_true")
    p.add_argument("--json", default="")
    args = p.parse_args()

    report = {"analytic": {r: analytic(r) for r in RUNS}}
    a = report["analytic"]
    terms = list(a["moe"]["forward_terms"])
    print("== BİRİNCİL: analitik aktif ileri FLOP / token (türetim: docs/flops.md) ==")
    print(f"{'terim':22s}" + "".join(f"{r:>14s}" for r in RUNS))
    for t in terms:
        print(f"{t:22s}" + "".join(f"{a[r]['forward_terms'][t]:>14,}" for r in RUNS))
    print(f"{'İLERİ toplam':22s}" + "".join(f"{a[r]['forward']:>14,}" for r in RUNS))
    print(f"{'EĞİTİM (3x) / token':22s}" + "".join(f"{a[r]['train_per_token']:>14,}" for r in RUNS))
    d, m = a["dense"], a["moe"]
    print(f"moe / dense eğitim FLOP oranı: {m['train_per_token'] / d['train_per_token']:.5f} "
          f"(fark = router: {m['forward_terms']['moe_router']:,} / token ileri)")
    steps = PRESETS["moe"][1].max_steps
    tokens = steps * m["tokens_per_step"]
    print(f"{steps} adım x {m['tokens_per_step']:,} token = {tokens / 1e6:.2f}M token: "
          f"dense {d['train_per_token'] * tokens:.4e}, moe {m['train_per_token'] * tokens:.4e} FLOP")
    same_flop_tokens = m["train_per_token"] * tokens / d["train_per_token"]
    print(f"moe'nin {steps} adımlık FLOP'una eşit dense bütçesi: {same_flop_tokens / 1e6:.3f}M token = "
          f"{same_flop_tokens / d['tokens_per_step']:.2f} adım")
    print("\n== İKİNCİL (yaklaşık, bütçelere girmez): eleman bazlı işlemler, ileri / token ==")
    for t in a["moe"]["secondary_forward_terms"]:
        print(f"{t:26s}" + "".join(f"{a[r]['secondary_forward_terms'][t]:>14,.0f}" for r in RUNS))
    for r in RUNS:
        s = a[r]["secondary_forward"]
        print(f"  {r}: ikincil / birincil = {s / a[r]['forward']:.2%}")
    with_sec = [(a[r]["forward"] + a[r]["secondary_forward"]) for r in ("dense", "moe")]
    print(f"ikincil dahil moe / dense oranı: {with_sec[1] / with_sec[0]:.5f}")
    print("aktif / toplam parametre: " + ", ".join(
        f"{r} {a[r]['active_params'] / 1e6:.2f}M / {a[r]['total_params'] / 1e6:.2f}M" for r in RUNS))

    if not args.no_profile and torch.cuda.is_available():
        print("\n== ÇAPRAZ KONTROL: profiler (FlopCounterMode), CUDA, bf16, ileri+geri / token ==")
        report["profiler"] = {}
        for preset, impl in (("dense", "grouped"), ("moe", "loop"), ("moe", "grouped")):
            name = preset if preset == "dense" else f"{preset}-{impl}"
            prof = profile(preset, impl)
            report["profiler"][name] = prof
            an = a[preset]
            fb = prof["forward+backward"]
            attn_ops = sum(v for op, v in fb["per_op"].items() if "attention" in op)
            linear_ops = sum(v for op, v in fb["per_op"].items() if "attention" not in op)
            an_linear = 3 * (an["forward"] - an["forward_terms"]["attn_context"])
            an_ctx = 3 * an["forward_terms"]["attn_context"]
            print(f"-- {name}: profiler toplam {fb['total_per_token']:,.0f} | analitik {an['train_per_token']:,} "
                  f"(oran {fb['total_per_token'] / an['train_per_token']:.3f})")
            print(f"   matris çarpımları: profiler {linear_ops:,.0f} vs analitik {an_linear:,} "
                  f"(fark {linear_ops - an_linear:+,.0f})")
            print(f"   attention çekirdekleri: profiler {attn_ops:,.0f} vs analitik causal bağlam {an_ctx:,}")
            print("   op'lar: " + ", ".join(f"{op} {v:,.0f}" for op, v in sorted(fb["per_op"].items())))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
