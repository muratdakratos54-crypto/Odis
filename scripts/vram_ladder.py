"""Sabit batch/sequence'te model boyutu merdiveni: tepe VRAM ve tok/s ölçer (FOTON kampanyası boyutlaması).

    .venv\\Scripts\\python scripts/vram_ladder.py --arch dense,moe --limit_gb 10.5

Neden: FOTON kampanyasının boyutu "parametre hedefi" değil **VRAM hedefi** ile seçiliyor — mevcut batch
(mikro-batch 32) ve dizi uzunluğunu (512) bozmadan 12 GB'a sığan en büyük boyut. Bu betik gerçek bir
eğitim adımı (ileri + geri + AdamW) koşturup `torch.cuda.max_memory_allocated`'ı okur; tahmin yapmaz.

FOTON henüz yazılmadığı için onun belleği ölçülemiyor; bunun yerine `--foton_modules N` ile
"N modülün ağırlık + gradyan + AdamW state'i" ek maliyeti ANALİTİK eklenir (aktivasyon tarafı
değişmez, çünkü sequence-level top-1'de dizi başına yalnızca BİR modül çalışır).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataclasses import replace  # noqa: E402

from src.config import PRESETS  # noqa: E402
from src.flops import train_flops_per_token  # noqa: E402
from src.model import GPT  # noqa: E402

BYTES_PER_PARAM = 16  # fp32 ağırlık 4 + gradyan 4 + AdamW exp_avg/exp_avg_sq 8


def step_probe(cfg, micro_batch: int, steps: int = 6) -> dict:
    """Gerçek bir eğitim adımı: rastgele token, bf16 autocast, geri yayılım, AdamW. Tepe belleği döner."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = GPT(cfg).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (micro_batch, cfg.max_seq_len), device="cuda", generator=g)
    y = torch.randint(0, cfg.vocab_size, (micro_batch, cfg.max_seq_len), device="cuda", generator=g)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i == 0:                      # ilk adım ısınma: zamanı ondan sonra ölç
            torch.cuda.synchronize()
            t0 = time.perf_counter()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / (steps - 1)
    peak = torch.cuda.max_memory_allocated() / 1e9
    n_par = sum(p.numel() for p in model.parameters())
    del model, opt, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return {"peak_gb": peak, "sec_per_micro_step": dt, "params": n_par,
            "tok_per_sec": micro_batch * cfg.max_seq_len / dt}


def ladder(d_model: int, n_layer: int) -> dict:
    """Merdiven basamağı: d_model'e bağlı head/ffn/expert boyutları, v2/v3 oranlarını korur."""
    return {"d_model": d_model, "n_layer": n_layer, "n_head": d_model // 64,
            "ffn_hidden": int(round(d_model * 2.75)), "expert_hidden": int(round(d_model * 2.75)) // 4}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="dense,moe")
    p.add_argument("--micro_batch", type=int, default=32)
    p.add_argument("--limit_gb", type=float, default=10.5, help="güvenli tepe VRAM tavanı")
    p.add_argument("--foton_modules", type=int, default=3, help="FOTON'un modül sayısı (analitik ek maliyet)")
    p.add_argument("--foton_module_layers", type=int, default=2, help="modül başına katman (paylaşılmayan)")
    p.add_argument("--sizes", default="512x8,576x9,640x10,704x11,768x12,832x12,896x12")
    p.add_argument("--vocab_size", type=int, default=0, help="0 = preset degeri; kampanya icin 32768")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--out", default="")
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)} | toplam "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB | mikro-batch {args.micro_batch} "
          f"x dizi 512 | güvenli tavan {args.limit_gb} GB")
    print(f"{'boyut':>10s} {'mimari':>6s} {'param':>9s} {'tepe VRAM':>10s} {'tok/s':>9s} "
          f"{'FLOP/token':>13s} {'+FOTON VRAM':>12s} {'sığar mı':>9s}")
    rows = []
    for spec in args.sizes.split(","):
        dm, nl = (int(v) for v in spec.split("x"))
        dims = ladder(dm, nl)
        for arch in args.arch.split(","):
            base = PRESETS[arch][0]
            cfg = replace(base, **dims)
            if args.vocab_size:
                cfg = replace(cfg, vocab_size=args.vocab_size)
            if arch == "moe":
                cfg = replace(cfg, n_experts=16, top_k=3, n_shared_experts=1, first_dense_layers=1)
            try:
                r = step_probe(cfg, args.micro_batch, args.steps)
            except torch.OutOfMemoryError:
                print(f"{spec:>10s} {arch:>6s} {'—':>9s} {'OOM':>10s}")
                rows.append({"size": spec, "arch": arch, "oom": True})
                continue
            # FOTON eki: (modül sayısı - 1) x modül parametresi x 16 B. Modül = module_layers katman
            # (attention + FFN), paylaşılan gövdeden ayrı. Aktivasyon değişmez (dizi başına tek modül).
            per_layer = 4 * dm * dm + 3 * dm * dims["ffn_hidden"]
            extra = (args.foton_modules - 1) * args.foton_module_layers * per_layer * BYTES_PER_PARAM / 1e9
            total = r["peak_gb"] + extra
            fits = "EVET" if total <= args.limit_gb else "hayır"
            print(f"{spec:>10s} {arch:>6s} {r['params']/1e6:8.1f}M {r['peak_gb']:9.2f}G "
                  f"{r['tok_per_sec']:9,.0f} {train_flops_per_token(cfg):13,d} {total:11.2f}G {fits:>9s}")
            rows.append({"size": spec, "arch": arch, **r, "foton_extra_gb": extra,
                         "foton_total_gb": total, "fits": total <= args.limit_gb,
                         "flops_per_token": train_flops_per_token(cfg)})
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=1)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
