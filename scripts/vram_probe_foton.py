"""Kampanyadan ÖNCE dört kolun GERÇEK tepe VRAM'ini ölçer: eğitim adımı + eval (FOTON'da oracle geçişi dahil).

    .venv\\Scripts\\python scripts/vram_probe_foton.py --out experiments/foton-v02/vram_probe.json

Neden ayrı bir betik: `scripts/vram_ladder.py` gerçek tepeyi SİSTEMATİK olarak az gösteriyor (ön-kayıt V8:
S0'da MoE +0.90 GB, dense +0.71 GB). Ön-kayıt bunu eval'deki `logits.float()` tensörüne bağlıyordu; bu betik
yazılırken ÖLÇÜLDÜ ki sebep o değil: eval tepesi eğitim tepesinin ALTINDA kalıyor. Asıl sebep eğitim
döngüsünün `_, loss, info = model(x, y)` ataması — `_` önceki mikro-batch'in bf16 logit tensörünü (0.54 GB)
bir sonraki forward bitene kadar canlı tutuyor. O atama taklit edilince prob gerçek koşuları üç ondalığa
kadar yeniden üretiyor (dense 6.48 / gerçek 6.480 GB; MoE 8.03 / gerçek 8.031 GB). Dört kolda da aynı kod
çalıştığı için bu bir kollar-arası fark değildir; S1'deki dar pay (~0.9 GB) için bilinmesi gerekir.
FOTON kollarında eval ayrıca üç modülün üçünü de koşturuyor (oracle eğrileri). Burada koşunun KENDİ kodu çalışır: `GPT.configure_optimizer`, gerçek korpustan
`sequence_stream` ile 16 x 4 mikro-batch'lik optimizer adımları, sonra `src.train.evaluate()`, sonra yine bir
eğitim adımı (eval'den sonraki parçalanmış ayırıcıyla). `vram_limit_frac = 0.92` koşulardaki gibi uygulanır:
sığmayan kol burada OutOfMemoryError verir, kampanyanın ortasında değil.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import replace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.config import PRESETS  # noqa: E402
from src.data_multi import MultiDomainDataset, multi_domain_paths  # noqa: E402
from src.flops import train_flops_per_token  # noqa: E402
from src.model import GPT  # noqa: E402
from src.tokenizer import load_tokenizer  # noqa: E402
from src.train import evaluate  # noqa: E402

GB = 1e9


def probe(preset: str, steps: int, eval_iters: int) -> dict:
    mcfg, tcfg = PRESETS[preset]
    tok = load_tokenizer(tcfg.data_dir)
    mcfg = replace(mcfg, vocab_size=tok.vocab_size)
    tcfg = replace(tcfg, eval_iters=eval_iters)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(101)
    train_ds = MultiDomainDataset(multi_domain_paths(tcfg.data_dir, "train"), mcfg.max_seq_len, "cuda")
    val_ds = MultiDomainDataset(multi_domain_paths(tcfg.data_dir, "val"), mcfg.max_seq_len, "cuda")
    model = GPT(mcfg).to("cuda")
    opt = model.configure_optimizer(tcfg.lr, tcfg.weight_decay, (tcfg.beta1, tcfg.beta2), "cuda")
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(101)
    out = {"preset": preset, "params_total": model.num_params(), "params_active": model.num_params(active_only=True),
           "flops_train_per_token": train_flops_per_token(mcfg), "batch_size": tcfg.batch_size,
           "grad_accum": tcfg.grad_accum}

    def train_steps(n: int) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            seqs = train_ds.sequence_stream(tcfg.batch_size * tcfg.grad_accum, gen)
            for i in range(0, len(seqs), tcfg.batch_size):
                x, y = train_ds.windows(seqs[i:i + tcfg.batch_size])
                # `src/train.py` ile BİREBİR aynı atama: `_` önceki mikro-batch'in logit tensörünü (bf16,
                # 16 x 512 x 32768 x 2 B = 0.54 GB) bir sonraki forward bitene kadar CANLI tutar. Gerçek koşuların
                # tepesi bu yüzden "yalın" bir eğitim adımının tepesinden ~0.5 GB yüksektir; prob bunu taklit
                # etmezse gerçek tepeyi az gösterir (ilk hâli dense'te 5.95 GB okudu, gerçek koşu 6.48 GB).
                with ctx:
                    _, loss, info = model(x, y)
                (loss / tcfg.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return n * tcfg.batch_size * tcfg.grad_accum * mcfg.max_seq_len / (time.perf_counter() - t0)

    train_steps(2)                                           # ısınma + AdamW state'in ayrılması
    out["tok_per_sec_probe"] = train_steps(steps)
    out["peak_train_gb"] = torch.cuda.max_memory_allocated() / GB
    ev = evaluate(model, val_ds, tcfg, ctx)
    out["peak_after_eval_gb"] = torch.cuda.max_memory_allocated() / GB
    train_steps(1)
    out["peak_gb"] = torch.cuda.max_memory_allocated() / GB
    out["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / GB
    out["eval_keys"] = sorted(k for k in ev if k != "raw")
    if mcfg.foton:
        out["foton_connections_counted"] = model.foton_connection_counts()
    del model, opt, train_ds, val_ds
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="foton-d,foton-m,foton-h,foton-x")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--eval_iters", type=int, default=3)
    ap.add_argument("--out", default="experiments/foton-v02/vram_probe.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA yok: VRAM ölçülemez")
    frac = PRESETS["foton-d"][1].vram_limit_frac
    torch.cuda.set_per_process_memory_fraction(frac)
    total = torch.cuda.get_device_properties(0).total_memory / GB
    res = {"gpu": torch.cuda.get_device_name(0), "total_gb": total, "vram_limit_frac": frac,
           "usable_gb": frac * total, "torch": torch.__version__, "steps": args.steps,
           "eval_iters": args.eval_iters, "arms": {}}
    for preset in args.presets.split(","):
        r = probe(preset.strip(), args.steps, args.eval_iters)
        r["margin_gb"] = res["usable_gb"] - r["peak_gb"]
        res["arms"][preset.strip()] = r
        print(f"{preset:12s} tepe {r['peak_gb']:.2f} GB (yalnız eğitim {r['peak_train_gb']:.2f}, eval sonrası "
              f"{r['peak_after_eval_gb']:.2f}) | ayrılmış {r['peak_reserved_gb']:.2f} | pay {r['margin_gb']:.2f} GB "
              f"| {r['tok_per_sec_probe']/1e3:.1f}k tok/s | FLOP/token {r['flops_train_per_token']:,}")
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"kullanılabilir tavan {res['usable_gb']:.2f} GB ({frac} x {total:.2f}) -> {args.out}")


if __name__ == "__main__":
    main()
