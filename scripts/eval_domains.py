"""Checkpoint'i ALAN BAŞINA değerlendirir (FOTON kampanyası: alan dağılımı raporlanması zorunlu).

    .venv\\Scripts\\python scripts/eval_domains.py --ckpt checkpoints/foton-v02/baseline-m-s11/best.pt

Neden ayrı betik: `src/train.py`'nin `evaluate()`'i karışık batch'lerden tek bir val loss üretir.
Ön-kaydın V11'i alan başına val loss'u zorunlu kılıyor, çünkü FOTON'un iddiası alan başına farklı
hesaplama yolu; hangi alanda kazandığı/kaybettiği birincil sonucu YORUMLAMAK için gerekli (teşhis).

Pencere sayısı ön-kayıtla aynı tutulur: alan başına batch_size x eval_iters = 1280 pencere,
sabit generator seed 12345. Yani bu ölçüm koşular arasında karşılaştırılabilir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.checkpoint import load_model  # noqa: E402
from src.data_multi import MultiDomainDataset, multi_domain_paths  # noqa: E402


def eval_per_domain(model, ds: MultiDomainDataset, batch_size: int, iters: int, device: str) -> dict:
    """Alan başına ortalama CE. Her alan için AYNI sayıda pencere ve aynı seed."""
    model.eval()
    out = {}
    ctx = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.no_grad():
        for dom in ds.names:
            g = torch.Generator().manual_seed(12345)     # ön-kayıtla aynı sabit seed
            tot = 0.0
            for _ in range(iters):
                x, y = ds.get_domain_batch(dom, batch_size, g)
                with ctx:
                    _, _, info = model(x, y)
                tot += info["ce"].item()
            out[dom] = tot / iters
    model.train()
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/foton")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_iters", type=int, default=80, help="batch_size x eval_iters = 1280 (ön-kayıt)")
    p.add_argument("--out", default="")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_model(args.ckpt, device=device)
    ds = MultiDomainDataset(multi_domain_paths(args.data, "val"), model.cfg.max_seq_len, device)
    per = eval_per_domain(model, ds, args.batch_size, args.eval_iters, device)
    mixed = sum(per.values()) / len(per)     # eşit alan ağırlığı = korpusun kendi oranı
    print(f"{args.ckpt} (adım {ckpt.get('step')}, kayıtlı val {ckpt.get('val_loss'):.6f})")
    print(f"  pencere/alan = {args.batch_size * args.eval_iters}")
    for dom, v in sorted(per.items()):
        print(f"  {dom:6s} {v:.6f}  (ppl {torch.tensor(v).exp().item():8.3f})")
    print(f"  {'eşit ağırlıklı ortalama':22s} {mixed:.6f}")
    res = {"ckpt": args.ckpt, "step": ckpt.get("step"), "recorded_val": ckpt.get("val_loss"),
           "per_domain": per, "equal_weight_mean": mixed,
           "windows_per_domain": args.batch_size * args.eval_iters}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
