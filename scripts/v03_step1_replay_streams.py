"""v03 adım 1 — KOŞUDAN ÖNCE akış provası (CPU, eğitim yok, model yok): veri sırası ve X-iid atama akışı yeniden oynatılır.

    .venv/Scripts/python scripts/v03_step1_replay_streams.py --out experiments/foton-v03-step1/stream_replay.log

İki akış da yalnızca seed'e bağlıdır (veri: `train_gen`; atama: ayrı generator, tohum = seed + 7,777,777), yani 12 koşunun
göreceği veri sırası ve X-iid'in uygulayacağı alan x modül tablosu koşulardan ÖNCE bilinir. Bu prova iki şeyi sınar:

  1. Analizin eşleme denetimi GEÇERLİ bir kampanyayı reddetmeyecek: yeniden oynatılan veri sırasının log aralığı başına alan
     sayımları, aynı seed'li v02-X koşusunun logladığıyla (88 aralığın 88'inde) birebir aynı olmalı.
  2. Analizin X-iid pay toleransı (`IID_SHARE_TOL`) GEÇERLİ bir X-iid koşusunu reddetmeyecek: gerçek tablonun 1/3'ten en büyük
     sapması toleransın yarısından küçük olmalı.

Beklentiler ASSERTION'dır: biri tutmazsa log'a KALDI yazılır ve çıkış kodu 1 olur.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v03_step1_analyze as an  # noqa: E402
from src import train  # noqa: E402
from src.config import PRESETS  # noqa: E402
from src.data_multi import MultiDomainDataset, multi_domain_paths  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--v02", default=os.path.join(ROOT, "checkpoints", "foton-v02-tarama"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tcfg = PRESETS["foton-x-iid"][1]
    ds = MultiDomainDataset(multi_domain_paths(os.path.join(ROOT, "data", "foton"), "train"), 512, "cpu")
    lines = [f"alanlar {ds.names} | adım başına {tcfg.batch_size} x {tcfg.grad_accum} dizi | {an.END_STEP} adım | log aralığı {tcfg.log_interval}", ""]
    checks = []
    for seed in an.dec.SEEDS:
        gen = torch.Generator().manual_seed(seed)
        assigner, _ = train.make_assigner("iid", 3, ds.names, seed=seed)
        table, interval, order = np.zeros((3, 3), dtype=np.int64), np.zeros(3, dtype=np.int64), []
        for step in range(1, an.END_STEP + 1):
            doms = np.array([ds.domain_id(d) for d, _ in ds.sequence_stream(tcfg.batch_size * tcfg.grad_accum, gen)])
            assign = torch.cat([assigner(None, tcfg.batch_size, "cpu") for _ in range(tcfg.grad_accum)]).numpy()
            np.add.at(table, (doms, assign), 1)
            interval += np.bincount(doms, minlength=3)
            if step % tcfg.log_interval == 0 or step >= an.END_STEP:
                order.append(interval.tolist())
                interval[:] = 0
        with open(os.path.join(args.v02, f"v7-x-s{seed}", "metrics.jsonl"), encoding="utf-8") as f:
            logged = [np.asarray(r["router_domain_counts"]).sum(axis=1).tolist() for r in map(json.loads, f) if r.get("type") == "train"]
        same = sum(a == b for a, b in zip(order, logged))
        worst = float(np.abs(table / table.sum(axis=1, keepdims=True) - 1.0 / 3.0).max())
        lines.append(f"seed {seed}: veri sırası v02-X loguyla {same}/{len(logged)} aralıkta aynı | X-iid alan x modül tablosu {table.tolist()} "
                     f"| 1/3'ten en büyük pay sapması {worst:.5f} (tolerans {an.IID_SHARE_TOL})")
        checks += [(f"seed {seed}: veri sırası bütün log aralıklarında v02-X'inkiyle aynı", len(order) == len(logged) and same == len(logged)),
                   (f"seed {seed}: X-iid pay sapması toleransın yarısından küçük", worst < an.IID_SHARE_TOL / 2),
                   (f"seed {seed}: toplam dizi {an.END_STEP * an.SEQS_PER_STEP}", int(table.sum()) == an.END_STEP * an.SEQS_PER_STEP)]
    lines += [""] + [f"{'GEÇTİ' if ok else '*** KALDI ***'}  {name}" for name, ok in checks]
    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    if not all(ok for _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
