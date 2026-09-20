"""v03 adım 1 — EĞİTİMSİZ GPU sağlaması: zorlanmış kolların eval yolu GERÇEK ağırlıklarla, bf16 / CUDA'da iki yoldan aynı mı.

    .venv/Scripts/python scripts/v03_step1_gpu_two_route.py --out experiments/foton-v03-step1/gpu_two_route.log

CPU testleri (tests/test_foton_assign.py) iki yolun eşitliğini minik bir modelde fp32'de sınar. Karar koşuları ise bf16 /
CUDA'da, 55M parametreli modelle koşar. İki yol 1e-3'ü aşarsa (ya da sağlama sonlu değilse) koşu KENDİNİ DURDURUR (train.py);
1e-4'ü aşarsa koşu sürer ama ANALİZ bütün kampanyayı reddeder — yani 1e-4 ile 1e-3 arasındaki bir fark 5 GPU saatini yakar.
İkisi de burada, koşudan ÖNCE görülür: bu betiğin eşiği analizinkidir (1e-4). Eğitim YOK: bitmiş bir v02-X koşusunun ağırlıkları zorlanmış
atama modeline yüklenir (aynı parametreler; tek fark `foton_assign`) ve yalnızca `train.evaluate()` çağrılır.

  1. yol  : eğitimin forward yolu, `model(x, y, assign=...)` (X-dom: alan kimliği; X-iid: m = 0..M-1 sabit atamalarının ortalaması)
  2. yol  : üç-modül geçişi `foton_eval_all` (`assigned_check`) ve ondan bağımsız olarak ham dökümden sonradan hesap
  çapa   : aynı ağırlıklarla oracle, v02 koşusunun kendi logladığı oracle'ı yeniden üretmeli (ağırlıklar doğru yüklendi mi)

Beklentiler ASSERTION'dır: biri tutmazsa log'a KALDI yazılır ve çıkış kodu 1 olur.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v03_step1_analyze as an  # noqa: E402
from src import train  # noqa: E402
from src.config import PRESETS, ModelConfig  # noqa: E402
from src.data_multi import MultiDomainDataset, multi_domain_paths  # noqa: E402
from src.model import GPT  # noqa: E402

TOL = 1e-4


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "checkpoints", "foton-v02-tarama", "v7-x-s101"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA yok: bu sağlama karar koşularının koştuğu yolu (bf16 / CUDA) sınar")
    with open(os.path.join(args.run, "metrics.jsonl"), encoding="utf-8") as f:
        logged = [r for r in map(json.loads, f) if r.get("type") == "val"][-1]
    state = torch.load(os.path.join(args.run, "last.pt"), map_location="cpu", weights_only=False)
    if state["step"] != logged["step"]:
        sys.exit(f"last.pt adım {state['step']}, son eval adım {logged['step']}: çapa kurulamaz")
    tcfg = PRESETS["foton-x-dom"][1]
    val_ds = MultiDomainDataset(multi_domain_paths(os.path.join(ROOT, "data", "foton"), "val"), 512, "cuda")
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    lines = [f"ağırlıklar: {os.path.relpath(args.run, ROOT)} adım {state['step']} | {torch.cuda.get_device_name(0)} | torch {torch.__version__} | bf16",
             f"koşunun kendi logu: val_loss (router) {logged['val_loss']:.6f}, oracle {logged['val_loss_oracle']:.6f}", ""]
    checks, digests = [], set()
    for mode in ("domain", "iid"):
        cfg = replace(ModelConfig(**state["model_cfg"]), foton_assign=mode)
        model = GPT(cfg)
        missing = model.load_state_dict(state["model"], strict=False)
        assert not missing.missing_keys and not missing.unexpected_keys, missing     # aynı parametreler, birebir
        model.cuda()
        torch.cuda.reset_peak_memory_stats()
        ev = train.evaluate(model, val_ds, tcfg, ctx)
        sm, dom = ev["raw"]["seq_masked"].astype(np.float64), ev["raw"]["domain"]
        posthoc = float(sm[np.arange(len(dom)), dom].mean() if mode == "domain" else sm.mean())
        d_check, d_post = abs(ev["primary"] - ev["assigned_check"]), abs(ev["primary"] - posthoc)
        lines.append(f"{mode:7s} 1. yol (eğitimin forward'ı) {ev['primary']:.6f} | 2. yol assigned_check {ev['assigned_check']:.6f} "
                     f"(fark {d_check:.2e}) | ham dökümden {posthoc:.6f} (fark {d_post:.2e}) | router_UNUSED {ev['router_unused']:.6f} "
                     f"| oracle {ev['oracle']:.6f} | eval tepe bellek {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        checks += [(f"{mode}: iki yol {TOL} içinde (assigned_check ve ham döküm)", d_check <= TOL and d_post <= TOL),
                   (f"{mode}: oracle, v02 koşusunun logladığını yeniden üretiyor ({TOL} içinde)",
                    abs(ev["oracle"] - logged["val_loss_oracle"]) <= TOL),
                   (f"{mode}: router'ın sayısı (UNUSED), v02 koşusunun router'lı val_loss'unu yeniden üretiyor ({TOL} içinde)",
                    abs(ev["router_unused"] - logged["val_loss"]) <= TOL)]
        digests.add(ev["val_windows_sha256"])
        del model
    # Val pencerelerinin içerik özeti: oracle v02 koşusunun logunu yeniden ürettiğine göre bu, v02'nin ölçüldüğü val kümesidir.
    # Analiz 12 karar koşusunun (D dahil) özetini BU değerle karşılaştırır (`v03_step1_analyze.VAL_WINDOWS_SHA256`).
    lines += ["", f"val_windows_sha256 = {sorted(digests)}"]
    checks.append(("val penceresi özeti iki kolda aynı ve analizin beklediği değer", digests == {an.VAL_WINDOWS_SHA256}))
    lines += [""] + [f"{'GEÇTİ' if ok else '*** KALDI ***'}  {name}" for name, ok in checks]
    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    if not all(ok for _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
