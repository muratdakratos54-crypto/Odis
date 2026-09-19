"""EŞİT TOKEN bağlam sayıları: tarama koşuları (kol X, %50 bütçe) ile dondurulmuş D / M baseline'ları (BAĞLAM; karar değil).

    .venv\\Scripts\\python scripts/screening_context.py --out experiments/foton-v02/screening_context.json

Ön-kayıt V11 "D'ye karşı bağlam sayıları (ii)": eşit token bütçesinde FOTON − D val farkı (hedefe-ulaşma DEĞİL, aynı-bütçe
karşılaştırması). Karar kampanyası koşulmadığı için tek FOTON verisi tarama koşularıdır; burada ortak eval adımlarında
(ölçüt: pozisyon 64..511, dört kolda aynı kod) kol ortalamaları yan yana konur.

OKUMA SINIRLARI — raporda da yazılır: tarama seed'leri {101,102,103}, baseline seed'leri {11,..,55}: EŞLEŞTİRİLMEMİŞ,
farklı başlangıç ve veri sırası; n = 3'e karşı 5; yalnızca bütçenin ilk yarısı (adım <= 2000); tarama koşusu karar
koşusunun yörüngesinin ilk yarısıdır (LR schedule tam bütçe üzerinden), baseline'lar da aynı schedule'ı izler. Bu sayılar
karar tablosuna GİRMEZ ve "D'yi / M'i yendik / yenemedik" cümlesi kurmaz; yalnızca oracle tavanının yanına ölçek koyar.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GROUPS = {"D": "checkpoints/foton-v02r/r-d-s*", "M": "checkpoints/foton-v02r/r-m-s*",
          "X-none": "checkpoints/foton-v02-tarama/v7-x-s1*", "X-entropy": "checkpoints/foton-v02-tarama/v7-x-ent-s1*",
          "X-lb": "checkpoints/foton-v02-tarama/v7-x-lb-s1*"}
STEPS = (500, 1000, 1500, 2000)


def val_curve(run_dir: str) -> dict[int, float]:
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    skips = {r.get("loss_prefix_skip") for r in rows if r.get("type") == "val"}
    if skips != {64}:
        raise ValueError(f"{run_dir}: val satırları prefix maskeli ölçütte değil (loss_prefix_skip = {skips})")
    return {r["step"]: r["val_loss"] for r in rows if r.get("type") == "val"}


def summarize(curves: dict[str, dict[str, dict[int, float]]], steps=STEPS) -> dict:
    """curves: {grup: {koşu adı: {adım: val}}} — koşu ADIYLA anahtarlı; eksik adım sessizce atlanmaz, hata verir."""
    out = {}
    for step in steps:
        row = {}
        for g, runs in curves.items():
            v = np.array([c[step] for c in runs.values()])
            row[g] = {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                      "runs": {name: c[step] for name, c in runs.items()}}
        for g in [k for k in row if k.startswith("X")]:
            row[g]["minus_D"] = row[g]["mean"] - row["D"]["mean"]
            row[g]["minus_M"] = row[g]["mean"] - row["M"]["mean"]
        out[str(step)] = row
    return out


def mean_curve(runs: dict[str, dict[int, float]]) -> dict[int, float]:
    """Grubun ortalama val eğrisi — yalnızca BÜTÜN koşularda bulunan adımlarda."""
    common = set.intersection(*(set(c) for c in runs.values()))
    return {s: float(np.mean([c[s] for c in runs.values()])) for s in sorted(common)}


def steps_to_reach(curve: dict[int, float], loss: float) -> float | None:
    """`curve`'ün `loss`'a ulaştığı adım — ardışık eval noktaları arasında ln(adım)'da doğrusal enterpolasyon.
    (Dört kolda token/adım aynıdır: adım oranı = token oranı.) Eğri o kayba ulaşmıyorsa None."""
    pts = [(s, v) for s, v in sorted(curve.items()) if s > 0]
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if v0 >= loss >= v1 and v0 != v1:
            w = (v0 - loss) / (v0 - v1)
            return float(np.exp(np.log(s0) + w * (np.log(s1) - np.log(s0))))
    return None


def equal_loss_token_ratio(x_curve: dict[int, float], base_curve: dict[int, float], steps) -> dict:
    """X'in `adım`da ulaştığı kayba baseline kaç adımda ulaştı -> X'in harcadığı token / baseline'ın harcadığı token.
    Nat farkı düzleşen eğride kendiliğinden küçülür; birincil ölçütün birimi (hedefe ulaşmak için gereken compute) budur."""
    out = {}
    for s in steps:
        if s in x_curve:
            b = steps_to_reach(base_curve, x_curve[s])
            out[str(s)] = None if b is None else s / b
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    curves = {}
    for g, pattern in GROUPS.items():
        dirs = sorted(d for d in glob.glob(os.path.join(ROOT, pattern)) if ".blok-iptal-" not in d and ".yarim-" not in d)
        if not dirs:
            sys.exit(f"{g}: koşu bulunamadı ({pattern})")
        curves[g] = {os.path.basename(d): val_curve(d) for d in dirs}
    res = {"what": "eşit token bağlam sayıları (V11-ii); karar tablosuna GİRMEZ",
           "caveats": "eşleştirilmemiş seed'ler (tarama 101-103, baseline 11-55); n = 3'e karşı 5; yalnızca bütçenin ilk yarısı",
           "metric": "val_loss, pozisyon 64..511 (loss_prefix_skip = 64)", "by_step": summarize(curves)}
    xs, ds, ms = mean_curve(curves["X-none"]), mean_curve(curves["D"]), mean_curve(curves["M"])
    ratio_steps = [s for s in sorted(xs) if s >= 750]
    res["equal_loss_token_ratio_X_none"] = {"vs_D": equal_loss_token_ratio(xs, ds, ratio_steps),
                                            "vs_M": equal_loss_token_ratio(xs, ms, ratio_steps),
                                            "note": "X'in harcadığı token / baseline'ın aynı kayba ulaşmak için harcadığı token (ortalama eğriler, ln-adım enterpolasyonu)"}
    # V11 bağlam sayısı (i): D'nin tam bütçe sonundaki val loss'u ve L*'e eksiği — yalnızca dondurulmuş baseline'lardan
    with open(os.path.join(ROOT, "experiments", "foton-v02-prereg", "lstar.json"), encoding="utf-8") as f:
        lstar_doc = json.load(f)
    lstar = lstar_doc["lstar"]
    end = {name: c[max(c)] for name, c in curves["D"].items()}
    res["D_full_budget"] = {"final_step": max(max(c) for c in curves["D"].values()), "runs": end,
                            "mean": float(np.mean(list(end.values()))), "L_star": lstar,
                            "deficit_to_L_star": None if lstar is None else float(np.mean(list(end.values())) - lstar)}
    for step, row in res["by_step"].items():
        print(f"adım {step}: " + " | ".join(f"{g} {v['mean']:.4f} (n={v['n']})" for g, v in row.items())
              + f" || X-none − D = {row['X-none']['minus_D']:+.4f}, X-none − M = {row['X-none']['minus_M']:+.4f} nat")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("->", args.out)


if __name__ == "__main__":
    main()
