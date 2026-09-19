"""Router SEBEP incelemesi (ön-kayıt V6: erken uyarı ateşlenirse "sebep incelenir") — yalnızca TEŞHİS, karar kuralı değil.

    .venv\\Scripts\\python scripts/router_diag.py --runs checkpoints/foton-v02-tarama --tag v7 \\
        --out experiments/foton-v02/v7_router_diag.json

Salt-okunur: koşuların `eval_raw/step_*.npz` ham dökümlerini okur (üç modülün dizi başına ölçüt-pozisyonu CE'si,
router seçimi / olasılıkları / logit'leri, alan kimliği). Val pencereleri SABİT olduğu için (ön-kayıt V1-ii) aynı
1280 dizi her eval noktasında yeniden görülür; bu, eğitim loglarından hesaplanamayan iki şeyi ölçülebilir kılar:

* `flip_rate`: ardışık iki eval noktası arasında AYNI dizinin modül değiştirme oranı (ön-kaydın V6'da istediği
  "aynı dizinin modül değiştirmesi" — eğitim pencereleri tekrar etmediği için orada tanımsızdı; Sapma-S4).
* `best_same_as_prev`, `advantage_corr_prev`: router'ın öğrenmesi gereken HEDEF durağan mı — aynı dizinin en iyi
  modülü ardışık eval noktalarında aynı kalıyor mu (şans ~1/3), modül avantajları korele mi.
* `router_is_best`: router'ın seçtiği modülün o dizide GERÇEKTEN en düşük kayıplı modül olma oranı (şans = 1/3).

Sorulan soru: router neden kararsız / neden öğrenmiyor? İki aday sebep ayrışır:
  (a) ÖĞRENİLECEK BİR ŞEY YOK — modüller dizi düzeyinde ayırt edilemiyor: `spread` (dizi içinde en iyi ile en kötü
      modül arasındaki fark) küçük, `best_module_mi_domain` ~0, oracle kazancı null'ınkine eşit;
  (b) SİNYAL VAR AMA ROUTER ALAMIYOR — `spread` büyük ve en iyi modül alanla öngörülebilir, ama `router_is_best`
      şans düzeyinde ve logit'ler büyümüyor.
Betik val loss'u bir SEÇİM için kullanmaz; V7 kazananı `scripts/router_health.py`'den gelir.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

N_MODULES = 3


def mutual_information_bits(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> float:
    """I(a; b) bit cinsinden, ortak sayım tablosundan (eklenti tahmini; küçük örneklemde yukarı yanlıdır)."""
    table = np.zeros((na, nb), dtype=np.float64)
    np.add.at(table, (a, b), 1.0)
    p = table / table.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pa @ pb)[nz])).sum())


def mi_bias_bits(n: int, na: int, nb: int) -> float:
    """Bağımsızlık altında eklenti MI tahmininin beklenen değeri (Miller-Madow): (na-1)(nb-1) / (2 n ln 2)."""
    return (na - 1) * (nb - 1) / (2.0 * n * math.log(2))


def snapshot(raw: dict) -> dict:
    sm = np.asarray(raw["seq_masked"], dtype=np.float64)          # [N, 3]
    choice = np.asarray(raw["choice"]).astype(np.int64)
    logits = np.asarray(raw["logits"], dtype=np.float64)
    probs = np.asarray(raw["probs"], dtype=np.float64)
    dom = np.asarray(raw["domain"]).astype(np.int64)
    n = len(choice)
    if sm.shape != (n, N_MODULES) or logits.shape != (n, N_MODULES) or dom.shape != (n,):
        raise ValueError(f"ham döküm biçimi beklenmedik: seq_masked {sm.shape}, logits {logits.shape}, domain {dom.shape}")
    best = sm.argmin(1)
    top2 = np.sort(logits, axis=1)[:, -2:]
    n_dom = int(dom.max()) + 1 if dom.min() >= 0 else 1
    dom0 = np.where(dom >= 0, dom, 0)
    return {
        "n": n,
        "routed": float(sm[np.arange(n), choice].mean()),
        "oracle": float(sm.min(1).mean()),
        "module_means": [float(v) for v in sm.mean(0)],
        "spread": float((sm.max(1) - sm.min(1)).mean()),
        "router_is_best": float((best == choice).mean()),
        "choice_counts": np.bincount(choice, minlength=N_MODULES).tolist(),
        "best_counts": np.bincount(best, minlength=N_MODULES).tolist(),
        "choice_mi_domain_bits": mutual_information_bits(dom0, choice, n_dom, N_MODULES),
        "best_module_mi_domain_bits": mutual_information_bits(dom0, best, n_dom, N_MODULES),
        "mi_bias_bits": mi_bias_bits(n, n_dom, N_MODULES),
        "logit_std": float(logits.std()),
        "margin": float((top2[:, 1] - top2[:, 0]).mean()),
        "max_prob": float(probs.max(1).mean()),
        "_choice": choice,
        "_best": best,
        "_adv": sm - sm.mean(1, keepdims=True),
    }


def run_diag(run_dir: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(run_dir, "eval_raw", "step_*.npz")))
    if not files:
        raise ValueError(f"{run_dir}: eval_raw dökümü yok (FOTON koşusu değil mi?)")
    out, prev, prev_best, prev_adv = [], None, None, None
    for f in files:
        with np.load(f) as z:
            snap = snapshot({k: z[k] for k in ("seq_masked", "choice", "logits", "probs", "domain")})
            step = int(z["step"])
        choice = snap.pop("_choice")
        # aynı sabit val dizilerinin bir önceki eval noktasına göre modül değiştirme oranı
        snap["flip_rate"] = None if prev is None or len(prev) != len(choice) else float((prev != choice).mean())
        prev = choice
        # ÖĞRENİLECEK HEDEF durağan mı: aynı dizinin en iyi modülü bir önceki eval noktasındakiyle aynı mı (şans ~1/3),
        # ve modül avantajları (loss_m - üç modülün ortalaması) ardışık eval noktaları arasında korele mi
        best, adv = snap.pop("_best"), snap.pop("_adv")
        same_n = prev_best is not None and len(prev_best) == len(best)
        snap["best_same_as_prev"] = float((prev_best == best).mean()) if same_n else None
        defined = same_n and adv.std() > 0 and prev_adv.std() > 0       # sabit avantajda korelasyon tanımsız (NaN yazma)
        snap["advantage_corr_prev"] = float(np.corrcoef(adv.ravel(), prev_adv.ravel())[0, 1]) if defined else None
        prev_best, prev_adv = best, adv
        out.append({"step": step, **snap})
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = sorted(d for d in glob.glob(os.path.join(args.runs, f"{args.tag}-*")) if os.path.isdir(d)
                  and ".blok-iptal-" not in d and ".yarim-" not in d)
    res = {"what": "router sebep incelemesi (teşhis; karar kuralı DEĞİL)", "chance_router_is_best": 1 / N_MODULES, "runs": {}}
    for d in runs:
        name = os.path.basename(d)
        try:
            res["runs"][name] = run_diag(d)
        except ValueError as e:
            print(f"atlandı: {e}")
            continue
        last = res["runs"][name][-1]
        print(f"{name:16s} adım {last['step']:5d} | routed {last['routed']:.4f} oracle {last['oracle']:.4f} | "
              f"yayılım {last['spread']:.4f} | router=en iyi {last['router_is_best']:.3f} | "
              f"I(alan;seçim) {last['choice_mi_domain_bits']:.3f} bit | I(alan;en iyi) {last['best_module_mi_domain_bits']:.4f} bit "
              f"(yanlılık {last['mi_bias_bits']:.4f}) | değiştirme {last['flip_rate']} | maks p {last['max_prob']:.3f} | "
              f"en iyi modül aynı kaldı {last['best_same_as_prev']} | avantaj kor. {last['advantage_corr_prev']}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("->", args.out)


if __name__ == "__main__":
    main()
