"""v03 ADIM 1 karar kuralının ULAŞILABİLİRLİK sınavı (veri görülmeden; koşu yok).

    .venv/Scripts/python scripts/v03_step1_reachability.py --out experiments/foton-v03-step1/reachability.log

Kural (experiments/foton-v03-step1/decision_rule.md): seed başına eşli fark
    delta_s = val_atanan(X-iid, s) - val_atanan(X-dom, s)      [nat; pozitif = alan ataması daha iyi]
EVET  <=> ort_s(delta) >= ESIK_NAT  VE  her seed'de delta_s > 0.      (n = 2 seed: 101, 102)
Burada kuralın, makul bir gürültü aralığında (sigma = aynı seed'li iki koşunun val farkının sd'si) hangi gerçek etkide
hangi olasılıkla EVET dediği sayılır. sigma'nın kalibrasyonu v02 taramasından: aynı seed'li üç koşulun val@2188 farkları
0.0026 / 0.0045 / 0.0073 nat (regularizasyon farkı + deterministik olmayan çekirdekler) -> taranan aralık 0.003-0.008.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

SLOPE = -0.3346                                   # ön-kayıt V9, L* yerel eğimi (v02'nin 1.34 puanıyla aynı çevirme)
THRESH_POINTS = 3.0
THRESH_NAT = -abs(SLOPE) * math.log(1 - THRESH_POINTS / 100)      # 0.01019 nat
SEEDS = 2


def decide(deltas) -> bool:
    d = np.asarray(deltas, dtype=float)
    return bool(d.mean() >= THRESH_NAT and (d > 0).all())


def p_yes(effect: float, sigma: float, sims: int, rng) -> float:
    d = effect + rng.normal(0, sigma, (sims, SEEDS))
    return float(((d.mean(1) >= THRESH_NAT) & (d > 0).all(1)).mean())


def points(nat: float) -> float:
    return 100 * (1 - math.exp(-nat / abs(SLOPE)))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20260920)
    effects = [0.0, 0.0034, 0.0068, THRESH_NAT, 0.0137, 0.0172, 0.0260]     # 0, 1, 2, 3, 4, 5, 7.5 puan
    sigmas = [0.003, 0.005, 0.008]
    lines = [f"esik = {THRESH_POINTS} puan = {THRESH_NAT:.5f} nat (egim {SLOPE}); n = {SEEDS} seed; {args.sims} simulasyon",
             "P(EVET) | gercek etki (puan) x sigma (nat, esli fark)", "etki_puan  " + "  ".join(f"s={s:.3f}" for s in sigmas)]
    table = {}
    for e in effects:
        row = [p_yes(e, s, args.sims, rng) for s in sigmas]
        table[e] = row
        lines.append(f"{points(e):8.2f}   " + "  ".join(f"{v:7.4f}" for v in row))
    # ASSERTION'lar: kural ulaşılabilir VE yanlış alarm vermiyor olmalı — düzyazı değil
    assert all(v < 0.005 for v in table[0.0][:2]) and table[0.0][2] < 0.05, "etki yokken EVET olasiligi yuksek"
    assert all(v > 0.90 for v in table[0.0172][:2]), "5 puanlik gercek etkide kural EVET diyemiyor (ulasilamaz)"
    assert all(0.35 < v < 0.55 for v in table[THRESH_NAT]), "esikteki etki ~yari yariya olmali"
    assert all(v > 0.99 for v in table[0.0260]), "7.5 puanlik etkide EVET kesin olmali"
    lines.append("ASSERTION'lar gecti: etki 0 -> P(EVET) < %0.5 (s<=0.005) / < %5 (s=0.008); etki 5 puan -> > %90 (s<=0.005); 7.5 puan -> > %99")
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
