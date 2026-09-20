"""v03 ADIM 1 karar kuralının ULAŞILABİLİRLİK sınavı (veri görülmeden; koşu yok) — n = 3 seed, iki soru.

    .venv/Scripts/python scripts/v03_step1_reachability.py --out experiments/foton-v03-step1/reachability.log

Kuralın TEK kopyası `scripts/v03_step1_decision.py`'dedir; burada o fonksiyonlar simüle edilmiş seed-anahtarlı farklarla
çağrılır (kural burada yeniden yazılmaz). Sorulan: makul bir gürültü aralığında kural hangi GERÇEK etkide hangi olasılıkla
EVET der — sahte EVET üretiyor mu, gerçek bir etkiyi yakalayabiliyor mu, ve 3–4 puan bölgesinde yazı-tura mı.

sigma = eşli farkın seed başına sd'si. Kalibrasyon (v02 taraması, aynı seed'li üç koşulun val@2188 farkları): 0.0026 /
0.0045 / 0.0073 nat -> (a) için taranan aralık 0.003–0.008. (b)'de D ile X-dom FARKLI mimarilerdir (aynı seed, aynı veri):
eşli farkın sd'si bilinmiyor; v02'de seed'ler arası val sd'si D 0.0027, X 0.0085 idi -> taranan aralık 0.003–0.010.

ÖN KOŞUL (X-iid, aynı seed'li v02-X'ten sapmamalı) da bir durdurma kuralıdır, o da sınanır: X-iid ile v02-X arasındaki GERÇEK
sistematik kayma (nat) x seed başına sd. Sorulan: kayma yokken kaç kez yanlış alarm verir; (a) eşiği kadar (0.010 nat) bir
kaymayı kaç kez "temiz" sayar. İşaretli-ortalama bacağı OLMADAN aynı sayı yan yana yazılır (o bacak incelemeden sonra eklendi).
Kalibrasyon: v02'nin aynı seed'li koşullarının eşli farkları 0.0026 / 0.0045 / 0.0073 nat -> taranan aralık 0.003–0.0075;
v02-X'te modül yükleri dengeliydi (pay 0.33 / 0.34 / 0.33), yani v02-X'in modülleri de ~iid veri gördü: iki taraf kıyaslanabilir.

Beklentiler ASSERTION'dır (düzyazı değil): başarısız olan log'a "KALDI" diye yazılır ve çıkış kodu 1 olur.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v03_step1_decision as dec  # noqa: E402

SIG_A, SIG_B = (0.003, 0.005, 0.008), (0.003, 0.006, 0.010)
EFFECTS_A, EFFECTS_B = (0, 1, 2, 2.5, 3, 3.5, 4, 5, 7.5), (-14, -5, 0, 2.5, 4, 5, 6, 7.5, 10)   # -14: v02'de X, D'nin 0.050 nat gerisinde
SIG_P, SHIFTS_P = (0.003, 0.0045, 0.0075), (0.0, 0.005, 0.0075, 0.010, 0.015)
# D akla-yatkınlık denetimi: yeni D'lerin dondurulmuş referanstan GERÇEK sistematik kayması x D'nin seed'ler arası sd'si
# (ölçülen 0.0027, n = 5; iki katına kadar tarandı)
SIG_D, SHIFTS_D = (0.0027, 0.004, 0.0055), (0.0, 0.005, 0.010, 0.015, 0.020)
# maske teşhisi: val(H-dom) − val(X-dom)'un GERÇEK değeri x eşli farkın sd'si
SIG_M, DIFFS_M = (0.003, 0.005, 0.008), (0.0, 0.0025, 0.005, 0.0075, 0.010)


def effect_nat(points: float) -> float:
    return dec.points_of_nat(0.0) if points == 0 else (dec.nat_of_points(points) if points > 0 else -dec.nat_of_points(-points))


def p_pass(rule, points: float, sigma: float, sims: int, rng, seeds=dec.SEEDS) -> float:
    draws = effect_nat(points) + rng.normal(0.0, sigma, (sims, len(seeds)))
    return float(np.mean([rule(dict(zip(seeds, row)))["passed"] for row in draws]))


def p_clean(shift: float, sigma: float, sims: int, rng, mean_leg: bool = True) -> float:
    """P(ön koşul 'temiz') — yalnızca mean_fixed bacakları (tavan farkı 0 verilir). `mean_leg=False`: eklenen bacak OLMADAN."""
    zero = dict.fromkeys(dec.SEEDS, 0.0)
    draws = shift + rng.normal(0.0, sigma, (sims, len(dec.SEEDS)))
    if not mean_leg:
        return float(np.mean((np.abs(draws) <= dec.PRE_MEAN_FIXED_NAT).all(1)))
    return float(np.mean([dec.precondition(dict(zip(dec.SEEDS, row)), zero, zero, zero)["clean"] for row in draws]))


def p_d_clean(shift: float, sigma: float, sims: int, rng, mean_leg: bool = True) -> float:
    draws = dec.D_REF_VAL + shift + rng.normal(0.0, sigma, (sims, len(dec.SEEDS)))
    if not mean_leg:
        return float(np.mean((np.abs(draws - dec.D_REF_VAL) <= dec.D_REF_TOL).all(1)))
    return float(np.mean([dec.d_sanity(dict(zip(dec.SEEDS, row)))["clean"] for row in draws]))


def p_measurable(diff: float, sigma: float, sims: int, rng, sign_leg: bool = True) -> float:
    draws = diff + rng.normal(0.0, sigma, (sims, len(dec.SEEDS)))
    if not sign_leg:
        return float(np.mean(np.abs(draws.mean(1)) >= dec.MASK_VS_WEIGHT_NAT))
    zero = dict.fromkeys(dec.SEEDS, 0.0)
    return float(np.mean([dec.mask_vs_weight(dict(zip(dec.SEEDS, row)), zero)["measurable"] for row in draws]))


def checks(ta: dict, tb: dict, tp: dict, td: dict, tm: dict) -> list[tuple[str, bool]]:
    return [
        ("D akla-yatkınlık: kayma 0 -> P(temiz) >= %99 (s=0.0027), >= %95 (s=0.004), >= %80 (s=0.0055)",
         td[0.0][0] >= 0.99 and td[0.0][1] >= 0.95 and td[0.0][2] >= 0.80),
        ("D akla-yatkınlık: kayma 0.010 nat -> P(temiz) <= %15; kayma 0.015 -> <= %1",
         all(v <= 0.15 for v in td[0.010]) and all(v <= 0.01 for v in td[0.015])),
        ("maske teşhisi: gerçek fark 0 -> P('ölçülebilir') <= %8 (s<=0.005), <= %20 (s=0.008)",
         all(v <= 0.08 for v in tm[0.0][:2]) and tm[0.0][2] <= 0.20),
        ("maske teşhisi: gerçek fark 0.010 nat -> P('ölçülebilir') >= %85 (s<=0.005), >= %60 (s=0.008)",
         all(v >= 0.85 for v in tm[0.010][:2]) and tm[0.010][2] >= 0.60),
        ("ön koşul: kayma 0 -> P(temiz) >= %99 (s=0.003), >= %97 (s=0.0045), >= %75 (s=0.0075)",
         tp[0.0][0] >= 0.99 and tp[0.0][1] >= 0.97 and tp[0.0][2] >= 0.75),
        ("ön koşul: kayma 0.010 nat ((a) eşiği kadar) -> P(temiz) <= %30; kayma 0.015 -> <= %5",
         all(v <= 0.30 for v in tp[0.010]) and all(v <= 0.05 for v in tp[0.015])),
        ("(a) etki 0 -> P(EVET) <= %0.2 (s<=0.005), <= %2 (s=0.008)", all(v <= 0.002 for v in ta[0][:2]) and ta[0][2] <= 0.02),
        ("(a) etki 1 puan -> <= %2 (s<=0.005), <= %8 (s=0.008)", all(v <= 0.02 for v in ta[1][:2]) and ta[1][2] <= 0.08),
        ("(a) etki 3 puan (eşik) -> %40-60", all(0.40 <= v <= 0.60 for v in ta[3])),
        ("(a) etki 4 puan -> >= %85 (s<=0.005), >= %70 (s=0.008)", all(v >= 0.85 for v in ta[4][:2]) and ta[4][2] >= 0.70),
        ("(a) etki 5 puan -> >= %95 (s<=0.005), >= %90 (s=0.008)", all(v >= 0.95 for v in ta[5][:2]) and ta[5][2] >= 0.90),
        ("(b) X-dom 14 puan geride (v02'deki X) -> <= %0.1; eşit -> <= %0.2 (s<=0.006), <= %2 (s=0.010)",
         all(v <= 0.001 for v in tb[-14]) and all(v <= 0.002 for v in tb[0][:2]) and tb[0][2] <= 0.02),
        ("(b) +7.5 puan -> >= %95 (s<=0.006); +10 puan -> >= %99 (s<=0.006)",
         all(v >= 0.95 for v in tb[7.5][:2]) and all(v >= 0.99 for v in tb[10][:2])),
    ]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20260920)
    lines = [f"n = {len(dec.SEEDS)} seed {dec.SEEDS}; {args.sims} simülasyon/hücre; çevirim eğimi {dec.SLOPE}",
             f"(a) MEKANİZMA eşiği {dec.A_POINTS} puan = {dec.A_NAT:.6f} nat | (b) KAPI eşiği {dec.B_POINTS} puan = {dec.B_NAT:.6f} nat",
             "kural: ort_s(fark) >= eşik VE üç seed'de de fark > 0", ""]
    ta = {p: [p_pass(dec.question_a, p, s, args.sims, rng) for s in SIG_A] for p in EFFECTS_A}
    lines += ["(a) P(EVET) | gerçek etki (puan) x sigma (nat)", "etki   " + "  ".join(f"s={s:.3f}" for s in SIG_A)]
    lines += [f"{p:5.1f}  " + "  ".join(f"{v:7.4f}" for v in row) for p, row in ta.items()]
    tb = {p: [p_pass(dec.gate_b, p, s, args.sims, rng) for s in SIG_B] for p in EFFECTS_B}
    lines += ["", "(b) P(KAPI GEÇİLDİ) | X-dom'un eşli D'ye gerçek üstünlüğü (puan; negatif = geride) x sigma (nat)",
              "etki   " + "  ".join(f"s={s:.3f}" for s in SIG_B)]
    lines += [f"{p:5.1f}  " + "  ".join(f"{v:7.4f}" for v in row) for p, row in tb.items()]

    # n = 2 ile kıyas ((a), aynı kural iki seed'le, sigma = 0.005): 3-4 puan bölgesi ne kadar daraldı
    def two_seed(points: float) -> float:
        rows = effect_nat(points) + rng.normal(0.0, 0.005, (args.sims, 2))
        return float(np.mean((rows.mean(1) >= dec.A_NAT) & (rows > 0).all(1)))
    lines += ["", "n = 2 ile kıyas ((a), sigma = 0.005): " + ", ".join(f"{p} puan -> n=2 {two_seed(p):.3f} / n=3 {ta[p][1]:.3f}" for p in (2, 3.5, 4, 5))]

    tp = {sh: [p_clean(sh, s, args.sims, rng) for s in SIG_P] for sh in SHIFTS_P}
    old = {sh: [p_clean(sh, s, args.sims, rng, mean_leg=False) for s in SIG_P] for sh in SHIFTS_P}
    lines += ["", f"ÖN KOŞUL P(temiz) | X-iid − v02-X gerçek kayması (nat) x sigma (nat); sınırlar: seed başına {dec.PRE_MEAN_FIXED_NAT}, "
                  f"işaretli ortalama {dec.PRE_MEAN_SHIFT_NAT}", "kayma    " + "  ".join(f"s={s:.4f}" for s in SIG_P)
              + "   | işaretli-ortalama bacağı OLMADAN"]
    lines += [f"{sh:6.4f}  " + "  ".join(f"{v:8.4f}" for v in tp[sh]) + "   | " + "  ".join(f"{v:8.4f}" for v in old[sh]) for sh in SHIFTS_P]

    td = {sh: [p_d_clean(sh, s, args.sims, rng) for s in SIG_D] for sh in SHIFTS_D}
    old_d = {sh: [p_d_clean(sh, s, args.sims, rng, mean_leg=False) for s in SIG_D] for sh in SHIFTS_D}
    lines += ["", f"D AKLA-YATKINLIK P(temiz) | yeni D'lerin referanstan gerçek kayması (nat) x D'nin seed'ler arası sd'si; sınırlar: seed "
                  f"başına {dec.D_REF_TOL}, işaretli ortalama {dec.D_REF_MEAN_TOL}", "kayma    " + "  ".join(f"s={s:.4f}" for s in SIG_D)
              + "   | işaretli-ortalama bacağı OLMADAN"]
    lines += [f"{sh:6.4f}  " + "  ".join(f"{v:8.4f}" for v in td[sh]) + "   | " + "  ".join(f"{v:8.4f}" for v in old_d[sh]) for sh in SHIFTS_D]

    tm = {df: [p_measurable(df, s, args.sims, rng) for s in SIG_M] for df in DIFFS_M}
    old_m = {df: [p_measurable(df, s, args.sims, rng, sign_leg=False) for s in SIG_M] for df in DIFFS_M}
    lines += ["", f"MASKE TEŞHİSİ P('ölçülebilir') | val(H-dom) − val(X-dom) gerçek değeri (nat) x sigma (nat); taban {dec.MASK_VS_WEIGHT_NAT} "
                  "+ üç seed'de aynı işaret", "fark     " + "  ".join(f"s={s:.4f}" for s in SIG_M) + "   | aynı-işaret şartı OLMADAN"]
    lines += [f"{df:6.4f}  " + "  ".join(f"{v:8.4f}" for v in tm[df]) + "   | " + "  ".join(f"{v:8.4f}" for v in old_m[df]) for df in DIFFS_M]

    res = checks(ta, tb, tp, td, tm)
    lines += [""] + [f"{'GEÇTİ' if ok else '*** KALDI ***'}  {name}" for name, ok in res]
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not all(ok for _, ok in res):
        sys.exit(1)


if __name__ == "__main__":
    main()
