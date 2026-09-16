"""FOTON v0.1 ön-kaydının güç (power) kalibrasyonu — YALNIZCA baseline gürültüsünden.

    .venv\\Scripts\\python scripts/power_analysis.py [--out experiments/foton-v01-prereg/calibration.log]

Soru: ön-kayıttaki "+%5 tasarruf ve %95 GA alt sınırı > 0" kuralı, 5 eşleştirilmiş seed ile GERÇEKTEN
saptanabilir mi, yoksa gözlenen seed gürültüsü zaten o mertebede olduğu için tasarım garantili "belirsiz"
mi üretir? FOTON koşulmadan cevaplanabilir, çünkü yalnızca dondurulmuş baseline koşularının varyansını
kullanıyor.

Dört bölüm:
  1. Koşullar içi (within-condition) seed varyansı: val loss (nat) ve hedefe-ulaşma-hesabı N* (%).
  2. Eşleştirilmiş (paired) vs eşleştirilmemiş (unpaired) fark varyansı — eşleştirme ne kazandırıyor.
  3. Karar kuralının gücü: Monte Carlo, ön-kayıttaki BİLEŞİK kuralla (s̄ >= 5 VE GA alt sınırı > 0).
  4. Sonuç ve öneri.

Önemli: güç, tek bir t-testinin gücü değil ön-kayıttaki karar kuralının gücüdür. "s̄ >= 5" nokta-tahmin
şartı, gerçek etki tam 5 iken gücü tanım gereği ~%50'ye çiviler; bu gürültüden bağımsızdır ve seed
sayısıyla düzelmez. Betik bunu ayrıca gösteriyor.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.config import PRESETS  # noqa: E402
from src.flops import train_flops_per_token  # noqa: E402

LSTAR = 1.3592            # PREREGISTRATION.md'de dondurulmuş
TOK_PER_STEP = 32768
B_REF = 3000 * TOK_PER_STEP
T975 = {2: 4.302653, 3: 3.182446, 4: 2.776445, 9: 2.262157, 19: 2.093024}

# S0'daki dondurulmuş koşular. "mimari" alanı eşleştirmenin hangi çiftte mimari-ötesi olduğunu belirler.
CONDITIONS = {
    "auxfree":   ("experiments/v3-router-ablation/runs/auxfree-s*", "moe"),
    "aux0.1":    ("experiments/v3-router-ablation/runs/aux0.1-s*", "moe"),
    "aux0.01":   ("experiments/v2-grouped-dispatch/runs/moe-tok-s*", "moe"),
    "aux0.001":  ("experiments/v3-router-ablation/runs/aux0.001-s*", "moe"),
    "aux1.0":    ("experiments/v3-router-ablation/runs/aux1.0-s*", "moe"),
    "aux0":      ("experiments/v2-grouped-dispatch/runs/moe-noaux-tok-s*", "moe"),
    "dense":     ("experiments/v2-grouped-dispatch/runs/dense-tok-s*", "dense"),
}
FT = {"moe": train_flops_per_token(PRESETS["moe"][0]), "dense": train_flops_per_token(PRESETS["dense"][0])}

out: list[str] = []


def say(s: str = "") -> None:
    print(s)
    out.append(s)


def curve(run_dir: str) -> list[tuple[int, float]]:
    pts = []
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "val":
                pts.append((r["step"] * TOK_PER_STEP, r["val_loss"]))
    return sorted(pts)


def n_star(pts: list[tuple[int, float]], L: float = LSTAR) -> float | None:
    """L*'i ilk kez aşağı yönde kesen yerde, token cinsinden lineer interpolasyon (PREREGISTRATION O3)."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 >= L >= y1:
            return x0 + (y0 - L) / (y0 - y1) * (x1 - x0)
    return None


def pooled_sd(groups: list[list[float]]) -> tuple[float, int]:
    """Havuzlanmış içsel sd ve serbestlik derecesi (her grup kendi ortalamasına göre)."""
    ss = df = 0.0
    for g in groups:
        if len(g) < 2:
            continue
        m = st.mean(g)
        ss += sum((x - m) ** 2 for x in g)
        df += len(g) - 1
    return math.sqrt(ss / df), int(df)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/foton-v01-prereg/calibration.log")
    ap.add_argument("--sims", type=int, default=200_000)
    args = ap.parse_args()

    data = {}
    for name, (pat, arch) in CONDITIONS.items():
        rows = []
        for d in sorted(glob.glob(os.path.join(ROOT, pat))):
            seed = int(d.split("-s")[-1])
            pts = curve(d)
            ns = n_star(pts)
            rows.append({"seed": seed, "final_val": pts[-1][1], "n_star": ns,
                         "flops": None if ns is None else ns * FT[arch]})
        data[name] = {"arch": arch, "rows": sorted(rows, key=lambda r: r["seed"])}

    say("# FOTON v0.1 ön-kaydı — güç kalibrasyonu (yalnızca baseline gürültüsü)")
    say(f"L* = {LSTAR} (dondurulmuş) | referans bütçe = {B_REF:,} token | koşul {len(data)}, koşu "
        f"{sum(len(v['rows']) for v in data.values())}, seed/koşul 3")

    # ---------------- 1. Koşullar içi seed varyansı ----------------
    say("\n## 1. Koşullar içi (within-condition) seed varyansı")
    say(f"  {'koşul':10s} {'val loss ort':>12s} {'val sd':>9s} {'N* ort (token)':>16s} {'N* sd':>12s} {'N* sd %':>8s}")
    val_groups, rel_groups = [], []
    for name, v in data.items():
        vals = [r["final_val"] for r in v["rows"]]
        ns = [r["n_star"] for r in v["rows"]]
        val_groups.append(vals)
        if all(x is not None for x in ns):
            m, s = st.mean(ns), st.stdev(ns)
            rel_groups.append([100.0 * x / m for x in ns])
            say(f"  {name:10s} {st.mean(vals):12.4f} {st.stdev(vals):9.4f} {m:16,.0f} {s:12,.0f} {100*s/m:7.2f}%")
        else:
            say(f"  {name:10s} {st.mean(vals):12.4f} {st.stdev(vals):9.4f} {'L*e ulaşmıyor':>16s}")
    sd_val, df_val = pooled_sd(val_groups)
    sd_rel, df_rel = pooled_sd(rel_groups)
    say(f"\n  HAVUZLANMIŞ içsel sd: val loss {sd_val:.5f} nat (df {df_val}) | N* {sd_rel:.2f}% (df {df_rel})")

    # val loss sd'sini hesap cinsine çevir: eğri eğimi
    ref = data["auxfree"]["rows"]
    c0 = dict(curve(sorted(glob.glob(os.path.join(ROOT, CONDITIONS["auxfree"][0])))[0]))
    x0, x1 = 2250 * TOK_PER_STEP, 2500 * TOK_PER_STEP
    slope = (c0[x0] - c0[x1]) / (math.log(x1) - math.log(x0))     # nat / ln(token)
    say(f"  eğri eğimi L* civarında: d(val)/d(ln token) = -{slope:.4f} nat")
    say(f"  -> val loss sd'si {sd_val:.5f} nat, hesap cinsinden {100*sd_val/slope:.2f}% "
        f"(doğrudan ölçülen N* sd'si {sd_rel:.2f}% ile tutarlı olmalı)")
    say(f"  KULLANICI SORUSU: gözlenen tek-koşu gürültüsü ±%6 DEĞİL, ±{sd_rel:.2f}% (1 sd).")

    # ---------------- 2. Paired vs unpaired ----------------
    say("\n## 2. Eşleştirilmiş (paired) vs eşleştirilmemiş (unpaired) fark varyansı")
    say("  s = 100 x (1 - N*_A / N*_B).  paired: aynı seed eşleşir (n=3).  unpaired: 3x3=9 çapraz eşleşme.")
    say(f"  {'çift':22s} {'tip':>10s} {'s̄':>8s} {'sd paired':>10s} {'sd unpaired':>12s} {'kazanç':>8s}")
    names = list(data)
    paired_groups_same, paired_groups_cross = [], []
    unp_same, unp_cross = [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = data[a]["rows"], data[b]["rows"]
            if any(r["n_star"] is None for r in ra + rb):
                continue
            seeds = sorted({r["seed"] for r in ra} & {r["seed"] for r in rb})
            if len(seeds) < 3:
                continue
            na = {r["seed"]: r["n_star"] for r in ra}
            nb = {r["seed"]: r["n_star"] for r in rb}
            paired = [100.0 * (1 - na[s] / nb[s]) for s in seeds]
            unpaired = [100.0 * (1 - na[s1] / nb[s2]) for s1 in seeds for s2 in seeds]
            sp, su = st.stdev(paired), st.stdev(unpaired)
            cross = data[a]["arch"] != data[b]["arch"]
            (paired_groups_cross if cross else paired_groups_same).append(paired)
            (unp_cross if cross else unp_same).append(unpaired)
            say(f"  {a + ' / ' + b:22s} {'mimari-ötesi' if cross else 'aynı mimari':>10s} "
                f"{st.mean(paired):+8.2f} {sp:10.2f} {su:12.2f} {su / sp if sp > 0 else float('inf'):7.2f}x")

    def pool_pairs(groups):
        # her grup kendi ortalamasına göre; paired icin df=2, unpaired icin 9 gozlem ama bagimsiz degil:
        # unpaired'i yalnizca sd tahmini olarak kullaniyoruz (n-1 = 8 yerine muhafazakar 2 df saymiyoruz)
        return pooled_sd(groups)

    sd_p_same, df_p_same = pool_pairs(paired_groups_same)
    sd_p_cross, df_p_cross = pool_pairs(paired_groups_cross)
    sd_u_same, _ = pool_pairs(unp_same)
    sd_u_cross, _ = pool_pairs(unp_cross)
    say(f"\n  HAVUZLANMIŞ  aynı mimari : paired sd {sd_p_same:.2f} puan (df {df_p_same}) | "
        f"unpaired sd {sd_u_same:.2f} puan | kazanç {sd_u_same / sd_p_same:.2f}x")
    say(f"  HAVUZLANMIŞ  mimari-ötesi: paired sd {sd_p_cross:.2f} puan (df {df_p_cross}) | "
        f"unpaired sd {sd_u_cross:.2f} puan | kazanç {sd_u_cross / sd_p_cross:.2f}x")
    say("  'mimari-ötesi' = dense ile MoE çiftleri: FOTON-baseline eşleştirmesinin en yakın proxy'si,")
    say("  çünkü orada da weight init birebir eşleşemiyor, yalnızca veri/batch sırası ortak.")

    # ---------------- 2b. Eşleştirme niye kazandırmıyor: blok (seed) etkisi ----------------
    say("\n## 2b. Eşleştirme niye kazandırmıyor: seed'in PAYLAŞILAN etkisi var mı?")
    say("  Eşleştirilmiş tasarım yalnızca seed'in koşullar arasında TUTARLI bir etkisi varsa varyans düşürür")
    say("  (Var_paired = Var_A + Var_B - 2*Cov). Rastgele blok ayrışımı: her koşulun N*'ı kendi ortalamasına")
    say("  normalize edilir, sonra seed sütunlarının ortalaması alınır.")
    seeds = sorted({r["seed"] for v in data.values() for r in v["rows"]})
    usable = [n for n, v in data.items() if all(r["n_star"] is not None for r in v["rows"])]
    R = {}
    for name in usable:
        rows = {r["seed"]: r["n_star"] for r in data[name]["rows"]}
        m = st.mean(rows.values())
        R[name] = {s: 100.0 * rows[s] / m for s in seeds}
    say(f"\n  {'koşul':10s}" + "".join(f"{('s' + str(s)):>10s}" for s in seeds))
    for name in usable:
        say(f"  {name:10s}" + "".join(f"{R[name][s]:9.2f}%" for s in seeds))
    block = {s: st.mean(R[n][s] for n in usable) - 100.0 for s in seeds}
    say(f"  {'SEED ETKİSİ':10s}" + "".join(f"{block[s]:+9.2f}%" for s in seeds))
    resid = [R[n][s] - 100.0 - block[s] for n in usable for s in seeds]
    within = [R[n][s] - 100.0 for n in usable for s in seeds]
    sd_block = math.sqrt(sum(b * b for b in block.values()) / (len(seeds) - 1))
    sd_within = math.sqrt(sum(w * w for w in within) / (len(within) - len(usable)))
    sd_resid = math.sqrt(sum(r * r for r in resid) / (len(resid) - len(usable) - len(seeds) + 1))
    say(f"\n  seed etkisinin sd'si          : {sd_block:.3f}%  (df {len(seeds)-1})")
    say(f"  koşul içi toplam sd           : {sd_within:.3f}%  (df {len(within)-len(usable)})")
    say(f"  seed etkisi çıkarıldıktan sonra: {sd_resid:.3f}%  (df {len(resid)-len(usable)-len(seeds)+1})")
    share = (sd_block / sd_within) ** 2 if sd_within > 0 else 0.0
    null_share = (len(seeds) - 1) / (len(within) - len(usable))
    say(f"  -> seed'in açıkladığı varyans payı ~ %{100*share:.1f}")
    say(f"     ama SAF GÜRÜLTÜDE beklenen pay = df_seed/df_toplam = {len(seeds)-1}/{len(within)-len(usable)} "
        f"= %{100*null_share:.1f}: gözlenen pay bundan AYIRT EDİLEMİYOR.")
    rhos = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            xa = [R[a][s] for s in seeds]
            xb = [R[b][s] for s in seeds]
            ma, mb = st.mean(xa), st.mean(xb)
            num = sum((x - ma) * (y - mb) for x, y in zip(xa, xb))
            den = math.sqrt(sum((x - ma) ** 2 for x in xa) * sum((y - mb) ** 2 for y in xb))
            if den > 0:
                rhos.append(num / den)
    say(f"  koşul çiftleri arası seed korelasyonu: ortalama ρ = {st.mean(rhos):+.3f} "
        f"(n={len(rhos)} çift, medyan {st.median(rhos):+.3f})")
    say("  ρ ~ 0 => aynı veri sırasını paylaşmak 3000 adım sonunda ortak bir sapma bırakmıyor: eğitim")
    say("  kaotik biçimde ayrışıyor. Eşleştirme bu yüzden kazanç sağlamıyor (v2'de de gözlenmişti).")

    # ---------------- 3. Karar kuralının gücü ----------------
    say("\n## 3. Karar kuralının gücü (Monte Carlo, ön-kayıttaki BİLEŞİK kural)")
    say("  Kural: 'başarılı' <=> s̄ >= 5 VE %95 GA alt sınırı > 0 (GA örneklem sd'siyle, t(n-1)).")
    say("  'belirsiz' <=> s̄ > 2 ve (GA sıfırı kesiyor veya s̄ < 5).  Tripwire: örneklem sd > 5 puan.")
    rng = np.random.default_rng(0)

    def simulate(mu: float, sd: float, n: int, sims: int) -> dict:
        x = rng.normal(mu, sd, size=(sims, n))
        m = x.mean(axis=1)
        s = x.std(axis=1, ddof=1)
        hw = T975[n - 1] * s / math.sqrt(n)
        lo, hi = m - hw, m + hw
        trip = s > 5
        basarili = (~trip) & (m >= 5) & (lo > 0)
        guclu = basarili & (m >= 10)
        belirsiz = (~trip) & (m > 2) & ~basarili
        notr = (~trip) & (m >= -2) & (m <= 2)
        basarisiz = (~trip) & (m < -2)
        return {"başarılı+": basarili.mean(), "  (güçlü)": guclu.mean(), "belirsiz": belirsiz.mean(),
                "nötr": notr.mean(), "başarısız": basarisiz.mean(), "tripwire": trip.mean(),
                "GA alt>0": (lo > 0).mean(), "ort GA yarı-gen.": hw.mean()}

    SDS = [("gözlenen, mimari-ötesi", sd_p_cross), ("gözlenen, aynı mimari", sd_p_same),
           ("kötümser 3.0", 3.0), ("çok kötümser 5.0", 5.0), ("kullanıcının korktuğu 6.0", 6.0)]
    for label, sd in SDS:
        say(f"\n  --- sd = {sd:.2f} puan ({label}) ---")
        say(f"  {'gerçek etki':>12s} {'n':>3s} {'başarılı+':>10s} {'belirsiz':>9s} {'nötr':>7s} "
            f"{'başarısız':>10s} {'tripwire':>9s} {'GA yarı-gen.':>13s}")
        for mu in (0.0, 2.0, 5.0, 7.0, 10.0):
            for n in (5, 10):
                r = simulate(mu, sd, n, args.sims)
                say(f"  {mu:+11.1f}% {n:3d} {100*r['başarılı+']:9.1f}% {100*r['belirsiz']:8.1f}% "
                    f"{100*r['nötr']:6.1f}% {100*r['başarısız']:9.1f}% {100*r['tripwire']:8.1f}% "
                    f"{r['ort GA yarı-gen.']:12.2f}p")

    # ---------------- 4. Sonuç ----------------
    say("\n## 4. Sonuç")
    hw5 = T975[4] * sd_p_cross / math.sqrt(5)
    say(f"  (a) Gözlenen gürültü: tek koşu ±{sd_rel:.2f}% (1 sd) hesap cinsinden; eşleştirilmiş farkın sd'si")
    say(f"      mimari-ötesi çiftlerde {sd_p_cross:.2f} puan. Kullanıcının hipotezindeki ±%6 DEĞİL.")
    say(f"  (b) n=5'te %95 GA yarı-genişliği ~{hw5:.2f} puan, yani 'GA alt sınırı > 0' şartı s̄ > {hw5:.2f}")
    say(f"      puandan itibaren sağlanıyor. Gürültü, +%5 eşiği için BAĞLAYICI DEĞİL.")
    say("  (c) Bağlayıcı olan şey nokta-tahmin şartı: gerçek etki tam +%5 iken P(s̄ >= 5) ~ %50, bu")
    say("      gürültüden bağımsızdır ve seed sayısıyla DÜZELMEZ (yukarıdaki tabloda n=5 ile n=10")
    say("      karşılaştırılabilir). Eşiğin sınırındaki bir etkinin yarı yarıya 'belirsiz' çıkması")
    say("      eşiğin tanımının sonucudur, tasarım hatası değildir; ölçek yükseltme hakkı bunun içindir.")
    say("  (d) Bu yüzden ne eşik yükseltilmeli ne seed artırılmalı. Gözlenen gürültü eşiği yutmuyor;")
    say("      seed artırmak GA'yı daraltır ama zaten bağlayıcı olmayan şartı iyileştirir.")
    say("      Tripwire (örneklem sd > 5 puan) gözlenen gürültünün ~4-10 katı: yalnızca gerçekten")
    say("      anormal varyansta ateşlenir, ki o zaman 10 seed'e çıkmak doğru tepkidir.")

    path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
