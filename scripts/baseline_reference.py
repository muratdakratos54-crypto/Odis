"""Baseline referans sayıları: `N*`, `F_base,i`, ve hedefe ulaşma süresi — TAM ölçümle.

    .venv\\Scripts\\python scripts/baseline_reference.py

Denetimin bulduğu üç hatayı düzeltir:

1. **Zamana oranlama yerine TAM zaman.** Önceki hesap `süre = toplam_süre × N*/toplam_token`
   yapıyordu, yani koşu içinde tok/s'in SABİT olduğunu varsayıyordu. Oysa aynı kayıt tok/s'in
   koşu içinde 64.7k → 50.5k salındığını gösteriyor; varsayım tam olarak ölçmek istediğimiz
   gürültüyü siliyor. `metrics.jsonl` her 25 adımda kümülatif `train_time` taşıdığı için hedefe
   ulaşma süresi doğrudan interpolasyonla çıkarılabilir. Etkisi ölçüldü: seed başına %-4.4 ile
   %+7.3 arası hata, ve p değeri 2 kat küçük çıkıyordu.

2. **tok/s cezası ile süre artışı karıştırılmıştı** (taban hatası). M aynı bütçeyi D'den
   %51 daha UZUN sürede işliyor; bu %51'lik bir tok/s düşüşü DEĞİLDİR, %33.8'lik bir düşüştür
   (1 − 1/1.51). Her ikisi de ayrı ayrı raporlanıyor, adları karıştırılmadan.

3. **Seed başına `N*` ve `F_base,i` hiçbir yerde kayıtlı değildi.** Ön-kaydın birincil istatistiği
   `s_i = 100 × (1 − F_foton,i / F_base,i)` seed başına paydaya dayanıyor; tek bir skaler
   pinlemek onu yeniden kurulamaz yapıyordu.

Ayrıca `N*` interpolasyonu artık açık bir kural: **token ekseninde doğrusal**, ve EK1-B14 gereği
kesişimden SONRAKİ eval noktasının da hedefin altında olması denetleniyor.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from dataclasses import replace  # noqa: E402

from src.config import PRESETS  # noqa: E402
from src.flops import train_flops_per_token  # noqa: E402
from src.stats import n_star, paired  # noqa: E402

STEP, STEPS = 32768, 4375
SEEDS = [11, 22, 33, 44, 55]
ARMS = {"m": "foton-m", "d": "foton-d"}


def load(arm: str, seed: int) -> dict:
    p = f"checkpoints/foton-v02/baseline-{arm}-s{seed}/metrics.jsonl"
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    val = sorted((x["step"] * STEP, x["val_loss"]) for x in rows if x["type"] == "val")
    # kümülatif eğitim süresi: (token, saniye) — train satırları her log aralığında
    clock = sorted((x["step"] * STEP, x["train_time"]) for x in rows
                   if x["type"] == "train" and "train_time" in x)
    end = [x for x in rows if x["type"] == "end"][0]
    clock.append((STEPS * STEP, end["train_time"]))
    tr = [x for x in rows if x["type"] == "train"]
    return {"val": val, "clock": clock, "full_sec": end["train_time"],
            "wall_sec": end.get("time", end["train_time"]),
            # peak VRAM `train` satırlarında; koşunun gördüğü EN YÜKSEK değeri al
            "peak_gb": max(x["peak_mem_gb"] for x in tr if "peak_mem_gb" in x),
            "rates": [x["tok_per_sec"] for x in tr][2:]}


def time_at(clock: list[tuple[int, float]], tokens: float) -> float:
    """Verilen token sayısına ulaşıldığı AN — kümülatif saatte doğrusal interpolasyon."""
    for (a, ta), (b, tb) in zip(clock, clock[1:]):
        if a <= tokens <= b:
            return ta + (tokens - a) / (b - a) * (tb - ta)
    raise ValueError(f"{tokens} kümülatif saatin dışında")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    LSTAR = json.load(open("experiments/foton-v02-prereg/lstar.json", encoding="utf-8"))["lstar"]
    ft = {a: train_flops_per_token(replace(PRESETS[p][0], vocab_size=32768))
          for a, p in ARMS.items()}
    data = {a: {s: load(a, s) for s in SEEDS} for a in ARMS}

    print(f"L* = {LSTAR} | FLOP/token M {ft['m']:,} D {ft['d']:,} "
          f"(fark %{100*(ft['m']/ft['d']-1):.4f})")

    # ---- 1) L* hedefinde seed basina N* ve F_base,i ----
    print("\n## L* hedefi: seed başına N* ve F_base,i  (ön-kaydın paydası)")
    rec: dict = {"l_star": LSTAR, "flops_per_token": ft, "seeds": SEEDS, "per_arm": {}}
    for a in ARMS:
        rows = []
        for s in SEEDS:
            n = n_star(data[a][s]["val"], LSTAR)
            rows.append({"seed": s, "n_star": n,
                         "f_base": None if n is None else n * ft[a],
                         "final_val": data[a][s]["val"][-1][1],
                         "peak_mem_gb": data[a][s]["peak_gb"]})
        rec["per_arm"][a] = rows
        if all(r["n_star"] for r in rows):
            ns = [r["n_star"] for r in rows]
            fs = [r["f_base"] for r in rows]
            print(f"  {a}: N* = " + ", ".join(f"{x/1e6:.3f}M" for x in ns))
            print(f"     ortalama {st.mean(ns)/1e6:.3f}M (bütçenin %{100*st.mean(ns)/(STEPS*STEP):.3f}'i)"
                  f" | sd %{100*st.stdev(ns)/st.mean(ns):.2f}")
            print(f"     F_base,i ortalaması {st.mean(fs):.6e}  "
                  f"| 0.8×bütçe×FLOP/token = {0.8*STEPS*STEP*ft[a]:.6e} "
                  f"(fark %{100*(st.mean(fs)/(0.8*STEPS*STEP*ft[a])-1):+.3f})")
        else:
            fin = [r["final_val"] for r in rows]
            print(f"  {a}: L*'e ULAŞMIYOR — son val {st.mean(fin):.6f} "
                  f"(aralık {min(fin):.4f}–{max(fin):.4f}), eksik {st.mean(fin)-LSTAR:+.6f} nat")

    # ---- 2) yardimci hedef: TAM zaman vs oranlanmis zaman ----
    LAUX = round(st.mean([dict(data["d"][s]["val"])[3500 * STEP] for s in SEEDS]), 4)
    print(f"\n## Yardımcı hedef L_aux = {LAUX} (dense'in %80 noktası) — TAM zaman ölçümü")
    aux: dict = {"l_aux": LAUX, "per_arm": {}}
    for a in ARMS:
        rows = []
        for s in SEEDS:
            d = data[a][s]
            n = n_star(d["val"], LAUX)
            exact = time_at(d["clock"], n)
            prorated = d["full_sec"] * n / (STEPS * STEP)
            rows.append({"seed": s, "n_star": n, "sec_exact": exact,
                         "sec_prorated": prorated, "full_sec": d["full_sec"],
                         "wall_sec": d["wall_sec"]})
        aux["per_arm"][a] = rows
        print(f"  {a}:")
        for r in rows:
            print(f"    s{r['seed']:<3d} N* {r['n_star']/1e6:7.3f}M | tam {r['sec_exact']:8.2f}s "
                  f"| oranlanmış {r['sec_prorated']:8.2f}s | hata %{100*(r['sec_prorated']/r['sec_exact']-1):+6.2f}")

    tm_e = [r["sec_exact"] for r in aux["per_arm"]["m"]]
    td_e = [r["sec_exact"] for r in aux["per_arm"]["d"]]
    tm_p = [r["sec_prorated"] for r in aux["per_arm"]["m"]]
    td_p = [r["sec_prorated"] for r in aux["per_arm"]["d"]]
    pe, pp = paired(tm_e, td_e, SEEDS, SEEDS), paired(tm_p, td_p, SEEDS, SEEDS)
    print(f"\n  {'yöntem':12s} {'fark':>12s} {'%':>8s} {'GA95 (dk)':>22s} {'p':>9s}")
    for lab, pr, base in (("TAM", pe, td_e), ("oranlanmış", pp, td_p)):
        print(f"  {lab:12s} {pr['mean']/60:+9.3f} dk {100*pr['mean']/st.mean(base):+7.1f}% "
              f"[{pr['ci95'][0]/60:+7.3f}, {pr['ci95'][1]/60:+7.3f}] {pr['p']:9.4f}")
    aux["paired_exact"] = {"mean_sec": pe["mean"], "pct": 100*pe["mean"]/st.mean(td_e),
                           "ci95_sec": pe["ci95"], "p": pe["p"], "t": pe["t"]}
    aux["paired_prorated"] = {"mean_sec": pp["mean"], "pct": 100*pp["mean"]/st.mean(td_p),
                              "ci95_sec": pp["ci95"], "p": pp["p"]}

    # ---- 3) tok/s cezasi: DOGRU taban ----
    print("\n## Gerçekleşen verim — iki ayrı sayı, adları karıştırılmadan")
    tok = STEPS * STEP
    thr = {a: tok / st.mean([data[a][s]["full_sec"] for s in SEEDS]) for a in ARMS}
    t_inc = st.mean([data["m"][s]["full_sec"] for s in SEEDS]) / \
        st.mean([data["d"][s]["full_sec"] for s in SEEDS]) - 1
    print(f"  M {thr['m']:,.0f} tok/s | D {thr['d']:,.0f} tok/s")
    print(f"  tok/s DÜŞÜŞÜ        %{100*(1-thr['m']/thr['d']):.2f}   (1 − M/D)")
    print(f"  süre ARTIŞI         %{100*t_inc:.2f}   (M/D − 1)   <- bunlar AYNI sayı değil")
    # tutarlilik: t_M/t_D = (N*_M/N*_D) x (tput_D/tput_M)
    ratio = (st.mean([r["n_star"] for r in aux["per_arm"]["m"]]) /
             st.mean([r["n_star"] for r in aux["per_arm"]["d"]])) * (thr["d"] / thr["m"])
    print(f"  tutarlılık: (N*_M/N*_D)×(tput_D/tput_M) = {ratio:.5f} -> süre farkı %{100*(ratio-1):+.2f}"
          f"  (ölçülen tam: %{100*pe['mean']/st.mean(td_e):+.2f})")
    aux["throughput"] = {"m_tok_s": thr["m"], "d_tok_s": thr["d"],
                         "tok_s_reduction_pct": 100*(1-thr["m"]/thr["d"]),
                         "time_increase_pct": 100*t_inc}

    # ---- 4) eval/ckpt yuku ve VRAM ----
    print("\n## Uçtan uca duvar saati ve VRAM (koşuların kendi kaydından)")
    for a in ARMS:
        tr = st.mean([data[a][s]["full_sec"] for s in SEEDS]) / 60
        wl = st.mean([data[a][s]["wall_sec"] for s in SEEDS]) / 60
        pk = [data[a][s]["peak_gb"] for s in SEEDS]
        print(f"  {a}: eğitim {tr:6.2f} dk | uçtan uca {wl:6.2f} dk | eval+ckpt yükü {wl-tr:5.2f} dk"
              f" | peak VRAM {st.mean(pk):.3f} GB (aralık {min(pk):.3f}–{max(pk):.3f})")
        aux.setdefault("overhead", {})[a] = {"train_min": tr, "wall_min": wl,
                                             "eval_ckpt_min": wl - tr, "peak_gb_mean": st.mean(pk)}

    # ---- 5) kampanya GPU saati tahmini ----
    print("\n## Kampanya GPU saati (ölçülen uçtan uca sürelerden)")
    w = {a: st.mean([data[a][s]["wall_sec"] for s in SEEDS]) / 3600 for a in ARMS}
    per_block = w["d"] + w["m"] + 2 * w["m"]          # H ve X ~ M mertebesinde varsayımı
    print(f"  D {w['d']:.3f} sa | M {w['m']:.3f} sa | H,X için M varsayıldı (ölçülmedi)")
    print(f"  S0 karar: 5 blok × (D+M+H+X) ≈ {5*per_block:.1f} GPU saati")
    print(f"  V7 ablasyon: 9 koşu × %50 bütçe ≈ {9*0.5*w['m']:.1f} sa")
    print(f"  tarama (3 seed × 4 kol × %50) ≈ {3*0.5*per_block:.1f} sa")
    print(f"  S1 yükseltme hakkı (640×10, kullanılırsa) ≈ {5*per_block*1.6:.1f} sa (1.6× ölçek varsayımı)")
    aux["gpu_hours"] = {"s0_decision": 5*per_block, "v7_ablation": 9*0.5*w["m"],
                        "screening": 3*0.5*per_block, "s1_escalation": 5*per_block*1.6,
                        "note": "H ve X'in süresi ÖLÇÜLMEDİ; M mertebesinde varsayıldı"}

    out = {"reference": rec, "auxiliary": aux}
    p = "experiments/foton-v02-prereg/baseline_reference.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
