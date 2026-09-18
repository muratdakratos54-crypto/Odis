"""FOTON v0.2 ön-kaydının güç kalibrasyonu — yalnızca baseline gürültüsünden, FOTON verisi yok.

    .venv\\Scripts\\python scripts/calibrate_v02.py --out experiments/foton-v02-prereg/calibration.log

v01'in kalibrasyon yöntemini bu korpusa uygular (yöntem devralındı, sayılar yeniden türetildi).
Dört soru:
  1. Aynı konfigürasyonun seed'ler arası gürültüsü ne (val loss ve hesap cinsinden)?
  2. Eşleştirilmiş `s_i`'nin sd'si ne (D ↔ M, gerçek iki mimari arası)?
  3. Bu gürültüde n=5 ile +%5'lik bir etkiyi ön-kayıttaki BİLEŞİK kuralla saptayabilir miyiz?
  4. Koşu-oturumu varyansı (tok/s) ne — duvar saati ölçütü ölçülebilir mi?

`s_i` FLOP cinsinden olduğu için (N* x FLOP/token) 2. ve 3. soru, koşuların kol bloğu hâlinde
koşmuş olmasından ETKİLENMEZ. 4. soru etkilenir ve zaten onu ölçmek için var.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from dataclasses import replace  # noqa: E402

from src.config import PRESETS  # noqa: E402
from src.flops import train_flops_per_token  # noqa: E402
from src.stats import SUCCESS, decide, n_star, paired, savings, t_crit  # noqa: E402

STEP, STEPS, P80 = 32768, 4375, 3500
RUNS, PREFIX = "checkpoints/foton-v02r", "r-"
SEEDS = [11, 22, 33, 44, 55]
ARMS = {"m": "foton-m", "d": "foton-d"}
out: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    out.append(s)


def load(arm: str, seed: int) -> dict:
    p = f"{RUNS}/{PREFIX}{arm}-s{seed}/metrics.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    val = sorted((x["step"] * STEP, x["val_loss"]) for x in rows if x["type"] == "val")
    tr = [x for x in rows if x["type"] == "train"][2:]
    end = [x for x in rows if x["type"] == "end"][0]
    return {"val": val, "rates": [x["tok_per_sec"] for x in tr], "time": end["train_time"],
            "at80": dict((s * STEP, v) for s, v in [(x["step"], x["val_loss"]) for x in rows
                                                    if x["type"] == "val"])[P80 * STEP]}




def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/foton-v02-prereg/calibration.log")
    ap.add_argument("--sims", type=int, default=200_000)
    args = ap.parse_args()

    LSTAR = json.load(open("experiments/foton-v02-prereg/lstar.json", encoding="utf-8"))["lstar"]
    ft = {a: train_flops_per_token(replace(PRESETS[p][0], vocab_size=32768)) for a, p in ARMS.items()}
    data = {a: {s: load(a, s) for s in SEEDS} for a in ARMS}

    say("# FOTON v0.2 guc kalibrasyonu (yalnizca baseline gurultusu, FOTON verisi YOK)")
    say(f"L* = {LSTAR} | butce {STEPS*STEP:,} token | %80 = adim {P80} (tam eval noktasi)")
    say(f"FLOP/token: M {ft['m']:,} | D {ft['d']:,} (fark %{100*(ft['m']/ft['d']-1):.2f})")

    # --- 1. kosullar ici gurultu ---
    say("\n## 1. Ayni konfigurasyonun seed'ler arasi gurultusu")
    say(f"  {'kol':4s} {'val@3500 ort':>13s} {'val sd':>9s} {'N* ort':>13s} {'N* sd %':>9s} {'butce %':>8s}")
    ns = {}
    for a in ARMS:
        v80 = [data[a][s]["at80"] for s in SEEDS]
        nn = [n_star(data[a][s]["val"], LSTAR) for s in SEEDS]
        if any(x is None for x in nn):
            say(f"  {a:4s} {st.mean(v80):13.6f} {st.stdev(v80):9.6f}   L*'e ULASMIYOR "
                f"(son val {data[a][SEEDS[0]]['val'][-1][1]:.4f})")
            ns[a] = None
            continue
        ns[a] = nn
        say(f"  {a:4s} {st.mean(v80):13.6f} {st.stdev(v80):9.6f} {st.mean(nn):13,.0f} "
            f"{100*st.stdev(nn)/st.mean(nn):8.2f}% {100*st.mean(nn)/(STEPS*STEP):7.1f}%")
    # egri egimi
    v = data["m"][SEEDS[0]]["val"]
    d = dict(v)
    x0, x1 = 3250 * STEP, 3750 * STEP
    slope = (d[x0] - d[x1]) / (math.log(x1) - math.log(x0))
    say(f"  L* civarinda egim d(val)/d(ln token) = -{slope:.4f} nat "
        f"-> %5 tasarruf ~ {slope*math.log(1/0.95):.5f} nat")

    # --- 2. eslestirilmis s_i ---
    say("\n## 2. Eslestirilmis s_i: D'nin M'e gore tasarrufu (gercek iki mimari)")
    if ns["d"] is None:
        say("  D kolu L*'e ULASMIYOR: on-kaydin sansur kurali (v01 O3/EK1-B8) geregi N*_d sansurlu.")
        dfin = {s: data["d"][s]["val"][-1][1] for s in SEEDS}     # seed KİMLİĞİYLE anahtarlı
        mfin = {s: data["m"][s]["val"][-1][1] for s in SEEDS}
        say(f"  D son val ort {st.mean(dfin.values()):.4f} | L* {LSTAR} | eksik {st.mean(dfin.values())-LSTAR:+.4f} nat")
        prq = paired(mfin, dfin)
        say(f"  esit tokende M-D farki {prq['mean']:+.4f} nat GA [{prq['ci95'][0]:+.4f}, "
            f"{prq['ci95'][1]:+.4f}] p={prq.get('p', float('nan')):.5f}  (TinyStories'te -0.0344 idi)")
        # Eslestirilmis sd'yi olcmek icin IKI KOLUN DA ulastigi yardimci hedef: dense'in kendi %80'i.
        # Bu bir karar hedefi DEGIL, yalnizca gurultu kalibrasyonu icin olcum araci.
        LAUX = round(st.mean([data["d"][s]["at80"] for s in SEEDS]), 4)
        say("")
        say(f"  Kalibrasyon icin yardimci hedef L_aux = {LAUX} (dense'in %80 noktasi; iki kol da ulasiyor)")
        NN = {a: {s: n_star(data[a][s]["val"], LAUX) for s in SEEDS} for a in ARMS}
        for a in ARMS:
            nn = list(NN[a].values())
            say(f"    {a}: N* ort {st.mean(nn)/1e6:7.2f}M = butcenin %{100*st.mean(nn)/(STEPS*STEP):5.1f}'i "
                f"| sd %{100*st.stdev(nn)/st.mean(nn):.2f}")
        # s_i `src/stats.savings()` ile kurulur: seed KİMLİĞİNE göre, oran olarak. İlk hâl listeyi
        # KONUMA göre eşliyordu — `paired()`'ın regresyona bağladığı hatanın aynısı, birincil
        # istatistiğin tam üstünde (ilk denetimin bulgusu). Artık veri seed kimliğiyle anahtarlı
        # taşınıyor: `src.stats` liste kabul ETMİYOR, konuma göre eşleme ifade edilemiyor.
        obs = savings({s: NN["m"][s] * ft["m"] for s in SEEDS},
                      {s: NN["d"][s] * ft["d"] for s in SEEDS})["obs"]
        s_i = [obs[s].s for s in SEEDS]
        p2 = paired({s: o.s for s, o in obs.items()}, dict.fromkeys(SEEDS, 0.0))
        sd_paired = p2["sd"]
        say(f"    s_i (M'in D'ye gore tasarrufu): {[round(x,2) for x in s_i]}")
        dres = decide(obs)      # w̄ bu kalibrasyonda verilmiyor; `decide()` bunu not olarak yazar
        say(f"    ort {p2['mean']:+.2f} puan | SD {sd_paired:.2f} puan | GA [{p2['ci95'][0]:+.2f}, "
            f"{p2['ci95'][1]:+.2f}] | p={p2.get('p', float('nan')):.5f} | karar: {dres['row']!r}")
        for note in dres["notes"]:          # notları DÜŞÜRME: "None sessiz değildir" garantisi
            say(f"      not: {note}")       # ancak not bir artifact'a ulaşırsa anlamlıdır
        say(f"    => OLCULMUS eslestirilmis sd(s_i) = {sd_paired:.2f} puan (gercek iki mimari arasi)")
    else:
        obs = savings({s: ns["d"][i] * ft["d"] for i, s in enumerate(SEEDS)},
                      {s: ns["m"][i] * ft["m"] for i, s in enumerate(SEEDS)})["obs"]
        s_i = [obs[s].s for s in SEEDS]
        say(f"  s_i (seed basina, puan): {[round(x,2) for x in s_i]}")
        pr = paired({s: o.s for s, o in obs.items()}, dict.fromkeys(SEEDS, 0.0))
        sd_paired = pr["sd"]
        say(f"  ort {pr['mean']:+.2f} puan | sd {sd_paired:.2f} puan | %95 GA "
            f"[{pr['ci95'][0]:+.2f}, {pr['ci95'][1]:+.2f}] | p={pr.get('p', float('nan')):.4f}")
        say(f"  -> dense, MoE'ye gore ayni hedefe {abs(pr['mean']):.1f} puan {'DAHA FAZLA' if pr['mean']<0 else 'daha az'} FLOP harciyor")
        say(f"  karar kurali bu s_i'ye uygulanirsa: {decide(obs)['row']!r}")

    # --- 3. guc ---
    say("\n## 3. Karar kuralinin gucu (Monte Carlo, on-kayittaki BILESIK kural)")
    rng = np.random.default_rng(0)

    def power(mu, sd, n, sims):
        """Ön-kayıttaki bileşik kuralın vektörize hâli — `src/stats.decide()` ile AYNI mantık.

        Tripwire artık KESİNLİK ölçütü (`src/stats.indeterminate`), mutlak `sd > 5` DEĞİL. Eski
        mutlak eşik n ile ölçeklenmediği için tek çaresi (10 seed) kendisini daha olası yapıyordu;
        bu betiğin ilk hâli o eski kuralı hesaplıyordu ve ön-kayıtla tutarsızdı.
        `scripts/rule_reachability.py --crosscheck` bu vektörize hâli `decide()` yoluna karşı
        doğruluyor (fark 0.00 puan).
        """
        x = rng.normal(mu, sd, size=(sims, n))
        m = x.mean(axis=1); s = x.std(axis=1, ddof=1)
        hw = t_crit(n - 1) * s / math.sqrt(n)
        lo, hi = m - hw, m + hw
        determinate = ((lo > 0) & (m >= SUCCESS)) | (hi < SUCCESS)
        trip = ~determinate
        ok = determinate & (m >= SUCCESS) & (lo > 0)
        und = determinate & (m > 2.0) & ~ok
        return ok.mean(), und.mean(), trip.mean(), hw.mean()

    cands = []
    if sd_paired:
        cands.append((f"olculen (D~M) {sd_paired:.2f}", sd_paired))
    cands += [("kotumser 3.0", 3.0), ("cok kotumser 5.0", 5.0)]
    for label, sd in cands:
        say(f"\n  --- sd = {sd:.2f} puan ({label}) ---")
        say(f"  {'gercek etki':>12s} {'n':>3s} {'basarili+':>10s} {'belirsiz':>9s} {'tripwire':>9s} {'GA yari':>9s}")
        for mu in (0.0, 2.0, 5.0, 7.0, 10.0):
            for n in (5, 10):
                ok, und, trip, hw = power(mu, sd, n, args.sims)
                say(f"  {mu:+11.1f}% {n:3d} {100*ok:9.1f}% {100*und:8.1f}% {100*trip:8.1f}% {hw:8.2f}p")

    # --- 4. kosu-oturumu varyansi ---
    say("\n## 4. Kosu-oturumu varyansi (tok/s) — duvar saati olcutu olculebilir mi?")
    say(f"  {'kol':4s} {'medyanlar (bin tok/s)':>34s} {'CV %':>7s} {'yavas/hizli':>12s}")
    for a in ARMS:
        meds = [st.median(data[a][s]["rates"]) for s in SEEDS]
        cv = 100 * st.stdev(meds) / st.mean(meds)
        say(f"  {a:4s} {', '.join(f'{m/1000:.1f}' for m in meds):>34s} {cv:6.2f}% "
            f"{max(meds)/min(meds):11.3f}x")
        hw = t_crit(4) * cv / math.sqrt(5)
        say(f"       -> bu CV ile n=5'te w̄'nin GA yari-genisligi ~{hw:.1f} puan")
    say("  KOL BLOGU halinde kosuldugu icin bu varyans kollara ESIT dagilmadi; V11a bu yuzden")
    say("  kampanyayi SEED BLOGU halinde kosturmayi zorunlu kiliyor.")

    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
