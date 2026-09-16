"""100 adımlık loop / grouped doğrulama koşularının loss eğrilerini karşılaştırır.

    .venv\\Scripts\\python scripts/compare_curves.py --dir checkpoints/v2-verify-sched \\
        --out experiments/v2-grouped-dispatch/verify_100step.json

Koşular (her preset ve seed için): <preset>-{loop,grouped,loopb,loopc}-s<seed>. loopb ve loopc, loop'un aynı seed'le iki
tekrarı: aynı başlatma, aynı veri sırası, aynı kod. Hepsi GERÇEK schedule'ın (3000 adım, warmup 200) ilk 100 adımı
(--stop_step 100). Ayrıca v1tag-<preset>-s1337: v1 tag'inin kendi kodu (aynı 100 adım).

ÖNCEDEN KAYDEDİLEN PROTOKOL (koşulardan önce yazıldı; ilk denemede warmup 20 adıma sıkıştırılmıştı ve eğitim 11-13.
adımda loss sıçramalarıyla kaotikleşiyordu, o yüzden adım adım KS testleri anlamsızdı ve bırakıldı):
  1. Ayrışma öncesi pencere. Seed başına T_c = |loss(loopb) - loss(loop)| farkının ilk kez 1e-3'ü aştığı adımdan
     önceki son adım (hiç aşmazsa 100). 1..T_c adımlarında S_x = sum |loss(x) - loss(loop)|. Oran R_g = S_grouped / S_loopb;
     sıfır dağılımı (aynı kodun iki tekrarı) R_c = S_loopc / S_loopb. grouped doğruysa R_g değerleri R_c değerleriyle aynı
     büyüklükte olmalı. Ayrıca loss'un loop'la bit bit aynı kaldığı ilk adım sayısı raporlanır.
  2. Uç noktalar: val loss @100 ve adım 81-100 ortalama train loss'u; seed başına eşleştirilmiş fark (grouped - loop),
     aynı şekilde (loopb - loop) ve (loopc - loop) sıfır dağılımı olarak. Ortalama, örneklem sd, %95 GA t(n-1).
  3. Eşdeğerlik marjı delta = 0.02 nat (v1'deki en küçük model farkı noaux - moe = 0.020): grouped - loop farkının
     %95 GA'sı [-delta, delta] içindeyse "eşdeğer". Aynı kodun tekrarı (loopb - loop) bile bu marja sığmıyorsa test
     bu çözünürlüğü sağlayamıyor demektir: sonuç "yetersiz güç" diye raporlanır, eşdeğerlik iddia edilmez.
  4. LR dizileri tüm koşularda birebir aynı olmalı; MoE yük dengesi (yük CV, ölü expert) adım 100'de eşleştirilmiş.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

T_CRIT_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
DELTA = 0.02
SPLIT_TOL = 1e-3


def load(run_dir: str) -> dict:
    rec = {"train": {}, "val": {}}
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] in rec:
                rec[r["type"]][r["step"]] = r
    return rec


def ci(xs: list[float]) -> dict:
    n = len(xs)
    m = st.mean(xs)
    if n < 2:
        return {"mean": m, "n": n}
    sd = st.stdev(xs)
    half = T_CRIT_975[n - 1] * sd / math.sqrt(n)
    return {"mean": m, "sd": sd, "ci95": [m - half, m + half], "n": n, "per_seed": xs}


def fmt(c: dict) -> str:
    if "ci95" not in c:
        return f"{c['mean']:+.4f}"
    return f"{c['mean']:+.4f} (sd {c['sd']:.4f}, %95 GA [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}])"


def within(c: dict, delta: float) -> bool:
    return "ci95" in c and -delta <= c["ci95"][0] and c["ci95"][1] <= delta


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="checkpoints/v2-verify-sched")
    p.add_argument("--out", default="experiments/v2-grouped-dispatch/verify_100step.json")
    p.add_argument("--seeds", default="1337,1338,1339,1340,1341,1342")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    report = {"protocol": {"delta": DELTA, "split_tol": SPLIT_TOL, "seeds": seeds}}

    for preset in ("moe", "moe-noaux"):
        impls = ("loop", "grouped", "loopb", "loopc")
        runs = {(i, s): load(os.path.join(args.dir, f"{preset}-{i}-s{s}")) for i in impls for s in seeds}
        steps = sorted(runs[("loop", seeds[0])]["train"])
        last = steps[-1]
        loss = {k: [r["train"][t]["loss"] for t in steps] for k, r in runs.items()}
        lr_ref = [runs[("loop", seeds[0])]["train"][t]["lr"] for t in steps]
        lr_same = all([r["train"][t]["lr"] for t in steps] == lr_ref for r in runs.values())
        v1_dir = os.path.join(args.dir, f"v1tag-{preset}-s{seeds[0]}")
        v1 = load(v1_dir) if os.path.exists(v1_dir) else None

        window, first_diff = {}, {}
        for s in seeds:
            L = loss[("loop", s)]
            split = next((i for i, (b, l) in enumerate(zip(loss[("loopb", s)], L)) if abs(b - l) > SPLIT_TOL), len(L))
            sums = {i: sum(abs(a - l) for a, l in zip(loss[(i, s)][:split], L[:split])) for i in ("grouped", "loopb", "loopc")}
            window[s] = {"T_c": steps[split - 1] if split > 0 else 0, "S": sums,
                         "R_grouped": sums["grouped"] / sums["loopb"] if sums["loopb"] > 0 else None,
                         "R_null_loopc": sums["loopc"] / sums["loopb"] if sums["loopb"] > 0 else None}
            first_diff[s] = {i: next((steps[j] for j, (a, l) in enumerate(zip(loss[(i, s)], L)) if a != l), None)
                             for i in ("grouped", "loopb", "loopc")}

        tail = [j for j, t in enumerate(steps) if t > last - 20]

        def tail_mean(k):
            return st.mean(loss[k][j] for j in tail)

        end = {}
        for name, key in (("val_at_end", None), ("tail_mean_train_loss", None)):
            for i in ("grouped", "loopb", "loopc"):
                if name == "val_at_end":
                    d = [runs[(i, s)]["val"][last]["val_loss"] - runs[("loop", s)]["val"][last]["val_loss"] for s in seeds]
                else:
                    d = [tail_mean((i, s)) - tail_mean(("loop", s)) for s in seeds]
                end[f"{name}:{i}_minus_loop"] = ci(d)
        balance = {}
        if "load_cv" in runs[("loop", seeds[0])]["train"][last]:
            for m in ("load_cv", "dead_experts"):
                balance[m] = {i: ci([runs[(i, s)]["train"][last][m] - runs[("loop", s)]["train"][last][m] for s in seeds])
                              for i in ("grouped", "loopb")}
        val_seed_sd = st.stdev(runs[("loop", s)]["val"][last]["val_loss"] for s in seeds)

        g_val, b_val = end["val_at_end:grouped_minus_loop"], end["val_at_end:loopb_minus_loop"]
        if within(g_val, DELTA) and within(b_val, DELTA):
            verdict = "eşdeğer (marj içinde)"
        elif not within(b_val, DELTA):
            verdict = "yetersiz güç: aynı kodun tekrarı bile marja sığmıyor"
        else:
            verdict = "eşdeğerlik gösterilemedi"
        r = {"steps": len(steps), "lr_sequences_identical": lr_same, "pre_split_window": window,
             "first_step_loss_differs_from_loop": first_diff, "endpoints": end, "moe_balance_at_end": balance,
             "val_seed_sd_loop": val_seed_sd, "verdict_val_at_end": verdict}
        if v1:
            v1_loss = [v1["train"][t]["loss"] for t in steps]
            L = loss[("loop", seeds[0])]
            r["v1tag_vs_new_loop_seed0"] = {
                "first_step_differs": next((steps[j] for j, (a, l) in enumerate(zip(v1_loss, L)) if a != l), None),
                "val_at_end_diff": v1["val"][last]["val_loss"] - runs[("loop", seeds[0])]["val"][last]["val_loss"],
                "lr_identical": [v1["train"][t]["lr"] for t in steps] == lr_ref,
            }
        report[preset] = r

        print(f"\n===== {preset}: {len(steps)} adım (gerçek schedule'ın başı) x {len(seeds)} seed =====")
        print(f"LR dizileri tüm koşularda birebir aynı: {lr_same}")
        print("1) ayrışma öncesi pencere (seed: T_c | S_grouped, S_loopb, S_loopc | R_grouped vs R_sıfır | loss'un loop'tan "
              "ilk ayrıldığı adım g/b/c):")
        for s in seeds:
            w, fd = window[s], first_diff[s]
            rg = "—" if w["R_grouped"] is None else f"{w['R_grouped']:.2f}"
            rc = "—" if w["R_null_loopc"] is None else f"{w['R_null_loopc']:.2f}"
            print(f"   {s}: T_c {w['T_c']:3d} | {w['S']['grouped']:.2e}, {w['S']['loopb']:.2e}, {w['S']['loopc']:.2e} | "
                  f"R {rg} vs {rc} | {fd['grouped']}/{fd['loopb']}/{fd['loopc']}")
        print("2) uç noktalar, seed başına eşleştirilmiş fark:")
        for k, c in end.items():
            print(f"   {k:42s} {fmt(c)}")
        print(f"   (loop'un seed'ler arası val sd'si: {val_seed_sd:.4f})")
        for m, d in balance.items():
            print(f"   adım {last} {m}: grouped-loop {fmt(d['grouped'])} | loopb-loop {fmt(d['loopb'])}")
        print(f"3) karar (val @{last}, marj ±{DELTA}): {verdict}")
        if v1:
            q = r["v1tag_vs_new_loop_seed0"]
            print(f"4) v1 tag kodu vs yeni loop (seed {seeds[0]}): LR aynı {q['lr_identical']}, loss ilk ayrıldığı adım "
                  f"{q['first_step_differs']}, val @{last} farkı {q['val_at_end_diff']:+.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
