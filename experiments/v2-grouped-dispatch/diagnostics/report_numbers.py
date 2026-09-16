"""README "Deney kaydı: v2-grouped-dispatch" bölümündeki türetilmiş sayıları kayıtlardan yeniden hesaplar.

    .venv\\Scripts\\python experiments/v2-grouped-dispatch/diagnostics/report_numbers.py > report_numbers.log

Amaç: rapordaki hiçbir türetilmiş sayı elle yazılmasın. (İlk taslakta elle yazılan iki sayı — "en büyük aşım 1.13
kat" ve "fp32'de çoğu durumda bit bit aynı" — bir denetimde yanlış çıktı; bu betik o hesapları tek yerde tutuyor.)
GPU gerekmez, yalnızca kayıtları okur.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics as st
import sys

REC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
T975 = {2: 4.302653, 5: 2.570582}


def ci(xs: list[float]) -> str:
    m, sd = st.mean(xs), st.stdev(xs)
    h = T975[len(xs) - 1] * sd / math.sqrt(len(xs))
    return f"{m:+.4g} (sd {sd:.3g}, %95 GA [{m - h:+.4g}, {m + h:+.4g}])"


def speed() -> None:
    s = json.load(open(os.path.join(REC, "summary.json"), encoding="utf-8"))["speed"]
    print("== Hız benchmark'ı (300 adım, 3 dönüşümlü tekrar, medyan tok/s) ==")
    for preset in ("dense", "moe", "moe-noaux"):
        for impl in ("loop", "grouped"):
            k = f"{preset}-{impl}"
            if k in s:
                print(f"  {k:20s} {[round(v / 1e3, 2) for v in s[k]]} ort {st.mean(s[k]) / 1e3:.2f}k tok/s")
    for preset in ("moe", "moe-noaux"):
        g, l = s[f"{preset}-grouped"], s[f"{preset}-loop"]
        rel = [100 * (a / b - 1) for a, b in zip(g, l)]  # tekrar başına eşleştirilmiş (aynı tur)
        print(f"  {preset}: grouped/loop tekrar başına % fark {[round(r, 2) for r in rel]} -> {ci(rel)} %")
    d = st.mean(s["dense-grouped"])
    for preset in ("moe", "moe-noaux"):
        print(f"  dense / {preset}-grouped = {d / st.mean(s[f'{preset}-grouped']):.3f}x, "
              f"dense / {preset}-loop = {d / st.mean(s[f'{preset}-loop']):.3f}x")


def bound(metric: str, r: dict, precision: str) -> float:
    FP32_TOL = {"logits": 1e-6, "grads": 1e-5, "grads_worst_tensor": 1e-5, "update": 1e-4, "loss_after_step": 1e-5}
    fl, sc = r["floor_loop_vs_loop"], r["scale_loop_bf16_vs_fp32"]
    if metric == "teacher_forced":
        return max(3 * r["teacher_forced"]["layer_output_floor_rel_l2_max"], 1e-6)
    return max(3 * fl[metric], 0.25 * sc[metric]) if precision == "bf16" else max(10 * fl[metric], FP32_TOL[metric])


def verify() -> None:
    files = sorted(glob.glob(os.path.join(REC, "verify_dispatch_default_run*.json")))
    metrics = ("teacher_forced", "logits", "grads", "grads_worst_tensor", "update", "loss_after_step")
    n_checks, fails, bit, routing, neg = 0, [], {"fp32": [0, 0, 0, 0], "bf16": [0, 0, 0, 0]}, {"fp32": [0, 0], "bf16": [0, 0]}, {"fp32": [], "bf16": []}
    for fn in files:
        run = re.search(r"run(\d+)", fn).group(1)
        d = json.load(open(fn, encoding="utf-8"))
        for case, r in d["results"].items():
            precision = case.split(" | ")[0]
            g = r["grouped_vs_loop"]
            for m in metrics:
                val = r["teacher_forced"]["layer_output_rel_l2_max"] if m == "teacher_forced" else g[m]
                b = bound(m, r, precision)
                n_checks += 1
                if val > b:
                    fails.append((f"run{run}", case, m, val, b, val / b))
            for i, key in enumerate(("logits_bitexact", "grads_bitexact", "params_bitexact")):
                bit[precision][i] += int(g[key])
            bit[precision][3] += 1
            routing[precision][0] += int(g["routing_identical_all_layers"])
            routing[precision][1] += 1
            for m in ("logits", "grads", "grads_worst_tensor"):
                if g[m] > 0:
                    neg[precision].append(r["negative_control_vs_loop"][m] / g[m])
    print(f"\n== Doğrulama (varsayılan çekirdekler), {len(files)} koşu x 6 durum x 6 metrik = {n_checks} sınır kontrolü ==")
    print(f"  sınırı aşan kontrol sayısı: {len(fails)}")
    for run, case, m, val, b, ratio in sorted(fails, key=lambda t: -t[5]):
        print(f"    {ratio:6.2f}x aşım | {run} | {case} | {m}: {val:.3e} > {b:.3e}")
    # Simetrik sıfır aday (loop tekrarı) aynı sınırlarda: kaç kontrolde ve ne kadar aşıyor?
    null_fails, n_null = [], 0
    for fn in files:
        run = re.search(r"run(\d+)", fn).group(1)
        for case, r in json.load(open(fn, encoding="utf-8"))["results"].items():
            nc = r.get("null_candidate_loop_vs_loop")
            if not nc:
                continue
            for m in ("logits", "grads", "grads_worst_tensor", "update", "loss_after_step"):
                n_null += 1
                b = bound(m, r, case.split(" | ")[0])
                if nc[m] > b:
                    null_fails.append((f"run{run}", case, m, nc[m] / b))
    print(f"  sıfır aday (loop tekrarı) aynı sınırlarda: {n_null} kontrol, {len(null_fails)} aşım" +
          "".join(f"\n    {r:6.2f}x aşım | {run} | {case} | {m}" for run, case, m, r in null_fails))
    print(f"  (sıfır aday run1'de yok, run2-run{re.search(r'run(.)', files[-1]).group(1)}'da var)")
    for p in ("fp32", "bf16"):
        lo, gr, pa, n = bit[p]
        print(f"  {p}: bit bit aynı durum sayısı logits {lo}/{n}, gradyan {gr}/{n}, adım sonrası parametre {pa}/{n} | "
              f"serbest koşuda routing birebir aynı {routing[p][0]}/{routing[p][1]}")
        if neg[p]:
            print(f"    negatif kontrol / grouped oranı: {min(neg[p]):.3g} - {max(neg[p]):.3g} "
                  f"({len(neg[p])} tanımlı oran; grouped'ın bit bit aynı olduğu durumlarda tanımsız)")
    det = json.load(open(os.path.join(REC, "verify_dispatch_deterministic.json"), encoding="utf-8"))["results"]
    n_det = sum(1 for _ in det)
    ok = sum(1 for r in det.values() if all(r["grouped_vs_loop"][k] for k in ("logits_bitexact", "grads_bitexact", "params_bitexact")))
    rout = sum(1 for r in det.values() if r["grouped_vs_loop"]["routing_identical_all_layers"])
    print(f"  deterministik mod: {ok}/{n_det} durumda grouped ile loop bit bit aynı (logits+gradyan+parametre), "
          f"routing {rout}/{n_det}")


def stages() -> None:
    """Aşama süreleri birden çok koşunun medyanı: tam katman ölçümü süreçler arası ~1 ms oynuyor, aşamalar kararlı."""
    logs = [open(p, encoding="utf-8").read() for p in sorted(glob.glob(os.path.join(REC, "diagnostics", "stage_times_moe_run*.log")))]
    names = [m for m in re.findall(r"^  (.+?)\s+fwd\+bwd", logs[0], re.M)]

    def vals(name: str) -> list[float]:
        pat = "^  " + re.escape(name) + r"\s+fwd\+bwd\s+([\d.]+) ms"
        return [float(re.search(pat, log, re.M).group(1)) for log in logs]

    print(f"\n== Aşama süreleri ({len(logs)} koşu; medyan [min-maks]) ==")
    med = {}
    for name in names:
        v = vals(name)
        med[name] = st.median(v)
        print(f"  {name:34s} {st.median(v):6.2f} ms [{min(v):.2f}-{max(v):.2f}]")
    parts = [float(re.search(r"parçalar toplamı \(grouped\): ([\d.]+) ms", log).group(1)) for log in logs]
    opt = [[float(x) for x in re.findall(r"clip \+ AdamW \([^)]*\): ([\d.]+) ms", log)] for log in logs]
    opt_d, opt_m = st.median(o[0] for o in opt), st.median(o[1] for o in opt)
    print(f"  {'parçaların toplamı (grouped)':34s} {st.median(parts):6.2f} ms [{min(parts):.2f}-{max(parts):.2f}]")
    print(f"  {'clip + AdamW dense / moe':34s} {opt_d:6.2f} / {opt_m:.2f} ms")
    s = json.load(open(os.path.join(REC, "summary.json"), encoding="utf-8"))["groups"]
    ms = {g: 32768 / st.mean(s[g]["tok_per_sec"]) * 1e3 for g in ("dense-tok", "moe-tok")}
    measured = ms["moe-tok"] - ms["dense-tok"]
    dense_ffn = med["dense FFN (h=1408)"]
    print(f"\n== Zaman muhasebesi (ölçülen adım farkı {measured:.1f} ms: dense {ms['dense-tok']:.1f}, moe {ms['moe-tok']:.1f}) ==")
    for label, moe_side in (("parçaların toplamı", st.median(parts)), ("tam katman ölçümü", med["MoE tam katman (grouped)"])):
        pred = (moe_side - dense_ffn) * 7 * 2 + (opt_m - opt_d)
        print(f"  {label}: ({moe_side:.2f} - {dense_ffn:.2f}) x 14 = {(moe_side - dense_ffn) * 14:.1f} ms + optimizer "
              f"{opt_m - opt_d:.2f} ms -> {pred:.1f} ms (ölçümün {pred / measured:.2f} katı)")


def flops() -> None:
    f = json.load(open(os.path.join(REC, "flops_report.json"), encoding="utf-8"))
    a, p = f["analytic"], f["profiler"]
    print("\n== FLOP (token başına, eğitim = 3x ileri) ==")
    for k in ("dense", "moe"):
        print(f"  {k}: ileri {a[k]['forward']:,} | eğitim {a[k]['train_per_token']:,}")
    print(f"  moe/dense analitik oran: {a['moe']['train_per_token'] / a['dense']['train_per_token']:.5f} "
          f"(fark router: {a['moe']['forward_terms']['moe_router']:,} / token ileri)")
    for name in p:
        tot = p[name]["forward+backward"]["total_per_token"]
        base = a["dense" if name == "dense" else "moe"]["train_per_token"]
        print(f"  profiler {name:14s} {tot:,.0f} (analitiğin {tot / base:.3f} katı)")
    loop, grp = (p["moe-loop"]["forward+backward"]["total_per_token"], p["moe-grouped"]["forward+backward"]["total_per_token"])
    routed = 3 * a["moe"]["forward_terms"]["moe_routed_experts"]
    print(f"  profiler grouped, loop'tan {100 * (1 - grp / loop):.1f}% düşük; eksik sayılan {loop - grp:,.0f} = "
          f"routed expert'lerin eğitim FLOP'u {routed:,} ({'aynı' if abs(loop - grp - routed) < 1 else 'FARKLI'})")


def runs() -> None:
    s = json.load(open(os.path.join(REC, "summary.json"), encoding="utf-8"))
    g = s["groups"]
    print("\n== Koşu grupları ==")
    for name in sorted(g):
        d = g[name]
        print(f"  {name:15s} son val {st.mean(d['final_val']):.4f} ± {st.stdev(d['final_val']):.4f} "
              f"(ppl {math.exp(st.mean(d['final_val'])):.3f}) | süre {st.mean(d['train_time_min']):.2f} dk | "
              f"tok/s {st.mean(d['tok_per_sec']) / 1e3:.1f}k | takılma maks {max(d['stall_ratio']):.4f} | "
              f"tepe bellek {st.mean(d['peak_mem_gb']):.2f} GB")
    # dense'in token ölçeklenmesi: eşit süre farkını kapatmak için gereken hız
    dt, dti = g["dense-tok"], g["dense-time"]
    slope = (st.mean(dt["final_val"]) - st.mean(dti["final_val"])) / math.log(st.mean(dti["tokens"]) / st.mean(dt["tokens"]))
    gap = st.mean(g["moe-tok"]["final_val"]) - st.mean(dti["final_val"])
    print(f"  dense token ölçeklenmesi: {slope:.3f} nat / ln(token) (3000 -> 4139 adım iki tam koşudan)")
    print(f"  eşit sürede moe - dense farkı {gap:+.4f} nat; aynı eğimle kapanması için moe'nin "
          f"{math.exp(gap / slope):.3f}x daha fazla token (yani ~{math.exp(gap / slope):.2f}x hız) gerekir; "
          f"hız farkının tamamı {st.mean(dt['tok_per_sec']) / st.mean(g['moe-tok']['tok_per_sec']):.2f}x")


def curves() -> None:
    d = json.load(open(os.path.join(REC, "verify_100step.json"), encoding="utf-8"))
    print("\n== 100 adımlık eşdeğerlik ==")
    for preset in ("moe", "moe-noaux"):
        r = d[preset]
        e = r["endpoints"]
        print(f"  {preset}: karar {r['verdict_val_at_end']} | LR aynı {r['lr_sequences_identical']}")
        for k in ("val_at_end", "tail_mean_train_loss"):
            for who in ("grouped", "loopb", "loopc"):
                c = e[f"{k}:{who}_minus_loop"]
                print(f"    {k:22s} {who:8s} {c['mean']:+.4f} GA [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] "
                      f"(işaret: {sum(1 for x in c['per_seed'] if x > 0)}/{len(c['per_seed'])} pozitif)")
        w = r["pre_split_window"]
        print(f"    ayrışma öncesi oran R_grouped {[round(v['R_grouped'], 2) for v in w.values()]} vs "
              f"R_sıfır {[round(v['R_null_loopc'], 2) for v in w.values()]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    speed()
    verify()
    stages()
    flops()
    runs()
    curves()
