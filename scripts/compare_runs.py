"""KARANTİNA — v2 DÖNEMİ BETİĞİ, FOTON HATTINDA KULLANILAMAZ. FOTON dizinlerine yöneltilirse hata verip çıkar.

Bu betik kendi `paired(a: list, b: list)` fonksiyonunu taşır ve iki listeyi `zip()` ile KONUMA göre
eşler. Konuma göre eşleme hatası bu depoda üç kez doğdu (`src/stats.paired` v3'te, `savings` ve
`decide_primary` FOTON v0.2 ön-kaydının denetimlerinde) ve her seferinde yanlış seed'leri sessizce
eşledi. FOTON'un karar yolu artık yalnızca seed kimliğiyle anahtarlı eşleme kabul eden `src/stats.py`'yi
kullanır (PREREGISTRATION_V02.md V4, "VERİ TAŞIMA KURALI"). Bu dosya yalnızca v2 sonuçlarının
(`checkpoints/v2`, `experiments/v2-grouped-dispatch`) yeniden üretilebilmesi için duruyor; burada
koşu adları sabit bir desenle ve sıralı seed listesiyle okunduğu için konum eşlemesi o veri üzerinde
doğru sonuç veriyordu. Dördüncü doğum yeri olmasın diye `refuse_foton()` her girdi yolunu ve
yüklenen her `config.json`'u denetler.

v2 ölçümleri: dispatch hız benchmark'ı + dense / moe / moe-noaux için dört karşılaştırma (3 seed).

    .venv\\Scripts\\python scripts/compare_runs.py --runs checkpoints/v2 --speed checkpoints/v2-speed \\
        --out experiments/v2-grouped-dispatch/summary.json --plot docs/v2_comparisons.png

Koşu adları (checkpoints/v2):  <model>-<bütçe>-s<seed>
  model : dense | moe | moe-noaux
  bütçe : tok  = 3000 adım x 32 768 token (aynı token; aktif parametreler %0.2 içinde eşit -> aynı aktif parametre)
          flop = dense'in moe-tok ile aynı analitik eğitim FLOP'u (src/flops.py)
          time = moe-tok'un ortalama eğitim duvar saatine eşit süre (bütçe token olarak verilir, LR token'a bağlı)
Hız benchmark'ı (checkpoints/v2-speed): speed-<preset>-<impl>-r<tekrar>, dönüşümlü sırayla koşulmuş kısa koşular.

İstatistik: n = 3 seed. Ortalama ± örneklem standart sapması; model farkı seed başına eşleştirilir (aynı seed = aynı
veri sırası), eşleştirilmiş farkın %95 güven aralığı t(2) = 4.303 ile. n = 3'te aralıklar geniştir; bu bilerek raporlanır.

Duvar saati: train_time (eval/checkpoint hariç). Her koşu için takılma denetimi: train_time / (token / kararlı tok/s)
> 1.03 ise uyarı (arka plan işi, termal kısma ...). Eşit süre bütçesi ham ortalama süreden değil kararlı tok/s'ten
hesaplanır: dense_time_tokens = moe-tok token'ı x ort(dense-tok kararlı tok/s) / ort(moe-tok kararlı tok/s),
tam adıma yukarı yuvarlanır (--budget ile yazdırılır ve summary.json'a girer).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics as st
import sys
from dataclasses import fields

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import ModelConfig  # noqa: E402
from src.flops import train_flops_per_token  # noqa: E402
from src.model import GPT  # noqa: E402

T975_DF2 = 4.302653


def refuse_foton(path: str, config: dict | None = None) -> None:
    """FOTON verisine yöneltilirse ÇIK. Yol adı ya da koşunun kendi config'i FOTON'u gösteriyorsa durur.

    Yalnızca yol adına bakmak yetmez (dizin yeniden adlandırılabilir); config'in üç işareti de
    denetlenir: FOTON korpusu, prefix maskeli ölçüt, FOTON kolu.
    """
    why = []
    if "foton" in os.path.normpath(path).lower():
        why.append(f"yol FOTON'u gösteriyor: {path}")
    if config is not None:
        tr, mo = config.get("train", {}), config.get("model", {})
        if "foton" in str(tr.get("data_dir", "")).lower():
            why.append(f"data_dir = {tr.get('data_dir')!r}")
        if tr.get("loss_prefix_skip", 0):
            why.append(f"loss_prefix_skip = {tr.get('loss_prefix_skip')} (FOTON ölçütü)")
        if mo.get("foton"):
            why.append(f"model.foton = {mo.get('foton')!r}")
    if why:
        sys.exit("KARANTİNA: scripts/compare_runs.py FOTON verisinde KULLANILAMAZ (" + "; ".join(why) + "). "
                 "Bu betik iki listeyi zip() ile KONUMA göre eşler. FOTON analizi için src/stats.py "
                 "(savings -> decide_primary -> abandon) kullanılır.")


def load(run_dir: str) -> dict:
    refuse_foton(run_dir)
    rec = {"train": [], "val": [], "end": None}
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] in ("train", "val"):
                rec[r["type"]].append(r)
            elif r["type"] == "end":
                rec["end"] = r
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        rec["config"] = json.load(f)
    refuse_foton(run_dir, rec["config"])
    return rec


def steady_tok_s(rec: dict) -> float:
    """Log aralıklarının medyan tok/s'i; ilk iki aralık (CUDA/allocator ısınması) ve kısa son aralık hariç."""
    log_interval = rec["config"]["train"]["log_interval"]
    xs = [r["tok_per_sec"] for r in rec["train"][2:] if r["step"] % log_interval == 0]
    return st.median(xs)


def ms(xs: list[float], fmt: str = ".4f") -> str:
    return f"{st.mean(xs):{fmt}} ± {st.stdev(xs):{fmt}}" if len(xs) > 1 else f"{xs[0]:{fmt}}"


def paired(a: list[float], b: list[float]) -> dict:
    d = [x - y for x, y in zip(a, b)]
    out = {"per_seed": d, "mean": st.mean(d)}
    if len(d) > 1:
        sd = st.stdev(d)
        half = T975_DF2 * sd / math.sqrt(len(d)) if len(d) == 3 else float("nan")
        out.update(sd=sd, ci95=[out["mean"] - half, out["mean"] + half])
    return out


def fmt_paired(p: dict) -> str:
    s = f"{p['mean']:+.4f}"
    if "sd" in p:
        s += f" (sd {p['sd']:.4f}, %95 GA [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}])"
    return s + " | seed başına " + ", ".join(f"{x:+.4f}" for x in p["per_seed"])


def speed_section(speed_dir: str) -> dict:
    groups: dict[tuple[str, str], list[float]] = {}
    for name in sorted(os.listdir(speed_dir)):
        m = re.fullmatch(r"speed-(.+)-(loop|grouped)-r(\d+)", name)
        if not m or not os.path.exists(os.path.join(speed_dir, name, "metrics.jsonl")):
            continue
        rec = load(os.path.join(speed_dir, name))
        if rec["end"] is None:
            continue
        groups.setdefault((m.group(1), m.group(2)), []).append(steady_tok_s(rec))
    print("\n== Dispatch hız benchmark'ı (tam eğitim adımı, medyan tok/s, dönüşümlü tekrarlar) ==")
    out = {}
    for (preset, impl), xs in sorted(groups.items()):
        out[f"{preset}-{impl}"] = xs
        print(f"  {preset:10s} {impl:8s}: " + ", ".join(f"{x / 1e3:.1f}k" for x in xs) + f"  -> ort. {st.mean(xs) / 1e3:.2f}k")
    for preset in sorted({p for p, _ in groups}):
        if (preset, "loop") in groups and (preset, "grouped") in groups:
            r = st.mean(groups[(preset, "grouped")]) / st.mean(groups[(preset, "loop")])
            out[f"{preset}-grouped_over_loop"] = r
            print(f"  {preset}: grouped / loop = {r:.4f} ({(r - 1) * 100:+.1f}%)")
    dense = [xs for (p, _), xs in groups.items() if p == "dense"]
    if dense:
        d = st.mean(dense[0])
        for preset in ("moe", "moe-noaux"):
            for impl in ("loop", "grouped"):
                if (preset, impl) in groups:
                    ratio = d / st.mean(groups[(preset, impl)])
                    out[f"dense_over_{preset}-{impl}"] = ratio
                    print(f"  dense / {preset}-{impl} = {ratio:.3f}x yavaşlama")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="checkpoints/v2")
    p.add_argument("--speed", default="checkpoints/v2-speed")
    p.add_argument("--out", default="experiments/v2-grouped-dispatch/summary.json")
    p.add_argument("--plot", default="docs/v2_comparisons.png")
    p.add_argument("--budget", action="store_true", help="yalnızca eşit süre bütçesini hesapla ve yazdır")
    args = p.parse_args()
    for path in (args.runs, args.speed, args.out, args.plot):
        refuse_foton(path)
    summary = {}
    if os.path.isdir(args.speed) and not args.budget:
        summary["speed"] = speed_section(args.speed)

    groups: dict[str, dict[int, dict]] = {}
    for name in sorted(os.listdir(args.runs) if os.path.isdir(args.runs) else []):
        m = re.fullmatch(r"(dense|moe|moe-noaux)-(tok|flop|time)-s(\d+)", name)
        if not m:
            continue
        rec = load(os.path.join(args.runs, name))
        if rec["end"] is None:
            print(f"[uyarı] {name} bitmemiş, atlandı")
            continue
        mcfg = ModelConfig(**{f.name: rec["config"]["model"][f.name] for f in fields(ModelConfig)
                              if f.name in rec["config"]["model"]})
        final = rec["val"][-1]
        assert final["step"] == rec["end"]["step"], name
        tokens = rec["end"]["tokens"]
        rec["summary"] = {
            "steps": rec["end"]["step"], "tokens": tokens, "train_flops": train_flops_per_token(mcfg) * tokens,
            "train_time_min": rec["end"]["train_time"] / 60, "tok_per_sec": steady_tok_s(rec),
            "final_val": final["val_loss"], "best_val": rec["end"]["best_val"],
            "active_params": GPT(mcfg).num_params(active_only=True), "moe_impl": mcfg.moe_impl,
            "peak_mem_gb": max(r["peak_mem_gb"] for r in rec["train"]),
        }
        last = rec["train"][-1]
        if "load_cv" in last:
            rec["summary"].update(load_cv=last["load_cv"], dead_experts=last["dead_experts"], aux_loss=last["aux_loss"])
        stall = rec["end"]["train_time"] / (tokens / rec["summary"]["tok_per_sec"])
        rec["summary"]["stall_ratio"] = stall
        if stall > 1.03:
            print(f"[uyarı] {name}: eğitim süresi kararlı hızdan beklenenin {stall:.3f} katı (takılma / arka plan yükü?)")
        groups.setdefault(f"{m.group(1)}-{m.group(2)}", {})[int(m.group(3))] = rec

    tps = 32768

    def steady(g: str) -> float:
        return st.mean(r["summary"]["tok_per_sec"] for r in groups[g].values())

    if "moe-tok" in groups and "dense-tok" in groups:
        moe_tokens = st.mean(r["summary"]["tokens"] for r in groups["moe-tok"].values())
        budget = {"reference": "moe-tok kararlı süre", "moe_tok_steady_time_min": moe_tokens / steady("moe-tok") / 60,
                  "dense_time_tokens": math.ceil(moe_tokens * steady("dense-tok") / steady("moe-tok") / tps) * tps}
        if "moe-noaux-tok" in groups:
            budget["noaux_over_moe_steady_speed"] = steady("moe-noaux-tok") / steady("moe-tok")
            budget["noaux_time_tokens"] = math.ceil(moe_tokens * steady("moe-noaux-tok") / steady("moe-tok") / tps) * tps
        summary["equal_time_budget"] = budget
        print(f"\n== Eşit süre bütçesi (kararlı tok/s'ten) ==\n  {json.dumps(budget, ensure_ascii=False)}")
        if args.budget:
            return

    print("\n== Koşu grupları (ortalama ± sd, n = seed sayısı) ==")
    table = {}
    for g, runs in groups.items():
        s = [r["summary"] for r in runs.values()]
        table[g] = {"seeds": sorted(runs), **{k: [x[k] for x in s] for k in s[0]}}
        print(f"  {g:14s} n={len(s)} | adım {s[0]['steps']} | {s[0]['tokens'] / 1e6:.2f}M token | "
              f"FLOP {s[0]['train_flops']:.4e} | eğitim {ms([x['train_time_min'] for x in s], '.2f')} dk | "
              f"{ms([x['tok_per_sec'] / 1e3 for x in s], '.1f')}k tok/s | takılma {max(x['stall_ratio'] for x in s):.3f} maks | "
              f"son val {ms([x['final_val'] for x in s])} | "
              f"aktif {s[0]['active_params'] / 1e6:.2f}M" + (f" | yük CV {ms([x['load_cv'] for x in s], '.3f')}, ölü "
                                                              f"{ms([x['dead_experts'] for x in s], '.1f')}" if "load_cv" in s[0] else ""))
    summary["groups"] = table

    comps = {}
    # 4. karşılaştırma (aynı aktif parametre) ayrı koşu değildir: presetler aktif parametre eşit olacak şekilde kurulduğu
    # için (moe/dense = 1.0019x) aynı token koşularıyla birebir aynıdır; bu tabloda bunu açıkça göstermek için tekrarlanır.
    spec = [
        ("1_ayni_token", "dense-tok", ["moe-tok", "moe-noaux-tok"]),
        ("2_ayni_analitik_flop", "dense-flop", ["moe-tok", "moe-noaux-tok"]),
        ("3_ayni_duvar_saati", "dense-time", ["moe-tok", "moe-noaux-time" if "moe-noaux-time" in groups else "moe-noaux-tok"]),
        ("4_ayni_aktif_parametre (= 1 ile aynı koşular)", "dense-tok", ["moe-tok", "moe-noaux-tok"]),
    ]
    if "dense-flop" in groups and "dense-tok" in groups:
        seeds = sorted(set(groups["dense-flop"]) & set(groups["dense-tok"]))
        comps["dense_flop_minus_dense_tok"] = paired([groups["dense-flop"][s]["summary"]["final_val"] for s in seeds],
                                                     [groups["dense-tok"][s]["summary"]["final_val"] for s in seeds])
        print(f"\n  dense-flop (3005 adım) - dense-tok (3000 adım), aynı seed: {fmt_paired(comps['dense_flop_minus_dense_tok'])}"
              f"  (5 adımlık bütçe farkının beklenen etkisi ~1e-4 mertebesi; bu satır aynı kodun tekrar gürültüsünü de gösterir)")
    print("\n== Dört karşılaştırma: son val loss farkı (model - dense), seed başına eşleştirilmiş ==")
    for key, base, others in spec:
        if base not in groups or any(o not in groups for o in others):
            print(f"  {key}: eksik grup, atlandı")
            continue
        seeds = sorted(set(groups[base]).intersection(*(groups[o] for o in others)))
        c = {"baseline": base, "seeds": seeds, "matched": {}}
        for o in others + [base]:
            s = [groups[o][sd]["summary"] for sd in seeds]
            c["matched"][o] = {"tokens": st.mean(x["tokens"] for x in s), "train_flops": st.mean(x["train_flops"] for x in s),
                               "train_time_min": [x["train_time_min"] for x in s], "active_params": s[0]["active_params"]}
        print(f"  {key} (taban {base}, seed {seeds}):")
        b = [groups[base][sd]["summary"]["final_val"] for sd in seeds]
        for o in others:
            v = [groups[o][sd]["summary"]["final_val"] for sd in seeds]
            c[o] = paired(v, b)
            mo, mb = c["matched"][o], c["matched"][base]
            print(f"    {o:14s} - {base}: {fmt_paired(c[o])}")
            print(f"       eşleşme: token {mo['tokens'] / mb['tokens']:.4f}x, analitik FLOP {mo['train_flops'] / mb['train_flops']:.4f}x, "
                  f"eğitim süresi {st.mean(mo['train_time_min']) / st.mean(mb['train_time_min']):.4f}x, "
                  f"aktif parametre {mo['active_params'] / mb['active_params']:.4f}x")
        if len(others) == 2:
            c["noaux_minus_moe"] = paired([groups[others[1]][sd]["summary"]["final_val"] for sd in seeds],
                                          [groups[others[0]][sd]["summary"]["final_val"] for sd in seeds])
            print(f"    {others[1]} - {others[0]}: {fmt_paired(c['noaux_minus_moe'])}")
        comps[key] = c
    summary["comparisons"] = comps

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n-> {args.out}")

    if args.plot and groups:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        style = {"dense-tok": ("#1f77b4", "-"), "dense-flop": ("#17becf", ":"), "dense-time": ("#1f77b4", "--"),
                 "moe-tok": ("#d62728", "-"), "moe-noaux-tok": ("#ff7f0e", "-"), "moe-noaux-time": ("#ff7f0e", "--")}
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
        for g, runs in sorted(groups.items()):
            color, ls = style.get(g, ("gray", "-"))
            vals = [r["val"] for r in runs.values()]
            n = min(len(v) for v in vals)
            mcfg = ModelConfig(**{f.name: next(iter(runs.values()))["config"]["model"][f.name] for f in fields(ModelConfig)
                                  if f.name in next(iter(runs.values()))["config"]["model"]})
            fpt = train_flops_per_token(mcfg)
            mean = [st.mean(v[i]["val_loss"] for v in vals) for i in range(n)]
            lo = [min(v[i]["val_loss"] for v in vals) for i in range(n)]
            hi = [max(v[i]["val_loss"] for v in vals) for i in range(n)]
            xs = {"tok": [vals[0][i]["tokens"] / 1e6 for i in range(n)],
                  "flop": [vals[0][i]["tokens"] * fpt / 1e16 for i in range(n)],
                  "time": [st.mean(v[i]["train_time"] for v in vals) / 60 for i in range(n)]}
            for ax, key in zip(axes, ("tok", "flop", "time")):
                ax.plot(xs[key], mean, color=color, ls=ls, label=f"{g} (n={len(vals)})")
                ax.fill_between(xs[key], lo, hi, color=color, alpha=0.15)
        for ax, lab in zip(axes, ("görülen token (M)", "analitik eğitim FLOP (1e16)", "eğitim duvar saati (dk, eval hariç)")):
            ax.set_xlabel(lab)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("val loss (ortalama, bant = min-maks seed)")
        axes[0].set_ylim(1.28, 1.8)
        axes[0].legend()
        fig.suptitle("v2: dense vs MoE — aynı token / aynı analitik FLOP / aynı duvar saati")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"-> {args.plot}")


if __name__ == "__main__":
    main()
