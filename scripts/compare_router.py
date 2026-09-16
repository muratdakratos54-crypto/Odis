"""Router dengeleme ablasyonu: altı koşulu 3 seed ile karşılaştırır (denge, uzmanlaşma, sahte denge testleri).

    .venv\\Scripts\\python scripts/compare_router.py --analysis experiments/v3-router-ablation/analysis \\
        --out experiments/v3-router-ablation/summary.json --plot docs/v3_router.png

Koşullar (hepsi aynı veri, aynı seed'ler, 3000 adım x 32 768 token):
  a) aux 0.01  (v1/v2'deki normal ayar)          -> checkpoints/v2/moe-tok-s<seed>
  b) aux 0.001 (zayıflatılmış)                   -> checkpoints/v3/aux0.001-s<seed>
  c) aux yok                                      -> checkpoints/v2/moe-noaux-tok-s<seed>
  d) aux yok + DeepSeek-V3/K2 tarzı dengeleme     -> checkpoints/v3/auxfree-s<seed>   (sigmoid gating + seçim bias'ı)
  e) aux 0.1   (güçlendirilmiş)                   -> checkpoints/v3/aux0.1-s<seed>
  f) aux 1.0   (100x güçlendirilmiş)              -> checkpoints/v3/aux1.0-s<seed>
(e) "aux'u güçlendirmek uzmanlaşmayı bozuyor mu" sorusu için eklendi; (f) ise (e)'nin sonucu görüldükten SONRA,
çünkü (e)'de dengeleme kuvveti hâlâ görev gradyanının binde 2'siydi. (f) bulguları keşifseldir.
DİKKAT: (a) ve (c) v2 koduyla, (b)/(d)/(e)/(f) v3 koduyla eğitildi (softmax yolunun eşdeğerliği
experiments/v3-router-ablation/checks/ altında ayrıca sınandı) ve (a)/(c) farklı bir koşu oturumunda koştu:
tok/s karşılaştırmaları bu yüzden oturum etkisiyle karışıktır.

val loss ve tok/s eğitim kayıtlarından; denge, uzmanlaşma ve karşı-olgusal ölçütler scripts/router_analysis.py
çıktılarından gelir. İstatistik: n = 3 seed, ortalama ± örneklem sd; fark SEED KİMLİĞİNE göre eşleştirilir, %95 GA t(n-1).
Referans (a)'dır, ama serinin kendi içindeki karşılaştırmalar (ör. f vs e) da basılır: "optimumdan kötüleşme"
iddiası (a)'ya karşı test edilemez. Çoklu karşılaştırma: tüm eşleştirilmiş testlere Holm (FWER) ve
Benjamini-Hochberg (FDR) uygulanır; nominal yıldızlar düzeltmesizdir ve tek başına okunmamalıdır.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

T975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582}
CONDITIONS = [
    ("a_aux0.01", "aux 0.01 (normal)", "checkpoints/v2", "moe-tok-s{s}"),
    ("b_aux0.001", "aux 0.001 (zayıf)", "checkpoints/v3", "aux0.001-s{s}"),
    ("c_aux0", "aux yok", "checkpoints/v2", "moe-noaux-tok-s{s}"),
    ("d_auxfree", "aux'suz dengeleme", "checkpoints/v3", "auxfree-s{s}"),
    ("e_aux0.1", "aux 0.1 (güçlü)", "checkpoints/v3", "aux0.1-s{s}"),
    ("f_aux1.0", "aux 1.0 (çok güçlü)", "checkpoints/v3", "aux1.0-s{s}"),
]


def load_training(run_dir: str) -> dict:
    train, val, end = [], [], None
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "train":
                train.append(r)
            elif r["type"] == "val":
                val.append(r)
            elif r["type"] == "end":
                end = r
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    log_interval = cfg["train"]["log_interval"]
    steady = st.median([r["tok_per_sec"] for r in train[2:] if r["step"] % log_interval == 0])
    last = train[-1]
    return {"cv_curve": [(r["step"], r["load_cv"]) for r in train if "load_cv" in r],
            "val_curve": [(r["step"], r["val_loss"]) for r in val],
            "final_val": val[-1]["val_loss"], "ppl": math.exp(val[-1]["val_loss"]), "tok_per_sec": steady,
            "train_time_min": end["train_time"] / 60, "load_cv_train": last.get("load_cv"),
            "dead_train": last.get("dead_experts"), "aux_value": last.get("aux_loss"),
            "bias_range": last.get("bias_range"), "steps": end["step"], "tokens": end["tokens"]}


def ms(xs: list[float], fmt: str = ".4f") -> str:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "—"
    return f"{st.mean(xs):{fmt}} ± {st.stdev(xs):{fmt}}" if len(xs) > 1 else f"{xs[0]:{fmt}}"


def paired(a: list[float], b: list[float], seeds_a=None, seeds_b=None) -> dict:
    """Seed KİMLİĞİNE göre eşleştirilmiş fark. Konuma göre zip, bir koşulda eksik analiz olduğunda sessizce
    yanlış seed'leri eşleştiriyordu; t kritik değeri de n'den bağımsız sabitti (n=2'de GA 3 kat dar çıkardı)."""
    if seeds_a is not None and seeds_b is not None:
        pos = {s: i for i, s in enumerate(seeds_b)}
        pairs = [(a[i], b[pos[s]]) for i, s in enumerate(seeds_a) if s in pos]
    else:
        pairs = list(zip(a, b))
    d = [x - y for x, y in pairs]
    out = {"per_seed": d, "n": len(d), "mean": st.mean(d) if d else float("nan")}
    if len(d) > 1:
        sd = st.stdev(d)
        out["sd"] = sd
        if sd > 0:
            out["t"] = out["mean"] / (sd / math.sqrt(len(d)))
            out["p"] = _t_sf(abs(out["t"]), len(d) - 1) * 2
        h = T975.get(len(d) - 1, 1.96) * sd / math.sqrt(len(d))
        out["ci95"] = [out["mean"] - h, out["mean"] + h]
    return out


def _t_sf(t: float, df: int) -> float:
    """P(T > t), Student-t. Küçük df için düzenlenmiş inkomplet beta; scipy yok."""
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Düzenlenmiş inkomplet beta I_x(a,b), sürekli kesir (Numerical Recipes 6.4)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta)
    if x < (a + 1) / (a + b + 2):
        return front * _bcf(a, b, x) / a
    return 1.0 - math.exp(math.log(1 - x) * b + math.log(x) * a - lbeta) * _bcf(b, a, 1 - x) / b


def _bcf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-16) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def multiplicity(tests: list[tuple[str, float]]) -> dict:
    """Holm (FWER .05) ve Benjamini-Hochberg (FDR .05): hangi nominal anlamlı testler ayakta kalıyor."""
    rows = sorted(((n, p) for n, p in tests if p == p), key=lambda r: r[1])
    m = len(rows)
    holm, bh = set(), set()
    for i, (name, pv) in enumerate(rows):  # Holm: adım adım, ilk başarısızlıkta dur
        if pv <= 0.05 / (m - i):
            holm.add(name)
        else:
            break
    kmax = max((i for i, (_, pv) in enumerate(rows) if pv <= 0.05 * (i + 1) / m), default=None)
    if kmax is not None:
        bh = {name for name, _ in rows[:kmax + 1]}
    return {"n_tests": m, "nominal": [n for n, pv in rows if pv <= 0.05], "holm": sorted(holm), "bh": sorted(bh)}


def fmt_paired(p: dict, fmt: str = "+.4f") -> str:
    s = f"{p['mean']:{fmt}}"
    return s + (f" GA [{p['ci95'][0]:{fmt}}, {p['ci95'][1]:{fmt}}]" if "ci95" in p else "")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", default="experiments/v3-router-ablation/analysis")
    p.add_argument("--out", default="experiments/v3-router-ablation/summary.json")
    p.add_argument("--plot", default="docs/v3_router.png")
    p.add_argument("--seeds", default="1337,1338,1339")
    p.add_argument("--analyze", action="store_true",
                   help="eksik analiz dosyalarını router_analysis.py ile üret (GPU), sonra tabloyu bas")
    p.add_argument("--batches", type=int, default=20)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.analyze:
        import subprocess
        os.makedirs(args.analysis, exist_ok=True)
        for key, _, run_root, pattern in CONDITIONS:
            for s in seeds:
                ckpt = os.path.join(run_root, pattern.format(s=s), "best.pt")
                out = os.path.join(args.analysis, f"{key}-s{s}.json")
                if os.path.exists(out):
                    print(f"[atla] {out} var")
                    continue
                if not os.path.exists(ckpt):
                    print(f"[uyarı] {ckpt} yok, atlandı")
                    continue
                cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "router_analysis.py"),
                       "--ckpt", ckpt, "--out", out, "--batches", str(args.batches)]
                print(f"[analiz] {key}-s{s}")
                rc = subprocess.run(cmd, env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
                if rc != 0:
                    sys.exit(f"router_analysis başarısız: {ckpt}")

    data: dict[str, dict] = {}
    for key, label, run_root, pattern in CONDITIONS:
        rows = []
        for s in seeds:
            run_dir = os.path.join(run_root, pattern.format(s=s))
            apath = os.path.join(args.analysis, f"{key}-s{s}.json")
            if not (os.path.exists(os.path.join(run_dir, "metrics.jsonl")) and os.path.exists(apath)):
                print(f"[uyarı] eksik: {run_dir} / {apath}")
                continue
            with open(apath, encoding="utf-8") as f:
                a = json.load(f)
            # H(expert|token) = H(E) - I(T;E): routing kararının token kimliğiyle açıklanamayan (bağlama bağlı) kısmı.
            # Katman ortalaması, per_layer'dan türetilir. mi_bits ile birlikte okunur: mi_bits yüksek + bu düşük =
            # routing bir sözlüğe (öğrenilmiş hash) yakın; ikisi de yüksek = token kimliği kadar bağlam da belirliyor.
            a["specialization"].setdefault(
                "h_cond_bits", st.mean(r["h_expert_bits"] - r["mi_bits"] for r in a["per_layer"]))
            rows.append({"seed": s, "train": load_training(run_dir), "analysis": a})
        if rows:
            data[key] = {"label": label, "rows": rows}

    def col(key: str, path: list[str]) -> list[float]:
        out = []
        for r in data[key]["rows"]:
            v = r
            for p_ in path:
                v = v.get(p_) if isinstance(v, dict) else None
            out.append(v)
        return out

    FIELDS = [
        ("val loss", ["train", "final_val"], ".4f"),
        ("ppl", ["train", "ppl"], ".3f"),
        ("tok/s (k)", ["train", "tok_per_sec"], ".0f"),
        ("yük CV", ["analysis", "balance", "load_cv"], ".3f"),
        ("ölü expert", ["analysis", "balance", "dead_experts"], ".1f"),
        ("maks pay", ["analysis", "balance", "load_max_ratio"], ".2f"),
        ("mi_bits", ["analysis", "specialization", "mi_bits"], ".3f"),
        ("NMI", ["analysis", "specialization", "nmi"], ".4f"),
        ("H(expert|token) bit", ["analysis", "specialization", "h_cond_bits"], ".3f"),
        ("yük entropisi (top-k)", ["analysis", "balance", "load_entropy"], ".4f"),
        ("top-1 kullanım entropisi", ["analysis", "specialization", "utilization_entropy"], ".4f"),
        ("mi_bits null", ["analysis", "specialization", "mi_bits_null"], ".3f"),
        ("tutarlılık null", ["analysis", "specialization", "consistency_null"], ".3f"),
        ("tutarlılık", ["analysis", "specialization", "consistency"], ".3f"),
        ("expert çeşitliliği", ["analysis", "specialization", "expert_diversity"], ".3f"),
        ("embedding tutarlılığı", ["analysis", "specialization", "embedding_coherence"], ".3f"),
        ("seçim marjı", ["analysis", "router", "margin"], ".3f"),
        ("top-1 gate", ["analysis", "router", "gate_top1"], ".3f"),
        ("ΔCE permütasyon", ["analysis", "counterfactual_ce", "delta_permuted"], "+.4f"),
        ("ΔCE rastgele", ["analysis", "counterfactual_ce", "delta_random"], "+.4f"),
        ("ΔCE ikinci-en-iyi", ["analysis", "counterfactual_ce", "delta_next"], "+.4f"),
        ("ΔCE bias'sız", ["analysis", "counterfactual_ce", "delta_nobias"], "+.4f"),
        ("|aux grad|/|görev grad|", ["analysis", "gradient_tension", "ratio_mean"], ".4f"),
        ("aux-görev kosinüs", ["analysis", "gradient_tension", "cosine_mean"], "+.4f"),
        ("bias ezme oranı", ["analysis", "router", "bias_override"], ".3f"),
    ]
    print("== Router dengeleme ablasyonu: koşullar x 3 seed (ortalama ± sd) ==")
    header = f"{'ölçüt':24s}" + "".join(f"{data[k]['label'][:20]:>22s}" for k in data)
    print(header)
    table = {}
    for name, path, fmt in FIELDS:
        cells = []
        for k in data:
            vals = col(k, path)
            vals = [v * 1e-3 if name == "tok/s (k)" and v else v for v in vals]
            table.setdefault(name, {})[k] = vals
            cells.append(f"{ms(vals, fmt):>22s}")
        if any(c.strip() != "—" for c in cells):
            print(f"{name:24s}" + "".join(cells))

    base = "a_aux0.01"
    print("\n== (a) ile eşleştirilmiş farklar (seed kimliğine göre; %95 GA t(n-1)) ==")
    diffs = {}
    for name, path, fmt in FIELDS:
        if name in ("ΔCE bias'sız", "bias ezme oranı", "|aux grad|/|görev grad|", "aux-görev kosinüs"):
            continue
        b = table[name][base]
        if any(v is None for v in b):
            continue
        for k in data:
            if k == base:
                continue
            v = table[name][k]
            if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in v):
                continue
            d = paired(v, b, [r["seed"] for r in data[k]["rows"]], [r["seed"] for r in data[base]["rows"]])
            diffs.setdefault(name, {})[k] = d
    for name in diffs:
        print(f"  {name:24s}" + " | ".join(f"{k.split('_')[0]}: {fmt_paired(diffs[name][k])}" for k in diffs[name]))

    # Seri içi karşılaştırmalar: "optimumdan kötüleşme" iddiası (a)'ya karşı test EDİLEMEZ, çünkü referans
    # serinin optimumu değil. aux serisinin komşu çiftleri + serinin en iyi val loss'lu kolu ayrıca hesaplanır.
    SERIES = ["c_aux0", "b_aux0.001", "a_aux0.01", "e_aux0.1", "f_aux1.0"]
    order = [k for k in SERIES if k in data]
    best = min(order, key=lambda k: st.mean(table["val loss"][k]))
    pairs = [(order[i + 1], order[i]) for i in range(len(order) - 1)] + [(k, best) for k in order if k != best]
    KEYS = ("val loss", "mi_bits", "H(expert|token) bit", "tutarlılık", "yük CV", "seçim marjı", "ΔCE permütasyon")
    series_diffs = {}
    print(f"\n== Seri içi eşleştirilmiş farklar (komşu çiftler + serinin en iyisi = {data[best]['label']}) ==")
    for hi, lo in pairs:
        cells = []
        for name in KEYS:
            v, b = table.get(name, {}).get(hi), table.get(name, {}).get(lo)
            if not v or not b or any(x is None for x in v) or any(x is None for x in b):
                continue
            d = paired(v, b, [r["seed"] for r in data[hi]["rows"]], [r["seed"] for r in data[lo]["rows"]])
            series_diffs.setdefault(f"{hi} - {lo}", {})[name] = d
            star = "*" if "ci95" in d and (d["ci95"][0] > 0 or d["ci95"][1] < 0) else ""
            cells.append(f"{name} {d['mean']:+.4f}{star}")
        print(f"  {hi:12s} - {lo:12s}: " + " | ".join(cells))

    # Çoklu karşılaştırma: tüm eşleştirilmiş testler tek havuzda (nominal yıldızlar düzeltmesizdir).
    pool = [(f"{n} | {k} vs a", diffs[n][k].get("p", float("nan"))) for n in diffs for k in diffs[n]]
    pool += [(f"{n} | {pair}", d.get("p", float("nan"))) for pair, ds in series_diffs.items() for n, d in ds.items()]
    mult = multiplicity(pool)
    print(f"\n== Çoklu karşılaştırma ({mult['n_tests']} eşleştirilmiş test, hepsi n=3) ==")
    print(f"  nominal p<=.05: {len(mult['nominal'])} | Holm (FWER .05): {len(mult['holm'])} "
          f"| BH (FDR .05): {len(mult['bh'])}")
    print("  Holm'da ayakta kalanlar:")
    for name in mult["holm"]:
        print(f"    {name}")

    summary = {"conditions": {k: {"label": v["label"], "seeds": [r["seed"] for r in v["rows"]]} for k, v in data.items()},
               "table": {n: {k: table[n][k] for k in table[n]} for n in table},
               "paired_vs_a": {n: {k: diffs[n][k] for k in diffs[n]} for n in diffs},
               "series_paired": series_diffs, "multiplicity": mult, "series_best": best}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n-> {args.out}")

    print("\n== Katman 4 lift örnekleri (seed 1337) ==")
    for k in data:
        row = next((r for r in data[k]["rows"] if r["seed"] == seeds[0]), None)
        if row and "lift_examples" in row["analysis"]:
            print(f"  {data[k]['label']}:")
            for e, toks in enumerate(row["analysis"]["lift_examples"]["experts"][:6]):
                print(f"    expert {e}: {' '.join(toks[:6])}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        CV_X = ("yük CV", "yük CV (0 = mükemmel denge)")
        panels = [  # (x, y): son panel eğitim eğrisi, ondan öncekiler saçılım
            (CV_X, ("mi_bits", "I(token; expert) [bit]")),
            (CV_X, ("ΔCE permütasyon", "ΔCE: expert kimlikleri permüte")),
            (CV_X, ("val loss", "val loss")),
            (("H(expert|token) bit", "H(expert|token) [bit] = routing'in bağlama bağlı kısmı"), ("val loss", "val loss")),
        ]
        fig, axes = plt.subplots(1, len(panels) + 1, figsize=(5.5 * (len(panels) + 1), 5))
        palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#e377c2"]
        colors = {k: palette[i % len(palette)] for i, k in enumerate(data)}
        # Permütasyon karşı-olgusalı ölü expert olan kollarda okunamaz (tokenların büyük kısmı fiilen
        # eğitilmemiş modüllere gider), o kollar içi boş işaretle çizilir.
        invalid_perm = {k for k in data if st.mean(table["ölü expert"][k]) > 0}
        for ax, ((xm, xlabel), (ym, ylabel)) in zip(axes, panels):
            for k in data:
                x, y = table[xm][k], table[ym][k]
                if any(v is None for v in x) or any(v is None for v in y):
                    continue
                hollow = ym == "ΔCE permütasyon" and k in invalid_perm
                ax.scatter(x, y, s=60, label=data[k]["label"], edgecolors=colors[k],
                           facecolors="none" if hollow else colors[k])
                if not hollow:
                    ax.scatter([st.mean(x)], [st.mean(y)], color=colors[k], marker="x", s=200)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if ym == "ΔCE permütasyon":
                ax.set_title("içi boş = ölü expert var, test geçersiz", fontsize=8)
        for k in data:  # son panel: dengenin eğitim boyunca gelişimi (seed ortalaması)
            curves = [r["train"]["cv_curve"] for r in data[k]["rows"]]
            n = min(len(c) for c in curves)
            steps = [curves[0][i][0] for i in range(n)]
            mean = [st.mean(c[i][1] for c in curves) for i in range(n)]
            axes[-1].plot(steps, mean, color=colors[k], label=data[k]["label"])
        axes[-1].set_xlabel("adım")
        axes[-1].set_ylabel("yük CV (eğitim logu, seed ortalaması)")
        axes[-1].set_yscale("log")
        axes[-1].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle("v3: router dengeleme ablasyonu (3 seed, nokta = seed, x = ortalama). Dengeli kollar arasında "
                     "yük CV val loss'u açıklamıyor; H(expert|token) ile ilişki güçlü ama gözlemsel ve "
                     "mi_bits'in aynası (r = -0.97), bağımsız kanıt değil.")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"-> {args.plot}")


if __name__ == "__main__":
    main()
