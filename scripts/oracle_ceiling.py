"""ORACLE TAVANI: her dizide en iyi modülü MÜKEMMEL seçen bir router ne kazanırdı? (TEŞHİS; karar kuralı değil)

    .venv\\Scripts\\python scripts/oracle_ceiling.py --runs checkpoints/foton-v02-tarama --tag v7 \\
        --out experiments/foton-v02/oracle_ceiling.json

Mevcut veriden, yeni koşu olmadan: koşuların `eval_raw/step_*.npz` ham dökümleri sabit 1280 val dizisinin HER
BİRİ için üç modülün ölçüt-pozisyonu (64..511) CE'sini taşır (`seq_masked` [N, 3]). Buradan:

  * `fixed[m]`      : bütün diziler modül m'den geçseydi val loss (sabit-tek-modül mimarisi);
  * `best_fixed`    : min_m fixed[m] — TABAN. En iyi modül aynı veride sonradan seçildiği için olabilecek en GÜÇLÜ
                      tabandır ve tavanı küçültür; tavan olduğundan küçük görünmesin diye aynı fark en ZAYIF makul
                      tabanla (`mean_fixed`, üç modülün ortalaması) ve router'ın kendisiyle (`routed`) de raporlanır;
  * `oracle`        : ort_b min_m seq_masked[b, m] — geriye dönük MÜKEMMEL dizi-düzeyi seçim (hedefleri görerek);
  * `domain_oracle` : alan başına en iyi modül — prefix alanı taşıdığı için (prob %97) bir prefix router'ın
                      GERÇEKÇİ olarak öğrenebileceği en iyi politika. Hem örneklem-içi hem ikiye-bölmeli (modül bir
                      yarıda seçilir, öbür yarıda değerlendirilir; 200 rastgele bölmenin ortalaması) verilir;
  * `routed`        : router'ın gerçekte seçtikleri (koşunun `val_loss`'u);
  * `oracle_null`   : koşunun logladığı B11 null'ı (modül kayıpları pozisyon başına permüte edilmiş sözde-modüllere
                      aynı oracle) — oracle kazancının SEÇİM GÜRÜLTÜSÜ payı. Üç modül istatistiksel olarak aynı olsa
                      bile "dizide en küçüğü seç" eğriyi aşağı çeker; tavanın yapısal kısmı `oracle_null - oracle`'dır.

PUANA ÇEVİRME. Birincil ölçüt compute-to-L* tasarrufudur (%); nat farkı eğri eğimiyle puana çevrilir:
`puan = 100 * (1 - exp(-delta / |d val / d ln token|))`. Üç eğim raporlanır: ön-kaydın V9'da çivilediği L* çevresi
yerel eğim (-0.3346; "%5 = 0.01716 nat"), V9'un duyarlılık ucu olan geç sekant (-0.2385), ve ölçümün yapıldığı yerde
koşunun KENDİ val eğrisinin yerel eğimi (son üç eval noktası). Başarı eşiği +%5'tir (ön-kayıt KARAR TABLOSU).

KAPSAM SINIRI — raporda da yazılmalı: bu tavan, RASTGELE'ye yakın bir yönlendirmeyle eğitilmiş modüller içindir ve
%50 bütçededir. "Daha iyi bir ROUTER bu modüllerle ne kazanırdı"yı sınırlar; "kararlı yönlendirmeyle birlikte eğitilen
modüller ne kadar ayrışırdı"yı SINIRLAMAZ. Val loss burada bir SEÇİM için kullanılmaz.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

SLOPE_LSTAR_LOCAL = -0.3346        # ön-kayıt V9: d(val)/d(ln token), [3250, 3750] penceresi, foton-m
SLOPE_LATE_SECANT = -0.2385        # ön-kayıt V9: [3500, 4375] sekantı (duyarlılık ucu)
SUCCESS_THRESHOLD_POINTS = 5.0     # ön-kayıt KARAR TABLOSU: başarı eşiği +%5
ORACLE_ABANDON_POINTS = 2.0        # ön-kayıt VAZGEÇME KOŞULU: oracle kazancı (null'ın üstünde) < %2


def points(delta_nat: float, slope: float) -> float:
    """Nat cinsinden val loss farkının compute-to-target tasarrufu karşılığı (%), verilen d(val)/d(ln token) eğiminde."""
    if not slope < 0:
        raise ValueError(f"eğim negatif olmalı (loss token'la AZALIR), verilen: {slope}")
    return 100.0 * (1.0 - math.exp(-delta_nat / abs(slope)))


def domain_oracle(sm: np.ndarray, dom: np.ndarray, fit: np.ndarray, test: np.ndarray) -> float:
    """Alan başına en iyi modül `fit` dizilerinde SEÇİLİR, kayıp `test` dizilerinde ölçülür (diziler üzerinden ortalama)."""
    total, n = 0.0, 0
    for d in np.unique(dom):
        f, t = fit & (dom == d), test & (dom == d)
        if not t.any():
            continue
        m = sm[f].mean(0).argmin() if f.any() else sm[fit].mean(0).argmin()
        total += sm[t, m].sum()
        n += int(t.sum())
    return total / n


def ceiling(sm: np.ndarray, choice: np.ndarray, dom: np.ndarray, n_split: int = 200, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bir eval noktasının tavan ölçüleri. sm: [N, M] dizi başına ölçüt-pozisyonu CE'si (float64'e çevrilir)."""
    sm = np.asarray(sm, dtype=np.float64)
    n, n_mod = sm.shape
    if choice.shape != (n,) or dom.shape != (n,) or n_mod < 2:
        raise ValueError(f"biçim beklenmedik: sm {sm.shape}, choice {choice.shape}, domain {dom.shape}")
    rng = np.random.default_rng(seed)
    allm = np.ones(n, dtype=bool)
    fixed = sm.mean(0)
    best_m = int(fixed.argmin())
    seq_min = sm.min(1)
    out = {
        "n": n,
        "fixed": [float(v) for v in fixed],
        "best_fixed_module": best_m,
        "best_fixed": float(fixed[best_m]),
        "mean_fixed": float(fixed.mean()),
        "routed": float(sm[np.arange(n), choice].mean()),
        "oracle": float(seq_min.mean()),
        "domain_oracle_in_sample": float(domain_oracle(sm, dom, allm, allm)),
    }
    halves = []
    for _ in range(n_split):
        fit = rng.permutation(n) < n // 2
        halves.append(domain_oracle(sm, dom, fit, ~fit) - sm[~fit].mean(0)[sm[fit].mean(0).argmin()])
    # ikiye-bölmeli: alan-oracle'ının, AYNI bölmede seçilmiş en iyi sabit modüle karşı farkı (negatif = alan-oracle daha iyi)
    out["domain_oracle_minus_best_fixed_split_half"] = float(np.mean(halves))
    out["ceiling_vs_best_fixed"] = out["best_fixed"] - out["oracle"]
    out["ceiling_vs_mean_fixed"] = out["mean_fixed"] - out["oracle"]
    out["ceiling_vs_routed"] = out["routed"] - out["oracle"]
    out["domain_gain_vs_best_fixed_in_sample"] = out["best_fixed"] - out["domain_oracle_in_sample"]
    # diziler üzerinden bootstrap: tavanın (en iyi sabit modül her yeniden örneklemde YENİDEN seçilir) %95 aralığı
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = sm[idx].mean(0).min() - seq_min[idx].mean()
    out["ceiling_vs_best_fixed_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def token_level_raw(ce_tok: np.ndarray, k: int) -> dict:
    """HAM token-düzeyi "her pozisyonda en iyi modül" minimumu. ce_tok: [N, M, T]; ölçüt pozisyonları k..T-1.

    BİR TAVAN DEĞİLDİR ve tek başına raporlanmamalıdır: (i) neredeyse tamamı seçim gürültüsüdür ve v02'nin B11 null'ı bu
    düzeyde DEJENEREDİR (pozisyon-başı permütasyon pozisyon-başı minimumu değiştirmez: null ≡ oracle — `null_gain` bu
    yüzden kurgu gereği 0 okur ve burada bunu GÖSTERMEK için hesaplanır); (ii) kayıplar dizinin TAMAMI tek modüle
    zorlanarak ölçüldü — dizi ortasında modül değiştirmek sonraki gizli durumları değiştirir, yani gerçekleştirilemez bir
    karşı-olgusaldır. v03 notlarının Yön 3'ü için: "mevcut dökümden ucuz bir token-düzeyi ön-sınav" YOKTUR.
    """
    x = np.asarray(ce_tok[:, :, k:], dtype=np.float64)
    fixed = x.mean(axis=(0, 2))
    tok_min = x.min(axis=1)
    rng = np.random.default_rng(0)
    perm = rng.permuted(x, axis=1)                         # modül kayıpları her (dizi, pozisyon)'da kendi aralarında karışır
    return {"best_fixed": float(fixed.min()), "token_oracle": float(tok_min.mean()),
            "raw_gap_vs_best_fixed": float(fixed.min() - tok_min.mean()),
            "per_token_module_spread": float((x.max(axis=1) - tok_min).mean()),
            "null_gain": float(perm.min(axis=1).mean() - tok_min.mean())}


def local_slope(val_rows: list[dict], k: int = 3) -> float:
    """Koşunun kendi val eğrisinin son k eval noktasındaki d(val)/d(ln token) eğimi (en küçük kareler)."""
    rows = [r for r in val_rows if r["tokens"] > 0][-k:]
    if len(rows) < 2:
        raise ValueError("yerel eğim için en az iki eval noktası gerekir")
    x, y = np.log([r["tokens"] for r in rows]), np.array([r["val_loss"] for r in rows])
    return float(np.polyfit(x, y, 1)[0])


def run_ceiling(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        val = {r["step"]: r for r in map(json.loads, f) if r.get("type") == "val"}
    files = sorted(glob.glob(os.path.join(run_dir, "eval_raw", "step_*.npz")))
    if not files:
        raise ValueError(f"{run_dir}: eval_raw dökümü yok")
    steps = []
    for f in files:
        with np.load(f) as z:
            step, sm, choice, dom = int(z["step"]), z["seq_masked"], z["choice"], z["domain"]
        c = ceiling(sm, choice, dom, n_boot=2000 if f == files[-1] else 200)
        v = val[step]
        # ham döküm koşunun logladığı eğrilerle AYNI mı (başka bir dosyayı okumadığımızın sağlaması)
        if abs(c["oracle"] - v["val_loss_oracle"]) > 1e-4 or abs(c["routed"] - v["val_loss"]) > 1e-4:
            raise ValueError(f"{f}: ham döküm metrics.jsonl ile uyuşmuyor ({c['oracle']} / {v['val_loss_oracle']})")
        c["oracle_null"] = v["val_loss_oracle_null"]
        c["structured_gain_over_null"] = v["val_loss_oracle_null"] - c["oracle"]
        steps.append({"step": step, "tokens": v["tokens"], **c})
    # SEYİR PUAN CİNSİNDEN: her eval noktasında koşunun kendi val eğrisinin o noktaya kadarki yerel eğimiyle (son üç
    # nokta). Nat cinsinden küçülen bir tavan, eğri düzleştiği için puan cinsinden küçülmüyor olabilir.
    ordered = [val[s] for s in sorted(val) if val[s]["tokens"] > 0]
    for st in steps:
        upto = [r for r in ordered if r["step"] <= st["step"]]
        if len(upto) >= 2:
            sl = local_slope(upto)
            st["own_local_slope_here"] = sl
            st["ceiling_points_own_slope"] = points(max(st["ceiling_vs_best_fixed"], 0.0), sl)
            st["structured_points_own_slope"] = points(max(st["structured_gain_over_null"], 0.0), sl)
    slope = local_slope([val[s] for s in sorted(val)])
    last = steps[-1]
    with np.load(files[-1]) as z:
        if "ce_tok" in z.files:
            last["token_level_raw"] = token_level_raw(z["ce_tok"], int(val[last["step"]].get("loss_prefix_skip", 0)))
    conv = {}
    for name, sl in (("lstar_local_-0.3346", SLOPE_LSTAR_LOCAL), ("late_secant_-0.2385", SLOPE_LATE_SECANT),
                     (f"own_local_{slope:.4f}", slope)):
        conv[name] = {k: points(max(last[k], 0.0), sl) for k in
                      ("ceiling_vs_best_fixed", "ceiling_vs_mean_fixed", "ceiling_vs_routed", "structured_gain_over_null",
                       "domain_gain_vs_best_fixed_in_sample")}
        conv[name]["ceiling_vs_best_fixed_ci95"] = [points(max(v, 0.0), sl) for v in last["ceiling_vs_best_fixed_ci95"]]
    return {"own_local_slope": slope, "final_step_points": conv, "steps": steps}


def best_module_agreement(best: dict[str, np.ndarray]) -> dict:
    """Aynı sabit val dizilerinde, FARKLI koşuların "en iyi modül"ü ne kadar örtüşüyor — koşu ADIYLA anahtarlı.

    Maske kaynaklı, koşudan koşuya TEKRARLANAN bir tercih varsa (ör. "bu dizi hep dilated'i sever") örtüşme, iki koşunun
    marjinal dağılımlarından beklenen şans düzeyinin üstünde olur. Şans düzeyindeyse dizi başına "en iyi modül" seçim
    gürültüsüdür — B11 null'ının okumasına dayanmayan, bağımsız bir sağlama.
    """
    names = sorted(best)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = best[a], best[b]
            if len(x) != len(y):
                raise ValueError(f"{a} / {b}: dizi sayıları farklı ({len(x)} / {len(y)})")
            m = int(max(x.max(), y.max())) + 1
            px, py = np.bincount(x, minlength=m) / len(x), np.bincount(y, minlength=m) / len(y)
            pairs.append({"a": a, "b": b, "agreement": float((x == y).mean()), "expected_if_independent": float(px @ py)})
    return {"n_pairs": len(pairs), "mean_agreement": float(np.mean([p["agreement"] for p in pairs])),
            "mean_expected_if_independent": float(np.mean([p["expected_if_independent"] for p in pairs])), "pairs": pairs}


def seed_of(run_name: str) -> str:
    return run_name.rsplit("-s", 1)[-1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(args.runs, f"{args.tag}-*")) if os.path.isdir(d)
                  and ".blok-iptal-" not in d and ".yarim-" not in d)
    res = {"what": "oracle tavanı (TEŞHİS; karar kuralı değil; mevcut tarama verisinden, yeni koşu yok)",
           "success_threshold_points": SUCCESS_THRESHOLD_POINTS, "oracle_abandon_threshold_points": ORACLE_ABANDON_POINTS,
           "slopes": {"lstar_local": SLOPE_LSTAR_LOCAL, "late_secant": SLOPE_LATE_SECANT},
           "scope": "rastgeleye yakın yönlendirmeyle eğitilmiş modüller, %50 bütçe; daha iyi bir ROUTER'ın bu modüllerle "
                    "kazanabileceğini sınırlar, birlikte eğitimle oluşabilecek ayrışmayı sınırlamaz",
           "runs": {}}
    for d in dirs:
        name = os.path.basename(d)
        res["runs"][name] = r = run_ceiling(d)
        last, p = r["steps"][-1], r["final_step_points"]["lstar_local_-0.3346"]
        print(f"{name:15s} adım {last['step']} | en iyi sabit {last['best_fixed']:.4f} (M{last['best_fixed_module']}) | oracle "
              f"{last['oracle']:.4f} | TAVAN {last['ceiling_vs_best_fixed']:.4f} nat = {p['ceiling_vs_best_fixed']:.2f} puan "
              f"[{p['ceiling_vs_best_fixed_ci95'][0]:.2f}, {p['ceiling_vs_best_fixed_ci95'][1]:.2f}] | null üstü "
              f"{last['structured_gain_over_null']:.4f} nat = {p['structured_gain_over_null']:.2f} puan | alan-oracle "
              f"{last['domain_gain_vs_best_fixed_in_sample']:.5f} nat (bölmeli {-last['domain_oracle_minus_best_fixed_split_half']:+.5f})")
    keys = ("ceiling_vs_best_fixed", "ceiling_vs_mean_fixed", "ceiling_vs_routed", "structured_gain_over_null",
            "domain_gain_vs_best_fixed_in_sample")
    summary = {}
    for conv in ("lstar_local_-0.3346", "late_secant_-0.2385"):
        summary[conv] = {}
        for k in keys:
            v = [r["final_step_points"][conv][k] for r in res["runs"].values()]
            summary[conv][k] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n_runs": len(v)}
    res["summary_points_final_step"] = summary
    # 9 koşu = 3 seed x 3 koşul: aynı seed'in koşuları init'i ve veri sırasını paylaşır -> BAĞIMSIZ tekrar sayısı seed sayısıdır
    by_seed: dict[str, list[float]] = {}
    for name, r in res["runs"].items():
        by_seed.setdefault(seed_of(name), []).append(r["final_step_points"]["lstar_local_-0.3346"]["ceiling_vs_best_fixed"])
    seed_means = {s: float(np.mean(v)) for s, v in sorted(by_seed.items())}
    res["ceiling_points_by_seed"] = {"means": seed_means, "n_independent": len(seed_means),
                                     "sd_of_seed_means": float(np.std(list(seed_means.values()), ddof=1)) if len(seed_means) > 1 else None,
                                     "note": "aynı seed'in koşuları init ve veri sırasını paylaşır; [min, maks] betimleyicidir, güven aralığı değildir"}
    traj = {}
    for i, st in enumerate(next(iter(res["runs"].values()))["steps"]):
        rows = [r["steps"][i] for r in res["runs"].values()]
        if any(x["step"] != st["step"] for x in rows):
            raise ValueError("koşuların eval adımları aynı değil")
        traj[str(st["step"])] = {k: float(np.mean([x[k] for x in rows])) for k in
                                 ("ceiling_vs_best_fixed", "structured_gain_over_null", "ceiling_points_own_slope",
                                  "structured_points_own_slope", "own_local_slope_here") if all(k in x for x in rows)}
    res["trajectory_mean_over_runs"] = traj
    best = {}
    for d in dirs:
        with np.load(sorted(glob.glob(os.path.join(d, "eval_raw", "step_*.npz")))[-1]) as z:
            best[os.path.basename(d)] = z["seq_masked"].argmin(1)
    agree = best_module_agreement(best)
    same = [p for p in agree["pairs"] if seed_of(p["a"]) == seed_of(p["b"])]
    diff = [p for p in agree["pairs"] if seed_of(p["a"]) != seed_of(p["b"])]
    for label, grp in (("same_seed", same), ("different_seed", diff)):
        agree[label] = {"n_pairs": len(grp), "mean_agreement": float(np.mean([p["agreement"] for p in grp])) if grp else None,
                        "mean_expected_if_independent": float(np.mean([p["expected_if_independent"] for p in grp])) if grp else None}
    res["cross_run_best_module_agreement_final_step"] = agree
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    s = summary["lstar_local_-0.3346"]["ceiling_vs_best_fixed"]
    print(f"TAVAN (9 koşu, L* yerel eğimi): ort {s['mean']:.2f} puan [{s['min']:.2f}, {s['max']:.2f}] | eşik +{SUCCESS_THRESHOLD_POINTS:.0f} -> {args.out}")


if __name__ == "__main__":
    main()
