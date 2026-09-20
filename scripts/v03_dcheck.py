"""v03 adım 1 — DENSE TEŞHİS KOŞUSUNUN ÖLÇÜTÜ (koşu görülmeden yazıldı; sahibi teşhis koşusunu onayladı).

    .venv/Scripts/python scripts/v03_dcheck.py --predict --out experiments/foton-v03-step1/dcheck_schedule.log   (koşudan ÖNCE)
    .venv/Scripts/python scripts/v03_dcheck.py --out experiments/foton-v03-step1/dcheck_result.json              (koşudan SONRA)

SORU: bugünkü kod, dondurulmuş dense referansını AYNI SEED'de yeniden üretiyor mu? Yani D-101'in referans ailesinden
sapması (pencere ortalaması +0.0253 nat) kodun mu, seed çekiminin mi?

TASARIM: eşli karşılaştırma. Yeni koşu `--preset foton-d --seed 11 --stop_step 2188`; referans `r-d-s11` (aynı seed, aynı
veri sırası, aynı başlangıç; tam bütçeli). Ölçü TEK NOKTA DEĞİL, EĞRİ BOYUNCA: D-101'i yakalayan şey adım 2000 değil,
750–1750 aralığıydı (orada z = +3.8 … +9.6 iken adım 2000'de sınırı yalnızca %1.1 aştı).

    d(s) = val(yeni D-s11, s) − val(r-d-s11, s),   ortak eval adımlarında
    PENCERE = (750, 1000, 1250, 1500, 1750)        D-101'i yakalayan aralık
    TÜM     = (250, 500, 750, 1000, 1250, 1500, 1750, 2000)

ÇAPALAR — hepsi koşudan ÖNCE, diskteki veriden ölçüldü; eşikler bunlardan TÜRETİLDİ:
  * AYNI KODLA tekrar (`foton-v02r/r-m-s11` ↔ `foton-v02r.kirli-src-iptal/r-m-s11`; İKİSİ DE src b645ec94, config'leri
    birebir aynı, seed 11): pencere ortalaması +0.001133, maks_TÜM |d| 0.007758 (adım 250), maks_PENCERE 0.005055,
    adım 0 farkı −8.3e-08. BU, gürültü zarfının ASIL ölçüsüdür: aynı kod, aynı seed, iki koşu.
  * FARKLI KOD sürümüyle tekrar (`r-d-s11` çifti; b645ec94 / d1b93ff0, dense kol): pencere ortalaması −0.000177,
    maks_TÜM 0.005801 (adım 500), adım 0 farkı TAM 0. İkinci, daha zayıf çapa.
  * D-101 ANOMALİSİ (5 seed'lik referans ortalamasına karşı): pencere ortalaması +0.025295, pencerede her adım pozitif.
  Ders: aynı kodla bile eğitim ortasında ±0.008 nat'a kadar ayrışma OLAĞANdır (kaotik büyüme) ve sonra kapanır; ama
  İŞARETLİ pencere ortalaması gürültüde ≤0.0012'de kalır. Ölçüt bu yüzden işaretli pencere ortalamasına dayanır.

KARAR — BEŞ hâl (veri görülmeden yazıldı, sonradan değiştirilmeyecek):
  KOD SAĞLAM          : |ort_PENCERE d| ≤ 0.004 VE maks_TÜM |d| ≤ 0.012
  KOD SUÇLU           : ort_PENCERE d ≥ +0.012648 VE pencerede HER adımda d > 0   (anomalinin EN AZ YARISI kodun)
  KOD SAPIYOR (kısmi) : +0.004 < ort_PENCERE d < +0.012648 VE pencerede HER adımda d > 0
  TERS YÖNDE SAPMA    : ort_PENCERE d < −0.004 VE pencerede HER adımda d < 0      (kod referanstan DAHA İYİ)
  OKUNMAZ             : hiçbiri (karışık işaretli sapma, ya da ortalama temiz ama tek noktada büyük sapma).
                        Bu hâlde basılan metin ÖLÇÜLEN sayılardan üretilir, sabit değildir.

Eşiklerin gerekçesi ve gücü (σ_pencere-ortalaması ≈ 0.0017; iki tekrar çiftinin adım-başı sd'si ~0.0035, beş adım, hafif
pozitif ilişki): 0.004 = aynı-kod çiftinde ölçülen |ort|'un (0.001133) 3.5 katı → kod özdeşken P(SAĞLAM) ≈ %97.
0.012 = aynı-kod çiftinde ölçülen maks_TÜM'ün (0.007758) 1.55 katı → tek noktadan yanlış alarm ≈ %0.5.
+0.012648 = D-101 anomalisinin YARISI. DÜRÜSTLÜK: bu eşik tam o değerde yazı-turadır (%50 güç); gerçek kod etkisi 0.025
iken güç ~%100, 0.008 iken KOD SAPIYOR (kısmi) hâli yakalar. Yani "yarısını açıklıyorsa suçlu" iddiası ancak eşiğin
belirgin üstünde geçerlidir; ara bölge için ayrı bir hâl (kısmi) vardır ve o da ayrı bir işe yönlendirir.

ÖN KOŞULLAR — tutmazsa TEŞHİS OKUNMAZ (hiçbir hâl açıklanmaz):
  P1 TAKVİM : ortak loglanan HER eğitim adımında lr BİREBİR aynı (kayan nokta eşitliği). Referans tam bütçeli, yeni koşu
              tarama bütçeli; takvim ayrışsaydı teşhis koddan değil takvimden sapardı ve yorumlanamazdı.
  P2 BAŞLANGIÇ : |d(0)| ≤ 1e-6 (aynı seed → bit-özdeş init; iki tekrar çiftinde ölçülen fark 0 ve −8.3e-08).
  P3 AYARLAR : config'ler yalnızca `run_name` / `out_dir` / `stop_step` / `budget_mult` ve varsayılan `foton_*` alanlarında
              ayrılır, ve seed iki tarafta da 11'dir.
  P4 BÜTÜNLÜK : iki koşu da bitmiş, ortak eval adımlarının hepsi var ve sonlu.

SONRAKİ ADIM (sahibin talimatı, koşu görülmeden): BEŞ hâlde de kalan 8 koşu koşulur; kuyruk kısaltılmaz. Her hâlde (b)
kapısı bu turda OKUNMAZ (sahibin kararı, 2026-09-20); tolerans gevşetilmez.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.config import PRESETS  # noqa: E402
from src.train import lr_at, token_budget  # noqa: E402

REF_DIR = os.path.join(ROOT, "checkpoints", "foton-v02r", "r-d-s11")
NEW_DIR = os.path.join(ROOT, "checkpoints", "foton-v03-dcheck", "dcheck-d-s11")
WINDOW = (750, 1000, 1250, 1500, 1750)
ALL_STEPS = (250, 500, 750, 1000, 1250, 1500, 1750, 2000)
END_STEP, SEED = 2188, 11
# Eşikler: aşağıdaki ÇAPALARDAN türetildi, koşu görülmeden.
SANE_WINDOW_MEAN = 0.004         # aynı-kod çiftinde ölçülen |ort| (0.001133) x 3.5
SANE_MAX_ABS = 0.012             # aynı-kod çiftinde ölçülen maks_TÜM (0.007758) x 1.55
GUILTY_WINDOW_MEAN = 0.012648    # D-101 pencere anomalisinin (0.025295) yarısı
INIT_TOL = 1e-6                  # ölçülen: 0.0 ve -8.3e-08
SAME_CODE_PAIR = {"runs": "foton-v02r/r-m-s11 ↔ foton-v02r.kirli-src-iptal/r-m-s11 (ikisi de src b645ec94)",
                  "window_mean": 0.001133, "max_abs_all": 0.007758, "max_abs_window": 0.005055, "init_diff": -8.3e-08,
                  "train_ce_window_mean": 0.000670}
OTHER_CODE_PAIR = {"runs": "foton-v02r/r-d-s11 ↔ foton-v02r.kirli-src-iptal/r-d-s11 (b645ec94 / d1b93ff0)",
                   "window_mean": -0.000177, "max_abs_all": 0.005801, "max_abs_window": 0.002575, "init_diff": 0.0,
                   "train_ce_window_mean": -0.001576}
D101_ANOMALY = {"window_mean": 0.025295, "all_positive": True}
CFG_MAY_DIFFER_TRAIN = {"run_name", "out_dir", "stop_step", "budget_mult"}
NEXT_STEP = {
    "kod_saglam": "KOD SAĞLAM: bugünkü kod dondurulmuş referansı aynı seed'de yeniden üretti; D-101'in sapması seed "
                  "çekimidir. Kayda geçer, kalan 8 koşu (sahibin onayıyla) koşulur. (b) bu turda yine OKUNMAZ.",
    "kod_suclu": "KOD SUÇLU: sapma aynı seed'de de tekrarlandı ve anomalinin en az yarısını açıklıyor; dense yolunda "
                 "sistematik bir fark var. Kalan 8 koşu yine koşulur ((a) okunabilir: X ailesi v02-X'e karşı doğrulandı), "
                 "dense farkı AYRI bir iş olarak açılır.",
    "kod_sapiyor_kismi": "KOD SAPIYOR (kısmi): sapma aynı seed'de de var ve D-101'le AYNI yönde, ama anomalinin yarısından "
                         "küçük; D-101'i tek başına açıklamıyor. Dense farkı AYRI bir iş olarak açılır (D gelecekteki her "
                         "karşılaştırmanın tabanıdır), D-101'in kalanı seed çekimidir. Kalan 8 koşu yine koşulur.",
    "ters_yonde_sapma": "TERS YÖNDE SAPMA: bugünkü kod referanstan sistematik olarak DAHA İYİ; D-101'in sapması kodla "
                        "AÇIKLANAMAZ (yön ters). Dense yolundaki fark yine AYRI bir iş olarak açılır. Kalan 8 koşu yine koşulur.",
    "okunmaz": "OKUNMAZ: ölçülen desen beş hâlin hiçbirine uymuyor; yorumlanmaz, sahibe sorulur. Kalan 8 koşuya devam "
               "kararı sahibindir.",
    "blocked": "TEŞHİS OKUNMAZ: ön koşullardan biri tutmadı (takvim / başlangıç / ayarlar / bütünlük). Önce o giderilir. "
               "Kalan 8 koşuya devam kararı sahibindir.",
}


def _curve(x, name: str, steps=ALL_STEPS) -> dict[int, float]:
    """Adım kimliğiyle anahtarlı eşleme; konuma göre eşleme kabul EDİLMEZ (bu depoda o hata dört kez doğdu)."""
    if not isinstance(x, Mapping):
        raise TypeError(f"{name}: adım kimliğiyle anahtarlı bir eşleme olmalı, verilen {type(x).__name__}")
    if not set(steps) <= set(x):
        raise ValueError(f"{name}: eksik adım(lar) {sorted(set(steps) - set(x))}")
    out = {}
    for s in steps:
        v = x[s]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name}[{s}] sonlu bir sayı olmalı, verilen {v!r}")
        out[s] = float(v)
    return out


def paired_diff(new, ref, steps=ALL_STEPS) -> dict[int, float]:
    """d(s) = yeni − referans, ADIM KİMLİĞİYLE eşli."""
    a, b = _curve(new, "yeni", steps), _curve(ref, "referans", steps)
    return {s: a[s] - b[s] for s in steps}


def verdict(d) -> dict:
    """BEŞ hâlli karar. TEK NOKTA DEĞİL: işaretli pencere ortalaması + tüm eğri üzerinde maksimum sapma."""
    diff = _curve(d, "d")
    win = [diff[s] for s in WINDOW]
    mean, max_abs = sum(win) / len(win), max(abs(v) for v in diff.values())
    all_pos, all_neg = all(v > 0 for v in win), all(v < 0 for v in win)
    if abs(mean) <= SANE_WINDOW_MEAN and max_abs <= SANE_MAX_ABS:
        outcome = "kod_saglam"
    elif all_pos and mean >= GUILTY_WINDOW_MEAN:
        outcome = "kod_suclu"
    elif all_pos and mean > SANE_WINDOW_MEAN:
        outcome = "kod_sapiyor_kismi"
    elif all_neg and mean < -SANE_WINDOW_MEAN:
        outcome = "ters_yonde_sapma"
    else:
        outcome = "okunmaz"
    text = NEXT_STEP[outcome]
    if outcome == "okunmaz":            # gerekçe SABİT DEĞİL, ölçülenden üretilir (yanlış açıklama basılmasın)
        why = (f"pencere ortalaması {mean:+.6f} nat (|.| {'>' if abs(mean) > SANE_WINDOW_MEAN else '≤'} {SANE_WINDOW_MEAN}), "
               f"maks_TÜM |d| {max_abs:.6f} ({'>' if max_abs > SANE_MAX_ABS else '≤'} {SANE_MAX_ABS}), "
               f"pencerede hepsi pozitif: {all_pos}, hepsi negatif: {all_neg}")
        text = f"{text} Ölçülen: {why}."
    return {"per_step_nat": diff, "window_mean_nat": mean, "max_abs_nat": max_abs,
            "max_abs_window_nat": max(abs(v) for v in win), "window_all_positive": all_pos, "window_all_negative": all_neg,
            "thresholds": {"sane_window_mean": SANE_WINDOW_MEAN, "sane_max_abs": SANE_MAX_ABS,
                           "guilty_window_mean": GUILTY_WINDOW_MEAN},
            "anchors": {"same_code_pair": SAME_CODE_PAIR, "other_code_pair": OTHER_CODE_PAIR, "d101_anomaly": D101_ANOMALY},
            "outcome": outcome, "next_step": text}


def rows(run_dir: str, kind: str) -> list[dict]:
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        raise ValueError(f"{run_dir}: metrics.jsonl yok")
    with open(path, encoding="utf-8") as f:
        return [r for r in (json.loads(line) for line in f if line.strip()) if r.get("type") == kind]


def read_curves(run_dir: str) -> dict:
    if not any(r.get("step") for r in rows(run_dir, "end")) and not os.path.exists(os.path.join(run_dir, "DONE")):
        raise ValueError(f"{run_dir}: koşu bitmemiş")
    val = {r["step"]: r["val_loss"] for r in rows(run_dir, "val")}
    train = {r["step"]: r for r in rows(run_dir, "train")}
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return {"val": val, "lr": {s: r["lr"] for s, r in train.items()}, "ce": {s: r["ce"] for s, r in train.items()}, "config": cfg}


def schedule_identical(new_lr: Mapping, ref_lr: Mapping) -> dict:
    """P1: ortak loglanan HER eğitim adımında lr BİREBİR aynı (== ; yuvarlama toleransı YOK)."""
    common = sorted(set(new_lr) & set(ref_lr))
    bad = [s for s in common if new_lr[s] != ref_lr[s]]
    return {"common_steps": len(common), "mismatched_steps": bad, "identical": bool(common and not bad)}


def predicted_schedule(preset: str = "foton-d", stop_step: int = END_STEP) -> dict[int, float]:
    """Yeni koşunun LOGLAYACAĞI lr — koddan, koşmadan. Bütçe, eğitimin KENDİ yardımcısıyla (`token_budget`) türetilir:
    `stop_step` takvime GİRMEZ. Loglanan lr, o adımdan ÖNCE görülen token'la hesaplanır (adım S → (S−1)·tps)."""
    mcfg, tcfg = PRESETS[preset]
    tps = tcfg.batch_size * tcfg.grad_accum * mcfg.max_seq_len
    warm, mx = token_budget(tcfg, tps)
    return {s: lr_at((s - 1) * tps, tps, warm, mx, tcfg) for s in range(tcfg.log_interval, stop_step + 1, tcfg.log_interval)}


# Ulaşılabilirlik benzetiminin gürültü ölçeği: İKİ tekrar çiftinin 16 adım-başı farkının kök-ortalama-karesi.
STEP_SD = 0.0035235


def reachability(sims: int = 40_000, seed: int = 20260920, rho: float = 0.0) -> dict:
    """Her hâl ULAŞILABİLİR mi ve kod özdeşken yanlış alarm ne kadar — benzetim (çare yönü sayıyla ölçülür).
    Model: d(s) = mu + gürültü (adım-başı sd 0.003528, iki tekrar çiftinden); `rho` > 0 komşu adımlar arası ilişki."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = {}
    for mu in (0.0, 0.004, 0.008, GUILTY_WINDOW_MEAN, 0.020, D101_ANOMALY["window_mean"], -0.008, -0.025295):
        z = rng.normal(0.0, 1.0, (sims, len(ALL_STEPS)))
        if rho:
            z[:, 1:] = rho * z[:, :-1] + math.sqrt(1 - rho ** 2) * z[:, 1:]
        draws = mu + STEP_SD * z
        counts: dict[str, int] = {}
        for row in draws:
            o = verdict(dict(zip(ALL_STEPS, (float(v) for v in row))))["outcome"]
            counts[o] = counts.get(o, 0) + 1
        out[round(mu, 6)] = {k: v / sims for k, v in sorted(counts.items())}
    return out


def config_ok(new_cfg: dict, ref_cfg: dict) -> dict:
    m = {k for k in set(new_cfg["model"]) | set(ref_cfg["model"])
         if new_cfg["model"].get(k, "<yok>") != ref_cfg["model"].get(k, "<yok>") and not k.startswith("foton")}
    t = {k for k in set(new_cfg["train"]) | set(ref_cfg["train"])
         if new_cfg["train"].get(k, "<yok>") != ref_cfg["train"].get(k, "<yok>")} - CFG_MAY_DIFFER_TRAIN
    seed_ok = new_cfg["train"].get("seed") == ref_cfg["train"].get("seed") == SEED
    foton_off = not new_cfg["model"].get("foton", "") and new_cfg["model"].get("n_experts", 0) == 0
    return {"model_unexpected": sorted(m), "train_unexpected": sorted(t), "same_seed": seed_ok, "dense_arm": foton_off,
            "ok": bool(not m and not t and seed_ok and foton_off)}


def diagnose(new_dir: str = NEW_DIR, ref_dir: str = REF_DIR) -> dict:
    new, ref = read_curves(new_dir), read_curves(ref_dir)
    sched = schedule_identical(new["lr"], ref["lr"])
    cfg = config_ok(new["config"], ref["config"])
    init = {"diff": new["val"][0] - ref["val"][0], "tol": INIT_TOL} if 0 in new["val"] and 0 in ref["val"] else None
    pre = {"P1_schedule": sched, "P2_init": init, "P3_config": cfg,
           "clean": bool(sched["identical"] and cfg["ok"] and init is not None and abs(init["diff"]) <= INIT_TOL)}
    d = paired_diff(new["val"], ref["val"])
    ce_steps = tuple(s for s in WINDOW if s in new["ce"] and s in ref["ce"])
    ce = paired_diff(new["ce"], ref["ce"], steps=ce_steps) if ce_steps else {}
    v = verdict(d)
    if not pre["clean"]:
        v = {**v, "outcome": "blocked", "next_step": NEXT_STEP["blocked"]}
    return {"what": "v03 adım 1 — dense teşhis koşusu (ölçüt koşu görülmeden yazıldı: scripts/v03_dcheck.py)",
            "new": os.path.relpath(new_dir, ROOT), "ref": os.path.relpath(ref_dir, ROOT), "preconditions": pre,
            "train_ce_window_diff_NOT_binding": {"steps": list(ce_steps), "per_step": ce,
                                                 "mean": sum(ce.values()) / len(ce) if ce else None,
                                                 "anchor_same_code_pair": SAME_CODE_PAIR["train_ce_window_mean"]},
            **v}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true", help="koşudan ÖNCE: LR takvimi eşleşmesini koddan doğrula")
    ap.add_argument("--reach", action="store_true", help="koşudan ÖNCE: beş hâlin ulaşılabilirliği (benzetim)")
    ap.add_argument("--new", default=NEW_DIR)
    ap.add_argument("--ref", default=REF_DIR)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.reach:
        lines = [f"Ulaşılabilirlik — beş hâl (benzetim; adım-başı gürültü sd {STEP_SD}, iki tekrar çiftinden; {40_000} çekim/hücre)",
                 f"eşikler: sağlam |ort| ≤ {SANE_WINDOW_MEAN} ve maks_TÜM ≤ {SANE_MAX_ABS} | suçlu ort ≥ {GUILTY_WINDOW_MEAN}", ""]
        checks = []
        for rho in (0.0, 0.3):
            tab = reachability(rho=rho)
            lines.append(f"komşu adım ilişkisi rho = {rho}")
            for mu, dist in tab.items():
                lines.append(f"  gerçek etki {mu:+.6f} -> " + ", ".join(f"{k} {v:.3f}" for k, v in dist.items()))
            g = lambda mu, k: tab[round(mu, 6)].get(k, 0.0)                                      # noqa: E731
            checks += [(f"rho={rho}: etki 0 -> P(SAĞLAM) >= %90", g(0.0, "kod_saglam") >= 0.90),
                       (f"rho={rho}: etki 0 -> P(SUÇLU) <= %0.1 ve P(KISMİ) <= %5",
                        g(0.0, "kod_suclu") <= 0.001 and g(0.0, "kod_sapiyor_kismi") <= 0.05),
                       (f"rho={rho}: etki = anomalinin tamamı -> P(SUÇLU) >= %95", g(D101_ANOMALY["window_mean"], "kod_suclu") >= 0.95),
                       (f"rho={rho}: etki +0.008 -> P(SUÇLU ya da KISMİ) >= %80",
                        g(0.008, "kod_suclu") + g(0.008, "kod_sapiyor_kismi") >= 0.80),
                       (f"rho={rho}: etki -0.025295 -> P(TERS YÖNDE) >= %95", g(-0.025295, "ters_yonde_sapma") >= 0.95),
                       (f"rho={rho}: etki tam suçlu eşiğinde -> P(SUÇLU) %40-60 (eşikte yazı-tura, dürüstlük notu)",
                        0.40 <= g(GUILTY_WINDOW_MEAN, "kod_suclu") <= 0.60)]
            lines.append("")
        lines += [f"{'GEÇTİ' if ok else '*** KALDI ***'}  {name}" for name, ok in checks]
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return 0 if all(ok for _, ok in checks) else 1
    if args.predict:
        ref = read_curves(args.ref)
        pred = predicted_schedule()
        sched = schedule_identical(pred, ref["lr"])
        lines = [f"Yeni koşunun LOGLAYACAĞI lr (koddan, preset foton-d, --stop_step {END_STEP}) ile referans {os.path.basename(args.ref)}"
                 f" (tam bütçe) karşılaştırması", f"ortak loglanan adım: {sched['common_steps']} | birebir aynı: {sched['identical']}"
                 f" | ayrışan adım: {sched['mismatched_steps']}",
                 f"örnek: adım 25 lr {pred[25]:.17g} | adım 2000 lr {pred[2000]:.17g} | adım {max(pred)} lr {pred[max(pred)]:.17g}",
                 "not: bütçe eğitimin kendi `token_budget()` yardımcısından; `stop_step` takvime girmez (takvim max_steps = 4375 üzerinden)."]
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return 0 if sched["identical"] else 1
    res = diagnose(args.new, args.ref)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "anchors"}, indent=1, ensure_ascii=False))
    print("SONUÇ:", res["outcome"].upper(), "|", res["next_step"])
    return 0 if res["outcome"] != "blocked" else 2


if __name__ == "__main__":
    sys.exit(main())
