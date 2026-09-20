"""v03 ADIM 1 analizi — karar kuralı VERİ GÖRÜLMEDEN yazıldı (experiments/foton-v03-step1/decision_rule.md).

    .venv/Scripts/python scripts/v03_step1_analyze.py --out experiments/foton-v03-step1/result.json

12 koşu (X-dom, H-dom, X-iid, D × seed 101/102/103; `checkpoints/foton-v03-step1/v03s1-<kol>-s<seed>`) + aynı seed'li v02-X
(`checkpoints/foton-v02-tarama/v7-x-s<seed>`). Kuralın kendisi `scripts/v03_step1_decision.py`'dedir; burası yalnızca
ölçüleri koşulardan ÇIKARIR ve o fonksiyonlara seed-anahtarlı verir. Aşağıdakilerden biri tutmazsa KARAR ÜRETİLMEZ:

  KİMLİK   bir koşunun kim olduğu KLASÖR ADINDAN değil, koşunun kendi yazdığı kayıtlardan okunur: `config.json` (seed, kol,
           atama kuralı, bütçe), meta satırı (kol harfi, maske harfleri, parametre + FLOP sayıları, bitiş adımı) ve
           `provenance.json`'daki komut satırı. Klasör adı yalnızca bunlarla UYUŞMAK zorundadır. Eğitim ayarları aynı
           seed'li v02-X'inkiyle (run_name / out_dir dışında) BİREBİR, model ayarları yalnızca `foton` / `foton_assign`'da farklı.
  EŞLEME   "aynı seed = aynı veri sırası, aynı val kümesi, aynı başlangıç ağırlıkları" iddiası veriden doğrulanır: log aralığı
           başına alan sayımları (D dahil bütün kollar + v02-X), val pencerelerinin içerik özeti (`val_windows_sha256`; 12 koşunun
           her eval noktasında TEK değer — D'nin val kümesi yalnızca buradan doğrulanır, D ham döküm yazmaz), val `domain`
           dizisi ve adım 0'daki ham val kayıpları (FOTON kollarını v02-X'e bağlar; v02 koşularında özet alanı yoktur).
  ATAMA    X-dom / H-dom'da alan x modül tablosunun köşegen DIŞI sıfır; X-iid'de her alanın modül payları 1/3 civarı.
  İKİ YOL  `val_atanan` iki ayrı yoldan 1e-4 içinde (NaN "uyuşuyor" sayılmaz): koşunun eval yolu (`val_loss`, eğitimin forward
           yolu) ile `eval_raw` ham dökümünden sonradan hesap (`seq_masked[b, atama(b)]`; X-iid'de üç modülün ortalaması).
  KOD      bütün karar koşularında `code_dirty = False` ve TEK src özeti, istisnasız.

Router'ın (eğitilmemiş) seçimiyle hesaplanan `val_loss_router_UNUSED` hiçbir tabloya kol kaybı olarak GİRMEZ.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v03_step1_decision as dec  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oc = _load("oracle_ceiling")
sc = _load("screening_context")
rd = _load("router_diag")

ARMS = {"d": None, "x-dom": "domain", "h-dom": "domain", "x-iid": "iid"}        # kol -> beklenen val_assign_mode
# Kurulduğu hâliyle sayılar: docs/ARCHITECTURE_AS_BUILT.md (testle kilitli). Zorlanmış kollarda router modelde DURUR; FLOP aynı loglanır.
IDENTITY = {
    "d": {"preset": "foton-d", "foton": "", "foton_assign": "", "meta_arm": "d", "masks": None,
          "params_total": 42_476_032, "flops_train_per_token": 267_411_456},
    "x-dom": {"preset": "foton-x-dom", "foton": "x", "foton_assign": "domain", "meta_arm": "x", "masks": ["A", "B", "C"],
              "params_total": 55_326_720, "flops_train_per_token": 265_838_610},
    "h-dom": {"preset": "foton-h-dom", "foton": "h", "foton_assign": "domain", "meta_arm": "h", "masks": ["A", "A", "A"],
              "params_total": 55_326_720, "flops_train_per_token": 265_834_602},
    "x-iid": {"preset": "foton-x-iid", "foton": "x", "foton_assign": "iid", "meta_arm": "x", "masks": ["A", "B", "C"],
              "params_total": 55_326_720, "flops_train_per_token": 265_838_610},
}
END_STEP, MAX_STEPS, TOKENS_PER_STEP, SEQS_PER_STEP = 2188, 4375, 32768, 64
VAL_STEPS = tuple(range(0, 2001, 250)) + (END_STEP,)
VAL_DOMAIN_COUNTS = [431, 434, 415]                  # dondurulmuş val kümesi: 1.280 pencere (code / math / nl)
# Aynı 1.280 pencerenin TOKEN içeriğinin özeti (16'lık batch'lerle, `train.evaluate`'in hesapladığı biçimde). Koşulardan ÖNCE
# kaydedildi: `scripts/v03_step1_gpu_two_route.py` bu pencerelerle v02-X koşusunun logladığı oracle'ı 1e-4 içinde yeniden üretir,
# yani bu, v02'nin ölçüldüğü val kümesidir. D'yi (ham döküm yazmaz) v02'nin val kümesine bağlayan TEK veri budur.
VAL_WINDOWS_SHA256 = "9773961275510800cc4345523184278f7601f5d59d005174143906894888845e"
TWO_ROUTE_TOL = 1e-4
INIT_FINGERPRINT_TOL = 1e-3                          # aynı seed'li v02 koşularında ölçülen fark 0.0; başka seed'de ortalama 0.06
IID_SHARE_TOL = 0.01                                 # alan başına ~46.700 çekim: payın sd'si 0.0022 -> 4.5 sd
MODEL_KEYS_MAY_DIFFER = {"foton", "foton_assign"}
TRAIN_KEYS_MAY_DIFFER = {"run_name", "out_dir"}


def _fail(run_dir: str, what: str):
    raise ValueError(f"{run_dir}: {what}")


def _rows(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        _fail(run_dir, "metrics.jsonl yok")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json(run_dir: str, name: str) -> dict:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        _fail(run_dir, f"{name} yok")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _val_rows(run_dir: str, rows: list[dict], need_digest: bool = True) -> dict[int, dict]:
    """Bitmiş, tam bütçeli, prefix maskeli, her eval noktası SONLU bir val eğrisi — yoksa ValueError. Karar koşularında her
    eval noktası val pencerelerinin içerik özetini de taşımalı ve koşu boyunca TEK özet olmalı (v02 koşularında alan yok)."""
    ends = [r for r in rows if r.get("type") == "end"]
    if len(ends) != 1:
        _fail(run_dir, "koşu bitmemiş ('end' kaydı yok ya da birden fazla)")
    if ends[0].get("step") != END_STEP or ends[0].get("tokens") != END_STEP * TOKENS_PER_STEP:
        _fail(run_dir, f"bitiş adımı / token'ı {ends[0].get('step')} / {ends[0].get('tokens')}, beklenen {END_STEP}")
    val = {r["step"]: r for r in rows if r.get("type") == "val"}
    if tuple(sorted(val)) != VAL_STEPS:
        _fail(run_dir, f"eval adımları {sorted(val)}, beklenen {list(VAL_STEPS)} (son eval adımı {END_STEP} olmalı)")
    if {r.get("loss_prefix_skip") for r in val.values()} != {64}:
        _fail(run_dir, "ölçüt prefix maskeli değil (loss_prefix_skip != 64)")
    bad = [s for s, r in val.items() if not _finite(r.get("val_loss"))]
    if bad:
        _fail(run_dir, f"val_loss sonlu değil (adım {bad})")
    digests = {r.get("val_windows_sha256") for r in val.values()}
    if need_digest and (len(digests) != 1 or not all(isinstance(d, str) and len(d) == 64 for d in digests)):
        _fail(run_dir, f"val pencerelerinin özeti (val_windows_sha256) yok ya da koşu içinde değişiyor: {sorted(map(str, digests))}")
    return val


def _config_diff(a: dict, b: dict) -> set[str]:
    return {k for k in set(a) | set(b) if a.get(k, "") != b.get(k, "")}


def check_identity(run_dir: str, arm: str, seed: int, tag: str, rows: list[dict], ref_cfg: dict) -> dict:
    """Koşu GERÇEKTEN (kol, seed) mi — koşunun kendi kayıtlarından. `ref_cfg` = aynı seed'li v02-X'in config.json'u."""
    want, name = IDENTITY[arm], f"{tag}-{arm}-s{seed}"
    if os.path.basename(os.path.normpath(run_dir)) != name:
        _fail(run_dir, f"klasör adı {name!r} değil")
    cfg, prov = _json(run_dir, "config.json"), _json(run_dir, "provenance.json")
    m, t = cfg["model"], cfg["train"]
    got = {"run_name": t.get("run_name"), "seed": t.get("seed"), "foton": m.get("foton"), "foton_assign": m.get("foton_assign", ""),
           "n_experts": m.get("n_experts"), "resume": t.get("resume")}
    exp = {"run_name": name, "seed": seed, "foton": want["foton"], "foton_assign": want["foton_assign"], "n_experts": 0, "resume": False}
    if got != exp:
        _fail(run_dir, f"koşu kimliği uyuşmuyor: config.json {got}, beklenen {exp}")
    d_model, d_train = _config_diff(m, ref_cfg["model"]), _config_diff(t, ref_cfg["train"])
    if not d_model <= MODEL_KEYS_MAY_DIFFER or not d_train <= TRAIN_KEYS_MAY_DIFFER:
        _fail(run_dir, f"ayarlar aynı seed'li v02-X'ten izinli alanlar dışında ayrılıyor: model {sorted(d_model - MODEL_KEYS_MAY_DIFFER)}, "
                       f"eğitim {sorted(d_train - TRAIN_KEYS_MAY_DIFFER)}")
    args = ["--preset", want["preset"], "--seed", str(seed), "--stop_step", str(END_STEP)]
    if prov.get("args") != args:
        _fail(run_dir, f"komut satırı {prov.get('args')}, beklenen {args}")
    metas = [r for r in rows if r.get("type") == "meta"]
    if len(metas) != 1:
        _fail(run_dir, "meta satırı yok ya da birden fazla")
    meta = metas[0]
    masks = None if "foton_connections_counted" not in meta else [k.split(":")[1] for k in sorted(meta["foton_connections_counted"])]
    got_meta = {"arm": meta.get("arm"), "end_step": meta.get("end_step"), "max_steps": meta.get("max_steps"),
                "tokens_per_step": meta.get("tokens_per_step"), "loss_prefix_skip": meta.get("loss_prefix_skip"),
                "params_total": meta.get("params_total"), "flops_train_per_token": meta.get("flops_train_per_token"),
                "masks": masks, "foton_assign": meta.get("foton_assign", "")}
    exp_meta = {"arm": want["meta_arm"], "end_step": END_STEP, "max_steps": MAX_STEPS, "tokens_per_step": TOKENS_PER_STEP,
                "loss_prefix_skip": 64, "params_total": want["params_total"], "flops_train_per_token": want["flops_train_per_token"],
                "masks": want["masks"], "foton_assign": want["foton_assign"]}
    if got_meta != exp_meta:
        _fail(run_dir, f"meta satırı uyuşmuyor: {got_meta}, beklenen {exp_meta}")
    return {k: prov.get(k) for k in ("git_head", "src_sha256", "code_dirty", "session")}


def _data_order(run_dir: str, rows: list[dict], field: str = "data_domain_counts") -> list[list[int]]:
    """Veri sırasının parmak izi: log aralığı başına alan sayımları. v02-X'te aynı bilgi `router_domain_counts`'un satır toplamı."""
    train = [r for r in rows if r.get("type") == "train"]
    if not train or any(r.get(field) is None for r in train):
        _fail(run_dir, f"eğitim satırlarında {field} yok (veri sırası doğrulanamaz)")
    out = [[int(v) for v in (r[field] if field == "data_domain_counts" else np.asarray(r[field]).sum(axis=1))] for r in train]
    if sum(map(sum, out)) != END_STEP * SEQS_PER_STEP:
        _fail(run_dir, f"{field} toplamı {sum(map(sum, out))}, beklenen {END_STEP * SEQS_PER_STEP} dizi")
    return out


def _raw(run_dir: str, step: int) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(run_dir, "eval_raw", f"step_{step:05d}.npz")
    if not os.path.exists(path):
        _fail(run_dir, f"eval_raw/step_{step:05d}.npz yok")
    with np.load(path) as z:
        if int(z["step"]) != step:
            _fail(run_dir, f"ham döküm başka adımın ({int(z['step'])} != {step})")
        sm, dom = z["seq_masked"].astype(np.float64), z["domain"].astype(np.int64)
    if np.bincount(dom, minlength=3).tolist() != VAL_DOMAIN_COUNTS or sm.shape != (len(dom), 3) or not np.isfinite(sm).all():
        _fail(run_dir, f"adım {step} ham dökümü dondurulmuş val kümesi değil ya da sonlu değil")
    return sm, dom


def check_assignment_table(run_dir: str, rows: list[dict], assign_mode: str, data_order: list[list[int]]) -> dict:
    """Eğitimde UYGULANAN atama kuralı, koşunun logladığı alan x modül tablosundan."""
    train = [r for r in rows if r.get("type") == "train"]
    tables = [np.asarray(r["assign_domain_counts"], dtype=np.int64) for r in train]
    if [t.sum(axis=1).tolist() for t in tables] != data_order:
        _fail(run_dir, "atama tablosunun alan toplamları veri sırasının parmak iziyle uyuşmuyor")
    if any(r.get(f) is not None for r in train for f in ("router_counts", "router_entropy_marginal")):
        _fail(run_dir, "zorlanmış kolda router metriği basılmış (UYGULANAMAZ olmalıydı)")
    total = np.sum(tables, axis=0)
    if assign_mode == "domain":
        if total.sum() - np.trace(total) != 0:
            _fail(run_dir, f"alan ataması köşegen dışına taşmış: {total.tolist()}")
    else:
        share = total / total.sum(axis=1, keepdims=True)
        if not (np.abs(share - 1.0 / 3.0) <= IID_SHARE_TOL).all():
            _fail(run_dir, f"iid atama payları 1/3'ten {IID_SHARE_TOL}'den fazla sapıyor: {share.round(4).tolist()}")
    return {"assign_table": total.tolist(),
            "steps_with_gradless_module": int(sum(r["assign_steps_gradless_module"] for r in train)),
            "microbatches_missing_module": int(sum(r["assign_microbatches_missing_module"] for r in train))}


def read_run(run_dir: str, arm: str, seed: int, tag: str, ref_cfg: dict) -> dict:
    """Bir karar koşusunun ölçüleri. Bitmemiş / başka bütçeli / başka kimlikli / yanlış kurallı koşu -> ValueError."""
    rows, assign_mode = _rows(run_dir), ARMS[arm]
    val = _val_rows(run_dir, rows)
    out = {"dir": run_dir, "val_curve": {s: r["val_loss"] for s, r in val.items()}, "val": val[END_STEP]["val_loss"],
           "provenance": check_identity(run_dir, arm, seed, tag, rows, ref_cfg), "data_order": _data_order(run_dir, rows),
           "val_digest": val[END_STEP]["val_windows_sha256"]}
    modes = {r.get("val_assign_mode") for r in val.values()}
    if modes != {assign_mode}:
        _fail(run_dir, f"val_assign_mode {modes}, beklenen {assign_mode!r}")
    if assign_mode is None:
        return out
    worst = 0.0
    for step, r in val.items():                           # İKİ YOL, her eval noktasında
        sm, dom = _raw(run_dir, step)
        posthoc = sm[np.arange(len(dom)), dom].mean() if assign_mode == "domain" else sm.mean()
        check = r.get("val_loss_assigned_check")
        diffs = [abs(r["val_loss"] - posthoc), abs(r["val_loss"] - check) if _finite(check) else float("nan")]
        if not all(math.isfinite(float(x)) for x in diffs):     # NaN, `max` ve `>` karşılaştırmalarından sessizce geçerdi
            _fail(run_dir, f"atanan val loss iki yoldan uyuşmuyor (adım {step}: sağlama sonlu değil)")
        worst = max(worst, *map(float, diffs))
    if not worst <= TWO_ROUTE_TOL:
        _fail(run_dir, f"atanan val loss iki yoldan uyuşmuyor (en büyük fark {worst:.2e} > {TWO_ROUTE_TOL})")
    out["two_route_max_diff"] = worst
    out["val_router_UNUSED"] = val[END_STEP]["val_loss_router_UNUSED"]
    out.update(module_measures(run_dir, val))
    out.update(check_assignment_table(run_dir, rows, assign_mode, out["data_order"]))
    out["init_fingerprint"], out["val_domains"] = _raw(run_dir, 0)
    return out


def read_v02(run_dir: str, seed: int) -> dict:
    """Aynı seed'li v02-X referansı (ön koşulun karşı tarafı). Karar koşularıyla AYNI sıkılıkta denetlenir."""
    rows = _rows(run_dir)
    val = _val_rows(run_dir, rows, need_digest=False)
    cfg, prov = _json(run_dir, "config.json"), _json(run_dir, "provenance.json")
    m, t = cfg["model"], cfg["train"]
    got = {"seed": t.get("seed"), "foton": m.get("foton"), "foton_assign": m.get("foton_assign", ""), "foton_reg": m.get("foton_reg"),
           "n_experts": m.get("n_experts"), "layout": [m.get("foton_pre"), m.get("foton_routed"), m.get("foton_post")],
           "stop_step": t.get("stop_step"), "max_steps": t.get("max_steps"), "seqs_per_step": t.get("batch_size", 0) * t.get("grad_accum", 0),
           "eval_iters": t.get("eval_iters"), "loss_prefix_skip": t.get("loss_prefix_skip"), "resume": t.get("resume"),
           "code_dirty": prov.get("code_dirty")}
    exp = {"seed": seed, "foton": "x", "foton_assign": "", "foton_reg": "none", "n_experts": 0, "layout": [3, 2, 3],
           "stop_step": END_STEP, "max_steps": MAX_STEPS, "seqs_per_step": SEQS_PER_STEP, "eval_iters": 80, "loss_prefix_skip": 64,
           "resume": False, "code_dirty": False}
    if got != exp:
        _fail(run_dir, f"v02-X referansı beklenen koşu değil: {got}, beklenen {exp}")
    if any("val_loss_routed_check" not in r or "val_assign_mode" in r for r in val.values()):
        _fail(run_dir, "v02-X referansı öğrenilmiş-router'lı bir koşu değil")
    out = {"dir": run_dir, "config": cfg, "val": val[END_STEP]["val_loss"], "src_sha256": str(prov.get("src_sha256"))[:12],
           "data_order": _data_order(run_dir, rows, "router_domain_counts")}
    out.update(module_measures(run_dir, val))
    out["init_fingerprint"], out["val_domains"] = _raw(run_dir, 0)
    return out


def module_measures(run_dir: str, val: dict[int, dict]) -> dict:
    """SON eval noktasında (adım 2188, açıkça): modüller arası fark, oracle tavanı, null'ı aşan kısım, I(alan; en iyi modül)."""
    sm, dom = _raw(run_dir, END_STEP)
    with np.load(os.path.join(run_dir, "eval_raw", f"step_{END_STEP:05d}.npz")) as z:
        # yalnızca NOKTA değerler kullanılır (fixed / best / mean / oracle / tavan): v02'nin 1.34 puanıyla AYNI fonksiyon;
        # ikiye-bölme ve bootstrap (n_split, n_boot) bu değerleri etkilemez, burada en küçük hâlleriyle çağrılır
        c = oc.ceiling(z["seq_masked"], z["choice"], z["domain"], n_split=1, n_boot=1)
    row = val[END_STEP]
    if not all(_finite(row.get(k)) for k in ("val_loss_oracle", "val_loss_oracle_null")) or abs(c["oracle"] - row["val_loss_oracle"]) > 1e-4:
        _fail(run_dir, "ham döküm metrics.jsonl ile uyuşmuyor (oracle)")
    over_null = row["val_loss_oracle_null"] - c["oracle"]
    best = sm.argmin(1)
    return {"mean_fixed": c["mean_fixed"], "best_fixed": c["best_fixed"], "fixed": c["fixed"],
            "spread_nat": float((sm.max(1) - sm.min(1)).mean()),
            "ceiling_nat": c["ceiling_vs_best_fixed"], "ceiling_points": dec.points_of_nat(c["ceiling_vs_best_fixed"]),
            "over_null_nat": over_null, "over_null_points": dec.points_of_nat(over_null),
            "mi_domain_best_bits": rd.mutual_information_bits(dom, best, 3, sm.shape[1]),
            "best_is_domain_share": float((best == dom).mean())}


def check_pairing(runs: dict, v02: dict) -> dict:
    """Eşli tasarımın dayandığı üç iddia, VERİDEN (seed başına): aynı veri sırası (D dahil), aynı val kümesi, aynı başlangıç."""
    worst = 0.0
    digests = {runs[arm][s]["val_digest"] for arm in ARMS for s in dec.SEEDS}
    if len(digests) != 1:                                  # D dahil 12 koşu: val pencerelerinin TOKEN içeriği tek ve aynı
        raise ValueError(f"12 koşu aynı val pencerelerinde ölçülmemiş (val_windows_sha256 {len(digests)} ayrı değer) — eşleme yok")
    if digests != {VAL_WINDOWS_SHA256}:                    # kendi içinde tutarlı ama BAŞKA bir val kümesi (ör. val dosyası yeniden üretilmiş)
        raise ValueError(f"12 koşunun val kümesi koşulardan önce kaydedilen (v02'nin ölçüldüğü) val kümesi değil: {sorted(digests)}")
    for s in dec.SEEDS:
        order = v02[s]["data_order"]
        for arm in ARMS:
            if runs[arm][s]["data_order"] != order:
                raise ValueError(f"seed {s}: {arm} kolunun veri sırası aynı seed'li v02-X'inkiyle aynı değil — eşleme yok")
        for arm in ("x-dom", "h-dom", "x-iid"):
            if not np.array_equal(runs[arm][s]["val_domains"], v02[s]["val_domains"]):
                raise ValueError(f"seed {s}: {arm} kolunun val kümesi v02-X'inkiyle aynı değil")
        ref = v02[s]["init_fingerprint"]
        for arm, cols in (("x-dom", slice(None)), ("x-iid", slice(None)), ("h-dom", slice(0, 1))):     # H: yalnız modül 0 (maske A) ortak
            diff = float(np.abs(runs[arm][s]["init_fingerprint"][:, cols] - ref[:, cols]).max())
            worst = max(worst, diff)
            if not diff <= INIT_FINGERPRINT_TOL:
                raise ValueError(f"seed {s}: {arm} adım 0'da v02-X ile aynı ağırlıklardan başlamamış (en büyük fark {diff:.3g})")
    for a, b in ((101, 102), (101, 103), (102, 103)):                                # seed'ler GERÇEKTEN farklı koşular mı
        same_init = float(np.abs(v02[a]["init_fingerprint"] - v02[b]["init_fingerprint"]).max()) <= INIT_FINGERPRINT_TOL
        if v02[a]["data_order"] == v02[b]["data_order"] or same_init:
            raise ValueError(f"seed {a} ve {b} aynı veri sırasını / başlangıcı taşıyor — koşular birbirinin kopyası")
    return {"same_data_order_all_arms_incl_D": True, "same_val_windows_all_12_runs_incl_D": True, "val_windows_sha256": digests.pop(),
            "same_val_domains_and_init_as_v02_x_FOTON_arms_only": True, "init_fingerprint_max_diff": worst,
            "note": "D, FOTON kollarıyla veri sırasında (alan sayımları) ve val kümesinde (pencere içerik özeti) eşli; başlangıç "
                    "ağırlıklarında DEĞİL (başka mimari). v02-X'e bağ: FOTON kollarının val alan dizisi + adım 0 ham kayıpları "
                    "(v02 koşularında özet alanı yok)"}


def direct_savings(curve_x: dict[int, float], curve_base: dict[int, float]) -> float | None:
    """EĞİMSİZ okuma (%): X, taban kolun son adımdaki kaybına kaç adımda ulaştı -> 100·(1 − N_X / son adım). X oraya
    ulaşamıyorsa tersinden: taban X'in son kaybına N_B adımda ulaştı -> −100·(1 − N_B / son adım). Karar kuralına GİRMEZ."""
    end = max(curve_base)
    n_x = sc.steps_to_reach(curve_x, curve_base[end])
    if n_x is not None:
        return 100.0 * (1.0 - n_x / end)
    n_b = sc.steps_to_reach(curve_base, curve_x[max(curve_x)])
    return None if n_b is None else -100.0 * (1.0 - n_b / max(curve_x))


def direct_reading(per_seed: dict, mean_nat: float) -> dict:
    """Eğimsiz okumanın nat farkıyla ÇELİŞKİSİ, veri görülmeden yazılı ölçütle: İŞARET. Nat farkının ortalaması pozitifken
    (X-dom daha iyi) doğrudan okumanın ortalaması <= 0 ise, ya da tersi, çelişki vardır. Bir seed'de okuma yapılamadıysa
    (eğriler birbirinin son kaybına hiç ulaşmıyor) bayrak `None`'dır ve bu da raporun başında yazılır. Karar DEĞİŞMEZ."""
    vals = [per_seed[s] for s in dec.SEEDS]
    if any(v is None for v in vals):
        return {"per_seed_percent": per_seed, "mean_percent": None, "contradicts_sign_of_nat_difference": None}
    mean = sum(vals) / len(vals)
    return {"per_seed_percent": per_seed, "mean_percent": mean, "contradicts_sign_of_nat_difference": bool((mean > 0) != (mean_nat > 0))}


def analyze(runs_dir: str, v02_dir: str, tag: str = "v03s1") -> dict:
    v02 = {s: read_v02(os.path.join(v02_dir, f"v7-x-s{s}"), s) for s in dec.SEEDS}
    runs = {arm: {s: read_run(os.path.join(runs_dir, f"{tag}-{arm}-s{s}"), arm, s, tag, v02[s]["config"]) for s in dec.SEEDS}
            for arm in ARMS}
    pairing = check_pairing(runs, v02)
    prov = [r["provenance"] for arm in runs.values() for r in arm.values()]
    provenance = {"all_code_clean": all(p["code_dirty"] is False for p in prov),
                  "src_sha256": sorted({str(p["src_sha256"])[:12] for p in prov}),
                  "sessions": sorted({str(p["session"]) for p in prov}),
                  "v02_src_sha256": sorted({v02[s]["src_sha256"] for s in dec.SEEDS})}
    if not provenance["all_code_clean"] or len(provenance["src_sha256"]) != 1:
        raise ValueError(f"karar koşularında code_dirty = False ve TEK src özeti zorunlu, istisnasız: {provenance}")
    by = lambda arm, key: {s: runs[arm][s][key] for s in dec.SEEDS}                     # noqa: E731 — SEED anahtarlı
    delta = dec.paired_difference(by("x-iid", "val"), by("x-dom", "val"), "delta")
    delta_d = dec.paired_difference(by("d", "val"), by("x-dom", "val"), "delta_D")
    pre = dec.precondition(by("x-iid", "mean_fixed"), {s: v02[s]["mean_fixed"] for s in dec.SEEDS},
                           by("x-iid", "ceiling_points"), {s: v02[s]["ceiling_points"] for s in dec.SEEDS})
    d_ok = dec.d_sanity({s: runs["d"][s]["val_curve"][dec.D_REF_STEP] for s in dec.SEEDS})
    decision = dec.decide(delta, delta_d, pre, d_ok)
    # BAĞLAYICI DEĞİL: delta'nın "ortalama sabit modül" yerine "en iyi sabit modül"e karşı hâli (delta, tanım gereği >= 0 olan
    # mean_fixed − best_fixed bileşenini içerir; v02'de 0.0007–0.0011 nat). Karar değişmez; ÇELİŞKİ varsa raporda öne yazılır.
    a_best = dec.question_a(dec.paired_difference(by("x-iid", "best_fixed"), by("x-dom", "val"), "delta_best"))
    measures = {arm: {s: {k: runs[arm][s][k] for k in MEASURES} for s in dec.SEEDS} for arm in ARMS if ARMS[arm]}
    measures["v02-x"] = {s: {k: v02[s][k] for k in MEASURES if k in v02[s]} for s in dec.SEEDS}
    return {
        "what": "v03 adım 1 — zorlanmış yönlendirme: modüller ayrışıyor mu (kural veri görülmeden yazıldı)",
        "conversion": f"puan = 100·(1 − exp(−Δ/{abs(dec.SLOPE)})); kural NAT cinsinden bağlayıcı",
        "decision": decision, "precondition": pre, "d_sanity": d_ok,
        "delta_vs_best_fixed_NOT_binding": {**a_best, "contradicts_question_a": a_best["passed"] != decision["question_a"]["passed"]},
        "mask_vs_weight": dec.mask_vs_weight(by("h-dom", "val"), by("x-dom", "val")),
        "val_assigned": {arm: by(arm, "val") for arm in ARMS},
        "direct_savings_percent_NOT_binding": {
            "x_dom_vs_x_iid": direct_reading({s: direct_savings(runs["x-dom"][s]["val_curve"], runs["x-iid"][s]["val_curve"])
                                              for s in dec.SEEDS}, decision["question_a"]["mean_nat"]),
            "x_dom_vs_d": direct_reading({s: direct_savings(runs["x-dom"][s]["val_curve"], runs["d"][s]["val_curve"])
                                          for s in dec.SEEDS}, decision["gate_b"]["mean_nat"])},
        "module_measures": measures,
        "tautological_in_dom_arms": "X-dom / H-dom'da modüller arası fark, oracle tavanı, null'ı aşan kısım (over_null), I(alan; en iyi "
                                    "modül) ve 'en iyi sabit tek modüle karşı puan' TANIM GEREĞİ büyüktür (modül alan dışını hiç "
                                    "görmedi): raporlanır, kanıt sayılmaz",
        "assigned_vs_best_fixed_points_TAUTOLOGICAL": {arm: {s: dec.points_of_nat(runs[arm][s]["best_fixed"] - runs[arm][s]["val"])
                                                             for s in dec.SEEDS} for arm in ("x-dom", "h-dom")},
        "pairing": pairing, "provenance": provenance,
    }


MEASURES = ("val", "val_router_UNUSED", "two_route_max_diff", "mean_fixed", "best_fixed", "spread_nat", "ceiling_nat",
            "ceiling_points", "over_null_nat", "over_null_points", "mi_domain_best_bits", "best_is_domain_share",
            "assign_table", "steps_with_gradless_module", "microbatches_missing_module")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "checkpoints", "foton-v03-step1"))
    ap.add_argument("--v02", default=os.path.join(ROOT, "checkpoints", "foton-v02-tarama"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = analyze(args.runs, args.v02)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    d = res["decision"]
    for name, q in (("(a) MEKANİZMA  delta", d["question_a"]), ("(b) KAPI       delta_D", d["gate_b"])):
        print(f"{name}: ort {q['mean_nat']:+.5f} nat = {q['mean_points']:+.2f} puan | seed başına "
              f"{ {s: round(v, 5) for s, v in q['per_seed_nat'].items()} } | eşik {q['threshold_nat']:.5f} | "
              f"{'GEÇTİ' if q['passed'] else 'GEÇMEDİ'}")
    print("ön koşul (X-iid null'ı temiz mi):", res["precondition"]["clean"], "| D akla yatkın mı:", res["d_sanity"]["clean"])
    if res["delta_vs_best_fixed_NOT_binding"]["contradicts_question_a"]:
        print("UYARI: 'en iyi sabit modüle karşı delta' (a)'nın cevabıyla ÇELİŞİYOR — raporda öne yazılır (karar değişmez)")
    for name, r in res["direct_savings_percent_NOT_binding"].items():
        if r["contradicts_sign_of_nat_difference"] is not False:
            print(f"UYARI: doğrudan (eğimsiz) okuma {name}: ort {r['mean_percent']} — nat farkının İŞARETİYLE çelişiyor ya da "
                  "okunamadı; raporda öne yazılır (karar değişmez)")
    print("SONRAKİ ADIM:", d["next_step"])
    print("->", args.out)


if __name__ == "__main__":
    main()
