"""v03 adım 1 — karar kuralı (`v03_step1_decision.py`) ve analizi (`v03_step1_analyze.py`), cevabı BİLİNEN sentetik koşularda.

Kural veri görülmeden yazıldı; bu testler de öyle. Sınanan sınıflar: seed'lerin KONUMA göre eşlenmesi, eşiklerin kayması,
ön koşulun atlanması, kısmi / kirli / yanlış kurallı koşularla karar üretilmesi, iki yolun uyuşmazlığının (ya da NaN'ın)
yutulması, router'ın seçimiyle hesaplanan sayının kol kaybı yerine geçmesi, koşu kimliğinin KLASÖR ADINDAN okunması (yanlış
seed'li / yanlış kollu / yanlış bütçeli bir koşunun doğru adlı klasörde durması), eşleme iddiasının (aynı veri sırası, aynı
başlangıç) denetlenmeden kabul edilmesi, atama kuralının eğitimde gerçekten uygulanmamış olması.

Sentetik koşuların `config.json`'u GERÇEK preset'lerden (`src.config.PRESETS`) üretilir: bir preset değişirse (ör. lr) kimlik
denetimi burada patlar — 5 GPU saatlik koşudan SONRA analizde değil.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.config import PRESETS  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dec = _load("v03_step1_decision")
an = _load("v03_step1_analyze")
S = dec.SEEDS


def by_seed(*values):
    return dict(zip(S, values))


# ------------------------------------------------------------------------------------------------ kural
def test_thresholds_are_the_approved_numbers_under_the_single_conversion():
    assert dec.A_NAT == pytest.approx(0.010192, abs=1e-6) and dec.B_NAT == pytest.approx(0.017163, abs=1e-6)
    assert dec.points_of_nat(dec.A_NAT) == pytest.approx(3.0) and dec.points_of_nat(-dec.B_NAT) == pytest.approx(-5.0)
    assert dec.points_of_nat(0.0045) == pytest.approx(1.34, abs=0.01)          # v02'nin tavanıyla AYNI çevirim
    assert S == (101, 102, 103)


def test_question_a_needs_the_mean_at_threshold_and_every_seed_positive():
    assert dec.question_a(by_seed(0.011, 0.012, 0.009))["passed"] is True
    assert dec.question_a(by_seed(0.0101, 0.0101, 0.0101))["passed"] is False            # ortalama eşiğin hemen altında
    assert dec.question_a(by_seed(0.020, 0.020, -0.0001))["passed"] is False             # ortalama yüksek ama bir seed negatif
    assert dec.question_a(by_seed(0.020, 0.020, 0.0))["passed"] is False                 # sıfır pozitif DEĞİL
    out = dec.question_a(by_seed(0.03, 0.01, 0.02))
    assert out["mean_nat"] == pytest.approx(0.02) and out["per_seed_nat"] == {101: 0.03, 102: 0.01, 103: 0.02}


def test_gate_b_is_the_five_point_gate_against_the_paired_dense_arm():
    assert dec.gate_b(by_seed(0.018, 0.018, 0.018))["passed"] is True
    assert dec.gate_b(by_seed(0.016, 0.016, 0.016))["passed"] is False                   # (a)'nın eşiğini geçer, kapıyı geçmez
    assert dec.gate_b(by_seed(-0.05, -0.05, -0.05))["passed"] is False
    assert dec.gate_b(by_seed(0.06, 0.06, -0.001))["passed"] is False


@pytest.mark.parametrize("bad", [[0.02, 0.02, 0.02], (0.02, 0.02, 0.02), np.array([0.02, 0.02, 0.02]),
                                 {101: 0.02, 102: 0.02}, {101: 0.02, 102: 0.02, 103: 0.02, 104: 0.02},
                                 {11: 0.02, 22: 0.02, 33: 0.02}, {101: 0.02, 102: float("nan"), 103: 0.02},
                                 {101: 0.02, 102: None, 103: 0.02}, {101: True, 102: 0.02, 103: 0.02}])
def test_positional_or_incomplete_inputs_do_not_run_they_raise(bad):
    with pytest.raises((TypeError, ValueError)):
        dec.question_a(bad)
    with pytest.raises((TypeError, ValueError)):
        dec.paired_difference(bad, by_seed(1.0, 1.0, 1.0), "x")
    with pytest.raises((TypeError, ValueError)):
        dec.d_sanity(bad)


def test_pairing_is_by_seed_identity_not_by_position():
    """İki eşleme aynı değerleri BAŞKA eklenme sırasıyla taşır: fark konumdan değil seed kimliğinden çıkmalı."""
    a = {101: 3.10, 102: 3.20, 103: 3.30}
    b = {103: 3.29, 101: 3.05, 102: 3.21}                                     # sözlük sırası karışık
    assert dec.paired_difference(a, b, "x") == pytest.approx({101: 0.05, 102: -0.01, 103: 0.01})
    assert dec.question_a(dec.paired_difference(a, b, "x"))["all_seeds_positive"] is False


def test_the_next_step_map_has_exactly_the_preregistered_branches():
    clean, d_ok, d_bad = {"clean": True}, {"clean": True}, {"clean": False}
    yes, no = by_seed(0.02, 0.02, 0.02), by_seed(0.001, 0.001, 0.001)
    big, small = by_seed(0.03, 0.03, 0.03), by_seed(-0.04, -0.04, -0.04)
    assert "asıl deneyi" in dec.decide(yes, big, clean, d_ok)["next_step"] and dec.decide(yes, big, clean, d_ok)["outcome"] == [True, True]
    assert "ÖNCE 4/8, router SONRA" in dec.decide(yes, small, clean, d_ok)["next_step"]
    assert "2/8 → 4/8" in dec.decide(no, small, clean, d_ok)["next_step"]
    blocked = dec.decide(yes, big, {"clean": False}, d_ok)
    assert blocked["outcome"] == "blocked" and "OKUNMAZ" in blocked["next_step"] and blocked["question_a"]["passed"] is True
    with pytest.raises(TypeError):
        dec.decide(yes, big, {"clean": "evet"}, d_ok)
    with pytest.raises(TypeError):
        dec.decide(yes, big, clean, {"clean": 1})
    assert set(dec.NEXT_STEP) == {"blocked", (False, False), (False, None), (False, True), (True, None), (True, True), (True, False)}


def test_gate_passed_without_mechanism_is_unexpected_and_opens_nothing():
    """(b) EVET + (a) HAYIR, X-iid'in eşli D'yi yakaladığı anlamına gelir: v02 ile çelişir. Ne 4/8'e ne router'a geçilir."""
    out = dec.decide(by_seed(0.001, 0.001, 0.001), by_seed(0.03, 0.03, 0.03), {"clean": True}, {"clean": True})
    assert out["outcome"] == [False, True] and "BEKLENMEYEN" in out["next_step"] and "OKUNMAZ" in out["next_step"]
    assert "asıl deneyi" not in out["next_step"] and "AYNI ölçüm" not in out["next_step"]


def test_an_implausible_dense_arm_makes_only_the_gate_unreadable():
    yes, no, big = by_seed(0.02, 0.02, 0.02), by_seed(0.001, 0.001, 0.001), by_seed(0.03, 0.03, 0.03)
    out = dec.decide(yes, big, {"clean": True}, {"clean": False})
    assert out["outcome"] == [True, None] and "OKUNAMADI" in out["next_step"] and "AÇILMAZ" in out["next_step"]
    assert out["gate_b"]["passed"] is True and "asıl deneyi" not in out["next_step"]      # sayı raporlanır, karar açılmaz
    assert "2/8 → 4/8" in dec.decide(no, big, {"clean": True}, {"clean": False})["next_step"]
    ok = dec.d_sanity(by_seed(3.1198 + 0.0119, 3.1198 - 0.0119, 3.1198))
    assert ok["clean"] is True and ok["deviation_nat"][101] == pytest.approx(0.0119) and ok["mean_deviation_nat"] == pytest.approx(0.0, abs=1e-12)
    assert dec.d_sanity(by_seed(3.1198, 3.1198 - 0.005, 3.1198 + 0.0121))["clean"] is False        # yalnız seed-başı bacak
    assert dec.d_sanity(by_seed(3.1198 - 0.0121, 3.1198 + 0.005, 3.1198))["clean"] is False
    # SİSTEMATİK kayma: her seed kendi sınırının (0.012) içinde, ama üçü de aynı yöne 0.008 — kapıyı 0.008 nat kaydırırdı
    high, low = dec.d_sanity(by_seed(3.1278, 3.1278, 3.1278)), dec.d_sanity(by_seed(3.1118, 3.1118, 3.1118))
    assert high["clean"] is False and low["clean"] is False and high["mean_deviation_nat"] == pytest.approx(0.008)
    assert dec.d_sanity(by_seed(3.1198 + 0.0059, 3.1198 + 0.0059, 3.1198 + 0.0059))["clean"] is True
    assert (dec.D_REF_STEP, dec.D_REF_VAL, dec.D_REF_TOL, dec.D_REF_MEAN_TOL) == (2000, 3.1198, 0.012, 0.006)


def test_precondition_is_per_seed_two_sided_and_has_a_signed_mean_leg():
    same = by_seed(3.13, 3.14, 3.15)
    ceil = by_seed(1.4, 1.5, 1.2)
    ok = dec.precondition(same, same, ceil, by_seed(1.42, 1.38, 1.10))
    assert ok["clean"] is True
    assert dec.precondition(by_seed(3.13, 3.14, 3.166), same, ceil, ceil)["clean"] is False
    assert dec.precondition(by_seed(3.13, 3.124, 3.15), same, ceil, ceil)["clean"] is False
    assert dec.precondition(same, same, by_seed(1.4, 1.5, 1.7), ceil)["clean"] is False
    # SİSTEMATİK kayma: her seed kendi sınırının (0.015) içinde, ama üçü de aynı yöne 0.010 — (a) eşiği kadar bir kayma
    shifted = dec.precondition(by_seed(3.14, 3.15, 3.16), same, ceil, ceil)
    assert shifted["clean"] is False and shifted["mean_shift_nat"] == pytest.approx(0.010)
    assert dec.precondition(by_seed(3.12, 3.13, 3.14), same, ceil, ceil)["clean"] is False           # öbür yöne de
    mixed = dec.precondition(by_seed(3.14, 3.13, 3.15), same, ceil, ceil)                          # +0.010, −0.010, 0: kayma yok
    assert mixed["clean"] is True and mixed["mean_shift_nat"] == pytest.approx(0.0, abs=1e-12)
    assert dec.PRE_MEAN_SHIFT_NAT == 0.0075


def test_mask_vs_weight_is_a_diagnostic_with_a_measurability_floor_and_a_sign_condition():
    assert dec.mask_vs_weight(by_seed(3.12, 3.12, 3.12), by_seed(3.11, 3.11, 3.11))["measurable"] is True
    assert dec.mask_vs_weight(by_seed(3.112, 3.112, 3.112), by_seed(3.11, 3.11, 3.11))["measurable"] is False
    noisy = dec.mask_vs_weight(by_seed(3.13, 3.13, 3.109), by_seed(3.11, 3.11, 3.11))      # ortalama +0.013 ama bir seed ters
    assert noisy["mean_nat"] > dec.MASK_VS_WEIGHT_NAT and noisy["measurable"] is False
    assert dec.mask_vs_weight(by_seed(3.10, 3.10, 3.10), by_seed(3.11, 3.11, 3.11))["measurable"] is True   # negatif yönde de


# ------------------------------------------------------------------------------------------------ analiz (sentetik koşular)
# Temel kampanya KASTEN asimetriktir: her (kol, seed)'in düzeyi ve gürültüsü ayrıdır. İlk hâlinde D, v02, H-dom ve X-iid üç
# seed'de aynıydı ve gürültü her yerde aynı tohumdandı: delta_D'nin, ön koşulun ve maske teşhisinin seed-eşlemesi bozulunca
# hiçbir sayı değişmiyordu (ikinci incelemenin 5 sonda mutantı hayatta kaldı).
END = an.END_STEP
DOM = np.repeat(np.arange(3), an.VAL_DOMAIN_COUNTS)
N = len(DOM)
ROW_STEPS = (700, 700, 788)                                  # üç "log aralığı": toplam 2188 adım x 64 dizi
KIND = {"d": ("foton-d", None), "x-dom": ("foton-x-dom", "domain"), "h-dom": ("foton-h-dom", "domain"),
        "x-iid": ("foton-x-iid", "iid"), "v02": ("foton-x", None)}
LEVEL = {"x-dom": (3.100, 3.104, 3.097), "h-dom": (3.109, 3.115, 3.108), "x-iid": (3.130, 3.133, 3.128),
         "d": (3.121, 3.118, 3.123), "v02": (3.131, 3.130, 3.129)}
NOISE_SD = {101: 0.004, 102: 0.006, 103: 0.008}              # seed'e göre tavan ~1.0 / 1.5 / 2.0 puan: seed karışırsa ön koşul kirlenir
DIGEST = an.VAL_WINDOWS_SHA256


def _data_order(seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [rng.multinomial(n * an.SEQS_PER_STEP, [1 / 3] * 3).tolist() for n in ROW_STEPS]


def _init(seed: int, kind: str) -> np.ndarray:
    """Adım 0: aynı seed -> aynı ağırlıklar -> aynı ham kayıplar. H'nin maskeleri (A, A, A): yalnızca modül 0 (maske A) X ile
    ortaktır, 1. ve 2. sütunlar GERÇEKTE de ayrışır — analiz H-dom'da yalnız 0. sütunu karşılaştırmak ZORUNDA."""
    base = 10.4 + np.random.default_rng(1000 + seed).normal(0, 0.05, (N, 3))
    return base + (np.array([0.0, 0.005, -0.005]) if kind == "h-dom" else 0.0)


def _write_run(root, name, *, kind, seed, final, sm_of_step=None, dirty=False, finished=True, check_shift=0.0, src="a" * 64,
               cfg_patch=(), meta_patch=None, args=None, order_seed=None, init_seed=None, table="ok", end_step=END, drop_val=(),
               nan_check=False, nan_val=False, session=None, val_dom=None, oracle_shift=0.0, raw_step_shift=0, router_printed=False,
               no_routed_check=False, val_mode_label=None, raw_shift=0.0, check_only_shift=0.0, digest=DIGEST, digest_changes=False,
               val_skip_label=None, drop_train_row=False, raw_inf=False, end_tokens_shift=0):
    preset, mode = KIND[kind]
    dom = {None: DOM, "permuted": np.repeat(np.arange(3), [434, 431, 415]), "reordered": DOM[::-1].copy()}[val_dom]
    d = root / name
    os.makedirs(d / "eval_raw", exist_ok=True)
    m, t = (asdict(x) for x in PRESETS[preset])
    m["vocab_size"] = 32768
    t.update(run_name=name, seed=seed, stop_step=END, out_dir=str(root))
    if kind == "v02":
        m.pop("foton_assign")                                # v02 koşuları bu alan eklenmeden ÖNCE yazıldı
    for section, key, value in cfg_patch:
        {"model": m, "train": t}[section][key] = value
    (d / "config.json").write_text(json.dumps({"model": m, "train": t}), encoding="utf-8")
    want = an.IDENTITY.get(kind, {"meta_arm": "x", "masks": ["A", "B", "C"], "params_total": 55_326_720,
                                  "flops_train_per_token": 265_838_610, "foton_assign": ""})
    meta = {"type": "meta", "step": 0, "arm": want["meta_arm"], "end_step": END, "max_steps": an.MAX_STEPS,
            "tokens_per_step": an.TOKENS_PER_STEP, "loss_prefix_skip": 64, "params_total": want["params_total"],
            "flops_train_per_token": want["flops_train_per_token"]}
    if want["masks"]:
        meta["foton_connections_counted"] = {f"M{i}:{x}": 128.0 for i, x in enumerate(want["masks"])}
        meta["foton_assign"] = want["foton_assign"]
    meta.update(meta_patch or {})
    rows = [meta]
    for step in an.VAL_STEPS:
        if step in drop_val:
            continue
        shift = 0.00002 * (END - step)
        rec = {"type": "val", "step": step, "tokens": step * an.TOKENS_PER_STEP, "loss_prefix_skip": 64}
        if kind != "v02" and digest is not None:             # v02 koşuları özet alanı eklenmeden ÖNCE yazıldı
            rec["val_windows_sha256"] = ("cd" * 32) if digest_changes and step == 1750 else digest
        if sm_of_step is None:                                                  # D kolu: yalnızca val eğrisi
            rec["val_loss"] = final + shift
        else:
            sm = (_init(seed if init_seed is None else init_seed, kind) if step == 0 else sm_of_step(dom) + shift).astype(np.float32)
            choice = np.zeros(N, dtype=np.int64)
            raw = sm + (np.float32(raw_shift) if step == 750 else np.float32(0.0))           # YALNIZ ham döküm kaymış
            if raw_inf and step == 1250:
                raw = raw.copy()
                raw[0, 2] = np.inf                                              # atanan hücre DEĞİL (dom[0] = 0): val_loss etkilenmez
            np.savez_compressed(d / "eval_raw" / f"step_{step:05d}.npz", step=step + (raw_step_shift if step == 500 else 0),
                                seq_masked=raw, choice=choice, domain=dom)
            oracle = float(sm.astype(np.float64).min(1).mean())
            rec.update(val_loss_oracle=oracle + oracle_shift, val_loss_oracle_null=oracle + 0.001)
            if kind == "v02":                                                   # v02-X: val_loss = router'ın seçimi
                rec.update(val_loss=float(sm[np.arange(N), choice].astype(np.float64).mean()))
                if not no_routed_check:
                    rec.update(val_loss_routed_check=0.0)
            else:
                assigned = float(sm[np.arange(N), dom].astype(np.float64).mean() if mode == "domain" else sm.astype(np.float64).mean())
                check = assigned + (check_only_shift if step == 750 else 0.0)  # YALNIZ ikinci yol kaymış
                rec.update(val_loss=assigned + check_shift, val_loss_assigned_check=float("nan") if nan_check else check,
                           val_assign_mode=mode, val_loss_router_UNUSED=float(sm[:, 0].mean()))
        if nan_val and step == 1000:
            rec["val_loss"] = float("nan")
        if val_mode_label and step == 1500:                                     # TEK bir eval noktası başka kuralla hesaplanmış
            rec["val_assign_mode"] = val_mode_label
        if val_skip_label is not None and step == 1500:                         # TEK bir eval noktası maskesiz ölçütle
            rec["loss_prefix_skip"] = val_skip_label
        rows.append(rec)
    orders = _data_order(seed if order_seed is None else order_seed)
    for i, counts in enumerate(orders[:-1] if drop_train_row else orders):
        rec = {"type": "train", "step": sum(ROW_STEPS[:i + 1])}
        if kind == "v02":
            rec["router_domain_counts"] = [[c, 0, 0] for c in counts]           # çökmüş router: hepsi modül 0
        else:
            rec["data_domain_counts"] = counts
        if mode:
            tab = np.diag(counts) if mode == "domain" else np.array([[c // 3, c // 3, c - 2 * (c // 3)] for c in counts])
            if table == "leak":                                                 # TEK bir dizi yanlış modüle gitmiş (alan toplamı aynı)
                tab = tab.copy()
                tab[0, 0], tab[0, 1] = tab[0, 0] - 1, tab[0, 1] + 1
            if table == "skewed":                                               # iid DEĞİL: modül 0 fazla yükleniyor
                tab = np.array([[c // 2, c // 4, c - c // 2 - c // 4] for c in counts])
            if table == "short":                                                # tablo, verinin gördüğünden bir dizi EKSİK sayıyor
                tab = np.asarray(tab).copy()
                tab[0, 0] -= 1
            rec.update(assign_domain_counts=np.asarray(tab).tolist(), assign_steps_gradless_module=0,
                       assign_microbatches_missing_module=2, router_counts=[5, 5, 5] if router_printed else None,
                       router_entropy_marginal=None)
        rows.append(rec)
    if finished:
        rows.append({"type": "end", "step": end_step, "tokens": end_step * an.TOKENS_PER_STEP + end_tokens_shift})
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    default_args = ["--preset", preset, "--seed", str(seed), "--stop_step", str(END)]
    (d / "provenance.json").write_text(json.dumps({"git_head": "h", "src_sha256": src, "code_dirty": dirty, "session": session or name,
                                                   "args": default_args if args is None else args}), encoding="utf-8")


def specialised(level):                        # X-dom / H-dom: modül yalnızca kendi alanında iyi
    def sm(dom):
        x = np.full((N, 3), level + 0.6)
        x[np.arange(N), dom] = level
        return x
    return sm


def generalist(level, arm="x-iid", seed=101, sd=None):
    """X-iid / v02-X: modüller ayırt edilemez. Gürültü (kol, seed)'e özgüdür; genliği seed'e bağlıdır (tavan seed'e göre değişir)."""
    def sm(dom):
        rng = np.random.default_rng([seed, sorted(KIND).index(arm)])
        noise = rng.normal(0, NOISE_SD[seed] if sd is None else sd, (N, 3))
        return level + noise - noise.mean(0)                                    # sütun ortalamaları TAM `level`: kesin assert'ler için
    return sm


def _write_arm(runs, v02_dir, arm, s, level, **kw):
    if arm == "d":
        return _write_run(runs, f"v03s1-d-s{s}", kind="d", seed=s, final=level, **kw)
    sm = specialised(level) if arm.endswith("dom") else generalist(level, arm, s)
    root, name = (v02_dir, f"v7-x-s{s}") if arm == "v02" else (runs, f"v03s1-{arm}-s{s}")
    _write_run(root, name, kind=arm, seed=s, final=level, **{"sm_of_step": sm, **kw})


def campaign(tmp_path, *, x_dom=LEVEL["x-dom"], x_iid=LEVEL["x-iid"], d=LEVEL["d"], h_dom=LEVEL["h-dom"], v02=LEVEL["v02"],
             overrides=None, only_seed=None):
    """`overrides` = {kol: _write_run anahtarları}; `only_seed` verilirse bozulma YALNIZCA o seed'in koşusuna uygulanır."""
    kw = overrides or {}
    runs, v02_dir = tmp_path / "runs", tmp_path / "v02"
    levels = {"x-dom": x_dom, "h-dom": h_dom, "x-iid": x_iid, "d": d, "v02": v02}
    for i, s in enumerate(S):
        for arm in KIND:
            _write_arm(runs, v02_dir, arm, s, levels[arm][i], **(kw.get(arm, {}) if only_seed in (None, s) else {}))
    return str(runs), str(v02_dir)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """TEMİZ bir kampanya, modül başına bir kez (150 dosya). Bozulma testleri onun TEK bir koşusunu bozup geri yazar."""
    return campaign(tmp_path_factory.mktemp("temel"))


@contextlib.contextmanager
def broken(base, arm, seed, kw):
    runs, v02_dir = (pathlib.Path(p) for p in base)
    level = LEVEL[arm][S.index(seed)]
    _write_arm(runs, v02_dir, arm, seed, level, **kw)
    try:
        yield base
    finally:
        _write_arm(runs, v02_dir, arm, seed, level)


def test_analysis_reads_the_assigned_loss_pairs_by_seed_and_walks_the_map(base):
    """Her sayı SEED BAŞINA kesin değerle sınanır: D'nin, v02-X'in ya da H-dom'un seed'leri kayarsa değerler değişir."""
    res = an.analyze(*base)
    a, b = res["decision"]["question_a"], res["decision"]["gate_b"]
    assert a["per_seed_nat"] == pytest.approx({101: 0.030, 102: 0.029, 103: 0.031}, abs=1e-6) and a["passed"] is True
    assert b["per_seed_nat"] == pytest.approx({101: 0.021, 102: 0.014, 103: 0.026}, abs=1e-6) and b["passed"] is True
    assert res["decision"]["outcome"] == [True, True] and res["precondition"]["clean"] is True and res["d_sanity"]["clean"] is True
    pre = res["precondition"]
    assert pre["mean_fixed_diff_nat"] == pytest.approx({101: -0.001, 102: 0.003, 103: -0.001}, abs=1e-6)
    assert pre["mean_shift_nat"] == pytest.approx(1e-3 / 3, abs=1e-6)
    ceil_iid = {s: res["module_measures"]["x-iid"][s]["ceiling_points"] for s in S}
    ceil_v02 = {s: res["module_measures"]["v02-x"][s]["ceiling_points"] for s in S}
    assert pre["ceiling_diff_points"] == pytest.approx({s: ceil_iid[s] - ceil_v02[s] for s in S}, abs=1e-9)
    assert ceil_v02[101] + 0.3 < ceil_v02[102] < ceil_v02[103] - 0.3 and all(abs(v) < 0.2 for v in pre["ceiling_diff_points"].values())
    assert res["d_sanity"]["deviation_nat"] == pytest.approx({101: 3.12476 - 3.1198, 102: 3.12176 - 3.1198, 103: 3.12676 - 3.1198}, abs=1e-6)
    mw = res["mask_vs_weight"]
    assert mw["per_seed_nat"] == pytest.approx({101: 0.009, 102: 0.011, 103: 0.011}, abs=1e-6) and mw["measurable"] is True
    # kol kaybı = ATANAN kayıp; router'ın seçimiyle hesaplanan sayı (modül 0'a zorlanmış gibi: +0.4) ona KARIŞMAZ
    assert res["val_assigned"]["x-dom"] == pytest.approx(dict(zip(S, LEVEL["x-dom"])), abs=1e-6)
    assert res["val_assigned"]["d"] == pytest.approx(dict(zip(S, LEVEL["d"])), abs=1e-9)
    assert all(v["val_router_UNUSED"] > 3.4 for v in res["module_measures"]["x-dom"].values())
    # totolojik ölçüler raporlanır ve öyle etiketlenir
    assert all(v > 15 for v in res["assigned_vs_best_fixed_points_TAUTOLOGICAL"]["x-dom"].values())
    assert res["module_measures"]["x-dom"][101]["best_is_domain_share"] == 1.0 and "over_null" in res["tautological_in_dom_arms"]
    assert res["provenance"]["all_code_clean"] is True and res["pairing"]["same_data_order_all_arms_incl_D"] is True
    assert res["pairing"]["val_windows_sha256"] == DIGEST and res["pairing"]["same_val_windows_all_12_runs_incl_D"] is True
    table = np.array(res["module_measures"]["x-dom"][101]["assign_table"])
    assert table.sum() == END * 64 and table.sum() == np.trace(table)
    best = res["delta_vs_best_fixed_NOT_binding"]
    assert best["contradicts_question_a"] is False and 0 < best["mean_nat"] <= a["mean_nat"]
    direct = res["direct_savings_percent_NOT_binding"]
    assert direct["x_dom_vs_x_iid"]["contradicts_sign_of_nat_difference"] is False and direct["x_dom_vs_x_iid"]["mean_percent"] > 0
    assert direct["x_dom_vs_d"]["contradicts_sign_of_nat_difference"] is False and set(direct["x_dom_vs_d"]["per_seed_percent"]) == set(S)


def test_delta_against_the_best_fixed_module_is_reported_and_a_contradiction_is_flagged(tmp_path):
    """delta eşiği geçiyor, ama 'en iyi sabit modüle karşı' hâli geçmiyor (X-iid'in bir modülü ötekilerden 0.003 nat iyi:
    ortalama 3.13, en iyisi 3.128): karar DEĞİŞMEZ, çelişki bayrağı kalkar."""
    def uneven(dom):
        return generalist(3.13)(dom) + np.array([-0.002, 0.001, 0.001])
    res = an.analyze(*campaign(tmp_path, x_dom=(3.119, 3.119, 3.119), overrides={"x-iid": {"sm_of_step": uneven}}))
    assert res["decision"]["question_a"]["passed"] is True and res["decision"]["question_a"]["mean_nat"] == pytest.approx(0.011, abs=1e-6)
    assert res["delta_vs_best_fixed_NOT_binding"]["passed"] is False and res["delta_vs_best_fixed_NOT_binding"]["contradicts_question_a"] is True


def test_direct_reading_flags_a_sign_contradiction_with_the_nat_difference():
    ok = an.direct_reading({101: 5.0, 102: 4.0, 103: 6.0}, 0.01)
    assert ok["mean_percent"] == pytest.approx(5.0) and ok["contradicts_sign_of_nat_difference"] is False
    assert an.direct_reading({101: -5.0, 102: -4.0, 103: -6.0}, -0.01)["contradicts_sign_of_nat_difference"] is False
    assert an.direct_reading({101: -5.0, 102: -4.0, 103: 2.0}, 0.01)["contradicts_sign_of_nat_difference"] is True
    assert an.direct_reading({101: 5.0, 102: 4.0, 103: 6.0}, -0.01)["contradicts_sign_of_nat_difference"] is True
    unread = an.direct_reading({101: 5.0, 102: None, 103: 6.0}, 0.01)
    assert unread["mean_percent"] is None and unread["contradicts_sign_of_nat_difference"] is None


def test_specialisation_without_paying_the_sparsity_cost_and_no_specialisation(tmp_path):
    pays_not = an.analyze(*campaign(tmp_path / "a", x_dom=(3.115, 3.115, 3.115)))
    assert pays_not["decision"]["outcome"] == [True, False] and "ÖNCE 4/8" in pays_not["decision"]["next_step"]
    none = an.analyze(*campaign(tmp_path / "b", x_dom=(3.128, 3.128, 3.128)))
    assert none["decision"]["outcome"] == [False, False]


def test_an_implausible_dense_run_blocks_only_the_gate(tmp_path):
    res = an.analyze(*campaign(tmp_path, d=(3.121, 3.118, 3.20)))                        # seed 103'ün D'si adım 2000'de 3.20376
    assert res["d_sanity"]["clean"] is False and res["decision"]["outcome"] == [True, None]
    assert res["d_sanity"]["deviation_nat"][103] == pytest.approx(3.20376 - 3.1198, abs=1e-6)


def test_a_single_seed_going_the_wrong_way_blocks_yes_and_seeds_are_not_mixed_up(tmp_path):
    """Seed 103'te X-dom, X-iid'den KÖTÜ. Seed'ler konuma göre eşlenseydi (ör. klasör sırası) bu işaret başka seed'e yazılırdı."""
    res = an.analyze(*campaign(tmp_path, x_dom=(3.08, 3.08, 3.135)))
    a = res["decision"]["question_a"]
    assert a["per_seed_nat"][103] < 0 < a["per_seed_nat"][101] and a["mean_nat"] > dec.A_NAT and a["passed"] is False


def test_a_dirty_null_blocks_the_decision_per_seed_on_the_mean_and_on_the_ceiling(tmp_path):
    res = an.analyze(*campaign(tmp_path / "a", x_iid=(3.130, 3.133, 3.16)))
    assert res["precondition"]["clean"] is False and res["decision"]["outcome"] == "blocked"
    assert res["precondition"]["mean_fixed_diff_nat"][103] == pytest.approx(0.031, abs=1e-6)
    res = an.analyze(*campaign(tmp_path / "b", x_iid=(3.141, 3.140, 3.139)))              # her seed sınırın içinde: +0.010
    assert all(abs(v) < dec.PRE_MEAN_FIXED_NAT for v in res["precondition"]["mean_fixed_diff_nat"].values())
    assert res["precondition"]["clean"] is False and res["decision"]["outcome"] == "blocked"
    # YALNIZ tavan bacağı, YALNIZ seed 103'te: X-iid'in modülleri v02-X'inkilerden çok daha ayrışık (gürültü 0.008 -> 0.014)
    wide = {"x-iid": {"sm_of_step": generalist(LEVEL["x-iid"][2], "x-iid", 103, sd=0.014)}}
    res = an.analyze(*campaign(tmp_path / "c", overrides=wide, only_seed=103))
    pre = res["precondition"]
    assert all(abs(v) < dec.PRE_MEAN_FIXED_NAT for v in pre["mean_fixed_diff_nat"].values()) and abs(pre["mean_shift_nat"]) < dec.PRE_MEAN_SHIFT_NAT
    assert pre["ceiling_diff_points"][103] > dec.PRE_CEILING_POINTS > abs(pre["ceiling_diff_points"][101])
    assert pre["clean"] is False and res["decision"]["outcome"] == "blocked"


FORCED, ALL4 = ("x-dom", "h-dom", "x-iid"), ("d", "x-dom", "h-dom", "x-iid")
BROKEN = [
    # --- iki yol: üç bacak AYRI AYRI (val_loss kaymış / yalnız ikinci yol kaymış / yalnız ham döküm kaymış / sağlama NaN)
    ("x-dom", {"check_shift": 3e-4}, "iki yoldan uyuşmuyor"),
    ("x-iid", {"check_shift": -3e-4}, "iki yoldan uyuşmuyor"),
    ("x-dom", {"check_only_shift": 3e-4}, "iki yoldan uyuşmuyor"),
    ("x-iid", {"check_only_shift": -3e-4}, "iki yoldan uyuşmuyor"),
    ("h-dom", {"raw_shift": 3e-4}, "iki yoldan uyuşmuyor"),
    ("x-iid", {"raw_shift": -3e-4}, "iki yoldan uyuşmuyor"),
    ("x-dom", {"nan_check": True}, "sağlama sonlu değil"),
    ("h-dom", {"nan_val": True}, "val_loss sonlu değil"),
    ("d", {"nan_val": True}, "val_loss sonlu değil"),
    ("d", {"finished": False}, "bitmemiş"),
    ("x-iid", {"end_step": 2000}, "bitiş adımı"),
    ("h-dom", {"end_tokens_shift": 1}, "bitiş adımı"),
    ("d", {"drop_val": (2188,)}, "eval adımları"),
    ("h-dom", {"drop_val": (250,)}, "eval adımları"),
    ("x-dom", {"val_skip_label": 0}, "prefix maskeli değil"),
    ("d", {"val_skip_label": 0}, "prefix maskeli değil"),
    ("x-iid", {"dirty": True}, "code_dirty"),
    ("d", {"src": "b" * 64}, "TEK src"),
    # --- kimlik: doğru ADLI klasörde yanlış koşu
    ("x-dom", {"cfg_patch": [("train", "seed", 103)]}, "koşu kimliği"),
    ("d", {"cfg_patch": [("train", "seed", 11)]}, "koşu kimliği"),
    ("x-dom", {"cfg_patch": [("model", "foton_assign", "iid")]}, "koşu kimliği"),
    ("x-iid", {"cfg_patch": [("model", "foton_assign", "")]}, "koşu kimliği"),
    ("h-dom", {"cfg_patch": [("model", "foton", "x")]}, "koşu kimliği"),
    ("d", {"cfg_patch": [("model", "foton", "x")]}, "koşu kimliği"),
    ("d", {"cfg_patch": [("model", "n_experts", 8)]}, "koşu kimliği"),
    ("x-dom", {"cfg_patch": [("train", "resume", True)]}, "koşu kimliği"),
    ("x-iid", {"cfg_patch": [("train", "run_name", "v03s1-x-dom-s101")]}, "koşu kimliği"),
    # --- bütçe / ayarlar: aynı seed'li v02-X'le birebir olmalı
    ("x-dom", {"cfg_patch": [("train", "stop_step", 4375)]}, "izinli alanlar dışında"),
    ("d", {"cfg_patch": [("train", "max_steps", 8750)]}, "izinli alanlar dışında"),
    ("x-iid", {"cfg_patch": [("train", "lr", 0.002)]}, "izinli alanlar dışında"),
    ("h-dom", {"cfg_patch": [("train", "batch_size", 8)]}, "izinli alanlar dışında"),
    ("x-dom", {"cfg_patch": [("model", "foton_routed", 4)]}, "izinli alanlar dışında"),
    ("x-dom", {"cfg_patch": [("model", "foton_reg", "entropy")]}, "izinli alanlar dışında"),
    ("d", {"cfg_patch": [("train", "loss_prefix_skip", 0)]}, "izinli alanlar dışında"),
    ("x-iid", {"args": ["--preset", "foton-x-iid", "--seed", "101", "--stop_step", "2188", "--lr", "0.002"]}, "komut satırı"),
    ("h-dom", {"meta_patch": {"foton_connections_counted": {"M0:A": 1.0, "M1:B": 1.0, "M2:C": 1.0}}}, "meta satırı"),
    ("x-dom", {"meta_patch": {"end_step": 4375}}, "meta satırı"),
    ("d", {"meta_patch": {"params_total": 91_734_528, "arm": "m"}}, "meta satırı"),
    ("h-dom", {"meta_patch": {"flops_train_per_token": 265_838_610}}, "meta satırı"),
    ("x-iid", {"meta_patch": {"foton_assign": "domain"}}, "meta satırı"),
    # --- eşleme VERİDEN: bozulma x kol ÇAPRAZ ÇARPIMI ("kol X, denetim Y'nin dışında" sınıfı her hücrede kanıtlı olsun)
    *[(arm, {"order_seed": 999}, "veri sırası") for arm in ALL4],
    *[(arm, {"drop_train_row": True}, "toplamı") for arm in ALL4],
    *[(arm, {"init_seed": 999}, "aynı ağırlıklardan başlamamış") for arm in FORCED],
    *[(arm, {"val_dom": "reordered"}, "v02-X'inkiyle aynı değil") for arm in FORCED],
    *[(arm, {"val_dom": "permuted"}, "dondurulmuş val kümesi") for arm in FORCED],
    *[(arm, {"digest": "ef" * 32}, "aynı val pencerelerinde ölçülmemiş") for arm in ALL4],
    *[(arm, {"digest": None}, "val_windows_sha256") for arm in ALL4],
    ("d", {"digest_changes": True}, "koşu içinde değişiyor"),
    ("x-dom", {"digest": "kisa"}, "val_windows_sha256"),
    # --- atama kuralı eğitimde GERÇEKTEN uygulanmış mı
    ("x-dom", {"table": "leak"}, "köşegen dışına"),
    ("h-dom", {"table": "leak"}, "köşegen dışına"),
    ("x-iid", {"table": "skewed"}, "iid atama payları"),
    # --- eval'in kuralı: config / meta doğru, ama bir eval noktası BAŞKA kuralla etiketli (U19 ilk sınavda hayatta kalmıştı)
    ("x-dom", {"val_mode_label": "iid"}, "val_assign_mode"),
    ("x-iid", {"val_mode_label": "domain"}, "val_assign_mode"),
    ("d", {"val_mode_label": "domain"}, "val_assign_mode"),
    ("x-dom", {"table": "short"}, "alan toplamları"),
    ("x-iid", {"router_printed": True}, "router metriği basılmış"),
    # --- ham dökümler: doğru adım, sonlu, metrics.jsonl ile tutarlı
    ("x-iid", {"oracle_shift": 0.01}, "metrics.jsonl ile uyuşmuyor"),
    ("x-dom", {"raw_step_shift": 1}, "ham döküm başka adımın"),
    ("x-dom", {"raw_inf": True}, "sonlu değil"),
]


@pytest.mark.parametrize("arm, kw, msg", BROKEN, ids=[f"{i}-{a}-{m}" for i, (a, _, m) in enumerate(BROKEN)])
def test_partial_dirty_misidentified_or_unpaired_campaigns_produce_no_decision(base, arm, kw, msg):
    """Bozulma TEK bir koşuda (seed 102): 'bütün koşularda aynı hata' değil, tek yanlış koşu da yakalanmalı."""
    with broken(base, arm, 102, kw), pytest.raises(ValueError, match=msg):
        an.analyze(*base)


V02_BROKEN = [({"cfg_patch": [("train", "seed", 101)]}, "v02-X referansı beklenen koşu değil"),
              ({"cfg_patch": [("model", "foton_reg", "entropy")]}, "v02-X referansı beklenen koşu değil"),
              ({"cfg_patch": [("train", "stop_step", 0)]}, "v02-X referansı beklenen koşu değil"),
              ({"cfg_patch": [("train", "grad_accum", 8)]}, "v02-X referansı beklenen koşu değil"),
              ({"dirty": True}, "v02-X referansı beklenen koşu değil"),
              ({"finished": False}, "bitmemiş"),
              ({"drop_val": (2188,)}, "eval adımları"),
              ({"nan_val": True}, "val_loss sonlu değil"),
              ({"no_routed_check": True}, "öğrenilmiş-router'lı bir koşu değil"),
              ({"val_dom": "permuted"}, "dondurulmuş val kümesi"),
              ({"oracle_shift": 0.01}, "metrics.jsonl ile uyuşmuyor"),
              ({"order_seed": 999}, "veri sırası"),
              ({"init_seed": 999}, "aynı ağırlıklardan başlamamış")]


@pytest.mark.parametrize("kw, msg", V02_BROKEN, ids=[str(i) for i in range(len(V02_BROKEN))])
def test_the_v02_reference_is_checked_as_strictly_as_the_decision_runs(base, kw, msg):
    with broken(base, "v02", 102, kw), pytest.raises(ValueError, match=msg):
        an.analyze(*base)


def test_the_shared_base_campaign_is_clean_again_after_every_breakage(base):
    assert an.analyze(*base)["decision"]["outcome"] == [True, True]


def test_a_forced_run_in_the_v02_slot_or_a_run_in_the_wrong_folder_is_refused(tmp_path):
    runs, v02 = campaign(tmp_path)
    _write_run(tmp_path / "v02", "v7-x-s103", kind="x-iid", seed=103, final=3.13, sm_of_step=generalist(3.13))
    with pytest.raises(ValueError, match="v02-X referansı"):
        an.analyze(runs, v02)
    runs, v02 = campaign(tmp_path / "b")
    os.rename(os.path.join(runs, "v03s1-x-dom-s101"), os.path.join(runs, "tmp"))          # iki klasör yer değiştirmiş
    os.rename(os.path.join(runs, "v03s1-x-dom-s102"), os.path.join(runs, "v03s1-x-dom-s101"))
    os.rename(os.path.join(runs, "tmp"), os.path.join(runs, "v03s1-x-dom-s102"))
    with pytest.raises(ValueError, match="koşu kimliği"):
        an.analyze(runs, v02)


@pytest.mark.parametrize("copied", [{"order_seed": 101}, {"init_seed": 101}, {"order_seed": 101, "init_seed": 101}],
                         ids=["yalniz-veri-sirasi", "yalniz-baslangic", "ikisi"])
def test_copied_seeds_are_refused(tmp_path, copied):
    """Üç 'seed' aslında aynı koşunun kopyasıysa n = 3 değil n = 1'dir. Veri sırasının YA DA başlangıcın kopya olması yeter
    (ikisi birlikte sınanırsa `or` ile `and` ayırt edilemez)."""
    runs, v02 = campaign(tmp_path, overrides={k: copied for k in KIND})
    with pytest.raises(ValueError, match="birbirinin kopyası"):
        an.analyze(runs, v02)


def test_a_self_consistent_campaign_on_another_validation_set_is_refused(tmp_path):
    """12 koşu kendi aralarında AYNI val kümesinde, ama o küme koşulardan önce kaydedilen (v02'nin ölçüldüğü) küme değil:
    val dosyası arada yeniden üretilmişse kimlik, veri sırası ve kod özeti denetimlerinin HEPSİ geçer."""
    runs, v02 = campaign(tmp_path, overrides={k: {"digest": "ef" * 32} for k in ALL4})
    with pytest.raises(ValueError, match="kaydedilen"):
        an.analyze(runs, v02)
    assert len(an.VAL_WINDOWS_SHA256) == 64 and an.VAL_WINDOWS_SHA256 != "ef" * 32


def test_a_missing_run_or_missing_raw_dump_produces_no_decision(tmp_path):
    runs, v02 = campaign(tmp_path / "a")
    os.remove(os.path.join(runs, "v03s1-h-dom-s102", "metrics.jsonl"))
    with pytest.raises(ValueError, match="metrics.jsonl yok"):
        an.analyze(runs, v02)
    runs, v02 = campaign(tmp_path / "b")
    os.remove(os.path.join(runs, "v03s1-x-iid-s103", "eval_raw", "step_00500.npz"))
    with pytest.raises(ValueError, match="step_00500.npz yok"):
        an.analyze(runs, v02)
    runs, v02 = campaign(tmp_path / "c")
    os.remove(os.path.join(runs, "v03s1-d-s101", "config.json"))
    with pytest.raises(ValueError, match="config.json yok"):
        an.analyze(runs, v02)


def test_the_queue_on_disk_is_exactly_the_twelve_runs_the_analysis_expects():
    with open(os.path.join(ROOT, "experiments", "foton-v03-step1", "queues", "step1.json"), encoding="utf-8") as f:
        q = json.load(f)
    want = {f"v03s1-{arm}-s{s}": ["--preset", an.IDENTITY[arm]["preset"], "--seed", str(s), "--stop_step", str(END)]
            for arm in an.ARMS for s in S}
    assert {r["name"]: r["args"] for r in q["runs"]} == want and len(q["runs"]) == 12 and q["_strict"] is True
    assert os.path.normpath(q["out_dir"]) == os.path.normpath("checkpoints/foton-v03-step1")


def test_the_written_map_in_the_decision_document_is_the_code_map_verbatim():
    """'Kuralın TEK kopyası koddadır': belgedeki harita NEXT_STEP metinlerini KELİMESİ KELİMESİNE taşır; eşikler de öyle."""
    with open(os.path.join(ROOT, "experiments", "foton-v03-step1", "decision_rule.md"), encoding="utf-8") as f:
        doc = " ".join(f.read().split())
    for key, text in dec.NEXT_STEP.items():
        assert " ".join(text.split()) in doc, key
    for number in (f"{dec.A_NAT:.6f}", f"{dec.B_NAT:.6f}", str(dec.PRE_MEAN_FIXED_NAT), str(dec.PRE_MEAN_SHIFT_NAT),
                   str(dec.PRE_CEILING_POINTS), str(dec.D_REF_VAL), str(dec.D_REF_TOL), str(dec.MASK_VS_WEIGHT_NAT)):
        assert number in doc, number


def test_real_presets_differ_from_the_v02_arm_only_where_the_identity_check_allows():
    ref_m, ref_t = (asdict(x) for x in PRESETS["foton-x"])
    for arm, want in an.IDENTITY.items():
        m, t = (asdict(x) for x in PRESETS[want["preset"]])
        assert an._config_diff(m, ref_m) <= an.MODEL_KEYS_MAY_DIFFER and an._config_diff(t, ref_t) <= {"run_name"}
        assert (m["foton"], m["foton_assign"], m["n_experts"]) == (want["foton"], want["foton_assign"], 0)


def test_direct_savings_reads_tokens_not_a_slope():
    import math
    base = {s: 10.0 - math.log(s) for s in (500, 1000, 2000, 2188)}
    faster = {s: 10.0 - math.log(s * 1.25) for s in (500, 1000, 2000, 2188)}          # aynı kayba %20 daha az adımla
    assert an.direct_savings(faster, base) == pytest.approx(20.0, abs=0.5)
    assert an.direct_savings(base, faster) == pytest.approx(-20.0, abs=0.5)
    assert an.direct_savings(base, base) == pytest.approx(0.0, abs=1e-6)


def test_identity_table_matches_what_the_training_code_would_log_for_the_real_presets():
    """Analizin beklediği meta sayıları (parametre, FLOP/token, maske harfleri, kol harfi) GERÇEK modelden ve train.py'nin meta
    satırını kuran ifadelerden üretilir — koşu YOK. Yanlış bir beklenti, 5 GPU saatlik temiz bir kampanyayı analizde reddederdi."""
    from dataclasses import replace

    from src.flops import train_flops_per_token
    from src.model import GPT
    for arm, want in an.IDENTITY.items():
        mcfg = replace(PRESETS[want["preset"]][0], vocab_size=32768)
        model = GPT(mcfg)
        assert model.num_params() == want["params_total"], arm
        assert train_flops_per_token(mcfg) == want["flops_train_per_token"], arm
        assert (mcfg.foton or ("m" if mcfg.n_experts > 0 else "d")) == want["meta_arm"], arm            # train.py: meta["arm"]
        masks = [k.split(":")[1] for k in sorted(model.foton_connection_counts())] if mcfg.foton else None
        assert masks == want["masks"], arm
        tcfg = PRESETS[want["preset"]][1]
        assert (tcfg.max_steps, tcfg.batch_size * tcfg.grad_accum, tcfg.eval_interval, tcfg.loss_prefix_skip) == (an.MAX_STEPS, an.SEQS_PER_STEP, 250, 64)
        assert tcfg.batch_size * tcfg.grad_accum * mcfg.max_seq_len == an.TOKENS_PER_STEP
