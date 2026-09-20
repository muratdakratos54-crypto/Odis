"""Dense teşhis koşusunun ölçütü (`scripts/v03_dcheck.py`), koşu görülmeden yazıldı; bu testler de öyle.

Sınanan sınıflar: ölçünün TEK NOKTAYA çökmesi (D-101'i yakalayan şey adım 2000 değil, 750–1750 aralığıydı), eşiklerin
kayması, işaret şartının düşmesi, "arada kaldı" bandının yutulması (iki hâlden birine zorlama), ön koşulların (takvim,
başlangıç, ayarlar, bütünlük) atlanması, adım kimliği yerine konuma göre eşleme, ve ölçütün BAŞARISIZ OLAMAMASI.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dc = _load("v03_dcheck")
REF, NEW = dc.REF_DIR, dc.NEW_DIR
# Diskteki koşulara dayanan testler: mutasyon kumu yalnız src/ + tests/ + scripts/ kopyalar, checkpoints/ orada yoktur.
# O testler orada ATLANIR; mutantları öldüren testlerin HEPSİ saf koddur (kum içinde de koşar).
REPEAT_DIR = os.path.join(ROOT, "checkpoints", "foton-v02r.kirli-src-iptal", "r-d-s11")
needs_runs = pytest.mark.skipif(not (os.path.isdir(REF) and os.path.isdir(REPEAT_DIR)),
                                reason="diskteki v02 koşuları yok (mutasyon kumu): çapa/takvim testleri atlanır")


def d_of(value, window=None):
    """TÜM adımlarda `value`, pencerede istenirse başka bir değer."""
    out = {s: value for s in dc.ALL_STEPS}
    if window is not None:
        out.update({s: window for s in dc.WINDOW})
    return out


def test_thresholds_are_the_anchored_numbers_and_the_window_is_the_one_that_caught_d101():
    assert dc.WINDOW == (750, 1000, 1250, 1500, 1750) and 2000 not in dc.WINDOW
    assert dc.ALL_STEPS == (250, 500, 750, 1000, 1250, 1500, 1750, 2000)
    assert (dc.SANE_WINDOW_MEAN, dc.SANE_MAX_ABS) == (0.004, 0.012)
    assert dc.GUILTY_WINDOW_MEAN == pytest.approx(dc.D101_ANOMALY["window_mean"] / 2, abs=1e-6)   # anomalinin YARISI
    # Eşikler AYNI KODLA koşmuş tekrar çiftinden türetildi (farklı kod sürümlü çiftten DEĞİL): zarf orada daha geniş
    assert dc.SAME_CODE_PAIR["max_abs_all"] > dc.OTHER_CODE_PAIR["max_abs_all"]
    assert abs(dc.SAME_CODE_PAIR["window_mean"]) > abs(dc.OTHER_CODE_PAIR["window_mean"])
    assert dc.SANE_WINDOW_MEAN >= abs(dc.SAME_CODE_PAIR["window_mean"]) * 3                       # gürültünün üstünde
    assert dc.SANE_MAX_ABS >= dc.SAME_CODE_PAIR["max_abs_all"] * 1.5                              # gerçek zarfa pay bırakır
    assert dc.SANE_WINDOW_MEAN < dc.GUILTY_WINDOW_MEAN and dc.INIT_TOL > abs(dc.SAME_CODE_PAIR["init_diff"]) * 10


@needs_runs
def _prov_src(d):
    with open(os.path.join(d, "provenance.json"), encoding="utf-8") as f:
        return json.load(f)["src_sha256"]


@needs_runs
def test_the_measured_anchors_are_the_ones_on_disk_and_the_main_one_is_a_same_code_repeat():
    """Eşikler diskteki ölçümlerden türetildi; burada yeniden hesaplanır (çapa kayarsa test düşer). ASIL çapa AYNI KODLA
    koşmuş çifttir: `code_hash` (src/**/*.py) iki koşuda da birebir aynı olmalı, yoksa ölçülen şey gürültü değil kod farkıdır."""
    def val(d):
        return {r["step"]: r["val_loss"] for r in dc.rows(d, "val")}
    diffs = []
    for label, arm, anchor, same_code in (("aynı kod", "m", dc.SAME_CODE_PAIR, True), ("farklı kod", "d", dc.OTHER_CODE_PAIR, False)):
        x = os.path.join(ROOT, "checkpoints", "foton-v02r", f"r-{arm}-s11")
        y = os.path.join(ROOT, "checkpoints", "foton-v02r.kirli-src-iptal", f"r-{arm}-s11")
        assert (_prov_src(x) == _prov_src(y)) is same_code, label
        a, b = val(x), val(y)
        rep = dc.paired_diff(b, a)
        assert np.mean([rep[s] for s in dc.WINDOW]) == pytest.approx(anchor["window_mean"], abs=1e-6), label
        assert max(abs(v) for v in rep.values()) == pytest.approx(anchor["max_abs_all"], abs=1e-6), label
        assert max(abs(rep[s]) for s in dc.WINDOW) == pytest.approx(anchor["max_abs_window"], abs=1e-6), label
        assert b[0] - a[0] == pytest.approx(anchor["init_diff"], abs=1e-9), label
        ce = dc.paired_diff({r["step"]: r["ce"] for r in dc.rows(y, "train")},
                            {r["step"]: r["ce"] for r in dc.rows(x, "train")}, steps=dc.WINDOW)
        assert np.mean(list(ce.values())) == pytest.approx(anchor["train_ce_window_mean"], abs=1e-6), label
        diffs.extend(rep.values())
    # Benzetimin adım-başı gürültü ölçeği = İKİ çiftin 16 adım farkının kök-ortalama-karesi
    assert dc.STEP_SD == pytest.approx(float(np.sqrt(np.mean(np.square(diffs)))), abs=1e-6) and len(diffs) == 16
    refs = [val(os.path.join(ROOT, "checkpoints", "foton-v02r", f"r-d-s{s}")) for s in (11, 22, 33, 44, 55)]
    d101 = val(os.path.join(ROOT, "checkpoints", "foton-v03-step1", "v03s1-d-s101"))
    anom = {s: d101[s] - float(np.mean([c[s] for c in refs])) for s in dc.ALL_STEPS}
    assert np.mean([anom[s] for s in dc.WINDOW]) == pytest.approx(dc.D101_ANOMALY["window_mean"], abs=1e-6)
    assert all(anom[s] > 0 for s in dc.WINDOW) is dc.D101_ANOMALY["all_positive"] is True


STATES = ("kod_saglam", "kod_suclu", "kod_sapiyor_kismi", "ters_yonde_sapma", "okunmaz")


def test_the_verdict_has_five_reachable_states_and_can_fail():
    assert dc.verdict(d_of(0.0))["outcome"] == "kod_saglam"
    assert dc.verdict(d_of(0.001133, window=0.001133))["outcome"] == "kod_saglam"                # ölçülen aynı-kod çifti
    assert dc.verdict(d_of(0.025295, window=0.025295))["outcome"] == "kod_suclu"                 # ölçülen anomali
    assert dc.verdict(d_of(0.008, window=0.008))["outcome"] == "kod_sapiyor_kismi"               # gerçek ama kısmi sapma
    assert dc.verdict(d_of(-0.020, window=-0.020))["outcome"] == "ters_yonde_sapma"              # kod DAHA İYİ
    mixed = {**d_of(0.0), 750: +0.02, 1000: +0.02, 1250: -0.02, 1500: +0.02, 1750: +0.02}
    assert dc.verdict(mixed)["outcome"] == "okunmaz"                                             # karışık işaret
    for out in STATES:
        assert out in dc.NEXT_STEP and "kalan 8" in dc.NEXT_STEP[out].lower()      # her hâlde kuyruk kısaltılmaz
    assert set(dc.NEXT_STEP) == set(STATES) | {"blocked"}


def test_the_unreadable_message_is_computed_from_the_measured_numbers_not_fixed():
    """Ters yönde BÜYÜK bir sapmada sabit bir 'anomalinin yarısından küçük' cümlesi basmak olgusal olarak YANLIŞ olurdu."""
    rev = dc.verdict({**d_of(0.0), 750: -0.03, 1000: -0.03, 1250: +0.001, 1500: -0.03, 1750: -0.03})
    assert rev["outcome"] == "okunmaz" and "-0.0238" in rev["next_step"].replace("−", "-")
    assert "hepsi negatif: False" in rev["next_step"] and rev["window_mean_nat"] < -dc.GUILTY_WINDOW_MEAN
    tidy = dc.verdict({**d_of(0.0), 250: 0.02})                                  # ortalama temiz, tek nokta büyük
    assert tidy["outcome"] == "okunmaz" and "0.020000" in tidy["next_step"] and "≤ 0.004" in tidy["next_step"]


@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_every_state_is_reachable_under_the_measured_noise_scale(rho):
    """Durdurma/karar kuralı ulaşılabilirlik sınavından geçer (benzetim; adım-başı sd iki tekrar çiftinden)."""
    tab = dc.reachability(sims=4000, rho=rho)
    # Gürültü ölçeği GERÇEK olmalı: etki 0'da karar ne kesin (ölçek küçültülmüş) ne de zayıf olmalı
    assert 0.92 <= tab[0.0]["kod_saglam"] <= 0.995 and tab[0.0].get("kod_suclu", 0.0) <= 0.002
    assert tab[round(dc.D101_ANOMALY["window_mean"], 6)]["kod_suclu"] >= 0.95
    assert tab[-0.025295]["ters_yonde_sapma"] >= 0.95
    assert tab[0.008].get("kod_sapiyor_kismi", 0) + tab[0.008].get("kod_suclu", 0) >= 0.80
    assert 0.40 <= tab[round(dc.GUILTY_WINDOW_MEAN, 6)]["kod_suclu"] <= 0.60          # eşikte yazı-tura (dürüstlük notu)
    assert 0.30 <= tab[0.004]["kod_saglam"] <= 0.70                                   # sağlam/kısmi sınırı da yazı-tura


def test_a_single_point_cannot_decide_the_verdict():
    """D-101 adım 2000'de sınırı %1.1 aştı ama 750-1750'de tamamen dışarıdaydı: ölçü tek noktaya çökerse bunu kaçırır."""
    only_2000 = {**d_of(0.0), 2000: 0.0121}                    # yalnız son nokta sapıyor: pencere temiz -> SAĞLAM değil
    assert dc.verdict(only_2000)["outcome"] == "okunmaz" and dc.verdict(only_2000)["max_abs_nat"] > dc.SANE_MAX_ABS
    only_window = {**d_of(0.0), **{s: 0.0253 for s in dc.WINDOW}}
    v = dc.verdict(only_window)
    assert v["outcome"] == "kod_suclu" and v["per_step_nat"][2000] == 0.0    # son nokta temizken bile pencere karar verir
    assert v["max_abs_window_nat"] == pytest.approx(0.0253) and v["max_abs_nat"] == pytest.approx(0.0253)


def test_the_window_mean_is_signed_and_the_sign_condition_is_required():
    mixed = {**d_of(0.0), 750: +0.06, 1000: +0.06, 1250: -0.06, 1500: +0.06, 1750: +0.045}
    v = dc.verdict(mixed)                                       # ortalama eşiğin çok üstünde ama bir adım TERS
    assert v["window_mean_nat"] > dc.GUILTY_WINDOW_MEAN and v["window_all_positive"] is False and v["outcome"] == "okunmaz"
    # İşaret şartları YALNIZ pencerede bakılır: pencere dışı adımlar 0 olsa da hâl pencereden çıkar
    assert dc.verdict(d_of(0.0, window=-0.02))["outcome"] == "ters_yonde_sapma"
    assert dc.verdict(d_of(-0.02))["window_all_negative"] is True and dc.verdict(d_of(0.02))["window_all_positive"] is True
    cancel = {**d_of(0.0), 750: +0.02, 1000: -0.02, 1250: +0.02, 1500: -0.02, 1750: 0.0}
    assert abs(dc.verdict(cancel)["window_mean_nat"]) < 1e-9 and dc.verdict(cancel)["outcome"] == "okunmaz"  # maks sapma büyük


@pytest.mark.parametrize("mean, mx, want", [(0.004, 0.012, "kod_saglam"), (0.0041, 0.012, "kod_sapiyor_kismi"),
                                            (-0.004, 0.012, "kod_saglam"), (-0.0041, 0.012, "ters_yonde_sapma"),
                                            (0.004, 0.0121, "okunmaz"), (0.012648 + 1e-9, 0.012648 + 1e-9, "kod_suclu"),
                                            (0.012648 - 1e-9, 0.012648 - 1e-9, "kod_sapiyor_kismi")])
def test_the_thresholds_are_inclusive_exactly_where_written(mean, mx, want):
    """Eşikler `<=` ve `>=`: tam eşikteki değer İÇERİDEDİR. Suçlu eşiğinde tam eşitlik kayan noktada temsil edilemediği
    için ±1e-9 ile sınanır; sağlam eşiği (0.003) tam olarak temsil edilebiliyor ve tam eşikte SAĞLAM veriyor."""
    d = {**d_of(0.0), **{s: mean for s in dc.WINDOW}, 250: mx if abs(mx) > abs(mean) else mean}
    assert dc.verdict(d)["outcome"] == want


@pytest.mark.parametrize("bad", [[0.0] * 8, (0.0,) * 8, np.zeros(8), {s: 0.0 for s in dc.WINDOW},
                                 {**{s: 0.0 for s in dc.ALL_STEPS}, 1000: float("nan")},
                                 {**{s: 0.0 for s in dc.ALL_STEPS}, 1000: None},
                                 {**{s: 0.0 for s in dc.ALL_STEPS}, 1000: True}])
def test_positional_or_incomplete_inputs_do_not_run_they_raise(bad):
    with pytest.raises((TypeError, ValueError)):
        dc.verdict(bad)


def test_pairing_is_by_step_identity_not_by_position_or_by_sorted_value():
    """Eğriler AZALAN (gerçek val eğrisi gibi) ve referans BAŞKA sırada yazılmış: konuma ya da sıralanmış değere göre
    eşleyen bir uygulama burada başka sayılar üretir."""
    new = {250: 8.0, 500: 7.0, 750: 6.0, 1000: 5.0, 1250: 4.0, 1500: 3.0, 1750: 2.0, 2000: 1.0}
    ref = {2000: 1.5, 750: 6.25, 250: 8.5, 1500: 3.25, 500: 7.5, 1750: 2.5, 1000: 5.5, 1250: 4.5}
    assert dc.paired_diff(new, ref) == {250: -0.5, 500: -0.5, 750: -0.25, 1000: -0.5, 1250: -0.5, 1500: -0.25, 1750: -0.5, 2000: -0.5}
    assert dc.verdict(dc.paired_diff(new, ref))["window_mean_nat"] == pytest.approx(-0.4, abs=1e-12)


# ------------------------------------------------------------------------------------ takvim (ön koşul P1)
def test_the_schedule_does_not_depend_on_the_truncated_budget():
    """`stop_step` koşuyu yalnızca KESER: tarama bütçesiyle hesaplanan lr dizisi, tam bütçeninkinin ön ekidir.
    (Saf kod testi: diskteki koşulara ihtiyaç duymaz, mutasyon kumunda da koşar.)"""
    short, full = dc.predicted_schedule(stop_step=2188), dc.predicted_schedule(stop_step=4375)
    assert short == {s: v for s, v in full.items() if s <= 2188} and len(short) == 87 and max(short) == 2175
    assert short[2000] == pytest.approx(0.0006469248450683149, abs=1e-15) and short[25] == pytest.approx(0.000125, abs=1e-15)


@needs_runs
def test_the_predicted_schedule_matches_the_frozen_reference_on_every_common_logged_step():
    """Sahibin şartı: takvim ortak adımlarda birebir aynı olmalı, yoksa teşhis koddan değil takvimden sapar.
    Tahmin KODDAN üretilir (koşu gerekmez) ve referansın LOGLADIĞI değerlerle karşılaştırılır."""
    ref = dc.read_curves(REF)
    sched = dc.schedule_identical(dc.predicted_schedule(), ref["lr"])
    assert sched["identical"] is True and sched["mismatched_steps"] == [] and sched["common_steps"] == 87
    assert dc.predicted_schedule()[2000] == ref["lr"][2000]


def test_a_schedule_mismatch_is_detected_exactly_not_within_a_tolerance():
    a = {25: 0.000125, 50: 0.00025, 75: 0.000375}
    assert dc.schedule_identical(a, a)["identical"] is True
    off = {**a, 50: 0.00025 + 1e-12}
    assert dc.schedule_identical(off, a)["identical"] is False and dc.schedule_identical(off, a)["mismatched_steps"] == [50]
    assert dc.schedule_identical({}, a)["identical"] is False                      # ortak adım yoksa "aynı" DEĞİLDİR


# ------------------------------------------------------------------------------------ uçtan uca (sentetik koşu klasörleri)
def _write(tmp, name, val_curve, lr=None, ce=None, cfg_patch=(), finished=True, seed=11):
    from dataclasses import asdict

    from src.config import PRESETS
    d = tmp / name
    os.makedirs(d, exist_ok=True)
    m, t = (asdict(x) for x in PRESETS["foton-d"])
    t.update(run_name=name, seed=seed, out_dir=str(tmp))
    for sec, k, v in cfg_patch:
        {"model": m, "train": t}[sec][k] = v
    (d / "config.json").write_text(json.dumps({"model": m, "train": t}), encoding="utf-8")
    rows = [{"type": "val", "step": s, "val_loss": v} for s, v in val_curve.items()]
    rows += [{"type": "train", "step": s, "lr": (lr or {}).get(s, 0.0005), "ce": (ce or {}).get(s, 3.0)} for s in (25, 1000, 2000)]
    if finished:
        rows.append({"type": "end", "step": 2188})
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(d)


def _curves(shift):
    base = {0: 10.483, **{s: 3.5 - 0.0002 * s for s in dc.ALL_STEPS}}
    return base, {s: v + (shift if s in dc.WINDOW else 0.0) for s, v in base.items()}


@pytest.mark.parametrize("shift, want", [(0.0, "kod_saglam"), (0.0253, "kod_suclu"), (0.008, "kod_sapiyor_kismi"),
                                         (-0.008, "ters_yonde_sapma")])
def test_end_to_end_reads_both_runs_and_reaches_each_verdict(tmp_path, shift, want):
    ref, new = _curves(shift)
    res = dc.diagnose(_write(tmp_path, "yeni", new), _write(tmp_path, "ref", ref))
    assert res["outcome"] == want and res["preconditions"]["clean"] is True
    assert res["window_mean_nat"] == pytest.approx(shift, abs=1e-9)


@pytest.mark.parametrize("kw, leg", [({"lr": {1000: 0.0004}}, "P1_schedule"),
                                     ({"cfg_patch": [("train", "lr", 0.002)]}, "P3_config"),
                                     ({"cfg_patch": [("train", "seed", 22)]}, "P3_config"),
                                     ({"cfg_patch": [("model", "n_layer", 6)]}, "P3_config"),
                                     ({"cfg_patch": [("model", "foton", "x")]}, "P3_config"),
                                     ({"cfg_patch": [("model", "n_experts", 8)]}, "P3_config")])
def test_a_failed_precondition_blocks_the_verdict_even_when_the_curves_agree(tmp_path, kw, leg):
    ref, new = _curves(0.0)
    res = dc.diagnose(_write(tmp_path, "yeni", new, **kw), _write(tmp_path, "ref", ref))
    assert res["outcome"] == "blocked" and res["preconditions"]["clean"] is False and "OKUNMAZ" in res["next_step"]
    assert res["preconditions"][leg] != {} and not (res["preconditions"][leg].get("identical") or res["preconditions"][leg].get("ok"))


def test_a_paired_comparison_at_the_wrong_seed_is_blocked_even_when_both_runs_agree(tmp_path):
    """İki koşu da seed 22 ile koşmuşsa config'ler BİREBİR aynıdır ve fark denetimi hiçbir şey görmez; ama teşhis
    dondurulmuş referansın seed 11 üyesine karşı yapılır. Seed şartı düşerse yanlış referansla 'sağlam' denirdi."""
    ref, new = _curves(0.0)
    res = dc.diagnose(_write(tmp_path, "yeni", new, seed=22), _write(tmp_path, "ref", ref, seed=22))
    assert res["outcome"] == "blocked" and res["preconditions"]["P3_config"]["same_seed"] is False
    assert res["preconditions"]["P3_config"]["train_unexpected"] == [] and dc.SEED == 11
    ok = dc.diagnose(_write(tmp_path / "b", "yeni", new), _write(tmp_path / "b", "ref", ref))
    assert ok["preconditions"]["P3_config"]["same_seed"] is True and ok["outcome"] == "kod_saglam"


def test_a_different_initialisation_blocks_the_verdict(tmp_path):
    ref, new = _curves(0.0)
    new = {**new, 0: ref[0] + 2e-6}                                  # aynı seed'de init AYNI olmalı; değilse kod testi değildir
    res = dc.diagnose(_write(tmp_path, "yeni", new), _write(tmp_path, "ref", ref))
    assert res["outcome"] == "blocked" and abs(res["preconditions"]["P2_init"]["diff"]) > dc.INIT_TOL
    ok = dc.diagnose(_write(tmp_path / "b", "yeni", {**new, 0: ref[0] + 9e-7}), _write(tmp_path / "b", "ref", ref))
    assert ok["outcome"] == "kod_saglam"


def test_an_unfinished_or_missing_run_raises(tmp_path):
    ref, new = _curves(0.0)
    with pytest.raises(ValueError, match="bitmemiş"):
        dc.diagnose(_write(tmp_path, "yeni", new, finished=False), _write(tmp_path, "ref", ref))
    with pytest.raises(ValueError, match="metrics.jsonl yok"):
        dc.diagnose(str(tmp_path / "yok"), _write(tmp_path, "ref", ref))
    ref2, new2 = _curves(0.0)
    del new2[1500]
    with pytest.raises(ValueError, match="eksik adım"):
        dc.diagnose(_write(tmp_path / "c", "yeni", new2), _write(tmp_path / "c", "ref", ref2))


def test_the_diagnostic_queue_is_one_run_outside_the_decision_campaign():
    with open(os.path.join(ROOT, "experiments", "foton-v03-step1", "queues", "dcheck_s11.json"), encoding="utf-8") as f:
        q = json.load(f)
    assert q["runs"] == [{"name": "dcheck-d-s11", "args": ["--preset", "foton-d", "--seed", "11", "--stop_step", "2188"]}]
    assert q["_strict"] is True and os.path.normpath(q["out_dir"]) == os.path.normpath("checkpoints/foton-v03-dcheck")
    an = _load("v03_step1_analyze")
    assert os.path.normpath(q["out_dir"]) != os.path.normpath("checkpoints/foton-v03-step1")      # analiz onu GÖRMEZ
    assert not any(q["runs"][0]["name"] == f"v03s1-{arm}-s{s}" for arm in an.ARMS for s in an.dec.SEEDS)
    assert os.path.normpath(dc.NEW_DIR).endswith(os.path.normpath("checkpoints/foton-v03-dcheck/dcheck-d-s11"))


def test_the_written_rule_document_carries_the_code_thresholds_verbatim():
    with open(os.path.join(ROOT, "experiments", "foton-v03-step1", "dcheck_rule.md"), encoding="utf-8") as f:
        doc = " ".join(f.read().split())
    for number in ("0.004", "0.012", "0.012648", "0.025295", "0.001133", "0.007758", "0.000177", "1e-6", "2188"):
        assert number in doc, number
    for state in ("KOD SAĞLAM", "KOD SUÇLU", "KOD SAPIYOR", "TERS YÖNDE", "OKUNMAZ"):
        assert state in doc
    assert "87 ortak loglanan adımın tamamında birebir aynı" in doc
