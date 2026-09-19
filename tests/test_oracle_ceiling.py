"""`scripts/oracle_ceiling.py`: tavan ölçüleri cevabı BİLİNEN sentetik dökümlerde; puana çevirme ön-kaydın sayısıyla."""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("oracle_ceiling", os.path.join(ROOT, "scripts", "oracle_ceiling.py"))
oc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oc)


def test_points_reproduces_the_preregistered_conversion_and_rejects_a_positive_slope():
    """Ön-kayıt V9: eğim -0.3346'da %5 tasarruf = 0.01716 nat; geç sekantta (-0.2385) 0.0122 nat."""
    assert oc.points(0.01716, oc.SLOPE_LSTAR_LOCAL) == pytest.approx(5.0, abs=0.01)
    assert oc.points(0.0122, oc.SLOPE_LATE_SECANT) == pytest.approx(5.0, abs=0.02)
    assert oc.points(0.0, -0.3) == 0.0 and oc.points(0.01, -0.2) > oc.points(0.01, -0.3)
    with pytest.raises(ValueError):
        oc.points(0.01, 0.3)


def test_ceiling_when_the_best_module_is_the_domain():
    """Alan d'nin en iyi modülü d (fark 1 nat): dizi-oracle = alan-oracle = 3.0; en iyi sabit modül 4 - 1/3."""
    n = 300
    dom = np.tile(np.arange(3), n // 3)
    sm = np.full((n, 3), 4.0)
    sm[np.arange(n), dom] = 3.0
    c = oc.ceiling(sm, np.zeros(n, dtype=int), dom, n_split=20, n_boot=50)
    assert c["oracle"] == pytest.approx(3.0) and c["domain_oracle_in_sample"] == pytest.approx(3.0)
    assert c["best_fixed"] == pytest.approx(4 - 1 / 3) and c["mean_fixed"] == pytest.approx(4 - 1 / 3)
    assert c["ceiling_vs_best_fixed"] == pytest.approx(2 / 3) and c["ceiling_vs_routed"] == pytest.approx(2 / 3)
    assert c["domain_gain_vs_best_fixed_in_sample"] == pytest.approx(2 / 3)
    # bölmeli de aynı kazanç (biraz fazlası: üç sabit modül berabere olduğundan seçim yarısında "kazanan" sabit modül,
    # alanı o yarıda fazla temsil edilmiş olandır ve değerlendirme yarısında ortalamanın biraz altında kalır)
    assert c["domain_oracle_minus_best_fixed_split_half"] == pytest.approx(-2 / 3, abs=0.05)
    # Bootstrap en iyi sabit modülü her yeniden örneklemde YENİDEN seçer (tahmin edicinin kendisi gibi). Üç modül tam
    # berabereyken bu seçim tabanı yalnızca aşağı çekebilir, yani aralık 2/3'ün hemen ALTINDA kalır: tavan tahmini
    # berabere modüllerde hafifçe aşağı yanlıdır (bu yüzden `ceiling_vs_mean_fixed` de raporlanır — seçimsiz taban).
    lo, hi = c["ceiling_vs_best_fixed_ci95"]
    assert 0.6 < lo <= hi <= 2 / 3 + 1e-9


def test_pure_selection_noise_gives_a_positive_hindsight_ceiling_but_no_domain_gain():
    """Üç modül istatistiksel olarak AYNI (bağımsız gürültü): geriye dönük oracle yine de 'kazanır' (seçim gürültüsü),
    ama alan düzeyinde öğrenilebilir hiçbir şey yoktur — ikiye-bölmeli alan kazancı ~0."""
    rng = np.random.default_rng(0)
    n = 3000
    sm = 3.0 + rng.normal(0, 0.01, (n, 3))
    dom = rng.integers(0, 3, n)
    c = oc.ceiling(sm, rng.integers(0, 3, n), dom, n_split=50, n_boot=50)
    assert c["ceiling_vs_best_fixed"] == pytest.approx(0.01 * 0.846, abs=0.0015)      # E[min of 3 N(0,1)] = -0.846
    assert abs(c["domain_oracle_minus_best_fixed_split_half"]) < 3e-4
    assert c["ceiling_vs_mean_fixed"] >= c["ceiling_vs_best_fixed"] >= 0


def test_a_single_dominant_module_gives_zero_ceiling():
    n = 90
    sm = np.tile(np.array([3.0, 3.5, 3.7]), (n, 1))
    c = oc.ceiling(sm, np.ones(n, dtype=int), np.tile(np.arange(3), n // 3), n_split=5, n_boot=20)
    assert c["best_fixed_module"] == 0 and c["ceiling_vs_best_fixed"] == 0.0
    assert c["ceiling_vs_routed"] == pytest.approx(0.5) and c["ceiling_vs_best_fixed_ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        oc.ceiling(sm, np.ones(n - 1, dtype=int), np.zeros(n, dtype=int))


def test_run_ceiling_cross_checks_the_dump_against_the_logged_curves(tmp_path):
    n = 60
    dom = np.tile(np.arange(3), n // 3)
    sm = np.full((n, 3), 4.0, dtype=np.float32)
    sm[np.arange(n), dom] = 3.5
    choice = np.zeros(n, dtype=np.int64)
    os.makedirs(tmp_path / "eval_raw")
    rows = []
    for step, tok, shift in ((250, 1e6, 0.4), (500, 2e6, 0.2), (750, 3e6, 0.0)):
        s = sm + shift
        np.savez_compressed(tmp_path / "eval_raw" / f"step_{step:05d}.npz", step=step, seq_masked=s, choice=choice, domain=dom)
        rows.append({"type": "val", "step": step, "tokens": int(tok), "val_loss": float(s[np.arange(n), choice].mean()),
                     "val_loss_oracle": float(s.min(1).mean()), "val_loss_oracle_null": float(s.min(1).mean()) + 0.01})
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = oc.run_ceiling(str(tmp_path))
    assert [s["step"] for s in out["steps"]] == [250, 500, 750] and out["own_local_slope"] < 0
    assert out["steps"][-1]["structured_gain_over_null"] == pytest.approx(0.01)
    assert out["steps"][-1]["ceiling_vs_best_fixed"] == pytest.approx(4 - 0.5 / 3 - 3.5, abs=1e-6)
    p = out["final_step_points"]["lstar_local_-0.3346"]
    assert p["structured_gain_over_null"] == pytest.approx(oc.points(0.01, -0.3346), rel=1e-4)
    # döküm ile log uyuşmuyorsa (yanlış dosya, bozuk döküm) sessizce hesaplamak yerine DUR
    rows[-1]["val_loss_oracle"] += 0.01
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        oc.run_ceiling(str(tmp_path))


def test_cross_run_agreement_separates_a_repeated_preference_from_selection_noise():
    """Koşudan koşuya TEKRARLANAN bir tercih (maske kaynaklı) örtüşmeyi şansın üstüne çıkarır; bağımsız gürültü çıkarmaz."""
    rng = np.random.default_rng(0)
    n = 6000
    shared = rng.integers(0, 3, n)
    noisy = {f"v7-x-s{s}": np.where(rng.random(n) < 0.5, shared, rng.integers(0, 3, n)) for s in (101, 102, 103)}
    rep = oc.best_module_agreement(noisy)
    assert rep["n_pairs"] == 3 and rep["mean_agreement"] > rep["mean_expected_if_independent"] + 0.15
    indep = oc.best_module_agreement({f"v7-x-s{s}": rng.integers(0, 3, n) for s in (101, 102, 103)})
    assert abs(indep["mean_agreement"] - indep["mean_expected_if_independent"]) < 0.02
    skew = oc.best_module_agreement({"a-s1": np.zeros(n, dtype=int), "b-s2": np.zeros(n, dtype=int)})
    assert skew["mean_agreement"] == 1.0 == skew["mean_expected_if_independent"]      # tek modül hep kazanıyorsa "örtüşme" bilgi değil
    # ÇARPIK ama BAĞIMSIZ etiketler (bir modül çoğu dizide kazanıyor): örtüşme 1/3'ün çok üstünde, ama şans düzeyi de öyle.
    # Şans düzeyi 1/M alınsaydı bu, sahte bir "tekrarlanan tercih" okurdu (mutant Z21).
    lopsided = {f"v7-x-s{s}": rng.choice(3, size=n, p=[0.8, 0.1, 0.1]) for s in (101, 102)}
    lop = oc.best_module_agreement(lopsided)
    assert lop["mean_agreement"] > 0.6 and abs(lop["mean_agreement"] - lop["mean_expected_if_independent"]) < 0.02
    with pytest.raises(ValueError):
        oc.best_module_agreement({"a-s1": np.zeros(5, dtype=int), "b-s2": np.zeros(6, dtype=int)})
    assert oc.seed_of("v7-x-ent-s102") == "102" and oc.seed_of("v7-x-s101") == "101"


def test_trajectory_in_points_uses_the_local_slope_at_each_eval_point(tmp_path):
    """Sabit bir nat tavanı, eğri düzleştikçe DAHA ÇOK puan eder: seyir yalnızca nat cinsinden okunmamalı."""
    n = 30
    dom = np.tile(np.arange(3), n // 3)
    base = np.full((n, 3), 4.0, dtype=np.float32)
    base[np.arange(n), dom] = 3.97                          # tavan her adımda aynı: 0.02 nat
    choice = np.zeros(n, dtype=np.int64)
    os.makedirs(tmp_path / "eval_raw")
    rows = []
    for step, tok, shift in ((250, 1e6, 1.0), (500, 2e6, 0.5), (750, 3e6, 0.3), (1000, 4e6, 0.25)):   # düzleşen eğri
        s = base + np.float32(shift)
        np.savez_compressed(tmp_path / "eval_raw" / f"step_{step:05d}.npz", step=step, seq_masked=s, choice=choice, domain=dom)
        rows.append({"type": "val", "step": step, "tokens": int(tok), "val_loss": float(s[np.arange(n), choice].mean()),
                     "val_loss_oracle": float(s.min(1).mean()), "val_loss_oracle_null": float(s.min(1).mean())})
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    steps = oc.run_ceiling(str(tmp_path))["steps"]
    assert "ceiling_points_own_slope" not in steps[0]                                  # tek noktadan eğim çıkmaz
    nat = [s["ceiling_vs_best_fixed"] for s in steps]
    assert max(nat) - min(nat) < 1e-5
    pts = [s["ceiling_points_own_slope"] for s in steps[1:]]
    assert pts[0] < pts[1] < pts[2]                                                    # aynı nat, artan puan
    assert steps[-1]["ceiling_points_own_slope"] == pytest.approx(oc.points(nat[-1], steps[-1]["own_local_slope_here"]))


def test_token_level_raw_gap_is_selection_noise_and_the_b11_null_is_degenerate_there():
    """Üç modül istatistiksel olarak AYNI olsa bile token-düzeyi minimum büyük bir "kazanç" okur; pozisyon-başı permütasyon
    null'ı bu düzeyde hiçbir şeyi düzeltemez (null = oracle). Sayı bu yüzden tek başına raporlanmaz."""
    rng = np.random.default_rng(0)
    ce = 3.0 + rng.normal(0, 0.1, (200, 3, 96))
    t = oc.token_level_raw(ce, k=32)
    assert t["raw_gap_vs_best_fixed"] == pytest.approx(0.1 * 0.846, abs=0.01)         # saf gürültüden gelen "tavan"
    assert t["null_gain"] == 0.0                                                       # dejenere: null = oracle
    assert t["per_token_module_spread"] == pytest.approx(0.1 * 1.693, abs=0.01)        # E[max - min of 3 N(0,1)] = 1.693
    ce[:, :, :32] += 5.0                                                               # prefix pozisyonları ölçüte GİRMEZ
    assert oc.token_level_raw(ce, k=32)["token_oracle"] == pytest.approx(t["token_oracle"])
