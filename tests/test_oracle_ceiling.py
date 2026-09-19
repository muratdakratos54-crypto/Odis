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
