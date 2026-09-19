"""`scripts/router_diag.py`: router sebep incelemesinin ölçüleri, cevabı BİLİNEN sentetik dökümlerde."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("router_diag", os.path.join(ROOT, "scripts", "router_diag.py"))
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)


def _raw(sm, choice, dom, logits=None):
    sm = np.asarray(sm, dtype=np.float32)
    n = len(sm)
    logits = np.zeros((n, 3), dtype=np.float32) if logits is None else np.asarray(logits, dtype=np.float32)
    e = np.exp(logits - logits.max(1, keepdims=True))
    return {"seq_masked": sm, "choice": np.asarray(choice), "logits": logits, "probs": e / e.sum(1, keepdims=True),
            "domain": np.asarray(dom)}


def test_mutual_information_is_zero_for_independent_and_log2_3_for_identical_labels():
    a = np.tile(np.arange(3), 300)
    assert rd.mutual_information_bits(a, a, 3, 3) == pytest.approx(np.log2(3))
    b = np.repeat(np.arange(3), 300)                       # her (a, b) hücresi eşit dolu: bağımsız
    assert rd.mutual_information_bits(a, b, 3, 3) == pytest.approx(0.0, abs=1e-12)
    assert rd.mi_bias_bits(1280, 3, 3) == pytest.approx(4 / (2 * 1280 * np.log(2)))


def test_snapshot_reads_a_perfect_router_and_a_blind_router():
    n = 300
    dom = np.tile(np.arange(3), n // 3)
    sm = np.full((n, 3), 4.0)
    sm[np.arange(n), dom] = 3.0                            # alan d'nin en iyi modülü d, fark 1 nat
    perfect = rd.snapshot(_raw(sm, dom, dom, logits=np.eye(3)[dom] * 5))
    assert perfect["router_is_best"] == 1.0 and perfect["routed"] == pytest.approx(3.0) == perfect["oracle"]
    assert perfect["spread"] == pytest.approx(1.0) and perfect["margin"] == pytest.approx(5.0)
    assert perfect["choice_mi_domain_bits"] == pytest.approx(np.log2(3))
    assert perfect["best_module_mi_domain_bits"] == pytest.approx(np.log2(3))
    blind = rd.snapshot(_raw(sm, np.zeros(n, dtype=int), dom))          # hep modül 0, logit'ler 0
    assert blind["router_is_best"] == pytest.approx(1 / 3) and blind["routed"] == pytest.approx(4 - 1 / 3)
    assert blind["choice_mi_domain_bits"] == 0.0 and blind["max_prob"] == pytest.approx(1 / 3)
    assert blind["choice_counts"] == [n, 0, 0] and blind["best_counts"] == [100, 100, 100]


def test_flip_rate_is_measured_on_the_same_fixed_sequences_between_consecutive_evals(tmp_path):
    n = 12
    dom = np.tile(np.arange(3), 4)
    sm = np.full((n, 3), 4.0)
    first = np.zeros(n, dtype=int)
    second = first.copy()
    second[:3] = 1                                          # 12 diziden 3'ü modül değiştirdi
    os.makedirs(tmp_path / "eval_raw")
    for step, ch in ((0, first), (250, second), (500, second)):
        np.savez_compressed(tmp_path / "eval_raw" / f"step_{step:05d}.npz", step=step, **_raw(sm, ch, dom))
    out = rd.run_diag(str(tmp_path))
    assert [o["step"] for o in out] == [0, 250, 500]
    assert [o["flip_rate"] for o in out] == [None, 0.25, 0.0]
    with pytest.raises(ValueError):
        rd.run_diag(str(tmp_path / "yok"))


def test_malformed_dump_is_rejected_rather_than_silently_read():
    with pytest.raises(ValueError):
        rd.snapshot(_raw(np.zeros((5, 2)), np.zeros(5, dtype=int), np.zeros(5, dtype=int), logits=np.zeros((5, 3))))
