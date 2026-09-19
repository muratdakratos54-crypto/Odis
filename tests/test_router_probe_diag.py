"""`scripts/router_probe_diag.py`: prob, öğrenilebilir bir hedefi BULUYOR ve öğrenilemez bir hedefte şansta KALIYOR mu."""
from __future__ import annotations

import importlib.util
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("router_probe_diag", os.path.join(ROOT, "scripts", "router_probe_diag.py"))
pd_ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd_)


def test_probe_finds_a_linearly_predictable_target_and_stays_at_chance_on_an_unpredictable_one():
    rng = np.random.default_rng(0)
    n, d = 600, 16
    y = rng.integers(0, 3, n)
    centers = rng.normal(0, 1, (3, d))
    x = centers[y] * 2.0 + rng.normal(0, 1, (n, d))          # sınıf merkezleri ayrık: doğrusal olarak öğrenilebilir
    good = pd_.probe(x, y, 3, n_perm=5)
    assert good["cv_accuracy"] > 0.9 and good["cv_accuracy"] > good["permutation_null_max"] + 0.3
    assert good["permutation_p"] == 1 / 6                     # 5 permütasyonla ulaşılabilecek en küçük p
    noise = pd_.probe(x, rng.integers(0, 3, n), 3, n_perm=5)  # etiketler girdiden BAĞIMSIZ
    assert abs(noise["cv_accuracy"] - 1 / 3) < 0.08
    assert 0.3 < noise["majority_baseline"] < 0.4 and noise["n"] == n


def test_standardisation_is_learned_on_the_training_fold_only():
    """Sabit (sıfır varyanslı) bir özellik ve çok büyük ölçekli bir özellik probu bozmamalı."""
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 2, n)
    x = np.stack([y + rng.normal(0, 0.1, n), np.full(n, 7.0), rng.normal(0, 1e6, n)], 1)
    assert pd_.cv_accuracy(x, y, 2) > 0.95
