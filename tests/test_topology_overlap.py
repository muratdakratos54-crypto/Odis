"""`scripts/topology_overlap.py`: ofset kümeleri üzerinden sayım, KURULMUŞ maskelerden kaba kuvvet sayımla aynı mı."""
from __future__ import annotations

import importlib.util
import itertools
import os

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("topology_overlap", os.path.join(ROOT, "scripts", "topology_overlap.py"))
to = importlib.util.module_from_spec(spec)
spec.loader.exec_module(to)

from src import attn_patterns as ap  # noqa: E402


@pytest.mark.parametrize("T", [64, 512, 700])
def test_offset_arithmetic_matches_brute_force_counting_on_the_built_masks(T):
    res = to.analyze(T)
    union = {m: ap.PATTERNS[m].masks(T, ap.N_HEAD_KAMPANYA).any(axis=0) for m in to.MODULES}      # [T, T] head birleşimi
    full = np.tril(np.ones((T, T), dtype=bool))
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    for m in to.MODULES:
        got = res["modules"][m]
        assert got["union_coverage_of_causal_pairs"] == pytest.approx(union[m].sum() / full.sum())
        assert got["mean_reach_tokens"] == pytest.approx(dist[union[m]].mean())
        assert got["max_reach_tokens"] == dist[union[m]].max()
        masks = ap.PATTERNS[m].masks(T, ap.N_HEAD_KAMPANYA)
        assert got["avg_connections_per_token_head"] == pytest.approx(masks.sum() / (masks.shape[0] * T))
    for a, b in itertools.combinations(to.MODULES, 2):
        j = (union[a] & union[b]).sum() / (union[a] | union[b]).sum()
        assert res["pairs"][f"{a}-{b}"]["jaccard"] == pytest.approx(j)


def test_campaign_length_reads_the_frozen_densities():
    r = to.analyze(512)
    assert r["full_causal_avg_connections"] == 256.5
    assert [round(r["modules"][m]["avg_connections_per_token_head"], 4) for m in "ABC"] == [128.1738, 128.5, 128.375]
    assert all(0.49 < r["modules"][m]["density_vs_full_causal"] < 0.51 for m in "ABC")
