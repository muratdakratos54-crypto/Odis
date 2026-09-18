"""Kampanya kuyruğu üretecinin regresyon testleri (ön-kayıt V11a: seed bloğu + Latin kare).

Mutasyon sınavının yakaladığı boşluk: Latin kare denge assertion'ı yalnızca `blok % kol == 0`
iken çalışıyordu, yani kampanyanın GERÇEK biçiminde (5 seed × 4 kol) hiç çalışmıyordu. Rotasyon
satırı silinip sıra sabitlendiğinde hiçbir kontrol ateşlenmiyordu.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("mcq", os.path.join(ROOT, "scripts", "make_campaign_queue.py"))
mcq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcq)

ARMS = ["foton-d", "foton-m", "foton-h", "foton-x"]
SEEDS = [11, 22, 33, 44, 55]


def test_blocks_are_seed_blocks_and_every_arm_runs_once_per_block():
    runs = mcq.build_runs(ARMS, SEEDS)
    assert len(runs) == 20
    for bi, s in enumerate(SEEDS):
        block = runs[bi * 4:(bi + 1) * 4]
        assert {r["_block"] for r in block} == {f"s{s}"}, "blok bitişik değil: seed blokları iç içe geçmiş"
        assert sorted(r["_arm"] for r in block) == ["d", "h", "m", "x"]
        assert all(r["args"][-1] == str(s) for r in block)


def test_latin_square_rotation_balances_positions():
    runs = mcq.build_runs(ARMS, SEEDS)
    pos = mcq.positions(runs, ARMS, SEEDS)
    for a, p in pos.items():
        assert all(p.count(k) >= 1 for k in range(4)), f"{a} bir pozisyonu hiç görmedi: {p}"
    mcq.check_balance(runs, ARMS, SEEDS)                 # gerçek kampanya biçiminde de ÇALIŞIR


def test_check_balance_catches_a_fixed_order_queue():
    """Sabit sıralı (rotasyonsuz) bir kuyruk REDDEDİLMELİ — 5 × 4 biçiminde de."""
    fixed = []
    for s in SEEDS:
        for a in ARMS:
            fixed.append({"name": f"{a}-s{s}", "args": [], "_block": f"s{s}", "_arm": a.replace("foton-", "")})
    with pytest.raises(AssertionError, match="Latin kare dengesi"):
        mcq.check_balance(fixed, ARMS, SEEDS)


def test_check_balance_catches_a_missing_arm():
    runs = [r for r in mcq.build_runs(ARMS, SEEDS) if r["name"] != "x-s33"]
    with pytest.raises(AssertionError, match="kol kümesi bozuk"):
        mcq.check_balance(runs, ARMS, SEEDS)


def test_residual_position_imbalance_is_the_arithmetic_minimum():
    """5 blok × 4 kolda tam denge İMKÂNSIZ: ilk 4 blok tam Latin kare (kol başına toplam 6),
    beşinci blok 0..3'ü bir permütasyonla ekler -> ortalamalar daima {1.2, 1.4, 1.6, 1.8}.
    Ön-kayıt bu artık dengesizliği sayısıyla yazıyor; test sayının doğru olduğunu kilitler."""
    pos = mcq.positions(mcq.build_runs(ARMS, SEEDS), ARMS, SEEDS)
    means = sorted(round(sum(p) / len(p), 2) for p in pos.values())
    assert means == [1.2, 1.4, 1.6, 1.8]
    assert round(sum(pos["x"]) / 5 - sum(pos["m"]) / 5, 2) == 0.4
    assert round(sum(pos["x"]) / 5 - sum(pos["h"]) / 5, 2) == 0.2
