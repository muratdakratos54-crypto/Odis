"""V7 kazanan seçimi ve V6 erken uyarı kuralı — ön-kaydın açık bıraktığı okumalar TARAMADAN ÖNCE burada kilitli.

V7'nin kazananı karar koşularının config'ine girer (ve SHA-256'sı pinlenir): seçim kuralının kendisi ölçümü
belirleyen koddur. Kural val loss'a BAKAMAZ.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("router_health", os.path.join(ROOT, "scripts", "router_health.py"))
rh = importlib.util.module_from_spec(spec)
sys.modules["router_health"] = rh
spec.loader.exec_module(rh)

SEEDS = (101, 102, 103)


def rows(per_seq: float, dead=0, marginal: float = 0.9, last: int = 2200, early=None) -> list[dict]:
    """25'lik log ızgarasında sentetik router satırları; `dead` sayı ya da {adım: sayı} olabilir."""
    out = []
    for step in range(25, last + 1, 25):
        d = dead.get(step, 0) if isinstance(dead, dict) else dead
        r = {"step": step, "router_entropy_marginal": marginal, "router_entropy_per_seq": per_seq,
             "router_dead": d, "router_load_cv": 0.1, "router_margin": 1.0, "router_switch_rate": 0.0,
             "router_counts": [500, 600, 500], "router_domain_counts": [[1, 1, 1]] * 3}
        if early and step in early:
            r.update(early[step])
        out.append(r)
    return out


def campaign(none=0.30, entropy=0.60, lb=0.50, dead=None) -> dict:
    vals, dead = {"none": none, "entropy": entropy, "lb": lb}, dead or {}
    return {c: {s: rows(vals[c], dead=dead.get(c, 0)) for s in SEEDS} for c in rh.CONDITIONS}


def test_winner_is_the_most_decisive_router_among_conditions_without_dead_modules():
    res = rh.v7_select(campaign())
    assert res["winner"] == "none" and res["winner_preset"] == "foton-x" and res["fallback"] is False
    assert res["selection_reads_val_loss"] is False
    # "none" en kararlı ama ÖLÜ modülü var -> süzgeçten geçemez, sıradaki en kararlı kazanır
    res = rh.v7_select(campaign(none=0.10, dead={"none": 1}))
    assert res["winner"] == "lb" and res["conditions"]["none"]["passes_dead_filter"] is False


def test_dead_filter_needs_every_seed_and_every_row_of_the_last_500_steps():
    c = campaign(none=0.10)
    c["none"][102] = rows(0.10, dead={2150: 1})             # TEK seed'in penceredeki TEK satırında ölü modül
    assert rh.v7_select(c)["winner"] == "lb"
    c["none"][102] = rows(0.10, dead={1600: 1})             # pencerenin DIŞINDA (2200 - 500 = 1700'den önce): sayılmaz
    assert rh.v7_select(c)["winner"] == "none"
    assert rh.WINDOW_STEPS == 500 and len(rh.window(rows(0.1))) == 20


def test_ties_follow_the_preregistered_order_and_the_tolerance_is_pinned():
    assert rh.CONDITIONS == ("none", "entropy", "lb") and rh.TIE_TOL == 1e-3
    assert rh.v7_select(campaign(none=0.3004, entropy=0.3000, lb=0.3008))["winner"] == "none"     # fark < 1e-3
    res = rh.v7_select(campaign(none=0.3020, entropy=0.3000, lb=0.3005))
    assert res["winner"] == "entropy" and res["tied"] == ["entropy", "lb"]


def test_fallback_when_no_condition_passes_is_flagged():
    res = rh.v7_select(campaign(none=0.2, entropy=0.1, lb=0.3, dead={"none": 1, "entropy": 2, "lb": 1}))
    assert res["fallback"] is True and res["winner"] == "none"      # en az ölü: none ve lb; aralarında en kararlı none
    assert "YEDEK" in res["reason"]


def test_selection_cannot_see_val_loss_and_input_is_keyed(tmp_path):
    run = tmp_path / "v7-x-s101"
    run.mkdir()
    lines = [{"type": "meta", "step": 0}, {"type": "val", "step": 0, "val_loss": 1.0}]
    lines += [{"type": "train", "val_loss": 9.9, "loss": 3.0, **r} for r in rows(0.3)]
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    got = rh.load_router_rows(str(run))
    assert all(set(r) == {"step", *rh.ROUTER_FIELDS} for r in got)   # val_loss / loss satıra GİRMEZ
    with pytest.raises(ValueError, match="tam üç koşul"):
        rh.v7_select({"none": {}, "lb": {}})
    with pytest.raises(ValueError, match="tam üç koşul"):
        rh.v7_select([rows(0.3)] * 3)
    c = campaign()
    del c["lb"][103]
    with pytest.raises(ValueError, match="AYNI seed"):
        rh.v7_select(c)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "metrics.jsonl").write_text(json.dumps({"type": "train", "step": 25, "loss": 1.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="router alanı"):
        rh.load_router_rows(str(tmp_path / "d"))


def test_early_warning_reads_the_three_intervals_ending_at_step_500():
    assert rh.EARLY_STEPS == (450, 475, 500)
    assert rh.early_warning(rows(0.5)) == []
    # init'te dizi başına entropi ZORUNLU olarak ~1.0: ilk aralıklar kuralı TETİKLEMEZ
    warm = {s: {"router_entropy_per_seq": 0.99} for s in range(25, 426, 25)}
    assert rh.early_warning(rows(0.5, early=warm)) == []
    stuck = {s: {"router_entropy_per_seq": 0.95} for s in (450, 475, 500)}
    assert any("ÖĞRENMİYOR" in w for w in rh.early_warning(rows(0.5, early=stuck)))
    two = {s: {"router_entropy_per_seq": 0.95} for s in (475, 500)}
    assert rh.early_warning(rows(0.5, early=two)) == []             # üçü birden gerekir
    dead = {s: {"router_entropy_marginal": 0.05} for s in (450, 475, 500)}
    assert any("ÇÖKME" in w for w in rh.early_warning(rows(0.5, early=dead)))
    with pytest.raises(ValueError, match="log adımları yok"):
        rh.early_warning(rows(0.5, last=400))


def test_v6g_collapse_criterion():
    assert rh.collapsed(rows(0.2))["collapsed"] is False
    assert rh.collapsed(rows(0.2, dead={2200: 2}))["collapsed"] is True
    assert rh.collapsed(rows(0.2, marginal=0.05))["collapsed"] is True
