"""Preset hash'i: karar koşularının config'i yazıldığı andan sonra DEĞİŞEMEZ (ön-kayıt V10)."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("preset_hash", os.path.join(ROOT, "scripts", "preset_hash.py"))
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)


def test_hash_covers_every_campaign_preset_and_is_stable():
    a, b = ph.compute(), ph.compute()
    assert a == b and set(a["presets"]) == set(ph.CAMPAIGN)
    assert ph.CAMPAIGN == ("foton-d", "foton-m", "foton-h", "foton-x",
                           "foton-d-s1", "foton-m-s1", "foton-h-s1", "foton-x-s1")
    assert len(set(a["presets"].values())) == len(ph.CAMPAIGN)          # her preset ayrı
    assert all(len(h) == 64 for h in a["presets"].values()) and len(a["total"]) == 64


@pytest.mark.parametrize("field,value,which", [
    ("foton_reg", "lb", "model"), ("foton_prefix_k", 32, "model"), ("foton_reg_coef", 0.02, "model"),
    ("lr", 9e-4, "train"), ("loss_prefix_skip", 0, "train"), ("eval_iters", 40, "train"), ("compile", True, "train"),
])
def test_any_change_to_a_preset_is_caught_by_verify(tmp_path, monkeypatch, field, value, which):
    path = str(tmp_path / "preset_hashes.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ph.compute(), f)
    assert ph.verify(path) == []
    m, t = ph.PRESETS["foton-x"]
    changed = (replace(m, **{field: value}), t) if which == "model" else (m, replace(t, **{field: value}))
    monkeypatch.setitem(ph.PRESETS, "foton-x", changed)
    bad = ph.verify(path)
    assert len(bad) == 1 and bad[0].startswith("foton-x:")


def test_missing_file_and_unknown_presets_fail(tmp_path):
    assert ph.verify(str(tmp_path / "yok.json"))
    path = str(tmp_path / "p.json")
    res = ph.compute()
    res["presets"]["foton-y"] = "0" * 64
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f)
    assert any("foton-y" in b for b in ph.verify(path))


def test_run_name_seed_and_out_dir_do_not_enter_because_they_are_preset_constants_not_run_values():
    """Hash PRESET'in hash'idir: koşuya özgü `--seed` / `--run_name` ezmeleri preset'i değiştirmez."""
    h = ph.preset_hash("foton-x")
    assert ph.PRESETS["foton-x"][1].seed == 1337 and h == ph.preset_hash("foton-x")
