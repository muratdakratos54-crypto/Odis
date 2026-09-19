"""`scripts/compare_runs.py` KARANTİNADA: FOTON verisine yöneltilirse hata verip çıkar.

O betik kendi `paired(a: list, b: list)` fonksiyonunu taşır ve `zip()` ile KONUMA göre eşler.
Konuma göre eşleme hatası bu depoda üç kez doğdu; bu test dördüncü doğum yerini kilitler.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("compare_runs", os.path.join(ROOT, "scripts", "compare_runs.py"))
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def _run_dir(tmp_path, name: str, config: dict) -> str:
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "metrics.jsonl").write_text(json.dumps({"type": "end", "step": 1}) + "\n", encoding="utf-8")
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(d)


V2_CFG = {"model": {"d_model": 512}, "train": {"data_dir": "data/sp8k", "loss_prefix_skip": 0}}


def test_v2_data_still_loads(tmp_path):
    rec = cr.load(_run_dir(tmp_path, "dense-tok-s1337", V2_CFG))
    assert rec["end"]["step"] == 1 and rec["config"] == V2_CFG


@pytest.mark.parametrize("path", ["checkpoints/foton-v02", "checkpoints/FOTON-v02r/r-m-s11",
                                  "experiments/foton-v02/summary.json"])
def test_foton_paths_are_refused(path):
    with pytest.raises(SystemExit, match="KARANTİNA"):
        cr.refuse_foton(path)


@pytest.mark.parametrize("config", [
    {"model": {}, "train": {"data_dir": "data/foton", "loss_prefix_skip": 0}},      # FOTON korpusu
    {"model": {}, "train": {"data_dir": "data/x", "loss_prefix_skip": 64}},         # prefix maskeli ölçüt
    {"model": {"foton": "x"}, "train": {"data_dir": "data/x"}},                     # FOTON kolu
])
def test_runs_are_refused_by_their_own_config_even_under_an_innocent_directory_name(tmp_path, config):
    """Dizin yeniden adlandırılsa da koşunun KENDİ config'i FOTON'u ele verir.

    Testin ADI bilerek "foton" içermez: pytest `tmp_path`'i test adından türetir ve ilk hâlde yolun kendisi
    "foton" içerdiği için karantinanın YOL bacağı ateşleniyor, config bacağı hiç sınanmıyordu (mutant K02
    hayatta kaldı). Aşağıdaki ilk assertion bu tuzağın geri gelmesini engeller."""
    run_dir = _run_dir(tmp_path, "dense-tok-s11", config)
    assert "foton" not in run_dir.lower()
    with pytest.raises(SystemExit, match="KARANTİNA") as e:
        cr.load(run_dir)
    assert "yol FOTON'u gösteriyor" not in str(e.value)


def test_main_refuses_foton_arguments_before_doing_anything(monkeypatch, tmp_path):
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["compare_runs.py", "--runs", "checkpoints/foton-v02r", "--out", str(out)])
    with pytest.raises(SystemExit, match="KARANTİNA"):
        cr.main()
    assert not out.exists()


def test_the_legacy_paired_is_positional_which_is_why_it_is_quarantined():
    """Belge niteliğinde: bu fonksiyon sırası farklı iki listeyi sessizce yanlış eşler."""
    a, b = [1.0, 2.0, 3.0], [2.5, 0.5, 1.5]          # b, a ile aynı seed'lerin başka sırası olsaydı...
    assert cr.paired(a, b)["per_seed"] == [-1.5, 1.5, 1.5]      # ...doğru cevap [0.5, 0.5, 0.5] olurdu
