"""`scripts/screening_context.py`: kol ortalamaları ve farklar koşu ADIYLA anahtarlı; eksik adım sessizce atlanmaz."""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("screening_context", os.path.join(ROOT, "scripts", "screening_context.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def test_group_means_and_gaps():
    curves = {"D": {"d1": {1000: 3.0, 2000: 2.0}, "d2": {1000: 3.2, 2000: 2.2}},
              "M": {"m1": {1000: 2.9, 2000: 1.9}},
              "X-none": {"x1": {1000: 3.3, 2000: 2.3}, "x2": {1000: 3.5, 2000: 2.5}}}
    out = sc.summarize(curves, steps=(1000, 2000))
    assert out["1000"]["D"]["mean"] == pytest.approx(3.1) and out["1000"]["D"]["n"] == 2
    assert out["1000"]["M"]["sd"] is None                                   # tek koşuda sd tanımsız: NaN yazma
    assert out["2000"]["X-none"]["minus_D"] == pytest.approx(0.3) and out["2000"]["X-none"]["minus_M"] == pytest.approx(0.5)
    assert out["2000"]["X-none"]["runs"] == {"x1": 2.3, "x2": 2.5}
    with pytest.raises(KeyError):                                           # bir koşuda o adım yoksa ortalama ALINMAZ
        sc.summarize({**curves, "X-none": {"x1": {1000: 3.3}}}, steps=(2000,))


def test_val_curve_refuses_the_unmasked_metric(tmp_path):
    rows = [{"type": "val", "step": 0, "val_loss": 10.0, "loss_prefix_skip": 64}, {"type": "train", "step": 1},
            {"type": "val", "step": 250, "val_loss": 5.0, "loss_prefix_skip": 64}]
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    assert sc.val_curve(str(tmp_path)) == {0: 10.0, 250: 5.0}
    rows[2]["loss_prefix_skip"] = 0
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        sc.val_curve(str(tmp_path))


def test_equal_loss_token_ratio_reads_how_many_more_tokens_x_needs():
    """Kayıp ln(adım)'da doğrusalsa ve X her adımda baseline'ın 2 kat token'ıyla aynı kayba ulaşıyorsa oran tam 2'dir —
    nat farkı adımdan adıma küçülse bile (düzleşen eğri) oran SABİT kalır."""
    import math
    base = {s: 10.0 - math.log(s) for s in (250, 500, 1000, 2000, 4000)}
    x = {s: 10.0 - math.log(s / 2) for s in (500, 1000, 2000, 4000)}
    out = sc.equal_loss_token_ratio(x, base, (500, 1000, 2000, 4000))
    assert all(v == pytest.approx(2.0) for v in out.values()) and set(out) == {"500", "1000", "2000", "4000"}
    assert sc.steps_to_reach(base, 10.0 - math.log(700)) == pytest.approx(700.0)      # eval noktaları ARASINDA enterpolasyon
    assert sc.steps_to_reach(base, 0.5) is None                                        # eğri o kayba hiç ulaşmıyor: uydurma
    assert sc.equal_loss_token_ratio({8000: 0.5}, base, (8000,)) == {"8000": None}
    assert sc.mean_curve({"a": {1: 2.0, 2: 1.0}, "b": {1: 4.0}}) == {1: 3.0}            # yalnızca ORTAK adımlar
