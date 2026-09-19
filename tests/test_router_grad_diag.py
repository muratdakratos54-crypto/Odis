"""`scripts/router_grad_diag.py`: yeniden kurulan trunk modelin kendisiyle aynı mı, dL/dg gerçekten dL/dg mi."""
from __future__ import annotations

import contextlib
import importlib.util
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("router_grad_diag", os.path.join(ROOT, "scripts", "router_grad_diag.py"))
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)

from src.config import PRESETS  # noqa: E402
from src.model import GPT  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = replace(PRESETS["tiny-foton"][0], vocab_size=61, max_seq_len=48)
    model = GPT(cfg).double().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    idx = torch.randint(0, 61, (5, 48))
    tgt = torch.randint(0, 61, (5, 48))
    return model, idx, tgt


def test_rebuilt_trunk_gives_the_models_own_forced_module_loss(tiny):
    model, idx, tgt = tiny
    for m in range(3):
        _, _, info = model(idx, tgt, force_module=m)
        ref = info["ce"]                                # saf CE: regularizasyonlu koşullarda `loss` ek terim taşır
        seq, g, loss = gd.forced_loss_and_gate_grad(model, idx, tgt, m, contextlib.nullcontext())
        assert loss.item() == pytest.approx(ref.item(), abs=1e-6)
        assert seq.shape == (5,) and g.shape == (5,) and seq.mean().item() == pytest.approx(loss.item(), abs=1e-6)


def test_gate_gradient_matches_a_finite_difference_of_the_per_sequence_gate(tiny):
    """dL/dg_b: YALNIZCA b dizisinin gate'i 1 -> 1 + eps yapıldığında batch loss'unun değişimi / eps."""
    model, idx, tgt = tiny
    m, eps = 1, 1e-4
    _, g, base = gd.forced_loss_and_gate_grad(model, idx, tgt, m, contextlib.nullcontext())
    cos, sin = model.rope_cos[:48], model.rope_sin[:48]

    def loss_with(gate: torch.Tensor) -> float:
        with torch.no_grad():
            x = model.tok_emb(idx)
            for blk in model.blocks[: model.cfg.foton_pre]:
                x, _ = blk(x, cos, sin)
            x = x + gate[:, None, None] * (model._foton_module(m, x, cos, sin) - x)
            for blk in model.blocks[model.cfg.foton_pre:]:
                x, _ = blk(x, cos, sin)
            logits = model.lm_head(model.norm(x))
            return torch.nn.functional.cross_entropy(logits.view(-1, 61), tgt.reshape(-1)).item()

    assert loss_with(torch.ones(5, dtype=torch.float64)) == pytest.approx(base.item(), abs=1e-6)
    for b in range(5):
        up, dn = torch.ones(5, dtype=torch.float64), torch.ones(5, dtype=torch.float64)
        up[b] += eps
        dn[b] -= eps
        assert (loss_with(up) - loss_with(dn)) / (2 * eps) == pytest.approx(g[b].item(), rel=5e-3, abs=1e-7)


def test_summarize_reads_a_signal_that_points_to_the_best_module_and_one_that_does_not():
    rng = np.random.default_rng(0)
    loss = rng.normal(3.0, 0.1, (200, 3))
    adv = loss - loss.mean(1, keepdims=True)
    good = gd.summarize(loss, 0.01 * adv + 5e-4)             # daha iyi modül = daha negatif gradyan (+ ortak kip)
    assert good["grad_points_to_best"] == 1.0 and good["grad_points_to_worst"] == 0.0
    assert good["corr_grad_advantage_within_sequence"] == pytest.approx(1.0)
    blind = gd.summarize(loss, rng.normal(0, 1e-3, (200, 3)))   # sinyal modülün avantajından bağımsız
    assert abs(blind["grad_points_to_best"] - 1 / 3) < 0.1 and abs(blind["corr_grad_advantage_within_sequence"]) < 0.15
    inverted = gd.summarize(loss, -0.01 * adv)
    assert inverted["grad_points_to_best"] == 0.0 and inverted["grad_points_to_worst"] == 1.0
