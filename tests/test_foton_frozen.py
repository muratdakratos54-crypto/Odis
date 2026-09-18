"""Ön-kaydın DONDURDUĞU ama hiçbir testin kilitlemediği üç şeyin regresyon testleri.

Mutasyon sınavı tasarlanırken bulundu: aşağıdakilerin her biri karar-kritik, ön-kayıtta sayısıyla
yazılı, ve bozulduğunda hiçbir guard ateşlenmiyordu.

  1. `evaluate()`'in prefix maskesi — raporlanan val loss pozisyon 64..511 üzerinden. Bozulursa
     FOTON'un dizi düzeyi router'ının nedensellik sızıntısı ölçüte geri girer (+0.64 puan bedava).
  2. `foton-d` / `foton-m` preset'lerinin dondurulmuş değerleri — `L*` bu değerlerle türetildi.
  3. `MultiDomainDataset.REF_BATCH = 16` — dondurulmuş val penceresi kümesi bu sabitin ürettiği
     tüketim sırasıdır; 32'ye çekilirse pencereler SESSİZCE değişir ve `L*` başka bir kümede
     ölçülmüş olur. Mevcut testler akışı `ds.REF_BATCH` ile karşılaştırdığı için sabiti
     değiştirmek iki tarafı birlikte kaydırıyor ve test geçmeye devam ediyordu.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PRESETS, ModelConfig, TrainConfig  # noqa: E402
from src.data_multi import MultiDomainDataset  # noqa: E402
from src.model import GPT  # noqa: E402
from src.train import evaluate  # noqa: E402

T = 32


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(0)
    paths = {}
    for name in ("aaa", "bbb", "ccc"):
        p = tmp_path / f"{name}.bin"
        rng.integers(0, 64, size=4000, dtype=np.uint16).tofile(p)
        paths[name] = str(p)
    ds = MultiDomainDataset(paths, block_size=T, device="cpu")
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=64, max_seq_len=T, d_model=32, n_layer=1, n_head=2, ffn_hidden=64))
    ctx = torch.autocast("cpu", enabled=False)
    return model, ds, ctx


def _independent_masked_ce(model, ds, cfg, k: int) -> float:
    """Maskeli CE'nin BAĞIMSIZ hesabı: hedefleri −1'leyip modelin kendi `ignore_index` yolundan."""
    pairs = ds.frozen_val_windows(cfg.batch_size * cfg.eval_iters, 12345)
    tot, nb = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), cfg.batch_size):
            x, y = ds.windows(pairs[i:i + cfg.batch_size])
            y = y.clone()
            if k > 0:
                y[:, :k] = -1
            _, _, info = model(x, y)
            tot += info["ce"].item()
            nb += 1
    return tot / nb


def test_evaluate_primary_metric_skips_exactly_the_prefix(setup):
    """`primary` = pozisyon k..T−1 üzerinden CE; `full` = hepsi. İki bağımsız yol aynı sayıyı verir."""
    model, ds, ctx = setup
    cfg = TrainConfig(batch_size=16, eval_iters=2, loss_prefix_skip=8)
    ev = evaluate(model, ds, cfg, ctx)
    assert ev["primary"] == pytest.approx(_independent_masked_ce(model, ds, cfg, 8), abs=1e-5)
    assert ev["full"] == pytest.approx(_independent_masked_ce(model, ds, cfg, 0), abs=1e-5)
    assert ev["primary"] != pytest.approx(ev["full"], abs=1e-7), "maske hiçbir şey değiştirmiyor"


def test_evaluate_without_mask_reports_one_number_twice(setup):
    model, ds, ctx = setup
    ev = evaluate(model, ds, TrainConfig(batch_size=16, eval_iters=2, loss_prefix_skip=0), ctx)
    assert ev["primary"] == ev["full"]


def test_evaluate_mask_is_a_prefix_not_a_suffix(setup):
    """Maske İLK k pozisyonu atar. Son k'yı atan bir hata da 'bir şeyi maskeliyor' görünür."""
    model, ds, ctx = setup
    cfg = TrainConfig(batch_size=16, eval_iters=2, loss_prefix_skip=8)
    pairs = ds.frozen_val_windows(32, 12345)
    tot = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, 32, 16):
            x, y = ds.windows(pairs[i:i + 16])
            y = y.clone()
            y[:, -8:] = -1                                   # YANLIŞ maske: son 8
            tot += model(x, y)[2]["ce"].item()
    assert evaluate(model, ds, cfg, ctx)["primary"] != pytest.approx(tot / 2, abs=1e-6)


@pytest.mark.parametrize("preset", ["foton-d", "foton-m"])
def test_frozen_baseline_presets_match_the_preregistration(preset):
    """Ön-kayıt V1/V2'nin sayıları: bunlarla türetilen `L*` başka değerlerle taşınamaz."""
    mcfg, tcfg = PRESETS[preset]
    assert tcfg.loss_prefix_skip == 64, "raporlanan ölçüt pozisyon 64..511 (router nedenselliği)"
    assert tcfg.batch_size * tcfg.grad_accum * mcfg.max_seq_len == 32_768, "adım başına token PİNLİ"
    assert tcfg.batch_size * tcfg.eval_iters == 1280, "val penceresi sayısı PİNLİ"
    assert tcfg.batch_size % MultiDomainDataset.REF_BATCH == 0
    assert tcfg.max_steps == 4375 and tcfg.eval_interval == 250
    assert (0.8 * tcfg.max_steps) % tcfg.eval_interval == 0, "%80 noktası tam bir eval noktası olmalı"
    assert tcfg.data_dir == "data/foton" and tcfg.compile is False and tcfg.vram_limit_frac == 0.92
    assert (tcfg.lr, tcfg.min_lr, tcfg.warmup_steps, tcfg.weight_decay, tcfg.grad_clip) == \
        (1e-3, 1e-4, 200, 0.1, 1.0)
    assert (mcfg.d_model, mcfg.n_layer, mcfg.n_head, mcfg.ffn_hidden, mcfg.dropout) == (512, 8, 8, 1408, 0.0)


def test_foton_m_is_the_preregistered_moe_baseline():
    m = PRESETS["foton-m"][0]
    assert (m.n_experts, m.top_k, m.expert_hidden, m.n_shared_experts, m.first_dense_layers) == (16, 3, 352, 1, 1)
    assert (m.aux_loss_coef, m.router_gate, m.bias_update_rate, m.moe_impl) == (0.0, "sigmoid", 0.001, "grouped")
    assert m.expert_hidden % 8 == 0, "8'in katı değilse _grouped_mm sessizce döngü yoluna düşer"
    assert PRESETS["foton-d"][0].n_experts == 0


def test_ref_batch_is_the_frozen_literal_16(setup):
    """Dondurulmuş val penceresi kümesi `get_batch(16, g)`'nin tüketim sırasıdır — LİTERAL 16.

    `ds.REF_BATCH` ile karşılaştıran testler sabiti değiştirince iki tarafı birlikte kaydırır ve
    geçmeye devam eder; burada beklenen taraf literal ile kuruluyor.
    """
    _, ds, _ = setup
    assert MultiDomainDataset.REF_BATCH == 16
    g = torch.Generator().manual_seed(12345)
    expected = [ds.get_batch(16, g) for _ in range(4)]            # literal 16
    got = ds.frozen_val_windows(64, 12345)
    for m, (xo, yo) in enumerate(expected):
        xn, yn = ds.windows(got[m * 16:(m + 1) * 16])
        assert torch.equal(xo, xn) and torch.equal(yo, yn), f"pencere kümesi kaydı (parça {m})"
