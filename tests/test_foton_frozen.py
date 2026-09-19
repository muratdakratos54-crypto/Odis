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


@pytest.mark.parametrize("preset", ["foton-d", "foton-m", "foton-h", "foton-x", "foton-x-ent", "foton-x-lb"])
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


S0_ARMS = ["foton-d", "foton-m", "foton-h", "foton-x", "foton-x-ent", "foton-x-lb"]
S1_ARMS = ["foton-d-s1", "foton-m-s1", "foton-h-s1", "foton-x-s1"]


def test_closed_v2_list_is_identical_in_every_campaign_preset():
    """Ön-kayıt V2'nin KAPALI listesi: dört kolda (ve S1'de, ve V7 koşullarında) TrainConfig BİREBİR aynı.

    Tek fark `run_name`. Bir kolun derleyici bayrağı, VRAM sınırı ya da log aralığı farklı olsaydı
    `quality/sec` bir ayarla oynatılabilirdi ("başarılı" -> "güçlü başarı")."""
    from dataclasses import asdict
    ref = {k: v for k, v in asdict(PRESETS["foton-d"][1]).items() if k != "run_name"}
    for name in S0_ARMS + S1_ARMS:
        got = {k: v for k, v in asdict(PRESETS[name][1]).items() if k != "run_name"}
        assert got == ref, f"{name}: {[k for k in ref if got[k] != ref[k]]}"
        assert PRESETS[name][1].run_name == name
        assert PRESETS[name][1].budget_mult == 1 and PRESETS[name][0].dropout == 0.0
        assert PRESETS[name][0].moe_impl == "grouped" and PRESETS[name][0].tie_embeddings is True


def test_foton_arms_differ_from_the_dense_baseline_only_in_the_foton_fields():
    """H ve X, D ile AYNI boyutlarda (d 512, 8 katman, 8 head, ffn 1408, MoE yok): fark yalnızca yönlendirmede."""
    from dataclasses import asdict
    d = asdict(PRESETS["foton-d"][0])
    for name, arm, reg in (("foton-h", "h", "none"), ("foton-x", "x", "none"),
                           ("foton-x-ent", "x", "entropy"), ("foton-x-lb", "x", "lb")):
        m = asdict(PRESETS[name][0])
        assert {k for k in m if m[k] != d[k]} <= {"foton", "foton_reg"}, name
        assert (m["foton"], m["foton_reg"], m["foton_reg_coef"]) == (arm, reg, 0.01)
        assert (m["foton_prefix_k"], m["foton_pre"], m["foton_routed"], m["foton_post"]) == (64, 3, 2, 3)
        assert m["n_experts"] == 0
    assert asdict(PRESETS["foton-d"][0])["foton"] == "" and asdict(PRESETS["foton-m"][0])["foton"] == ""
    # H ile X'in tek farkı `foton` alanı (aynı regularizasyon): birincil-2 yalnızca heterojenliği izole eder
    h, x = asdict(PRESETS["foton-h"][0]), asdict(PRESETS["foton-x"][0])
    assert {k for k in h if h[k] != x[k]} == {"foton"}


def test_s1_presets_are_the_preregistered_640x10_step_with_a_4_2_4_split():
    for name in S1_ARMS:
        m = PRESETS[name][0]
        assert (m.d_model, m.n_layer, m.n_head, m.ffn_hidden, m.max_seq_len) == (640, 10, 10, 1760, 512)
    m = PRESETS["foton-m-s1"][0]
    assert (m.n_experts, m.top_k, m.expert_hidden, m.n_shared_experts, m.first_dense_layers) == (16, 3, 440, 1, 1)
    assert (m.aux_loss_coef, m.router_gate, m.bias_update_rate) == (0.0, "sigmoid", 0.001) and m.expert_hidden % 8 == 0
    for name, arm in (("foton-h-s1", "h"), ("foton-x-s1", "x")):
        m = PRESETS[name][0]
        assert (m.foton, m.foton_pre, m.foton_routed, m.foton_post, m.foton_prefix_k) == (arm, 4, 2, 4, 64)
        assert m.foton_routed == PRESETS["foton-x"][0].foton_routed, "modül başına katman sayısı donmuş eksen"


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
