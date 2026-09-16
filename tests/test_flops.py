"""Analitik FLOP formülünün (src/flops.py) modelin kendisine ve bağımsız bir sayaca karşı denetimi (CPU)."""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import pytest
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PRESETS, ModelConfig  # noqa: E402
from src.flops import forward_flops_per_token, secondary_forward_flops_per_token, train_flops_per_token  # noqa: E402
from src.model import GPT  # noqa: E402

CFG_MOE = ModelConfig(vocab_size=97, max_seq_len=16, d_model=32, n_layer=3, n_head=4, ffn_hidden=88,
                      n_experts=4, top_k=2, expert_hidden=24, n_shared_experts=1, first_dense_layers=1)
CFG_DENSE = replace(CFG_MOE, n_experts=0)


def active_matmul_params(model: GPT) -> int:
    """Formülden bağımsız sayım: token başına çalışan tüm lineer katman ağırlıkları, modülün kendisinden.
    MoE katmanında E expert'ten top_k tanesi (hepsi aynı boyutta) + router + shared."""
    n = model.lm_head.weight.numel()  # tied olsa da çıkışta tam bir matris çarpımı
    for block in model.blocks:
        n += sum(m.weight.numel() for m in block.attn.modules() if isinstance(m, nn.Linear))
        if block.is_moe:
            moe = block.ffn
            per_expert = sum(m.weight.numel() for m in moe.experts[0].modules() if isinstance(m, nn.Linear))
            n += moe.router.weight.numel() + moe.top_k * per_expert
            n += sum(m.weight.numel() for m in moe.shared.modules() if isinstance(m, nn.Linear))
        else:
            n += sum(m.weight.numel() for m in block.ffn.modules() if isinstance(m, nn.Linear))
    return n


@pytest.mark.parametrize("preset", ["dense", "moe", "moe-noaux", "tiny", "tiny-moe"])
def test_matmul_terms_equal_twice_active_params(preset):
    mcfg, _ = PRESETS[preset]
    f = forward_flops_per_token(mcfg)
    matmul = sum(v for k, v in f.items() if k != "attn_context")
    assert matmul == 2 * active_matmul_params(GPT(mcfg))


def test_dense_vs_moe_preset_differ_only_by_router():
    """Presetler aynı aktif FFN'le kuruldu (3 x 352 + 352 = 1408): tek analitik fark router çarpımları."""
    dense, moe = PRESETS["dense"][0], PRESETS["moe"][0]
    fd, fm = forward_flops_per_token(dense), forward_flops_per_token(moe)
    assert sum(fm.values()) - sum(fd.values()) == fm["moe_router"] == 7 * 2 * 512 * 16
    assert train_flops_per_token(moe) == 3 * sum(fm.values())


@pytest.mark.parametrize("cfg", [CFG_DENSE, replace(CFG_MOE, moe_impl="loop")], ids=["dense", "moe-loop"])
def test_profiler_matches_analytic_linear_terms(cfg):
    """Bağımsız sayaç (FlopCounterMode, çarpma+toplama = 2 FLOP) ile formülün matris çarpımı terimleri, ileri ve
    ileri+geri için birebir. CPU'da SDPA bu sayaçta 0 sayıldığı için attention bağlam terimi karşılaştırmaya girmez;
    o terim ve gruplu çarpımın (profiler'da formülü yok) durumu scripts/flops_report.py'de raporlanır."""
    torch.manual_seed(0)
    model = GPT(cfg)
    B, T = 3, cfg.max_seq_len
    x, y = torch.randint(0, cfg.vocab_size, (B, T)), torch.randint(0, cfg.vocab_size, (B, T))
    f = forward_flops_per_token(cfg)
    matmul_per_token = sum(v for k, v in f.items() if k != "attn_context")
    with FlopCounterMode(display=False) as fwd:
        model(x, y)
    assert fwd.get_total_flops() == B * T * matmul_per_token
    with FlopCounterMode(display=False) as both:
        _, loss, _ = model(x, y)
        loss.backward()
    assert both.get_total_flops() == 3 * B * T * matmul_per_token


@pytest.mark.parametrize("T", [1, 16, 512])
def test_attn_context_equals_explicit_causal_sum(T):
    """Kapalı form 2*d*(T+1), konum konum açık toplamla aynı olmalı: t. konum t+1 anahtara bakar,
    skorlar 2*d*(t+1) + değerlerin ağırlıklı toplamı 2*d*(t+1)."""
    cfg = replace(PRESETS["dense"][0], max_seq_len=T)
    d, L = cfg.d_model, cfg.n_layer
    total = L * sum(4 * d * (t + 1) for t in range(T))
    assert total % T == 0 and forward_flops_per_token(cfg)["attn_context"] == total // T


def test_attn_context_against_profiler_math_attention():
    """Bağımsız sayaç: SDPA'yı matris çarpımlarına açan MATH arka ucunda FlopCounterMode, attention'ı causal maskeden
    bağımsız T x T sayar (katman başına q.k ve a.v için 2 * B*H*T*T*hd). Analitik causal terim bunun tam
    (T+1)/(2T) katı olmalı."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    cfg = CFG_DENSE
    torch.manual_seed(0)
    model = GPT(cfg)
    B, T = 2, cfg.max_seq_len
    x = torch.randint(0, cfg.vocab_size, (B, T))
    f = forward_flops_per_token(cfg)
    matmul = sum(v for k, v in f.items() if k != "attn_context")
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        model(x, x)
    attn_full = counter.get_total_flops() - B * T * matmul  # lineer katmanlar dışında kalan = attention bmm'leri
    assert attn_full == cfg.n_layer * B * 4 * cfg.d_model * T * T
    assert f["attn_context"] * 2 * T == attn_full // B // T * (T + 1)


def test_secondary_terms_match_docs():
    """docs/flops.md bölüm 5'teki sayılar: ikincil/birincil %0.32 (dense) ve %0.36 (moe), oran 1.00219, 3006.6 adım."""
    (pd, sd), (pm, sm) = ((sum(forward_flops_per_token(PRESETS[p][0]).values()),
                           sum(secondary_forward_flops_per_token(PRESETS[p][0]).values())) for p in ("dense", "moe"))
    assert round(100 * sd / pd, 2) == 0.32 and round(100 * sm / pm, 2) == 0.36
    assert round((pm + sm) / (pd + sd), 5) == 1.00219
    assert round(3000 * (pm + sm) / (pd + sd), 1) == 3006.6
    assert round(3000 * pm / pd, 2) == 3005.38  # birincil tanımla eşit FLOP bütçesi (docs bölüm 4)


@pytest.mark.parametrize("first_dense", [0, 1, 3, 5])
def test_layer_counts_follow_model(first_dense):
    """first_dense_layers >= n_layer iken bile MoE katman sayısı modelinkiyle aynı (negatif değil)."""
    cfg = replace(CFG_MOE, first_dense_layers=first_dense)
    model = GPT(cfg)
    n_moe = sum(b.is_moe for b in model.blocks)
    f = forward_flops_per_token(cfg)
    assert f["moe_router"] == n_moe * 2 * cfg.d_model * cfg.n_experts
    assert f["ffn_dense"] == (cfg.n_layer - n_moe) * 6 * cfg.d_model * cfg.ffn_hidden
    assert sum(v for k, v in f.items() if k != "attn_context") == 2 * active_matmul_params(model)
