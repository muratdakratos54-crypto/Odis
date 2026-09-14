"""Modelin temel doğruluk testleri (CPU'da saniyeler sürer).

    .venv\\Scripts\\python -m pytest -q tests

Neden bu testler?  Bir dil modeli hatalı olsa bile loss genellikle yine düşer; hatalar sessizdir.
Bu testler en yaygın sessiz hataları yakalar:
  * nedensellik (causal mask) sızıntısı -> model geleceği "görür", train loss sahte şekilde iyi olur
  * RoPE'nin yanlış uygulanması         -> konum bilgisi bozulur
  * yanlış başlatma                     -> başlangıç loss'u ln(V)'den çok farklı olur
  * gradyan akışı kopukluğu             -> model tek bir batch'i bile ezberleyemez
Model seviyesindeki testler hem dense hem MoE konfigürasyonunda çalışır.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import ModelConfig  # noqa: E402
from src.model import GPT, apply_rope, precompute_rope  # noqa: E402

CFG = ModelConfig(vocab_size=97, max_seq_len=32, d_model=64, n_layer=2, n_head=4, ffn_hidden=176)
CFG_MOE = ModelConfig(vocab_size=97, max_seq_len=32, d_model=64, n_layer=3, n_head=4, ffn_hidden=176,
                      n_experts=6, top_k=2, expert_hidden=44, n_shared_experts=1, first_dense_layers=1)
BOTH = pytest.mark.parametrize("cfg", [CFG, CFG_MOE], ids=["dense", "moe"])


@BOTH
def test_causality(cfg):
    """t konumunun logit'i, t'den sonraki tokenlar değişince DEĞİŞMEMELİ.
    (MoE'de routing token bazında olduğu için bu özellik korunmalı.)"""
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    x2 = x.clone()
    x2[:, 10:] = torch.randint(0, cfg.vocab_size, (2, 6))
    y = torch.zeros_like(x)  # targets verince tüm konumların logit'i döner
    with torch.no_grad():
        l1, _, _ = model(x, y)
        l2, _, _ = model(x2, y)
    assert torch.allclose(l1[:, :10], l2[:, :10], atol=1e-5), "gelecekten bilgi sızıyor!"
    assert not torch.allclose(l1[:, 10:], l2[:, 10:]), "girdi değişti ama çıktı değişmedi?"


def test_rope_relative():
    """RoPE sonrası <q_m, k_n> yalnızca m - n'e bağlı olmalı."""
    torch.manual_seed(0)
    hd = 16
    cos, sin = precompute_rope(hd, 64, 10000.0)
    q = torch.randn(1, 1, 1, hd)
    k = torch.randn(1, 1, 1, hd)

    def score(m: int, n: int) -> float:
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum().item()

    assert math.isclose(score(5, 2), score(20, 17), rel_tol=1e-4)
    assert math.isclose(score(0, 0), score(40, 40), rel_tol=1e-4)
    # ...ama farklı göreli mesafeler farklı skor vermeli (yoksa hiç döndürme yapılmıyor demektir)
    assert not math.isclose(score(5, 2), score(5, 4), rel_tol=1e-3)
    # döndürme normu korumalı
    assert torch.allclose(apply_rope(q, cos[7:8], sin[7:8]).norm(), q.norm(), atol=1e-5)


def test_model_uses_positions():
    """Konum bilgisi olmayan tek katmanlı causal attention, önekteki tokenların SIRASINA kayıtsızdır
    (aynı kümeye aynı ağırlıklı ortalama). RoPE gerçekten (q VE k'ye) uygulanıyorsa [a,b,c,d] ile
    [b,a,c,d]'nin son konum çıktısı farklı olmalı. (Aynı tokenı tekrarlayan bir girdi bu testi
    yapamaz: tüm v vektörleri aynı olur ve RoPE doğru olsa bile çıktılar eşit çıkar.)"""
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=97, max_seq_len=32, d_model=64, n_layer=1, n_head=4, ffn_hidden=176)).eval()
    with torch.no_grad():
        la, _, _ = model(torch.tensor([[5, 17, 42, 9]]))
        lb, _, _ = model(torch.tensor([[17, 5, 42, 9]]))
    assert not torch.allclose(la, lb, atol=1e-5)


@BOTH
def test_initial_loss_near_uniform(cfg):
    """Rastgele başlatılmış model tüm tokenlara ~eşit olasılık vermeli: CE ≈ ln(V)."""
    torch.manual_seed(0)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (8, 32))
    # Hedefler girdiden BAĞIMSIZ olmalı: hedef=girdi verilirse weight tying yüzünden model
    # daha başlangıçta "şu anki token"a yüksek skor verir (h_t ~ e_x, logit = <h_t, e_x>)
    # ve loss yapay olarak ln(V)'nin altına düşer.
    y = torch.randint(0, cfg.vocab_size, (8, 32))
    _, loss, info = model(x, y)
    assert abs(info["ce"].item() - math.log(cfg.vocab_size)) < 0.3, info["ce"].item()
    if cfg.n_experts:
        # Başlangıçta router neredeyse düzgün -> aux loss ≈ 1 (denge)
        assert 0.9 < info["aux_loss"].item() < 1.3, info["aux_loss"].item()
        expected = info["ce"] + cfg.aux_loss_coef * info["aux_loss"] + cfg.z_loss_coef * info["z_loss"]
        assert torch.allclose(loss.detach(), expected)


@BOTH
def test_overfit_single_batch(cfg):
    """Gradyanlar tüm yolu doğru akıyorsa model tek bir batch'i hızla ezberler."""
    torch.manual_seed(0)
    model = GPT(cfg)
    opt = model.configure_optimizer(3e-3, 0.0, (0.9, 0.95), "cpu")
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.roll(x, -1, dims=1)
    for _ in range(150):
        _, loss, info = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert info["ce"].item() < 0.1, info["ce"].item()


def test_noncontiguous_targets():
    """Veri yükleyici y'yi xy[:, 1:] dilimi olarak (bitişik olmayan görünüm) verir; loss bunu kabul etmeli."""
    model = GPT(CFG)
    xy = torch.randint(0, CFG.vocab_size, (2, 17))
    x, y = xy[:, :-1], xy[:, 1:]
    assert not y.is_contiguous()
    _, loss, _ = model(x, y)
    assert torch.isfinite(loss)


def test_weight_tying_and_param_count():
    model = GPT(CFG)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    d, h, L, V = CFG.d_model, CFG.ffn_hidden, CFG.n_layer, CFG.vocab_size
    expected = V * d + L * (4 * d * d + 3 * d * h + 2 * d) + d
    assert model.num_params() == expected
    assert model.num_params(active_only=True) == expected  # dense: hepsi aktif


def test_aux_loss_reaches_router():
    """aux_loss_coef değişince router gradyanı değişmeli: aux terimi GPT.forward -> MoE -> router
    yolunda gerçekten türevlenebilir bağlı (ör. yanlışlıkla detach edilmemiş)."""
    x = torch.randint(0, CFG_MOE.vocab_size, (2, 16))
    y = torch.randint(0, CFG_MOE.vocab_size, (2, 16))
    grads = []
    for coef in (0.0, 1.0):
        torch.manual_seed(0)
        model = GPT(replace(CFG_MOE, aux_loss_coef=coef, z_loss_coef=0.0))
        _, loss, _ = model(x, y)
        loss.backward()
        grads.append([b.ffn.router.weight.grad.clone() for b in model.blocks if b.is_moe])
    for g0, g1 in zip(*grads):
        assert not torch.allclose(g0, g1)


def test_model_bf16_cast():
    """Model bütünüyle bf16'ya çevrildiğinde de (autocast'siz) MoE router'ı çalışmalı."""
    torch.manual_seed(0)
    model = GPT(CFG_MOE).to(torch.bfloat16).eval()
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), 3)
    assert out.shape == (1, 4)


def test_moe_param_count():
    c = CFG_MOE
    d, V, E, k = c.d_model, c.vocab_size, c.n_experts, c.top_k
    attn_norms = 4 * d * d + 2 * d
    dense_layer = attn_norms + 3 * d * c.ffn_hidden
    expert = 3 * d * c.expert_hidden
    shared = 3 * d * c.expert_hidden * c.n_shared_experts
    router = d * E
    n_moe = c.n_layer - c.first_dense_layers
    base = V * d + d + c.first_dense_layers * dense_layer + n_moe * (attn_norms + shared + router)
    model = GPT(c)
    assert model.num_params() == base + n_moe * E * expert
    assert model.num_params(active_only=True) == base + n_moe * k * expert
    assert [b.is_moe for b in model.blocks] == [False, True, True]


@BOTH
def test_generate_stops_at_eos(cfg):
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    start = torch.zeros(3, 1, dtype=torch.long)
    out = model.generate(start, 20, temperature=1.0, top_k=5)
    assert out.shape == (3, 21)  # eos_id yok -> tam uzunluk
    # Greedy'de 3 özdeş girdi aynı ilk tokenı üretir; onu EOS sayarsak üretim 1 adımda durmalı
    first = int(model.generate(start, 1, temperature=0)[0, -1])
    out = model.generate(start, 20, temperature=0, eos_id=first)
    assert out.shape == (3, 2)
    # top_k=0 filtreyi kapatır (çökmemeli)
    assert model.generate(start, 3, temperature=1.0, top_k=0).shape == (3, 4)


def test_get_batch_bounds(tmp_path):
    """Pencereler dosya sınırını aşmamalı, y = x'in bir kaydırılmışı olmalı ve son token da hedef olabilmeli."""
    from src.data import TokenDataset

    path = tmp_path / "toy.bin"
    data = np.arange(20, dtype=np.uint16)
    data.tofile(path)
    ds = TokenDataset(str(path), block_size=8, device="cpu")
    g = torch.Generator().manual_seed(0)
    seen_last = False
    for _ in range(200):
        x, y = ds.get_batch(4, g)
        assert torch.equal(y[:, :-1], x[:, 1:]) and torch.equal(y, x + 1)
        assert int(y.max()) <= 19
        seen_last |= bool((y[:, -1] == 19).any())
    assert seen_last, "dosyanın son tokenı hiç hedef olmadı (off-by-one)"
