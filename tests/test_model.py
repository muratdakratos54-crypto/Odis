"""Modelin temel doğruluk testleri (CPU'da saniyeler sürer).

    .venv\\Scripts\\python -m pytest -q tests

Neden bu testler?  Bir dil modeli hatalı olsa bile loss genellikle yine düşer; hatalar sessizdir.
Bu testler en yaygın sessiz hataları yakalar:
  * nedensellik (causal mask) sızıntısı -> model geleceği "görür", train loss sahte şekilde iyi olur
  * RoPE'nin yanlış uygulanması         -> konum bilgisi bozulur
  * yanlış başlatma                     -> başlangıç loss'u ln(V)'den çok farklı olur
  * gradyan akışı kopukluğu             -> model tek bir batch'i bile ezberleyemez
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import ModelConfig  # noqa: E402
from src.model import GPT, apply_rope, precompute_rope  # noqa: E402

CFG = ModelConfig(vocab_size=97, max_seq_len=32, d_model=64, n_layer=2, n_head=4, ffn_hidden=176)


def test_causality():
    """t konumunun logit'i, t'den sonraki tokenlar değişince DEĞİŞMEMELİ."""
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    x = torch.randint(0, CFG.vocab_size, (2, 16))
    x2 = x.clone()
    x2[:, 10:] = torch.randint(0, CFG.vocab_size, (2, 6))
    y = torch.zeros_like(x)  # targets verince tüm konumların logit'i döner
    with torch.no_grad():
        l1, _ = model(x, y)
        l2, _ = model(x2, y)
    assert torch.allclose(l1[:, :10], l2[:, :10], atol=1e-5), "gelecekten bilgi sızıyor!"
    assert not torch.allclose(l1[:, 10:], l2[:, 10:]), "girdi değişti ama çıktı değişmedi?"


def test_rope_relative():
    """RoPE sonrası <q_m, k_n> yalnızca m - n'e bağlı olmalı."""
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
    # döndürme normu korumalı
    assert torch.allclose(apply_rope(q, cos[7:8], sin[7:8]).norm(), q.norm(), atol=1e-5)


def test_initial_loss_near_uniform():
    """Rastgele başlatılmış model tüm tokenlara ~eşit olasılık vermeli: loss ≈ ln(V)."""
    torch.manual_seed(0)
    model = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (8, 32))
    # Hedefler girdiden BAĞIMSIZ olmalı: hedef=girdi verilirse weight tying yüzünden model
    # daha başlangıçta "şu anki token"a yüksek skor verir (h_t ~ e_x, logit = <h_t, e_x>)
    # ve loss yapay olarak ln(V)'nin altına düşer.
    y = torch.randint(0, CFG.vocab_size, (8, 32))
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.3, loss.item()


def test_overfit_single_batch():
    """Gradyanlar tüm yolu doğru akıyorsa model tek bir batch'i hızla ezberler."""
    torch.manual_seed(0)
    model = GPT(CFG)
    opt = model.configure_optimizer(3e-3, 0.0, (0.9, 0.95), "cpu")
    x = torch.randint(0, CFG.vocab_size, (4, 32))
    y = torch.roll(x, -1, dims=1)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, loss.item()


def test_noncontiguous_targets():
    """Veri yükleyici y'yi xy[:, 1:] dilimi olarak (bitişik olmayan görünüm) verir; loss bunu kabul etmeli."""
    model = GPT(CFG)
    xy = torch.randint(0, CFG.vocab_size, (2, 17))
    x, y = xy[:, :-1], xy[:, 1:]
    assert not y.is_contiguous()
    _, loss = model(x, y)
    assert torch.isfinite(loss)


def test_weight_tying_and_param_count():
    model = GPT(CFG)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    d, h, L, V = CFG.d_model, CFG.ffn_hidden, CFG.n_layer, CFG.vocab_size
    expected = V * d + L * (4 * d * d + 3 * d * h + 2 * d) + d
    assert model.num_params() == expected


def test_generate_stops_at_eos():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    out = model.generate(torch.zeros(3, 1, dtype=torch.long), 20, temperature=1.0, top_k=5)
    assert out.shape == (3, 21)
