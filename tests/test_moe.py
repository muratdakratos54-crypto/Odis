"""MoE katmanı testleri (CPU).

En kritik test: hızlı sıralama tabanlı dispatch/combine, her tokenı tek tek expert'lere gönderen
naif referans uygulamayla BİREBİR aynı sonucu vermeli. MoE kodundaki indeksleme hataları loss'u
patlatmaz, sadece modeli sessizce kötüleştirir; bu yüzden bu eşdeğerliği açıkça kanıtlıyoruz.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import SwiGLU  # noqa: E402
from src.moe import MoE, load_balancing_loss, router_z_loss  # noqa: E402

D, E, K, H = 32, 6, 2, 24


def make_moe(shared: bool = True, norm: bool = True, k: int = K) -> MoE:
    torch.manual_seed(0)
    return MoE(D, E, k, make_expert=lambda: SwiGLU(D, H),
               make_shared=(lambda: SwiGLU(D, H)) if shared else None, norm_topk_prob=norm)


def counts_of(idx: torch.Tensor) -> torch.Tensor:
    return torch.bincount(idx.reshape(-1), minlength=E)


def naive_forward(moe: MoE, x: torch.Tensor) -> torch.Tensor:
    """Referans: her token için döngü, seçilen expert'leri tek tek çalıştır ve ağırlıklı topla."""
    h = x.reshape(-1, x.size(-1))
    probs = F.softmax(moe.router(h), dim=-1)
    topk_p, topk_idx = probs.topk(moe.top_k, dim=-1)
    normalize = moe.norm_topk_prob and moe.top_k > 1
    gates = topk_p / topk_p.sum(-1, keepdim=True) if normalize else topk_p
    y = torch.zeros_like(h)
    for t in range(h.size(0)):
        for j in range(moe.top_k):
            e = int(topk_idx[t, j])
            y[t] += gates[t, j] * moe.experts[e](h[t : t + 1])[0]
    if moe.shared is not None:
        y = y + moe.shared(h)
    return y.view_as(x)


def test_dispatch_matches_naive():
    for shared in (True, False):
        for norm in (True, False):
            for k in (1, K, E):  # top-1, top-k ve "hepsi" uç durumları
                moe = make_moe(shared, norm, k)
                x = torch.randn(3, 7, D)
                y, _ = moe(x)
                assert torch.allclose(y, naive_forward(moe, x), atol=1e-5), (shared, norm, k)


def test_top1_router_gets_task_gradient():
    """k=1'de normalize edilen gate p/p = 1 sabit olurdu ve router görev loss'undan gradyan alamazdı."""
    moe = make_moe(shared=False, norm=True, k=1)
    y, _ = moe(torch.randn(2, 5, D))
    y.pow(2).sum().backward()  # yalnızca "görev" loss'u, aux yok
    assert moe.router.weight.grad.abs().sum() > 1e-3


def test_gradients_match_naive():
    """Geri yayılım da aynı olmalı (router dahil): index_add_/argsort yolu gradyanı bozmamalı."""
    moe = make_moe()
    x = torch.randn(2, 5, D)
    y, _ = moe(x)
    y.pow(2).sum().backward()
    g_fast = {n: p.grad.clone() for n, p in moe.named_parameters() if p.grad is not None}
    moe.zero_grad()
    naive_forward(moe, x).pow(2).sum().backward()
    for n, p in moe.named_parameters():
        if p.grad is not None:
            assert torch.allclose(g_fast[n], p.grad, atol=1e-5), n
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0


def test_counts_sum_to_n_times_k():
    moe = make_moe()
    x = torch.randn(4, 9, D)
    _, stats = moe(x)
    assert stats["counts"].sum().item() == 4 * 9 * K


def test_aux_loss_range():
    N = 60
    # Mükemmel denge: olasılıklar düzgün, atamalar expert'lere eşit dağılmış -> 1.0
    probs = torch.full((N, E), 1.0 / E)
    idx = torch.arange(N * K).remainder(E).view(N, K)
    assert math.isclose(load_balancing_loss(probs, counts_of(idx)).item(), 1.0, rel_tol=1e-6)
    # Tam çöküş: her token hep expert 0 ve 1'e gidiyor, olasılık kütlesi de orada -> E/k
    probs = torch.zeros(N, E)
    probs[:, :K] = 1.0 / K
    idx = torch.arange(K).expand(N, K)
    assert math.isclose(load_balancing_loss(probs, counts_of(idx)).item(), E / K, rel_tol=1e-6)


def test_aux_loss_gradient_pushes_away_from_overloaded_expert():
    """Aşırı yüklü expert'in logit'i aux loss gradyanıyla AŞAĞI itilmeli (grad > 0)."""
    logits = torch.zeros(10, E)
    logits[:, 0] = 2.0  # expert 0 herkesin favorisi
    logits.requires_grad_(True)
    probs = logits.softmax(-1)
    idx = probs.topk(K, dim=-1).indices
    load_balancing_loss(probs, counts_of(idx)).backward()
    assert logits.grad[:, 0].mean() > 0  # gradient descent: logit -= lr * grad -> azalır
    assert logits.grad[:, 0].mean() > logits.grad[:, 1:].mean()


def test_z_loss_penalizes_large_logits():
    small = router_z_loss(torch.randn(8, E) * 0.1)
    large = router_z_loss(torch.randn(8, E) * 10)
    assert large > small


def test_empty_expert_ok():
    """Hiç token almayan expert çökme yaratmamalı (dropless routing'de sık görülür)."""
    moe = make_moe()
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] = 5.0  # hepsi expert 0'a (ve ikinci olarak eşit skorlulardan birine)
    x = torch.randn(2, 4, D).abs()
    y, stats = moe(x)
    assert (stats["counts"] == 0).sum() >= E - K
    y.sum().backward()
    assert torch.allclose(y, naive_forward(moe, x), atol=1e-5)
