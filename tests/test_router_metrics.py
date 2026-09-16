"""scripts/router_analysis.py ölçütlerinin sentetik verilerle denetimi (CPU).

Her ölçütün cevabı bilinen uç durumlarda sabitlenir: mükemmel uzmanlaşma, rastgele routing, çökmüş routing,
birbirinin kopyası olan expert'ler. Ayrıca karşı-olgusal testlerde kullanılan replika forward'ın gerçek
MoE.forward ile bit bit aynı olduğu kanıtlanır (script bunu çalışırken de assert ediyor).
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.model import SwiGLU  # noqa: E402
from src.moe import MoE  # noqa: E402

_spec = importlib.util.spec_from_file_location("router_analysis", os.path.join(ROOT, "scripts", "router_analysis.py"))
ra = importlib.util.module_from_spec(_spec)
sys.modules["router_analysis"] = ra
_spec.loader.exec_module(ra)

D, E, K, H = 32, 4, 2, 24


def make_moe(gate: str = "softmax", bias_rate: float = 0.0, k: int = K, impl: str = "grouped") -> MoE:
    torch.manual_seed(0)
    return MoE(D, E, k, make_expert=lambda: SwiGLU(D, H), make_shared=lambda: SwiGLU(D, H),
               impl=impl, gate=gate, bias_rate=bias_rate)


def test_mutual_information_extremes():
    # Mükemmel uzmanlaşma: her token türü hep kendi expert'ine, yük eşit -> I = H(E) = log2(E), NMI = 1
    C = torch.zeros(E, E, dtype=torch.long)
    C.fill_diagonal_(100)
    mi = ra.mutual_information(C)
    assert mi["nmi"] == pytest.approx(1.0) and mi["mi_bits"] == pytest.approx(math.log2(E))
    assert mi["utilization_entropy"] == pytest.approx(1.0)
    # Rastgele/bağımsız routing: her token her expert'e eşit -> I = 0, ama kullanım entropisi 1 (dengeli!)
    mi = ra.mutual_information(torch.full((8, E), 50, dtype=torch.long))
    assert mi["mi_bits"] == pytest.approx(0.0, abs=1e-12) and mi["nmi"] == pytest.approx(0.0, abs=1e-12)
    assert mi["utilization_entropy"] == pytest.approx(1.0)
    # Çöküş: her şey expert 0'a -> H(E) = 0, NMI tanımsız yerine 0 döner, kullanım entropisi 0
    C = torch.zeros(8, E, dtype=torch.long)
    C[:, 0] = 10
    mi = ra.mutual_information(C)
    assert mi["h_expert_bits"] == pytest.approx(0.0) and mi["nmi"] == 0.0
    assert mi["utilization_entropy"] == pytest.approx(0.0)
    assert ra.mutual_information(torch.zeros(4, E, dtype=torch.long))["nmi"] == 0.0


def test_routing_consistency_and_min_count():
    C = torch.zeros(3, E, dtype=torch.long)
    C[0, 1] = 100                      # hep aynı expert -> tutarlı
    C[1] = torch.tensor([25, 25, 25, 25])  # eşit dağılmış -> 1/E
    C[2, 0] = 10                       # nadir token: min_count altında, hesaba girmez
    assert ra.routing_consistency(C[:1], min_count=50) == pytest.approx(1.0)
    assert ra.routing_consistency(C[1:2], min_count=50) == pytest.approx(1.0 / E)
    assert ra.routing_consistency(C, min_count=50) == pytest.approx((100 + 25) / 200)
    assert math.isnan(ra.routing_consistency(C[2:], min_count=50))


def test_expert_output_diversity_identical_vs_different():
    moe = make_moe()
    h = torch.randn(64, D)
    with torch.no_grad():  # hepsini expert 0'ın kopyası yap -> çeşitlilik ~0
        for e in moe.experts[1:]:
            e.load_state_dict(moe.experts[0].state_dict())
    assert ra.expert_output_diversity(moe, h) == pytest.approx(0.0, abs=1e-6)
    fresh = make_moe()  # bağımsız başlatılmış expert'ler -> belirgin çeşitlilik
    assert ra.expert_output_diversity(fresh, h) > 0.3


def test_embedding_coherence_clustered_vs_random():
    V = 64
    C = torch.zeros(V, E, dtype=torch.long)
    for t in range(V):  # token türleri expert'lere bloklar halinde dağılmış
        C[t, t % E] = 100
    torch.manual_seed(0)
    centers = torch.randn(E, D) * 3
    clustered = torch.stack([centers[t % E] + 0.05 * torch.randn(D) for t in range(V)])
    random_emb = torch.randn(V, D)
    assert ra.embedding_coherence(C, clustered, min_count=50) > 0.5
    assert abs(ra.embedding_coherence(C, random_emb, min_count=50)) < 0.15


def test_top_lift_finds_exclusive_tokens():
    C = torch.zeros(5, E, dtype=torch.long)
    C[0, 2] = 500            # yalnızca expert 2'ye giden token
    C[1] = torch.tensor([100, 100, 100, 100])
    C[2:] = 100
    lifts = ra.top_lift(C, lambda i: f"t{i}", min_count=200, top=2)
    assert lifts[2][0].startswith("t0(") and float(lifts[2][0].split("(")[1][:-1]) > 1.5
    assert all(not s.startswith("t0(") for e in (0, 1, 3) for s in lifts[e])


@pytest.mark.parametrize("gate,bias", [("softmax", 0.0), ("sigmoid", 0.0), ("sigmoid", 0.001)])
def test_route_modes(gate, bias):
    moe = make_moe(gate=gate, bias_rate=bias)
    if bias:
        with torch.no_grad():
            moe.expert_bias.copy_(torch.tensor([0.5, 0.0, 0.0, -0.5]))
    h = torch.randn(40, D)
    logits = torch.nn.functional.linear(h, moe.router.weight)
    scores, sel, idx, gates = ra.route(moe, logits)
    # learned: modülün kendi seçimiyle aynı
    _, stats = moe(h.unsqueeze(0))
    assert torch.equal(stats["topk_idx"], idx)
    assert torch.allclose(gates.sum(-1), torch.ones(40), atol=1e-6)  # k>1: normalize
    # next: top-k ile kesişmeyen, sıradaki k expert
    _, _, nxt, _ = ra.route(moe, logits, "next")
    assert nxt.shape == idx.shape and not (nxt.unsqueeze(-1) == idx.unsqueeze(1)).any()
    # random: token başına k farklı expert
    g = torch.Generator().manual_seed(0)
    _, _, rnd, _ = ra.route(moe, logits, "random", g)
    assert rnd.shape == idx.shape and all(len(set(r.tolist())) == K for r in rnd)
    # nobias: bias yok sayılır
    _, _, nb, _ = ra.route(moe, logits, "nobias")
    assert torch.equal(nb, scores.topk(K, dim=-1).indices)
    if bias:
        assert not torch.equal(nb, idx)  # bias gerçekten seçimi değiştiriyor


@pytest.mark.parametrize("gate,bias", [("softmax", 0.0), ("sigmoid", 0.001)])
def test_replica_forward_is_bitexact(gate, bias):
    """Karşı-olgusal testler MoE.forward'ın kopyasını kullanıyor: 'learned' kipinde gerçekle bit bit aynı olmalı."""
    moe = make_moe(gate=gate, bias_rate=bias)
    if bias:
        with torch.no_grad():
            moe.expert_bias.copy_(torch.linspace(-0.2, 0.2, E))
    x = torch.randn(2, 9, D)
    y_real, s_real = moe(x)
    real_forward = MoE.forward
    try:
        MoE.forward = ra.moe_forward_variant("learned")
        y_rep, s_rep = moe(x)
    finally:
        MoE.forward = real_forward
    assert torch.equal(y_real, y_rep) and torch.equal(s_real["topk_idx"], s_rep["topk_idx"])
    assert torch.equal(s_real["counts"], s_rep["counts"])
