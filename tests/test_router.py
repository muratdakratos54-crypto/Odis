"""Router dengeleme yolları: sigmoid gating ve aux-loss-free seçim bias'ı (CPU).

Aux loss'un aksine bias gradyanla değil, yük sapmasının işaretiyle güncellenir ve YALNIZCA seçime girer.
Bu iki şey testlerle sabitlenmiştir: bias gate değerini değiştirmemeli, görev gradyanına karışmamalı.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PRESETS, ModelConfig  # noqa: E402
from src.model import GPT, SwiGLU  # noqa: E402
from src.moe import MoE, load_balancing_loss, update_expert_biases  # noqa: E402

D, E, K, H = 32, 6, 2, 24
pytestmark = pytest.mark.filterwarnings("error::UserWarning")


def make_moe(gate: str = "softmax", bias_rate: float = 0.0, k: int = K, impl: str = "grouped") -> MoE:
    torch.manual_seed(0)
    return MoE(D, E, k, make_expert=lambda: SwiGLU(D, H), make_shared=lambda: SwiGLU(D, H),
               norm_topk_prob=True, impl=impl, gate=gate, bias_rate=bias_rate)


def routing_of(moe: MoE, x: torch.Tensor):
    """Router'ı testin kendi hesabıyla yeniden kur: skorlar, seçim ve gate değerleri."""
    h = x.reshape(-1, x.size(-1))
    logits = F.linear(h, moe.router.weight)
    scores = logits.sigmoid() if moe.gate == "sigmoid" else logits.softmax(-1)
    sel = (scores + moe.expert_bias) if moe.bias_rate > 0 else scores
    idx = sel.topk(moe.top_k, dim=-1).indices
    top = scores.gather(-1, idx)
    # modülle aynı kural: k=1'de normalize edilmez (bkz. MoE.forward)
    gates = top / top.sum(-1, keepdim=True) if moe.norm_topk_prob and moe.top_k > 1 else top
    return scores, idx, gates


def test_softmax_default_matches_reference_routing():
    """Varsayılan (softmax, bias yok) yol: seçim ve gate değerleri referans hesapla bit bit aynı."""
    moe = make_moe()
    x = torch.randn(3, 7, D)
    y, stats = moe(x)
    _, idx, gates = routing_of(moe, x)
    assert torch.equal(stats["topk_idx"], idx)
    # gate'ler çıktıya doğrudan giriyor: naif yeniden hesapla karşılaştır
    h = x.reshape(-1, D)
    ref = sum(gates[:, j : j + 1] * torch.stack([moe.experts[int(e)](h[t : t + 1])[0] for t, e in enumerate(idx[:, j])])
              for j in range(moe.top_k)) + moe.shared(h)
    assert torch.allclose(y.reshape(-1, D), ref, atol=1e-6)


def test_sigmoid_gate_selects_and_normalizes_sigmoid_scores():
    moe = make_moe(gate="sigmoid")
    x = torch.randn(4, 5, D)
    y, stats = moe(x)
    scores, idx, gates = routing_of(moe, x)
    assert torch.equal(stats["topk_idx"], idx)
    assert torch.allclose(gates.sum(-1), torch.ones(20), atol=1e-6)  # seçilenler normalize
    assert (scores > 0).all() and (scores < 1).all() and not torch.allclose(scores.sum(-1), torch.ones(20))
    # aux değeri (loglama için) normalize skorlarla hesaplanır
    counts = torch.bincount(idx.reshape(-1), minlength=E)
    assert torch.allclose(stats["aux_loss"], load_balancing_loss(scores / scores.sum(-1, keepdim=True), counts))


def test_bias_changes_selection_but_not_gate_values():
    """Bias yalnızca seçime girer: seçilen expert değişir, gate değerleri ham skorlardan gelir."""
    moe = make_moe(gate="sigmoid", bias_rate=0.001, k=1)
    x = torch.randn(2, 6, D)
    _, idx0, gates0 = routing_of(moe, x)
    with torch.no_grad():
        moe.expert_bias[E - 1] = 10.0  # bu expert her zaman seçilsin
    scores, idx1, gates1 = routing_of(moe, x)
    _, stats = moe(x)
    assert (idx1 == E - 1).all() and torch.equal(stats["topk_idx"], idx1)
    assert not torch.equal(idx0, idx1)
    # k=1'de normalize edilmez: gate = seçilen expert'in HAM sigmoid skoru; bias 10 olmasına rağmen gate <= 1
    assert torch.equal(gates1.squeeze(1), scores[:, E - 1]) and (gates1 <= 1).all()


def test_bias_is_buffer_without_gradient():
    moe = make_moe(gate="sigmoid", bias_rate=0.001)
    y, stats = moe(torch.randn(2, 5, D))
    (y.pow(2).sum() + stats["aux_loss"]).backward()
    assert moe.expert_bias.grad is None and not moe.expert_bias.requires_grad
    assert "expert_bias" not in {n for n, _ in moe.named_parameters()}
    assert moe.router.weight.grad.abs().sum() > 0


def test_update_bias_moves_against_load_and_resets_counter():
    moe = make_moe(gate="sigmoid", bias_rate=0.01)
    with torch.no_grad():  # expert 0 aşırı yüklü, 1 ideal payda, geri kalanlar az yüklü
        moe.bias_counts.zero_()
        moe.bias_counts[0] = 100.0
        moe.bias_counts[1] = 120.0 / E
        moe.bias_counts[2:] = (120.0 - 100.0 - 120.0 / E) / (E - 2)
    before = moe.expert_bias.clone()
    rng = moe.update_bias()
    delta = moe.expert_bias - before
    assert math.isclose(delta[0].item(), -0.01, rel_tol=1e-6)      # aşırı yüklü: bias düştü
    assert all(math.isclose(d.item(), +0.01, rel_tol=1e-6) for d in delta[2:])  # az yüklü: yükseldi
    assert math.isclose(rng, 0.02, rel_tol=1e-6) and moe.bias_counts.sum() == 0


def test_bias_counts_only_accumulate_in_training():
    moe = make_moe(gate="sigmoid", bias_rate=0.001)
    x = torch.randn(2, 8, D)
    moe.eval()
    with torch.no_grad():
        moe(x)
    assert moe.bias_counts.sum() == 0
    moe.train()
    moe(x)
    assert moe.bias_counts.sum() == 2 * 8 * K


def test_expert_bias_round_trips_and_old_checkpoints_load():
    """expert_bias modelin parçası (çıkarımda da seçimi etkiler) -> state_dict'e yazılır; eski checkpoint'lerde
    yok, o durumda sıfır kabul edilir (strict=True yine çalışır)."""
    moe = make_moe(gate="sigmoid", bias_rate=0.001)
    with torch.no_grad():
        moe.expert_bias.copy_(torch.arange(E, dtype=torch.float32))
    sd = moe.state_dict()
    assert "expert_bias" in sd and "bias_counts" not in sd
    fresh = make_moe(gate="sigmoid", bias_rate=0.001)
    fresh.load_state_dict(sd)
    assert torch.equal(fresh.expert_bias, moe.expert_bias)
    old = {k: v for k, v in sd.items() if k != "expert_bias"}  # v1/v2 biçimi
    fresh2 = make_moe(gate="sigmoid", bias_rate=0.001)
    fresh2.load_state_dict(old)
    assert torch.count_nonzero(fresh2.expert_bias) == 0


@pytest.mark.parametrize("impl", ["loop", "grouped"])
def test_sigmoid_and_bias_work_in_both_dispatch_paths(impl):
    a, b = make_moe("sigmoid", 0.001, impl="loop"), make_moe("sigmoid", 0.001, impl=impl)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        a.expert_bias.copy_(torch.linspace(-0.1, 0.1, E))
        b.expert_bias.copy_(a.expert_bias)
    x = torch.randn(3, 9, D)
    ya, sa = a(x)
    yb, sb = b(x)
    assert torch.equal(sa["topk_idx"], sb["topk_idx"])
    assert torch.allclose(ya, yb, rtol=1e-5, atol=1e-6)


def test_auxfree_balancing_recovers_from_collapse():
    """Mekanizmanın işe yaradığının işlevsel kanıtı: router tek expert'e kilitlenmiş bir modelde, yalnızca bias
    güncellemesi (gradyan yok) yükü ideal paya yaklaştırmalı; bias kapalıyken çöküş sürmeli."""
    torch.manual_seed(3)
    x = torch.randn(64, D).abs()  # girdiler pozitif: expert 0'ın sabit satırı her token için yüksek skor verir
    loads = {}
    for rate in (0.0, 0.05):
        moe = make_moe(gate="sigmoid", bias_rate=rate, k=1)
        with torch.no_grad():
            # Tokenların kendi tercihleri farklı olsun (rastgele router), ama expert 0 hepsini kazanacak kadar
            # önde başlasın: bias'sız çöküş, bias'lı dağılma beklenir.
            moe.router.weight.normal_(0, 0.3)
            moe.router.weight[0] = 1.0
        moe.train()
        for _ in range(200):
            with torch.no_grad():
                _, stats = moe(x.unsqueeze(0))
            update_expert_biases(moe)
        counts = stats["counts"].double()
        loads[rate] = (counts / counts.sum() * E).tolist()  # ideal pay = 1.0
    assert max(loads[0.0]) == pytest.approx(E, rel=1e-6)  # bias yok: hâlâ tek expert
    assert max(loads[0.05]) < 2.0 and min(loads[0.05]) > 0.3  # bias var: yük dağıldı


def test_preset_and_config_plumbing():
    mcfg, _ = PRESETS["moe-auxfree"]
    assert (mcfg.router_gate, mcfg.bias_update_rate, mcfg.aux_loss_coef) == ("sigmoid", 0.001, 0.0)
    model = GPT(mcfg)
    moes = [b.ffn for b in model.blocks if b.is_moe]
    assert len(moes) == 7 and all(m.gate == "sigmoid" and m.bias_rate == 0.001 for m in moes)
    # softmax presetlerinde bias kapalı ve buffer sıfır
    dense_moe = GPT(PRESETS["moe"][0])
    assert all(b.ffn.bias_rate == 0.0 and b.ffn.gate == "softmax" for b in dense_moe.blocks if b.is_moe)
    assert update_expert_biases(dense_moe) == 0.0


def test_train_step_updates_bias_and_logs_range(tmp_path):
    """Eğitim döngüsü bias'ı adım başına bir kez güncellemeli ve aralığı metrikte loglamalı."""
    import json

    import numpy as np

    import src.train as train

    rng = np.random.default_rng(0)
    for name, n in (("train.bin", 20_000), ("val.bin", 5_000)):
        rng.integers(0, 64, n, dtype=np.uint16).tofile(tmp_path / name)

    class Tok:
        vocab_size, bos_id, eos_id = 64, 1, 2

        def decode(self, ids):
            return ""

    old_argv, old_tok, old_cuda = sys.argv, train.load_tokenizer, torch.cuda.is_available
    sys.argv = ["train", "--preset", "tiny-moe", "--router_gate", "sigmoid", "--bias_update_rate", "0.01",
                "--aux_loss_coef", "0", "--data_dir", str(tmp_path), "--out_dir", str(tmp_path), "--run_name", "r",
                "--max_steps", "6", "--warmup_steps", "2", "--log_interval", "2", "--eval_interval", "100",
                "--eval_iters", "1", "--sample_tokens", "0", "--save_checkpoints", "false", "--batch_size", "4"]
    train.load_tokenizer, torch.cuda.is_available = lambda d: Tok(), lambda: False
    try:
        train.main()
    finally:
        sys.argv, train.load_tokenizer, torch.cuda.is_available = old_argv, old_tok, old_cuda
    recs = [json.loads(line) for line in open(tmp_path / "r" / "metrics.jsonl", encoding="utf-8")]
    trains = [r for r in recs if r["type"] == "train"]
    assert [r["step"] for r in trains] == [2, 4, 6]
    # 6 adımda her adım bias'a +-gamma ekler: aralık 0'dan büyür ve gamma'nın katlarında kalır
    assert all("bias_range" in r for r in trains)
    assert 0 < trains[0]["bias_range"] <= 6 * 0.01 + 1e-9 and trains[-1]["bias_range"] >= trains[0]["bias_range"]
