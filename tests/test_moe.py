"""MoE katmanı testleri (CPU).

En kritik test: hızlı sıralama tabanlı dispatch/combine, her tokenı tek tek expert'lere gönderen
naif referans uygulamayla BİREBİR aynı sonucu vermeli. MoE kodundaki indeksleme hataları loss'u
patlatmaz, sadece modeli sessizce kötüleştirir; bu yüzden bu eşdeğerliği açıkça kanıtlıyoruz.
Aynı testler iki expert yolunda da ("loop" ve "grouped") koşar; ayrıca iki yol birbirine karşı
küçük, deterministik bir CPU kurulumunda bit bit karşılaştırılır.
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
from src.config import ModelConfig  # noqa: E402
from src.model import GPT, SwiGLU  # noqa: E402
from src.moe import MoE, _CastGatherRows, grouped_mm_supported, load_balancing_loss, router_z_loss  # noqa: E402

# "grouped" yol desteklenmeyen boyutta sessizce "loop"a düşerse yalnızca bir UserWarning verir; o durumda
# grouped-vs-loop testleri loop'u loop'la karşılaştırıp boşuna geçerdi. Uyarıyı hata say (fallback testi
# pytest.warns ile bunu kendisi yakalar).
pytestmark = pytest.mark.filterwarnings("error::UserWarning")

D, E, K, H = 32, 6, 2, 24
IMPLS = pytest.mark.parametrize("impl", ["loop", "grouped"])


def make_moe(shared: bool = True, norm: bool = True, k: int = K, impl: str = "grouped", hidden: int = H,
             d: int = D) -> MoE:
    torch.manual_seed(0)
    return MoE(d, E, k, make_expert=lambda: SwiGLU(d, hidden),
               make_shared=(lambda: SwiGLU(d, hidden)) if shared else None, norm_topk_prob=norm, impl=impl)


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


@pytest.fixture
def one_thread():
    """Bit bit karşılaştırmalar tek iş parçacığında: çok iş parçacıklı eleman bazlı çekirdekler tensörü
    boyuna göre farklı bölüp farklı kod yollarından hesaplar (bkz. test_grouped_matches_loop_bitexact)."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture
def grouped_calls(monkeypatch):
    """torch._grouped_mm çağrı sayacı: "grouped" yolun gerçekten gruplu çekirdeği çalıştırdığını doğrulamak için."""
    calls = [0]
    real = torch._grouped_mm

    def counting(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "_grouped_mm", counting)
    return calls


@IMPLS
def test_dispatch_matches_naive(impl):
    for shared in (True, False):
        for norm in (True, False):
            for k in (1, K, E):  # top-1, top-k ve "hepsi" uç durumları
                moe = make_moe(shared, norm, k, impl)
                x = torch.randn(3, 7, D)
                y, _ = moe(x)
                assert torch.allclose(y, naive_forward(moe, x), atol=1e-5), (shared, norm, k)


@pytest.mark.parametrize("impl, per_forward", [("grouped", 3), ("loop", 0)])
def test_selected_impl_runs_its_kernel(grouped_calls, impl, per_forward):
    """grouped: MoE ileri yönü başına 3 gruplu çarpım (gate, up, down); loop: hiç."""
    moe = make_moe(impl=impl)
    moe(torch.randn(2, 5, D))
    assert grouped_calls[0] == per_forward


@IMPLS
def test_top1_router_gets_task_gradient(impl):
    """k=1'de normalize edilen gate p/p = 1 sabit olurdu ve router görev loss'undan gradyan alamazdı."""
    moe = make_moe(shared=False, norm=True, k=1, impl=impl)
    y, _ = moe(torch.randn(2, 5, D))
    y.pow(2).sum().backward()  # yalnızca "görev" loss'u, aux yok
    assert moe.router.weight.grad.abs().sum() > 1e-3


@IMPLS
def test_gradients_match_naive(impl):
    """Geri yayılım da aynı olmalı (router dahil): index_add_/argsort yolu gradyanı bozmamalı."""
    moe = make_moe(impl=impl)
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


@IMPLS
@pytest.mark.parametrize("favorites", [(0, 1), (E - 2, E - 1)])  # boş dilimler sonda / başta
def test_empty_expert_ok(impl, favorites):
    """Hiç token almayan expert çökme yaratmamalı (dropless routing'de sık görülür). Grouped yolda boş dilimler
    offs'ta tekrarlanan değerlerdir; baştaki ve sondaki boş dilimler ayrı uç durumlar."""
    moe = make_moe(impl=impl)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[favorites[0]] = 5.0  # tüm tokenlar bu iki expert'e
        moe.router.weight[favorites[1]] = 4.0
    x = torch.randn(2, 4, D).abs()
    y, stats = moe(x)
    assert (stats["counts"] > 0).tolist() == [e in favorites for e in range(E)]
    y.sum().backward()
    assert torch.allclose(y, naive_forward(moe, x), atol=1e-5)
    # Token almayan expert'in gradyanı None değil sıfır (iki yolda da): AdamW onu atlamaz, aynı davranır
    assert all(torch.count_nonzero(moe.experts[e].w_gate.weight.grad) == 0 for e in range(E) if e not in favorites)


# --- "grouped" ile "loop" birbirine karşı -------------------------------------------------------------------

def _run(moe: MoE, x: torch.Tensor, g: torch.Tensor):
    """Çıktı, istatistikler, girdi gradyanı ve tüm parametre gradyanları (aux ve z loss dahil)."""
    x = x.clone().requires_grad_(True)
    y, stats = moe(x)
    loss = (y.float() * g).sum() + 0.3 * stats["aux_loss"] + 0.1 * stats["z_loss"]
    loss.backward()
    grads = {n: p.grad.clone() for n, p in moe.named_parameters() if p.grad is not None}
    return y.detach(), stats, x.grad, grads


def _loop_and_grouped(**kw):
    loop, grouped = make_moe(impl="loop", **kw), make_moe(impl="grouped", **kw)
    grouped.load_state_dict(loop.state_dict())
    return loop, grouped


def _assert_bitexact(r1, r2, n_params: int):
    (y1, s1, gx1, gr1), (y2, s2, gx2, gr2) = r1, r2
    assert torch.equal(s1["topk_idx"], s2["topk_idx"]) and torch.equal(s1["counts"], s2["counts"])
    assert torch.equal(y1, y2)
    assert torch.equal(s1["aux_loss"], s2["aux_loss"]) and torch.equal(s1["z_loss"], s2["z_loss"])
    assert torch.equal(gx1, gx2)
    assert gr1.keys() == gr2.keys() and len(gr1) == n_params
    for n in gr1:
        assert torch.equal(gr1[n], gr2[n]), n


@pytest.mark.parametrize("bt", [(3, 13), (8, 13)])
@pytest.mark.parametrize("k", [1, K, E])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("norm", [True, False])
def test_grouped_matches_loop_bitexact(one_thread, grouped_calls, bt, k, shared, norm):
    """Küçük, deterministik CPU kurulumunda iki yol çıktıda, istatistiklerde ve TÜM gradyanlarda bit bit aynı.

    Kurulumun iki şartı ölçümle belirlendi (torch 2.14, MKL, AVX2, 20 çekirdek):
      * tek iş parçacığı: çok iş parçacığında SiLU büyük tensörü (grouped) ve küçük dilimleri (loop) farklı
        bölüp farklı yollardan hesaplıyor; sonuçlar son bitte ayrışabiliyor. (8, 13) durumu bu şartın gerçekten
        gerektiğini gösterecek kadar büyük: bu makinede 20 iş parçacığıyla 12 kombinasyondan 4'ünde bit bit tutmadı.
      * expert gizli boyutu 8'in katı ve >= 256: SiLU'nun CPU çekirdeği ~256 elemandan kısa tensörleri ve
        uzunluğu 8'in katı olmayanların son elemanlarını skaler koddan geçiriyor; bu SIMD sonucuyla bit bit aynı
        değil. Gizli boyut 256 iken tek satırlık bir dilim bile SIMD yolunda kalır.
    Matris çarpımları (mm / _grouped_mm) bu ölçümlerde her boyutta bit bit aynıydı. Bu şartlar dışında iki yol
    yalnızca yakın (allclose) olmalı: bkz. test_grouped_matches_loop_allclose_any_shape.
    """
    loop, grouped = _loop_and_grouped(k=k, shared=shared, norm=norm, hidden=256)
    torch.manual_seed(1)
    x, g = torch.randn(*bt, D), torch.randn(*bt, D)
    r1 = _run(loop, x, g)
    assert grouped_calls[0] == 0
    r2 = _run(grouped, x, g)
    assert grouped_calls[0] == 3
    _assert_bitexact(r1, r2, len(list(loop.parameters())))


def test_grouped_matches_loop_under_autocast_bitexact(one_thread, grouped_calls):
    """bf16 autocast altında da bit bit aynı: gate ve up kollarının girdi gradyanları grouped yolda da "loop"taki
    gibi float32'de toplanmalı (_CastGatherRows'un iki çıktısı). Tek çıktılı ilk sürüm burada ~1 ulp ayrışıyordu."""
    loop, grouped = _loop_and_grouped(hidden=256)
    torch.manual_seed(3)
    x, g = torch.randn(3, 13, D), torch.randn(3, 13, D)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        r1 = _run(loop, x, g)
        r2 = _run(grouped, x, g)
    assert grouped_calls[0] == 3
    assert r1[0].dtype == r2[0].dtype == torch.bfloat16 and r2[2].dtype == torch.float32
    assert all(v.dtype == torch.float32 for v in r2[3].values())
    _assert_bitexact(r1, r2, len(list(loop.parameters())))


def test_grouped_matches_loop_allclose_any_shape(grouped_calls):
    """Bit bit şartları sağlanmayan boyutlarda ve varsayılan iş parçacığı sayısında: yakın olmalı."""
    loop, grouped = _loop_and_grouped(hidden=H)  # H = 24: dilimler skaler SiLU yoluna düşer
    torch.manual_seed(2)
    x, g = torch.randn(4, 11, D), torch.randn(4, 11, D)
    y1, s1, gx1, gr1 = _run(loop, x, g)
    y2, s2, gx2, gr2 = _run(grouped, x, g)
    assert grouped_calls[0] == 3
    assert torch.equal(s1["topk_idx"], s2["topk_idx"])  # routing, expert hesabından önce: her zaman aynı
    assert torch.allclose(y1, y2, rtol=1e-5, atol=1e-6) and torch.allclose(gx1, gx2, rtol=1e-5, atol=1e-6)
    for n in gr1:
        assert torch.allclose(gr1[n], gr2[n], rtol=1e-5, atol=1e-6), n


def test_cast_gather_rows_accumulates_in_float32():
    """Her token k kez, iki kol (iki çıktı) ayrı gradyanla: toplam float32'de, bf16 yuvarlaması olmadan."""
    torch.manual_seed(0)
    N, C, k = 16, 32, 3
    h = torch.randn(N, C, requires_grad=True)
    idx = torch.randperm(N * k) // k  # her satır k kez, karışık sırada
    a, b = _CastGatherRows.apply(h, idx, torch.bfloat16)
    assert a.dtype == torch.bfloat16 and torch.equal(a, h.detach().to(torch.bfloat16).index_select(0, idx))
    assert a.data_ptr() == b.data_ptr()  # ikinci çıktı kopya değil
    ga, gb = torch.randn(N * k, C).to(torch.bfloat16), torch.randn(N * k, C).to(torch.bfloat16)
    torch.autograd.backward([a, b], [ga, gb])
    ref = torch.zeros(N, C).index_add_(0, idx, ga.float() + gb.float())
    assert torch.equal(h.grad, ref)
    bf16_sum = torch.zeros(N, C, dtype=torch.bfloat16).index_add_(0, idx, ga + gb).float()
    assert not torch.equal(h.grad, bf16_sum)  # test gerçekten ayırt ediyor: bf16'da toplamak farklı sonuç verir


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_grouped_mm_supported_matches_kernel(dtype):
    """grouped_mm_supported, çekirdeğin gerçekten kabul ettiği boyutlarla birebir aynı olmalı (fwd + bwd):
    yanlış True -> eğitim çöker; yanlış False -> sessizce yavaş yola düşülür. torch değişirse burası yakalar."""
    for C in range(1, 21):
        for Hd in range(1, 21):
            try:
                x = torch.randn(7, C, dtype=dtype, requires_grad=True)
                w = torch.randn(3, Hd, C, dtype=dtype, requires_grad=True)
                y = torch._grouped_mm(x, w.transpose(1, 2), offs=torch.tensor([2, 2, 7], dtype=torch.int32))
                y.backward(torch.randn_like(y))
                works = True
            except RuntimeError:
                works = False
            assert works == grouped_mm_supported(dtype, C, Hd), (dtype, C, Hd)


@pytest.mark.parametrize("d, hidden", [(D, 44), (36, 32)])  # gizli boyut / d_model 8'in katı değil
def test_grouped_falls_back_to_loop_when_kernel_unsupported(grouped_calls, d, hidden):
    """bf16'da 8'in katı olmayan boyut çekirdekçe desteklenmez: uyarı + loop ile aynı sonuç, gruplu çağrı yok."""
    loop, grouped = _loop_and_grouped(hidden=hidden, d=d)
    loop, grouped = loop.to(torch.bfloat16), grouped.to(torch.bfloat16)
    x = torch.randn(2, 5, d, dtype=torch.bfloat16)
    with pytest.warns(UserWarning, match="loop"):
        y2, _ = grouped(x)
    y1, _ = loop(x)
    assert torch.equal(y1, y2) and grouped_calls[0] == 0


def test_grouped_falls_back_under_autocast_with_float32_params(grouped_calls):
    """Eğitimdeki kurulum: float32 parametre + bf16 autocast. 44 float32'de hizalı (4'ün katı) ama bf16'da değil:
    karar parametrenin değil autocast'in dtype'ına göre verilmeli, yoksa çekirdek eğitimde çökerdi."""
    loop, grouped = _loop_and_grouped(hidden=44)
    torch.manual_seed(4)
    x, g = torch.randn(2, 5, D), torch.randn(2, 5, D)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        r1 = _run(loop, x, g)
        with pytest.warns(UserWarning, match="loop"):
            r2 = _run(grouped, x, g)
    assert grouped_calls[0] == 0
    assert torch.equal(r1[0], r2[0]) and torch.equal(r1[2], r2[2])


def test_whole_model_bf16_grouped_matches_loop(grouped_calls):
    """Model bütünüyle bf16'ya çevrilince (autocast'siz, desteklenen boyutlarla) grouped yol çalışmalı ve loop'la
    aynı logit'leri vermeli (generate dahil)."""
    cfg = ModelConfig(vocab_size=97, max_seq_len=16, d_model=32, n_layer=3, n_head=4, ffn_hidden=88,
                      n_experts=4, top_k=2, expert_hidden=48, n_shared_experts=1, first_dense_layers=1)
    outs = []
    for impl in ("loop", "grouped"):
        torch.manual_seed(0)
        model = GPT(replace(cfg, moe_impl=impl)).to(torch.bfloat16).eval()
        x = torch.randint(0, 97, (2, 16))
        with torch.no_grad():
            logits, _, _ = model(x, x)
        outs.append(logits)
        assert model.generate(x[:, :1], 3).shape == (2, 4)
    assert grouped_calls[0] > 0
    assert torch.equal(outs[0], outs[1])


def test_training_steps_bitexact_cpu(one_thread, grouped_calls):
    """Model düzeyinde: aynı seed'le kurulan iki model (loop / grouped) 3 AdamW adımı boyunca bit bit aynı loss
    ve parametreler üretmeli (bit bit şartları için bkz. test_grouped_matches_loop_bitexact). ModelConfig.moe_impl'in
    gerçekten MoE katmanlarına ulaştığı çağrı sayısıyla doğrulanır."""
    cfg = ModelConfig(vocab_size=97, max_seq_len=16, d_model=32, n_layer=3, n_head=4, ffn_hidden=88,
                      n_experts=4, top_k=2, expert_hidden=256, n_shared_experts=1, first_dense_layers=1)
    torch.manual_seed(0)
    x = torch.randint(0, 97, (4, 16))
    y = torch.randint(0, 97, (4, 16))
    runs = []
    for impl in ("loop", "grouped"):
        torch.manual_seed(0)  # parametre düzeni aynı olduğundan aynı seed -> bit bit aynı başlatma
        model = GPT(replace(cfg, moe_impl=impl))
        assert all(b.ffn.impl == impl for b in model.blocks if b.is_moe)
        opt = model.configure_optimizer(1e-2, 0.1, (0.9, 0.95), "cpu")
        losses = []
        grouped_calls[0] = 0
        for _ in range(3):
            _, loss, _ = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        n_moe = sum(b.is_moe for b in model.blocks)
        assert grouped_calls[0] == (3 * 3 * n_moe if impl == "grouped" else 0)
        runs.append((losses, {n: p.detach().clone() for n, p in model.named_parameters()}))
    (l1, p1), (l2, p2) = runs
    assert all(torch.equal(a, b) for a, b in zip(l1, l2)), (l1, l2)
    assert l1[-1] < l1[0]  # gerçekten öğreniyor (sabit bir şeyi karşılaştırmıyoruz)
    for n in p1:
        assert torch.equal(p1[n], p2[n]), n
