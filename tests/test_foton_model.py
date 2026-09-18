"""FOTON v0.2 kollarının ZORUNLU DOĞRULAMASI (sahibin 06 talimatı) + modelin ölçümü üreten davranışları.

Tasarımın tamamı şu üç doğrulamaya dayanıyor:

  1. Her modülün token başına ORTALAMA attention bağlantısı NEDENSEL MASKE ALTINDA, T = 512'de FİİLEN
     sayılır: üçü de 128.4 ± 0.4. Teorik formülle değil, modelin `scaled_dot_product_attention`'a verdiği
     maske sayılarak. İlk tasarımda tam bu adım atlandığı için 192 / 128 / 88 farkı gözden kaçmıştı.
  2. Router NEDENSELDİR: k = 64'ten sonraki hiçbir token modül seçimini etkileyemez (prefix sabit, kuyruk
     değişir -> seçim değişmez).
  3. Her modülün attention grafiği tasarlandığı desendedir (yerel / dilated / karışık) — maskeye bakarak
     değil, attention katmanının ÇIKTISININ girdiye fonksiyonel bağımlılığı ölçülerek.

Beklenen desenler bu dosyada `src/attn_patterns`'ten BAĞIMSIZ, elle yazılmış formüllerle kurulur: aynı
kaynaktan türetilen iki şeyi karşılaştırmak bir şey kanıtlamaz.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src import attn_patterns  # noqa: E402
from src.config import PRESETS, ModelConfig  # noqa: E402
from src.model import FOTON_ARMS, GPT, CausalSelfAttention  # noqa: E402

T, K = 512, 64
# kampanyanın T / k / katman bölüşümü / head sayısı, ama dar model: CPU'da saniyeler içinde koşsun
SMALL = ModelConfig(vocab_size=97, max_seq_len=T, d_model=64, n_layer=8, n_head=8, ffn_hidden=128, foton="x")


def _diverse(model: GPT, B: int, seed: int) -> torch.Tensor:
    """Yönlendirmesi ÇEŞİTLİ bir batch. Init'te 64 rastgele token'ın ortalaması diziler arasında neredeyse
    aynıdır ve top-1 hepsini tek modüle yollar (top-1 yönlendirmenin bilinen başlangıç davranışı; V6/V7'nin
    konusu). Seçimin gerçekten girdiye bağlı olduğu testler için prefix'i dizi başına TEK bir token'dan
    kurup router ağırlığını büyütüyoruz: havuzlanan vektörler belirgin biçimde ayrışır."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 97, (B, T), generator=g)
    x[:, :K] = torch.arange(B)[:, None] * 7 + 1
    with torch.no_grad():
        model.foton_router.weight.mul_(200.0)
    return x


def _expected(kind: str) -> np.ndarray:
    """[T, T] beklenen desen — ELLE yazılmış tanımdan (attn_patterns kullanılmaz). True = (i, j) bağlı."""
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    off = i - j
    if kind == "local150":
        return (off >= 0) & (off < 150)
    if kind == "local256":
        return (off >= 0) & (off < 256)
    if kind == "dilated2":                                   # her 2. pozisyon, en çok 256 tane, kendisi dahil
        return (off >= 0) & (off % 2 == 0) & (off // 2 < 256)
    if kind == "strided4":
        return (off >= 0) & (off % 4 == 0) & (off // 4 < 256)
    if kind == "full":
        return off >= 0
    raise KeyError(kind)


# ------------------------------------------------------------------ ZORUNLU DOĞRULAMA 1: bağlantıları SAY

def test_counted_connections_per_module_are_128_4_pm_0_4_under_the_causal_mask_at_T512():
    """Gerçek preset'lerin (`foton-x`, `foton-h`) kurduğu modelde, SDPA'ya giden maske buffer'ı SAYILIR."""
    report = {}
    for arm in ("foton-x", "foton-h"):
        torch.manual_seed(0)
        model = GPT(replace(PRESETS[arm][0], vocab_size=64))          # vocab küçük: yalnızca kurulum hızı
        assert model.cfg.max_seq_len == T and model.cfg.n_head == 8 and model.cfg.foton_prefix_k == K
        for m, layers in enumerate(model.foton_modules):
            assert len(layers) == 2
            for blk in layers:
                mask = blk.attn.attn_mask
                assert mask.dtype == torch.bool and tuple(mask.shape) == (8, T, T)
                assert not torch.triu(mask, diagonal=1).any(), "nedensel değil"
                counted = mask.sum(dim=-1).double().mean().item()      # head ve pozisyon ortalaması, SAYILARAK
                assert abs(counted - 128.4) <= 0.4, f"{arm} modül {m}: {counted}"
                report[(arm, m)] = counted
    assert report[("foton-x", 0)] == pytest.approx(128.173828125, abs=1e-12)     # A: yerel W=150
    assert report[("foton-x", 1)] == pytest.approx(128.5, abs=1e-12)             # B: dilated r=2 W=256
    assert report[("foton-x", 2)] == pytest.approx(128.375, abs=1e-12)           # C: yarım yerel + yarım strided
    assert {report[("foton-h", m)] for m in range(3)} == {report[("foton-x", 0)]}  # H: üçü de A
    assert max(report.values()) - min(report.values()) < 0.33                    # kalan eşitsizlik
    # modelin kendi raporu da aynı sayıları veriyor (rapora giren bu)
    torch.manual_seed(0)
    got = GPT(replace(PRESETS["foton-x"][0], vocab_size=64)).foton_connection_counts()
    assert got == pytest.approx({"M0:A": 128.173828125, "M1:B": 128.5, "M2:C": 128.375}, abs=1e-12)
    # baseline'ların gördüğü: 256.5 — kalan eşitsizliğin YÖNÜ FOTON'un aleyhine
    assert _expected("full").sum(axis=1).mean() == 256.5


def test_the_stale_ox1_design_cannot_be_built():
    """İKİNCİ KİLİT. Ön-kaydın O-X1 bölümünde kalan ESKİ tasarım (A bitişik 256 / B dilated 256 / C 128+128)
    nedensel maske altında 192.25 / 128.50 / 88.31 bağlantı verir — reddedilen compute confound'un kendisi.
    O desenlerle model KURULAMAZ: `verify_foton_masks()` her kurulumda sayar ve patlar."""
    old = {"A": attn_patterns.local(256), "B": attn_patterns.dilated(2, 256),
           "C": attn_patterns.head_split("C", attn_patterns.Head(1, 128, "yerel W=128"),
                                         attn_patterns.Head(4, 128, "strided s=4 W=128"))}
    assert [round(p.avg(T, 8), 2) for p in old.values()] == [192.25, 128.5, 88.31]
    saved = dict(attn_patterns.PATTERNS)
    try:
        for name in ("A", "C"):                            # tek bir modülün eskiye dönmesi de yakalanır
            attn_patterns.PATTERNS.clear()
            attn_patterns.PATTERNS.update({**saved, name: old[name]})
            with pytest.raises(AssertionError, match="SAYILMIŞ bağlantı"):
                GPT(replace(SMALL, foton="x"))
    finally:
        attn_patterns.PATTERNS.clear()
        attn_patterns.PATTERNS.update(saved)
    GPT(SMALL)                                             # donmuş tasarım kurulur
    # desen parametreleri config'ten ya da komut satırından EZİLEMEZ: ModelConfig'te öyle bir alan yok
    fields = set(ModelConfig.__dataclass_fields__)
    assert not {f for f in fields if any(t in f.lower() for t in ("window", "dilat", "stride", "local_w"))}
    # donmuş sabitlerin kendisi
    assert (attn_patterns.A_LOCAL_W, attn_patterns.B_DILATION_R, attn_patterns.B_BUDGET_W,
            attn_patterns.C_LOCAL_W, attn_patterns.C_STRIDE_S, attn_patterns.C_STRIDE_W) == (150, 2, 256, 256, 4, 256)
    assert FOTON_ARMS == {"h": ("A", "A", "A"), "x": ("A", "B", "C")}


def test_frozen_constant_drift_is_caught_even_if_the_counts_still_pass(monkeypatch):
    """Sayım toleransın içinde kalsa bile (W=151 -> 128.9 DEĞİL ama W=150±küçük oynamalar) sabit pinlidir."""
    monkeypatch.setattr(attn_patterns, "B_BUDGET_W", 255)
    with pytest.raises(AssertionError, match="donmuş spesifikasyondan sapmış"):
        GPT(SMALL)


# ------------------------------------------------------------------ ZORUNLU DOĞRULAMA 2: router nedensel

@pytest.mark.parametrize("arm", ["x", "h"])
@pytest.mark.parametrize("train_mode", [False, True])
def test_router_is_causal_no_token_after_k_can_change_the_module_choice(arm, train_mode):
    """Prefix (0..63) sabit, kuyruk (64..511) TAMAMEN değiştirilir: seçim, olasılıklar ve logit'ler BİREBİR aynı."""
    torch.manual_seed(1)
    model = GPT(replace(SMALL, foton=arm))
    model.train(train_mode)
    g = torch.Generator().manual_seed(2)
    x1 = _diverse(model, 12, seed=2)
    x2 = x1.clone()
    x2[:, K:] = torch.randint(0, 97, (12, T - K), generator=g)
    assert (x1[:, K:] != x2[:, K:]).float().mean() > 0.9      # kuyruk gerçekten değişti
    y = torch.zeros_like(x1)
    with torch.no_grad():
        _, _, a = model(x1, y)
        _, _, b = model(x2, y)
    assert torch.equal(a["foton_choice"], b["foton_choice"])
    assert torch.equal(a["foton_logits"], b["foton_logits"]) and torch.equal(a["foton_probs"], b["foton_probs"])
    assert len(set(a["foton_choice"].tolist())) > 1, "test anlamsız: bütün diziler aynı modüle gitti"
    # ters yön: prefix'in İÇİNDEKİ tek bir token seçimi etkileyebilir (router gerçekten prefix'i okuyor)
    x3 = x1.clone()
    x3[:, K - 1] = (x3[:, K - 1] + 1) % 97
    with torch.no_grad():
        _, _, c = model(x3, y)
    assert not torch.equal(a["foton_logits"], c["foton_logits"])
    # sınırın tam yeri: pozisyon k-1 okunur, pozisyon k OKUNMAZ
    x4 = x1.clone()
    x4[:, K] = (x4[:, K] + 1) % 97
    with torch.no_grad():
        _, _, d = model(x4, y)
    assert torch.equal(a["foton_logits"], d["foton_logits"])


def test_predictions_after_the_prefix_are_strictly_causal():
    """Pozisyon t >= k'nın logit'i yalnızca 0..t token'larına bağlıdır: ölçüte giren pozisyonlarda sızıntı YOK.

    (t < k pozisyonları prefix'in TAMAMINDAN seçilen yoldan geçer — ön-kaydın "eşitlenemeyen eksen 6"sı;
    ölçüt bu yüzden 64..511 üzerinden hesaplanır.)"""
    torch.manual_seed(3)
    model = GPT(SMALL).eval()
    g = torch.Generator().manual_seed(4)
    x1 = torch.randint(0, 97, (4, T), generator=g)
    p = 300
    x2 = x1.clone()
    x2[:, p:] = torch.randint(0, 97, (4, T - p), generator=g)
    y = torch.zeros_like(x1)
    with torch.no_grad():
        l1, _, _ = model(x1, y)
        l2, _, _ = model(x2, y)
    assert torch.allclose(l1[:, :p], l2[:, :p], atol=1e-5), "gelecek token geçmiş pozisyonun tahminini değiştirdi"
    assert not torch.allclose(l1[:, p:], l2[:, p:], atol=1e-5)


# ------------------------------------------------------------------ ZORUNLU DOĞRULAMA 3: attention grafiği

def _dependency_rows(attn: CausalSelfAttention, rows: list[int], heads: slice | None = None) -> np.ndarray:
    """Attention katmanının ÇIKTISININ girdiye fonksiyonel bağımlılığı: [len(rows), T] bool.

    out[i]'nin x[j]'ye türevi sıfırdan farklıysa i, j'ye bağlıdır. Maskelenen çiftlerde softmax ağırlığı
    TAM sıfırdır, dolayısıyla türev de tam sıfırdır. `heads` verilirse diğer head'lerin değer (v)
    projeksiyonu sıfırlanır: çıktı yalnızca o head'lerin grafiğini yansıtır (C'nin yarıları ayrı sınanır).
    """
    d, H = attn.qkv.in_features, attn.n_head
    hd = d // H
    if heads is not None:
        with torch.no_grad():
            keep = torch.zeros(H, dtype=torch.bool)
            keep[heads] = True
            v = attn.qkv.weight[2 * d:].view(H, hd, d)
            v[~keep] = 0.0
    from src.model import precompute_rope
    cos, sin = precompute_rope(hd, T, 10000.0)
    x = torch.randn(1, T, d, requires_grad=True)
    out = attn(x, cos, sin)
    dep = np.zeros((len(rows), T), dtype=bool)
    for r, i in enumerate(rows):
        (grad,) = torch.autograd.grad(out[0, i].sum(), x, retain_graph=True)
        dep[r] = (grad[0].abs().sum(dim=-1) > 0).numpy()
    return dep


ROWS = [0, 1, 2, 3, 4, 148, 149, 150, 151, 254, 255, 256, 257, 300, 301, 509, 510, 511]


@pytest.mark.parametrize("pattern,kind", [("A", "local150"), ("B", "dilated2")])
def test_module_attention_graph_is_the_designed_pattern(pattern, kind):
    torch.manual_seed(5)
    attn = CausalSelfAttention(SMALL, pattern).double()
    dep = _dependency_rows(attn.float(), ROWS)
    want = _expected(kind)[ROWS]
    assert np.array_equal(dep, want), f"{pattern}: grafik tasarlanan desen değil (ilk fark satır " \
                                      f"{ROWS[int(np.argwhere((dep != want).any(axis=1))[0])]})"


def test_module_c_is_half_local_half_strided_and_the_halves_are_the_first_and_last_heads():
    torch.manual_seed(6)
    local_heads = _dependency_rows(CausalSelfAttention(SMALL, "C"), ROWS, heads=slice(0, 4))
    torch.manual_seed(6)
    strided_heads = _dependency_rows(CausalSelfAttention(SMALL, "C"), ROWS, heads=slice(4, 8))
    assert np.array_equal(local_heads, _expected("local256")[ROWS]), "C'nin ilk 4 head'i yerel W=256 değil"
    assert np.array_equal(strided_heads, _expected("strided4")[ROWS]), "C'nin son 4 head'i strided s=4 değil"
    # head'ler üzerinden ortalama bağlantı: (192.25 + 64.5) / 2 = 128.375
    avg = (_expected("local256").sum(1).mean() + _expected("strided4").sum(1).mean()) / 2
    assert avg == 128.375


def test_shared_blocks_are_full_causal_like_the_baselines():
    torch.manual_seed(7)
    dep = _dependency_rows(CausalSelfAttention(SMALL, None), ROWS)
    assert np.array_equal(dep, _expected("full")[ROWS])
    model = GPT(SMALL)
    assert all(b.attn.pattern is None and not hasattr(b.attn, "attn_mask") for b in model.blocks)
    assert len(model.blocks) == 6 and [len(m) for m in model.foton_modules] == [2, 2, 2]
    assert [[b.attn.pattern for b in m] for m in model.foton_modules] == [["A"] * 2, ["B"] * 2, ["C"] * 2]
    h = GPT(replace(SMALL, foton="h"))
    assert [[b.attn.pattern for b in m] for m in h.foton_modules] == [["A"] * 2] * 3


def test_masked_sdpa_equals_a_manual_masked_softmax():
    """SDPA'nın maskeyi "True = izinli" diye uyguladığının bağımsız kanıtı (ters okunsa grafik tümleyen olurdu)."""
    torch.manual_seed(8)
    attn = CausalSelfAttention(SMALL, "B")
    from src.model import apply_rope, precompute_rope
    cos, sin = precompute_rope(8, T, 10000.0)
    x = torch.randn(2, T, 64)
    got = attn(x, cos, sin)
    q, k, v = attn.qkv(x).split(64, dim=2)
    q, k, v = (t.view(2, T, 8, 8).transpose(1, 2) for t in (q, k, v))
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = (q @ k.transpose(-1, -2)) / 8 ** 0.5
    scores = scores.masked_fill(~torch.from_numpy(_expected("dilated2")), float("-inf"))
    want = attn.proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(2, T, 64))
    assert torch.allclose(got, want, atol=1e-5)


# ------------------------------------------------------------------ yönlendirme, gate, oracle

def _xy(B=10, seed=9):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 97, (B, T), generator=g)
    return x, torch.randint(0, 97, (B, T), generator=g)


def test_routed_forward_equals_forcing_each_sequence_through_its_chosen_module():
    """Dağıtım (index_select / index_copy) dizileri karıştırmıyor ve gate'in DEĞERİ tam 1.0."""
    torch.manual_seed(10)
    model = GPT(SMALL).eval()
    _, y = _xy()
    x = _diverse(model, 10, seed=9)
    with torch.no_grad():
        logits, _, info = model(x, y)
        choice = info["foton_choice"]
        assert len(set(choice.tolist())) > 1
        for b in range(x.shape[0]):
            forced, _, _ = model(x[b:b + 1], y[b:b + 1], force_module=int(choice[b]))
            assert torch.allclose(logits[b], forced[0], atol=1e-5), f"dizi {b}"
            other = (int(choice[b]) + 1) % 3
            wrong, _, _ = model(x[b:b + 1], y[b:b + 1], force_module=other)
            assert not torch.allclose(logits[b], wrong[0], atol=1e-5)


def test_straight_through_gate_gives_the_router_a_task_gradient_and_unused_modules_get_none():
    torch.manual_seed(11)
    model = GPT(SMALL)
    assert model.cfg.foton_reg == "none"
    x, y = _xy(B=4, seed=12)
    _, loss, info = model(x, y)
    assert loss.item() == pytest.approx(info["ce"].item(), abs=0)      # "none": router'a HİÇBİR ek terim yok
    loss.backward()
    assert model.foton_router.weight.grad is not None and model.foton_router.weight.grad.abs().sum() > 0
    used = set(info["foton_choice"].tolist())
    for m, layers in enumerate(model.foton_modules):
        grads = [p.grad for p in layers.parameters()]
        if m in used:
            assert all(g is not None for g in grads)
        else:                                               # dizi almayan modül ÇALIŞMADI: AdamW onu atlar
            assert all(g is None for g in grads)


def test_router_tie_goes_to_the_lowest_index_and_short_sequences_pool_what_exists():
    torch.manual_seed(13)
    model = GPT(SMALL).eval()
    with torch.no_grad():
        model.foton_router.weight.zero_()                   # bütün logit'ler eşit
        x, y = _xy(B=3)
        _, _, info = model(x, y)
        assert info["foton_choice"].tolist() == [0, 0, 0]
        assert torch.allclose(info["foton_probs"], torch.full((3, 3), 1 / 3))
    torch.manual_seed(13)
    short = GPT(SMALL).eval()
    with torch.no_grad():
        _, _, a = short(x[:, :20], y[:, :20])               # T < k: eldeki 20 pozisyon havuzlanır, hata yok
    assert a["foton_logits"].shape == (3, 3)


def test_oracle_pass_matches_the_routed_forward_and_is_never_worse():
    torch.manual_seed(14)
    model = GPT(SMALL).eval()
    x, y = _xy(B=8, seed=15)
    ev = model.foton_eval_all(x, y)
    assert tuple(ev["ce_tok"].shape) == (3, 8, T)
    per_seq = ev["ce_tok"][:, :, K:].mean(dim=-1)                       # [3, B], ölçüt pozisyonları
    with torch.no_grad():
        logits, _, info = model(x, y)
    assert torch.equal(ev["choice"], info["foton_choice"])
    ce = torch.nn.functional.cross_entropy(logits.view(-1, 97), y.reshape(-1), reduction="none").view(8, T)
    routed = ce[:, K:].mean(dim=-1)
    gathered = per_seq.gather(0, ev["choice"][None, :])[0]
    assert torch.allclose(routed, gathered, atol=1e-5)                  # aynı nicelik, iki yoldan
    assert bool((per_seq.min(dim=0).values <= gathered + 1e-7).all())   # oracle, router'ın seçtiğinden kötü OLAMAZ


def test_v7_regularizers_do_what_their_names_say():
    x, y = _xy(B=6, seed=16)
    out = {}
    for reg in ("none", "entropy", "lb"):
        torch.manual_seed(17)
        model = GPT(replace(SMALL, foton_reg=reg, foton_reg_coef=0.01))
        _, loss, info = model(x, y)
        out[reg] = (loss.item(), info["ce"].item(), info.get("foton_reg"))
    ce = out["none"][1]
    assert out["entropy"][1] == pytest.approx(ce, abs=1e-6) and out["lb"][1] == pytest.approx(ce, abs=1e-6)
    ent, aux = out["entropy"][2].item(), out["lb"][2].item()
    assert 0 < ent <= np.log(3) + 1e-6 and aux >= 1.0 - 1e-6            # Switch aux'un alt sınırı 1 (tam denge)
    assert out["entropy"][0] == pytest.approx(ce - 0.01 * ent, abs=1e-6)   # BONUS: loss'u düşürür
    assert out["lb"][0] == pytest.approx(ce + 0.01 * aux, abs=1e-6)
    with pytest.raises(ValueError, match="foton_reg"):
        GPT(replace(SMALL, foton_reg="zloss"))


# ------------------------------------------------------------------ yapı: init, parametreler, D/M yolu

def test_h_and_x_start_from_identical_weights_and_modules_are_independent_draws():
    torch.manual_seed(18)
    x = GPT(SMALL)
    torch.manual_seed(18)
    h = GPT(replace(SMALL, foton="h"))
    sx, sh = x.state_dict(), h.state_dict()
    assert list(sx) == list(sh) and all(torch.equal(sx[k], sh[k]) for k in sx)
    w = [m[0].attn.qkv.weight for m in h.foton_modules]
    assert not torch.equal(w[0], w[1]) and not torch.equal(w[1], w[2])  # H: aynı desen, FARKLI ağırlık
    assert not any("attn_mask" in k for k in sx)                        # maske checkpoint'e GİRMEZ


def test_init_scale_uses_the_traversed_depth_not_the_constructed_block_count():
    torch.manual_seed(19)
    model = GPT(replace(SMALL, d_model=256, ffn_hidden=512))
    want = 0.02 / (2 * 8) ** 0.5
    for blk in list(model.blocks) + [b for m in model.foton_modules for b in m]:
        for lin in (blk.attn.proj, blk.ffn.w_down):
            assert lin.weight.std().item() == pytest.approx(want, rel=0.05)
    assert model.foton_router.weight.std().item() == pytest.approx(0.02, rel=0.15)
    assert model.foton_router.bias is None


def test_active_parameters_equal_a_single_module_model_plus_the_router():
    torch.manual_seed(20)
    x = GPT(SMALL)
    torch.manual_seed(20)
    d = GPT(replace(SMALL, foton=""))
    assert x.num_params(active_only=True) == d.num_params() + x.foton_router.weight.numel()
    per_module = sum(p.numel() for p in x.foton_modules[0].parameters())
    assert x.num_params() == d.num_params() + 2 * per_module + 3 * 64


def test_foton_config_is_validated():
    for bad in (dict(foton="y"), dict(foton_pre=2), dict(n_experts=4), dict(foton_prefix_k=0)):
        with pytest.raises(ValueError):
            GPT(replace(SMALL, **bad))


def test_d_and_m_models_are_bit_identical_to_the_tagged_code():
    """`foton == ""` iken kurulma sırası ve init RNG tüketimi DEĞİŞMEDİ: D ve M, dondurulmuş baseline'larla
    aynı modeldir. Tag'li `src/model.py` git'ten alınıp aynı seed'le kurulur ve ağırlıklar karşılaştırılır."""
    old_src = subprocess.run(["git", "show", "foton-v02-prereg:src/model.py"], cwd=ROOT, capture_output=True)
    if old_src.returncode != 0:
        pytest.skip("tag erişilemiyor")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model_tagged.py")
        with open(path, "wb") as f:
            f.write(old_src.stdout)
        spec = importlib.util.spec_from_file_location("model_tagged", path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
    small = dict(vocab_size=97, max_seq_len=64, d_model=64, n_layer=3, n_head=4, ffn_hidden=128)
    for extra in ({}, dict(n_experts=4, top_k=2, expert_hidden=32, n_shared_experts=1, router_gate="sigmoid",
                           bias_update_rate=0.001, aux_loss_coef=0.0, moe_impl="loop")):
        cfg = ModelConfig(**small, **extra)
        torch.manual_seed(21)
        new = GPT(cfg)
        torch.manual_seed(21)
        ref = old.GPT(cfg)
        a, b = new.state_dict(), ref.state_dict()
        assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
        x = torch.randint(0, 97, (2, 64), generator=torch.Generator().manual_seed(22))
        l1, l2 = new(x, x)[1], ref(x, x)[1]
        assert torch.equal(l1, l2)
