"""Analitik FLOP formülünün (src/flops.py) modelin kendisine ve bağımsız bir sayaca karşı denetimi (CPU)."""
from __future__ import annotations

import inspect
import os
import sys
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src import attn_patterns  # noqa: E402
from src.config import PRESETS, ModelConfig  # noqa: E402
from src.flops import (CONTEXT_KEYS, foton_arm_connections, foton_doc_tables, foton_flops_realized,  # noqa: E402
                       foton_module_connections, foton_module_keys, forward_flops_per_token,
                       secondary_forward_flops_per_token, train_flops_per_token)
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


# ============================================================================================ FOTON v0.2
# Ön-kayıt V3 "Analitik hesap src/flops.py ..." paragrafı + v01 EK1-B13. Türetim: docs/flops.md bölüm 8.
# Bu testlerdeki beklenen değerler flops.py'nin yardımcılarından DEĞİL, bağımsız yoldan kurulur: donmuş
# spesifikasyondan (W=150 / r=2 W=256 / yarım yerel W=256 + yarım strided s=4 W=256) elle kurulan maske
# sayılır. flops.py ile attn_patterns birlikte yanlışa kayarsa da test düşsün diye.

V_FOTON = 32768                       # data/foton tokenizer'ı; train.py vocab_size'ı oradan okur
FOTON_CTX_KEYS = ("attn_context", "foton_attn_routed")
S1_SHAPE = dict(d_model=640, n_layer=10, n_head=10, foton_pre=4, foton_routed=2, foton_post=4)
FROZEN_PATTERNS = dict(attn_patterns.PATTERNS)      # içe aktarma anındaki (donmuş) kayıt; sızıntı testleri bunu bozar


def foton_cfg(arm: str, **over) -> ModelConfig:
    return replace(PRESETS[f"foton-{arm}"][0], vocab_size=V_FOTON, **over)


def spec_masks(name: str, T: int, n_head: int) -> np.ndarray:
    """DONMUŞ spesifikasyonun maskesi, attn_patterns'ten BAĞIMSIZ kaba kuvvetle: [n_head, T, T] bool."""
    off = np.arange(T)[:, None] - np.arange(T)[None, :]

    def local(W):
        return (off >= 0) & (off < W)

    def strided(s, W):
        return (off >= 0) & (off % s == 0) & (off // s < W)

    assert n_head % 2 == 0
    heads = {"A": [local(150)] * n_head,
             "B": [strided(2, 256)] * n_head,
             "C": [local(256)] * (n_head // 2) + [strided(4, 256)] * (n_head // 2)}[name]
    return np.stack(heads)


def spec_c(name: str, T: int, n_head: int) -> Fraction:
    """Token/head başına ortalama nedensel bağlantı: maske kurulup True girdiler SAYILARAK, tam kesir."""
    return Fraction(int(spec_masks(name, T, n_head).sum()), n_head * T)


def spec_masks_full_c(T: int) -> float:
    """Tam nedensel attention'ın ortalama bağlantısı (D/M ve paylaşılan katmanlar), alt üçgen SAYILARAK."""
    return float(Fraction(int(np.tril(np.ones((T, T), dtype=bool)).sum()), T))


def arm_spec_c(arm: str, T: int, n_head: int) -> Fraction:
    """Pinli karar 2: H -> A; X -> üç modülün MAKSİMUMU."""
    return spec_c("A", T, n_head) if arm == "h" else max(spec_c(n, T, n_head) for n in "ABC")


def foton_active_matmul_params(model: GPT) -> int:
    """Bir dizinin GEÇTİĞİ lineer ağırlıklar: lm_head + paylaşılan bloklar + TEK modülün blokları (router hariç)."""
    def linears(block):
        return sum(m.weight.numel() for part in (block.attn, block.ffn) for m in part.modules()
                   if isinstance(m, nn.Linear))

    per_module = [sum(linears(b) for b in mod) for mod in model.foton_modules]
    assert len(set(per_module)) == 1, "modüller aynı şekilde olmalı, yoksa aktif FLOP seçime bağlı olurdu"
    return model.lm_head.weight.numel() + sum(linears(b) for b in model.blocks) + per_module[0]


def _dotted(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _old_ox1_patterns() -> dict:
    """Ön-kaydın O-X1 bölümünde kalan ESKİ (reddedilen) tasarım: A bitişik 256 / B dilated 256 / C 128+128.
    Yalnızca sızıntı testleri için kurulur; donmuş kayıt (`attn_patterns.PATTERNS`) değişmez."""
    H = attn_patterns.Head
    old = dict(FROZEN_PATTERNS)
    old.update(A=attn_patterns.local(256), B=attn_patterns.dilated(2, 256),
               C=attn_patterns.head_split("C", H(1, 128, "yerel W=128"), H(4, 128, "strided s=4 W=128")))
    return old


@pytest.mark.parametrize("preset, frozen", [("foton-d", 267_411_456), ("foton-m", 267_755_520)])
def test_frozen_baseline_train_flops_unchanged(preset, frozen):
    """FOTON genişletmesi D ve M'in FLOP/token'ını DEĞİŞTİREMEZ: ön-kaydın `F_base` referansı ve
    baseline_reference.json bu sayılarla donduruldu. Sayı hem sabit olarak hem de donmuş belgenin metninde aranır."""
    got = train_flops_per_token(replace(PRESETS[preset][0], vocab_size=V_FOTON))
    assert got == frozen
    with open(os.path.join(ROOT, "PREREGISTRATION_V02.md"), encoding="utf-8") as fh:
        assert _dotted(got) in fh.read()


def _pre_foton_forward(cfg: ModelConfig, T: int) -> dict[str, int]:
    """FOTON genişletmesinden ÖNCEKİ formülün birebir kopyası (commit 0970b8b'deki src/flops.py) — referans."""
    d, V = cfg.d_model, cfg.vocab_size
    n_moe = sum(1 for i in range(cfg.n_layer) if cfg.n_experts > 0 and i >= cfg.first_dense_layers)
    n_dense = cfg.n_layer - n_moe
    return {"attn_qkv_proj": cfg.n_layer * 2 * 4 * d * d,
            "attn_context": cfg.n_layer * 2 * d * (T + 1),
            "ffn_dense": n_dense * 2 * 3 * d * cfg.ffn_hidden,
            "moe_router": n_moe * 2 * d * cfg.n_experts,
            "moe_routed_experts": n_moe * cfg.top_k * 2 * 3 * d * cfg.expert_hidden,
            "moe_shared_experts": n_moe * cfg.n_shared_experts * 2 * 3 * d * cfg.expert_hidden,
            "lm_head": 2 * d * V}


def _pre_foton_secondary(cfg: ModelConfig, T: int) -> dict[str, float]:
    d, V, nh = cfg.d_model, cfg.vocab_size, cfg.n_head
    n_moe = sum(1 for i in range(cfg.n_layer) if cfg.n_experts > 0 and i >= cfg.first_dense_layers)
    n_dense, E, k, he = cfg.n_layer - n_moe, cfg.n_experts, cfg.top_k, cfg.expert_hidden
    return {"rmsnorm": (2 * cfg.n_layer + 1) * 4 * d, "rope_qk": cfg.n_layer * 2 * 3 * d,
            "attn_scale_softmax": cfg.n_layer * 4 * nh * ((T + 1) / 2), "residual_add": cfg.n_layer * 2 * d,
            "ffn_dense_act": n_dense * 4 * cfg.ffn_hidden,
            "moe_expert_act": n_moe * (k + cfg.n_shared_experts) * 4 * he,
            "moe_router_softmax_topk": n_moe * (3 * E + E + 2 * k),
            "moe_combine": n_moe * (2 * k * d + cfg.n_shared_experts * d),
            "moe_aux_z_loss": n_moe * 3 * E, "cross_entropy": 3 * V}


@pytest.mark.parametrize("preset", sorted(p for p, (m, _) in PRESETS.items() if not m.foton))
@pytest.mark.parametrize("seq_len", [None, 256])
def test_non_foton_configs_bit_identical(preset, seq_len):
    """FOTON olmayan her preset: eski anahtarların HEPSİ aynı sırada ve aynı değerde, yeni `foton_*` anahtarları 0,
    toplamlar birebir (birincil tamsayı; ikincil float, 0.0 eklemek biti değiştirmez)."""
    cfg = PRESETS[preset][0]
    T = seq_len or cfg.max_seq_len
    for got, ref in ((forward_flops_per_token(cfg, seq_len), _pre_foton_forward(cfg, T)),
                     (secondary_forward_flops_per_token(cfg, seq_len), _pre_foton_secondary(cfg, T))):
        assert list(got)[:len(ref)] == list(ref)
        assert {k: got[k] for k in ref} == ref
        extra = [k for k in got if k not in ref]
        assert extra and all(k.startswith("foton_") and got[k] == 0 for k in extra)
        assert sum(got.values()) == sum(ref.values())
    assert train_flops_per_token(cfg, seq_len) == 3 * sum(_pre_foton_forward(cfg, T).values())


@pytest.mark.parametrize("preset", ["foton-h", "foton-x", "tiny-foton"])
def test_foton_matmul_counts_traversed_layers_not_constructed_blocks(preset):
    """Pinli karar 1: aktif FLOP bir dizinin GEÇTİĞİ n_layer katmanı sayar (paylaşılan + TEK modül), kurulan
    n_layer + 2*routed bloğu değil. Formülden bağımsız sayım: modelin kendi Linear ağırlıkları."""
    cfg = PRESETS[preset][0]
    model = GPT(cfg)
    assert len(model.blocks) + len(model.foton_modules[0]) == cfg.n_layer
    n_constructed = len(model.blocks) + sum(len(mod) for mod in model.foton_modules)
    assert n_constructed == cfg.n_layer + 2 * cfg.foton_routed
    f = forward_flops_per_token(cfg)
    # Tüketicilere verilen sabit (profiler çapraz kontrolü bağlamı bununla ayırır) testin elle yazdığı kümeyle aynı:
    # yönlendirilen bağlam terimi sabitten düşerse aşağıdaki eşitlik de o tüketicilerde sessizce bozulurdu.
    assert CONTEXT_KEYS == FOTON_CTX_KEYS and set(CONTEXT_KEYS) <= set(f)
    matmul = sum(v for k, v in f.items() if k not in FOTON_CTX_KEYS + ("foton_router",))
    assert matmul == 2 * foton_active_matmul_params(model)
    assert f["moe_router"] == f["moe_routed_experts"] == f["moe_shared_experts"] == 0
    assert f["attn_qkv_proj"] == cfg.n_layer * 8 * cfg.d_model ** 2            # 8 katman, 12 değil
    assert f["ffn_dense"] == cfg.n_layer * 6 * cfg.d_model * cfg.ffn_hidden
    assert f["attn_context"] == (cfg.foton_pre + cfg.foton_post) * 2 * cfg.d_model * (cfg.max_seq_len + 1)


@pytest.mark.parametrize("arm", ["h", "x"])
def test_foton_routed_context_is_counted_from_frozen_spec_masks(arm):
    """Pinli karar 2: yönlendirilen katman başına 4*d*c; c donmuş spesifikasyonun maskesinden SAYILIR.
    H -> A (128.17), X -> maksimum = B (128.50). Beklenen değer burada kaba kuvvet maskeden kurulur."""
    cfg = foton_cfg(arm)
    d, T, nh = cfg.d_model, cfg.max_seq_len, cfg.n_head
    c = arm_spec_c(arm, T, nh)
    f = forward_flops_per_token(cfg)
    assert f["foton_attn_routed"] == round(cfg.foton_routed * 4 * d * c)
    assert isinstance(f["foton_attn_routed"], int) and all(isinstance(v, int) for v in f.values())
    assert foton_arm_connections(cfg) == float(c)
    assert round(float(c), 2) == {"h": 128.17, "x": 128.50}[arm]
    names = {"h": "AAA", "x": "ABC"}[arm]
    assert foton_module_connections(cfg) == {f"M{m}:{n}": float(spec_c(n, T, nh)) for m, n in enumerate(names)}


def test_foton_connections_match_model_mask_buffers():
    """flops.py'nin saydığı grafik, modelin SDPA'ya verdiği `attn_mask` buffer'larından sayılanla aynı — ve
    modül kimlikleri (`M<indeks>:<desen>`) modelinkiyle birebir (mix anahtarları bu kimliklerdir)."""
    for preset in ("tiny-foton", "foton-x", "foton-h"):
        cfg = PRESETS[preset][0]
        counts = GPT(cfg).foton_connection_counts()
        assert tuple(counts) == foton_module_keys(cfg)
        assert foton_module_connections(cfg) == pytest.approx(counts, abs=1e-12)


def test_foton_arms_cheaper_than_dense_and_moe():
    """H < D ve X < D (ve ikisi de < M): yönlendirilen 2 katman 256.5 yerine ~128.4 bağlantı sayar; router 6 FLOP.
    Tam-nedensel terimi modül katmanlarına uygulayan bir formülde H = D + router olurdu (ön-kayıt V3'ün uyarısı)."""
    t = {a: train_flops_per_token(foton_cfg(a)) for a in "dmhx"}
    assert t["h"] < t["x"] < t["d"] < t["m"]
    cfg = foton_cfg("h")
    d, T = cfg.d_model, cfg.max_seq_len
    full_two_layers = cfg.foton_routed * 2 * d * (T + 1)
    fh = forward_flops_per_token(cfg)
    assert t["d"] - t["h"] == 3 * (full_two_layers - fh["foton_attn_routed"] - fh["foton_router"])


def test_x_minus_h_is_exactly_the_routed_context_difference():
    """X − H = 3 (eğitim) * 2 katman * 4*d*(c_X − c_H), yuvarlama payıyla (iki terim ayrı yuvarlanır: <= 3 FLOP).
    Başka HİÇBİR terim iki kol arasında farklı olamaz."""
    h, x = foton_cfg("h"), foton_cfg("x")
    fh, fx = forward_flops_per_token(h), forward_flops_per_token(x)
    assert {k for k in fh if fh[k] != fx[k]} == {"foton_attn_routed"}
    d, T, nh = x.d_model, x.max_seq_len, x.n_head
    exact = 3 * x.foton_routed * 4 * d * (arm_spec_c("x", T, nh) - arm_spec_c("h", T, nh))
    diff = train_flops_per_token(x) - train_flops_per_token(h)
    assert diff > 0 and abs(diff - exact) <= 3


def test_primary_flops_do_not_depend_on_router_selection():
    """Birincil istatistiğin paydası router'ın kontrolünde olamaz (ön-kayıt V3 "FOTON'a özgü uyarı"): birincil
    fonksiyonlar karışım parametresi ALMAZ, ve hiçbir gerçekleşen karışım birincil değeri AŞAMAZ."""
    for fn in (forward_flops_per_token, train_flops_per_token, secondary_forward_flops_per_token):
        assert list(inspect.signature(fn).parameters) == ["cfg", "seq_len"]
    x = foton_cfg("x")
    primary = train_flops_per_token(x)
    keys = foton_module_keys(x)
    assert keys == ("M0:A", "M1:B", "M2:C")
    rng = np.random.default_rng(0)
    for _ in range(50):
        mix = {key: float(share) for key, share in zip(keys, rng.dirichlet(np.ones(len(keys))))}
        assert foton_flops_realized(x, mix) <= primary


def test_realized_mix_corners():
    """Duyarlılık satırının uçları: X'te hepsi-A karışımı = H'nin değeri (iki kolun tek farkı c), hepsi-B = X'in
    birincil değeri (c_X = B), hepsi-C ikisinin arasında. H'de karışım değeri değiştirmez (üç modül de A)."""
    h, x = foton_cfg("h"), foton_cfg("x")
    th, tx = train_flops_per_token(h), train_flops_per_token(x)
    corner = {k: foton_flops_realized(x, {kk: float(kk == k) for kk in foton_module_keys(x)})
              for k in foton_module_keys(x)}
    assert corner["M0:A"] == th
    assert corner["M1:B"] == tx
    assert th < corner["M2:C"] < tx
    assert isinstance(corner["M2:C"], int)
    assert foton_flops_realized(h, {"M0:A": 0.2, "M1:A": 0.5, "M2:A": 0.3}) == th
    # seq_len de taşınır. T=256'da maksimum artık B değil A'dır (B'nin tavanı T ile küçülür): birincil değer
    # "B" diye sabitlenmiş bir modüle değil, o T'de SAYILAN maksimuma eşittir.
    corner_256 = {k: foton_flops_realized(x, {kk: float(kk == k) for kk in foton_module_keys(x)}, 256)
                  for k in foton_module_keys(x)}
    assert max(corner_256.values()) == corner_256["M0:A"] == train_flops_per_token(x, 256)
    assert corner_256["M1:B"] < corner_256["M0:A"]


def test_realized_mix_is_paired_by_module_identity_not_position():
    """Paylar modüle KİMLİKLE eşlenir, eşlemenin yineleme SIRASIYLA değil. `zip(mix.values(), modüller)` biçiminde
    konumsal bir eşleme, anahtarları başka sırada kurulmuş bir karışımda (ör. JSON'dan okunmuş, sıklığa göre
    sıralanmış) payları sessizce YANLIŞ modüle yazardı — projede üç kez doğan paralel-liste hatasının aynısı.
    Öteki karışım testlerinin hepsi anahtarları modül sırasında kurar; bu test sırayı bilerek bozar."""
    h, x = foton_cfg("h"), foton_cfg("x")
    keys = foton_module_keys(x)
    shares = {"M0:A": 0.7, "M1:B": 0.2, "M2:C": 0.1}
    assert tuple(shares) == keys
    in_order = foton_flops_realized(x, shares)
    assert foton_flops_realized(x, {k: shares[k] for k in reversed(keys)}) == in_order
    assert foton_flops_realized(x, {k: shares[k] for k in ("M1:B", "M2:C", "M0:A")}) == in_order
    # köşeler ters sırada: "hepsi A" hâlâ H'nin, "hepsi B" hâlâ X'in değeri (konumsal eşlemede ikisi de C'ye düşerdi)
    assert foton_flops_realized(x, {"M2:C": 0.0, "M1:B": 0.0, "M0:A": 1.0}) == train_flops_per_token(h)
    assert foton_flops_realized(x, {"M2:C": 0.0, "M0:A": 0.0, "M1:B": 1.0}) == train_flops_per_token(x)
    # payların kendisi farklı modüllere gidince sonuç DEĞİŞMELİ (test, sıradan bağımsız sabit bir değerle geçmesin)
    assert foton_flops_realized(x, {"M0:A": 0.1, "M1:B": 0.2, "M2:C": 0.7}) != in_order


@pytest.mark.parametrize("mix, exc", [
    ([1.0, 0.0, 0.0], TypeError),                                         # paralel liste: kimlik yok
    ({"A": 1.0, "B": 0.0, "C": 0.0}, ValueError),                         # desen adı tek başına kimlik değil
    ({"M0:A": 0.5, "M1:B": 0.5}, ValueError),                             # eksik modül
    ({"M0:A": 0.5, "M1:B": 0.5, "M2:C": 0.0, "M3:D": 0.0}, ValueError),   # fazla modül
    ({"M0:A": 0.5, "M1:B": 0.3, "M2:C": 0.1}, ValueError),                # toplam 0.9
    ({"M0:A": 0.5, "M1:B": 0.5, "M2:C": 1e-6}, ValueError),               # toplam 1 + 1e-6 > tolerans
    ({"M0:A": 1.5, "M1:B": -0.5, "M2:C": 0.0}, ValueError),               # toplam 1 ama pay aralık dışı
    # toplam 1 ve HİÇBİR pay 1'i aşmıyor: yalnızca ALT sınır yakalar. Üstteki durumu 1.5 tek başına düşürür, yani
    # "pay >= 0" denetimi silinse o durum yine hata verirdi; negatif pay ucuz modülü eksi ağırlıkla sayar.
    ({"M0:A": 0.6, "M1:B": 0.6, "M2:C": -0.2}, ValueError),
    ({"M0:A": float("nan"), "M1:B": 0.5, "M2:C": 0.5}, ValueError),       # NaN karşılaştırmaları False döner
    ({"M0:A": "1", "M1:B": 0.0, "M2:C": 0.0}, ValueError),
], ids=["liste", "desen-adi", "eksik", "fazla", "toplam-0.9", "toplam-1+1e-6", "negatif", "yalniz-negatif", "nan",
        "metin"])
def test_realized_mix_rejects_malformed(mix, exc):
    with pytest.raises(exc):
        foton_flops_realized(foton_cfg("x"), mix)


def test_realized_mix_tolerance_and_non_foton():
    x = foton_cfg("x")
    ok = {"M0:A": 0.25, "M1:B": 0.5, "M2:C": 0.25 + 5e-10}          # 1e-9 toleransının İÇİNDE
    assert foton_flops_realized(x, ok) <= train_flops_per_token(x)
    with pytest.raises(ValueError, match="yalnızca FOTON"):
        foton_flops_realized(foton_cfg("d"), ok)


def test_router_is_per_sequence_amortised_over_T():
    """Pinli karar 3: router DİZİ başına tek d -> 3 çarpımı = 2*d*3; token başına round(2*d*3/T).
    Token başına saymak (2*d*3 = 3072) router'ı T kat şişirirdi."""
    x = foton_cfg("x")
    assert forward_flops_per_token(x)["foton_router"] == 6 == round(2 * x.d_model * 3 / x.max_seq_len)
    assert forward_flops_per_token(x, 256)["foton_router"] == 12           # T yarıya inince pay ikiye katlanır
    assert forward_flops_per_token(x, 1)["foton_router"] == 2 * x.d_model * 3
    assert forward_flops_per_token(foton_cfg("h"))["foton_router"] == 6
    s1 = foton_cfg("x", **S1_SHAPE)
    assert forward_flops_per_token(s1)["foton_router"] == round(Fraction(2 * 640 * 3, 512))


def test_profiler_foton_linear_terms_and_dense_mask_artefact():
    """Bağımsız sayaç (FlopCounterMode), küçük FOTON modelinde, CPU:
      * `mm` toplamı = B*T * (token başına matris çarpımları) + B * 2*d*3: router TAM OLARAK dizi başına bir
        çarpımdır (token başına olsaydı B*T*2*d*3 çıkardı); ileri ve ileri+geri (3x) için birebir.
      * boolean maskeli SDPA matematik yoluna açılır ve sayaç onu maskeden BAĞIMSIZ yoğun T x T sayar
        (katman başına 4*d*T*T / dizi). Analitik terim ALGORİTMİK bağlantı sayısıdır (4*d*c*T / dizi) ve
        bundan küçüktür: docs/flops.md bölüm 8.6'da beyan edilen profiler artefaktı budur."""
    cfg = replace(PRESETS["tiny-foton"][0], vocab_size=97, max_seq_len=16, d_model=32, ffn_hidden=88,
                  foton_prefix_k=4)
    torch.manual_seed(0)
    model = GPT(cfg)
    B, T, d = 6, cfg.max_seq_len, cfg.d_model
    x = torch.randint(0, cfg.vocab_size, (B, T))
    f = forward_flops_per_token(cfg)
    matmul = sum(v for k, v in f.items() if k not in FOTON_CTX_KEYS + ("foton_router",))
    router_per_seq = 2 * model.foton_router.weight.numel()          # modelin kendi router ağırlığından
    assert router_per_seq == 2 * d * len(model.foton_modules)
    assert abs(f["foton_router"] * T - router_per_seq) <= T / 2     # formülün token başına payı x T = dizi başına çarpım
    linear = B * T * matmul + B * router_per_seq

    def by_op(counter, name):
        return sum(v for op, v in counter.get_flop_counts()["Global"].items() if op.__name__ == name)

    with FlopCounterMode(display=False) as fwd:
        model(x, x)
    assert by_op(fwd, "mm") == linear
    with FlopCounterMode(display=False) as both:
        _, loss, _ = model(x, x)
        loss.backward()
    assert by_op(both, "mm") == 3 * linear
    dense_mask = cfg.foton_routed * B * 4 * d * T * T
    assert by_op(fwd, "bmm") == dense_mask and by_op(both, "bmm") == 3 * dense_mask
    assert B * T * f["foton_attn_routed"] < dense_mask


def test_s1_shape_works_and_uses_cfg_n_head():
    """S1 (d=640, 10 katman = 4+2+4, n_head=10): T ve n_head attn_patterns'e AÇIKÇA geçer. attn_patterns'in
    varsayılanı n_head=8'dir; varsayılan kullanılsaydı n_head=5 (C ikiye bölünemez) hata VERMEZDİ."""
    for arm in "hx":
        cfg = foton_cfg(arm, **S1_SHAPE)
        f = forward_flops_per_token(cfg)
        assert f["foton_attn_routed"] == round(2 * 4 * 640 * arm_spec_c(arm, 512, 10))
        assert f["attn_context"] == 8 * 2 * 640 * 513
        assert f["attn_qkv_proj"] == 10 * 8 * 640 * 640
        assert foton_module_connections(cfg) == pytest.approx(
            {k: float(spec_c(k[-1], 512, 10)) for k in foton_module_keys(cfg)}, abs=1e-12)
        assert train_flops_per_token(cfg) == 3 * sum(f.values())
    odd = dict(S1_SHAPE, n_head=5)
    with pytest.raises(ValueError, match="n_heads=5"):
        forward_flops_per_token(foton_cfg("x", **odd))
    assert forward_flops_per_token(foton_cfg("h", **odd))["foton_attn_routed"] > 0   # A head düzeyinde homojen


def test_seq_len_reaches_the_patterns():
    """c, T'ye bağlıdır (T=256'da A 106.35, T=512'de 128.17): seq_len attn_patterns'e taşınmalı."""
    cfg = foton_cfg("h")
    for T in (256, 512, 64):
        expected = round(cfg.foton_routed * 4 * cfg.d_model * spec_c("A", T, cfg.n_head))
        assert forward_flops_per_token(cfg, T)["foton_attn_routed"] == expected
    assert forward_flops_per_token(cfg, 256)["foton_attn_routed"] != forward_flops_per_token(cfg)["foton_attn_routed"]


@pytest.mark.parametrize("arm", ["h", "x"])
def test_implied_connections_are_the_frozen_target_not_the_old_design(arm):
    """FLOP'un İMA ETTİĞİ c (terim / (2 katman * 4d)) donmuş hedefte: 128.4 ± 0.4. ESKİ O-X1 tasarımının
    sayıları (192.25 / 88.31) ya da tam-nedensel 256.5 formüle sızarsa — desenle ya da elle yazılmış bir
    sabitle — bu test düşer."""
    cfg = foton_cfg(arm)
    implied = Fraction(forward_flops_per_token(cfg)["foton_attn_routed"], cfg.foton_routed * 4 * cfg.d_model)
    assert abs(implied - Fraction("128.4")) <= Fraction("0.4")
    for stale in (192.25, 88.31, 256.5):
        assert abs(float(implied) - stale) > 30


def test_old_ox1_design_leak_stops_the_flop_computation(monkeypatch):
    """İkinci kilidin FLOP tarafı: attn_patterns ESKİ O-X1 tasarımına (A bitişik 256 / B dilated 256 /
    C 128+128) dönerse FLOP hesabı sessizce 192.25 / 88.31 ile sürmez, DURUR."""
    old = _old_ox1_patterns()
    got = {k: round(old[k].avg(512, 8), 2) for k in "ABC"}
    assert got == {"A": 192.25, "B": 128.5, "C": 88.31}, "test gerçekten ESKİ tasarımı kurmalı"
    monkeypatch.setattr(attn_patterns, "PATTERNS", old)
    for arm in "hx":
        cfg = foton_cfg(arm)
        for call in (forward_flops_per_token, train_flops_per_token, secondary_forward_flops_per_token,
                     foton_module_connections, foton_arm_connections,
                     lambda c: foton_flops_realized(c, {k: 1 / 3 for k in foton_module_keys(c)})):
            with pytest.raises(AssertionError, match="O-X1"):
                call(cfg)
    # yalnızca C sızsa bile (A ve B yerinde): X durur, H (C kullanmaz) etkilenmez
    monkeypatch.setattr(attn_patterns, "PATTERNS", dict(FROZEN_PATTERNS, C=old["C"]))
    with pytest.raises(AssertionError, match="M2:C"):
        train_flops_per_token(foton_cfg("x"))
    assert train_flops_per_token(foton_cfg("h")) > 0


def test_counted_mask_must_agree_with_closed_form(monkeypatch):
    """c maskeden SAYILIR ve kapalı form `avg()` ile karşılaştırılır: ikisi ayrışırsa (ör. head
    ağırlıklandırması bir yolda değişirse) hesap durur."""
    monkeypatch.setattr(attn_patterns.Pattern, "avg", lambda self, T=512, n_heads=8: 128.4)
    with pytest.raises(AssertionError, match="kapalı form"):
        forward_flops_per_token(foton_cfg("x"))


@pytest.mark.parametrize("over, match", [
    (dict(n_experts=16), "n_experts = 0"),
    (dict(foton_pre=4), "katman bölüşümü"),
    (dict(foton_routed=0, foton_post=5), "katman bölüşümü"),
    (dict(foton="z"), "içinden olmalı"),
], ids=["moe", "bolusum-toplami", "bolusum-min", "bilinmeyen-kol"])
def test_foton_flops_reject_configs_the_model_would_reject(over, match):
    """Formül `GPT._build_foton`'un kabul etmediği bir config için sayı ÜRETMEZ (modeli kurmadan FLOP isteyen
    betiklerde yanlış bölüşüm sessizce paydaya girmesin)."""
    with pytest.raises(ValueError, match=match):
        forward_flops_per_token(foton_cfg("x", **over))
    with pytest.raises(ValueError):
        GPT(replace(PRESETS["tiny-foton"][0], **over))


def test_secondary_foton_terms():
    """İkincil (bütçeye girmeyen) FOTON terimleri, docs/flops.md bölüm 8.5'teki türetimle: prefix ortalaması
    (k*d + d)/T, router softmax+argmax 4*3/T, gate+residual 3*d + 1/T, regularizer dizi başına / T."""
    x = foton_cfg("x")
    d, T, k = x.d_model, x.max_seq_len, x.foton_prefix_k
    s = secondary_forward_flops_per_token(x)
    assert s["foton_prefix_pool"] == (k * d + d) / T == 65.0
    assert s["foton_router_softmax_argmax"] == 12 / T
    assert s["foton_gate_residual"] == 3 * d + 1 / T
    assert s["foton_reg"] == 0
    assert secondary_forward_flops_per_token(replace(x, foton_reg="entropy"))["foton_reg"] == 12 / T
    assert secondary_forward_flops_per_token(replace(x, foton_reg="lb"))["foton_reg"] == 6 / T
    with pytest.raises(ValueError, match="foton_reg"):
        secondary_forward_flops_per_token(replace(x, foton_reg="l2"))
    # modelin kurmayı reddettiği prefix uzunluğu için ikincil sayım da sayı üretmez (k = 0 havuzlamayı d/T'ye indirirdi)
    with pytest.raises(ValueError, match="foton_prefix_k"):
        secondary_forward_flops_per_token(replace(x, foton_prefix_k=0))
    with pytest.raises(ValueError, match="foton_prefix_k"):
        GPT(replace(PRESETS["tiny-foton"][0], foton_prefix_k=0))
    # yönlendirilen katmanlarda softmax c bağlantı üzerinde: D'ninkinden küçük, paylaşılan 6 katman aynı
    sd = secondary_forward_flops_per_token(foton_cfg("d"))
    c = float(arm_spec_c("x", T, x.n_head))
    assert s["attn_scale_softmax"] == 6 * 4 * x.n_head * (T + 1) / 2 + 2 * 4 * x.n_head * c
    assert s["attn_scale_softmax"] < sd["attn_scale_softmax"]
    # FOTON'a özgü ikincil terimlerin tamamı birincil ilerinin yüz binde 2'sinin altında (B13: kapasite oraya saklanamaz)
    foton_only = sum(v for key, v in s.items() if key.startswith("foton_"))
    assert foton_only / sum(forward_flops_per_token(x).values()) < 2e-5
    # prefix T'den uzunsa eldeki pozisyonlar havuzlanır (model: k = min(foton_prefix_k, T))
    assert secondary_forward_flops_per_token(x, 16)["foton_prefix_pool"] == (16 * d + d) / 16


# docs/flops.md bölüm 8'de koddan ÜRETİLMEYEN iki tablo (başlık satırlarıyla). İkisi de sayı tablosu değildir: sembol
# tablosunun değerleri `test_docs_foton_prose_numbers_are_computed`'da config'ten kurulur, işlem listesi B13 beyanıdır.
DOC_HAND_TABLES = ("| sembol | anlam | değer |",
                   "| işlem (kod) | neden birincil tanımın dışında | büyüklük (ileri / token) |")


def _doc_section8() -> list[str]:
    with open(os.path.join(ROOT, "docs", "flops.md"), encoding="utf-8") as fh:
        lines = [line.rstrip() for line in fh]
    start = lines.index("## 8. FOTON v0.2 (V=32768)")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start:end]


def _md_tables(lines: list[str]) -> dict[str, list[str]]:
    """Ardışık "|" satırları bir tablodur; başlık satırı -> tablonun bütün satırları. Başlık tekrar ederse hata."""
    tables, cur = [], []
    for line in list(lines) + [""]:
        if line.startswith("|"):
            cur.append(line.rstrip())
        elif cur:
            tables.append(cur)
            cur = []
    assert len({t[0] for t in tables}) == len(tables), "aynı başlıklı iki tablo"
    return {t[0]: t for t in tables}


def test_docs_foton_tables_are_generated_by_the_code():
    """docs/flops.md bölüm 8'in tabloları `python -m src.flops` çıktısıdır ve belgede TABLO TABLO BİREBİR durur:
    eksik satır da, FAZLA satır da düşürür. Yalnızca "üretilen her satır belgede var mı" diye bakmak yetmez —
    güncel satırın yanında elle eklenmiş ya da eski koşudan kalmış ÇELİŞEN bir satır (ör. ikinci bir "X − M") o
    denetimden geçerdi. Üretilmeyen tablolar açık bir beyaz listededir; listede olmayan yeni bir sayı tablosu düşer."""
    generated = _md_tables(foton_doc_tables().splitlines())
    doc = _md_tables(_doc_section8())
    assert len(generated) == 8
    for header, table in generated.items():
        assert header in doc, f"docs/flops.md bölüm 8'de tablo yok: {header}"
        assert doc[header] == table, ("docs/flops.md güncel değil (`python -m src.flops` ile yeniden üretin):\n"
                                      + "\n".join(sorted(set(doc[header]) ^ set(table))))
    assert sorted(h for h in doc if h not in generated) == sorted(DOC_HAND_TABLES)


def test_docs_foton_prose_numbers_are_computed():
    """Bölüm 8'in DÜZ METNİNDEKİ sayılar da hesaplanan değerlere bağlıdır (tablolar yukarıdaki testte). Beklenen
    her ifade burada config'ten / formülden / kaba kuvvet maskeden kurulur ve belgede aranır; config ya da formül
    değişip metin eski sayıyla kalırsa düşer. Satır kırılmaları karşılaştırmadan önce tek boşluğa indirilir."""
    from src.model import FOTON_CONN_TARGET, FOTON_CONN_TOL
    text = " ".join(" ".join(_doc_section8()).split())
    cfgs = {a: foton_cfg(a) for a in "dmhx"}
    x = cfgs["x"]
    d, T, L, nh = x.d_model, x.max_seq_len, x.n_layer, x.n_head
    fwd = {a: forward_flops_per_token(c) for a, c in cfgs.items()}
    train = {a: train_flops_per_token(c) for a, c in cfgs.items()}
    n_mod, shared = len(foton_module_keys(x)), x.foton_pre + x.foton_post
    full_c = spec_masks_full_c(T)
    old = {k: float(Fraction(int(p.masks(T, nh).sum()), nh * T)) for k, p in _old_ox1_patterns().items() if k in "ABC"}
    expected = [
        f"hepsinde d = {d}, L = {L}, n_h = {nh}, h = {x.ffn_hidden}, T = {T}, V = {V_FOTON}",
        f"| {x.foton_pre} + {x.foton_routed} + {x.foton_post} = L |",
        f"| n_m | modül sayısı (router çıkışı) | {n_mod} |",
        f"| k | router'ın gördüğü nedensel prefix uzunluğu | {x.foton_prefix_k} |",
        f"### 8.1 Bir dizinin geçtiği katmanlar: {L}, kurulan {shared + n_mod * x.foton_routed} değil",
        f"FOTON modeli {shared + n_mod * x.foton_routed} blok KURAR ({shared} paylaşılan + {n_mod} modül × "
        f"{x.foton_routed} katman), ama bir dizi bunların yalnızca {L}'inden geçer",
        f"`cfg.n_layer` = {L} katmanı sayar",
        f"Paylaşılan {shared} blok tam nedenseldir",
        f"uygulamak c yerine {full_c} sayar",
        f"\"ortak hedef {float(arm_spec_c('x', T, nh)):.2f}\"",
        f"(tablodaki {fwd['h']['foton_attn_routed']:,} ve {fwd['x']['foton_attn_routed']:,})",
        f"nedensel maske altında {old['A']:.2f} / {old['B']:.2f} / {old['C']:.2f} bağlantı",
        f"SAYILMIŞ c'si {FOTON_CONN_TARGET} ± {FOTON_CONN_TOL} dışındaysa",
        f"Router, ilk k = {x.foton_prefix_k} konumun ortalaması",
        f"2·d·n_m = {2 * d * n_mod:,} FLOP",
        f"`round(2·d·n_m / T)` = {fwd['x']['foton_router']} (`foton_router`)",
        f"ön-kayıttaki {_dotted(train['d'])} ve {_dotted(train['m'])}",
        f"yönlendirilen {x.foton_routed} katman {full_c} yerine ~{FOTON_CONN_TARGET} bağlantı sayar",
        f"ölçülen {L * 4 * d * T * 7 // 2:,} = {L} · 4dT · 3.5",
    ]
    # B13'ün 5 puan kuralının okunduğu sayılar: üretilen öngörü tablosunun "fark (%)" sütunundan
    pred_table = next(t for h, t in _md_tables(foton_doc_tables().splitlines()).items() if "ÖNGÖRÜ" in h)
    pred = {cells[1].strip(): float(cells[4]) for cells in (row.split("|") for row in pred_table[2:])}
    assert set(pred) >= {"D", "M (`loop`)", "H", "X"}
    grouped = next(v for k, v in pred.items() if k.startswith("M (`grouped`"))
    expected.append(f"\"baseline'ın farkı\" {grouped:+.1f} puan, FOTON'unki {pred['X']:+.1f} puan okunur")
    assert round(pred["H"], 1) == round(pred["X"], 1)                  # metin tek bir FOTON sayısı veriyor
    assert pred["H"] - grouped > 5 and pred["X"] - grouped > 5         # "5 puanı kat kat aşar": kural tetiklenir
    assert max(pred["H"], pred["X"]) - min(pred["D"], pred["M (`loop`)"]) < 1      # "1 puanın altındadır"
    missing = [s for s in expected if s not in text]
    assert not missing, "docs/flops.md bölüm 8 metni hesaplanan değerlerle uyuşmuyor:\n" + "\n".join(missing)
