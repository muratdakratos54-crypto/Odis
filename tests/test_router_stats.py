"""v3 ablasyonunun istatistik ve null (taban) hesaplarının denetimi (CPU).

Bu dosya, v3 denetiminin bulduğu dört sorunun testleridir:
  * eşleştirilmiş fark seed KİMLİĞİNE göre yapılmalı (konuma göre zip, eksik analizde yanlış seed eşliyordu)
  * t kritik değeri n'e bağlı olmalı (sabit t(2), n=2'de GA'yı 3 kat daraltıyordu)
  * çoklu karşılaştırma düzeltmesi (Holm / Benjamini-Hochberg) hesaplanmalı
  * ham `consistency` ve `mi_bits` koşullar arası karşılaştırılamaz: kendi bağımsızlık NULL'larına göre okunmalı.
    Ayrıca `expert_diversity`'nin null'ı TAVAN (eğitilmemiş expert'ler ~1.0), yani yüksek değeri uzmanlaşma
    kanıtı olamaz; ve "belirleyici kanıt" dediğimiz ΔCE permütasyonun sıfır noktası sınanmalı.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.config import PRESETS  # noqa: E402
from src.model import GPT  # noqa: E402
from src.moe import MoE  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ra = _load("router_analysis")
cr = _load("compare_router")


# --- istatistik ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("df,t", sorted(cr.T975.items()))
def test_t_critical_values_give_five_percent(df, t):
    """Tablodaki her t kritik değeri, kendi df'inde tam olarak iki yanlı %5 vermeli."""
    assert cr._t_sf(t, df) * 2 == pytest.approx(0.05, abs=1e-6)


def test_t_sf_known_values():
    assert cr._t_sf(1.0, 2) * 2 == pytest.approx(0.42264973, abs=1e-7)   # tablo değeri
    assert cr._t_sf(0.0, 3) * 2 == pytest.approx(1.0, abs=1e-12)
    assert cr._t_sf(2.919986, 2) * 2 == pytest.approx(0.1, abs=1e-6)


def test_paired_matches_by_seed_not_position():
    a, seeds_a = [1.0, 2.0, 3.0], [1337, 1338, 1339]
    b_shuffled, seeds_b = [2.5, 0.5, 1.5], [1339, 1337, 1338]
    d = cr.paired(a, b_shuffled, seeds_a, seeds_b)
    assert d["per_seed"] == [0.5, 0.5, 0.5] and d["n"] == 3
    # Konuma göre zip yanlış cevabı verirdi: sıralama bozuk olduğu için farklar eşit olmazdı.
    assert cr.paired(a, b_shuffled)["per_seed"] != [0.5, 0.5, 0.5]


def test_paired_drops_unmatched_seed_and_widens_ci():
    b, seeds_b = [0.5, 1.5, 2.5], [1337, 1338, 1339]
    full = cr.paired([1.0, 2.0, 3.0], b, seeds_b, seeds_b)
    part = cr.paired([1.0, 3.0], b, [1337, 1339], seeds_b)          # s1338'in analizi eksik
    assert part["n"] == 2 and part["per_seed"] == [0.5, 0.5]
    # Aynı (sıfır) sd'de n=2'nin GA'sı n=3'ten geniş olmalı; sabit t(2) bunu tersine çeviriyordu.
    wide = cr.paired([1.0, 3.5], b, [1337, 1339], seeds_b)
    narrow = cr.paired([1.0, 2.5, 3.5], b, seeds_b, seeds_b)
    assert (wide["ci95"][1] - wide["ci95"][0]) > (narrow["ci95"][1] - narrow["ci95"][0])
    assert full["n"] == 3


def test_multiplicity_holm_and_bh():
    tests = [("A", 0.001), ("B", 0.01), ("C", 0.04), ("D", 0.5)]
    m = cr.multiplicity(tests)
    assert m["n_tests"] == 4 and m["nominal"] == ["A", "B", "C"]
    assert m["holm"] == ["A", "B"]          # .001<=.0125, .01<=.0167, .04>.025 -> dur
    assert m["bh"] == ["A", "B"]            # en büyük k: .01 <= .05*2/4
    assert cr.multiplicity([("X", 0.2)])["holm"] == []
    assert cr.multiplicity([("X", float("nan"))])["n_tests"] == 0   # sd=0 dejenere testler havuza girmez


def test_multiplicity_holm_never_exceeds_nominal():
    rng = np.random.default_rng(0)
    tests = [(f"t{i}", float(p)) for i, p in enumerate(rng.uniform(0, 1, 200))]
    m = cr.multiplicity(tests)
    assert set(m["holm"]) <= set(m["bh"]) <= set(m["nominal"])


# --- null'lar: ham değer değil, tabana göre okunmalı -----------------------------------------------------------

def test_consistency_null_rises_with_imbalance():
    """Ham tutarlılık, routing rastgele olsa bile yük dengesizliğiyle yükselir: null olmadan yanıltır."""
    V, E, n = 200, 4, 100
    rng = np.random.default_rng(0)
    perfect = torch.zeros(V, E, dtype=torch.long)
    for t in range(V):
        perfect[t, t % E] = n
    balanced_random = torch.from_numpy(rng.multinomial([n] * V, [1 / E] * E))
    skewed_random = torch.from_numpy(rng.multinomial([n] * V, [0.7, 0.2, 0.07, 0.03]))

    # mükemmel uzmanlaşma: null'un çok üstünde
    assert ra.routing_consistency(perfect) == pytest.approx(1.0)
    assert ra.consistency_null(perfect) < 0.45
    # dengeli + rastgele: ham değer ~ null
    assert ra.routing_consistency(balanced_random) == pytest.approx(ra.consistency_null(balanced_random), abs=0.02)
    # DENGESİZ + rastgele: ham değer YÜKSEK (~0.7) ama null'u da aynı -> "yüksek tutarlılık" artefakt
    raw = ra.routing_consistency(skewed_random)
    assert raw > 0.6 and raw == pytest.approx(ra.consistency_null(skewed_random), abs=0.02)


def test_mi_null_is_plugin_floor():
    """I(T;E)'nin plug-in tahmincisi bağımsızlıkta bile sıfır değil; taban token bütçesiyle küçülür."""
    V, E = 500, 8
    rng = np.random.default_rng(1)
    few = torch.from_numpy(rng.multinomial([20] * V, [1 / E] * E))
    many = torch.from_numpy(rng.multinomial([2000] * V, [1 / E] * E))
    n_few = ra.mi_null(few)["mi_bits_null"]
    n_many = ra.mi_null(many)["mi_bits_null"]
    assert n_few > n_many > 0                                        # yanlılık bütçeyle azalır
    assert ra.mutual_information(few)["mi_bits"] == pytest.approx(n_few, abs=0.05)
    # H(E|T) tabanı: bağımsızlıkta H(E)'ye yakın olmalı (bilgi yok)
    null = ra.mi_null(many)
    assert null["h_cond_null"] == pytest.approx(math.log2(E), abs=0.05)


def test_expert_diversity_null_is_the_ceiling():
    """Eğitilmemiş expert'ler ~1.0 veriyor: yüksek çeşitlilik uzmanlaşma kanıtı DEĞİL, eğitim bunu düşürür."""
    from src.model import SwiGLU
    torch.manual_seed(0)
    moe = MoE(64, 8, 2, make_expert=lambda: SwiGLU(64, 128), make_shared=None, impl="loop")
    h = torch.randn(256, 64)
    fresh = ra.expert_output_diversity(moe, h)
    assert fresh == pytest.approx(1.0, abs=0.05)
    with torch.no_grad():  # birbirinin kopyası -> 0
        for e in moe.experts[1:]:
            e.load_state_dict(moe.experts[0].state_dict())
    assert ra.expert_output_diversity(moe, h) == pytest.approx(0.0, abs=1e-6)


# --- ΔCE permütasyonun sıfır noktası (kaydın "belirleyici kanıt"ı) ---------------------------------------------

def test_permutation_cost_is_zero_when_experts_are_identical():
    """Expert'ler birbirinin kopyasıysa hangi tokenın hangisine gittiği önemsizdir: ΔCE permütasyon ~ 0.
    Ölçütün sıfır noktası budur; gerçek koşullarda 1.06-1.20 nat okunması bu tabanla anlam kazanır."""
    cfg = PRESETS["tiny-moe"][0]
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, MoE):
                for e in m.experts[1:]:
                    e.load_state_dict(m.experts[0].state_dict())
    g = torch.Generator().manual_seed(0)
    batches = [(torch.randint(0, cfg.vocab_size, (2, 32), generator=g),
                torch.randint(0, cfg.vocab_size, (2, 32), generator=g)) for _ in range(2)]
    out = ra.counterfactual_ce(model, batches, "cpu")
    assert abs(out["delta_permuted"]) < 1e-4, out
    assert abs(out["delta_random"]) < 1e-2                  # gate değerleri değiştiği için tam sıfır değil

    # Kontrol: expert'ler farklıysa aynı kurulumda permütasyon maliyeti sıfırdan belirgin uzaklaşır.
    torch.manual_seed(0)
    varied = GPT(cfg).eval()
    out2 = ra.counterfactual_ce(varied, batches, "cpu")
    assert abs(out2["delta_permuted"]) > 20 * max(abs(out["delta_permuted"]), 1e-6)


def test_next_mode_refuses_when_experts_too_few():
    """'next' kipi k tane 'sıradaki' expert gerektirir; E < 2k'da sessiz yanlış eşleşme yerine hata."""
    from src.model import SwiGLU
    moe = MoE(32, 4, 3, make_expert=lambda: SwiGLU(32, 64), make_shared=None, impl="loop")
    logits = torch.randn(10, 4)
    with pytest.raises(ValueError, match="E >= 2k"):
        ra.route(moe, logits, "next")
    ra.route(moe, logits, "learned")        # diğer kipler etkilenmez
