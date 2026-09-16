"""src/stats.py regresyon testleri — FOTON kararının geçerliliği bunlara bağlı.

PREREGISTRATION.md'nin karar kuralı eşleştirilmiş güven aralığının İŞARETİNE bakıyor. v3 denetimi bu
koddaki iki hatayı buldu ve ikisi de karar kuralını sessizce geçersiz kılardı:

  * eşleştirme seed kimliği yerine liste konumuna göre yapılıyordu
  * t kritik değeri n'den bağımsız sabitti

Bu dosya ikisini de regresyona bağlar; ayrıca karar tablosunun her satırını ve sınır durumlarını sınar.
"""
from __future__ import annotations

import math
import os
import statistics as st
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.stats import (T975, decide, multiplicity, paired, t_crit, t_sf)  # noqa: E402


# --- t dağılımı -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("df,t", sorted(T975.items()))
def test_t_table_is_the_975_quantile(df, t):
    assert t_sf(t, df) * 2 == pytest.approx(0.05, abs=1e-6)


def test_t_sf_known_values():
    assert t_sf(1.0, 2) * 2 == pytest.approx(0.42264973, abs=1e-7)
    assert t_sf(2.919986, 2) * 2 == pytest.approx(0.1, abs=1e-6)
    assert t_sf(0.0, 3) * 2 == pytest.approx(1.0, abs=1e-12)
    assert t_sf(1.959964, 10_000) * 2 == pytest.approx(0.05, abs=1e-4)   # büyük df -> normal


def test_t_crit_depends_on_df_and_is_conservative_when_missing():
    assert t_crit(4) == T975[4]
    assert t_crit(1) > t_crit(2) > t_crit(4) > t_crit(9)      # df büyüdükçe daralır
    assert t_crit(12) == T975[10]                            # tabloda yok -> en yakın ALT df (muhafazakâr)
    assert t_crit(1000) == T975[29]


# --- REGRESYON: eşleştirme seed kimliğine göre ------------------------------------------------------------

def test_paired_matches_by_seed_identity_not_position():
    """v3'te bulunan hata: konuma göre zip. Sıralar farklıysa yanlış seed'ler eşlenir."""
    a, seeds_a = [1.0, 2.0, 3.0], [11, 22, 33]
    b, seeds_b = [2.5, 0.5, 1.5], [33, 11, 22]               # aynı seed'ler, başka sıra
    d = paired(a, b, seeds_a, seeds_b)
    assert d["per_seed"] == [0.5, 0.5, 0.5] and d["n"] == 3
    assert d["seeds"] == [11, 22, 33]
    assert paired(a, b)["per_seed"] != [0.5, 0.5, 0.5]       # konuma göre: yanlış cevap


def test_paired_drops_unmatched_seeds():
    b, seeds_b = [0.5, 1.5, 2.5], [11, 22, 33]
    d = paired([1.0, 3.0], b, [11, 33], seeds_b)             # 22'nin koşusu eksik
    assert d["n"] == 2 and d["seeds"] == [11, 33] and d["per_seed"] == [0.5, 0.5]
    assert paired([1.0], b, [99], seeds_b)["n"] == 0         # hiç eşleşme yok


def test_paired_ci_uses_t_of_n_minus_one():
    """v3'te bulunan hata: sabit t(2). n küçüldükçe GA GENİŞLEMELİ."""
    vals5, vals3 = [4.0, 5.0, 6.0, 5.0, 5.0], [4.0, 5.0, 6.0]
    s5 = [11, 22, 33, 44, 55]
    d5 = paired(vals5, [0.0] * 5, s5, s5)
    d3 = paired(vals3, [0.0] * 3, s5[:3], s5[:3])
    hw = lambda d: d["ci95"][1] - d["ci95"][0]
    assert hw(d3) > hw(d5)
    # elle kontrol: n=3, sd=1 -> yarı-genişlik t(2)*1/sqrt(3)
    assert hw(d3) / 2 == pytest.approx(T975[2] * 1.0 / math.sqrt(3), rel=1e-9)
    assert hw(d5) / 2 == pytest.approx(T975[4] * st.stdev(vals5) / math.sqrt(5), rel=1e-9)


def test_paired_zero_variance_has_no_p_but_has_ci():
    d = paired([3.0, 3.0, 3.0], [0.0, 0.0, 0.0], [11, 22, 33], [11, 22, 33])
    assert d["sd"] == 0 and "p" not in d and d["ci95"] == [3.0, 3.0]


# --- çoklu karşılaştırma ----------------------------------------------------------------------------------

def test_multiplicity_holm_and_bh():
    m = multiplicity([("A", 0.001), ("B", 0.01), ("C", 0.04), ("D", 0.5)])
    assert m["n_tests"] == 4 and m["nominal"] == ["A", "B", "C"]
    assert m["holm"] == ["A", "B"]      # .001<=.0125, .01<=.0167, .04>.025 -> dur
    assert m["bh"] == ["A", "B"]        # en büyük k: .01 <= .05*2/4
    assert multiplicity([("X", float("nan"))])["n_tests"] == 0
    assert set(m["holm"]) <= set(m["bh"]) <= set(m["nominal"])


def test_multiplicity_matches_v3_audit_shape():
    """v3'te 151 testten 82'si nominal anlamlıydı, Holm sonrası 9 kaldı: çok test -> sert budama.
    Burada aynı şekil sentetik olarak üretiliyor: Holm < BH < nominal, hepsi test sayısından çok küçük."""
    ps = [1e-8] * 3 + [0.002] * 6 + [0.03] * 10 + [0.2 + 0.005 * i for i in range(132)]
    m = multiplicity([(f"t{i}", p) for i, p in enumerate(ps)])
    assert m["n_tests"] == 151
    assert len(m["nominal"]) == 19           # p <= .05 olanlar
    assert len(m["holm"]) == 3 and len(m["bh"]) == 9
    assert len(m["holm"]) < len(m["bh"]) < len(m["nominal"]) < m["n_tests"]


# --- karar tablosu ----------------------------------------------------------------------------------------

def _tight(mean: float, n: int = 5) -> list[float]:
    """Verilen ortalamada, GA'yı sıfırın üstünde tutacak kadar dar bir örnek."""
    step = 0.05
    xs = [mean + step * (i - (n - 1) / 2) for i in range(n)]
    assert st.mean(xs) == pytest.approx(mean)
    return xs


def test_decide_rows():
    assert decide(_tight(12.0), quality_per_sec=1.0)["row"] == "güçlü başarı"
    assert decide(_tight(7.0), quality_per_sec=1.0)["row"] == "başarılı"
    assert decide(_tight(3.0), quality_per_sec=1.0)["row"] == "belirsiz"
    assert decide(_tight(0.0), quality_per_sec=1.0)["row"] == "nötr"
    assert decide(_tight(-5.0), quality_per_sec=1.0)["row"] == "başarısız"
    assert decide([-10, 0, 10, 2, 3], quality_per_sec=1.0)["row"] == "yüksek varyans / belirsiz"


def test_decide_strong_needs_positive_quality_per_sec():
    """FLOP kazancı >= %10 ama duvar saati kaybı: 'güçlü başarı' değil, çelişki etiketli 'başarılı'."""
    r = decide(_tight(12.0), quality_per_sec=-3.0)
    assert r["row"] == "başarılı"
    assert any("mimari başarı / sistem başarısızlığı" in n for n in r["notes"])
    assert decide(_tight(12.0), quality_per_sec=None)["row"] == "başarılı"


def test_decide_wide_ci_downgrades_to_belirsiz():
    """s̄ >= 5 ama GA sıfırı kesiyorsa 'başarılı' DEĞİL (tablo: 'veya GA sıfırı kesiyor -> belirsiz')."""
    xs = [5.0 + d for d in (-6.0, -3.0, 0.0, 3.0, 6.0)]      # ortalama 5, sd 4.74: tripwire altı ama geniş
    r = decide(xs, quality_per_sec=1.0)
    assert r["mean"] == pytest.approx(5.0) and r["sd"] < 5.0
    assert r["ci95"][0] <= 0 and r["row"] == "belirsiz"


def test_decide_boundaries_are_inclusive_as_written():
    assert decide(_tight(5.0), quality_per_sec=1.0)["row"] == "başarılı"       # >= %5
    assert decide(_tight(10.0), quality_per_sec=1.0)["row"] == "güçlü başarı"  # >= %10
    assert decide(_tight(2.0), quality_per_sec=1.0)["row"] == "nötr"           # -2..+2 bandı kapalı
    assert decide(_tight(-2.0), quality_per_sec=1.0)["row"] == "nötr"
    assert decide(_tight(2.001), quality_per_sec=1.0)["row"] == "belirsiz"


def test_decide_censored_seeds_block_success():
    r = decide(_tight(12.0), quality_per_sec=1.0, censored=1)
    assert r["row"] == "belirsiz" and any("sansürlü" in n for n in r["notes"])
    r2 = decide(_tight(-5.0), quality_per_sec=1.0, censored=2)
    assert r2["row"] == "başarısız"                          # sansür yalnızca başarıyı engeller


def test_decide_row_names_are_from_the_frozen_list():
    from src.stats import ROWS
    for mean in (-8, -2, 0, 3, 6, 12):
        assert decide(_tight(float(mean)), quality_per_sec=1.0)["row"] in ROWS


def test_decide_needs_at_least_two_seeds():
    with pytest.raises(ValueError):
        decide([5.0])
