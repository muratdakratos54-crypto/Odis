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
from src.stats import (T975, Obs, decide, multiplicity, paired, t_crit, t_sf)  # noqa: E402

SEEDS10 = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]        # ön-kayıtlı karar seed'leri (EK1-B3)


def T(values, cens=(), slow=()) -> dict:
    """Değer listesini seed KİMLİĞİYLE anahtarlı tabloya çevirir; `cens`/`slow` = o KONUMDAKİ seed'in durumu.

    `src.stats` liste kabul etmez (konuma göre eşleme kapalı); testler senaryoyu liste olarak
    yazıp buradan geçirir. Kimlikler ön-kayıtlı seed'lerdir.
    """
    assert len(values) <= len(SEEDS10)
    return {s: Obs(v, "sansürlü" if i in cens else "yavaş" if i in slow else "hızlı")
            for i, (s, v) in enumerate(zip(SEEDS10, values))}

# "Duvar saati her iki baseline'a karşı da pozitif" hâli. Ön-kayıt "güçlü başarı"yı `w̄(M) > 0`
# VE `w̄(D) > 0` şartına bağlıyor, bu yüzden testler skaler değil EŞLEME geçiyor. Skaler geçmek
# artık o satırı ERİŞİLEMEZ kılıyor (bkz. test_scalar_wbar_cannot_reach_strong_success): tek
# skalerin hangi baseline'a ait olduğu bilinmediği için ikisinin de pozitif olduğu kurulamaz.
W_BOTH = {"M": 1.0, "D": 1.0}


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
    a = {11: 1.0, 22: 2.0, 33: 3.0}
    b = {33: 2.5, 11: 0.5, 22: 1.5}                          # aynı seed'ler, başka EKLEME sırası
    d = paired(a, b)
    assert d["per_seed"] == [0.5, 0.5, 0.5] and d["n"] == 3
    assert d["seeds"] == [11, 22, 33] and d["dropped"] == []
    # konuma göre eşleme bu girdide [-1.5, 1.5, 1.5] verirdi; imza onu İFADE EDEMEZ:
    assert [x - y for x, y in zip(a.values(), b.values())] != d["per_seed"]
    for bad in ([1.0, 2.0, 3.0], (1.0, 2.0, 3.0)):
        with pytest.raises(TypeError, match="anahtarlanmış eşleme"):
            paired(bad, b)
        with pytest.raises(TypeError, match="anahtarlanmış eşleme"):
            paired(a, bad)


def test_paired_drops_unmatched_seeds():
    b = {11: 0.5, 22: 1.5, 33: 2.5}
    d = paired({11: 1.0, 33: 3.0}, b)                        # 22'nin koşusu eksik
    assert d["n"] == 2 and d["seeds"] == [11, 33] and d["per_seed"] == [0.5, 0.5]
    assert d["dropped"] == [22]                              # düşen seed SESSİZ değil
    none = paired({99: 1.0}, b)
    assert none["n"] == 0 and none["dropped"] == [11, 22, 33, 99]   # hiç eşleşme yok


def test_paired_ci_uses_t_of_n_minus_one():
    """v3'te bulunan hata: sabit t(2). n küçüldükçe GA GENİŞLEMELİ."""
    vals5, vals3 = [4.0, 5.0, 6.0, 5.0, 5.0], [4.0, 5.0, 6.0]
    s5 = [11, 22, 33, 44, 55]
    d5 = paired(dict(zip(s5, vals5)), dict.fromkeys(s5, 0.0))
    d3 = paired(dict(zip(s5[:3], vals3)), dict.fromkeys(s5[:3], 0.0))
    hw = lambda d: d["ci95"][1] - d["ci95"][0]
    assert hw(d3) > hw(d5)
    # elle kontrol: n=3, sd=1 -> yarı-genişlik t(2)*1/sqrt(3)
    assert hw(d3) / 2 == pytest.approx(T975[2] * 1.0 / math.sqrt(3), rel=1e-9)
    assert hw(d5) / 2 == pytest.approx(T975[4] * st.stdev(vals5) / math.sqrt(5), rel=1e-9)


def test_paired_zero_variance_has_no_p_but_has_ci():
    d = paired(dict.fromkeys([11, 22, 33], 3.0), dict.fromkeys([11, 22, 33], 0.0))
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
    assert decide(T(_tight(12.0)), quality_per_sec=W_BOTH)["row"] == "güçlü başarı"
    assert decide(T(_tight(7.0)), quality_per_sec=W_BOTH)["row"] == "başarılı"
    assert decide(T(_tight(3.0)), quality_per_sec=W_BOTH)["row"] == "belirsiz"
    assert decide(T(_tight(0.0)), quality_per_sec=W_BOTH)["row"] == "nötr"
    assert decide(T(_tight(-5.0)), quality_per_sec=W_BOTH)["row"] == "başarısız"
    assert decide(T([-10, 0, 10, 2, 3]), quality_per_sec=W_BOTH)["row"] == "yüksek varyans / belirsiz"


def test_decide_strong_needs_positive_quality_per_sec():
    """FLOP kazancı >= %10 ama duvar saati kaybı: 'güçlü başarı' değil, çelişki etiketli 'başarılı'."""
    r = decide(T(_tight(12.0)), quality_per_sec=-3.0)
    assert r["row"] == "başarılı"
    assert any("mimari başarı / sistem başarısızlığı" in n for n in r["notes"])
    assert decide(T(_tight(12.0)), quality_per_sec=None)["row"] == "başarılı"


def test_decide_conflict_label_whenever_wallclock_loses():
    """EK 1 B16: çelişki etiketi s̄ >= 10 şartına bağlı DEĞİL; s̄ > 0 ve w̄ <= 0 olan her durumda eklenir."""
    for mean, row in ((6.0, "başarılı"), (3.0, "belirsiz"), (1.0, "nötr")):
        r = decide(T(_tight(mean)), quality_per_sec=-2.0)
        assert r["row"] == row
        assert any("mimari başarı / sistem başarısızlığı" in n for n in r["notes"]), (mean, r)
    # s̄ <= 0 iken FLOP kazancı yok, çelişki de yok
    assert not any("çelişki" in n for n in decide(T(_tight(-4.0)), quality_per_sec=-2.0)["notes"])
    assert not any("mimari başarı" in n for n in decide(T(_tight(6.0)), quality_per_sec=+2.0)["notes"])


def test_decide_wide_ci_is_a_precision_problem_not_a_scale_problem():
    """s̄ >= 5 ama GA eşiği çözemiyorsa sonuç 'yüksek varyans / belirsiz' — 'belirsiz' DEĞİL.

    Bu bir DAVRANIŞ DEĞİŞİKLİĞİ ve kasıtlı. Eski hâl bu örneği "belirsiz" sayıyordu ve "belirsiz"in
    çaresi bir ÜST ÖLÇEK basamağıdır. Ama buradaki sorun ölçek değil KESİNLİK: s̄ = 5.0, sd = 4.74,
    GA [-0.89, +10.89]. Bu aralık "etki eşiği geçiyor" ile "etki sıfır" arasında ayrım yapamıyor,
    dolayısıyla doğru çare daha büyük model değil daha çok SEED'dir. Ölçek basamağı çıkmak burada
    GPU'yu, üstelik tek yükseltme hakkını, yanlış soruya harcamak olurdu.

    Ölçüm (scripts/rule_reachability.py): kesinlik ölçütü σ = 3'te n=5'te %20.7 tetiklenirken
    n=10'da %0.4'e düşüyor — yani çare kuralı gevşetiyor. Mutlak sd > 5 eşiği bu sınavı geçemiyordu
    (σ = 6'da %59.7 -> %71.5, yani çare kuralı SIKIŞTIRIYORDU).
    """
    xs = [5.0 + d for d in (-6.0, -3.0, 0.0, 3.0, 6.0)]      # ortalama 5, sd 4.74
    r = decide(T(xs), quality_per_sec=W_BOTH)
    assert r["mean"] == pytest.approx(5.0) and r["sd"] == pytest.approx(4.743, abs=1e-3)
    assert r["ci95"][0] <= 0 < 5.0 <= r["ci95"][1]
    assert r["row"] == "yüksek varyans / belirsiz" and r["determinate"] is False
    # Daralt: aynı ortalama, dar aralık -> artık belirli ve 'başarılı'
    assert decide(T(_tight(5.0)), quality_per_sec=W_BOTH)["row"] == "başarılı"


def test_decide_boundaries_are_inclusive_as_written():
    assert decide(T(_tight(5.0)), quality_per_sec=W_BOTH)["row"] == "başarılı"       # >= %5
    assert decide(T(_tight(10.0)), quality_per_sec=W_BOTH)["row"] == "güçlü başarı"  # >= %10
    assert decide(T(_tight(2.0)), quality_per_sec=W_BOTH)["row"] == "nötr"           # -2..+2 bandı kapalı
    assert decide(T(_tight(-2.0)), quality_per_sec=W_BOTH)["row"] == "nötr"
    assert decide(T(_tight(2.001)), quality_per_sec=W_BOTH)["row"] == "belirsiz"


def test_decide_one_censored_seed_blocks_success():
    xs = _tight(12.0)
    r = decide(T(xs, cens=[4]), quality_per_sec=W_BOTH)
    assert r["row"] == "belirsiz" and r["n_censored"] == 1
    assert any("sansürlü" in n and "LEHİNE" not in n for n in r["notes"])


def test_decide_two_censored_seeds_fail_outright():
    """EK 1 B8: >= 2 sansürlü seed -> doğrudan 'başarısız', tripwire'a bile bakılmaz."""
    xs = [8.0, 9.0, 8.5, -150.0, -150.0]
    r = decide(T(xs, cens=[3, 4]), quality_per_sec=W_BOTH)
    assert r["row"] == "başarısız" and r["n_censored"] == 2
    assert r["sd"] > 5.0                                     # ham sd tripwire'ı aşıyor ama satır 0 DEĞİL


def test_decide_tripwire_uses_uncensored_sd_only():
    """EK 1 B8: sansürlü seed sd'yi şişirip sonucu 'yüksek varyans / belirsiz' gibi göstermemeli.

    Sansürlü seed'in s_i'si tanım gereği 100*(1 - 2/0.8) = -150 puandır (2x bütçe ikamesi), yani n=5'te
    tek bir sansür ortalamayı -25'e çeker ve sonuç zaten 'başarısız' olur. Kritik olan şudur: sonuç
    'varyans yüksek, daha çok seed lazım' DEĞİL, 'FOTON hedefe ulaşamıyor' demeli."""
    xs = [6.0, 6.1, 5.9, 6.05, -150.0]
    r = decide(T(xs, cens=[4]), quality_per_sec=W_BOTH)
    assert r["sd"] > 5.0 and r["sd_fast"] < 0.2              # ham sd büyük, hızlı seed'lerin sd'si küçük
    assert r["row"] != "yüksek varyans / belirsiz"           # B8'in engellediği yanlış okuma
    assert r["row"] == "başarısız" and r["n_censored"] == 1
    # Sansür olmadan aynı büyüklükte yayılım GERÇEKTEN tripwire üretir (mekanizma çalışıyor).
    assert decide(T([-10, 0, 10, 2, 3]), quality_per_sec=W_BOTH)["row"] == "yüksek varyans / belirsiz"
    # Ve tek sansür, ortalama ezilmese bile başarıyı engeller (kemer + askı).
    r2 = decide(T([12.0, 12.1, 11.9, 12.05, 12.0], cens=[4]), quality_per_sec=W_BOTH)
    assert r2["row"] == "belirsiz"


def test_decide_rejects_positional_input_and_bad_status():
    """Liste/tuple ÇALIŞMAZ: durum artık ayrı bir maske değil, değerle aynı kayıtta (`Obs`)."""
    for bad in (_tight(6.0), tuple(_tight(6.0))):
        with pytest.raises(TypeError, match="anahtarlanmış eşleme"):
            decide(bad, quality_per_sec=W_BOTH)
    with pytest.raises(ValueError, match="status"):
        Obs(1.0, "censored")
    with pytest.raises(ValueError, match="aşamaz"):
        Obs(150.0)                                           # tasarruf +100 puanı aşamaz (birim hatası)
    with pytest.raises(TypeError):
        Obs(True)


def test_decide_row_names_are_from_the_frozen_list():
    from src.stats import ROWS
    for mean in (-8, -2, 0, 3, 6, 12):
        assert decide(T(_tight(float(mean))), quality_per_sec=W_BOTH)["row"] in ROWS


def test_decide_needs_at_least_two_seeds():
    with pytest.raises(ValueError):
        decide(T([5.0]))
