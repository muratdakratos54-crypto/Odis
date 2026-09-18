"""v02 denetiminin bulduğu delikleri regresyona bağlar — `src/stats.py`'nin YENİ parçaları.

Dört mercekli adversaryal denetim `PREREGISTRATION_V02.md`'de karar kuralına dair şu delikleri buldu;
her biri aşağıda en az bir testle kilitli:

  * `n_star` hiçbir yerde uygulanmamıştı; var olan tek uygulama EK1-B14'ün "kesişim sürdürülmüş
    olmalı" şartını atlıyordu. Ölçülü etki: 2750'de sarkıp geri çıkan bir eğride `s_i` ~21 puan
    şişiyor, "nötr" bir sonuç "güçlü başarı" oluyor.
  * `s_i`'yi kuran hiçbir fonksiyon yoktu; tek uygulama eşleştirmeyi liste KONUMUNA göre yapıyordu —
    yani `paired()`'ın regresyona bağladığı hatanın aynısı, birincil istatistiğin tam üstünde.
  * `decide()` iki seed'le "başarılı" damgası basıyordu; kilit cümle BEŞ seed istiyor.
  * tek sansürsüz seed varken tripwire ham sd'ye düşüyor ve "sansürsüz seed'lerin sd'si 110.31"
    diye VAR OLMAYAN bir niceliği raporluyordu.
  * çelişki etiketi yalnızca `mean > 0` ile tetikleniyordu: s̄ = +0.01 puanlık gürültüye
    "mimari başarı" damgası vurulabiliyordu.
  * `quality_per_sec=None` sessizce geçiyordu; rapor ikincil ölçütün hiç değerlendirilmediğini
    göstermiyordu.
  * `multiplicity` aile büyüklüğünü p üretebilen test sayısından alıyordu, yani hangi teşhis
    ölçütünün hesaplanabildiği Holm'un sıkılığını SONRADAN değiştirebiliyordu.
  * vazgeçme koşulu σ >~ 1.72'de aritmetik olarak ulaşılamazdı (v01 B10 hatasının ikinci kez
    ortaya çıkışı).
"""
from __future__ import annotations

import math
import os
import statistics as st
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.stats import (  # noqa: E402
    SUCCESS,
    abandon,
    ci95,
    decide,
    indeterminate,
    multiplicity,
    n_star,
    reachability,
    savings,
    t_crit,
)

# "Duvar saati her iki baseline'a karşı da pozitif" hâli. Ön-kayıt "güçlü başarı"yı `w̄(M) > 0`
# VE `w̄(D) > 0` şartına bağlıyor, bu yüzden testler skaler değil EŞLEME geçiyor. Skaler geçmek
# artık o satırı ERİŞİLEMEZ kılıyor (bkz. test_scalar_wbar_cannot_reach_strong_success): tek
# skalerin hangi baseline'a ait olduğu bilinmediği için ikisinin de pozitif olduğu kurulamaz.
W_BOTH = {"M": 1.0, "D": 1.0}

STEP = 32768


def _curve(vals: list[float], first: int = 0, interval: int = 250) -> list[tuple[float, float]]:
    """(token, val) eğrisi: `interval` adımlık eval ızgarası."""
    return [((first + i * interval) * STEP, v) for i, v in enumerate(vals)]


# --------------------------------------------------------------------------- n_star / B14

def test_n_star_interpolates_linearly_in_tokens():
    """Kural PİNLİ: token ekseninde doğrusal. Tam ortada kesen bir eğri ortayı vermeli."""
    c = _curve([3.0, 2.0])                       # 0 -> 8.192M token arası
    assert n_star(c, 2.5) == pytest.approx(0.5 * 250 * STEP)


def test_n_star_returns_none_when_target_never_reached():
    """Hedefe ulaşılmazsa None; çağıran taraf bunu SANSÜR olarak işlemek zorunda."""
    assert n_star(_curve([4.0, 3.5, 3.2, 3.1]), 2.0) is None


def test_n_star_requires_sustained_crossing_b14():
    """EK1-B14: kesişimden sonra hedefin ÜSTÜNE çıkan bir eğride tek bir `N*` anlamlı değildir.

    Bu, denetimin en somut karar-bozan bulgusuydu. Eğri 2750'de hedefin altına sarkıp 3000'de geri
    çıkıyor, 3500'de kalıcı kesiyor. Sürdürme kontrolü OLMADAN ilk kesişim okunur ve `N*` ~%21
    küçük çıkar; kontrol VARSA seed sansürlenir. Aradaki fark tek bir `if`.
    """
    #                 2500  2750  3000  3250  3500  3750
    vals = [2.95, 2.89, 2.93, 2.92, 2.88, 2.86]   # hedef 2.90: 2750'de sark, 3000'de geri çık
    c = _curve(vals, first=2500)
    assert n_star(c, 2.90) is None
    # Aynı eğride sarkma OLMASA kesişim geçerli ve erken:
    ok = _curve([2.95, 2.89, 2.88, 2.87, 2.86, 2.85], first=2500)
    got = n_star(ok, 2.90)
    assert got is not None and 2500 * STEP < got < 2750 * STEP


def test_n_star_sustained_check_allows_dip_before_crossing():
    """Kesişimden ÖNCEKİ dalgalanma kesişimi geçersiz kılmaz; yalnızca sonrası sayılır."""
    c = _curve([3.1, 3.3, 3.0, 2.8, 2.75], first=1000)     # 1250'de yukarı zıpla, sonra kes
    got = n_star(c, 2.9)
    assert got is not None and got > 1500 * STEP


def test_n_star_handles_flat_curve_exactly_at_target():
    c = _curve([3.0, 2.5, 2.5, 2.4])
    assert n_star(c, 2.5) == pytest.approx(250 * STEP)


# --------------------------------------------------------------------------- savings / s_i

def test_savings_matches_by_seed_identity_not_position():
    """Birincil istatistik seed KİMLİĞİNE göre eşlenmeli.

    Denetimin senaryosu: baseline listesi koşu defterinden kuruluyor ve OOM sonrası yeniden koşulan
    s22 listeye SONDA ekleniyor. Konum eşlemesi FOTON-22'yi baseline-33 ile eşlerdi.
    """
    seeds_f = [11, 22, 33]
    f_foton = [90.0, 80.0, 70.0]
    seeds_b = [11, 33, 22]                                   # s22 sonda!
    f_base = [100.0, 100.0, 200.0]                           # s22'nin baseline'ı 200
    r = savings(f_foton, f_base, seeds_f, seeds_b)
    # s11: 1-90/100 = 10 | s22: 1-80/200 = 60 | s33: 1-70/100 = 30
    assert r["s_i"] == pytest.approx([10.0, 60.0, 30.0])
    assert r["seeds"] == [11, 22, 33]


def test_savings_rejects_mismatched_seed_sets():
    """Eksik/fazla seed SESSİZCE atlanmaz. `paired()` atlıyor; birincil istatistikte atlamak yasak."""
    with pytest.raises(ValueError, match="seed kümeleri"):
        savings([1.0, 2.0], [1.0, 2.0], [11, 22], [11, 33])


def test_savings_rejects_duplicate_seeds():
    with pytest.raises(ValueError, match="tekrar"):
        savings([1.0, 2.0], [1.0, 2.0], [11, 11], [11, 11])


def test_savings_rejects_nonpositive_denominator():
    with pytest.raises(ValueError, match="pozitif"):
        savings([1.0, 2.0], [0.0, 2.0], [11, 22], [11, 22])


def test_savings_is_a_ratio_not_a_difference():
    """`paired()` FARK hesaplar, `s_i` ORAN. Karıştırılmasın: aynı girdide farklı sayılar."""
    r = savings([50.0], [100.0], [11], [11])
    assert r["s_i"] == pytest.approx([50.0])                 # %50 tasarruf, 50 puan "fark" değil


# --------------------------------------------------------------------------- kesinlik ölçütü

def test_indeterminate_is_false_when_success_established():
    assert indeterminate(mean=7.0, lo=1.0, hi=13.0) is False          # lo > 0 ve mean >= 5


def test_indeterminate_is_false_when_success_excluded():
    assert indeterminate(mean=0.0, lo=-4.0, hi=4.0) is False          # hi < 5


def test_indeterminate_is_true_when_ci_straddles_the_threshold():
    """Aralık "eşiği geçiyor" ile "geçmiyor" arasında ayrım yapamıyorsa karar belirsizdir."""
    assert indeterminate(mean=4.9, lo=3.0, hi=6.8) is True            # eşik aralığın içinde
    assert indeterminate(mean=7.0, lo=-1.0, hi=15.0) is True          # büyük ama sıfırı kesiyor


def test_precision_tripwire_remedy_loosens_the_rule():
    """ÇARENİN YÖNÜ: seed artırmak tripwire'ı GEVŞETMELİ, sıkıştırmamalı.

    Eski mutlak sd > 5 eşiği bu sınavı geçemiyordu: n ile ölçeklenmediği için gerçek sd = 6'da
    P(tetiklenme) %59.7'den %71.5'e ÇIKIYORDU — tripwire'ın tek çaresi tripwire'ı daha olası
    yapıyordu, ve tabloda "n=10'da tetiklendi" satırı yoktu. Bu test o örüntünün geri gelmesini
    engeller.
    """
    def fires(n):
        return lambda s: decide(s, quality_per_sec=W_BOTH)["row"] == "yüksek varyans / belirsiz"
    p5 = reachability(fires(5), mu=0.0, sigma=5.0, n=5, sims=3000, seed=1)
    p10 = reachability(fires(10), mu=0.0, sigma=5.0, n=10, sims=3000, seed=1)
    assert p5 > p10, f"çare kuralı sıkıştırıyor: n=5 %{100*p5:.1f} -> n=10 %{100*p10:.1f}"


def test_precision_tripwire_silent_at_measured_noise():
    """Ölçülen sd(s_i) = 0.75 puanda tripwire tetiklenmemeli; yoksa her koşu 10 seed'e giderdi."""
    p = reachability(lambda s: decide(s, quality_per_sec=W_BOTH)["row"] == "yüksek varyans / belirsiz",
                     mu=6.0, sigma=0.75, n=5, sims=3000, seed=2)
    assert p < 0.02, f"ölçülen gürültüde tripwire %{100*p:.1f} tetikleniyor"


# --------------------------------------------------------------------------- decide() sertleştirmeleri

def test_decide_requires_preregistered_seed_count():
    """Kilit cümle BEŞ seed istiyor; `decide()` iki seed'le "başarılı" damgası basıyordu."""
    with pytest.raises(ValueError, match="bağlayıcı karar"):
        decide([6.0, 6.1], quality_per_sec=W_BOTH)
    with pytest.raises(ValueError, match="bağlayıcı karar"):
        decide([6.0, 6.1, 6.05], quality_per_sec=W_BOTH)
    for n in (5, 10):
        xs = [6.0 + 0.01 * i for i in range(n)]
        assert decide(xs, quality_per_sec=W_BOTH)["row"] == "başarılı"


def test_decide_strict_false_blocks_success_rows():
    """Keşifsel kullanım mümkün ama başarı İLAN EDEMEZ."""
    r = decide([6.0, 6.1, 6.05], quality_per_sec=W_BOTH, strict=False)
    assert r["row"] == "belirsiz"
    assert any("strict=False" in n for n in r["notes"])


def test_decide_does_not_invent_an_uncensored_sd_from_one_seed():
    """Tek sansürsüz seed varken sd YOKTUR; eski kod ham sd'yi "sansürsüz sd" diye raporluyordu.

    O rapor, -150'lik sansür ikamesinden gelen 110.31'i "sansürsüz seed'lerin sd'si" olarak
    yazıyordu: var olmayan bir nicelik, ve tam olarak B8'in engellemek için yazıldığı yanlış okuma
    ("hedefe ulaşamıyor" yerine "varyans yüksek, daha çok seed lazım").
    """
    xs = [6.0, 6.1, 6.05, -150.0, -150.0]
    r = decide(xs, quality_per_sec=W_BOTH, censored_mask=[False] * 3 + [True] * 2)
    assert r["row"] == "başarısız"                      # >= 2 sansür: B8 doğrudan başarısız
    # Tek sansür + tek sansürsüz: kesinlik DEĞERLENDİRİLEMEZ, uydurulmaz
    r2 = decide([6.0, -150.0, -150.0, -150.0, -150.0], quality_per_sec=W_BOTH,
                censored_mask=[False] + [True] * 4)
    assert r2["row"] == "başarısız"
    assert not any("110" in n for n in r2["notes"])


def test_decide_one_uncensored_seed_cannot_claim_precision():
    """n=5, 4 sansürlü değil ama sansürsüz sayısı 1 olan kurgu: kesinlik değerlendirilemez."""
    # 1 sansür (B8 sınırı içinde) ama diğer 4'ü de... hayır: sansürsüz >= 2 olması için
    # en fazla 3 sansür gerekir ve B8 zaten >=2'de başarısız veriyor. Bu yüzden tek sansürsüz
    # durumu ancak >=2 sansürle oluşur ve B8 önce devreye girer — kilitlenen şey o sıralama.
    r = decide([7.0, -150.0, -150.0, -150.0, -150.0], quality_per_sec=W_BOTH,
               censored_mask=[False] + [True] * 4)
    assert r["row"] == "başarısız" and r["n_censored"] == 4
    assert any("B8" in n for n in r["notes"])


def test_decide_conflict_label_requires_statistical_flop_gain():
    """B16 çelişki etiketi GÜRÜLTÜYE vurulmamalı: FLOP kazancının GA'sı sıfırın üstünde olmalı.

    Eski kod `mean > 0` ile yetiniyordu, yani s̄ = +0.01 puanlık bir farka "mimari başarı /
    sistem başarısızlığı" damgası basılabiliyordu — ve ön-kayıt bu ifadeyi bir SATIR gibi
    anıyordu, dolayısıyla null bir sonuç mimari başarı iddiasına dönüşebiliyordu.
    """
    noisy = [0.4, 1.2, -0.3, 0.9, 0.1]                  # ortalama 0.46, GA sıfırı kesiyor
    r = decide(noisy, quality_per_sec=-30.0)
    assert r["row"] == "nötr" and r["ci95"][0] < 0
    assert not any("mimari başarı" in n for n in r["notes"])
    assert any("UYGULANMAZ" in n for n in r["notes"])
    # Gerçek ve istatistiksel kazançta etiket VERİLİR
    r2 = decide([6.0, 6.2, 5.9, 6.1, 6.0], quality_per_sec=-30.0)
    assert any("mimari başarı" in n for n in r2["notes"])


def test_decide_reports_missing_wall_clock_metric():
    """`quality_per_sec=None` SESSİZ olmamalı: ikincil ölçüt değerlendirilmedi, kayda geçsin.

    V11a duvar saatini eksik bir seed bloğunda geçersiz saydığı için bu ulaşılabilir bir durum:
    tek bir çöken koşu w̄'yi tanımsız bırakır, ve o hâlde "başarılı" satırının yanında ikincil
    ölçütün hiç bakılmadığının yazılı olması gerekir.
    """
    r = decide(_five(12.0), quality_per_sec=None)
    assert r["row"] == "başarılı"                        # 'güçlü başarı' erişilemez
    assert any("VERİLMEDİ" in n for n in r["notes"])
    assert decide(_five(12.0), quality_per_sec=W_BOTH)["row"] == "güçlü başarı"


def test_decide_ci_lower_bound_exactly_zero_is_not_success():
    """GA alt sınırı matematiksel olarak tam sıfırken kayan nokta bit çevirmesi kararı DEĞİŞTİRMESİN.

    Denetim, `lo`'nun 8.88e-16 çıkıp "başarılı" verdiği bir örnek kurdu. Ölçü sıfır ama bir
    başarı beyanının tek bir bite bağlı olduğu tek yerdi; tolerans eklendi.
    """
    n, mean = 5, 5.0
    sd = mean * math.sqrt(n) / t_crit(n - 1)             # h = mean -> lo = 0
    xs = [mean + sd * z for z in _zscores(n)]
    m, lo, hi = ci95(xs)
    assert m == pytest.approx(mean) and abs(lo) < 1e-9
    assert decide(xs, quality_per_sec=W_BOTH)["row"] != "başarılı"


def test_decide_censored_mask_is_foton_only_by_contract():
    """Sözleşme: maske YALNIZCA FOTON kolunun sansürü. Baseline'ın sansürü girerse felaket.

    v02'de dense kolu `L*`'e ulaşmıyor (5 seed'in 5'i). Maskeye onun sansürü konursa %7 tasarruf
    eden bir FOTON, BASELINE'ın yetersizliği yüzünden "başarısız" ilan edilir. Bu test o sonucu
    gösterir ki sözleşme kodda değil belgede korunduğu unutulmasın.
    """
    xs = _five(7.0)
    assert decide(xs, quality_per_sec=W_BOTH)["row"] == "başarılı"
    wrong = decide(xs, quality_per_sec=W_BOTH, censored_mask=[True] * 5)
    assert wrong["row"] == "başarısız"                   # maske yanlış doldurulursa olan bu


# --------------------------------------------------------------------------- vazgeçme koşulu

def test_abandon_fires_for_a_break_even_foton_at_measured_noise():
    """Ölçülen σ'da başabaş bir FOTON TERK EDİLEBİLİR olmalı. Eski biçim σ>=1.72'de ölüydü."""
    r5 = decide(_five(0.0, sd=0.75), quality_per_sec=-1.0)
    r6 = decide(_five(0.1, sd=0.75), quality_per_sec=-1.0)
    res = abandon([r5, r6], oracle_gain_over_null=0.5)
    assert res["abandon"] is True and "GA üst sınırı" in res["reason"]


def test_abandon_blocked_by_oracle_headroom():
    """Mükemmel yönlendirmede hâlâ kazanç varsa hipotez terk EDİLMEZ — küçük ama gerçek etkiyi korur."""
    rs = [decide(_five(0.0, sd=0.75), quality_per_sec=-1.0) for _ in range(2)]
    res = abandon(rs, oracle_gain_over_null=5.0)
    assert res["abandon"] is False and "oracle" in res["reason"]


def test_abandon_blocked_when_effect_could_still_be_large():
    """GA üst sınırı eşiğin üstündeyse "etki eşik kadar büyük olabilir": terk edilemez."""
    rs = [decide(_five(4.0, sd=1.0), quality_per_sec=-1.0) for _ in range(2)]
    assert all(r["ci95"][1] >= SUCCESS for r in rs)          # GA [2.76, 5.24]
    assert abandon(rs, oracle_gain_over_null=0.0)["abandon"] is False


def test_abandon_fires_on_a_real_but_sub_threshold_effect():
    """ANLAM KARARI, açıkça kilitlenmiştir: +%3'lük GERÇEK bir kazanç da vazgeçmeyi tetikler.

    `s̄ = 3.0, sd = 0.75` -> GA [2.07, 3.93]: etki sıfırdan farklı (kazanç gerçek) ama eşiğin
    tamamen altında. İki ardışık ölçekte böyleyse ve oracle headroom'u < %2 ise ön-kayıtlı hipotez
    TERK EDİLİR. Gerekçe: kilit cümlenin hipotezi "en az %5 tasarruf"tur, "sıfırdan büyük tasarruf"
    değil. İki ölçekte kesin olarak %5'in altında kaldığı ve mükemmel yönlendirmenin de kayda değer
    bir pay bırakmadığı gösterilmişse, ön-kayıtlı iddia çürütülmüştür.

    Terk etmek bulguyu SİLMEK değildir: +%3 sonuç olarak raporlanır. Terk edilen şey "dizi düzeyi
    top-1 yönlendirme %5+ getirir" iddiasıdır.

    Bu testin varlık sebebi: eski biçimdeki `s̄ <= 0` şartı tam bu durumu korumak için vardı ve
    onu attık. Attığımızın farkında olduğumuz burada kayıtlı; oracle şartı tek koruma olarak kalıyor.
    """
    rs = [decide(_five(3.0, sd=0.75), quality_per_sec=-1.0) for _ in range(2)]
    assert all(r["ci95"][0] > 0 for r in rs)                 # kazanç GERÇEK
    assert all(r["ci95"][1] < SUCCESS for r in rs)           # ama eşiğin altında
    assert abandon(rs, oracle_gain_over_null=0.5)["abandon"] is True
    # Oracle headroom varsa korunur:
    assert abandon(rs, oracle_gain_over_null=3.0)["abandon"] is False


def test_abandon_distinguishes_insufficient_precision_from_hypothesis_standing():
    """Kural tetiklenmediğinde "hipotez ayakta" ile "ölçemedik" AYRI raporlanmalı.

    Eski hâlin en sinsi sonucu buydu: geniş varyansta vazgeçme koşulu hiç tetiklenmiyor, sonuç da
    "nötr / sonuçsuz" olarak park ediyordu — yani değersiz bir FOTON süresiz yaşayabiliyordu.
    """
    wide = decide([0.0 + d for d in (-9, -4.5, 0, 4.5, 9)], quality_per_sec=-1.0)
    assert wide["row"] == "yüksek varyans / belirsiz"
    res = abandon([wide, wide], oracle_gain_over_null=0.0)
    assert res["abandon"] is False and "yetersiz kesinlik" in res["reason"]
    assert "ölçemedik" in res["reason"]


def test_abandon_needs_two_scales():
    r = decide(_five(0.0, sd=0.75), quality_per_sec=-1.0)
    assert abandon([r], oracle_gain_over_null=0.0)["abandon"] is False


def test_abandon_reachability_beats_the_old_form():
    """Sayısal sınav: sağlam biçim σ=3'te başabaş FOTON'u eski biçimden ÇOK daha sık terk eder."""
    def robust(s):
        return ci95(s)[2] < SUCCESS

    def old(s):
        m, lo, hi = ci95(s)
        return m <= 0 and hi < 2.0
    pr = reachability(robust, mu=0.0, sigma=3.0, n=5, sims=20000, seed=3)
    po = reachability(old, mu=0.0, sigma=3.0, n=5, sims=20000, seed=3)
    assert pr > 0.7 and po < 0.25, f"sağlam %{100*pr:.1f} vs eski %{100*po:.1f}"


# --------------------------------------------------------------------------- multiplicity

def test_multiplicity_family_size_must_be_pinned():
    """Aile büyüklüğü p ÜRETEBİLEN test sayısından alınırsa Holm sonradan gevşetilebilir.

    Denetimin örneği: p(D2) = 0.02 iken aile 10 ise Holm D2'yi reddeder, aile 2'ye düşerse KABUL
    eder. Yani "hangi teşhis ölçütü hesaplanabildi" sorusunun cevabı anlamlılığı değiştiriyordu —
    ve teşhislerin bir kısmı (entropi, yük CV, ölü modül sayısı) tek sayıdır, null'ı yoktur, p
    üretmez. Aile büyüklüğü ön-kayıttan gelir.
    """
    tests = [("D1", 0.0001), ("D2", 0.02)]
    assert multiplicity(tests) ["holm"] == ["D1", "D2"]              # m=2: D2 geçer
    assert multiplicity(tests, family_size=10)["holm"] == ["D1"]     # m=10: D2 geçmez
    assert multiplicity(tests, family_size=10)["n_tests"] == 10
    assert multiplicity(tests, family_size=10)["n_with_p"] == 2


def test_multiplicity_family_size_cannot_be_smaller_than_reality():
    with pytest.raises(ValueError):
        multiplicity([("a", 0.01), ("b", 0.02), ("c", 0.03)], family_size=2)


# --------------------------------------------------------------------------- yardımcılar

def _zscores(n: int) -> list[float]:
    """Ortalaması 0, örnek sd'si 1 olan n değer."""
    raw = [i - (n - 1) / 2 for i in range(n)]
    s = st.stdev(raw)
    return [r / s for r in raw]


def _five(mean: float, sd: float = 0.05, n: int = 5) -> list[float]:
    """Verilen ortalama ve ÖRNEK sd'siyle tam n değer."""
    return [mean + sd * z for z in _zscores(n)]


def test_helpers_produce_the_requested_moments():
    xs = _five(7.0, sd=1.25)
    assert st.mean(xs) == pytest.approx(7.0) and st.stdev(xs) == pytest.approx(1.25)


# ------------------------------------------------- ikinci denetimin bulduklari

def test_strong_success_needs_both_baselines():
    """"güçlü başarı" için `w̄(M) > 0` VE `w̄(D) > 0` gerekir; eksik ölçüm o satırı ERİŞİLEMEZ kılar.

    Ön-kayıt bu şartı şu gerekçeyle koyuyor: M bu kartta kendi başına D'den token başına %27.9
    yavaştır (16 expert'in AdamW state'i + dispatch), FOTON kollarında ise ne o expert'ler ne o
    dispatch var. Dolayısıyla yalnızca M'e karşı `w̄ > 0`, FOTON'un kendi hız katkısı OLMADAN da
    sağlanabilir — v01 EK1-B16'nın aynı kolda bulduğu sorun.

    İki aşamalı bir hata kilitleniyor:
      1. İlk hâl tek skaler alıyordu ve `decide(s̄=11, quality_per_sec=1.0)` "güçlü başarı"
         döndürüyordu — ön-kaydın "başarılı" dediği durum.
      2. İlk DÜZELTME eşleme kabul etti ama anahtar doğrulaması yapmadı: `{"M": 1.0}` (D hiç
         ölçülmemiş) hem satırı veriyor hem HİÇ not yazmıyordu, çünkü uyarı yalnızca skaler
         yolun "?" anahtarında tetikleniyordu. Blocker sessiz hâliyle geri gelmişti.
    """
    xs = _five(11.0)
    assert decide(xs, quality_per_sec={"M": 1.0, "D": 1.0})["row"] == "güçlü başarı"
    # Bir baseline'a karşı kaybediyorsa satır DÜŞER; şart simetrik
    for w, neg in (({"M": 1.0, "D": -5.0}, "D"), ({"M": -5.0, "D": 1.0}, "M")):
        r = decide(xs, quality_per_sec=w)
        assert r["row"] == "başarılı", w
        assert any("çelişki kuralı" in n and neg in n for n in r["notes"]), r["notes"]
    # EKSİK ölçüm: tek baselinelik eşleme satırı vermez VE eksikliği yazar
    for w, eksik in (({"M": 1.0}, "D"), ({"D": 1.0}, "M")):
        r = decide(xs, quality_per_sec=w)
        assert r["row"] == "başarılı", f"{w} 'güçlü başarı' vermemeli"
        assert any("ERİŞİLEMEZ" in n and eksik in n for n in r["notes"]), r["notes"]
    # Tek skaler: satır erişilemez ve hangi baseline olduğunun bilinmediği yazılır
    r2 = decide(xs, quality_per_sec=1.0)
    assert r2["row"] == "başarılı"
    assert any("BİLİNMİYOR" in n for n in r2["notes"])
    # Alakasız anahtar `min()` üzerinden meşru bir sonucu düşürmesin: reddedilir
    with pytest.raises(ValueError, match="anahtarları"):
        decide(xs, quality_per_sec={"M": 1.0, "D": 1.0, "Z": -1.0})
    # İki not BAĞIMSIZ olmalı: eksik ölçüm + negatif w̄ aynı anda yazılabilmeli.
    # İlk hâl bunları `elif` ile zincirlemişti ve skaler −3.0'da B16'nın etiketi atlanıyordu.
    r3 = decide(xs, quality_per_sec=-3.0)
    assert any("BİLİNMİYOR" in n for n in r3["notes"])
    assert any("mimari başarı / sistem başarısızlığı" in n for n in r3["notes"])


def test_precision_criterion_has_a_terminal_row_at_max_n():
    """n = 10'da kesinlik yine çözemiyorsa satır "yetersiz kesinlik" — "seed 10'a çıkar" DEĞİL.

    Eski mutlak sd tripwire'ının çıkmazı "n=10'da ateşlendi" satırının olmamasıydı. Kesinlik
    ölçütü o çıkmazı tekrar etmemeli: B3 ikinci bir artırma hakkı vermiyor ve ölçek yükseltmesi
    kesinlik sorununun çaresi değil, dolayısıyla o noktada raporu olmayan bir çareye yönlendirmek
    hatanın aynısı olurdu. "nötr"den ayrımı da kritik: nötr ölçülmüş bir sıfırdır, bu ölçülememiş
    bir eşiktir.
    """
    n5 = _five(4.0, sd=1.5)
    n10 = _five(4.0, sd=2.1, n=10)
    r5, r10 = decide(n5, quality_per_sec={"M": 1.0, "D": 1.0}), decide(n10, quality_per_sec={"M": 1.0, "D": 1.0})
    assert r5["row"] == "yüksek varyans / belirsiz" and "seed 10'a çıkar" in r5["notes"][0]
    assert r10["row"] == "yetersiz kesinlik"
    assert "azami seed" in r10["notes"][0] and "'nötr' DEĞİLDİR" in r10["notes"][0]
    from src.stats import ROWS
    assert "yetersiz kesinlik" in ROWS


def test_abandon_is_reachable_when_a_seed_is_censored():
    """Eşdeğerlik testi SANSÜRSÜZ alt örneğin GA'sına bakar; yoksa kural sansürde ulaşılamaz olur.

    Ölçülen senaryo: `s_i = [0.2, −0.1, 0.3, 0.0, −149.82]` (dördü başabaş, biri 2x bütçe
    ikamesiyle sansürlü). Tam örnek GA'sı **[−113.13, +53.37]** — `hi < 5` ASLA sağlanamaz, yani
    vazgeçme kuralı sansürlü her senaryoda ölürdü. Bu, `decide()`'ın kesinlik ölçütünde zaten
    kapatılmış hatanın vazgeçme kuralındaki kopyasıydı; üstelik V12'nin sayısal sınavı onu
    göremiyordu çünkü sınav yalnızca Gauss örnekleri çiziyordu (artık sansür de tarıyor).
    """
    s_i = [0.2, -0.1, 0.3, 0.0, -149.82]
    r = decide(s_i, quality_per_sec={"M": -1.0, "D": -1.0}, censored_mask=[False] * 4 + [True])
    assert r["ci95"][1] > SUCCESS, "tam örnek GA'sı gerçekten patlamalı (testin varlık sebebi)"
    assert r["ci95_uncensored"][1] < SUCCESS
    assert abandon([r, r], oracle_gain_over_null=0.0)["abandon"] is True


def test_abandon_from_basarisiz_needs_only_one_scale():
    """v01 EK1-B10: "başarısız" satırı TEK ölçekle terk etmeye yeter; oracle şartı yine uygulanır."""
    bad = decide(_five(-8.0), quality_per_sec={"M": -1.0, "D": -1.0})
    assert bad["row"] == "başarısız"
    ok = abandon([bad], oracle_gain_over_null=0.5)
    assert ok["abandon"] is True and ok["single_scale"] is True
    blocked = abandon([bad], oracle_gain_over_null=5.0)
    assert blocked["abandon"] is False and "oracle" in blocked["reason"]


def test_abandon_two_censored_seeds_is_a_single_scale_abandonment():
    """>= 2 sansür -> "başarısız" -> tek ölçekle terk. Eşdeğerlik testine sokmak anlamsız olurdu:
    GA ikame yüzünden patlar ve mümkün olan EN GÜÇLÜ olumsuz sonuç "terk edilemez" çıkardı."""
    two = decide([0.2, -0.1, 0.3, -149.8, -149.9], quality_per_sec={"M": -1.0, "D": -1.0},
                 censored_mask=[False] * 3 + [True] * 2)
    assert two["row"] == "başarısız"
    assert abandon([two], oracle_gain_over_null=0.5)["abandon"] is True


def test_abandon_blocked_by_indeterminate_terminal_row():
    """"yetersiz kesinlik" de vazgeçmeyi ENGELLER: ölçemediğimiz durumda hipotez terk edilmez."""
    r = decide(_five(4.0, sd=2.1, n=10), quality_per_sec={"M": -1.0, "D": -1.0})
    assert r["row"] == "yetersiz kesinlik"
    res = abandon([r, r], oracle_gain_over_null=0.0)
    assert res["abandon"] is False and "yetersiz kesinlik" in res["reason"]
