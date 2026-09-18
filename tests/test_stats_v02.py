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
    Obs,
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

SEEDS10 = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]        # ön-kayıtlı karar seed'leri (EK1-B3)


def T(values, cens=(), slow=()) -> dict:
    """Değer listesini seed KİMLİĞİYLE anahtarlı tabloya çevirir; `cens`/`slow` = o KONUMDAKİ seed'in durumu.

    `src.stats` liste kabul etmez (konuma göre eşleme kapalı); testler senaryoyu liste olarak
    yazıp buradan geçirir. Kimlikler ön-kayıtlı seed'lerdir.
    """
    assert len(values) <= len(SEEDS10)
    return {s: Obs(v, "sansürlü" if i in cens else "yavaş" if i in slow else "hızlı")
            for i, (s, v) in enumerate(zip(SEEDS10, values))}


def C(diffs, **kw):
    """Birincil-2'yi verilen FARKLARLA puanlar: X = farklar, H = 0 (aynı seed'ler)."""
    from src.stats import contribution
    return contribution(T(diffs), T([0.0] * len(diffs)), **kw)

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
    f_foton = {11: 90.0, 22: 80.0, 33: 70.0}
    f_base = {11: 100.0, 33: 100.0, 22: 200.0}               # s22 sonda! baseline'ı 200
    r = savings(f_foton, f_base)
    # s11: 1-90/100 = 10 | s22: 1-80/200 = 60 | s33: 1-70/100 = 30
    assert {k: o.s for k, o in r["obs"].items()} == pytest.approx({11: 10.0, 22: 60.0, 33: 30.0})
    assert list(r["obs"]) == [11, 22, 33]                    # kanonik sıra: ekleme sırasından bağımsız
    # konuma göre eşleme bu girdide s22 -> 1-80/100 = 20 verirdi; imza onu İFADE EDEMEZ
    with pytest.raises(TypeError, match="anahtarlanmış eşleme"):
        savings([90.0, 80.0, 70.0], [100.0, 100.0, 200.0])
    with pytest.raises(TypeError, match="anahtarlanmış eşleme"):
        savings(f_foton, [100.0, 100.0, 200.0])


def test_savings_rejects_mismatched_seed_sets():
    """Eksik/fazla seed SESSİZCE atlanmaz. `paired()` atlıyor; birincil istatistikte atlamak yasak."""
    with pytest.raises(ValueError, match="seed kümeleri"):
        savings({11: 1.0, 22: 2.0}, {11: 1.0, 33: 2.0})
    with pytest.raises(ValueError, match="seed kümeleri"):
        savings({11: 1.0, 22: 2.0}, {11: 1.0, 22: 2.0, 33: 3.0})       # fazla seed de sessiz değil
    with pytest.raises(ValueError, match="seed kümeleri"):
        savings({11: 1.0, 22: 2.0}, {11: 1.0, 22: 2.0}, status={11: "hızlı"})


def test_keyed_rejects_duplicate_seeds():
    """`dict(pairs)` tekrar eden seed'de öncekini SESSİZCE ezer; yükleyicinin kapısı `keyed()` hata verir."""
    from src.stats import keyed
    assert keyed([(11, 1.0), (22, 2.0)]) == {11: 1.0, 22: 2.0}
    with pytest.raises(ValueError, match="iki kez"):
        keyed([(11, 1.0), (22, 2.0), (11, 3.0)])                   # yeniden koşulan s11'in iki kaydı


def test_savings_rejects_nonpositive_denominator():
    with pytest.raises(ValueError, match="pozitif"):
        savings({11: 1.0, 22: 2.0}, {11: 0.0, 22: 2.0})


def test_savings_is_a_ratio_not_a_difference():
    """`paired()` FARK hesaplar, `s_i` ORAN. Karıştırılmasın: aynı girdide farklı sayılar."""
    r = savings({11: 50.0}, {11: 100.0})
    assert r["obs"][11].s == pytest.approx(50.0)             # %50 tasarruf, 50 puan "fark" değil
    assert r["obs"][11].status == "hızlı"
    slow = savings({11: 250.0}, {11: 100.0}, status={11: "yavaş"})
    assert slow["obs"][11] == Obs(-150.0, "yavaş")           # durum değerle AYNI kayıtta taşınır
    # durum da KİMLİKLE eşlenir: status eşlemesinin ekleme sırası başka olsa da s22 yavaş kalır
    two = savings({11: 90.0, 22: 150.0}, {11: 100.0, 22: 100.0}, status={22: "yavaş", 11: "hızlı"})
    assert (two["obs"][11].status, two["obs"][22].status) == ("hızlı", "yavaş")


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
        return lambda s: decide(T(s), quality_per_sec=W_BOTH)["row"] == "yüksek varyans / belirsiz"
    p5 = reachability(fires(5), mu=0.0, sigma=5.0, n=5, sims=3000, seed=1)
    p10 = reachability(fires(10), mu=0.0, sigma=5.0, n=10, sims=3000, seed=1)
    assert p5 > p10, f"çare kuralı sıkıştırıyor: n=5 %{100*p5:.1f} -> n=10 %{100*p10:.1f}"


def test_precision_tripwire_silent_at_measured_noise():
    """Ölçülen sd(s_i) = 0.75 puanda tripwire tetiklenmemeli; yoksa her koşu 10 seed'e giderdi."""
    p = reachability(lambda s: decide(T(s), quality_per_sec=W_BOTH)["row"] == "yüksek varyans / belirsiz",
                     mu=6.0, sigma=0.75, n=5, sims=3000, seed=2)
    assert p < 0.02, f"ölçülen gürültüde tripwire %{100*p:.1f} tetikleniyor"


# --------------------------------------------------------------------------- decide() sertleştirmeleri

def test_decide_requires_preregistered_seed_count():
    """Kilit cümle BEŞ seed istiyor; `decide()` iki seed'le "başarılı" damgası basıyordu."""
    with pytest.raises(ValueError, match="bağlayıcı karar"):
        decide(T([6.0, 6.1]), quality_per_sec=W_BOTH)
    with pytest.raises(ValueError, match="bağlayıcı karar"):
        decide(T([6.0, 6.1, 6.05]), quality_per_sec=W_BOTH)
    for n in (5, 10):
        xs = [6.0 + 0.01 * i for i in range(n)]
        assert decide(T(xs), quality_per_sec=W_BOTH)["row"] == "başarılı"


def test_decide_strict_false_blocks_success_rows():
    """Keşifsel kullanım mümkün ama başarı İLAN EDEMEZ."""
    r = decide(T([6.0, 6.1, 6.05]), quality_per_sec=W_BOTH, strict=False)
    assert r["row"] == "belirsiz"
    assert any("strict=False" in n for n in r["notes"])


def test_decide_does_not_invent_an_uncensored_sd_from_one_seed():
    """Tek sansürsüz seed varken sd YOKTUR; eski kod ham sd'yi "sansürsüz sd" diye raporluyordu.

    O rapor, -150'lik sansür ikamesinden gelen 110.31'i "sansürsüz seed'lerin sd'si" olarak
    yazıyordu: var olmayan bir nicelik, ve tam olarak B8'in engellemek için yazıldığı yanlış okuma
    ("hedefe ulaşamıyor" yerine "varyans yüksek, daha çok seed lazım").
    """
    xs = [6.0, 6.1, 6.05, -150.0, -150.0]
    r = decide(T(xs, cens=[3, 4]), quality_per_sec=W_BOTH)
    assert r["row"] == "başarısız"                      # >= 2 sansür: B8 doğrudan başarısız
    # Tek sansür + tek sansürsüz: kesinlik DEĞERLENDİRİLEMEZ, uydurulmaz
    r2 = decide(T([6.0, -150.0, -150.0, -150.0, -150.0], cens=[1, 2, 3, 4]), quality_per_sec=W_BOTH)
    assert math.isnan(r2["sd_fast"]) and "ci95_fast" not in r2
    assert r2["row"] == "başarısız"
    assert not any("110" in n for n in r2["notes"])


def test_decide_one_uncensored_seed_cannot_claim_precision():
    """n=5, 4 sansürlü değil ama sansürsüz sayısı 1 olan kurgu: kesinlik değerlendirilemez."""
    # 1 sansür (B8 sınırı içinde) ama diğer 4'ü de... hayır: sansürsüz >= 2 olması için
    # en fazla 3 sansür gerekir ve B8 zaten >=2'de başarısız veriyor. Bu yüzden tek sansürsüz
    # durumu ancak >=2 sansürle oluşur ve B8 önce devreye girer — kilitlenen şey o sıralama.
    r = decide(T([7.0, -150.0, -150.0, -150.0, -150.0], cens=[1, 2, 3, 4]), quality_per_sec=W_BOTH)
    assert r["row"] == "başarısız" and r["n_censored"] == 4
    assert any("B8" in n for n in r["notes"])


def test_decide_conflict_label_requires_statistical_flop_gain():
    """B16 çelişki etiketi GÜRÜLTÜYE vurulmamalı: FLOP kazancının GA'sı sıfırın üstünde olmalı.

    Eski kod `mean > 0` ile yetiniyordu, yani s̄ = +0.01 puanlık bir farka "mimari başarı /
    sistem başarısızlığı" damgası basılabiliyordu — ve ön-kayıt bu ifadeyi bir SATIR gibi
    anıyordu, dolayısıyla null bir sonuç mimari başarı iddiasına dönüşebiliyordu.
    """
    noisy = [0.4, 1.2, -0.3, 0.9, 0.1]                  # ortalama 0.46, GA sıfırı kesiyor
    r = decide(T(noisy), quality_per_sec=-30.0)
    assert r["row"] == "nötr" and r["ci95"][0] < 0
    assert not any("mimari başarı" in n for n in r["notes"])
    assert any("UYGULANMAZ" in n for n in r["notes"])
    # Gerçek ve istatistiksel kazançta etiket VERİLİR
    r2 = decide(T([6.0, 6.2, 5.9, 6.1, 6.0]), quality_per_sec=-30.0)
    assert any("mimari başarı" in n for n in r2["notes"])


def test_decide_reports_missing_wall_clock_metric():
    """`quality_per_sec=None` SESSİZ olmamalı: ikincil ölçüt değerlendirilmedi, kayda geçsin.

    V11a duvar saatini eksik bir seed bloğunda geçersiz saydığı için bu ulaşılabilir bir durum:
    tek bir çöken koşu w̄'yi tanımsız bırakır, ve o hâlde "başarılı" satırının yanında ikincil
    ölçütün hiç bakılmadığının yazılı olması gerekir.
    """
    r = decide(T(_five(12.0)), quality_per_sec=None)
    assert r["row"] == "başarılı"                        # 'güçlü başarı' erişilemez
    assert any("VERİLMEDİ" in n for n in r["notes"])
    assert decide(T(_five(12.0)), quality_per_sec=W_BOTH)["row"] == "güçlü başarı"


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
    assert decide(T(xs), quality_per_sec=W_BOTH)["row"] != "başarılı"


def test_decide_status_is_foton_only_by_contract():
    """Sözleşme: durum YALNIZCA FOTON kolunun durumu. Baseline'ın sansürü girerse felaket.

    v02'de dense kolu `L*`'e ulaşmıyor (5 seed'in 5'i). Maskeye onun sansürü konursa %7 tasarruf
    eden bir FOTON, BASELINE'ın yetersizliği yüzünden "başarısız" ilan edilir. Bu test o sonucu
    gösterir ki sözleşme kodda değil belgede korunduğu unutulmasın.
    """
    xs = _five(7.0)
    assert decide(T(xs), quality_per_sec=W_BOTH)["row"] == "başarılı"
    wrong = decide(T(xs, cens=range(5)), quality_per_sec=W_BOTH)
    assert wrong["row"] == "başarısız"                   # durum yanlış doldurulursa olan bu


# --------------------------------------------------------------------------- vazgeçme koşulu

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
    assert decide(T(xs), quality_per_sec={"M": 1.0, "D": 1.0})["row"] == "güçlü başarı"
    # Bir baseline'a karşı kaybediyorsa satır DÜŞER; şart simetrik
    for w, neg in (({"M": 1.0, "D": -5.0}, "D"), ({"M": -5.0, "D": 1.0}, "M")):
        r = decide(T(xs), quality_per_sec=w)
        assert r["row"] == "başarılı", w
        assert any("çelişki kuralı" in n and neg in n for n in r["notes"]), r["notes"]
    # EKSİK ölçüm: tek baselinelik eşleme satırı vermez VE eksikliği yazar
    for w, eksik in (({"M": 1.0}, "D"), ({"D": 1.0}, "M")):
        r = decide(T(xs), quality_per_sec=w)
        assert r["row"] == "başarılı", f"{w} 'güçlü başarı' vermemeli"
        assert any("ERİŞİLEMEZ" in n and eksik in n for n in r["notes"]), r["notes"]
    # Tek skaler: satır erişilemez ve hangi baseline olduğunun bilinmediği yazılır
    r2 = decide(T(xs), quality_per_sec=1.0)
    assert r2["row"] == "başarılı"
    assert any("BİLİNMİYOR" in n for n in r2["notes"])
    # Alakasız anahtar `min()` üzerinden meşru bir sonucu düşürmesin: reddedilir
    with pytest.raises(ValueError, match="anahtarları"):
        decide(T(xs), quality_per_sec={"M": 1.0, "D": 1.0, "Z": -1.0})
    # İki not BAĞIMSIZ olmalı: eksik ölçüm + negatif w̄ aynı anda yazılabilmeli.
    # İlk hâl bunları `elif` ile zincirlemişti ve skaler −3.0'da B16'nın etiketi atlanıyordu.
    r3 = decide(T(xs), quality_per_sec=-3.0)
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
    r5, r10 = decide(T(n5), quality_per_sec={"M": 1.0, "D": 1.0}), decide(T(n10), quality_per_sec={"M": 1.0, "D": 1.0})
    assert r5["row"] == "yüksek varyans / belirsiz" and "seed 10'a çıkar" in r5["notes"][0]
    assert r10["row"] == "yetersiz kesinlik"
    assert "azami seed" in r10["notes"][0] and "'nötr' DEĞİLDİR" in r10["notes"][0]
    from src.stats import ROWS
    assert "yetersiz kesinlik" in ROWS


# ------------------------------------------------- birincil-2 ve Holm'lu bileşik karar

def test_contribution_four_states_partition_the_plane():
    """Dört durum BİRBİRİNİ DIŞLAR ve TÜKETİR — sınırlar dahil.

    İlk hâl üç durumluydu ve "katkı var" (lo > 0) ile "katkı yok" (hi < MDE) birlikte doğru
    olabiliyordu; o hâl ("gerçek ama küçük") tabloda karşılıksızdı ve V12'nin taraması bunu
    olasılıkların NEGATİFE dönmesiyle yakalamıştı.
    """
    from src.stats import MDE_PRIMARY2, contribution
    cases = {
        "katkı var, büyük": _five(3.0, sd=0.5),          # lo>0, hi>=MDE
        "katkı var, küçük": _five(1.0, sd=0.2),          # lo>0, hi<MDE
        "katkı MDE kadar değil": _five(0.0, sd=0.5),     # lo<=0, hi<MDE
        "ölçemedik": _five(0.5, sd=2.0),                 # lo<=0, hi>=MDE
    }
    for want, xs in cases.items():
        r = C(xs)
        lo, hi = r["ci"]
        assert r["state"] == want, f"{want}: GA [{lo:.2f}, {hi:.2f}] -> {r['state']}"
        assert (lo > 0) == want.startswith("katkı var")
        assert (hi < MDE_PRIMARY2) == (want in ("katkı var, küçük", "katkı MDE kadar değil"))
    assert "çürütüldü" in C(cases["ölçemedik"])["note"]
    assert C(cases["katkı var, büyük"])["note"] == ""


def test_contribution_uses_holm_level_not_five_percent():
    """GA, Holm'un İLK adımındaki α = .025 düzeyinde — %95 değil %97.5 aralığı.

    Taramanın ilk hâli α = .05 aralıkları kullanıyordu ve "ölçemedik" payını gerçek katkı sıfırken
    %24.6 gösteriyordu; kuralın kendi düzeyinde %42.8. Kuralı başka bir düzeyde taramak, başka bir
    kuralı sınamaktır.
    """
    from src.stats import contribution, t_crit_alpha
    xs = _five(1.2, sd=1.0)
    r = C(xs)
    assert r["alpha"] == pytest.approx(0.025)
    half = (r["ci"][1] - r["ci"][0]) / 2
    assert half == pytest.approx(t_crit_alpha(4, 0.025) * 1.0 / math.sqrt(5), rel=1e-6)
    assert t_crit_alpha(4, 0.025) == pytest.approx(3.4954, abs=2e-4)
    assert t_crit_alpha(4, 0.05) == pytest.approx(t_crit(4), abs=1e-5)       # tabloyla tutarlı
    # %95'te "katkı var" olacak bir örnek, Holm düzeyinde "ölçemedik"e düşmeli:
    ys = _five(1.3, sd=1.0)                          # t = 2.91: .025 < p < .05
    m, lo95, _ = ci95(ys)
    assert lo95 > 0 and not C(ys)["state"].startswith("katkı var")


def test_contribution_is_not_scored_with_the_decision_table():
    """MDE kadar gerçek bir heterojenlik katkısı `decide()`'da "nötr"e düşerdi — o yüzden ayrı kural."""
    from src.stats import MDE_PRIMARY2, contribution
    xs = _five(MDE_PRIMARY2, sd=0.3)
    assert decide(T(xs), quality_per_sec=W_BOTH)["row"] == "nötr"       # tasarruf eşikleriyle yanlış etiket
    assert C(xs)["state"].startswith("katkı var")                     # doğru kuralla doğru etiket


# ------------------------------------------------- mutasyon sınavının bulduğu guard boşlukları

def test_decision_thresholds_are_locked_literally_and_behaviourally():
    """Karar tablosunun eşikleri ön-kayıtta donmuştur; burada HEM sabit HEM davranış kilitli.

    Mutasyon sınavı buldu: `SUCCESS = 5.0 -> 4.0` mutantı HAYATTA KALIYORDU, çünkü hiçbir test
    [4, 5) aralığında bir ortalamayı sınamıyordu (3.0 -> "belirsiz", 5.0 -> "başarılı" ikisi de
    eşik 4'te geçer). Yani ön-kaydın en önemli sayısı olan +%5 eşiği, hiçbir guard ateşlenmeden
    değiştirilebiliyordu.

    Aralık içi değerler dar GA ile seçilir ki kesinlik ölçütü araya girmesin (s̄ = 4.99 gibi
    eşiğe yapışık bir değer GA eşiği kestiği için "yüksek varyans / belirsiz" verir — o ayrı test).
    """
    from src import stats
    assert (stats.SUCCESS, stats.STRONG, stats.NEUTRAL) == (5.0, 10.0, 2.0)
    assert stats.REQUIRED_N == (5, 10) and stats.WBAR_KEYS == ("M", "D")
    assert stats.PRIMARY_FAMILY == 2 and stats.MDE_PRIMARY2 == 1.88
    for mean, row in ((4.5, "belirsiz"), (5.5, "başarılı"), (9.5, "başarılı"), (10.5, "güçlü başarı"),
                      (1.5, "nötr"), (2.5, "belirsiz"), (-1.5, "nötr"), (-2.5, "başarısız")):
        assert decide(T(_five(mean)), quality_per_sec=W_BOTH)["row"] == row, (mean, row)


def test_two_censored_seeds_give_basarisiz_with_the_b8_note_even_when_the_mean_would_not():
    """EK1-B8(iii): >= 2 sansür -> doğrudan "başarısız". Kuralın KENDİSİ kilitleniyor.

    Mutasyon sınavı gösterdi ki bu kural gerçekçi girdilerde fiilen yedekli: −150 ikamesi tam
    örnek ortalamasını zaten negatife çekiyor ve satır `s̄ < −2` dalından yine "başarısız"
    çıkıyor. Kuralın gerçekten bağladığı bölge, sansürsüz seed'lerin ÇOK büyük tasarruf ettiği
    uç durumdur: [100, 100, 100, −150, −150] -> ortalama 0 -> kural OLMADAN satır "nötr" olurdu
    (hedefe ulaşamayan iki seed'e rağmen). Test o bölgeyi ve notun varlığını kilitler.
    """
    r = decide(T([100.0, 100.0, 100.0, -150.0, -150.0], cens=[3, 4]), quality_per_sec=W_BOTH)
    assert r["mean"] == pytest.approx(0.0)
    assert r["row"] == "başarısız"
    assert any("B8" in n and "doğrudan 'başarısız'" in n for n in r["notes"])


# =====================================================================================
# ÜÇÜNCÜ DENETİM TURU: n_star (B14), sansüre duyarlı üst sınır, vazgeçmenin iki yolu,
# Holm'lu bileşik karar. Bu blok, o turun 8 HIGH bulgusunun her birini kilitler.
# =====================================================================================
NEG = {"M": -1.0, "D": -1.0}
CENS = -149.82                                   # 2x bütçe ikamesinin puan karşılığı
OK1 = {"S0": 0.5}                                # oracle kazancı, ölçek KİMLİĞİYLE
OK2 = {"S0": 0.5, "S1": 0.5}


def _h(s_xm: list[float], diffs: list[float]) -> dict:
    """X − H = `diffs` olacak H tablosu (X ile aynı seed'ler)."""
    return T([a - d for a, d in zip(s_xm, diffs)])


def test_n_star_takes_the_next_sustained_crossing_b14():
    """EK1-B14: "kesişimden sonraki eval noktası da ≤ L* olmalı; değilse SONRAKİ kesişime bakılır."

    Eğri 2750'de hedefin altına sarkıp 3000'de geri çıkıyor, 3250–3500 arasında KALICI kesiyor.
    Doğru `N*` o kalıcı kesişimdir. İki yanlış okuma da kilitli:
      * erken sarkmayı `N*` saymak (`s_i`'yi ~21 puan şişirir, "nötr"ü "güçlü başarı" yapar);
      * seed'i SANSÜRLEMEK (bir önceki uygulama buydu — B14'ün tersi; tek bir 0.0005 nat'lık
        sıçrama, hedefin altında biten bir koşuyu −150 puana çeviriyordu).
    """
    #        2500  2750  3000  3250  3500  3750
    vals = [2.95, 2.89, 2.93, 2.92, 2.88, 2.86]          # hedef 2.90
    got = n_star(_curve(vals, first=2500), 2.90)
    assert got is not None, "sürdürülen kesişim VAR; seed sansürlenmemeli"
    assert 3250 * STEP < got < 3500 * STEP, "erken sarkma N* sayıldı"
    assert got == pytest.approx((3250 + 250 * (2.92 - 2.90) / (2.92 - 2.88)) * STEP)
    # son nokta hedefin ÜSTÜNDEYSE (ulaşıp kaybetti) sansür
    assert n_star(_curve([2.95, 2.89, 2.88, 2.93], first=2500), 2.90) is None


def test_ub95_equals_plain_ci_without_censoring_and_is_monotone_in_censoring():
    """Sansür yokken düz %95 GA üst sınırı; her sansürlü seed onu AŞAĞI çeker (monoton).

    Üçüncü denetimin bulduğu delik: sansürsüz ALT örneğin aralığına bakan kural k=3'te (2 seed,
    t(1) = 12.7) patlıyor ve en net başarısızlık terk edilemiyordu; kural k'da monoton değildi.
    """
    from src.stats import ub95
    xs = _five(0.0, sd=0.88)
    assert ub95(T(xs)) == pytest.approx(ci95(xs)[2])
    prev = ub95(T(xs))
    for k in range(1, 5):
        cur = ub95(T(xs[:5 - k] + [CENS] * k, cens=range(5 - k, 5)))
        assert cur < prev, f"k={k}: üst sınır monoton değil ({cur:.2f} >= {prev:.2f})"
        prev = cur
    # 3/5 sansür + geniş iki seed: eski kural +6.4 görüp terk etmeyi engelliyordu
    assert ub95(T([0.5, -0.5, CENS, CENS, CENS], cens=[2, 3, 4])) < -80
    with pytest.raises(TypeError, match="anahtarlanmış eşleme"):
        ub95(xs)


def test_abandon_two_scale_rule_uses_the_censor_aware_bound():
    """İki ölçek yolu: her iki ölçekte `ub95 < +5` ve oracle (ölçek BAŞINA) < 2."""
    be = decide(T(_five(0.0, sd=0.88)), quality_per_sec=NEG)
    assert abandon({"S0": be, "S1": be}, OK2, escalation_used=True)["abandon"] is True
    # bir ölçekte tek seed sansürlü: düz aralık [−113, +53] olurdu; sansüre duyarlı sınır −29
    s1 = decide(T([0.2, -0.1, 0.3, 0.0, CENS], cens=[4]), quality_per_sec=NEG)
    assert s1["ci95"][1] > SUCCESS and s1["ub95"] < SUCCESS
    assert abandon({"S0": be, "S1": s1}, OK2, escalation_used=True)["abandon"] is True
    # bir ölçekte etki eşiğin ÜSTÜNDE kurulmuş (GA [6.0, 8.0]) -> terk yok
    near = decide(T(_five(7.0, sd=0.8)), quality_per_sec=NEG)
    assert near["determinate"] and near["ub95"] >= SUCCESS
    r = abandon({"S0": be, "S1": near}, OK2, escalation_used=True)
    assert r["abandon"] is False and "GA üst sınırı" in r["reason"]


def test_abandon_needs_both_scales_not_just_the_last_one():
    """`[S0 "başarılı", S1 "başarısız"]` TERK EDİLMEZ — üçüncü denetimin bulduğu regresyon.

    Önceki hâl tek ölçek çıkışını elde iki ölçek varken de deniyordu ve yalnızca SON ölçeğe
    bakıp hipotezi terk ettiriyordu; oysa S0'ın GA'sı [6.5, 7.5], tamamen +5'in ÜSTÜNDE.
    Varsayılan `escalation_used=False` ile de aynı olmalı (v01 B10: "başarılı"da S1 isteğe bağlı,
    hak harcanmaz — yani bu çağrı meşru).
    """
    s0 = decide(T([6.5, 7.0, 7.5, 6.8, 7.2]), quality_per_sec={"M": 1.0, "D": -1.0})
    s1 = decide(T([-4.0, -3.5, -4.5, -3.8, -4.2]), quality_per_sec=NEG)
    assert (s0["row"], s1["row"]) == ("başarılı", "başarısız")
    for flag in (False, True):
        r = abandon({"S0": s0, "S1": s1}, OK2, escalation_used=flag)
        assert r["abandon"] is False and r["path"] == "iki ölçek", (flag, r)


def test_abandon_single_scale_path_follows_b10():
    """Tek ölçek yolu (v01 EK1-B10): satır "başarısız" + hak KULLANILMAMIŞ + ub95 < 5 + oracle."""
    bad = decide(T(_five(-8.0)), quality_per_sec=NEG)
    ok = abandon({"S0": bad}, OK1, escalation_used=False)
    assert ok["abandon"] is True and ok["single_scale"] is True and "B10" in ok["reason"]
    assert ok["escalation_used"] is False and ok["scales"] == ["S0"]
    assert abandon({"S0": bad}, OK1, escalation_used=True)["abandon"] is False       # hak kullanıldı
    assert abandon({"S0": bad}, {"S0": 5.0}, escalation_used=False)["abandon"] is False   # oracle headroom'u
    neutral = decide(T(_five(0.0, sd=0.88)), quality_per_sec=NEG)
    r = abandon({"S0": neutral}, OK1, escalation_used=False)
    assert r["abandon"] is False and "'başarısız' satırından" in r["reason"]   # "nötr"den açılmaz


def test_abandon_with_a_censored_seed_follows_the_preregistered_statistic():
    """ANLAM KARARI, açıkça kilitli: `4 × +%30 + 1 sansür`, hak kullanılmazsa TERK EDİLİR.

    Birincil istatistik `s̄`, hedefe ulaşamayan seed'i de içerir (EK1-B8 ikamesi): burada
    s̄ ≈ −5.9, satır "başarısız". v01 B10 harfiyen: "başarısız"ta yükseltme hakkı kullanılabilir;
    kullanılmazsa hipotez terk edilir. Yani sahibin önünde iki yol var — S1'e çıkmak ya da terk —
    üçüncü bir "başarısız ama ne terk ne devam" hâli YOKTUR (o hâl bir park yörüngesiydi: bir
    önceki uygulama sansürsüz GA'ya bakıp bu sonucu ne terk ediyor ne ilerletiyordu, üstelik
    bu şart hiçbir bağlayıcı metinde yazmıyordu).
    """
    pos = decide(T([30.0, 30.1, 29.9, 30.05, CENS], cens=[4]), quality_per_sec=NEG)
    assert pos["row"] == "başarısız" and pos["mean"] == pytest.approx(-5.954, abs=1e-3)
    assert pos["ub95"] == pytest.approx((4 * ci95([30.0, 30.1, 29.9, 30.05])[2] + CENS) / 5)
    assert pos["ub95"] < SUCCESS
    assert abandon({"S0": pos}, OK1, escalation_used=False)["abandon"] is True
    assert abandon({"S0": pos}, OK1, escalation_used=True)["abandon"] is False         # hak kullanıldı: S1 beklenir
    assert abandon({"S0": pos}, {"S0": 35.0}, escalation_used=False)["abandon"] is False   # oracle korur


def test_abandon_is_reachable_with_three_of_five_seeds_censored():
    """5 seed'in 3'ü hedefe ulaşamadı: mümkün olan EN NET başarısızlık, her iki yolda terk edilebilir."""
    r = decide(T([0.5, -0.5, CENS, CENS, CENS], cens=[2, 3, 4]), quality_per_sec=NEG)
    assert r["row"] == "başarısız"
    assert abandon({"S0": r}, OK1, escalation_used=False)["abandon"] is True
    assert abandon({"S0": r, "S1": r}, OK2, escalation_used=True)["abandon"] is True


def test_abandon_blocked_by_indeterminate_rows_and_says_so():
    """"yüksek varyans / belirsiz" ve "yetersiz kesinlik" vazgeçmeyi ENGELLER: ölçemedik ≠ hipotez ayakta."""
    wide = decide(T([d for d in (-9, -4.5, 0, 4.5, 9)]), quality_per_sec=NEG)
    term = decide(T(_five(4.0, sd=2.1, n=10)), quality_per_sec=NEG)
    assert (wide["row"], term["row"]) == ("yüksek varyans / belirsiz", "yetersiz kesinlik")
    for r in (wide, term):
        res = abandon({"S0": r, "S1": r}, {"S0": 0.0, "S1": 0.0}, escalation_used=True)
        assert res["abandon"] is False and "yetersiz kesinlik" in res["reason"] and "ölçemedik" in res["reason"]


def test_abandon_oracle_is_per_scale_and_must_be_finite():
    """İki ölçekte tek sayı KABUL EDİLMEZ: S0 = 1.5 / S1 = 2.5 iken sonuç verilen sayıya bağlı olurdu."""
    be = decide(T(_five(0.0, sd=0.88)), quality_per_sec=NEG)
    two = {"S0": be, "S1": be}
    with pytest.raises(TypeError, match="anahtarlanmış eşleme"):
        abandon(two, 0.5, escalation_used=True)                      # tek sayı
    with pytest.raises(TypeError, match="anahtarlanmış eşleme"):
        abandon(two, [0.5, 0.5], escalation_used=True)               # paralel dizi: ölçekle KONUMLA eşlenirdi
    with pytest.raises(TypeError, match="anahtarlanmış eşleme"):
        abandon([be, be], {"S0": 0.5, "S1": 0.5}, escalation_used=True)
    with pytest.raises(ValueError, match="ölçek BAŞINA"):
        abandon(two, {"S0": 0.5}, escalation_used=True)              # S1'in oracle'ı yok
    assert abandon(two, {"S0": 1.5, "S1": 2.5}, escalation_used=True)["abandon"] is False   # S1'de headroom var
    assert abandon(two, {"S0": 1.5, "S1": 1.9}, escalation_used=True)["abandon"] is True
    with pytest.raises(ValueError, match="sonlu"):
        abandon(two, {"S0": 0.5, "S1": float("nan")}, escalation_used=True)
    # ölçek anahtarları pinli ve BAŞTAN ardışık: S0'sız S1 ya da tanımsız bir ölçek kabul edilmez
    for bad in ({"S1": be}, {"S0": be, "S2": be}):
        with pytest.raises(ValueError, match="BAŞTAN ardışık"):
            abandon(bad, dict.fromkeys(bad, 0.5))


def test_abandon_refuses_non_binding_or_hand_built_results():
    """Keşifsel (strict=False) bir sonuç ya da elle kurulmuş bir sözlük vazgeçme kararını SÜREMEZ."""
    expl = decide(T([-4.0, -3.5, -4.5]), quality_per_sec=NEG, strict=False)
    with pytest.raises(ValueError, match="bağlayıcı olmayan"):
        abandon({"S0": expl}, OK1)
    with pytest.raises(ValueError, match="decide"):
        abandon({"S0": {"row": "nötr", "ci95": [-1.0, 1.0], "n": 5}}, OK1)


def test_seed_right_spent_makes_the_precision_row_terminal_at_n5():
    """Tek seed artırma hakkı KAMPANYA başınadır. S0'da harcandıysa S1'de (n=5) kesinlik ölçütü
    tetiklenince satır doğrudan "yetersiz kesinlik"tir; "seed 10'a çıkar" olmayan bir çaredir."""
    xs = _five(4.0, sd=1.5)
    assert decide(T(xs), quality_per_sec=W_BOTH)["row"] == "yüksek varyans / belirsiz"
    r = decide(T(xs), quality_per_sec=W_BOTH, seed_right_available=False)
    assert r["row"] == "yetersiz kesinlik" and "zaten harcandı" in r["notes"][0]


def test_mandatory_notes_survive_every_early_return():
    """w̄ ve B8(i) notları sansür / kesinlik erken dönüşlerinde ATLANMAZ (rapor bunları zorunlu kılıyor)."""
    two = decide(T([0.2, -0.1, 0.3, CENS, CENS], cens=[3, 4]))
    assert any("VERİLMEDİ" in n for n in two["notes"]) and any("LEHİNE yanlıdır" in n for n in two["notes"])
    term = decide(T(_five(4.2, sd=1.5, n=10)), quality_per_sec=NEG)       # GA [3.13, 5.27]: alt > 0, eşiği kesiyor
    assert term["row"] == "yetersiz kesinlik" and term["ci95"][0] > 0
    assert any("mimari başarı / sistem başarısızlığı" in n for n in term["notes"])


def test_wbar_rejects_nan():
    """NaN w̄ `min()`'i sıra-bağımlı yapıyor ve ikinci sırada "güçlü başarı" kapısını GEÇİYORDU."""
    for w in ({"M": float("nan"), "D": 1.0}, {"D": 1.0, "M": float("nan")}, float("nan")):
        with pytest.raises(ValueError, match="sonlu"):
            decide(T(_five(11.0)), quality_per_sec=w)


def test_decide_primary_holm_precedes_the_row_name_for_both_success_rows():
    """Holm'dan GEÇMEYEN birincil-1 ne "başarılı" ne "güçlü başarı" OLABİLİR.

    s̄ gürültülü (p₁ ≈ .04), birincil-2 null. `decide()` tek başına başarı der; Holm'un ilk adımı
    .025 olduğu için bileşik karar satırı "belirsiz"e düşürür. "güçlü başarı" kolu ayrıca kilitli:
    mutasyon denetimi, yalnızca "başarılı"yı düşüren bir mutantın hayatta kaldığını buldu.
    """
    from src.stats import decide_primary
    for mean, sd, solo in ((6.0, 4.4, "başarılı"), (12.0, 8.8, "güçlü başarı")):
        s_xm = _five(mean, sd=sd)                        # t = 3.05, p ≈ .038
        assert decide(T(s_xm), quality_per_sec=W_BOTH)["row"] == solo
        out = decide_primary(T(s_xm), _h(s_xm, _five(0.0, sd=1.0)), quality_per_sec=W_BOTH)   # X−H null
        assert 0.025 < out["primary1"]["p"] < 0.05
        assert out["primary1"]["row"] == "belirsiz" and out["primary1"]["holm_passed"] is False
        assert any("Holm düzeltmesinden GEÇMEDİ" in n for n in out["primary1"]["notes"])
        assert out["holm"]["n_tests"] == 2


def test_decide_primary_holm_is_step_down_not_independent():
    """Holm ADIM ADIMdır: küçük p ilk adımı (.025) geçemezse büyük p HİÇ sınanmaz.

    p₂ ≈ .03 (küçük olan) .025'i geçemiyor; p₁ ≈ .04 tek başına .05'in altında ama sınanmamalı.
    Bağımsız eşiklerle (ya da `break` yerine `continue` ile) p₁ yanlışlıkla geçerdi.
    """
    from src.stats import decide_primary
    s_xm = _five(6.0, sd=4.4)                            # p₁ ≈ .038
    diffs = _five(1.5, sd=0.97)                          # p₂ ≈ .026..03, p₁'den KÜÇÜK
    out = decide_primary(T(s_xm), _h(s_xm, diffs), quality_per_sec=W_BOTH)
    p1, p2 = out["primary1"]["p"], out["primary2"]["p"]
    assert 0.025 < p2 < p1 < 0.05, (p1, p2)
    assert out["holm"]["holm"] == [] and out["primary1"]["row"] == "belirsiz"
    assert m_holm([("a", 0.04), ("b", 0.03)]) == []


def m_holm(tests):
    return multiplicity(tests, family_size=2)["holm"]


def test_decide_primary_scores_primary2_at_the_holm_level():
    """Birincil-2'nin aralığı α/2 = .025 düzeyindedir; `decide_primary` bunu `contribution()`'a
    GERÇEKTEN iletir (.05 ileten mutant hayatta kalıyordu)."""
    from src.stats import decide_primary
    s_xm = _five(7.0, sd=0.8)
    diffs = _five(1.3, sd=1.0)                           # %95'te "katkı var", %97.5'te değil
    out = decide_primary(T(s_xm), _h(s_xm, diffs), quality_per_sec=W_BOTH)
    assert out["primary2"]["alpha"] == pytest.approx(0.025)
    assert not out["primary2"]["state"].startswith("katkı var")
    clean = decide_primary(T(s_xm), _h(s_xm, _five(3.0, sd=0.5)), quality_per_sec=W_BOTH)
    assert clean["primary1"]["row"] == "başarılı" and clean["primary1"]["holm_passed"] is True
    assert clean["primary2"]["state"] == "katkı var, büyük" and clean["primary2"]["holm_passed"] is True


def test_decide_primary_forces_one_seed_set_and_never_drops_a_censored_seed():
    """İki endpoint AYNI seed kümesinde; sansürlü seed DÜŞÜRÜLMEZ, birincil-2 "ölçemedik" olur.

    Denetimin ölçtüğü seçim: H'nin bir seed'i sansürlüyken tam fark listesi "ölçemedik", o seed
    düşürülünce "katkı var, büyük". Artık farkı çağıran değil fonksiyon kuruyor.
    """
    from src.stats import decide_primary
    s_xm = [6.0, 6.5, 5.5, 6.2, 5.8]
    with pytest.raises(ValueError, match="seed kümeleri aynı"):
        decide_primary(T(s_xm), T([4.0, 4.4, 3.6, 4.1]), quality_per_sec=W_BOTH)
    out = decide_primary(T(s_xm), T([4.0, 4.4, 3.6, 4.1, CENS], cens=[4]), quality_per_sec=W_BOTH)
    assert out["primary2"]["state"] == "ölçemedik" and "DÜŞÜRÜLMEZ" in out["primary2"]["note"]
    assert out["primary2"]["p"] == 1.0 and out["primary1"]["row"] == "başarılı"
    assert out["primary2"]["failed_seeds"] == [55]
    with pytest.raises(ValueError, match="bağlayıcı birincil-2"):
        C([2.0, 2.1, 1.9])                               # 3 seed'den bağlayıcı durum çıkmaz


def test_contribution_flags_a_significantly_negative_contribution():
    r = C(_five(-3.0, sd=0.5))
    assert r["state"] == "katkı MDE kadar değil" and "NEGATİF" in r["note"]


# =====================================================================================
# DÖRDÜNCÜ DENETİM TURU + YAPISAL DÜZELTME
#   (1) konuma göre eşleme sınıfı: veri seed KİMLİĞİYLE anahtarlı taşınır, liste ÇALIŞMAZ
#   (2) yavaş seed (hedefe yalnızca zorunlu 2x koşusunda ulaşan) bir yöntem başarısızlığıdır
#   (3) ub95'in aritmetik tavanı HER durumda uygulanır
# =====================================================================================
# Seed etkisi İKİ kolda ortak ve büyük (gerçekçi: aynı seed = aynı veri sırası); katkı küçük ve düzgün.
# Böyle bir girdide yanlış seed'leri eşlemek farka seed etkisini SOKAR ve sonucu değiştirir.
X5 = [6.0, 9.0, 3.0, 7.5, 4.5]
D5 = [2.0, 2.3, 1.7, 2.2, 1.8]                               # X − H: "katkı var, büyük"


def _shuffled(table: dict, order: list[int]) -> dict:
    """Aynı tablo, başka EKLEME sırası (anahtarlar ve değerler aynı)."""
    keys = list(table)
    return {keys[i]: table[keys[i]] for i in order}


def test_every_decision_function_rejects_positional_input():
    """Yanlış kullanım ÇALIŞMAZ, patlar: paralel liste alan hiçbir karar fonksiyonu kalmadı.

    Konuma göre eşleme üç ayrı fonksiyonda üç kez çıktı (`paired`, `savings`, `decide_primary`);
    her seferinde fonksiyon yamalandı ve hata bir sonrakinde geri geldi. Sorun veri biçimiydi.
    """
    from src.stats import contribution, decide_primary, paired, ub95
    be = decide(T(_five(0.0, sd=0.88)), quality_per_sec=NEG)
    calls = {
        "paired": lambda: paired(X5, X5),
        "savings": lambda: savings(X5, X5),
        "decide": lambda: decide(X5, quality_per_sec=W_BOTH),
        "ub95": lambda: ub95(X5),
        "contribution": lambda: contribution(X5, X5),
        "contribution/H": lambda: contribution(T(X5), X5),
        "decide_primary": lambda: decide_primary(X5, X5, quality_per_sec=W_BOTH),
        "decide_primary/H": lambda: decide_primary(T(X5), tuple(X5), quality_per_sec=W_BOTH),
        "abandon": lambda: abandon([be, be], {"S0": 0.5, "S1": 0.5}),
        "abandon/oracle": lambda: abandon({"S0": be, "S1": be}, [0.5, 0.5]),
    }
    for name, call in calls.items():
        try:
            call()
        except TypeError as e:
            assert "anahtarlanmış eşleme" in str(e), name
        else:
            pytest.fail(f"{name}: konumlu girdi KABUL EDİLDİ")


def test_primary2_pairs_by_seed_identity_the_fourth_audit_scenario():
    """Dördüncü denetimin senaryosu: H'nin s22'si yeniden koşulup SONA düştü.

    Konum eşlemesi X-s22'yi H-s33 ile eşliyor ve "katkı var, büyük" olan sonuç "ölçemedik"
    çıkıyordu. Test önce konum eşlemesinin bu girdide GERÇEKTEN farklı sonuç verdiğini gösterir
    (yoksa hiçbir şeyi kilitlemiyor olurdu), sonra kimlik eşlemesinin doğru sonucu verdiğini.
    """
    from src.stats import decide_primary
    x = T(X5)
    h_true = _h(X5, D5)
    h_rerun = _shuffled(h_true, [0, 2, 3, 4, 1])             # 11, 33, 44, 55, 22
    positional = [a.s - b.s for a, b in zip(x.values(), h_rerun.values())]
    assert C(positional)["state"] == "ölçemedik"             # eski hatanın vereceği sonuç
    out = decide_primary(x, h_rerun, quality_per_sec=W_BOTH)
    assert out["primary2"]["state"] == "katkı var, büyük"
    assert out["primary2"]["mean"] == pytest.approx(st.mean(D5))
    assert out["primary2"]["seeds"] == SEEDS10[:5]


def test_results_do_not_depend_on_insertion_order_anywhere():
    """Her karar fonksiyonu, tabloların EKLEME sırasından bit düzeyinde bağımsızdır.

    Sıra karıştıran / eşlemeyi bir kaydıran bir mutant burada yakalanır: X ve H (ve ölçek
    sonuçları ile oracle kazançları) BAĞIMSIZ karıştırılıyor, çıktı birebir aynı kalmalı.
    """
    from src.stats import contribution, decide_primary, paired, ub95
    x, h = T(X5), _h(X5, D5)
    plain = lambda t: {k: o.s for k, o in t.items()}         # noqa: E731
    ref = decide_primary(x, h, quality_per_sec=W_BOTH)
    ref_c, ref_p = contribution(x, h), paired(plain(x), plain(h))
    fb = {s: 100.0 + i for i, s in enumerate(SEEDS10[:5])}
    ff = {s: 90.0 - 3 * i for i, s in enumerate(SEEDS10[:5])}
    ref_s = savings(ff, fb)
    for ox, oh in (([4, 3, 2, 1, 0], [0, 1, 2, 3, 4]), ([1, 2, 3, 4, 0], [2, 0, 4, 1, 3]),
                   ([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])):
        xs_, hs_ = _shuffled(x, ox), _shuffled(h, oh)
        assert decide_primary(xs_, hs_, quality_per_sec=W_BOTH) == ref
        assert contribution(xs_, hs_) == ref_c
        assert decide(xs_, quality_per_sec=W_BOTH) == decide(x, quality_per_sec=W_BOTH)
        assert ub95(xs_) == ub95(x)
        assert paired(plain(xs_), plain(hs_)) == ref_p
        assert savings(_shuffled(ff, ox), _shuffled(fb, oh)) == ref_s
    # durum da kimlikle taşınır: yavaş seed'i taşıyan kayıt sıra değişince başka seed'e GEÇMEZ
    slow = T([0.2, -0.1, 0.3, 0.0, -56.0], slow=[4])
    assert decide(_shuffled(slow, [4, 0, 1, 2, 3]), quality_per_sec=NEG) == decide(slow, quality_per_sec=NEG)
    # abandon: ölçek sonuçları ile oracle kazançları KİMLİKLE eşlenir, çıktı ölçek sırasındadır
    s0 = decide(T(_five(0.0, sd=0.88)), quality_per_sec=NEG)
    s1 = decide(T(_five(-4.0, sd=0.5)), quality_per_sec=NEG)
    a = abandon({"S0": s0, "S1": s1}, {"S0": 1.5, "S1": 2.5})
    b = abandon({"S1": s1, "S0": s0}, {"S1": 2.5, "S0": 1.5})
    assert a == b and a["scales"] == ["S0", "S1"] and a["ub95"] == [s0["ub95"], s1["ub95"]]
    assert "1.50, 2.50" in a["reason"]                       # gerekçede kazançlar ölçek SIRASIYLA


def test_resolve_n_star_applies_the_v3_procedure_mechanically():
    """V3'ün prosedürü: 1x kesiyor -> "hızlı"; kesmiyor, 2x kesiyor -> "yavaş"; o da kesmiyor -> "sansürlü"."""
    from src.stats import resolve_n_star
    budget = 4375 * STEP
    fast = _curve([3.0, 2.95, 2.88, 2.86], first=3000)
    never = _curve([3.0, 2.97, 2.95, 2.93], first=3000)
    late = _curve([2.93, 2.92, 2.89, 2.87], first=4250)
    assert resolve_n_star(fast, None, 2.90, budget) == (n_star(fast, 2.90), "hızlı")
    assert resolve_n_star(never, late, 2.90, budget) == (n_star(late, 2.90), "yavaş")
    assert resolve_n_star(never, never, 2.90, budget) == (2.0 * budget, "sansürlü")
    with pytest.raises(ValueError, match="2x koşu ZORUNLU"):
        resolve_n_star(never, None, 2.90, budget)            # 1x kesmiyorsa 2x eğrisi olmadan karar YOK
    # 1x eğrisi kesiyorsa 2x eğrisine HİÇ bakılmaz (analist "daha iyi" eğriyi seçemez)
    assert resolve_n_star(fast, late, 2.90, budget) == (n_star(fast, 2.90), "hızlı")


def test_slow_seed_is_a_method_failure_not_variance():
    """Yavaş seed aralığı ŞİŞİRMEZ: "4 başabaş + 1 yavaş (1.5x)" belirli bir "başarısız"dır ve terk edilebilir.

    Dördüncü denetim buldu: yavaş seed sıradan gözlem sayılınca düz GA [−42, +20] çıkıyor, kesinlik
    ölçütü tetikleniyor, n=10'da yine çözülmüyor ve satır terminal "yetersiz kesinlik" oluyordu —
    "başarısız" satırı ve vazgeçme, yavaş seed olan HİÇBİR kampanyada ulaşılamıyordu.
    """
    xs = [0.2, -0.1, 0.3, 0.0, -56.0]
    assert indeterminate(*ci95(xs)) is True                  # düz aralık bunu "varyans" diye okurdu
    r = decide(T(xs, slow=[4]), quality_per_sec=NEG)
    assert (r["row"], r["determinate"], r["n_slow"], r["n_censored"]) == ("başarısız", True, 1, 0)
    assert r["ub95"] == pytest.approx((4 * ci95(xs[:4])[2] - 56.0) / 5) and r["ub95"] < SUCCESS
    assert any("yalnızca ZORUNLU 2x koşusunda" in n for n in r["notes"])
    assert not any("LEHİNE yanlıdır" in n for n in r["notes"])      # o not İKAMEYE aittir, ölçülmüş değere değil
    assert abandon({"S0": r}, OK1, escalation_used=False)["abandon"] is True
    # n=10, 9 × −8 + 1 yavaş (1.9x): aynı şekilde belirli
    r10 = decide(T([-8.0 + 0.1 * i for i in range(9)] + [-140.0], slow=[9]), quality_per_sec=NEG)
    assert r10["row"] == "başarısız" and r10["determinate"] is True
    # durum bilgisi VERİLMEZSE (eski davranış) aynı sayılar ölçülemez çıkar: kilitlenen fark bu
    assert decide(T(xs), quality_per_sec=NEG)["row"] == "yüksek varyans / belirsiz"


def test_slow_and_censored_are_continuous_at_the_2x_boundary():
    """1.99x'te ulaşan seed (−148.75) ile hiç ulaşamayan seed (−149.82) ZIT sonuç vermez."""
    base = [0.2, -0.1, 0.3, 0.0]
    slow = decide(T(base + [-148.75], slow=[4]), quality_per_sec=NEG)
    cens = decide(T(base + [CENS], cens=[4]), quality_per_sec=NEG)
    assert slow["row"] == cens["row"] == "başarısız"
    assert slow["determinate"] is True and cens["determinate"] is True
    assert abs(slow["ub95"] - cens["ub95"]) < 0.5
    for r in (slow, cens):
        assert abandon({"S0": r}, OK1)["abandon"] is True


def test_slow_seed_does_not_count_toward_the_b8_censoring_rule():
    """B8'in ">= 2 sansür -> başarısız" ve "başarı yasak" hükümleri İKAMEYE bağlıdır; yavaş seed'in
    değeri ölçülmüştür ve sayıma GİRMEZ (ama üst sınıra sabit değer olarak girer)."""
    vals = [30.0, 30.1, 29.9, -30.0, -30.0]
    two_slow = decide(T(vals, slow=[3, 4]), quality_per_sec=W_BOTH)
    assert two_slow["n_censored"] == 0 and not any("B8" in n for n in two_slow["notes"])
    assert two_slow["row"] == "yüksek varyans / belirsiz"    # s̄ = 6, ub95 ≈ 6.1: eşik çözülemiyor
    two_cens = decide(T(vals, cens=[3, 4]), quality_per_sec=W_BOTH)
    assert two_cens["row"] == "başarısız" and any("B8" in n for n in two_cens["notes"])
    # n=10'da hafif bir yavaş seed başarıyı YASAKLAMAZ (değer ölçülmüş, ortalamanın içinde);
    # aynı seed sansürlü olsaydı B8 yasaklardı. Ön-kayıttaki anlam kararının iki yüzü:
    ten = [20.0 + 0.1 * i for i in range(9)] + [-30.0]
    slow10 = decide(T(ten, slow=[9]), quality_per_sec=W_BOTH)
    assert slow10["row"] == "güçlü başarı" and slow10["ci95"][0] == pytest.approx(3.957, abs=0.01)
    cens10 = decide(T(ten, cens=[9]), quality_per_sec=W_BOTH)
    assert cens10["row"] == "belirsiz" and any("1 sansürlü seed varken" in n for n in cens10["notes"])


def test_precision_criterion_has_one_form_and_uses_ub95():
    """Kesinlik ölçütü TEK biçimdir: belirli ⟺ (GA alt > 0 ve s̄ >= 5) ya da ub95 < 5.

    Başarısız seed yokken bu, düz aralıkla birebir aynıdır. Varken: "1 sansür + dağınık hızlı
    seed'ler" artık 20 anlamsız ek koşuya gitmez (üst sınır −11, eşik DIŞLANMIŞ).
    """
    import random
    rnd = random.Random(7)
    for _ in range(300):
        xs = [rnd.gauss(rnd.choice([0, 4, 5, 7]), rnd.choice([0.9, 3.0, 6.0])) for _ in range(5)]
        r = decide(T(xs), quality_per_sec=W_BOTH)
        assert (not r["determinate"]) == indeterminate(*ci95(xs)), xs
        assert r["ub95"] == pytest.approx(min(100.0, ci95(xs)[2]))
    r = decide(T([20.0, -5.0, 12.0, 1.0, CENS], cens=[4]), quality_per_sec=NEG)
    assert r["row"] == "başarısız" and r["determinate"] is True and r["ub95"] < -10
    # abandon()'daki "belirsiz satır yok" şartı ub95 < 5'ten ÇIKAR: ub95 < 5 olan belirsiz satır olamaz
    for xs, kw in (([-9, -4.5, 0, 4.5, 9], {}), ([30.0, 30.1, 29.9, -30.0, -30.0], {"slow": [3, 4]})):
        rr = decide(T(xs, **kw), quality_per_sec=NEG)
        assert rr["determinate"] is False and rr["ub95"] >= SUCCESS


def test_ub95_arithmetic_cap_applies_in_every_case():
    """Aritmetik tavan yalnızca n_h < 2'de değil HER durumda uygulanır (bağlayıcı metin de böyle).

    `[−62.5, −125, C, C, C]`: iki hızlı seed'in t(1) aralığı +303'e çıkıyor; tavansız biçim +31.4
    verip en net başarısızlığı terk edilemez kılardı. Tavanlı biçim −49.9.
    """
    from src.stats import ub95
    t = T([-62.5, -125.0, CENS, CENS, CENS], cens=[2, 3, 4])
    uncapped = (2 * ci95([-62.5, -125.0])[2] + 3 * CENS) / 5
    assert uncapped > 30
    assert ub95(t) == pytest.approx((2 * 100.0 + 3 * CENS) / 5) and ub95(t) < -49
    # yavaş seed'ler de "sabit" tarafta sayılır
    t2 = T([-62.5, -125.0, -80.0, CENS, CENS], cens=[3, 4], slow=[2])
    assert ub95(t2) == pytest.approx((2 * 100.0 - 80.0 + 2 * CENS) / 5)


def test_the_1x_boundary_is_the_only_status_discontinuity_and_it_is_documented():
    """BİLİNEN ve ön-kayıtlı sınır (KARAR TABLOSU, "tek durum süreksizliği"): 1x bütçenin hemen
    içinde ulaşan seed (−24) sıradan gözlemdir ve aralığı genişletir; hemen dışında ulaşan (−26)
    sabit sayılır. Birincisi "daha çok seed" ister, ikincisi belirli bir "belirsiz"dir. Hiçbiri
    başarı değildir ve hiçbiri TEK ölçekle terk edilemez; ama yavaş taraf iki ardışık ölçekte
    tekrarlanırsa vazgeçmeyi tetikler (ub95 = +3.4 < 5) — metin bunu açıkça yazıyor, test kilitliyor."""
    inside = decide(T([10.0, 10.5, 9.5, 10.2, -24.0]), quality_per_sec=W_BOTH)
    outside = decide(T([10.0, 10.5, 9.5, 10.2, -26.0], slow=[4]), quality_per_sec=W_BOTH)
    assert inside["row"] == "yüksek varyans / belirsiz"
    assert outside["row"] == "belirsiz" and outside["determinate"] is True
    assert outside["ub95"] == pytest.approx(3.38, abs=0.01)
    for r in (inside, outside):
        assert r["row"] not in ("güçlü başarı", "başarılı", "başarısız")
        assert abandon({"S0": r}, OK1)["abandon"] is False
    assert abandon({"S0": inside, "S1": inside}, OK2, escalation_used=True)["abandon"] is False
    assert abandon({"S0": outside, "S1": outside}, OK2, escalation_used=True)["abandon"] is True


def test_failed_seed_in_either_arm_makes_primary2_unmeasured_never_dropped():
    """Yavaş seed de sansürlü gibi birincil-2'yi "ölçemedik" yapar; hangi seed olduğu KİMLİKLE raporlanır."""
    from src.stats import decide_primary
    h = _h(X5, D5)
    out = decide_primary(T(X5[:4] + [-60.0], slow=[4]), h, quality_per_sec=W_BOTH)
    assert out["primary2"]["state"] == "ölçemedik" and out["primary2"]["failed_seeds"] == [55]
    assert out["primary2"]["p"] == 1.0 and out["primary2"]["holm_passed"] is False
    h_slow = {**h, 22: Obs(-70.0, "yavaş")}
    out2 = decide_primary(T(X5), h_slow, quality_per_sec=W_BOTH)
    assert out2["primary2"]["failed_seeds"] == [22] and out2["primary1"]["row"] == "başarılı"
