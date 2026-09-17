"""FOTON v0.2 attention deseni aritmetiğinin regresyon testleri.

Denetimde bulunan hata: ön-kayıt "A/B/C token başına 256 attention bağlantısında eşitlendi"
diyordu. T=512 ve nedensel maske altında bu imkânsız — i konumu yalnızca j <= i'ye bakabildiği
için tam nedensel desen bile 256.50'de, dilated r=2 ise 128.50'de tavan yapıyor. Eşitleme hedefi
bu yüzden dilated desenin tavanına indirildi.

Bu neden ciddiydi: "eşit hesap" iddiası X kolunun bütün karşılaştırmasını taşıyor. Sayı yanlışsa
modüller arası fark hesap farkından geliyor olabilir ve birincil sonuç yorumlanamaz hale gelir.
Sayılar ön-kayıtta düz metin kaldığı sürece bir parametre (W, r, s, T) değişince sessizce yanlışa
dönerdi; buradaki testler o sessiz değişimi gürültülü hale getirir.

Aşağıdaki testler beş şeyi kilitler:
  1. dört başlık sayısı (256.50 / 128.17 / 128.50 / 128.38) — parametre değişirse düşer,
  2. analitik kapalı form ile GERÇEKLEŞEN maskenin satır toplamlarının birebir uyuşması,
  3. nedensellik (j > i yok) ve her sorgunun en az kendisini görmesi,
  4. A/B/C arası kalan sapmanın 0.35 bağlantının altında kalması (ön-kaydın eşitleme iddiası),
  5. dilated r=2'nin tavanı: pencere ne kadar büyürse büyüsün 128.50 aşılamaz.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.attn_patterns import (EQ_TOL, FOTON_MODULES, PATTERNS, T_KAMPANYA, Head, Pattern,
                               baseline_gap, dilated, equalization, full, head_split, headline,
                               local, per_head_avg, self_check, strided)

T_LER = (1, 2, 3, 7, 16, 63, 128, 150, 256, 511, 512, 513)


def test_headline_numbers_at_512():
    """Dört başlık sayısı: 256.50 / 128.17 / 128.50 / 128.38.

    Bunlar ön-kayıtta yazılı. Bir parametre (A_LOCAL_W, B_DILATION_R, C_STRIDE_S, ...) veya T
    değişirse bu test düşer — ön-kayıt ile kod arasındaki bağ budur.
    """
    got = {k: round(PATTERNS[k].avg(T_KAMPANYA), 2) for k in ("full", "A", "B", "C")}
    assert got == {"full": 256.50, "A": 128.17, "B": 128.50, "C": 128.38}, got


def test_full_causal_average_is_exact_half_plus_half():
    """Tam nedensel ortalama tam olarak (T+1)/2 — baseline'ın gördüğü bağlantı sayısı."""
    for T in T_LER:
        assert full().avg(T) == pytest.approx((T + 1) / 2)


def test_local_256_and_the_88_31_row_reproduce():
    """Denetimin 'neden 256 tutulamadı' tablosundaki diğer iki satır: 192.25 ve 88.31."""
    assert round(local(256).avg(T_KAMPANYA), 2) == 192.25
    mix = head_split("128+128", Head(1, 128, "yerel W=128"), Head(4, 128, "strided s=4 W=128"))
    assert round(mix.avg(T_KAMPANYA), 2) == 88.31


@pytest.mark.parametrize("T", T_LER)
@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_mask_rows_equal_analytic_counts(name, T):
    """Kapalı form ile gerçekleşen maske BİREBİR uyuşmalı — modülün varlık sebebi bu.

    Formül ile eğitimde kullanılacak maske ayrışırsa raporlanan bağlantı sayıları modelin
    gerçekten gördüğü sayılar olmaz; 'eşit hesap' iddiası dayanaksız kalır.
    """
    for head in PATTERNS[name].heads:
        rows = head.mask(T).sum(axis=1)
        assert np.array_equal(rows, head.counts(T)), f"{name}/{head.label} T={T}"
        assert float(rows.sum()) / T == pytest.approx(head.avg(T)), "kapalı form ortalama sapıyor"


@pytest.mark.parametrize("T", T_LER)
@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_patterns_are_strictly_causal_and_self_attending(name, T):
    """Hiçbir desende j > i girdisi yok ve her sorgu en az kendisini görüyor.

    İkincisi önemsiz değil: strided/dilated ofset ailesinden 0 ofseti düşerse bazı konumlar
    hiçbir şeye bakamaz (maske satırı boş -> softmax NaN).
    """
    for head in PATTERNS[name].heads:
        m = head.mask(T)
        i = np.arange(T)[:, None]
        j = np.arange(T)[None, :]
        assert not bool((m & (j > i)).any()), f"{name}/{head.label}: nedensellik ihlali"
        assert bool(np.diag(m).all()), f"{name}/{head.label}: bir sorgu kendisini görmüyor"
        assert int(m.sum(axis=1).min()) >= 1


@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_position_zero_sees_exactly_one_position(name):
    """Sınır davranışı: 0 konumu HER desende tam 1 pozisyon görür (yalnızca kendisi)."""
    for head in PATTERNS[name].heads:
        assert int(head.counts(T_KAMPANYA)[0]) == 1
        assert int(head.mask(T_KAMPANYA)[0].sum()) == 1
        assert bool(head.mask(T_KAMPANYA)[0, 0])


def test_A_B_C_are_equalized_within_tolerance():
    """A/B/C arası kalan sapma <= 0.35 bağlantı: ön-kaydın eşitleme iddiası.

    Bir parametre değişirse bu test YÜKSEK SESLE düşmeli — eşitleme iddiası ön-kayıtta donmuş
    ve sessizce bozulması, X kolundaki modül farkını hesap farkıyla karıştırmak demektir.
    """
    # Eşiğin KENDİSİ de pinlenmeli. Aşağıdaki karşılaştırmalar EQ_TOL'ü kullandığı için, eşik
    # gevşetilirse (0.35 -> 0.40 ya da 100.0) hepsi geçmeye devam eder: mutasyon denemesinde
    # EQ_TOL=100.0 ile 118 testin 118'i geçti, yani ön-kayıtta donmuş tolerans sessizce
    # anlamsızlaştırılabiliyordu. Tripwire eşiği literal olarak sabitlenmezse tripwire değildir.
    assert EQ_TOL == 0.35, f"ön-kayıtta donmuş tolerans değişmiş: {EQ_TOL}"
    e = equalization(T_KAMPANYA)
    assert e["spread"] <= EQ_TOL, e
    assert round(e["spread"], 2) == 0.33, e["spread"]
    assert e["spread_pct"] < 0.30, e            # ortalamanın binde 2.5'i; %1'in çok altında
    assert e["within_tol"]
    for k in FOTON_MODULES:
        assert abs(e["per_module"][k] - e["mean"]) <= EQ_TOL
    # Sapma tam değerle de kilitli: 167/512 = 0.326171875. EQ_TOL'e payı yalnızca 0.024
    # bağlantı, yani tolerans BİLEREK dar — herhangi bir parametre oynaması eşiği geçirir.
    assert e["spread"] == pytest.approx(167 / 512)
    assert e["spread_pct"] == pytest.approx(0.2541, abs=1e-4)   # ön-kayıt %0.25 yazmalı, %0.26 değil


def test_dilated_rate_two_cannot_exceed_128_50():
    """r=2 dilated desenin TAVANI 128.50; pencere ne kadar büyürse büyüsün aşılamaz.

    Eşitleme hedefini 256'dan indiren olgu budur: r=2 ile geçmişin yarısı zaten erişilemez,
    yani T=512'de bütçeyi büyütmek 128.50'nin ötesine geçmez ve yerel pencere 256'nın 192.25'ine
    ASLA yetişemez. Hedef bu yüzden dilated tavanına çekildi; bu test o tavanı kilitler.
    """
    ceiling = 128.50
    for W in (1, 2, 64, 128, 255, 256, 257, 512, 1024, 10_000):
        v = dilated(2, W).avg(T_KAMPANYA)
        assert v <= ceiling + 1e-12, f"W={W} tavanı aştı: {v}"
    assert dilated(2, 256).avg(T_KAMPANYA) == pytest.approx(ceiling)
    # doygunluk: 256'dan sonra bütçe büyütmek HİÇBİR bağlantı eklemiyor
    for W in (257, 512, 10_000):
        assert dilated(2, W).avg(T_KAMPANYA) == pytest.approx(ceiling)
    assert dilated(2, 10_000).avg(T_KAMPANYA) < local(256).avg(T_KAMPANYA)


def test_budget_means_selected_positions_not_span():
    """W = SEÇİLEN POZİSYON SAYISI, açıklık değil. Yanlış okuma B'yi 96.25 yapardı.

    Ön-kayıt B'yi "256 pozisyon, seyrekleştirilmiş: aynı bütçeyle daha uzak geçmiş" diye
    tanımlıyor. W açıklık okunursa (256'lık şeritte her ikinci konum) bütçe 128'e düşer ve
    ortalama 96.25 olur — eşitleme hedefini 32 bağlantı kaydıran yorum hatası tam burada.
    """
    assert dilated(2, 256).avg(T_KAMPANYA) == pytest.approx(128.50)
    assert dilated(2, 128).avg(T_KAMPANYA) == pytest.approx(96.25)    # açıklık okuması
    assert PATTERNS["B"].heads[0].max_span(T_KAMPANYA) == 510         # açıklık 511 pozisyon


def test_C_head_averages_are_heterogeneous_by_design():
    """C'nin head'leri BİLEREK farklı: 192.25 (yerel) ve 64.50 (strided), ortalama 128.375.

    Ön-kaydın "eşitlenemeyen eksen 5"i bu. Eşitleme iddiası head ORTALAMASI üzerinden kurulduğu
    için head başına değerlerin ayrıca raporlanması şart; bu test o iki sayıyı kilitler.
    """
    per = per_head_avg("C", T_KAMPANYA)
    assert per == pytest.approx([192.25, 64.50])
    assert PATTERNS["C"].avg(T_KAMPANYA) == pytest.approx(sum(per) / 2)
    assert PATTERNS["A"].head_avgs(T_KAMPANYA) == pytest.approx([128.173828125])   # A homojen


def test_C_mask_has_half_heads_of_each_pattern():
    """C maskesi [n_head, T, T] ve head'lerin yarısı yerel, yarısı strided."""
    T, n_heads = 64, 8
    m = PATTERNS["C"].masks(T, n_heads)
    assert m.shape == (n_heads, T, T)
    per_head = m.sum(axis=(1, 2))
    lo, hi = sorted(set(per_head.tolist()))
    assert (per_head == hi).sum() == 4 and (per_head == lo).sum() == 4
    assert float(per_head.mean()) / T == pytest.approx(PATTERNS["C"].avg(T, n_heads))


def test_head_split_requires_divisible_head_count():
    """Tam bölünmeyen n_head sessizce kabul edilmemeli: 'yarısı/yarısı' iddiası yanlış olur."""
    with pytest.raises(ValueError, match="eşit bölünemiyor"):
        PATTERNS["C"].avg(T_KAMPANYA, n_heads=7)
    assert PATTERNS["A"].avg(T_KAMPANYA, n_heads=7) == pytest.approx(128.173828125)


def test_baseline_gap_direction_is_against_foton():
    """Açığın YÖNÜ: FOTON modülleri baseline'ın yarısı kadar bağlantı görüyor, tersi değil.

    Yön makine-okunur tutuluyor (`favors`) ki rapor üreteci bunu ters yazamasın; ön-kayıt bu
    cümlenin sonuç tablosunun başında durmasını şart koşuyor.
    """
    g = baseline_gap(T_KAMPANYA)
    assert g["favors"] == "baseline"
    assert g["foton_mean"] < g["baseline"]
    assert g["ratio"] == pytest.approx(0.5, abs=0.005)
    assert g["deficit"] == pytest.approx(128.15, abs=0.01)


def test_headline_states_the_direction_explicitly():
    """Başlık satırı 'ALEYHİNE' demek ZORUNDA — dipnot diline kaymasın."""
    h = headline(T_KAMPANYA)
    assert "ALEYHİNE" in h and "YARISI" in h
    assert "256.50" in h and "128.35" in h


def test_self_check_passes_for_every_registered_pattern():
    """Üç bağımsız yol (kapalı form / konum toplamı / maske satır toplamı) her desende uyuşuyor."""
    out = self_check(T_KAMPANYA)
    assert set(out) == set(PATTERNS)
    assert out["full"] == pytest.approx(256.5)


def test_mask_guard_actually_fires():
    """Maske–formül uyuşma kontrolü GERÇEKTEN çalışıyor mu: bozuk formül hata vermeli.

    Bu test olmadan yukarıdaki uyuşma testleri boş güvence olurdu (kontrol sessizce kaldırılsa
    kimse fark etmezdi). `python -O` ile de kalkmaması için kontrol çıplak assert değil.
    """
    class YanlisHead(Head):
        def counts(self, T: int) -> np.ndarray:                      # kasıtlı bozuk kapalı form
            return super().counts(T) + 1

    with pytest.raises(AssertionError, match="uyuşmuyor"):
        YanlisHead(1, 8, "bozuk").mask(16)


def test_causality_guard_actually_fires():
    """Nedensellik kontrolü de gerçek: nedensel olmayan bir desen ADIYLA yakalanmalı."""
    class IleriHead(Head):
        def mask(self, T: int) -> np.ndarray:                        # geleceğe bakan bozuk maske
            m = np.ones((T, T), dtype=bool)
            from src.attn_patterns import _verify_mask
            _verify_mask(m, self.counts(T), self.label)
            return m

    with pytest.raises(AssertionError, match="nedensel değil"):
        IleriHead(1, 8, "ileri").mask(16)


def test_strided_and_dilated_are_the_same_offset_family():
    """Ofset ailesi olarak strided ile dilated aynı; ayrım neyle birleştirildiğinden geliyor.

    Tek doğruluk kaynağı iddiasının testi: iki ad tek aritmetiğe bağlı olmasa C'nin strided
    head'i ile B ayrı ayrı bozulabilirdi.
    """
    for T in (16, 128, 512):
        a = strided(4, 32).heads[0]
        b = dilated(4, 32).heads[0]
        assert np.array_equal(a.mask(T), b.mask(T))
        assert a.avg(T) == b.avg(T)


def test_reach_metrics_report_the_non_equalized_axes():
    """Eşit bağlantıda bile alım alanı farklı: A 149/150, B 510/256, C head başına 255/256 ve 508/128.

    Ön-kaydın "eşitlenemeyen eksen 1 ve 2"si bu sayılarla raporlanıyor; eşitlenen tek eksen
    bağlantı SAYISI. C'nin iki değeri de ayrıca kilitli, çünkü ikisi de FOTON'un LEHİNE
    sapabilir (doygunluk gizlenirse 128 yerine 256 ayrık pozisyon, head'ler toplanırsa 255
    yerine 508 erişim okunur) ve bu modülün işi sapmanın yönünü gizlememek.
    """
    a = PATTERNS["A"].summary(T_KAMPANYA)
    b = PATTERNS["B"].summary(T_KAMPANYA)
    assert (a["max_span"], a["distinct_reachable"]) == (149, [150])
    assert (b["max_span"], b["distinct_reachable"]) == (510, [256])
    assert a["avg"] == pytest.approx(b["avg"], abs=EQ_TOL)          # bağlantı eşit
    assert b["max_span"] > 3 * a["max_span"]                        # alım alanı eşit DEĞİL

    # C de pinlenmeli, ve head BAŞINA. İki sayı da FOTON'un lehine sapabilecek türden:
    #  - strided head'in etiketi W=256 ama T=512'de 128 ayrık pozisyonda doygunluğa oturuyor;
    #    doygunluk raporlanmazsa C 256 pozisyon görüyormuş gibi okunur (mutasyon denemesinde
    #    128 -> 256 kaydırıldığında 118 testin hiçbiri düşmüyordu).
    #  - "en uzak" head'ler üzerinden max; C'nin yerel yarısı yalnızca 255 geriye uzanıyor,
    #    508 sayısı tek başına C'nin alım alanını olduğundan geniş gösterir.
    c = PATTERNS["C"].summary(T_KAMPANYA)
    assert c["distinct_reachable"] == [256, 128], c["distinct_reachable"]
    assert c["head_max_spans"] == [255, 508], c["head_max_spans"]
    assert c["max_span"] == max(c["head_max_spans"]) == 508


def test_invalid_parameters_are_rejected():
    """step/budget doğrulaması: sessiz 0 veya negatif bütçe boş maske üretirdi."""
    with pytest.raises(ValueError, match="step"):
        Head(0, 8, "sıfır adım")
    with pytest.raises(ValueError, match="budget"):
        Head(1, 0, "sıfır bütçe")


def test_pattern_registry_is_the_single_source():
    """Kayıt dört deseni de içeriyor ve X kolunun modülleri A/B/C."""
    assert set(PATTERNS) == {"full", "A", "B", "C"}
    assert FOTON_MODULES == ("A", "B", "C")
    assert all(isinstance(PATTERNS[k], Pattern) for k in PATTERNS)
