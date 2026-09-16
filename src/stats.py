"""Karar istatistiği: eşleştirilmiş farklar, çoklu karşılaştırma ve ön-kayıttaki karar tablosu — KOD olarak.

Neden ayrı bir modül: PREREGISTRATION.md'nin karar kuralı eşleştirilmiş güven aralıklarına dayanıyor ve
v3 denetimi bu koddaki iki hatayı buldu:

  1. `paired()` seed KİMLİĞİNE göre değil liste KONUMUNA göre eşliyordu. Bir koşulda tek bir analiz eksik
     olduğunda sessizce yanlış seed'leri eşliyordu -> eşleştirilmiş GA anlamsız olurdu.
  2. t kritik değeri n'den bağımsız sabitti (t(2)); n=2'de güven aralığını gerçeğin ~3'te birine daraltıyordu.

Bu hatalar FOTON kararında tekrarlanırsa karar GEÇERSİZ olur, çünkü karar kuralının tamamı o güven
aralığının işaretine bakıyor. Bu yüzden fonksiyonlar burada toplanıp `tests/test_stats.py` ile regresyona
bağlandı; FOTON'un karar betiği bu modülü kullanmak ZORUNDA (PREREGISTRATION O4).

`decide()` karar tablosunu birebir uygular ve satır ADINI döndürür. Rapor "iyi görünüyor" diyemez:
satır adı buradan gelir.
"""
from __future__ import annotations

import math
import statistics as st

# %97.5 t kritik değerleri (iki yanlı %95), df -> t
T975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582, 6: 2.446912, 7: 2.364624,
        8: 2.306004, 9: 2.262157, 10: 2.228139, 14: 2.144787, 19: 2.093024, 29: 2.045230}

SD_TRIPWIRE = 5.0       # puan; aşılırsa "yüksek varyans / belirsiz" (PREREGISTRATION seed politikası)
STRONG = 10.0           # puan; "güçlü başarı" eşiği
SUCCESS = 5.0           # puan; "başarılı" eşiği
NEUTRAL = 2.0           # puan; "nötr" bandının yarısı


def t_crit(df: int) -> float:
    """df için %97.5 t kritik değeri; tabloda yoksa en yakın alt df (muhafazakâr) ya da normal yaklaşım."""
    if df in T975:
        return T975[df]
    lower = [d for d in T975 if d < df]
    return T975[max(lower)] if lower else 1.959964


def t_sf(t: float, df: int) -> float:
    """P(T > t), Student-t dağılımı. Küçük df için düzenlenmiş inkomplet beta; scipy bağımlılığı yok."""
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Düzenlenmiş inkomplet beta I_x(a,b), sürekli kesir (Numerical Recipes 6.4)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1) / (a + b + 2):
        return math.exp(math.log(x) * a + math.log1p(-x) * b - lbeta) * _bcf(a, b, x) / a
    return 1.0 - math.exp(math.log1p(-x) * b + math.log(x) * a - lbeta) * _bcf(b, a, 1 - x) / b


def _bcf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-16) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / ((1.0 + aa * d) if abs(1.0 + aa * d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / ((1.0 + aa * d) if abs(1.0 + aa * d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def paired(a: list[float], b: list[float], seeds_a=None, seeds_b=None) -> dict:
    """Eşleştirilmiş fark (a - b), **seed kimliğine göre**. Ortalama, sd, %95 GA t(n-1), iki yanlı p.

    seeds_a/seeds_b verilmezse konuma göre eşler ve bu YALNIZCA aynı sırayla verildiği kesin olan
    listeler için doğrudur; karar hesabında daima seed listeleri verilmelidir (bkz. modül docstring'i).
    """
    if seeds_a is not None and seeds_b is not None:
        pos = {s: i for i, s in enumerate(seeds_b)}
        pairs = [(a[i], b[pos[s]]) for i, s in enumerate(seeds_a) if s in pos]
        matched = [s for s in seeds_a if s in pos]
    else:
        pairs = list(zip(a, b))
        matched = None
    d = [x - y for x, y in pairs]
    out = {"per_seed": d, "n": len(d), "seeds": matched, "mean": st.mean(d) if d else float("nan")}
    if len(d) > 1:
        sd = st.stdev(d)
        out["sd"] = sd
        if sd > 0:
            t = out["mean"] / (sd / math.sqrt(len(d)))
            out["t"], out["p"] = t, t_sf(abs(t), len(d) - 1) * 2
        h = t_crit(len(d) - 1) * sd / math.sqrt(len(d))
        out["ci95"] = [out["mean"] - h, out["mean"] + h]
    return out


def multiplicity(tests: list[tuple[str, float]], alpha: float = 0.05) -> dict:
    """Holm (FWER) ve Benjamini-Hochberg (FDR): hangi nominal anlamlı testler düzeltmeden geçiyor.

    NaN p değerli testler (sd = 0 dejenere) havuza girmez. Bir AİLE içinde uygulanır: birincil test
    düzeltmeye girmez (tek ve önceden belirli), teşhis ölçütleri kendi ailesi içinde düzeltilir.
    """
    rows = sorted(((n, p) for n, p in tests if p == p), key=lambda r: r[1])
    m = len(rows)
    holm: set[str] = set()
    for i, (name, pv) in enumerate(rows):
        if pv <= alpha / (m - i):
            holm.add(name)
        else:
            break
    bh: set[str] = set()
    kmax = max((i for i, (_, pv) in enumerate(rows) if pv <= alpha * (i + 1) / m), default=None)
    if kmax is not None:
        bh = {name for name, _ in rows[:kmax + 1]}
    return {"n_tests": m, "alpha": alpha, "nominal": [n for n, pv in rows if pv <= alpha],
            "holm": sorted(holm), "bh": sorted(bh)}


# Karar tablosu satır adları — rapor bunlardan birini BİREBİR yazmak zorunda.
ROWS = ("güçlü başarı", "başarılı", "belirsiz", "nötr", "başarısız", "yüksek varyans / belirsiz")


def decide(savings: list[float], quality_per_sec: float | None = None, censored: int = 0) -> dict:
    """PREREGISTRATION.md karar tablosunu (O5) birebir uygular.

    savings        : seed başına tasarruf, YÜZDE PUAN (s_i = 100*(1 - F_foton/F_base))
    quality_per_sec: w̄ ortalaması (yüzde puan); None ise "güçlü başarı" satırı değerlendirilemez
    censored       : L*'e ulaşmayan (sansürlü) seed sayısı; > 0 ise "başarılı"/"güçlü başarı" YASAK

    Döner: {"row": satır adı, "mean", "sd", "ci95", "n", "notes": [...]}
    """
    if len(savings) < 2:
        raise ValueError("karar için en az 2 seed gerekir")
    n = len(savings)
    mean, sd = st.mean(savings), st.stdev(savings)
    h = t_crit(n - 1) * sd / math.sqrt(n)
    lo, hi = mean - h, mean + h
    notes: list[str] = []
    res = {"mean": mean, "sd": sd, "n": n, "ci95": [lo, hi], "notes": notes}

    if sd > SD_TRIPWIRE:                                   # satır 0: tripwire her şeyin önünde
        notes.append(f"örneklem sd {sd:.2f} > {SD_TRIPWIRE} puan: seed 10'a çıkar")
        return {**res, "row": "yüksek varyans / belirsiz"}

    crosses_zero = lo <= 0 <= hi
    if mean >= STRONG and lo > 0 and quality_per_sec is not None and quality_per_sec > 0:
        row = "güçlü başarı"
    elif mean >= SUCCESS and lo > 0:
        row = "başarılı"
        if mean >= STRONG and (quality_per_sec is None or quality_per_sec <= 0):
            notes.append("FLOP kazancı >= %10 ama quality/sec pozitif değil -> "
                         "çelişki kuralı: mimari başarı / sistem başarısızlığı")
    elif mean > NEUTRAL and (crosses_zero or mean < SUCCESS):
        row = "belirsiz"
    elif -NEUTRAL <= mean <= NEUTRAL:
        row = "nötr"
    else:                                                  # mean < -2
        row = "başarısız"
        if crosses_zero:
            notes.append("ortalama < -%2 ama GA sıfırı kesiyor (O5: negatif ortalama 'belirsiz' üretmez)")

    if censored > 0 and row in ("güçlü başarı", "başarılı"):
        notes.append(f"{censored} seed L*'e ulaşmadı (sansürlü): O3 gereği 'başarılı' ilan edilemez")
        row = "belirsiz"
    elif censored > 0:
        notes.append(f"{censored} seed L*'e ulaşmadı (sansürlü); s̄ sansür ikamesiyle hesaplandı")
    return {**res, "row": row}
