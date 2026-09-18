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

STRONG = 10.0           # puan; "güçlü başarı" eşiği
SUCCESS = 5.0           # puan; "başarılı" eşiği
NEUTRAL = 2.0           # puan; "nötr" bandının yarısı
REQUIRED_N = (5, 10)    # bağlayıcı karar yalnızca 5 ya da 10 eşleştirilmiş seed'le verilir
# "güçlü başarı" için w̄'nin ölçülmüş olması GEREKEN baseline'lar. İkisi de olmadan o satır
# erişilemez: ön-kayıt `w̄(M) > 0` VE `w̄(D) > 0` istiyor, çünkü M bu kartta kendi başına D'den
# token başına %27.9 yavaştır ve yalnızca M'e karşı bir hız kazancı FOTON'un katkısı olmadan da
# sağlanabilir (v01 EK1-B16'nın aynı kolda bulduğu sorun).
WBAR_KEYS = ("M", "D")
CI_EPS = 1e-9           # GA sınırı sıfıra kayan nokta hassasiyetinde değerken kararı bit çevirmesin

# Mutlak sd tripwire'ı (v01: sd > 5 puan) KULLANIMDAN KALDIRILDI. Neden, ölçümle:
#
#   gerçek sd = 5 -> P(örnek sd > 5): n=5'te %40.6, n=10'da %43.8
#   gerçek sd = 6 -> P(örnek sd > 5): n=5'te %59.7, n=10'da %71.5
#
# Eşik mutlak olduğu ve n ile ölçeklenmediği için tripwire'ın TEK çaresi (seed'i 10'a çıkarmak)
# tripwire'ın kendisini DAHA olası yapıyordu; tabloda "n=10'da tripwire ateşlendi" satırı da yoktu,
# yani kural kendi çıkmazını üretiyordu. Yerine KESİNLİK ölçütü geldi (bkz. `indeterminate`):
# güven aralığı başarı eşiğini çözemiyorsa sonuç belirsizdir, ve o ölçüt n büyüdükçe KÜÇÜLÜR —
# çare artık kuralla aynı yöne çalışıyor.
SD_TRIPWIRE = None


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


def n_star(curve: list[tuple[float, float]], target: float) -> float | None:
    """`N*`: eğrinin `target`'ın altına KALICI olarak indiği token sayısı. Kural burada PİNLİ.

    curve : (token, val_loss) çiftleri; sıralı olmak zorunda değil, sıralanır.

    İki karar bu fonksiyonda dondurulmuştur, çünkü ikisi de birincil istatistiği oynatıyor ve
    ön-kayıtta serbest bırakılmaları "sonucu gördükten sonra seçme" kapısıydı:

    1. **İnterpolasyon TOKEN ekseninde doğrusal.** ln(token) ekseninde doğrusal yapmak `N*`'ı bir
       eval aralığı içinde ~%0.06 oynatıyor. Küçük ama serbest bir parametre olmamalı.
       (`L*`'in kendisi interpolasyonsuzdur — bütçe %80 noktası tam bir eval noktasına düşecek
       şekilde seçildi; interpolasyon yalnızca `N*`'da devreye girer.)

    2. **Kesişim SÜRDÜRÜLMÜŞ olmalı** (v01 EK1-B14: "kesişimden sonraki eval noktası da ≤ L* olmalı;
       değilse SONRAKİ kesişime bakılır"). Uygulama: `N*`, hedefin ÜSTÜNDEKİ son eval noktasından
       hemen sonraki kesişimdir — yani ardından eğrinin bir daha hedefin üstüne çıkmadığı ilk
       kesişim. Gürültülü bir eğri hedefin altına bir kez sarkıp geri çıkarsa o erken sarkma
       `N*` sayılmaz: ölçülü örnek, 2750'de sarkıp 3000'de geri çıkan bir eğride ilk kesişim `N*`'ı
       114.7M yerine 90.1M gösteriyor, yani `s_i`'yi ~21 puan şişirip "nötr"ü "güçlü başarı" yapıyor.

       Bu fonksiyonun bir önceki hâli böyle bir eğride None döndürüp seed'i SANSÜRLÜYORDU — B14'ün
       "sonraki kesişime bakılır" hükmünün tersi, ve ağır: 0.0005 nat'lık tek bir sıçrama, hedefin
       0.025 nat altında biten bir koşuyu −150 puana, dolayısıyla bütün sonucu "başarısız"a
       çeviriyordu. B14'ün kendisi uygulanıyor artık.

    Eğrinin SON noktası hedefin üstündeyse (hedefe ulaşılmadı ya da ulaşılıp kaybedildi) None döner.
    Çağıran taraf None'ı sansür olarak işlemek ZORUNDA (bkz. `decide`'ın censored_mask'i); sessizce
    atlamak seed'i kaybettirir ve eşleştirmeyi bozar.
    """
    pts = sorted(curve)
    if not pts or pts[-1][1] > target:
        return None
    above = [j for j, (_, y) in enumerate(pts) if y > target]
    if not above:
        return float(pts[0][0])                  # eğri baştan beri hedefin altında
    (a, ya), (b, yb) = pts[above[-1]], pts[above[-1] + 1]
    return a + (ya - target) / (ya - yb) * (b - a)


def savings(f_foton: list[float], f_base: list[float],
            seeds_f: list, seeds_b: list) -> dict:
    """Birincil istatistik: `s_i = 100 × (1 − F_foton,i / F_base,i)`, **seed KİMLİĞİNE göre**.

    Neden ayrı bir fonksiyon: `paired()` FARK hesaplar, `s_i` ise ORAN. Denetim, `s_i`'yi kuran tek
    kodun eşleştirmeyi liste KONUMUNA göre yaptığını buldu — yani `paired()`'ın regresyona bağladığı
    hatanın aynısı, birincil istatistiğin tam üstünde. Bir kol yeniden koşulup listeye sonda
    eklendiğinde (koşu defteri bunu yapar) konum eşlemesi FOTON-s22'yi baseline-s33 ile eşliyordu.
    Burada seed listeleri ZORUNLU ve eksik/fazla seed sessizce atlanmaz: hata verir.

    Ayrıca payda seed BAŞINA (`F_base,i`) — tek bir dondurulmuş skaler DEĞİL. Skaler payda
    kullanmak karşılaştırmayı eşleştirilmiş olmaktan çıkarır (baseline gürültüsü paydan silinir,
    GA yapay olarak daralır); ön-kayıt eşleştirilmiş GA'ya dayandığı için bu kararı geçersiz kılar.
    """
    if sorted(seeds_f) != sorted(seeds_b):
        raise ValueError(f"seed kümeleri aynı olmalı: FOTON {sorted(seeds_f)} vs "
                         f"baseline {sorted(seeds_b)}")
    if len(set(seeds_f)) != len(seeds_f):
        raise ValueError(f"FOTON seed listesinde tekrar var: {seeds_f}")
    if not (len(f_foton) == len(seeds_f) and len(f_base) == len(seeds_b)):
        raise ValueError("FLOP listeleri seed listeleriyle aynı uzunlukta olmalı")
    pos = {s: i for i, s in enumerate(seeds_b)}
    s_i = []
    for ff, s in zip(f_foton, seeds_f):          # FOTON'un sırası korunur; baseline KİMLİKLE bulunur
        fb = f_base[pos[s]]
        if fb <= 0:
            raise ValueError(f"seed {s}: F_base = {fb}, pozitif olmalı")
        s_i.append(100.0 * (1.0 - ff / fb))
    return {"s_i": s_i, "seeds": list(seeds_f), "n": len(s_i),
            "mean": st.mean(s_i), "sd": st.stdev(s_i) if len(s_i) > 1 else float("nan")}


def ci95(values: list[float]) -> tuple[float, float, float]:
    """(ortalama, GA alt, GA üst) — t(n−1) ile. Karar kuralının tek GA kaynağı."""
    n = len(values)
    if n < 2:
        raise ValueError("GA için en az 2 gözlem gerekir")
    m, sd = st.mean(values), st.stdev(values)
    h = t_crit(n - 1) * sd / math.sqrt(n)
    return m, m - h, m + h


def indeterminate(mean: float, lo: float, hi: float) -> bool:
    """Güven aralığı BAŞARI EŞİĞİNİ çözemiyor mu? (mutlak sd tripwire'ının yerini alan ölçüt)

    Karar belirlidir ancak şu ikisinden biri sağlanırsa:
      * `lo > 0` **ve** `mean >= SUCCESS`  -> başarı KURULDU
      * `hi < SUCCESS`                     -> başarı DIŞLANDI

    İkisi de sağlanmıyorsa aralık, "eşiği geçiyor" ile "geçmiyor" arasında ayrım yapamıyor demektir;
    sonuç "yüksek varyans / belirsiz"dir ve seed artırılır. Bu ölçütün mutlak sd eşiğine üstünlüğü:
    GA yarı-genişliği n ile küçüldüğü için **çare ölçütü gevşetir**, sıkılaştırmaz (bkz. SD_TRIPWIRE
    yorumundaki ölçüm ve `reachability`).
    """
    if lo > CI_EPS and mean >= SUCCESS:
        return False
    return hi >= SUCCESS


S_MAX = 100.0           # puan; tasarrufun aritmetik tavanı (F_foton = 0)


def ub95(savings: list[float], censored_mask: list[bool] | None = None) -> float:
    """`s̄`'nin SANSÜRE DUYARLI %95 üst güven sınırı — vazgeçme kuralının baktığı tek sayı.

        ÜS = ( n_u × (sansürsüz seed'lerin %95 GA üst sınırı) + Σ sansürlü ikameler ) / n

    Neden düz t-aralığı değil: sansürlü bir seed'in ikamesi (~−150 puan) bir GÖZLEM değil, tek
    yanlı bir SINIRDIR — gerçek değer ondan da küçüktür (koşu 2x bütçede bile hedefe ulaşmadı).
    Onu gözlem gibi t-aralığına sokmak sd'yi şişirir: `[0.2, −0.1, 0.3, 0.0, −149.82]` için düz
    aralık [−113, +53] çıkar ve "GA üst < +5" şartı sansürlü HER senaryoda sağlanamaz olur
    (ikinci denetimin bulgusu: kural ulaşılamazdı). Doğru muamele: sansürlü seed'ler yukarı yönde
    belirsizlik TAŞIMAZ, belirsizlik yalnızca sansürsüz seed'lerden gelir.

    Üçüncü denetimin bulduğu delik de burada kapanıyor: bir önceki hâl yalnızca sansürsüz ALT
    örneğin aralığına bakıyordu; 3/5 sansürde 2 seed kalıyor, t(1) = 12.7 aralığı patlatıyor ve
    EN NET başarısızlık (5 seed'in 3'ü hedefe ulaşamamış) terk edilemiyordu — kural sansür
    sayısında monoton bile değildi (k=3 < k=2 < k=4). Bu sınır k'da monotondur: her sansürlü seed
    onu ~30 puan aşağı çeker. Sansür yokken düz %95 GA üst sınırına eşittir, yani bağlayıcı
    metindeki "`s̄`'nin %95 GA'sının tamamı +%5'in altında" ifadesinin aynısıdır.

    Sansürsüz seed sayısı 2'den azsa aralık kurulamaz; o zaman aritmetik tavan (seed başına en çok
    +100 puan) kullanılır, ki 5 seed'in 4'ü sansürlüyken bile (100 − 600)/5 = −100'dür.
    """
    n = len(savings)
    mask = list(censored_mask) if censored_mask is not None else [False] * n
    if len(mask) != n:
        raise ValueError("censored_mask, savings ile aynı uzunlukta olmalı")
    unc = [s for s, c in zip(savings, mask) if not c]
    cens = [s for s, c in zip(savings, mask) if c]
    arith = (S_MAX * len(unc) + sum(cens)) / n
    if len(unc) >= 2:
        return min(arith, (len(unc) * ci95(unc)[2] + sum(cens)) / n)
    return arith


def abandon(scale_results: list[dict], oracle_gain_over_null, oracle_threshold: float = 2.0,
            escalation_used: bool = False) -> dict:
    """VAZGEÇME KOŞULU — ön-kayıttaki "Vazgeçmenin İKİ yolu" tablosunun birebir uygulaması.

    scale_results : ölçek başına BAĞLAYICI `decide()` çıktısı (strict=True, n ∈ {5, 10}), ölçek sırasıyla
    oracle_gain_over_null : ölçek BAŞINA oracle kazancının null'ın üstündeki payı (yüzde puan).
        Tek ölçekte sayı, iki ölçekte ölçek sırasıyla dizi. İki ölçekte tek sayı KABUL EDİLMEZ:
        S0 = 1.5 / S1 = 2.5 iken sonuç hangi sayının verildiğine bağlı olurdu.
    escalation_used : ölçek yükseltme hakkı kullanıldı mı (yalnızca tek ölçek yolunu etkiler)

    İKİ ÖLÇEK yolu (elde >= 2 ölçek sonucu varsa DAİMA bu): son iki ölçeğin ikisinde de
      `ub95 < +5`, hiçbiri "yüksek varyans / belirsiz" ya da "yetersiz kesinlik" değil, ve oracle
      kazancı her iki ölçekte < eşik.
    TEK ÖLÇEK yolu (v01 EK1-B10; yalnızca elde TEK ölçek sonucu varken): satır "başarısız",
      yükseltme hakkı KULLANILMAMIŞ, `ub95 < +5`, oracle kazancı < eşik.

    Üçüncü denetimin bulduğu regresyon: önceki hâl tek ölçek çıkışını elde İKİ ölçek varken de
    deniyordu — `[S0 "başarılı", S1 "başarısız"]` hipotezi terk ettiriyordu, oysa iki ölçek kuralı
    (S0'ın GA'sı [6.5, 7.5], tamamen +5'in ÜSTÜNDE) açıkça "terk yok" diyor.

    Eski biçim (`s̄ <= 0` VE `GA üst < +2`) neden atıldı: bir nokta tahminini bir aralık sınırıyla
    "ve" ile bağlıyordu; GA yarı-genişliği ~1.167σ olduğundan σ >~ 1.72'de tam başabaş bir FOTON
    asla terk edilemiyordu (v01 B10'da bir kez düzeltilen hatanın ikinci ortaya çıkışı; bkz. V12).
    """
    if not scale_results:
        return {"abandon": False, "reason": "ölçek sonucu yok", "path": None}
    for r in scale_results:
        if "ub95" not in r or "row" not in r:
            raise ValueError("abandon(): girdi `decide()` çıktısı olmalı (`ub95` / `row` yok)")
        if not r.get("strict", False) or r.get("n") not in REQUIRED_N:
            raise ValueError(f"abandon(): bağlayıcı olmayan sonuç (strict={r.get('strict')}, "
                             f"n={r.get('n')}) vazgeçme kararını süremez")
    two = len(scale_results) >= 2
    used = scale_results[-2:] if two else scale_results
    if isinstance(oracle_gain_over_null, (int, float)):
        if two:
            raise ValueError("iki ölçekte oracle kazancı ölçek BAŞINA verilmeli (dizi)")
        gains = [float(oracle_gain_over_null)]
    else:
        gains = [float(g) for g in oracle_gain_over_null][-len(used):]
    if len(gains) != len(used) or any(g != g or g in (float("inf"), float("-inf")) for g in gains):
        raise ValueError(f"oracle kazancı her ölçek için sonlu bir sayı olmalı: {gains}")
    oracle_ok = all(g < oracle_threshold for g in gains)
    ubs = [r["ub95"] for r in used]
    base = {"ub95": ubs, "oracle_ok": oracle_ok, "single_scale": not two,
            "path": "iki ölçek" if two else "tek ölçek"}
    why = []
    if two:
        if any(r["row"] in ("yüksek varyans / belirsiz", "yetersiz kesinlik") for r in used):
            return {**base, "abandon": False,
                    "reason": "yetersiz kesinlik: bir ölçekte GA başarı eşiğini çözemedi; bu "
                              "'hipotez ayakta' DEĞİL, 'ölçemedik' demektir"}
    else:
        r = used[0]
        if r["row"] != "başarısız":
            why.append(f"tek ölçek yolu yalnızca 'başarısız' satırından açılır (satır: '{r['row']}'); "
                       "iki ardışık ölçek sonucu gerekiyor")
        if escalation_used:
            why.append("yükseltme hakkı KULLANILDI: v01 B10 tek ölçekle terk etmeyi yalnızca hak "
                       "kullanılmadığında veriyor; iki ardışık ölçek sonucu gerekiyor")
    if not all(u < SUCCESS for u in ubs):
        why.append(f"bir ölçekte s̄'nin GA üst sınırı başarı eşiğinin üstünde "
                   f"({', '.join(f'{u:+.2f}' for u in ubs)}): etki eşik kadar büyük olabilir")
    if not oracle_ok:
        why.append(f"oracle kazancı null'ın {', '.join(f'{g:.2f}' for g in gains)} puan üstünde, "
                   f">= {oracle_threshold}: mükemmel yönlendirmede hâlâ kazanç var")
    if why:
        return {**base, "abandon": False, "reason": "; ".join(why)}
    return {**base, "abandon": True,
            "reason": f"{base['path']} yolu: s̄'nin GA üst sınırı < {SUCCESS} puan "
                      f"({', '.join(f'{u:+.2f}' for u in ubs)}) ve oracle kazancı null'ın "
                      f"{', '.join(f'{g:.2f}' for g in gains)} puan üstünde < {oracle_threshold}"
                      + ("" if two else "; satır 'başarısız' ve yükseltme hakkı kullanılmadı (v01 EK1-B10)")}


def t_crit_alpha(df: int, alpha: float) -> float:
    """İki yanlı `alpha` için t kritik değeri — Holm'un düzeltilmiş düzeylerinde GA kurmak için.

    `t_crit()` yalnızca α = .05'i biliyor. Birincil-2'nin MDE'si Holm'un İLK adımında
    (aile büyüklüğü 2 -> α/2 = .025) tanımlı ve o düzeyde t(4) = 3.4954, α=.05'in 2.7764'ü değil.
    İlk hâl MDE'yi .05'te türetiyordu ve +1.5 puandaki gücü %56.2 gösteriyordu; gerçek değer
    %38.8, MDE ise 1.49 değil **1.88** puan.
    """
    lo, hi = 1e-6, 100.0
    for _ in range(200):                     # ikiye bölme: t_sf monoton azalan
        mid = (lo + hi) / 2
        if t_sf(mid, df) * 2 > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# Birincil-2 (X↔H) için MDE ve aile büyüklüğü — ön-kayıtta pinli.
PRIMARY_FAMILY = 2       # birincil-1 (X↔M) + birincil-2 (X↔H); Holm bu büyüklükte uygulanır
MDE_PRIMARY2 = 1.88      # puan; n=5, sd(fark) 1.20, Holm ilk adımı α=.025 (bkz. ön-kayıt)


def contribution(diffs: list[float], mde: float = MDE_PRIMARY2,
                 alpha: float = 0.05 / PRIMARY_FAMILY, strict: bool = True) -> dict:
    """Birincil-2'nin DÖRT durumlu puanlaması. Karar tablosu birincil-2'ye UYGULANMAZ.

    diffs : seed başına `s_i(X) − s_i(H)` (yüzde puan); SANSÜRSÜZ seed'ler (bkz. `decide_primary`)

    Neden ayrı: karar tablosunun eşikleri (+%2/+%5/+%10) TASARRUF için kalibre edildi ve
    birincil-2 bir FARK kontrastıdır. `decide()`'a verilseydi +1.88 puanlık gerçek bir
    heterojenlik katkısı "nötr" satırına düşerdi (çünkü 1.88 < 2), yani deneyin asıl sorusunun
    olumlu cevabı "etki yok" diye etiketlenirdi.

    Dört durum BİRBİRİNİ DIŞLAR ve TÜKETİR. İlk hâl üç durum tanımlıyordu ve "katkı var"
    (`lo > 0`) ile "katkı yok" (`hi < MDE`) birbirini dışlamıyordu — ikisi birden doğru olabilir
    ve o hâl mümkün olan EN bilgilendirici sonuçtur: katkı gerçek ama MDE'nin altında.

    Aralık Holm'un İLK adımının düzeyindedir (α = .025, yani %97.5 GA) ve MDE çıtası n'den
    bağımsız SABİTTİR (1.88 puan); ikisi de ön-kayıtta pinli.
    """
    n = len(diffs)
    if n < 2:
        raise ValueError("kontrast için en az 2 seed gerekir")
    if strict and n not in REQUIRED_N:
        # `decide()`'daki aynı boşluğun kopyası: bağlayıcı giriş 3 seed'den "katkı var, büyük"
        # damgası basabiliyordu.
        raise ValueError(f"bağlayıcı birincil-2 için n {REQUIRED_N} olmalı, verilen n={n}")
    m, sd = st.mean(diffs), st.stdev(diffs)
    h = t_crit_alpha(n - 1, alpha) * sd / math.sqrt(n)
    lo, hi = m - h, m + h
    real, small = lo > CI_EPS, hi < mde        # CI_EPS: `decide()` ile aynı tolerans
    state = ("katkı var, büyük" if real and not small else
             "katkı var, küçük" if real and small else
             "katkı MDE kadar değil" if small else
             "ölçemedik")
    note = ""
    if state == "ölçemedik":
        note = "yeterince güçlü ölçemedik — 'heterojenlik çürütüldü' DEĞİL"
    elif hi < -CI_EPS:
        # Dört durum katkının İŞARETİNİ ayırmıyor; anlamlı NEGATİF bir katkı "MDE kadar değil"e
        # düşer ve orada sessiz kalırdı. Durum adı değişmez (tablo pinli), not düşülür.
        note = "katkı NEGATİF: aralığın tamamı sıfırın altında — heterojenlik H'ye göre ZARAR veriyor"
    return {"state": state, "mean": m, "sd": sd, "n": n, "ci": [lo, hi],
            "alpha": alpha, "mde": mde, "note": note}


def _p_two_sided(values: list[float]) -> float:
    """Ortalamanın sıfırdan farkı için iki yanlı t-testi p değeri (tek örneklem)."""
    n = len(values)
    m, sd = st.mean(values), st.stdev(values)
    if sd == 0:                                  # dejenere: bütün seed'ler aynı
        return 0.0 if m != 0 else 1.0
    return t_sf(abs(m / (sd / math.sqrt(n))), n - 1) * 2


def decide_primary(s_xm: list[float], s_hm: list[float],
                   quality_per_sec: float | dict[str, float] | None = None,
                   censored_mask: list[bool] | None = None,
                   censored_mask_h: list[bool] | None = None,
                   strict: bool = True, alpha: float = 0.05,
                   seed_right_available: bool = True) -> dict:
    """İKİ birincil uç noktayı BİRLİKTE karara bağlar: Holm (m = 2) satır adının ÖNÜNDEDİR.

    s_xm : seed başına `s_i(X↔M)` — birincil-1'in istatistiği
    s_hm : seed başına `s_i(H↔M)`, AYNI seed sırasıyla — birincil-2 `s_xm − s_hm` farkıdır
    censored_mask / censored_mask_h : X ve H kollarının seed başına sansür bayrakları

    Neden farkı çağıran değil BU fonksiyon kuruyor: iki liste ayrı ayrı verilirse iki endpoint'in
    farklı n'de (ör. birincil-1 n=5, birincil-2 n=10) ya da sansürlü bir seed DÜŞÜRÜLEREK
    hesaplanması mümkün oluyordu — ön-kaydın "sonuca göre seçilebilir" diye adını koyduğu seçim.
    Ölçülü örnek: H'nin bir seed'i sansürlüyken tam fark listesi "ölçemedik" veriyor, o seed
    düşürülünce "katkı var, büyük". Artık uzunluklar eşit olmak ZORUNDA ve sansür burada işleniyor.

    SANSÜR KURALI (birincil-2): X ya da H'nin herhangi bir seed'i sansürlüyse fark o seed'de
    tanımsızdır (ikame bir gözlem değil tek yanlı bir sınırdır; farkı ±150 puan oynatır). O zaman
    birincil-2 "ölçemedik" olarak raporlanır, p₂ = 1 alınır ve Holm'da birincil-1 α/2 düzeyiyle
    sınanır (muhafazakâr yön). Seed DÜŞÜRÜLMEZ.

    Sıra (ön-kayıttaki "Holm iki endpoint'e NASIL uygulanır" bölümünün birebir uygulaması):
      1. iki endpoint'in iki yanlı p değerleri,
      2. Holm, aile büyüklüğü PİNLİ (`PRIMARY_FAMILY`),
      3. birincil-1 satırı `decide()`'dan; Holm'dan geçmediyse başarı satırları "belirsiz"e düşer,
      4. birincil-2 durumu `contribution()`'dan. "Katkı var" durumları α/2 GA'sının alt sınırı > 0
         şartına bağlı, bu da `p₂ < α/2` demek — Holm'un HER adımından geçer; tutarlılık yine de
         denetlenir ve ihlali sessiz geçilmez.
    """
    if len(s_xm) != len(s_hm):
        raise ValueError(f"iki endpoint AYNI seed kümesinde hesaplanır: n(X↔M) = {len(s_xm)}, "
                         f"n(H↔M) = {len(s_hm)}")
    n = len(s_xm)
    mx = list(censored_mask) if censored_mask is not None else [False] * n
    mh = list(censored_mask_h) if censored_mask_h is not None else [False] * n
    if len(mx) != n or len(mh) != n:
        raise ValueError("sansür maskeleri seed listeleriyle aynı uzunlukta olmalı")
    r1 = decide(s_xm, quality_per_sec=quality_per_sec, censored_mask=mx, strict=strict,
                seed_right_available=seed_right_available)
    p1 = _p_two_sided(s_xm)
    if any(mx) or any(mh):
        k = sum(1 for a, b in zip(mx, mh) if a or b)
        c2 = {"state": "ölçemedik", "mean": None, "sd": None, "n": n, "ci": None,
              "alpha": alpha / PRIMARY_FAMILY, "mde": MDE_PRIMARY2,
              "note": f"{k} seed'de X ya da H sansürlü: fark tanımsız, seed DÜŞÜRÜLMEZ; birincil-2 "
                      "'ölçemedik' — 'heterojenlik çürütüldü' DEĞİL"}
        p2 = 1.0
    else:
        diffs = [a - b for a, b in zip(s_xm, s_hm)]
        c2 = contribution(diffs, alpha=alpha / PRIMARY_FAMILY, strict=strict)
        p2 = _p_two_sided(diffs)
    holm = multiplicity([("birincil-1", p1), ("birincil-2", p2)], alpha=alpha,
                        family_size=PRIMARY_FAMILY)
    passed = set(holm["holm"])
    row1, notes1 = r1["row"], list(r1["notes"])
    if row1 in ("güçlü başarı", "başarılı") and "birincil-1" not in passed:
        notes1.append(f"Holm düzeltmesinden GEÇMEDİ (p₁ = {p1:.4f}, p₂ = {p2:.4f}, aile "
                      f"büyüklüğü {PRIMARY_FAMILY}): satır '{row1}' -> 'belirsiz'")
        row1 = "belirsiz"
    if c2["state"].startswith("katkı var") and "birincil-2" not in passed:
        # Matematiksel olarak olamaz (`lo > 0` α/2 düzeyinde ⟹ p₂ < α/2 ⟹ Holm'un her adımı geçer);
        # yalnızca kayan nokta bıçak sırtında görülebilir. Bağlayıcı yolda istisna atmak yerine
        # MUHAFAZAKÂR yöne düşürülür ve kayda geçer.
        c2 = {**c2, "state": "katkı MDE kadar değil" if c2["ci"][1] < c2["mde"] else "ölçemedik",
              "note": f"Holm sınırında (p₂ = {p2:.6f}): 'katkı var' ilan EDİLMEDİ"}
    return {"primary1": {**r1, "row": row1, "notes": notes1, "p": p1,
                         "holm_passed": "birincil-1" in passed},
            "primary2": {**c2, "p": p2, "holm_passed": "birincil-2" in passed},
            "holm": holm}


def reachability(fires, mu: float, sigma: float, n: int, sims: int = 100_000,
                 seed: int = 0) -> float:
    """`fires(örnek) -> bool` kuralının, gerçek etki `mu` ve gürültü `sigma` iken ateşleme oranı.

    ÖN-KAYIT V12'nin yöntem şartı: yazılan her durdurma/tripwire kuralı, makul bir σ aralığında
    sayısal olarak sınanır. Gerekçe: aynı hata iki kez yapıldı (v01 B10, sonra v02'nin vazgeçme
    koşulu). Örüntü belli — nokta tahmini ile aralık sınırını "ve" ile bağlayan kurallar varyans
    büyüdükçe ölüyor, ve bu gözle görülmüyor, ancak hesaplanınca görülüyor.

    `numpy` varsa onu, yoksa `random`'ı kullanır (bağımlılık zorunlu olmasın).
    """
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        x = rng.normal(mu, sigma, size=(sims, n))
        return float(sum(1 for row in x if fires(row.tolist())) / sims)
    except ImportError:
        import random
        rnd = random.Random(seed)
        hit = sum(1 for _ in range(sims)
                  if fires([rnd.gauss(mu, sigma) for _ in range(n)]))
        return hit / sims


def multiplicity(tests: list[tuple[str, float]], alpha: float = 0.05,
                 family_size: int | None = None) -> dict:
    """Holm (FWER) ve Benjamini-Hochberg (FDR): hangi nominal anlamlı testler düzeltmeden geçiyor.

    family_size : düzeltmede kullanılacak aile büyüklüğü `m`. **Ön-kayıtta sayılan test sayısı
        buraya verilmek zorunda.** Verilmezse `m` = p değeri ÜRETEBİLEN test sayısı olur ve bu bir
        kapıdır: bazı teşhis ölçütleri (entropi, yük CV, ölü modül sayısı) tek sayıdır, null'ı yoktur
        ve p üretmez; onlar havuzdan düşünce `m` küçülür ve Holm GEVŞER. Ölçülü örnek: p(D2) = 0.02
        iken m=10'da Holm D2'yi reddeder, m=2'de KABUL eder — yani "hangi ölçütler hesaplanabildi"
        sorusunun cevabı, anlamlılığı sonradan değiştirebiliyordu. Aile büyüklüğü önceden pinlenir.
    """
    rows = sorted(((n, p) for n, p in tests if p == p), key=lambda r: r[1])
    m_eff = len(rows)
    m = family_size if family_size is not None else m_eff
    if m < m_eff:
        raise ValueError(f"family_size={m}, p üreten test sayısından ({m_eff}) küçük olamaz")
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
    return {"n_tests": m, "n_with_p": m_eff, "alpha": alpha,
            "nominal": [n for n, pv in rows if pv <= alpha],
            "holm": sorted(holm), "bh": sorted(bh)}


# Karar tablosu satır adları — rapor bunlardan birini BİREBİR yazmak zorunda.
#
# "yetersiz kesinlik" TERMİNAL satırdır ve n = max(REQUIRED_N)'de kesinlik ölçütü yine tetiklenirse
# döner. Ayrı bir ad olması şart: B3 ikinci bir seed artırma hakkı vermiyor, dolayısıyla o noktada
# "yüksek varyans / belirsiz" döndürmek raporu OLMAYAN bir çareye ("seed 10'a çıkar") yönlendirirdi.
# "nötr"den ayrımı da kritik: nötr, etkinin sıfıra yakın olduğunu ÖLÇTÜĞÜMÜZ durumdur; yetersiz
# kesinlik, ölçemediğimiz durumdur.
ROWS = ("güçlü başarı", "başarılı", "belirsiz", "nötr", "başarısız",
        "yüksek varyans / belirsiz", "yetersiz kesinlik")


def _wbar(quality_per_sec) -> tuple[dict[str, float] | None, float | None]:
    """`quality_per_sec`'i normalize eder: (baseline -> w̄ eşlemesi, en küçük w̄).

    Neden eşleme: ön-kayıt "güçlü başarı"yı **`w̄(M) > 0` VE `w̄(D) > 0`** şartına bağlıyor
    (gerekçe: M bu kartta kendi başına D'den token başına %27.9 yavaştır, dolayısıyla yalnızca
    M'e karşı `w̄ > 0` FOTON'un hız katkısı olmadan da sağlanabilir — v01 EK1-B16'nın aynı kolda
    bulduğu sorun). Tek bir skaler bu şartı İFADE EDEMEZ: ölçüldü, `decide(s̄=11, w=+1.0)`
    "güçlü başarı" döndürüyordu ve bu, ön-kaydın "başarılı" dediği duruma denk geliyordu.

    Skaler geçmek hâlâ mümkün (tarama, tek baseline'lı keşifsel kullanım) ama o zaman
    "güçlü başarı" satırı **erişilemez** ve notlarda gerekçesi yazılır.
    """
    if quality_per_sec is None:
        return None, None
    if isinstance(quality_per_sec, dict):
        if not quality_per_sec:
            raise ValueError("quality_per_sec boş eşleme olamaz")
        extra = set(quality_per_sec) - set(WBAR_KEYS)
        if extra:
            # Alakasız bir anahtar `min()` üzerinden meşru bir "güçlü başarı"yı düşürüyordu:
            # {"M":+1,"D":+1,"Z":−1} -> "başarılı". Anahtar kümesi kapalı.
            raise ValueError(f"quality_per_sec anahtarları {WBAR_KEYS} olmalı, fazlası: {sorted(extra)}")
        wmap = {k: float(v) for k, v in quality_per_sec.items()}
    else:
        wmap = {"?": float(quality_per_sec)}
    # NaN/inf REDDEDİLİR: `min()` NaN'la sıra-bağımlıdır ({"M": nan, "D": 1} -> nan, {"D": 1,
    # "M": nan} -> 1) ve `nan > 0` False olsa da ikinci sırada kapıyı GEÇİYORDU.
    bad = {k: v for k, v in wmap.items() if v != v or v in (float("inf"), float("-inf"))}
    if bad:
        raise ValueError(f"w̄ sonlu bir sayı olmalı: {bad} (ölçülemediyse anahtarı hiç verme)")
    return wmap, min(wmap.values())


def decide(savings: list[float], quality_per_sec: float | dict[str, float] | None = None,
           censored_mask: list[bool] | None = None, strict: bool = True,
           seed_right_available: bool = True) -> dict:
    """PREREGISTRATION_V02 karar tablosunu (öncelik tablosu + v01 EK 1 B8/B16) birebir uygular.

    **Bağlayıcı karar bu fonksiyonla TEK BAŞINA verilmez**: iki birincil uç nokta Holm'la birlikte
    `decide_primary()`'de karara bağlanır; `decide()` onun birincil-1 adımıdır.

    savings        : seed başına tasarruf, YÜZDE PUAN (`s_i`, bkz. `savings()`); sansürlü seed'ler
                     için 2x bütçe ikamesiyle hesaplanmış değer
    quality_per_sec: w̄ (yüzde puan), baseline başına eşleme `{"M": .., "D": ..}`. **None geçmek
                     sessiz değildir**: "güçlü başarı" erişilemez ve bu bir NOT olarak kayda geçer.
    censored_mask  : **YALNIZCA FOTON kolunun** seed başına sansür bayrağı (`n_star` None döndü).
                     Baseline'ın sansürü buraya GİRMEZ — v02'de dense kolu `L*`'e ulaşmıyor ve
                     maskeye onun beş sansürü konursa FOTON, baseline'ın yetersizliği yüzünden
                     "başarısız" ilan edilir. EK 1 B8:
                     - >= 1 sansürlü seed: "başarılı"/"güçlü başarı" YASAK
                     - >= 2 sansürlü seed: sonuç doğrudan "başarısız"
    strict         : True ise `n` ön-kayıtlı seed sayılarından biri olmak ZORUNDA (`REQUIRED_N`).
                     False yalnızca keşifsel kullanım içindir; başarı satırlarını BLOKLAR ve sonuç
                     `abandon()`'a verilemez.
    seed_right_available : tek seed artırma hakkı (EK1-B3, kampanya başına BİR kez) hâlâ duruyor mu.
                     False iken kesinlik ölçütü tetiklenirse satır doğrudan "yetersiz kesinlik"tir:
                     hak S0'da harcandıysa S1'de "seed 10'a çıkar" demek olmayan bir çaredir.

    Döner: {"row", "mean", "sd", "sd_uncensored", "ci95", "ci95_uncensored", "ub95", "n",
            "n_censored", "notes", "determinate", "strict", "w_bar"}
    """
    n = len(savings)
    if n < 2:
        raise ValueError("karar için en az 2 seed gerekir")
    # Kilit cümle BEŞ eşleştirilmiş seed istiyor, `decide()` ise iki seed'le "başarılı" damgası
    # basıyordu — ve `paired()` eşleşmeyen seed'leri SESSİZCE düşürdüğü için kısa seed kümesi
    # işleyişte ulaşılabilir bir durumdu. Artık bağlayıcı karar yalnızca 5 ya da 10 seed'le verilir.
    if strict and n not in REQUIRED_N:
        raise ValueError(f"bağlayıcı karar için n {REQUIRED_N} olmalı, verilen n={n}. "
                         f"Keşifsel kullanım için strict=False (başarı satırları bloklanır).")
    mask = list(censored_mask) if censored_mask is not None else [False] * n
    if len(mask) != n:
        raise ValueError("censored_mask, savings ile aynı uzunlukta olmalı")
    n_cens = sum(mask)
    wmap, wmin = _wbar(quality_per_sec)          # doğrulama HER dönüş yolundan önce
    mean, lo, hi = ci95(savings)
    sd = st.stdev(savings)
    unc = [s for s, c in zip(savings, mask) if not c]
    sd_unc = st.stdev(unc) if len(unc) > 1 else float("nan")
    res = {"mean": mean, "sd": sd, "sd_uncensored": sd_unc, "n": n, "n_censored": n_cens,
           "ci95": [lo, hi], "ub95": ub95(savings, mask), "strict": strict, "w_bar": wmap}
    if len(unc) >= 2:
        mu_, lo_, hi_ = ci95(unc)
        res["ci95_uncensored"], res["mean_uncensored"] = [lo_, hi_], mu_
    w_complete = wmap is not None and set(WBAR_KEYS) <= set(wmap)
    notes: list[str] = []

    def finish(row: str, determinate: bool) -> dict:
        """HER dönüş yolu buradan geçer: zorunlu notlar erken dönüşlerde ATLANMASIN.

        Üçüncü denetim buldu: sansür (>= 2) ve kesinlik dönüşleri w̄ notlarını ve B8(i)'in
        raporda zorunlu "ikame FOTON'un LEHİNE yanlıdır" cümlesini atlıyordu.
        """
        extra: list[str] = []
        if wmin is None:
            extra.append("w̄ (quality/sec) VERİLMEDİ: ikincil ölçüt değerlendirilmedi, bu yüzden "
                         "'güçlü başarı' satırı erişilemez ve çelişki kuralı uygulanamadı")
        else:
            # İKİ not BAĞIMSIZ: "w̄ eksik ölçüldü" ile "w̄ negatif" aynı anda doğru olabilir.
            if not w_complete and mean >= STRONG and lo > CI_EPS:
                missing = [k for k in WBAR_KEYS if k not in wmap]
                how = ("tek skaler olarak verildi: hangi baseline'a karşı olduğu BİLİNMİYOR"
                       if "?" in wmap else f"yalnızca {sorted(wmap)} için verildi")
                extra.append(f"w̄ {how}; eksik: {missing or list(WBAR_KEYS)}. Ön-kayıt 'güçlü başarı' "
                             f"için w̄({WBAR_KEYS[0]}) > 0 VE w̄({WBAR_KEYS[1]}) > 0 istiyor, bu yüzden "
                             "o satır ERİŞİLEMEZ. Gerekçe: yalnızca M'e karşı hız kazancı, M bu kartta "
                             "kendi başına D'den %27.9 yavaş olduğu için FOTON'un katkısı OLMADAN da "
                             "sağlanabilir (v01 EK1-B16).")
            # EK 1 B16: çelişki etiketi GÜRÜLTÜYE vurulmaz — FLOP kazancının GA'sı sıfırın üstünde olmalı.
            if wmin <= 0:
                neg = sorted(k for k, v in wmap.items() if v <= 0)
                which = f" (negatif olan: {', '.join(neg)})" if neg != ["?"] else ""
                if lo > CI_EPS:
                    extra.append("FLOP kazancı var (GA tamamen > 0), duvar saati kaybı var -> "
                                 f"çelişki kuralı: mimari başarı / sistem başarısızlığı{which}")
                elif mean > 0:
                    extra.append(f"s̄ pozitif ({mean:.2f}) ama GA sıfırı kesiyor ve w̄ <= 0: çelişki "
                                 "kuralı UYGULANMAZ (FLOP kazancı istatistiksel olarak kurulmadı)")
        if n_cens >= 1:
            extra.append(f"{n_cens} FOTON seed'i L*'e ulaşmadı; s̄ 2x bütçe ikamesiyle hesaplandı ve "
                         "bu ikame FOTON'un LEHİNE yanlıdır (EK 1 B8)")
        return {**res, "notes": notes + extra, "row": row, "determinate": determinate}

    if n_cens >= 2:                                        # EK 1 B8: iki veya daha fazla sansür -> başarısız
        notes.append(f"{n_cens} FOTON seed'i L*'e ulaşmadı: EK 1 B8 gereği sonuç doğrudan 'başarısız'")
        return finish("başarısız", True)

    # ---- kesinlik ölçütü (mutlak sd tripwire'ının yerini aldı; v01 B8(ii) gereği SANSÜRSÜZ
    #      seed'ler üzerinde: sansürlü seed "hedefe ulaşamadı"dır, "varyans yüksek" değil) ----
    if len(unc) >= 2:
        m_u, lo_u, hi_u = ci95(unc)
        indet = indeterminate(m_u, lo_u, hi_u)
        detail = f"sansürsüz {len(unc)} seed: s̄ {m_u:.2f}, GA [{lo_u:.2f}, {hi_u:.2f}]"
    else:
        indet, detail = True, f"yalnızca {len(unc)} sansürsüz seed: kesinlik DEĞERLENDİRİLEMEZ"
    if indet:
        if n >= max(REQUIRED_N) or not seed_right_available:
            # TERMİNAL: EK1-B3 tek bir seed artırma hakkı veriyor (kampanya başına). Hak yoksa
            # "seed 10'a çıkar" demek olmayan bir çareye yönlendirmek olurdu.
            neden = (f"n={n} zaten azami seed sayısı" if n >= max(REQUIRED_N)
                     else "seed artırma hakkı bu kampanyada zaten harcandı")
            notes.append(f"GA başarı eşiğini ({SUCCESS} puan) çözemiyor — {detail}; {neden} "
                         "(EK 1 B3 ikinci artırma hakkı vermiyor) ve ölçek yükseltmesi kesinlik "
                         "sorununun çaresi değil: etki, başarı eşiğinin bu deneyin kesinliğiyle ayırt "
                         "edilemeyeceği kadar yakınında. Ne kurulmuş ne dışlanmıştır; bu 'nötr' DEĞİLDİR.")
            return finish("yetersiz kesinlik", False)
        notes.append(f"GA başarı eşiğini ({SUCCESS} puan) çözemiyor — {detail}: "
                     "seed 10'a çıkar (EK 1 B3: {66,77,88,99,110})")
        return finish("yüksek varyans / belirsiz", False)

    crosses_zero = lo - CI_EPS <= 0 <= hi + CI_EPS
    if mean >= STRONG and lo > CI_EPS and wmin is not None and wmin > 0 and w_complete:
        row = "güçlü başarı"
    elif mean >= SUCCESS and lo > CI_EPS:
        row = "başarılı"
    elif mean > NEUTRAL and (crosses_zero or mean < SUCCESS):
        row = "belirsiz"
    elif -NEUTRAL <= mean <= NEUTRAL:
        row = "nötr"
    else:                                                  # mean < -2
        row = "başarısız"
        if crosses_zero:
            notes.append("ortalama < -%2 ama GA sıfırı kesiyor (O5: negatif ortalama 'belirsiz' üretmez)")

    if n_cens == 1 and row in ("güçlü başarı", "başarılı"):
        notes.append("EK 1 B8 gereği 1 sansürlü seed varken 'başarılı' ilan edilemez")
        row = "belirsiz"
    if not strict and row in ("güçlü başarı", "başarılı"):
        notes.append(f"strict=False (n={n} ön-kayıtlı değil): başarı satırı bloklandı, keşifsel sonuç")
        row = "belirsiz"
    return finish(row, True)
