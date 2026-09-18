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
    """`N*`: eğrinin `target`'ı ilk kez AŞAĞI kestiği token sayısı. Kural burada PİNLİ.

    curve : (token, val_loss) çiftleri; sıralı olmak zorunda değil, sıralanır.

    İki karar bu fonksiyonda dondurulmuştur, çünkü ikisi de birincil istatistiği oynatıyor ve
    ön-kayıtta serbest bırakılmaları "sonucu gördükten sonra seçme" kapısıydı:

    1. **İnterpolasyon TOKEN ekseninde doğrusal.** ln(token) ekseninde doğrusal yapmak `N*`'ı bir
       eval aralığı içinde ~%0.06 oynatıyor. Küçük ama serbest bir parametre olmamalı.
       (`L*`'in kendisi interpolasyonsuzdur — bütçe %80 noktası tam bir eval noktasına düşecek
       şekilde seçildi; interpolasyon yalnızca FOTON kollarının `N*`'ında devreye girer.)

    2. **Kesişim SÜRDÜRÜLMÜŞ olmalı** (v01 EK1-B14): kesişimden sonraki eval noktalarının HİÇBİRİ
       hedefin üstünde olamaz. Gürültülü bir eğri hedefin altına bir kez sarkıp geri çıkarsa, ilk
       kesişimi okumak sahte bir kazanç üretir — ölçülü örnek: 2750'de sarkıp 3000'de geri çıkan
       bir eğride ilk kesişim `N*`'ı 114.7M yerine 90.1M gösteriyor, yani `s_i`'yi ~21 puan
       şişiriyor ve "nötr" bir sonucu "güçlü başarı" yapıyor. Böyle bir eğride tek bir `N*` anlamlı
       değildir: None döner ve seed SANSÜRLÜ işlenir.

    Hedefe hiç ulaşılmazsa da None döner. Çağıran taraf None'ı sansür olarak işlemek ZORUNDA
    (bkz. `decide`'ın censored_mask'i); sessizce atlamak seed'i kaybettirir ve eşleştirmeyi bozar.
    """
    pts = sorted(curve)
    for j, ((a, ya), (b, yb)) in enumerate(zip(pts, pts[1:])):
        if ya >= target >= yb:
            if any(y > target for _, y in pts[j + 2:]):
                return None                      # B14: kesişim sürdürülmemiş
            if ya == yb:                         # yatay ve tam hedefte: ilk nokta
                return float(a)
            return a + (ya - target) / (ya - yb) * (b - a)
    return None


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
    s_i, order = [], []
    for i, s in enumerate(sorted(seeds_f, key=lambda x: seeds_f.index(x))):
        fb = f_base[pos[s]]
        if fb <= 0:
            raise ValueError(f"seed {s}: F_base = {fb}, pozitif olmalı")
        s_i.append(100.0 * (1.0 - f_foton[seeds_f.index(s)] / fb))
        order.append(s)
    return {"s_i": s_i, "seeds": order, "n": len(s_i),
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


def abandon(scale_results: list[dict], oracle_gain_over_null: float,
            oracle_threshold: float = 2.0, escalation_used: bool = False) -> dict:
    """VAZGEÇME KOŞULU — SAĞLAM biçim. İki ardışık ölçekte başarı eşiği DIŞLANMIŞ olmalı.

    scale_results : ölçek başına `decide()` çıktısı, ölçek sırasına göre
    oracle_gain_over_null : oracle routing kazancının null'ın ÜSTÜNDEKİ payı (yüzde puan)
    escalation_used : ölçek yükseltme hakkı kullanıldı mı (v01 B10'un tek-ölçek yolunun ön-koşulu)

    Neden eski biçim atıldı. v02'nin ilk hâli v01'den şunu devralmıştı: `s̄ <= 0` **ve**
    `GA üst sınırı < +%2`. Bu, bir NOKTA TAHMİNİ ile bir ARALIK SINIRINI "ve" ile bağlıyor ve
    GA yarı-genişliği ~1.167σ olduğu için koşul `s̄ < 2 − 1.167σ` demeye geliyordu. Ölçüldü:

        σ = 0.75 -> s̄ < +1.12 gerekir (ulaşılabilir)
        σ = 1.72 -> s̄ < 0.00     (sınır)
        σ = 3.00 -> s̄ < −1.50   (ULAŞILAMAZ: tam başabaş bir FOTON asla terk edilemez)
        σ = 6.00 -> s̄ < −5.00   (v01 O9 FOTON'un sd'sinin BÜYÜK olmasını bekliyordu)

    Yani kural tam ihtiyaç duyulduğu rejimde ölüyordu ve n=10 kurtarmıyordu. Bu, v01'in B10
    maddesinde bir kez düzeltilen hatanın ikinci kez ortaya çıkışıydı; örüntü artık kayıtlı
    (ön-kayıt V12: her durdurma kuralı `reachability` ile sayısal olarak sınanır).

    SAĞLAM biçim bir EŞDEĞERLİK testidir: aralığın tamamı başarı eşiğinin altında kalmalı
    (`hi < SUCCESS`), yani "etki eşik kadar büyük olabilir" olasılığı veriyle DIŞLANMALI. Nokta
    tahmini koşulu yok. σ büyürse bu da zorlaşır — ama ona doğru çare vardır (n artırmak `hi`'yi
    düşürür), ve zorlaştığında sonuç "terk edilmedi" değil **"yetersiz kesinlik"** olarak
    raporlanır; iki durum artık ayrı.
    """
    oracle_ok = oracle_gain_over_null < oracle_threshold

    def upper(r: dict) -> float:
        """Eşdeğerlik testinin baktığı GA ÜST SINIRI — sansürsüz alt örnekten.

        Neden tam örnekten değil: sansürlü bir seed'in `s_i`'si 2x bütçe ikamesiyle ~−150 puandır
        ve tek bir tanesi n=5'te GA'yı ±80 puana şişirir. Ölçüldü: `s_i = [0.2, −0.1, 0.3, 0.0,
        −149.82]` -> tam örnek GA [−113.13, +53.37], yani `hi < 5` ASLA sağlanamaz ve kural
        sansürlü her senaryoda ulaşılamaz hâle gelirdi. Bu, `decide()`'ın kesinlik ölçütünde
        zaten kapatılmış olan aynı hatanın vazgeçme kuralındaki kopyasıydı — ve V12'nin sayısal
        sınavı onu göremiyordu, çünkü sınav yalnızca Gauss örnekleri çiziyor, sansür çizmiyor.
        Artık `rule_reachability.py` sansürlü senaryoyu da tarıyor.
        """
        return r.get("ci95_uncensored", r["ci95"])[1]

    # B10: "başarısız" satırı TEK ölçekle vazgeçmeye yeter — AMA ÜÇ şartla, ve bu şartların
    # eksikliği bir regresyon üretmişti (aşağıda ölçümüyle).
    #
    # (i) yükseltme hakkı KULLANILMAMIŞ olmalı. v01 B10 birebir şöyle: "Yükseltme hakkı
    #     kullanılmazsa VE S0'da satır 'başarısız' ise hipotez tek ölçekle terk edilir."
    #     İlk uygulamada bu ön-koşul yoktu ve fonksiyonun böyle bir parametresi bile yoktu.
    # (ii) eşdeğerlik testi SANSÜRSÜZ alt örnekte de geçmeli. Ölçülen regresyon: tek bir sansürlü
    #     seed, −149.82 ikamesi yüzünden tam örnek ortalamasını ≈−25'e çekip satırı diğer dört
    #     seed'den BAĞIMSIZ olarak "başarısız" yapıyor. Sonuç: `4 × +%30 + 1 sansür` tek ölçekte
    #     terk ediliyordu, sansürsüz GA üst sınırı **+30.15** iken. Yani tam olarak düzeltmek için
    #     eklenen eşdeğerlik testi bu yolda hiç sorulmuyordu.
    # (iii) oracle şartı — ilk hâlde zaten vardı ve korunuyor.
    if scale_results and scale_results[-1]["row"] == "başarısız":
        last = scale_results[-1]
        hi_u = upper(last)
        why = []
        if escalation_used:
            why.append("yükseltme hakkı KULLANILDI: v01 B10 tek ölçekle terk etmeyi yalnızca hak "
                       "kullanılmadığında veriyor; iki ölçek kuralı uygulanır")
        if hi_u >= SUCCESS:
            why.append(f"satır 'başarısız' ama sansürsüz GA üst sınırı {hi_u:+.2f} >= {SUCCESS}: "
                       "sansürsüz seed'lerde etki hâlâ eşik kadar büyük olabilir (satır sansür "
                       "ikamesinden geliyor, ölçülen etkiden değil)")
        if not oracle_ok:
            why.append(f"oracle kazancı null'ın {oracle_gain_over_null:.2f} puan üstünde "
                       f">= {oracle_threshold}: mükemmel yönlendirmede hâlâ kazanç var")
        if not why:
            return {"abandon": True, "reason": "son ölçekte satır 'başarısız', yükseltme hakkı "
                    f"kullanılmadı, sansürsüz GA üst sınırı {hi_u:+.2f} < {SUCCESS} ve oracle "
                    f"kazancı null'ın {oracle_gain_over_null:.2f} puan üstünde < {oracle_threshold} "
                    "(v01 EK1-B10: tek ölçekle terk edilebilir)",
                    "oracle_ok": True, "single_scale": True, "ci_upper": [hi_u]}
        if not escalation_used and len(scale_results) < 2:
            return {"abandon": False, "oracle_ok": oracle_ok, "single_scale": True,
                    "ci_upper": [hi_u], "reason": "; ".join(why)}
        # Hak kullanıldıysa ya da elimizde iki ölçek varsa aşağıdaki iki-ölçek kuralına düş.

    if len(scale_results) < 2:
        return {"abandon": False, "reason": "iki ardışık ölçek sonucu gerekiyor",
                "n_scales": len(scale_results)}
    last2 = scale_results[-2:]
    excluded = [upper(r) < SUCCESS for r in last2]
    indet = [r["row"] in ("yüksek varyans / belirsiz", "yetersiz kesinlik") for r in last2]
    ok = all(excluded) and oracle_ok
    if any(indet):
        return {"abandon": False, "reason": "yetersiz kesinlik: bir ölçekte GA başarı eşiğini "
                "çözemedi; bu 'hipotez ayakta' DEĞİL, 'ölçemedik' demektir",
                "ci_upper": [upper(r) for r in last2], "oracle_ok": oracle_ok}
    if ok:
        return {"abandon": True, "reason": f"iki ardışık ölçekte GA üst sınırı < {SUCCESS} puan "
                f"({upper(last2[0]):.2f}, {upper(last2[1]):.2f}) ve oracle kazancı "
                f"null'ın {oracle_gain_over_null:.2f} puan üstünde < {oracle_threshold}",
                "ci_upper": [upper(r) for r in last2], "oracle_ok": True}
    why = []
    if not all(excluded):
        why.append("bir ölçekte GA üst sınırı başarı eşiğinin üstünde (etki eşik kadar büyük olabilir)")
    if not oracle_ok:
        why.append(f"oracle kazancı null'ın {oracle_gain_over_null:.2f} puan üstünde "
                   f">= {oracle_threshold}: mükemmel yönlendirmede hâlâ kazanç var")
    return {"abandon": False, "reason": "; ".join(why),
            "ci_upper": [upper(r) for r in last2], "oracle_ok": oracle_ok}


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
                 alpha: float = 0.05 / PRIMARY_FAMILY) -> dict:
    """Birincil-2'nin DÖRT durumlu puanlaması. Karar tablosu birincil-2'ye UYGULANMAZ.

    diffs : seed başına `s_i(X) − s_i(H)` (yüzde puan)

    Neden ayrı: karar tablosunun eşikleri (+%2/+%5/+%10) TASARRUF için kalibre edildi ve
    birincil-2 bir FARK kontrastıdır. `decide()`'a verilseydi +1.88 puanlık gerçek bir
    heterojenlik katkısı "nötr" satırına düşerdi (çünkü 1.88 < 2), yani deneyin asıl sorusunun
    olumlu cevabı "etki yok" diye etiketlenirdi.

    Dört durum BİRBİRİNİ DIŞLAR ve TÜKETİR. İlk hâl üç durum tanımlıyordu ve "katkı var"
    (`lo > 0`) ile "katkı yok" (`hi < MDE`) birbirini dışlamıyordu — ikisi birden doğru olabilir
    ve o hâl mümkün olan EN bilgilendirici sonuçtur: katkı gerçek ama MDE'nin altında.
    """
    if len(diffs) < 2:
        raise ValueError("kontrast için en az 2 seed gerekir")
    n = len(diffs)
    m, sd = st.mean(diffs), st.stdev(diffs)
    h = t_crit_alpha(n - 1, alpha) * sd / math.sqrt(n)
    lo, hi = m - h, m + h
    real, small = lo > CI_EPS, hi < mde        # CI_EPS: `decide()` ile aynı tolerans
    state = ("katkı var, büyük" if real and not small else
             "katkı var, küçük" if real and small else
             "katkı MDE kadar değil" if small else
             "ölçemedik")
    return {"state": state, "mean": m, "sd": sd, "n": n, "ci": [lo, hi],
            "alpha": alpha, "mde": mde,
            "note": ("yeterince güçlü ölçemedik — 'heterojenlik çürütüldü' DEĞİL"
                     if state == "ölçemedik" else "")}


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
        return dict(quality_per_sec), min(quality_per_sec.values())
    return {"?": float(quality_per_sec)}, float(quality_per_sec)


def decide(savings: list[float], quality_per_sec: float | dict[str, float] | None = None,
           censored_mask: list[bool] | None = None, strict: bool = True) -> dict:
    """PREREGISTRATION_V02 karar tablosunu (v01 O5 + EK 1 B8/B10/B16) birebir uygular.

    savings        : seed başına tasarruf, YÜZDE PUAN (`s_i`, bkz. `savings()`); sansürlü seed'ler
                     için 2x bütçe ikamesiyle hesaplanmış değer
    quality_per_sec: w̄ ortalaması (yüzde puan). **None geçmek sessiz değildir**: "güçlü başarı"
                     satırı değerlendirilemez ve bu bir NOT olarak kayda geçer (aşağıya bak).
    censored_mask  : **YALNIZCA FOTON kolunun** seed başına sansür bayrağı (`n_star` None döndü).
                     Baseline'ın sansürü buraya GİRMEZ — v02'de dense kolu `L*`'e ulaşmıyor ve
                     maskeye onun beş sansürü konursa FOTON, baseline'ın yetersizliği yüzünden
                     "başarısız" ilan edilir. EK 1 B8:
                     - >= 1 sansürlü seed: "başarılı"/"güçlü başarı" YASAK
                     - >= 2 sansürlü seed: sonuç doğrudan "başarısız"
    strict         : True ise `n` ön-kayıtlı seed sayılarından biri olmak ZORUNDA (`REQUIRED_N`).
                     False yalnızca keşifsel kullanım içindir ve başarı satırlarını BLOKLAR.

    Döner: {"row", "mean", "sd", "sd_uncensored", "ci95", "n", "n_censored", "notes", "determinate"}
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
    # Doğrulama sansürlü erken dönüşten ÖNCE: aksi hâlde `quality_per_sec={}` normal
    # yolda ValueError verirken `n_cens >= 2` yolunda sessizce kabul ediliyordu.
    wmap, wmin = _wbar(quality_per_sec)
    mean, lo, hi = ci95(savings)
    sd = st.stdev(savings)
    # EK 1 B8: sansürlü seed s_i ~ -150 puan verip sd'yi şişirir ve "hedefe ulaşamıyor" sonucunu
    # "varyans yüksek" gibi gösterirdi. Kesinlik ölçütü bu yüzden sansürsüz seed'lere bakar.
    unc = [s for s, c in zip(savings, mask) if not c]
    sd_unc = st.stdev(unc) if len(unc) > 1 else float("nan")
    notes: list[str] = []
    res = {"mean": mean, "sd": sd, "sd_uncensored": sd_unc, "n": n, "n_censored": n_cens,
           "ci95": [lo, hi], "notes": notes, "w_bar": wmap}
    # Sansürsüz alt örneğin GA'sı AYRI tutulur: `abandon()`'ın eşdeğerlik testi buna bakar, çünkü
    # tek bir sansürlü seed (~−150 puan ikame) tam örnek GA'sını ±80 puana şişirip kuralı
    # ulaşılamaz yapıyordu. `decide()`'ın kesinlik ölçütü aynı sebeple zaten sansürsüzlere bakıyor.
    if len(unc) >= 2:
        mu_, lo_, hi_ = ci95(unc)
        res["ci95_uncensored"] = [lo_, hi_]
        res["mean_uncensored"] = mu_

    if n_cens >= 2:                                        # EK 1 B8: iki veya daha fazla sansür -> başarısız
        notes.append(f"{n_cens} FOTON seed'i L*'e ulaşmadı: EK 1 B8 gereği sonuç doğrudan 'başarısız'")
        return {**res, "row": "başarısız", "determinate": True}

    # ---- kesinlik ölçütü: mutlak sd tripwire'ının yerini aldı (bkz. SD_TRIPWIRE yorumu) ----
    if len(unc) >= 2:
        m_u, lo_u, hi_u = ci95(unc)                        # sansürsüzlerin kendi aralığı
        indet = indeterminate(m_u, lo_u, hi_u)
        detail = f"sansürsüz {len(unc)} seed: s̄ {m_u:.2f}, GA [{lo_u:.2f}, {hi_u:.2f}]"
    else:
        # Tek sansürsüz seed'de sd yok. Eski kod bu durumda HAM sd'ye düşüyordu ve -150'lik ikameyi
        # "sansürsüz seed'lerin sd'si 110.31" diye raporluyordu — var olmayan bir niceliği rapor
        # eden bir yalan, ve tam olarak B8'in engellemek için yazıldığı yanlış okuma.
        indet = True
        detail = f"yalnızca {len(unc)} sansürsüz seed: kesinlik DEĞERLENDİRİLEMEZ"
    res["w_bar"] = wmap

    if indet:
        if n >= max(REQUIRED_N):
            # TERMİNAL: B3 ikinci artırma hakkı vermiyor, ölçek yükseltmesi de kesinlik sorununun
            # çaresi değil. "seed 10'a çıkar" demek olmayan bir çareye yönlendirmek olurdu.
            notes.append(f"GA başarı eşiğini ({SUCCESS} puan) çözemiyor — {detail}; n={n} zaten "
                         "azami seed sayısı (EK 1 B3 ikinci artırma hakkı vermiyor) ve ölçek "
                         "yükseltmesi kesinlik sorununun çaresi değil: etki, başarı eşiğinin bu "
                         "deneyin kesinliğiyle ayırt edilemeyeceği kadar yakınında. Ne kurulmuş "
                         "ne dışlanmıştır; bu 'nötr' DEĞİLDİR.")
            return {**res, "row": "yetersiz kesinlik", "determinate": False}
        notes.append(f"GA başarı eşiğini ({SUCCESS} puan) çözemiyor — {detail}: "
                     "seed 10'a çıkar (EK 1 B3: {66,77,88,99,110})")
        return {**res, "row": "yüksek varyans / belirsiz", "determinate": False}

    # "güçlü başarı" kapısı: w̄ HER İKİ baseline için de ölçülmüş VE ikisi de pozitif olmalı.
    # İlk düzeltme yalnızca `min()`'e bakıyordu ve eksik anahtarı görmüyordu: `{"M": +1.0}`
    # (D hiç ölçülmemiş) "güçlü başarı" döndürüyor, üstelik skaler yolun uyarısını da
    # atlıyordu — yani blocker geri gelmişti, sessiz hâliyle.
    w_complete = wmap is not None and set(WBAR_KEYS) <= set(wmap)
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

    # w̄ yoksa bunu SÖYLE. Eskiden None sessizce geçiyordu: "güçlü başarı" kapısı da B16'nın çelişki
    # etiketi de atlanıyor, ve rapor ikincil ölçütün hiç değerlendirilmediğini göstermiyordu.
    # V11a duvar saatini eksik bir seed bloğunda geçersiz saydığı için bu ulaşılabilir bir durum.
    if wmin is None:
        notes.append("w̄ (quality/sec) VERİLMEDİ: ikincil ölçüt değerlendirilmedi, bu yüzden "
                     "'güçlü başarı' satırı erişilemez ve çelişki kuralı uygulanamadı")
        if row == "güçlü başarı":                          # savunma: buraya düşülemez ama sabitlensin
            row = "başarılı"
    else:
        # İKİ not BAĞIMSIZ: "w̄ eksik ölçüldü" ile "w̄ negatif" aynı anda doğru olabilir. İlk hâl
        # bunları `elif` ile zincirlemişti ve skaler w̄ = −3.0 verildiğinde eksiklik notu yazılıp
        # B16'nın çelişki etiketi ATLANIYORDU — yani iki ayrı kural birbirini susturuyordu.
        if not w_complete and mean >= STRONG and lo > CI_EPS:
            missing = [k for k in WBAR_KEYS if k not in wmap]
            how = ("tek skaler olarak verildi: hangi baseline'a karşı olduğu BİLİNMİYOR"
                   if "?" in wmap else f"yalnızca {sorted(wmap)} için verildi")
            notes.append(f"w̄ {how}; eksik: {missing or list(WBAR_KEYS)}. Ön-kayıt 'güçlü başarı' "
                         f"için w̄({WBAR_KEYS[0]}) > 0 VE w̄({WBAR_KEYS[1]}) > 0 istiyor, bu yüzden "
                         "o satır ERİŞİLEMEZ. Gerekçe: yalnızca M'e karşı hız kazancı, M bu kartta "
                         "kendi başına D'den %27.9 yavaş olduğu için FOTON'un katkısı OLMADAN da "
                         "sağlanabilir (v01 EK1-B16).")
        # EK 1 B16: çelişki etiketi. Eskiden yalnızca `mean > 0` yetiyordu, yani s̄ = +0.01 puanlık
        # bir gürültüye "mimari başarı" damgası vurulabiliyordu. Artık FLOP kazancının kendisi de
        # İSTATİSTİKSEL olmak zorunda: GA'nın tamamı sıfırın üstünde.
        if wmin <= 0:
            neg = sorted(k for k, v in wmap.items() if v <= 0)
            which = f" (negatif olan: {', '.join(neg)})" if neg != ["?"] else ""
            if lo > CI_EPS:
                notes.append("FLOP kazancı var (GA tamamen > 0), duvar saati kaybı var -> "
                             f"çelişki kuralı: mimari başarı / sistem başarısızlığı{which}")
            elif mean > 0:
                notes.append(f"s̄ pozitif ({mean:.2f}) ama GA sıfırı kesiyor ve w̄ <= 0: çelişki "
                             "kuralı UYGULANMAZ (FLOP kazancı istatistiksel olarak kurulmadı)")
            if row == "güçlü başarı":   # wmin<=0 iken buraya düşülemez; kural kodda da sabit
                row = "başarılı"

    if n_cens == 1:
        # B8(i) açıklaması her sansürlü durumda verilir — eskiden yalnızca başarı bloklanmadığında.
        notes.append("1 FOTON seed'i L*'e ulaşmadı; s̄ 2x bütçe ikamesiyle hesaplandı ve bu ikame "
                     "FOTON'un LEHİNE yanlıdır (EK 1 B8)")
        if row in ("güçlü başarı", "başarılı"):
            notes.append("EK 1 B8 gereği 1 sansürlü seed varken 'başarılı' ilan edilemez")
            row = "belirsiz"
    if not strict and row in ("güçlü başarı", "başarılı"):
        notes.append(f"strict=False (n={n} ön-kayıtlı değil): başarı satırı bloklandı, keşifsel sonuç")
        row = "belirsiz"
    return {**res, "row": row, "determinate": True}
