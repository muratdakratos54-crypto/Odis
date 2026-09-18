"""Karar istatistiği: eşleştirilmiş farklar, çoklu karşılaştırma ve ön-kayıttaki karar tablosu — KOD olarak.

Neden ayrı bir modül: PREREGISTRATION.md'nin karar kuralı eşleştirilmiş güven aralıklarına dayanıyor ve
v3 denetimi bu koddaki iki hatayı buldu:

  1. `paired()` seed KİMLİĞİNE göre değil liste KONUMUNA göre eşliyordu. Bir koşulda tek bir analiz eksik
     olduğunda sessizce yanlış seed'leri eşliyordu -> eşleştirilmiş GA anlamsız olurdu.
  2. t kritik değeri n'den bağımsız sabitti (t(2)); n=2'de güven aralığını gerçeğin ~3'te birine daraltıyordu.

Bu hatalar FOTON kararında tekrarlanırsa karar GEÇERSİZ olur, çünkü karar kuralının tamamı o güven
aralığının işaretine bakıyor. Bu yüzden fonksiyonlar burada toplanıp `tests/test_stats.py` ile regresyona
bağlandı; FOTON'un karar betiği bu modülü kullanmak ZORUNDA (PREREGISTRATION O4).

VERİ TAŞIMA BİÇİMİ — seed başına sonuçlar **seed kimliğiyle anahtarlanmış eşleme** olarak taşınır
(`dict[seed] -> değer` ya da `dict[seed] -> Obs`). Paralel liste KABUL EDİLMEZ: `paired()`, `savings()`,
`decide()`, `ub95()`, `contribution()`, `decide_primary()` ve `abandon()` liste alınca `TypeError`
verir. (`paired()` tek istisnadır: o da yalnızca kimlikle eşler ama anahtar kümeleri farklıysa hata
vermez, ortak seed'leri eşleyip düşenleri `dropped`'ta raporlar; karar yolunda KULLANILMAZ.)
Bağlayıcı modda (`strict=True`) iki şey daha zorunludur: anahtarlar ÖN-KAYITLI seed'lerdir
(`DECISION_SEEDS`, seed artırıldıysa + `EXTRA_SEEDS`) ve her seed'in durumu AÇIKÇA verilir (`Obs`);
çıplak sayı "hızlı" sayılmaz, reddedilir. Gerekçe: konuma göre eşleme hatası ÜÇ ayrı fonksiyonda üç kez çıktı (`paired` v3'te, `savings`
v02'nin ilk denetiminde, `decide_primary` dördüncü denetimde) ve her seferinde fonksiyon yamalandı.
Sorun fonksiyonlar değildi: iki paralel liste, yanlış eşlemeyi İFADE EDEBİLEN bir veri biçimidir ve
sansür/yavaş maskeleri de aynı biçimde üçüncü, dördüncü paralel listelerdi. Eşlemede konum diye bir
şey yoktur; iki kol ancak anahtar kümeleri AYNIYSA birleştirilir, değilse hata verilir. Durum
(hızlı/yavaş/sansürlü) değerle AYNI kayıtta (`Obs`) taşınır, ayrı bir maskede değil.

`decide_primary()` bağlayıcı kararı döndürür (iki birincil uç nokta + Holm); `decide()` onun
birincil-1 adımıdır ve karar tablosunun satır ADINI döndürür. Rapor "iyi görünüyor" diyemez:
satır adı buradan gelir.
"""
from __future__ import annotations

import math
import statistics as st
from collections.abc import Mapping
from dataclasses import dataclass

# %97.5 t kritik değerleri (iki yanlı %95), df -> t
T975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582, 6: 2.446912, 7: 2.364624,
        8: 2.306004, 9: 2.262157, 10: 2.228139, 14: 2.144787, 19: 2.093024, 29: 2.045230}

STRONG = 10.0           # puan; "güçlü başarı" eşiği
SUCCESS = 5.0           # puan; "başarılı" eşiği
NEUTRAL = 2.0           # puan; "nötr" bandının yarısı
REQUIRED_N = (5, 10)    # bağlayıcı karar yalnızca 5 ya da 10 eşleştirilmiş seed'le verilir
DECISION_SEEDS = (11, 22, 33, 44, 55)       # ön-kayıt SEED POLİTİKASI: karar seed'leri (S0 ve S1)
EXTRA_SEEDS = (66, 77, 88, 99, 110)         # kesinlik ölçütü tetiklenirse eklenen ikinci beş (EK1-B3)
# "güçlü başarı" için w̄'nin ölçülmüş olması GEREKEN baseline'lar. İkisi de olmadan o satır
# erişilemez: ön-kayıt `w̄(M) > 0` VE `w̄(D) > 0` istiyor, çünkü M bu kartta kendi başına D'den
# token başına %27.9 yavaştır ve yalnızca M'e karşı bir hız kazancı FOTON'un katkısı olmadan da
# sağlanabilir (v01 EK1-B16'nın aynı kolda bulduğu sorun).
WBAR_KEYS = ("M", "D")
CI_EPS = 1e-9           # GA sınırı sıfıra kayan nokta hassasiyetinde değerken kararı bit çevirmesin
S_MAX = 100.0           # puan; tasarrufun aritmetik tavanı (F_foton = 0)
SCALES = ("S0", "S1")   # ölçek basamakları, SIRASIYLA (ön-kayıt V5/V8: en çok 2 ölçek)
STATUSES = ("hızlı", "yavaş", "sansürlü")    # bir FOTON seed'inin hedefe ulaşma durumu (V3)

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


# ------------------------------------------------------------------ seed kimliğiyle anahtarlı taşıma

@dataclass(frozen=True)
class Obs:
    """Bir seed'in tasarruf gözlemi: `s` (yüzde puan) + hedefe ulaşma durumu.

    Seed kimliği burada DEĞİL, eşlemenin ANAHTARINDADIR (`{11: Obs(5.2), 22: Obs(-56.0, "yavaş")}`):
    kimlik tek bir yerde durur, kayıtla anahtar çelişemez.

    status ∈ STATUSES (`resolve_n_star` üretir, elle doldurulmaz):
      "hızlı"    — 1x bütçede hedefe (kalıcı) ulaştı
      "yavaş"    — yalnızca ZORUNLU 2x koşusunda ulaştı; `s` 2x eğrisinden ÖLÇÜLDÜ (tipik −25 … −150)
      "sansürlü" — 2x'te de ulaşamadı; `s`, `N* = 2 × bütçe` İKAMESİDİR (≈ −150) ve FOTON lehine yanlıdır
    """
    s: float
    status: str = "hızlı"

    def __post_init__(self):
        if isinstance(self.s, bool) or not isinstance(self.s, (int, float)):
            raise TypeError(f"Obs.s sayı olmalı, {type(self.s).__name__} verildi")
        s = float(self.s)
        if s != s or s in (float("inf"), float("-inf")):
            raise ValueError(f"Obs.s sonlu olmalı: {self.s}")
        if s > S_MAX:
            raise ValueError(f"Obs.s = {s} > {S_MAX}: tasarruf tanım gereği +100 puanı aşamaz "
                             "(F_foton > 0); birim hatası olabilir (oran değil YÜZDE PUAN)")
        if self.status not in STATUSES:
            raise ValueError(f"Obs.status {STATUSES} içinden olmalı, verilen: {self.status!r}")
        object.__setattr__(self, "s", s)


def _need_mapping(x, name: str) -> Mapping:
    """Konuma göre eşlemeyi KAPATAN kapı: liste/tuple/dizi alan çağrı çalışmaz, patlar."""
    if not isinstance(x, Mapping):
        raise TypeError(f"{name}: seed kimliğiyle anahtarlanmış eşleme gerekli (dict[seed] -> değer), "
                        f"{type(x).__name__} verildi. Paralel listeler KABUL EDİLMEZ: konuma göre "
                        "eşleme üç kez yanlış seed'leri eşledi (bkz. modül docstring'i).")
    return x


def _ordered(keys) -> list:
    """Anahtarların KANONİK sırası: sonuç eşlemenin ekleme sırasına bit düzeyinde bile bağlı olmasın."""
    try:
        return sorted(keys)
    except TypeError:
        return sorted(keys, key=repr)


def keyed(pairs) -> dict:
    """`(seed, değer)` çiftlerinden eşleme kurar; aynı seed İKİ KEZ gelirse hata verir.

    Yükleyiciler bunu kullanmalı: `dict(pairs)` tekrar eden anahtarda öncekini SESSİZCE ezer — koşu
    defterinde yeniden koşulan bir seed'in iki kaydı varken hangisinin kazandığı sıraya bağlı olurdu.
    """
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"seed {k!r} iki kez verildi: hangisinin geçerli olduğu koşu defterinden "
                             "seçilmeli, sıradan değil")
        out[k] = v
    return out


def _keyed(x, name: str) -> dict:
    """`dict[seed] -> float`, kanonik sırada. Değerler sonlu sayı olmalı."""
    m = _need_mapping(x, name)
    out = {}
    for k in _ordered(m):
        v = m[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"{name}[{k!r}] sayı olmalı, {type(v).__name__} verildi")
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"{name}[{k!r}] sonlu olmalı: {v}")
        out[k] = v
    return out


def _table(x, name: str, strict: bool = False) -> dict:
    """`dict[seed] -> Obs`, kanonik sırada.

    Keşifsel modda çıplak sayı `Obs(sayı, "hızlı")` sayılır. Bağlayıcı modda (`strict=True`) SAYILMAZ:
    durumu verilmemiş bir seed'i sessizce "hızlı" saymak, sansürlü/yavaş bir seed'i sıradan gözleme
    çevirir (4. denetimin HIGH'ının aynısı; tur 3'te ölçüldü: aynı FLOP'lar durumla "başarısız",
    durumsuz "yüksek varyans / belirsiz"). Anahtarlar da ön-kayıtlı seed'ler olmak zorundadır: n
    doğru olsa bile 10 seed'in "en iyi 5'i" ya da `dict(enumerate(...))` bağlayıcı karar üretemez.
    """
    m = _need_mapping(x, name)
    if strict:
        bare = [k for k in _ordered(m) if not isinstance(m[k], Obs)]
        if bare:
            raise TypeError(f"{name}: strict=True iken her seed'in durumu AÇIKÇA verilmeli "
                            f"(`Obs(s, durum)`, `savings()` çıktısı); durumsuz seed'ler: {bare}")
        allowed = (set(DECISION_SEEDS), set(DECISION_SEEDS) | set(EXTRA_SEEDS))
        if set(m) not in allowed:
            raise ValueError(f"{name}: ön-kayıtlı seed kümesi gerekli (strict=True) — {DECISION_SEEDS} "
                             f"ya da + {EXTRA_SEEDS}; verilen: {_ordered(m)}")
    return {k: (m[k] if isinstance(m[k], Obs) else Obs(m[k])) for k in _ordered(m)}


def _same_seeds(a: Mapping, b: Mapping, name_a: str, name_b: str) -> None:
    if set(a) != set(b):
        raise ValueError(f"seed kümeleri aynı olmalı: {name_a} {_ordered(a)} vs {name_b} {_ordered(b)}")


def paired(a: Mapping, b: Mapping) -> dict:
    """Eşleştirilmiş fark (a − b), **seed kimliğine göre**. Ortalama, sd, %95 GA t(n−1), iki yanlı p.

    a, b : `dict[seed] -> değer`. Yalnızca İKİ eşlemede de bulunan seed'ler eşlenir; bir tarafta
    eksik olanlar `dropped` alanında raporlanır (sessiz değil). Birincil istatistik bunu KULLANMAZ:
    `savings()` eksik seed'de hata verir.
    """
    fa, fb = _keyed(a, "a"), _keyed(b, "b")
    matched = [s for s in fa if s in fb]
    d = [fa[s] - fb[s] for s in matched]
    dropped = _ordered(set(fa) ^ set(fb))
    out = {"per_seed": d, "n": len(d), "seeds": matched, "dropped": dropped,
           "mean": st.mean(d) if d else float("nan")}
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
       değilse SONRAKİ kesişime bakılır"). Uygulama B14'ün BİLİNÇLİ bir SIKILAŞTIRMASIDIR: `N*`,
       hedefin ÜSTÜNDEKİ son eval noktasından hemen sonraki kesişimdir — yani ardından eğrinin
       bir daha HİÇ hedefin üstüne çıkmadığı ilk kesişim (B14 yalnızca bir sonraki noktaya bakar;
       burada sonraki noktaların TÜMÜ ≤ L* olmalı). Sıkılaştırma `N*`'ı yalnızca büyütebilir. Gürültülü bir eğri hedefin altına bir kez sarkıp geri çıkarsa o erken sarkma
       `N*` sayılmaz: ölçülü örnek, 2750'de sarkıp 3000'de geri çıkan bir eğride ilk kesişim `N*`'ı
       114.7M yerine 90.1M gösteriyor, yani `s_i`'yi ~21 puan şişirip "nötr"ü "güçlü başarı" yapıyor.

       Bu fonksiyonun bir önceki hâli böyle bir eğride None döndürüp seed'i SANSÜRLÜYORDU — B14'ün
       "sonraki kesişime bakılır" hükmünün tersi, ve ağır: 0.0005 nat'lık tek bir sıçrama, hedefin
       0.025 nat altında biten bir koşuyu −150 puana, dolayısıyla bütün sonucu "başarısız"a
       çeviriyordu. Şimdiki uygulama B14'ün yukarıdaki SIKILAŞTIRILMIŞ hâlidir.

    Eğrinin SON noktası hedefin üstündeyse (hedefe ulaşılmadı ya da ulaşılıp kaybedildi) None döner.
    None "bu EĞRİ hedefi kalıcı kesmiyor" demektir, "seed sansürlü" DEĞİL: sansür kararı
    `resolve_n_star`'ındır (önce ZORUNLU 2x koşu; o da kesmezse sansür). Sessizce atlamak seed'i
    kaybettirir; `savings()` eksik seed'de hata verir.
    """
    pts = sorted(curve)
    if not pts or pts[-1][1] > target:
        return None
    above = [j for j, (_, y) in enumerate(pts) if y > target]
    if not above:
        return float(pts[0][0])                  # eğri baştan beri hedefin altında
    (a, ya), (b, yb) = pts[above[-1]], pts[above[-1] + 1]
    return a + (ya - target) / (ya - yb) * (b - a)


def resolve_n_star(curve_1x: list[tuple[float, float]], curve_2x: list[tuple[float, float]] | None,
                   target: float, budget_tokens: float) -> tuple[float, str]:
    """V3'ün sansür prosedürünü MEKANİK olarak uygular: (`N*`, durum), durum ∈ STATUSES.

      "hızlı"    — 1x bütçeli eğri hedefi (kalıcı) kesti; `N*` o eğriden.
      "yavaş"    — 1x kesmedi, ZORUNLU 2x koşu kesti; `N*` 2x eğrisinden.
      "sansürlü" — 2x de kesmedi; `N* = 2 × bütçe` ikamesi.

    Neden fonksiyon: hangi eğrinin okunacağını ve seed'in durumunu analist seçmesin. `Obs.status`
    buradan gelir, elle doldurulmaz.
    """
    budget = float(budget_tokens)
    if not curve_1x or budget <= 0:
        raise ValueError("resolve_n_star: 1x eğrisi boş ya da bütçe pozitif değil")
    # İki eğri de konumlu argümandır; 2x koşunun eğrisi 1x yuvasına düşerse yavaş bir seed "hızlı"
    # etiketlenir ve 4. denetimin HIGH'ı geri gelir (tur 3'te ölçüldü: N*/bütçe = 1.143, 'hızlı').
    # Eğrinin KENDİSİ hangi koşudan geldiğini söyler: 1x koşu bütçede biter, 2x koşu onu aşar.
    tol = 1e-9 * budget
    if max(t for t, _ in curve_1x) > budget + tol:
        raise ValueError(f"resolve_n_star: 1x eğrisi 1x bütçeyi ({budget:.0f} token) aşıyor — "
                         "2x koşunun eğrisi 1x yuvasına verilmiş olabilir")
    n1 = n_star(curve_1x, target)
    if n1 is not None:
        return n1, "hızlı"
    if curve_2x is None:
        raise ValueError("1x eğrisi hedefi kalıcı kesmiyor: 2x koşu ZORUNLU (V3 adım 2), eğrisi verilmeli")
    top2 = max(t for t, _ in curve_2x)
    if not (budget + tol < top2 <= 2.0 * budget + tol):
        raise ValueError(f"resolve_n_star: 2x eğrisi (1x, 2x] bütçe aralığında bitmiyor (son nokta "
                         f"{top2:.0f}, bütçe {budget:.0f}) — 2x koşunun eğrisi bu değil")
    n2 = n_star(curve_2x, target)
    if n2 is not None:
        return n2, "yavaş"
    return 2.0 * budget, "sansürlü"


def savings(f_foton: Mapping, f_base: Mapping, status: Mapping) -> dict:
    """Birincil istatistik: `s_i = 100 × (1 − F_foton,i / F_base,i)`, **seed KİMLİĞİNE göre**.

    f_foton, f_base : `dict[seed] -> FLOP`. Anahtar kümeleri AYNI olmak zorunda; eksik/fazla seed
                      sessizce atlanmaz, hata verir.
    status          : `dict[seed] -> durum` (FOTON kolunun, `resolve_n_star`'dan). ZORUNLU ve varsayılanı
                      YOK: "verilmediyse hepsi hızlı" varsayımı, sansürlü/yavaş bir seed'i sessizce
                      sıradan gözleme çevirirdi.

    Döner: {"obs": dict[seed] -> Obs, "n", "mean", "sd"} — `obs` doğrudan `decide_primary()`'ye verilir.

    Neden ayrı bir fonksiyon: `paired()` FARK hesaplar, `s_i` ise ORAN. Denetim, `s_i`'yi kuran tek
    kodun eşleştirmeyi liste KONUMUNA göre yaptığını buldu — yani `paired()`'ın regresyona bağladığı
    hatanın aynısı, birincil istatistiğin tam üstünde. Bir kol yeniden koşulup listeye sonda
    eklendiğinde (koşu defteri bunu yapar) konum eşlemesi FOTON-s22'yi baseline-s33 ile eşliyordu.

    Ayrıca payda seed BAŞINA (`F_base,i`) — tek bir dondurulmuş skaler DEĞİL. Skaler payda
    kullanmak karşılaştırmayı eşleştirilmiş olmaktan çıkarır (baseline gürültüsü paydan silinir,
    GA yapay olarak daralır); ön-kayıt eşleştirilmiş GA'ya dayandığı için bu kararı geçersiz kılar.
    """
    ff, fb = _keyed(f_foton, "f_foton"), _keyed(f_base, "f_base")
    _same_seeds(ff, fb, "FOTON", "baseline")
    stat = dict(_need_mapping(status, "status"))
    _same_seeds(ff, stat, "FOTON", "status")
    obs = {}
    for s in ff:
        if fb[s] <= 0:
            raise ValueError(f"seed {s}: F_base = {fb[s]}, pozitif olmalı")
        obs[s] = Obs(100.0 * (1.0 - ff[s] / fb[s]), stat[s])      # baseline KİMLİKLE bulunur
    vals = [o.s for o in obs.values()]
    return {"obs": obs, "n": len(vals), "mean": st.mean(vals) if vals else float("nan"),
            "sd": st.stdev(vals) if len(vals) > 1 else float("nan")}


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

    `decide()` `hi` olarak **`ub95`**'i verir: başarısız seed yokken düz GA üst sınırı, varken
    başarısız seed'leri aralığa değil SABİT değer olarak sokan sınır (bkz. `ub95`).

    İkisi de sağlanmıyorsa aralık, "eşiği geçiyor" ile "geçmiyor" arasında ayrım yapamıyor demektir;
    sonuç "yüksek varyans / belirsiz"dir ve seed artırılır. Bu ölçütün mutlak sd eşiğine üstünlüğü:
    GA yarı-genişliği n ile küçüldüğü için **çare ölçütü gevşetir**, sıkılaştırmaz (bkz. SD_TRIPWIRE
    yorumundaki ölçüm ve `reachability`).
    """
    if lo > CI_EPS and mean >= SUCCESS:
        return False
    return hi >= SUCCESS


def ub95(obs: Mapping) -> float:
    """`s̄`'nin BAŞARISIZ-SEED'E DUYARLI %95 üst sınırı — kesinlik ölçütünün (b) kolu ve vazgeçme kuralı buna bakar.

        ub95 = min( (100 × n_h + Σ sabit) / n ,  (n_h × (hızlı seed'lerin %95 GA üst sınırı) + Σ sabit) / n )

    n_h = 1x bütçede hedefe ulaşan ("hızlı") seed sayısı; "sabit" = başarısız seed'lerin değerleri:
    sansürlü seed'in ikamesi (≈ −150) ve yavaş seed'in 2x eğrisinden ÖLÇÜLEN `s_i`'si. Başarısız
    seed yokken bu, düz %95 GA üst sınırının aynısıdır (tavanla kırpılmış).

    Neden düz t-aralığı değil: başarısız bir seed'in değeri büyük ve negatiftir; onu t-aralığına
    sokmak sd'yi şişirir. `[0.2, −0.1, 0.3, 0.0, −149.82]` için düz aralık [−113, +53] çıkar ve
    "üst sınır < +5" şartı sağlanamaz (2. denetim: kural sansürde ulaşılamazdı). Sansürlü ikame
    tek yanlı bir sınırdır (gerçek değer daha da küçük), yavaş seed'in değeri ölçülmüş bir
    sabittir; ikisi de yukarı yönde belirsizlik TAŞIMAZ. Belirsizlik yalnızca hızlı seed'lerden gelir.

    Yalnızca sansürsüz ALT örneğe bakmak da yanlıştı (3. denetim): 3/5 sansürde 2 seed kalıyor,
    t(1) = 12.7 aralığı patlatıyor, en net başarısızlık terk edilemiyordu. Aritmetik tavan bu yüzden
    HER durumda uygulanır (tasarruf seed başına +100 puanı aşamaz): `[−62.5, −125, C, C, C]`
    için tavansız biçim +31.4, tavanlı biçim −49.9 verir.

    Yavaş seed'ler (4. denetim): hedefe yalnızca 2x koşuda ulaşan seed sıradan bir gözlem sayılınca
    1.99x'te (−148.75) sonuç "yetersiz kesinlik", sansürde (−149.82) "başarısız/terk" oluyordu —
    bir puanlık farkla zıt terminal sonuçlar. İkisi de "1x'te hedefe ulaşamadı"dır, aynı muameleyi görür.

    Bu, gözlenen başarısızlık örüntüsüne KOŞULLU bir üst sınırdır (başarısız seed oranının kendi
    örnekleme belirsizliğini içermez); hızlı seed sayısı < 2 ise yalnızca aritmetik tavan kalır.
    """
    tab = _table(obs, "obs")
    n = len(tab)
    if n == 0:
        raise ValueError("ub95: boş eşleme")
    fast = [o.s for o in tab.values() if o.status == "hızlı"]
    fixed = [o.s for o in tab.values() if o.status != "hızlı"]
    arith = (S_MAX * len(fast) + sum(fixed)) / n
    if len(fast) >= 2:
        return min(arith, (len(fast) * ci95(fast)[2] + sum(fixed)) / n)
    return arith


def abandon(scale_results: Mapping, oracle_gain_over_null: Mapping, *, escalation_used: bool,
            oracle_threshold: float = 2.0) -> dict:
    """VAZGEÇME KOŞULU — ön-kayıttaki "Vazgeçmenin İKİ yolu" tablosunun birebir uygulaması.

    scale_results : `dict[ölçek] -> decide_primary()["primary1"]` (ya da `decide()` çıktısı);
        anahtarlar `SCALES`'ten ve BAŞTAN ardışık ({"S0"} ya da {"S0", "S1"}). Bağlayıcı olmalı
        (strict=True, n ∈ {5, 10}).
    oracle_gain_over_null : `dict[ölçek] -> oracle kazancının null'ın üstündeki payı` (yüzde puan),
        AYNI anahtar kümesiyle. Ölçek sonuçları ile oracle kazançları KİMLİKLE eşlenir: iki paralel
        dizi S0'ın kazancını S1'in sonucuyla eşleyebilirdi (S0 = 1.5 / S1 = 2.5 iken sonuç değişir).
    escalation_used : ölçek yükseltme hakkı kullanıldı mı (yalnızca tek ölçek yolunu etkiler).
        ZORUNLU, yalnız anahtar kelimeyle ve varsayılanı YOK: sahibin seçimi S1'den ÖNCE loglanır
        (`scripts/owner_decision.py`) ve buraya o defterden gelir. (Konumlu üçüncü argüman eskiden
        `oracle_threshold`'a bağlanıyordu: `abandon(res, gains, True)` eşiği 1.0 yapıp hakkı
        "kullanılmadı" sayıyor ve terk kararını sessizce çeviriyordu.)

    İKİ ÖLÇEK yolu (elde 2 ölçek sonucu varsa DAİMA bu): ikisinde de `ub95 < +5`, hiçbiri "yüksek
      varyans / belirsiz" ya da "yetersiz kesinlik" değil, ve oracle kazancı her iki ölçekte < eşik.
      (`ub95 < +5` kesinlik ölçütünün (b) kolunun kendisidir, yani ikinci şart birincisinden çıkar;
      yine de ayrı denetlenir ki reddin GEREKÇESİ "ölçemedik" diye doğru yazılsın.)
    TEK ÖLÇEK yolu (v01 EK1-B10; yalnızca elde TEK ölçek sonucu varken): satır "başarısız",
      yükseltme hakkı KULLANILMAMIŞ, `ub95 < +5`, oracle kazancı < eşik.

    Üçüncü denetimin bulduğu regresyon: önceki hâl tek ölçek çıkışını elde İKİ ölçek varken de
    deniyordu — `[S0 "başarılı", S1 "başarısız"]` hipotezi terk ettiriyordu, oysa iki ölçek kuralı
    (S0'ın GA'sı [6.5, 7.5], tamamen +5'in ÜSTÜNDE) açıkça "terk yok" diyor.

    Eski biçim (`s̄ <= 0` VE `GA üst < +2`) neden atıldı: bir nokta tahminini bir aralık sınırıyla
    "ve" ile bağlıyordu; GA yarı-genişliği ~1.167σ olduğundan σ >~ 1.72'de tam başabaş bir FOTON
    asla terk edilemiyordu (v01 B10'da bir kez düzeltilen hatanın ikinci ortaya çıkışı; bkz. V12).
    """
    res = _need_mapping(scale_results, "scale_results")
    gains_in = _need_mapping(oracle_gain_over_null, "oracle_gain_over_null")
    if not res:
        return {"abandon": False, "reason": "ölçek sonucu yok", "path": None}
    unknown = [k for k in res if k not in SCALES]
    order = [k for k in SCALES if k in res]
    if unknown or tuple(order) != SCALES[:len(order)]:
        raise ValueError(f"abandon(): ölçek anahtarları {SCALES} içinden ve BAŞTAN ardışık olmalı, "
                         f"verilen: {list(res)}")
    if set(gains_in) != set(res):
        raise ValueError(f"oracle kazancı ölçek BAŞINA ve AYNI ölçeklerle verilmeli: sonuçlar "
                         f"{order}, oracle {list(gains_in)}")
    for k in order:
        r = res[k]
        if not isinstance(r, Mapping) or "ub95" not in r or "row" not in r:
            raise ValueError("abandon(): girdi `decide()` çıktısı olmalı (`ub95` / `row` yok)")
        if not r.get("strict", False) or r.get("n") not in REQUIRED_N:
            raise ValueError(f"abandon(): bağlayıcı olmayan sonuç (strict={r.get('strict')}, "
                             f"n={r.get('n')}) vazgeçme kararını süremez")
    two = len(order) >= 2
    used = order[-2:]
    if not isinstance(escalation_used, bool):
        raise TypeError("escalation_used bool olmalı (sahibin karar defterinden)")
    if isinstance(oracle_threshold, bool) or not isinstance(oracle_threshold, (int, float)):
        raise TypeError("oracle_threshold sayı olmalı")
    for k in used:
        g = gains_in[k]
        if isinstance(g, bool) or not isinstance(g, (int, float)) or g != g or g in (float("inf"), float("-inf")):
            raise ValueError(f"oracle kazancı her ölçek için sonlu bir sayı olmalı: {k} -> {g!r}")
    gains = [float(gains_in[k]) for k in used]                 # oracle ölçekle KİMLİKLE eşlenir
    oracle_ok = all(g < oracle_threshold for g in gains)
    ubs = [res[k]["ub95"] for k in used]
    base = {"ub95": ubs, "oracle_ok": oracle_ok, "single_scale": not two, "scales": used,
            "escalation_used": escalation_used, "path": "iki ölçek" if two else "tek ölçek"}
    why = []
    if two:
        if any(res[k]["row"] in ("yüksek varyans / belirsiz", "yetersiz kesinlik") for k in used):
            return {**base, "abandon": False,
                    "reason": "yetersiz kesinlik: bir ölçekte GA başarı eşiğini çözemedi; bu "
                              "'hipotez ayakta' DEĞİL, 'ölçemedik' demektir"}
    else:
        r = res[used[0]]
        if r["row"] in ("yüksek varyans / belirsiz", "yetersiz kesinlik"):
            why.append(f"satır '{r['row']}': ölçülemedi — bu satırdan ne tek ölçekle terk ne ölçek "
                       "yükseltme hakkı doğar ('hipotez ayakta' DEĞİL, 'ölçemedik')")
        elif r["row"] != "başarısız":
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


def contribution(s_x: Mapping, s_h: Mapping, mde: float = MDE_PRIMARY2,
                 alpha: float = 0.05 / PRIMARY_FAMILY, strict: bool = True) -> dict:
    """Birincil-2'nin DÖRT durumlu puanlaması. Karar tablosu birincil-2'ye UYGULANMAZ.

    s_x, s_h : `dict[seed] -> Obs` — X ve H kollarının M'e karşı tasarrufları (`savings()["obs"]`).
               Kontrast `s_i(X) − s_i(H)` BURADA, seed KİMLİĞİYLE kurulur; hazır bir fark listesi
               kabul edilmez, çünkü o liste bir yerde iki kolun KONUMA göre çıkarılmasıyla kurulmuş
               olurdu (dördüncü denetim: X-s22 ile H-s33 eşlenince "katkı var, büyük" olan sonuç
               "ölçemedik" çıkıyordu). Anahtar kümeleri aynı değilse hata.

    BAŞARISIZ SEED KURALI: X ya da H'nin herhangi bir seed'i 1x bütçede hedefe ulaşamadıysa
    (sansürlü ya da yavaş) fark o seed'de anlamlı bir gözlem değildir (farkı onlarca puan oynatır,
    t-aralığını patlatır). O zaman durum "ölçemedik"tir, p = 1 alınır (Holm'da birincil-1 α/2 ile
    sınanır: muhafazakâr yön) ve seed DÜŞÜRÜLMEZ — düşürmek sonuca göre seçilebilir bir analizdi:
    ölçülü örnekte tam liste "ölçemedik", seed düşürülünce "katkı var, büyük".

    Neden ayrı bir kural: karar tablosunun eşikleri (+%2/+%5/+%10) TASARRUF için kalibre edildi ve
    birincil-2 bir FARK kontrastıdır. `decide()`'a verilseydi +1.88 puanlık gerçek bir
    heterojenlik katkısı "nötr" satırına düşerdi (çünkü 1.88 < 2), yani deneyin asıl sorusunun
    olumlu cevabı "etki yok" diye etiketlenirdi.

    Dört durum BİRBİRİNİ DIŞLAR ve TÜKETİR. İlk hâl üç durum tanımlıyordu ve "katkı var"
    (`lo > 0`) ile "katkı yok" (`hi < MDE`) birbirini dışlamıyordu — ikisi birden doğru olabilir
    ve o hâl mümkün olan EN bilgilendirici sonuçtur: katkı gerçek ama MDE'nin altında.

    Aralık Holm'un İLK adımının düzeyindedir (α = .025, yani %97.5 GA) ve MDE çıtası n'den
    bağımsız SABİTTİR (1.88 puan); ikisi de ön-kayıtta pinli.
    """
    n = len(_need_mapping(s_x, "s_x"))
    if n < 2:
        raise ValueError("kontrast için en az 2 seed gerekir")
    if strict and n not in REQUIRED_N:
        # `decide()`'daki aynı boşluğun kopyası: bağlayıcı giriş 3 seed'den "katkı var, büyük"
        # damgası basabiliyordu.
        raise ValueError(f"bağlayıcı birincil-2 için n {REQUIRED_N} olmalı, verilen n={n}")
    x, h = _table(s_x, "s_x", strict), _table(s_h, "s_h", strict)
    _same_seeds(x, h, "X", "H")                # iki YÖNDE: H'nin fazla seed'i de (n=5 / n=10) hata
    seeds = list(x)
    failed = [s for s in seeds if x[s].status != "hızlı" or h[s].status != "hızlı"]
    if failed:
        return {"state": "ölçemedik", "mean": None, "sd": None, "n": n, "ci": None, "p": 1.0,
                "alpha": alpha, "mde": mde, "seeds": seeds, "failed_seeds": failed,
                "note": f"{len(failed)} seed'de ({failed}) X ya da H 1x bütçede hedefe ulaşamadı "
                        "(sansürlü/yavaş): fark anlamlı bir gözlem değil, seed DÜŞÜRÜLMEZ; birincil-2 "
                        "'ölçemedik' — 'heterojenlik çürütüldü' DEĞİL"}
    diffs = [x[s].s - h[s].s for s in seeds]                   # H, X ile KİMLİKLE eşlenir
    m, sd = st.mean(diffs), st.stdev(diffs)
    hw = t_crit_alpha(n - 1, alpha) * sd / math.sqrt(n)
    lo, hi = m - hw, m + hw
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
    return {"state": state, "mean": m, "sd": sd, "n": n, "ci": [lo, hi], "p": _p_two_sided(diffs),
            "alpha": alpha, "mde": mde, "seeds": seeds, "failed_seeds": [], "note": note}


def _p_two_sided(values: list[float]) -> float:
    """Ortalamanın sıfırdan farkı için iki yanlı t-testi p değeri (tek örneklem)."""
    n = len(values)
    m, sd = st.mean(values), st.stdev(values)
    if sd == 0:                                  # dejenere: bütün seed'ler aynı
        return 0.0 if m != 0 else 1.0
    return t_sf(abs(m / (sd / math.sqrt(n))), n - 1) * 2


def decide_primary(s_x: Mapping, s_h: Mapping,
                   quality_per_sec: float | dict[str, float] | None = None,
                   strict: bool = True, alpha: float = 0.05,
                   seed_right_available: bool = True) -> dict:
    """İKİ birincil uç noktayı BİRLİKTE karara bağlar: Holm (m = 2) satır adının ÖNÜNDEDİR.

    s_x : `dict[seed] -> Obs`, `s_i(X↔M)` — birincil-1'in istatistiği (`savings()["obs"]`)
    s_h : `dict[seed] -> Obs`, `s_i(H↔M)`, AYNI seed kümesiyle — birincil-2 `s_x − s_h` kontrastıdır

    İki kol seed KİMLİĞİYLE birleşir; konum diye bir şey yoktur (bkz. modül docstring'i). Bu
    fonksiyonun bir önceki hâli iki paralel listeyi `zip()` ile eşliyordu — `paired()` ve
    `savings()`'in regresyona bağladığı hatanın ÜÇÜNCÜ ortaya çıkışı, birincil-2'nin tam üstünde.

    Neden farkı çağıran değil kod kuruyor: iki endpoint'in farklı n'de (ör. birincil-1 n=5,
    birincil-2 n=10) ya da başarısız bir seed DÜŞÜRÜLEREK hesaplanması mümkün olmasın — ön-kaydın
    "sonuca göre seçilebilir" diye adını koyduğu seçim.

    Sıra (ön-kayıttaki "Holm iki endpoint'e NASIL uygulanır" bölümünün birebir uygulaması):
      1. iki endpoint'in iki yanlı p değerleri (birincil-2 BAŞARISIZ SEED yüzünden "ölçemedik"se
         p₂ = 1; GA yüzünden "ölçemedik"se p₂ hesaplanan değerdir — iki hâlde de p₂ > α/2),
      2. Holm ADIM ADIM, aile büyüklüğü PİNLİ (`PRIMARY_FAMILY`): küçük p ilk adımı geçemezse durur,
      3. birincil-1 satırı `decide()`'dan; Holm'dan geçmediyse başarı satırları "belirsiz"e düşer,
      4. birincil-2 durumu `contribution()`'dan (α/2 düzeyi). "Katkı var" `p₂ < α/2` demektir ve
         Holm'un her adımından geçer; kayan nokta bıçak sırtında geçmezse muhafazakâr yöne düşürülür.
    """
    x, h = _table(s_x, "s_x", strict), _table(s_h, "s_h", strict)
    _same_seeds(x, h, "X", "H")                # iki YÖNDE: H'nin fazla seed'i de (n=5 / n=10) hata
    r1 = decide(x, quality_per_sec=quality_per_sec, strict=strict,
                seed_right_available=seed_right_available)
    p1 = _p_two_sided([o.s for o in x.values()])
    c2 = contribution(x, h, alpha=alpha / PRIMARY_FAMILY, strict=strict)
    p2 = c2["p"]
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
            "primary2": {**c2, "holm_passed": "birincil-2" in passed},
            "holm": holm}


BOUNDARY_BAND = 0.10     # 1x bütçe sınırının ±%10'u; ön-kayıtta, veri görülmeden pinlendi (TEŞHİS bandı)


def boundary_diagnostic(s_x: Mapping, s_h: Mapping, ratio_x: Mapping, ratio_h: Mapping,
                        quality_per_sec: float | dict[str, float] | None = None,
                        strict: bool = True, alpha: float = 0.05,
                        seed_right_available: bool = True) -> dict:
    """1x SINIRI TEŞHİSİ — karar kuralı DEĞİLDİR. Bağlayıcı karar `decide_primary()`'ninkidir.

    ratio_x, ratio_h : `dict[seed] -> N* / (1x referans bütçe token'ı)`, `resolve_n_star()`'ın `N*`'ından.

    Ön-kayıt tek bir durum süreksizliğini bilerek kabul ediyor: 1x bütçenin hemen içinde hedefe
    ulaşan seed sıradan gözlemdir, hemen dışında ulaşan sabit değer sayılır. Sahibin kararı:
    kaldırma, GÖRÜNÜR yap. Bir seed'in `N*`'ı 1x bütçenin ±%10 bandına (`BOUNDARY_BAND`) düşerse,
    kararın sınırın DİĞER tarafında ne olacağı da raporlanır: banddaki seed'lerin durumu tersine
    çevrilir (hızlı <-> yavaş; `s_i` AYNI kalır, değişen yalnızca muameledir), hepsi birlikte, ve
    `decide_primary()` yeniden koşulur. Sansürlü seed banda giremez (oranı 2'dir).

    Durum ile oran tutarsızsa hata verir: "hızlı" ise oran <= 1 (1x eğrisi bütçede biter),
    "sansürlü" ise oran = 2 (ikame), "yavaş" ise 0 < oran <= 2. Yavaş seed'in oranı 1'in biraz
    ALTINDA olabilir: durum "hangi eğri kesti"ye göre atanır ve zorunlu 2x koşu ayrı bir koşudur,
    yeniden koşum gürültüsü içinde 1x bütçeden önce kesebilir (tur 3 denetimi, L2). O seed de
    banddadır ve raporlanır. Bu denetim, durumun elle değil `resolve_n_star()` ile atandığının
    mekanik kontrolüdür.
    """
    x, h = _table(s_x, "s_x", strict), _table(s_h, "s_h", strict)
    rx, rh = _keyed(ratio_x, "ratio_x"), _keyed(ratio_h, "ratio_h")
    _same_seeds(x, rx, "X", "ratio_x")
    _same_seeds(h, rh, "H", "ratio_h")
    for arm, tab, rat in (("X", x, rx), ("H", h, rh)):
        for s, o in tab.items():
            ok = (rat[s] <= 1.0 if o.status == "hızlı" else
                  0.0 < rat[s] <= 2.0 if o.status == "yavaş" else rat[s] == 2.0)
            if not ok:
                raise ValueError(f"{arm} seed {s}: durum {o.status!r} ile N*/bütçe = {rat[s]:.4f} tutarsız "
                                 "(durum `resolve_n_star()`'dan gelmeli)")
    lo_, hi_ = 1.0 - BOUNDARY_BAND, 1.0 + BOUNDARY_BAND

    def in_band(tab, rat):
        return [s for s, o in tab.items() if o.status != "sansürlü" and lo_ <= rat[s] <= hi_]

    def flipped(tab, seeds):
        return {s: (Obs(o.s, "yavaş" if o.status == "hızlı" else "hızlı") if s in seeds else o)
                for s, o in tab.items()}

    bx, bh = in_band(x, rx), in_band(h, rh)
    kw = dict(quality_per_sec=quality_per_sec, strict=strict, alpha=alpha,
              seed_right_available=seed_right_available)
    actual = decide_primary(x, h, **kw)
    out = {"band": BOUNDARY_BAND, "in_band": {"X": bx, "H": bh},
           "actual": {"row": actual["primary1"]["row"], "state": actual["primary2"]["state"],
                      "ub95": actual["primary1"]["ub95"]},
           "counterfactual": None, "changes": False,
           "note": "TEŞHİS: bağlayıcı karar mekanik durum atamasından çıkandır (`actual`)"}
    if bx or bh:
        cf = decide_primary(flipped(x, set(bx)), flipped(h, set(bh)), **kw)
        out["counterfactual"] = {"row": cf["primary1"]["row"], "state": cf["primary2"]["state"],
                                 "ub95": cf["primary1"]["ub95"]}
        out["changes"] = (out["counterfactual"]["row"] != out["actual"]["row"]
                          or out["counterfactual"]["state"] != out["actual"]["state"])
    return out


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
# "yetersiz kesinlik" TERMİNAL satırdır: kesinlik ölçütü n = max(REQUIRED_N)'de YA DA tek seed artırma
# hakkı bu kampanyada zaten harcanmışken (S0'da kullanıldıysa S1'de n = 5'te) tetiklenirse döner. Ayrı bir ad olması şart: B3 ikinci bir seed artırma hakkı vermiyor, dolayısıyla o noktada
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
        wmap = {k: float(quality_per_sec[k]) for k in WBAR_KEYS if k in quality_per_sec}   # kanonik sıra
    else:
        wmap = {"?": float(quality_per_sec)}
    # NaN/inf REDDEDİLİR: `min()` NaN'la sıra-bağımlıdır ({"M": nan, "D": 1} -> nan, {"D": 1,
    # "M": nan} -> 1) ve `nan > 0` False olsa da ikinci sırada kapıyı GEÇİYORDU.
    bad = {k: v for k, v in wmap.items() if v != v or v in (float("inf"), float("-inf"))}
    if bad:
        raise ValueError(f"w̄ sonlu bir sayı olmalı: {bad} (ölçülemediyse anahtarı hiç verme)")
    return wmap, min(wmap.values())


def decide(obs: Mapping, quality_per_sec: float | dict[str, float] | None = None,
           strict: bool = True, seed_right_available: bool = True) -> dict:
    """PREREGISTRATION_V02 karar tablosunu (öncelik tablosu + v01 EK 1 B8/B16) birebir uygular.

    **Bağlayıcı karar bu fonksiyonla TEK BAŞINA verilmez**: iki birincil uç nokta Holm'la birlikte
    `decide_primary()`'de karara bağlanır; `decide()` onun birincil-1 adımıdır.

    obs            : `dict[seed] -> Obs` — seed başına tasarruf (YÜZDE PUAN) ve durumu, **YALNIZCA
                     FOTON kolunun** (`savings()["obs"]`). Liste KABUL EDİLMEZ. Durum baseline'ın
                     değil FOTON'un durumudur: v02'de dense kolu `L*`'e ulaşmıyor ve onun
                     başarısızlığı buraya yazılırsa FOTON, baseline yüzünden "başarısız" ilan edilir.
                     - "sansürlü" (EK 1 B8): >= 1 -> "başarılı"/"güçlü başarı" YASAK; >= 2 -> doğrudan "başarısız"
                     - "yavaş": değer ÖLÇÜLMÜŞTÜR, ikame değildir -> B8'in sansür sayımına GİRMEZ;
                       ama sansürlü seed gibi `ub95`'e SABİT değer olarak girer (t-aralığına değil)
    quality_per_sec: w̄ (yüzde puan), baseline başına eşleme `{"M": .., "D": ..}`. **None geçmek
                     sessiz değildir**: "güçlü başarı" erişilemez ve bu bir NOT olarak kayda geçer.
    strict         : True ise `n` ön-kayıtlı seed sayılarından biri olmak ZORUNDA (`REQUIRED_N`).
                     False yalnızca keşifsel kullanım içindir; başarı satırlarını BLOKLAR ve sonuç
                     `abandon()`'a verilemez.
    seed_right_available : tek seed artırma hakkı (EK1-B3, kampanya başına BİR kez) hâlâ duruyor mu.
                     False iken kesinlik ölçütü tetiklenirse satır doğrudan "yetersiz kesinlik"tir:
                     hak S0'da harcandıysa S1'de "seed 10'a çıkar" demek olmayan bir çaredir.

    KESİNLİK ÖLÇÜTÜ (öncelik tablosu satır 1) TEK biçimdir: karar belirlidir ⟺
    (`GA alt > 0` ve `s̄ >= 5`) ya da `ub95 < 5`. Başarısız seed yokken `ub95` düz GA üst sınırıdır.
    Başarısız (sansürlü/yavaş) seed varken o seed aralığı ŞİŞİRMEZ: "1x'te hedefe ulaşamadı" bir
    yöntem başarısızlığıdır, varyans değil. Dördüncü denetim buldu: yavaş seed sıradan gözlem
    sayılınca tek bir seed (s_i −56 … −149) t-aralığını patlatıp her kampanyayı önce 10 seed'e,
    sonra terminal "yetersiz kesinlik"e sürüyor, "başarısız" satırı ve vazgeçme ulaşılamıyordu.

    Döner: {"row", "mean", "sd", "sd_fast", "ci95", "ci95_fast", "ub95", "n", "n_censored",
            "n_slow", "seeds", "notes", "determinate", "strict", "seed_right_available", "w_bar"}
    """
    n = len(_need_mapping(obs, "obs"))
    if n < 2:
        raise ValueError("karar için en az 2 seed gerekir")
    # Kilit cümle BEŞ eşleştirilmiş seed istiyor, `decide()` ise iki seed'le "başarılı" damgası
    # basıyordu — ve `paired()` eşleşmeyen seed'leri düşürdüğü için kısa seed kümesi işleyişte
    # ulaşılabilir bir durumdu. Artık bağlayıcı karar yalnızca 5 ya da 10 seed'le verilir.
    if strict and n not in REQUIRED_N:
        raise ValueError(f"bağlayıcı karar için n {REQUIRED_N} olmalı, verilen n={n}. "
                         f"Keşifsel kullanım için strict=False (başarı satırları bloklanır).")
    tab = _table(obs, "obs", strict)          # bağlayıcı modda: ön-kayıtlı seed'ler + AÇIK durum
    seeds = list(tab)
    vals = [o.s for o in tab.values()]
    n_cens = sum(o.status == "sansürlü" for o in tab.values())
    n_slow = sum(o.status == "yavaş" for o in tab.values())
    wmap, wmin = _wbar(quality_per_sec)          # doğrulama HER dönüş yolundan önce
    mean, lo, hi = ci95(vals)
    sd = st.stdev(vals)
    fast = [o.s for o in tab.values() if o.status == "hızlı"]
    ub = ub95(tab)
    res = {"mean": mean, "sd": sd, "sd_fast": st.stdev(fast) if len(fast) > 1 else None,   # None: NaN JSON'a yazılamaz
           "n": n, "n_censored": n_cens, "n_slow": n_slow, "seeds": seeds, "ci95": [lo, hi],
           "ub95": ub, "strict": strict, "seed_right_available": seed_right_available, "w_bar": wmap}
    if len(fast) >= 2:
        mu_, lo_, hi_ = ci95(fast)
        res["ci95_fast"], res["mean_fast"] = [lo_, hi_], mu_
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
        if n_slow >= 1:
            extra.append(f"{n_slow} FOTON seed'i L*'e yalnızca ZORUNLU 2x koşusunda ulaştı (yavaş): "
                         "s_i 2x eğrisinden ölçüldü ve üst sınıra SABİT değer olarak girdi (V3 adım 3)")
        return {**res, "notes": notes + extra, "row": row, "determinate": determinate}

    if n_cens >= 2:                                        # EK 1 B8: iki veya daha fazla sansür -> başarısız
        notes.append(f"{n_cens} FOTON seed'i L*'e ulaşmadı: EK 1 B8 gereği sonuç doğrudan 'başarısız'")
        return finish("başarısız", True)

    # ---- kesinlik ölçütü (mutlak sd tripwire'ının yerini aldı). Üst sınır `ub95`'tir: başarısız
    #      seed yokken düz GA üst sınırı, varken başarısız seed'leri SABİT sayan sınır ----
    if indeterminate(mean, lo, ub):
        detail = f"s̄ {mean:.2f}, GA alt {lo:.2f}, üst sınır (ub95) {ub:.2f}"
        if n_cens or n_slow:
            detail += f"; {n_cens} sansürlü + {n_slow} yavaş seed üst sınıra sabit değer olarak girdi"
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
