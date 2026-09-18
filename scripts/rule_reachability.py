"""Her durdurma/tripwire kuralını makul σ aralığında SAYISAL olarak sınar — ön-kayıt V12'nin şartı.

    .venv\\Scripts\\python scripts/rule_reachability.py --out experiments/foton-v02-prereg/reachability.log

NEDEN BU BETİK VAR. Aynı hata iki kez yapıldı:

  * v01 EK1-B10: vazgeçme koşulu "iki ardışık ölçek" kısmı yüzünden aritmetik olarak ulaşılamazdı.
  * v02'nin ilk hâli: aynı koşulun GA kısmı yüzünden yine ulaşılamazdı — `s̄ <= 0` **ve**
    `GA üst < +2` biçimi, GA yarı-genişliği ~1.167σ olduğu için `s̄ < 2 − 1.167σ` demeye geliyordu
    ve σ >~ 1.72'de tam başabaş bir FOTON asla terk edilemiyordu.

Örüntü belli: **bir NOKTA TAHMİNİNİ bir ARALIK SINIRIYLA "ve" ile bağlayan durdurma kuralları,
varyans büyüdükçe ölür.** Gözle görülmüyor; hesaplanınca görülüyor. Bu yüzden kural yazmak artık
yetmiyor, kuralın ulaşılabilirliğini ölçmek zorunlu.

SINANAN ÜÇ ÖZELLİK:
  1. Ulaşılabilirlik — kural, tetiklenmesi GEREKEN durumda makul σ aralığında tetikleniyor mu?
  2. Çarenin yönü — kuralın kendi çaresi (seed artırmak) kuralı GEVŞETİYOR mu, sıkıştırıyor mu?
     Mutlak sd eşiği bu sınavı geçemiyordu: çare tripwire'ı daha olası yapıyordu.
  3. Yanlış pozitif — gerçek etki sıfırken başarı ilan etme oranı %5'in altında mı?

Hız notu: `src.stats.reachability` genel ve yavaş (kuralı Python'da örnek başına çağırır). Buradaki
kurallar aynı matematiğin VEKTÖRİZE hâlidir ve `--crosscheck` ile genel yola karşı doğrulanır.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.stats import (  # noqa: E402
    MDE_PRIMARY2, PRIMARY_FAMILY, SUCCESS, Obs, abandon, ci95, contribution, decide, reachability,
    t_crit, t_crit_alpha, ub95,
)

SIGMAS = (0.88, 1.5, 1.72, 3.0, 4.28, 5.0, 6.0)      # 0.88 = ÖLÇÜLEN sd(s_i), ikinci küme
out: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    out.append(s)


def draw(mu: float, sd: float, n: int, sims: int, seed: int = 0, alpha: float = 0.05):
    """(ortalama, GA alt, GA üst, örnek sd) — sims x n örnekten, vektörize.

    `alpha`: GA düzeyi. Birincil-2'nin kuralı Holm'un ilk adımında (α=.025) tanımlı olduğu için
    o bölüm .025 veriyor; karar tablosu ve vazgeçme kuralı .05'te kalıyor (tek birincil test
    değil, aile düzeltmesi birincil-1'e Holm üzerinden uygulanıyor)."""
    x = np.random.default_rng(seed).normal(mu, sd, size=(sims, n))
    m = x.mean(axis=1)
    s = x.std(axis=1, ddof=1)
    h = (t_crit(n - 1) if alpha == 0.05 else t_crit_alpha(n - 1, alpha)) * s / math.sqrt(n)
    return m, m - h, m + h, s


# ---- kural aileleri, vektörize ----------------------------------------------------------------

def trip_precision(m, lo, hi, s):
    """YENİ tripwire: GA başarı eşiğini çözemiyor (bkz. src.stats.indeterminate)."""
    determinate = ((lo > 0) & (m >= SUCCESS)) | (hi < SUCCESS)
    return ~determinate


def trip_abs_sd(m, lo, hi, s):
    """ESKİ tripwire: örnek sd > 5 puan. Karşılaştırma için duruyor."""
    return s > 5.0


def abandon_robust(m, lo, hi, s):
    """YENİ vazgeçme: GA'nın tamamı başarı eşiğinin altında (eşdeğerlik testi)."""
    return hi < SUCCESS


def abandon_old(m, lo, hi, s):
    """ESKİ vazgeçme: `s̄ <= 0` VE `GA üst < +2`. Nokta tahmini + aralık sınırı, "ve" ile."""
    return (m <= 0) & (hi < 2.0)


def success(m, lo, hi, s):
    """Başarı satırı: `s̄ >= 5` ve GA tamamen sıfırın üstünde."""
    return (m >= SUCCESS) & (lo > 0)


def table(title: str, rules: dict, mu: float, ns=(5, 10), sims: int = 200_000) -> None:
    say(f"\n## {title}   (gerçek etki μ = {mu:+.1f} puan)")
    head = f"  {'σ':>6s}"
    for name in rules:
        for n in ns:
            head += f" {name + f' n={n}':>16s}"
    say(head)
    for sd in SIGMAS:
        row = f"  {sd:6.2f}"
        for fn in rules.values():
            for n in ns:
                m, lo, hi, s = draw(mu, sd, n, sims)
                row += f" {100 * float(fn(m, lo, hi, s).mean()):15.1f}%"
        say(row)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/foton-v02-prereg/reachability.log")
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--crosscheck", action="store_true",
                    help="vektörize kuralları src.stats'ın genel (yavaş) yoluna karşı doğrula")
    args = ap.parse_args()

    say("# Durdurma/tripwire kurallarının ulaşılabilirlik sınavı (ön-kayıt V12)")
    say(f"başarı eşiği = {SUCCESS} puan | GA = t(n−1) | sims = {args.sims:,}")
    say("ÖLÇÜLEN sd(s_i) = 0.88 puan (M↔D, L_aux); v01 O9 FOTON'un sd'sinin BÜYÜK olmasını bekliyor,")
    say("bu yüzden σ aralığı 6.0'a kadar taranıyor — kural orada da anlamlı davranmak zorunda.")

    table("TRIPWIRE: tetiklenme oranı", {"kesinlik": trip_precision, "eski sd>5": trip_abs_sd},
          mu=0.0, sims=args.sims)
    say("  YORUM — çarenin yönü: kesinlik ölçütünde n=5 -> n=10 oranı DÜŞÜYOR (çare kuralı")
    say("  gevşetiyor). Eski mutlak sd eşiğinde YÜKSELİYOR: tripwire'ın tek çaresi tripwire'ı daha")
    say("  olası yapıyordu ve tabloda 'n=10'da ateşlendi' satırı yoktu, yani kural kendi çıkmazını")
    say("  üretiyordu. Kesinlik ölçütü ölçülen σ = 0.88'de hiç tetiklenmiyor (%0.0).")

    table("VAZGEÇME: tek ölçekte tetiklenme (başabaş FOTON terk edilebiliyor mu?)",
          {"sağlam": abandon_robust, "eski": abandon_old}, mu=0.0, sims=args.sims)
    _r3 = float(abandon_robust(*draw(0.0, 3.0, 5, args.sims)).mean())
    _o3 = float(abandon_old(*draw(0.0, 3.0, 5, args.sims)).mean())
    say("  YORUM (HESAPLANAN değerlerden): eski biçim σ >= 1.72'de ÖLÜ (başabaş bir FOTON terk edilemez).")
    say(f"  Sağlam biçim σ = 3.0'da tek ölçekte %{100*_r3:.0f}, iki ardışık ölçekte %{100*_r3*_r3:.0f} — "
        f"eski biçimde iki ölçek %{100*_o3*_o3:.1f}'di.")
    say("  Sağlam biçim σ büyüdükçe de zorlaşır, AMA iki fark var: (i) n artırmak yardım eder,")
    say("  (ii) tetiklenmediğinde sonuç 'hipotez ayakta' değil 'yetersiz kesinlik' olarak yazılır.")

    say("\n## İki ARDIŞIK ölçek (kuralın gerçek hâli): P = tek ölçek²  (ölçekler bağımsız varsayımı)")
    say(f"  {'σ':>6s} {'sağlam n=5':>12s} {'sağlam n=10':>13s} {'eski n=5':>10s}")
    for sd in SIGMAS:
        r = []
        for fn, n in ((abandon_robust, 5), (abandon_robust, 10), (abandon_old, 5)):
            m, lo, hi, s = draw(0.0, sd, n, args.sims)
            p = float(fn(m, lo, hi, s).mean())
            r.append(100 * p * p)
        say(f"  {sd:6.2f} {r[0]:11.1f}% {r[1]:12.1f}% {r[2]:9.1f}%")

    table("YANLIŞ POZİTİF: gerçek etki sıfırken 'başarılı' ilan etme", {"başarı": success},
          mu=0.0, sims=args.sims)
    say("  YORUM: her σ'da %5'in ALTINDA (en kötü σ=6.0, n=5: ~%1.2) ve n artınca düşüyor.")

    say("\n## Gerçek etki EŞİKTE (μ = +5): başarı ilan etme gücü")
    say(f"  {'σ':>6s} {'n=5':>8s} {'n=10':>8s}")
    for sd in SIGMAS:
        r = [100 * float(success(*draw(5.0, sd, n, args.sims)).mean()) for n in (5, 10)]
        say(f"  {sd:6.2f} {r[0]:7.1f}% {r[1]:7.1f}%")
    say("  YORUM: tam eşikte güç ~%50 ve seed sayısıyla DÜZELMEZ — bağlayıcı olan nokta tahmini")
    say("  şartıdır (v01 O9'un yapısal sonucu, v02'de de geçerli). Eşik gevşetilmiyor.")

    # ---- BAŞARISIZ-SEED senaryoları: sınavın Gauss-dışı kör noktası ---------------------------
    say("\n## BAŞARISIZ-SEED senaryoları (sansürlü / yavaş) — vazgeçmenin İKİ yolu, beklentiler ASSERTION olarak")
    say("  Vazgeçme kuralı VE kesinlik ölçütü `s̄`'nin BAŞARISIZ-SEED'E DUYARLI üst sınırına bakar (`src.stats.ub95`):")
    say("  ub95 = min( (100·n_h + Σ sabit)/n , (n_h × hızlı seed'lerin GA üst sınırı + Σ sabit)/n ).")
    say("  'sabit' = 1x bütçede hedefe ulaşamayan seed'ler: sansürlü (ikame ≈ −150) ve yavaş (2x eğrisinden ölçülen).")
    say("  Bunları düz t-aralığına sokmak aralığı [−113, +53]'e şişirip kuralı ulaşılamaz yapıyordu (2. denetim);")
    say("  yalnızca sansürsüz ALT örneğe bakmak 3/5 sansürde (2 seed, t(1) = 12.7) en net başarısızlığı terk")
    say("  edilemez kılıyordu (3. denetim); yavaş seed'i sıradan gözlem saymak her yavaş-seed'li kampanyayı")
    say("  terminal 'yetersiz kesinlik'e sürüyordu (4. denetim). Aritmetik tavan HER durumda uygulanır.")
    say("  'tek' = tek ölçek yolu (v01 B10: satır 'başarısız' + hak kullanılmamış);")
    say("  'iki' = iki ardışık ölçek yolu. Oracle kazancı her senaryoda 0.5 (< 2).")
    say(f"  {'senaryo':34s} {'satır':>26s} {'s̄':>8s} {'düz GA üst':>11s} {'ub95':>8s} {'tek':>6s} {'iki':>6s}")
    CENS = -149.82                                    # 2x bütçe ikamesinin puan karşılığı
    SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)

    def tab(vals, cens=(), slow=()):
        """Senaryoyu seed KİMLİĞİYLE anahtarlı tabloya çevirir (`src.stats` liste kabul etmez)."""
        return {sd_: Obs(v, "sansürlü" if i in cens else "yavaş" if i in slow else "hızlı")
                for i, (sd_, v) in enumerate(zip(SEEDS, vals))}

    scen = [
        # (etiket, s_i, sansürlü konumlar, yavaş konumlar, tek ölçekte terk?, iki ölçekte terk?, SATIR)
        ("4 başabaş + 1 sansür", [0.2, -0.1, 0.3, 0.0, CENS], [4], [], True, True, "başarısız"),
        # başarısız seed İLK anahtarda (s11): sonuç seed'in HANGİ seed olduğuna bağlı olmamalı. Tur 3'te
        # hayatta kalan konum-sınıfı mutant (ub95'te "ilk n_h değer hızlıdır") yalnız burada görünür.
        ("1 sansür (s11'de) + 4 başabaş", [CENS, 0.2, -0.1, 0.3, 0.0], [0], [], True, True, "başarısız"),
        ("1 yavaş (s11'de, 1.5×) + 4 başabaş", [-56.0, 0.2, -0.1, 0.3, 0.0], [], [0], True, True, "başarısız"),
        # s̄ hedefe ulaşamayan seed'i İÇERİR: 4 × +%7 + 1 sansür -> s̄ ≈ −24, satır "başarısız".
        # v01 B10: hak kullanılmazsa terk (oracle şartı sağlanıyorsa); sahibin diğer yolu S1'dir.
        ("4 × +%7 + 1 sansür", [7.0, 7.1, 6.9, 7.05, CENS], [4], [], True, True, "başarısız"),
        ("4 × +%30 + 1 sansür", [30.0, 30.1, 29.9, 30.05, CENS], [4], [], True, True, "başarısız"),
        ("3 başabaş + 2 sansür", [0.2, -0.1, 0.3, CENS, CENS], [3, 4], [], True, True, "başarısız"),
        # 3. denetimin bulduğu delik: 2 sansürsüz seed'in aralığı t(1) = 12.7 ile patlıyor.
        ("2 geniş + 3 sansür", [0.5, -0.5, CENS, CENS, CENS], [2, 3, 4], [], True, True, "başarısız"),
        # 4. denetim (A): tavan HER durumda — iki hızlı seed'in t(1) aralığı +303'e çıkıyor.
        ("2 çok geniş hızlı + 3 sansür", [-62.5, -125.0, CENS, CENS, CENS], [2, 3, 4], [], True, True, "başarısız"),
        ("2 yavaş + 3 sansür", [-62.5, -125.0, CENS, CENS, CENS], [2, 3, 4], [0, 1], True, True, "başarısız"),
        # 4. denetim (C): yavaş seed bir yöntem başarısızlığıdır, varyans değil.
        ("4 başabaş + 1 yavaş (1.5×)", [0.2, -0.1, 0.3, 0.0, -56.0], [], [4], True, True, "başarısız"),
        ("4 başabaş + 1 yavaş (1.99×)", [0.2, -0.1, 0.3, 0.0, -148.75], [], [4], True, True, "başarısız"),
        ("4 × +%7 + 1 yavaş (1.5×)", [7.0, 7.1, 6.9, 7.05, -56.0], [], [4], True, True, "başarısız"),
        ("9 × −%8 + 1 yavaş (1.9×), n=10", [-8.0 + 0.02 * i for i in range(9)] + [-140.0], [], [9], True, True, "başarısız"),
        # "nötr" satırı tek ölçek yoluna GİRMEZ (o yol yalnızca "başarısız"tan açılır).
        ("başarısız seed yok, başabaş", [0.2, -0.1, 0.3, 0.0, 0.1], [], [], False, True, "nötr"),
        ("başarısız seed yok, −%8", [-8.0, -8.1, -7.9, -8.05, -7.95], [], [], True, True, "başarısız"),
        # etki eşiğin üstünde kurulmuş: hiçbir yolda terk edilemez
        ("başarısız seed yok, +%7", [7.0, 7.1, 6.9, 7.05, 6.95], [], [], False, False, "başarılı"),
        # 1x SINIRI (belgelenmiş tek durum süreksizliği): sınırın hemen içinde ulaşan seed aralığa
        # girer (kesinlik ölçütü tetiklenir), hemen dışında ulaşan sabit sayılır.
        ("4 × +%10 + 1 hızlı −24 (0.99×)", [10.0, 10.5, 9.5, 10.2, -24.0], [], [], False, False, "yüksek varyans / belirsiz"),
        ("4 × +%10 + 1 yavaş −26 (1.01×)", [10.0, 10.5, 9.5, 10.2, -26.0], [], [4], False, True, "belirsiz"),
        # İKİ sınır seed'i, n = 10, hızlı seed'ler ~+%35 (gerçekçi aralığın kenarı): satır 0 yalnızca
        # SANSÜRLÜ seed'leri sayar (ön-kayıtlı anlam kararı), dolayısıyla iki yavaş seed ile iki
        # sansürlü seed farklı satır verir. Tur 3 buldu: metin "2× sınırında süreksizlik YOKTUR"
        # diyordu; tek başarısız seed'de doğru, ikide değil. Artık metin de tablo da bunu söylüyor.
        ("8 × ~+%35 + 2 yavaş (1.99×), n=10", [28.0, 41.0, 33.0, 38.0, 34.0, 36.0, 35.0, 37.0, -148.75, -148.75],
         [], [8, 9], False, True, "nötr"),
        ("8 × ~+%35 + 2 sansür, n=10", [28.0, 41.0, 33.0, 38.0, 34.0, 36.0, 35.0, 37.0, CENS, CENS],
         [8, 9], [], True, True, "başarısız"),
    ]
    NEG = {"M": -1.0, "D": -1.0}
    rows = {}
    for label, s_i, cens, slow, exp1, exp2, exp_row in scen:
        r = decide(tab(s_i, cens, slow), quality_per_sec=NEG)
        rows[label] = r
        assert r["row"] == exp_row, f"{label}: satır {r['row']!r} != beklenen {exp_row!r}"   # V12: satır da assertion
        r1 = abandon({"S0": r}, {"S0": 0.5}, escalation_used=False)   # hak girdisi ZORUNLU, yalnız anahtar kelime
        r2 = abandon({"S0": r, "S1": r}, {"S0": 0.5, "S1": 0.5}, escalation_used=True)
        say(f"  {label:34s} {r['row']:>26s} {r['mean']:8.2f} {r['ci95'][1]:11.2f} {r['ub95']:8.2f} "
            f"{('EVET' if r1['abandon'] else 'hayır'):>6s} {('EVET' if r2['abandon'] else 'hayır'):>6s}")
        assert r1["abandon"] is exp1, f"{label} tek ölçek: {r1['abandon']} != {exp1} — {r1['reason'][:110]}"
        assert r2["abandon"] is exp2, f"{label} iki ölçek: {r2['abandon']} != {exp2} — {r2['reason'][:110]}"
    # başarısız seed'li HİÇBİR senaryo kesinlik satırına düşmemeli (4. denetimin bulduğu çıkmaz)
    for label, r in rows.items():
        if "yavaş" in label or "sansür" in label:
            assert r["determinate"], f"{label}: satır {r['row']!r} — başarısız seed aralığı şişirip kesinlik ölçütünü tetikliyor"
    # başarısız seed'in HANGİ seed olduğu sonucu değiştirmez (s11'de ya da s55'te)
    for first, last in (("1 sansür (s11'de) + 4 başabaş", "4 başabaş + 1 sansür"),
                        ("1 yavaş (s11'de, 1.5×) + 4 başabaş", "4 başabaş + 1 yavaş (1.5×)")):
        assert abs(rows[first]["ub95"] - rows[last]["ub95"]) < 1e-9 and rows[first]["row"] == rows[last]["row"], \
            f"sonuç başarısız seed'in konumuna bağlı: {first} ub95 {rows[first]['ub95']:+.2f} vs {rows[last]['ub95']:+.2f}"
    say("  başarısız seed s11'de ya da s55'te: AYNI satır, AYNI ub95 — OK")
    # 2x sınırında SÜREKLİLİK: 1.99x'te ulaşan ile hiç ulaşamayan aynı sonucu verir
    a_, b_ = rows["4 başabaş + 1 yavaş (1.99×)"], rows["4 başabaş + 1 sansür"]
    assert a_["row"] == b_["row"] and abs(a_["ub95"] - b_["ub95"]) < 0.5, "2x sınırında süreksizlik"
    say("  2x sınırı: 1.99× (−148.75, yavaş) ile sansür (−149.82) AYNI satır, ub95 farkı "
        f"{abs(a_['ub95'] - b_['ub95']):.2f} puan — OK")
    say("  1x sınırı (belgelenmiş): −24 hızlı -> kesinlik ölçütü tetiklenir; −26 yavaş -> belirli 'belirsiz'.")
    # ub95 sansür sayısında MONOTON olmalı (eski kural değildi: k=3 < k=2 < k=4)
    base5 = [0.2, -0.1, 0.3, 0.0, 0.1]
    ubs = [ub95(tab(base5[:5 - k] + [CENS] * k, cens=range(5 - k, 5))) for k in range(5)]
    assert all(a > b for a, b in zip(ubs, ubs[1:])), f"ub95 sansürde monoton değil: {ubs}"
    say("  ub95, sansürlü seed sayısı k = 0..4 için: " + ", ".join(f"{u:+.1f}" for u in ubs) + "  (monoton — OK)")
    # aritmetik tavan HER durumda: tavansız biçim bu senaryoda +31.4 verirdi
    capped = rows["2 çok geniş hızlı + 3 sansür"]["ub95"]
    uncapped = (2 * ci95([-62.5, -125.0])[2] + 3 * CENS) / 5
    assert capped < -49 < 5 < uncapped, f"aritmetik tavan uygulanmıyor: ub95 = {capped:+.1f} (tavansız {uncapped:+.1f})"
    say(f"  aritmetik tavan: [−62.5, −125, C, C, C] için ub95 = {capped:+.1f} (tavansız biçim {uncapped:+.1f} verirdi) — OK")
    # [S0 başarılı, S1 başarısız] TERK EDİLMEZ (3. denetimin bulduğu regresyon)
    s0 = decide(tab([6.5, 7.0, 7.5, 6.8, 7.2]), quality_per_sec=NEG)
    s1 = decide(tab([-4.0, -3.5, -4.5, -3.8, -4.2]), quality_per_sec=NEG)
    for flag in (False, True):
        assert abandon({"S0": s0, "S1": s1}, {"S0": 0.5, "S1": 0.5}, escalation_used=flag)["abandon"] is False, \
            "iki ölçek kuralı yalnızca SON ölçeğe bakıyor"
    say("  [S0 'başarılı', S1 'başarısız'] -> terk YOK (iki ölçek kuralı her iki ölçeğe bakıyor) — OK")
    # Oracle headroom'u HER yolda bağlayıcı, ve ölçek BAŞINA (ölçek KİMLİĞİYLE eşlenir):
    rb = decide(tab([-8.0, -8.1, -7.9, -8.05, -7.95]), quality_per_sec=NEG)
    two = {"S0": rb, "S1": rb}
    assert abandon({"S0": rb}, {"S0": 5.0}, escalation_used=False)["abandon"] is False, "oracle tek ölçekte bağlamıyor"
    assert abandon(two, {"S0": 0.5, "S1": 5.0}, escalation_used=True)["abandon"] is False, "oracle S1'de bağlamıyor"
    assert abandon(two, {"S1": 0.5, "S0": 5.0}, escalation_used=True)["abandon"] is False, "oracle S0'da bağlamıyor"
    say("  oracle headroom'u (5.0 >= 2.0) tek ölçekte ve iki ölçeğin HER BİRİNDE terk etmeyi engelliyor — OK")

    # ---- BİRİNCİL-2'nin dört durumlu kuralı --------------------------------------------------
    say("\n## BİRİNCİL-2'nin (X↔H) dört durumlu kuralı — V12 bunu da taramak zorunda")
    sdd = 0.88 * math.sqrt(2 * (1 - 0.07))            # kollar arası rho = +0.07
    a2 = 0.05 / PRIMARY_FAMILY                        # Holm'un İLK adımı
    say(f"  fark sd'si {sdd:.2f} puan | MDE çıtası {MDE_PRIMARY2} puan (n'den bağımsız SABİT) | "
        f"GA düzeyi α={a2} (%97.5 GA)")
    say("  Durumlar `src.stats.contribution()` ÇAĞRILARAK sayılıyor (ilk hâl kuralı satır içinde")
    say("  yeniden yazıyordu; o zaman tarama kuralı değil kendi kopyasını sınıyordu).")
    STATES = ("katkı var, büyük", "katkı var, küçük", "katkı MDE kadar değil", "ölçemedik")
    say(f"  {'gerçek X−H':>11s} {'n':>3s} " + " ".join(f"{s[:16]:>17s}" for s in STATES))
    sims2 = min(args.sims, 20_000)                    # contribution() Python'da: örnek başına çağrı
    share: dict[tuple[float, int], dict[str, float]] = {}
    for mu in (0.0, 0.5, 1.0, MDE_PRIMARY2, 3.0):
        for n in (5, 10):
            x = np.random.default_rng(11).normal(mu, sdd, size=(sims2, n))
            counts = dict.fromkeys(STATES, 0)
            zero = {sd_: Obs(0.0) for sd_ in SEEDS[:n]}    # H = 0: X − H = çekilen fark (aynı seed'ler)
            for row in x:                                # bağlayıcı mod: ön-kayıtlı seed'ler + AÇIK durum
                xt = {sd_: Obs(v) for sd_, v in zip(SEEDS, row.tolist())}
                counts[contribution(xt, zero)["state"]] += 1      # bilinmeyen durum -> KeyError
            share[(mu, n)] = {k: v / sims2 for k, v in counts.items()}
            say(f"  {mu:+10.2f}p {n:3d} " + " ".join(f"{100*share[(mu, n)][s]:16.1f}%" for s in STATES))
    c5, c10 = share[(0.0, 5)]["ölçemedik"], share[(0.0, 10)]["ölçemedik"]
    say(f"  YORUM (HESAPLANAN değerlerden): gerçek katkı sıfırken 'ölçemedik' payı n=5'te %{100*c5:.1f},")
    say(f"  n=10'da %{100*c10:.1f}. MDE'nin baştan ilan edilmesinin sebebi bu. Seed artırmak birincil-2 için de")
    say("  doğru yönde çalışıyor; ama seed hakkını YALNIZCA birincil-1'in kesinlik ölçütü tetikler.")
    assert c10 < c5, "seed artırmak birincil-2'de 'ölçemedik' payını düşürmüyor"
    pw5, pw10 = (sum(share[(MDE_PRIMARY2, n)][s] for s in STATES[:2]) for n in (5, 10))
    say(f"  Gerçek katkı tam MDE çıtasında ({MDE_PRIMARY2}) 'katkı var' oranı: n=5 %{100*pw5:.1f}, n=10 %{100*pw10:.1f}.")

    # ---- birincil-2'nin GÜÇ TABLOSU: ön-kayıttaki tablo BURADAN gelir ve hücre hücre denetlenir ----
    say("")
    say("## BİRİNCİL-2 güç tablosu (ön-kayıttaki tablo; `check_prereg_numbers.py` hücre hücre denetler)")
    say("  P('katkı var') = P(%97.5 GA alt sınırı > 0), fark sd'si 1.20 puan")
    say(f"  {'gerçek X−H':>11s} {'n=5':>8s} {'n=10':>8s} {'n=20':>8s}")
    for mu in (0.5, 1.0, 1.5, MDE_PRIMARY2, 2.0, 3.0):
        cells = []
        for n in (5, 10, 20):
            m, lo, hi, _ = draw(mu, sdd, n, args.sims, seed=23, alpha=a2)
            cells.append(100 * float((lo > 0).mean()))
        say(f"  MDE-GÜÇ {mu:+.2f} " + " ".join(f"{c:7.1f}%" for c in cells))

    if args.crosscheck:
        say("\n## Çapraz kontrol: vektörize kural == src.stats'ın genel yolu")
        for sd in (1.5, 5.0):
            m, lo, hi, s = draw(0.0, sd, 5, 4000, seed=7)
            fast = float(trip_precision(m, lo, hi, s).mean())
            slow = reachability(
                lambda x: decide({sd_: Obs(v) for sd_, v in zip(SEEDS, x)},
                                 quality_per_sec=1.0)["row"] == "yüksek varyans / belirsiz",
                0.0, sd, 5, sims=4000, seed=7)
            say(f"  σ={sd:.2f}: vektörize %{100*fast:.2f} | decide() yolu %{100*slow:.2f} "
                f"| fark {abs(fast-slow)*100:.2f} puan")

    p = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
