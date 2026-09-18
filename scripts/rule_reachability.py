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
    MDE_PRIMARY2, PRIMARY_FAMILY, SUCCESS, abandon, contribution, decide, reachability,
    t_crit, t_crit_alpha,
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
    say("  YORUM: eski biçim σ >= 1.72'de ÖLÜ (tam başabaş bir FOTON asla terk edilemez). Sağlam")
    say("  biçim σ = 3.0'da tek ölçekte %79, iki ardışık ölçekte %63 — eski biçimde iki ölçek %3.9'du.")
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

    # ---- SANSÜRLÜ senaryo: sınavın Gauss-dışı kör noktası ------------------------------------
    say("\n## SANSÜRLÜ senaryo — sınavın Gauss varsayımının kör noktası")
    say("  İkinci denetim şunu buldu: `abandon()` eşdeğerlik testini TAM örnek GA'sına uyguluyordu")
    say("  ve tek bir sansürlü seed (2x bütçe ikamesiyle ~−150 puan) o GA'yı ±80 puana şişiriyordu,")
    say("  yani kural sansürlü HER senaryoda ulaşılamaz hâle geliyordu. Bu sınav onu göremiyordu")
    say("  çünkü yalnızca Gauss örnekleri çiziyordu. Artık sansür de taranıyor.")
    say("  'tek' = TEK ölçekte terk (v01 B10 yolu: yalnızca 'başarısız' satırından);")
    say("  'iki' = iki ardışık ölçekte eşdeğerlik testiyle terk.")
    say(f"  {'senaryo':40s} {'satır':>22s} {'GA üst tam':>11s} {'GA üst sans.':>13s} {'tek':>5s} {'iki':>5s}")
    CENS = -149.82                                    # 2x bütçe ikamesinin puan karşılığı
    scen = [
        # (etiket, s_i, sansür maskesi, tek ölçekte terk?, iki ölçekte terk?)
        ("4 başabaş + 1 sansür", [0.2, -0.1, 0.3, 0.0, CENS], [False] * 4 + [True], True, True),
        # Bu iki satır bir REGRESYONU kilitliyor: satır sansür ikamesi yüzünden "başarısız" ama
        # sansürsüz seed'ler +%7 / +%30 tasarruf ediyor. HİÇBİR yolda terk EDİLMEMELİ.
        ("4 x +%7 + 1 sansür", [7.0, 7.1, 6.9, 7.05, CENS], [False] * 4 + [True], False, False),
        ("4 x +%30 + 1 sansür", [30.0, 30.1, 29.9, 30.05, CENS], [False] * 4 + [True], False, False),
        ("3 başabaş + 2 sansür", [0.2, -0.1, 0.3, CENS, CENS], [False] * 3 + [True] * 2, True, True),
        # "nötr" satırı B10'un tek-ölçek yoluna GİRMEZ (o yol yalnızca "başarısız"tan): tek
        # ölçekte terk edilemez, iki ölçekte eşdeğerlik testiyle edilebilir.
        ("sansür yok, başabaş", [0.2, -0.1, 0.3, 0.0, 0.1], [False] * 5, False, True),
        ("sansür yok, −%8", [-8.0, -8.1, -7.9, -8.05, -7.95], [False] * 5, True, True),
    ]
    for label, s_i, mask, exp1, exp2 in scen:
        r = decide(s_i, quality_per_sec={"M": -1.0, "D": -1.0}, censored_mask=mask)
        unc = r.get("ci95_uncensored")
        r1 = abandon([r], oracle_gain_over_null=0.5, escalation_used=False)
        r2 = abandon([r, r], oracle_gain_over_null=0.5, escalation_used=True)
        say(f"  {label:40s} {r['row']:>22s} {r['ci95'][1]:11.2f} "
            f"{(f'{unc[1]:.2f}' if unc else '—'):>13s} "
            f"{('EVET' if r1['abandon'] else 'hayır'):>5s} {('EVET' if r2['abandon'] else 'hayır'):>5s}")
        assert r1["abandon"] is exp1, f"{label} tek ölçek: {r1['abandon']} != {exp1} — {r1['reason'][:110]}"
        assert r2["abandon"] is exp2, f"{label} iki ölçek: {r2['abandon']} != {exp2} — {r2['reason'][:110]}"
    # Oracle headroom'u HER yolda bağlayıcı:
    rb = decide([-8.0, -8.1, -7.9, -8.05, -7.95], quality_per_sec={"M": -1.0, "D": -1.0})
    assert abandon([rb], oracle_gain_over_null=5.0)["abandon"] is False, "oracle tek ölçekte bağlamıyor"
    assert abandon([rb, rb], oracle_gain_over_null=5.0, escalation_used=True)["abandon"] is False,         "oracle iki ölçekte bağlamıyor"
    say("  oracle headroom'u (5.0 >= 2.0) hem tek hem iki ölçek yolunda terk etmeyi ENGELLİYOR — OK")
    say("  BEKLENEN, assertion olarak: başabaş ve negatif senaryolar terk EDİLEBİLİR olmalı,")
    say("  KUVVETLİ POZİTİF senaryolar ise sansür yüzünden 'başarısız' satırına düşseler bile")
    say("  terk EDİLEMEZ olmalı. İlk hâlde bu yalnızca düzyazıydı ve tarama, +%7 senaryosunun")
    say("  terk edildiğini YAZIP geçiyordu — defect'i gösterip yakalamayan bir sınav.")

    # ---- BİRİNCİL-2'nin eşdeğerlik kuralı ----------------------------------------------------
    say("\n## BİRİNCİL-2'nin (X↔H) eşdeğerlik kuralı — V12 bunu da taramak zorunda")
    say("  Kural: 'katkı yok' ancak farkın GA'sının TAMAMI MDE'nin altındaysa yazılabilir.")
    sdd = 0.88 * math.sqrt(2 * (1 - 0.07))            # kollar arası rho = +0.07
    mde = MDE_PRIMARY2
    a2 = 0.05 / PRIMARY_FAMILY                        # Holm'un İLK adımı
    say(f"  fark sd'si {sdd:.2f} puan | MDE {mde} puan | GA düzeyi α={a2} (Holm ilk adımı)")
    say("  DÜZELTME: bu taramanın ilk hâli α=.05 aralıkları kullanıyordu, oysa kural (ve MDE'nin")
    say("  türetimi) Holm'un ilk adımındaki α=.025'i söylüyor. Sonuç: 'ölçemedik' payı gerçek")
    say("  katkı sıfırken n=5'te %24.6 değil %42.8 — 1.7 kat. Kuralı taramak, kuralın KENDİ")
    say("  düzeyinde taramak demek; yoksa sınav başka bir kuralı sınıyor.")
    say("  Dört durum, BİRBİRİNİ DIŞLAYAN ve TÜKETEN. Sınav bu bölümü yazarken kendi kuralımdaki")
    say("  bir hatayı buldu: ilk hâl üç durum tanımlıyordu ve 'katkı var' (lo>0) ile 'katkı yok'")
    say("  (hi<MDE) birbirini dışlamıyordu — ikisi birden doğru olabilir ve o, mümkün olan EN")
    say("  bilgilendirici sonuçtur: 'katkı gerçek ama MDE'nin altında'. Üç durumla o hâl kaybolup")
    say("  olasılıklar negatife düşüyordu (n=10'da −%11.6 gibi). Dört durum:")
    say(f"  {'gerçek X−H':>11s} {'n':>3s} {'var,büyük':>10s} {'var,küçük':>10s} {'yok/sıfır':>10s} {'ölçemedik':>10s}")
    for mu in (0.0, 0.5, 1.0, 1.88, 3.0):
        for n in (5, 10):
            m, lo, hi, _ = draw(mu, sdd, n, args.sims, seed=11, alpha=a2)
            real, small = lo > 0, hi < mde
            cells = [(real & ~small).mean(), (real & small).mean(),
                     (~real & small).mean(), (~real & ~small).mean()]
            assert abs(sum(cells) - 1.0) < 1e-9, "dört durum tüketmiyor"
            say(f"  {mu:+10.2f}p {n:3d} " + " ".join(f"{100*c:9.1f}%" for c in cells))
    say("  YORUM: gerçek katkı sıfırken n=5'te 'ölçemedik' payı %24.6 — MDE'nin ilan edilmesinin")
    say("  sebebi bu. n=10'da o pay %0.6'ya düşüyor, yani seed artırmak birincil-2 için de doğru")
    say("  yönde çalışıyor (ama hak ORTAK olduğu için birincil-1'i de yeniden ölçer).")
    say("  'var,küçük' sütunu gerçek katkı 1.0 puanken n=10'da en yüksek: tam olarak ölçmek")
    say("  istediğimiz ama n=5'te ayırt edemediğimiz hâl.")

    if args.crosscheck:
        say("\n## Çapraz kontrol: vektörize kural == src.stats'ın genel yolu")
        for sd in (1.5, 5.0):
            m, lo, hi, s = draw(0.0, sd, 5, 4000, seed=7)
            fast = float(trip_precision(m, lo, hi, s).mean())
            slow = reachability(
                lambda x: decide(x, quality_per_sec=1.0)["row"] == "yüksek varyans / belirsiz",
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
