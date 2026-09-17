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
from src.stats import SUCCESS, decide, reachability, t_crit  # noqa: E402

SIGMAS = (0.75, 1.5, 1.72, 3.0, 4.28, 5.0, 6.0)
out: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    out.append(s)


def draw(mu: float, sd: float, n: int, sims: int, seed: int = 0):
    """(ortalama, GA alt, GA üst, örnek sd) — sims x n örnekten, vektörize."""
    x = np.random.default_rng(seed).normal(mu, sd, size=(sims, n))
    m = x.mean(axis=1)
    s = x.std(axis=1, ddof=1)
    h = t_crit(n - 1) * s / math.sqrt(n)
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
    say("ÖLÇÜLEN sd(s_i) = 0.75 puan (M↔D, L_aux); v01 O9 FOTON'un sd'sinin BÜYÜK olmasını bekliyor,")
    say("bu yüzden σ aralığı 6.0'a kadar taranıyor — kural orada da anlamlı davranmak zorunda.")

    table("TRIPWIRE: tetiklenme oranı", {"kesinlik": trip_precision, "eski sd>5": trip_abs_sd},
          mu=0.0, sims=args.sims)
    say("  YORUM — çarenin yönü: kesinlik ölçütünde n=5 -> n=10 oranı DÜŞÜYOR (çare kuralı")
    say("  gevşetiyor). Eski mutlak sd eşiğinde YÜKSELİYOR: tripwire'ın tek çaresi tripwire'ı daha")
    say("  olası yapıyordu ve tabloda 'n=10'da ateşlendi' satırı yoktu, yani kural kendi çıkmazını")
    say("  üretiyordu. Kesinlik ölçütü ölçülen σ = 0.75'te hiç tetiklenmiyor (%0.0).")

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
