"""Ön-kayıttaki sayıları ARTIFACT'LARA karşı doğrular — elle tutarlılık yerine mekanik kontrol.

    .venv\\Scripts\\python scripts/check_prereg_numbers.py

NEDEN BU BETİK VAR. İki bağımsız denetim, ön-kayıttaki sayı hatalarımı buldu ve hepsi aynı sınıftan:
bir sayı bir betikten okunup metne yazıldı, sonra betik değişti ve metin güncellenmedi. Somut
örnekler:

  * V9'un güç tablosu **kaldırılmış** mutlak sd tripwire'ının sayılarını taşıyordu; kötümser
    sütunlarda 32.2 puan sapma (σ=5, +%10 için %58.4 yazıyordu, gerçek %90.6).
  * İkincil ölçüt tablosundaki kol başına süreler (26.79 / 22.58 dk) hiçbir artifact'ta yoktu ve
    aralarındaki fark (+4.21 dk) iki satır aşağıdaki eşleştirilmiş farkla (+4.06 dk) çelişiyordu —
    eşleştirilmiş bir tasarımda imkânsız.
  * `L_aux`, `sd(s_i)`, duvar saati ve MDE değerleri baseline'lar yeniden koşulunca değişti ve
    metnin bazı yerlerinde eski hâlleri kaldı.

Elle tarama bu sınıfı yakalamıyor: belge 1400 satır ve aynı sayı beş yerde geçiyor. Bu betik
artifact'ları OKUYUP metinde o sayının geçtiği her yeri denetler. Tag'den önce koşması zorunlu
(ön-kayıt V10'un "karar koşularından önce" listesinde).

Kasıtlı TARİHSEL atıflar beyaz listede: "Sapmalar" ve sadakat kontrolü bölümleri eski değerleri
(2.9082 gibi) bilerek anıyor ve onları "eski sayı kalmış" diye işaretlemek yanlış olur.
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from dataclasses import replace  # noqa: E402

from src.config import PRESETS  # noqa: E402
from src.flops import train_flops_per_token  # noqa: E402

PREREG = "PREREGISTRATION_V02.md"
SEEDS = [11, 22, 33, 44, 55]

# Metinde geçmesi YASAK olan eski değerler ve nerede geçmesine izin verildiği.
# Anahtar: aranan dize. Değer: o dizeye izin veren bağlam parçaları (biri satırda varsa sorun yok).
FORBIDDEN = {
    "2.9082": ["Sapmalar", "dondurulmuş → yeniden koşum", "maskesiz", "superseded", "ilk küme"],
    "2.9646": [],                  # eski L_aux: hiçbir yerde kalmamalı
    "sd(s_i) = 0.75": [],          # eski ölçülen sd
    # V11a'nın oturum karşılaştırma tablosu 1. oturumun sayılarını BİLEREK anıyor
    # (iddia tam olarak "aynı nicelik iki oturumda 11 puan farklı"); o satır meşru.
    "+6.25 dk": [".0115"],
    "%+30.0": ["oturum", "1 |"],   # eski duvar saati yüzdesi (V11a'da oturum karşılaştırması meşru)
    "27.12 dk": [],
    "20.83 dk": [],
    "26.79 dk": [],
    "22.58 dk": [],
    "%51.04": ["süre artışı"],     # süre artışı olarak meşru, tok/s olarak değil
    # Bir ondalıklı biçim de yasak: belge "%33.8" yazıyordu ve iki ondalıklı deseni
    # arayan ilk hâl onu kaçırdı — denetlediği hata sınıfının aynısı.
    # V13 bu guard'ın kendi tarihçesini anlatırken deseni BİLEREK anıyor ("guard" bağlamı).
    "%33.79": ["guard"], "%33.8": ["guard"],     # ilk kümenin tok/s düşüşü
    # "%51.0" birincil-2 güç tablosunda MEŞRU bir hücre de olabilir ("puan |" bağlamı).
    "%51.0": ["süre artışı", "puan |"], "%30.4": ["oturum", "1 |"],
    "13 test": [],                 # eski test sayımı
    "1 + 2 + 10": [],
    "MDE'si ≈ **+1.5": [],
    "+1.3 puan": [],
    "0.87 puan": ["Monte Carlo"],  # MC ortalaması olarak anılması meşru
    "%13.56": [],                  # ilk kümenin FLOP tasarrufu
}

bad: list[str] = []
ok: list[str] = []


def check(cond: bool, msg: str) -> None:
    (ok if cond else bad).append(msg)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", default=os.path.join(ROOT, PREREG),
                    help="denetlenecek belge; mutasyon sınavı bozulmuş bir KOPYA verir (V13)")
    args = ap.parse_args()
    text = open(args.prereg, encoding="utf-8").read()
    lines = text.split("\n")

    print("# Ön-kayıt sayı denetimi\n")

    # ---- 1) yasaklı eski değerler --------------------------------------------------------------
    print("## 1. Kaldırılmış/eskimiş değerler metinde kalmış mı?")
    for needle, allowed in FORBIDDEN.items():
        hits = [(i + 1, l) for i, l in enumerate(lines) if needle in l]
        flagged = [(i, l) for i, l in hits if not any(a in l for a in allowed)]
        if flagged:
            for i, l in flagged:
                bad.append(f"satır {i}: eski değer {needle!r} -> {l.strip()[:95]}")
        else:
            ok.append(f"{needle!r}: temiz ({len(hits)} meşru atıf)")
    print(f"  {len(FORBIDDEN)} desen tarandı, {len([b for b in bad])} sorun")

    # ---- 2) L* artifact'la uyuşuyor mu ---------------------------------------------------------
    print("\n## 2. L* ve türevleri artifact'larla uyuşuyor mu?")
    ls = json.load(open(os.path.join(ROOT, "experiments/foton-v02-prereg/lstar.json"), encoding="utf-8"))
    check(f"L* = {ls['lstar']}" in text, f"L* = {ls['lstar']} metinde")
    # "Bir yerde geçiyor mu" yetmez: aynı nicelik belgede birden çok kez yazılı ve kopyalardan
    # BİRİ bozulursa alt-dize kontrolü yine geçer. Her geçiş denetlenir; tarihsel atıflar
    # (ilk kümenin maskesiz L*'i) bağlamıyla beyaz listededir.
    for i, l in enumerate(lines, 1):
        for m in re.finditer(r"L\\?\*`? = ([0-9]\.[0-9]+)", l):      # `L*` ya da Markdown kaçışlı `L\*`
            historical = any(w in l for w in ("İlk küme", "maskesiz", "verdi"))
            check(m.group(1) == str(ls["lstar"]) or historical,
                  f"satır {i}: L* = {m.group(1)} (artifact {ls['lstar']})")
    check(f"{ls['at80_mean']:.6f}" in text, f"L* ortalaması {ls['at80_mean']:.6f} metinde")
    check(f"{ls['at80_sd']:.6f}" in text, f"L* sd {ls['at80_sd']:.6f} metinde")
    per = [ls["at80_per_seed"][str(s)] for s in SEEDS]
    check(all(f"{v:.6f}" in text for v in per), "beş seed'in val@3500 değeri metinde")
    check(abs(st.mean(per) - ls["at80_mean"]) < 1e-9, "at80_mean gerçekten beşin ortalaması")

    # ---- 3) referans sayılar -------------------------------------------------------------------
    print("\n## 3. F_base, FLOP/token, L_aux, sd(s_i)")
    ref = json.load(open(os.path.join(ROOT, "experiments/foton-v02-prereg/baseline_reference.json"),
                         encoding="utf-8"))
    ft = {a: train_flops_per_token(replace(PRESETS[p][0], vocab_size=32768))
          for a, p in (("m", "foton-m"), ("d", "foton-d"))}
    check(f"{ft['m']:,}".replace(",", ".") in text, f"M FLOP/token {ft['m']:,} metinde (nokta ayırıcı)")
    check(f"{ft['d']:,}".replace(",", ".") in text, f"D FLOP/token {ft['d']:,} metinde")
    check(f"L_aux = {ref['auxiliary']['l_aux']}" in text, f"L_aux = {ref['auxiliary']['l_aux']} metinde")
    for i, l in enumerate(lines, 1):
        for m in re.finditer(r"L_aux`? = ([0-9]\.[0-9]+)", l):
            check(m.group(1) == str(ref["auxiliary"]["l_aux"]),
                  f"satır {i}: L_aux = {m.group(1)} (artifact {ref['auxiliary']['l_aux']})")
    fb = [r["f_base"] for r in ref["reference"]["per_arm"]["m"]]
    check(f"{st.mean(fb):.6e}".replace("e+16", "e16") in text
          or f"{st.mean(fb):.6e}" in text, f"F_base,i ortalaması {st.mean(fb):.6e} metinde")

    # ---- 4) duvar saati: tablo hücreleri eşleştirilmiş farkla tutarlı mı ----------------------
    print("\n## 4. Duvar saati tablosu kendi içinde tutarlı mı?")
    ex = {a: st.mean(r["sec_exact"] for r in ref["auxiliary"]["per_arm"][a]) for a in ("m", "d")}
    diff = ref["auxiliary"]["paired_exact"]["mean_sec"]
    check(abs((ex["m"] - ex["d"]) - diff) < 1.0,
          f"ortalamaların farkı ({(ex['m']-ex['d'])/60:.2f} dk) == eşleştirilmiş fark ({diff/60:.2f} dk)")
    for a in ("m", "d"):
        cell = f"{ex[a]/60:.2f} dk"
        check(cell in text, f"{a} kolunun hedefe süresi {cell} metinde")
    check(f"{diff/60:+.2f} dk".replace("+", "+") in text, f"eşleştirilmiş fark {diff/60:+.2f} dk metinde")
    thr = ref["auxiliary"]["throughput"]
    check(f"%{thr['tok_s_reduction_pct']:.2f}" in text, f"tok/s düşüşü %{thr['tok_s_reduction_pct']:.2f} metinde")
    check(f"%{thr['time_increase_pct']:.2f}" in text, f"süre artışı %{thr['time_increase_pct']:.2f} metinde")

    # ---- 5) güç tablosu calibration.log ile hücre hücre ---------------------------------------
    print("\n## 5. V9 güç tablosu calibration.log ile aynı mı?")
    log = open(os.path.join(ROOT, "experiments/foton-v02-prereg/calibration.log"), encoding="utf-8").read()
    # log satırları: "  +7.0%   5      91.6%      0.1%      8.3%     3.50p"
    cells: dict[tuple[str, float, int], float] = {}
    block = None
    for line in log.split("\n"):
        m = re.match(r"\s*---\s*sd = ([\d.]+) puan", line)
        if m:
            block = float(m.group(1))
        m = re.match(r"\s*([+-][\d.]+)%\s+(\d+)\s+([\d.]+)%", line)
        if m and block is not None:
            cells[(m.group(1), block, int(m.group(2)))] = float(m.group(3))
    # V9 tablosunun BÜTÜN satırları denetlenir. İlk hâl `mu in ("+5.0","+7.0","+10.0")` ile
    # sınırlıydı — yani tam olarak güncellenmiş satırlara bakıyordu ve `0%` / `+2%` satırlarında
    # kalan beş eski hücreyi yapısal olarak göremiyordu. Bir denetim betiği, denetlediği hatanın
    # bulunduğu yere bakmıyorsa denetim değildir.
    #
    # Ayrıca `token in text` yerine SATIRIN TAMAMI karşılaştırılıyor: bir yüzdenin belgenin
    # herhangi bir yerinde geçmesi o hücrenin doğru olduğunu göstermez.
    tbl = [l for l in lines if l.startswith("| 0%") or l.startswith("| +2%")
           or l.startswith("| **+5%**") or l.startswith("| +7%") or l.startswith("| +10%")]
    check(len(tbl) == 5, f"V9 güç tablosunun 5 satırı bulundu (bulunan: {len(tbl)})")
    label = {"+0.0": "| 0%", "+2.0": "| +2%", "+5.0": "| **+5%**", "+7.0": "| +7%", "+10.0": "| +10%"}
    for mu, pref in label.items():
        row = next((l for l in tbl if l.startswith(pref)), None)
        if row is None:
            bad.append(f"V9 tablosunda {pref} satırı yok")
            continue
        got = [c.strip().strip("*") for c in row.split("|")[2:-1]]
        want = [f"%{cells[(mu, sd, n)]:.1f}" for sd in (0.88, 3.0, 5.0) for n in (5, 10)]
        check(got == want, f"V9 satırı {pref}: {len(want)} hücre calibration.log ile aynı"
                           + ("" if got == want else f" (belge {got} != log {want})"))

    # ---- 6) çiviler ----------------------------------------------------------------------------
    # ---- 5b) birincil-2'nin güç tablosu ve dört durum dağılımı: reachability.log'dan -----------
    print("\n## 5b. Birincil-2 güç tablosu reachability.log ile aynı mı?")
    # Üçüncü denetim buldu: bu tablo HİÇBİR guard'a bağlı değildi ve `+1.88` satırı yanlıştı (satır
    # +1.8 puanda hesaplanıp +1.88 diye etiketlenmişti). Rapor bu tabloyu birebir basmak zorunda.
    rlog = open(os.path.join(ROOT, "experiments/foton-v02-prereg/reachability.log"), encoding="utf-8").read()
    mde_rows = re.findall(r"MDE-GÜÇ ([+-][0-9.]+)\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%", rlog)
    check(len(mde_rows) == 6, f"reachability.log'da 6 MDE-GÜÇ satırı (bulunan: {len(mde_rows)})")
    for mu, a, b, c in mde_rows:
        want = f"| {mu} puan | %{a} | %{b} | %{c} |"
        check(want in text, f"birincil-2 güç satırı {mu}: {want}")
    m = re.search(r"'ölçemedik' payı n=5'te %([0-9]+\.[0-9]),\s*n=10'da %([0-9]+\.[0-9])", rlog)
    check(m is not None, "reachability.log'da 'ölçemedik' payı YORUM satırı")
    if m:
        check(f'"ölçemedik" %{m.group(1)}' in text, f"'ölçemedik' payı n=5 %{m.group(1)} metinde")
        check(f'n=10\'da "ölçemedik" %{m.group(2)}' in text, f"'ölçemedik' payı n=10 %{m.group(2)} metinde")
        # HER geçiş denetlenir: sayı belgede iki kez yazılı ve mutasyon sınavı (N18) kopyalardan
        # yalnızca biri bozulduğunda alt-dize kontrolünün diğerinden geçtiğini gösterdi.
        for i, l in enumerate(lines, 1):
            for mm in re.finditer(r'"ölçemedik" %([0-9]+\.[0-9])', l):
                check(mm.group(1) in (m.group(1), m.group(2)),
                      f"satır {i}: \"ölçemedik\" %{mm.group(1)} (log: %{m.group(1)} / %{m.group(2)})")

    # ---- 5d) başarısız-seed senaryolarının metindeki sayıları: reachability.log'dan -------------
    print("\n## 5d. Başarısız-seed senaryo sayıları reachability.log ile aynı mı?")
    um = lambda x: x.replace("-", "−")               # noqa: E731  belge Unicode eksi kullanıyor
    scen = {}
    for l in rlog.split("\n"):
        ms = re.match(r"^  (\S.*?)\s{2,}(\S.*?)\s+(-?\d+\.\d\d)\s+(-?\d+\.\d\d)\s+(-?\d+\.\d\d)\s+(EVET|hayır)\s+(EVET|hayır)$", l)
        if ms:
            scen[ms.group(1)] = dict(row=ms.group(2), mean=ms.group(3), ub=ms.group(5),
                                     tek=ms.group(6), iki=ms.group(7))
    check(len(scen) >= 16, f"reachability.log'da >= 16 başarısız-seed senaryosu (bulunan: {len(scen)})")

    def scen_check(label, template):
        r = scen.get(label)
        check(r is not None, f"senaryo log'da: {label}")
        if r:
            ub = um(r["ub"]) if r["ub"].startswith("-") else "+" + r["ub"]
            want = template.format(mean=um(r["mean"]), ub=ub, row=r["row"])
            check(want in text, f"senaryo '{label}' metinde: {want!r}")

    scen_check("4 × +%30 + 1 sansür", '`s̄ = {mean}`, satır "{row}", `ub95 = {ub}`')
    scen_check("4 başabaş + 1 yavaş (1.5×)", '`s̄ = {mean}`,\n"{row}", `ub95 = {ub}`')
    scen_check("4 × +%10 + 1 yavaş −26 (1.01×)", 'belirli bir "{row}"dir (`ub95 = {ub}`)')
    for label, tek, iki in (("4 × +%10 + 1 hızlı −24 (0.99×)", "hayır", "hayır"),
                            ("4 × +%10 + 1 yavaş −26 (1.01×)", "hayır", "EVET"),
                            ("4 başabaş + 1 yavaş (1.99×)", "EVET", "EVET")):
        r = scen.get(label, {})
        check((r.get("tek"), r.get("iki")) == (tek, iki),
              f"senaryo '{label}' terk sonucu metindeki gibi (tek {tek} / iki {iki})")
    m = re.search(r"ub95 farkı ([0-9]+\.[0-9]+) puan", rlog)
    check(m is not None, "reachability.log'da 2x sınırı satırı")
    if m:
        check(f"`ub95` farkı {m.group(1)} puan" in text, f"2x sınırındaki ub95 farkı metinde: {m.group(1)}")
    m = re.search(r"ub95 = (-[0-9.]+) \(tavansız biçim (\+[0-9.]+) verirdi\)", rlog)
    check(m is not None, "reachability.log'da aritmetik tavan satırı")
    if m:
        want = f"tavansız biçim **{m.group(2)}**, tavanlı biçim **{um(m.group(1))}**"
        check(want in text, f"aritmetik tavan örneği metinde: {want}")
    m = re.search(r"k = 0\.\.4 için: ([^ ]+(?:, [^ ]+){4})", rlog)
    check(m is not None, "reachability.log'da ub95 monotonluk satırı")
    if m:
        want = "k = 0..4 için " + um(m.group(1))
        check(want in text, f"ub95 monotonluk dizisi metinde: {want}")

    # ---- 5c) V13'ün mutasyon sayımı mutation_log.md ile aynı mı --------------------------------
    print("\n## 5c. V13'teki mutasyon sayımı kayıtla aynı mı?")
    mlog = open(os.path.join(ROOT, "experiments/foton-v02-prereg/mutation_log.md"), encoding="utf-8").read()
    mm = re.search(r"\*\*(\d+) mutant: (\d+) yakalandı, (\d+) kanıtlı eşdeğer, (\d+) hayatta kaldı\.\*\*", mlog)
    check(mm is not None, "mutation_log.md özet satırı okunabiliyor")
    if mm:
        n_all, n_caught, n_eq, n_alive = mm.groups()
        check(n_alive == "0", f"hayatta kalan mutant yok (kayıt: {n_alive})")
        want = f"**{n_all} mutant — {n_caught} yakalandı, {n_eq} kanıtlı eşdeğer, {n_alive} hayatta kaldı.**"
        check(want in text, f"V13 sayımı metinde: {want}")

    print("\n## 6. Metindeki SHA-256'lar diskle uyuşuyor mu?")
    import hashlib
    pins = re.findall(r"^([0-9a-f]{64})  (\S+)$", text, re.M)
    for h, rel in pins:
        p = os.path.join(ROOT, *rel.split("/"))
        if not os.path.exists(p):
            bad.append(f"çivi dosyası yok: {rel}")
            continue
        actual = hashlib.sha256(open(p, "rb").read()).hexdigest()
        check(actual == h, f"çivi {rel}")

    # ---- 7) placeholder ------------------------------------------------------------------------
    print("\n## 7. Doldurulmamış yer tutucular")
    ph = [(i + 1, l.strip()) for i, l in enumerate(lines) if "DOLDURULACAK" in l]
    for i, l in ph:
        print(f"  satır {i}: {l[:100]}")
    print(f"  {len(ph)} yer tutucu (V10'un FOTON config hash'i karar koşularından önce dolar)")

    # ---- özet ----------------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"GEÇEN: {len(ok)} kontrol")
    if bad:
        print(f"BAŞARISIZ: {len(bad)}")
        for b in bad:
            print(f"  ✗ {b}")
        sys.exit(1)
    print("SONUÇ: ön-kayıttaki sayılar artifact'larla tutarlı.")


if __name__ == "__main__":
    main()
