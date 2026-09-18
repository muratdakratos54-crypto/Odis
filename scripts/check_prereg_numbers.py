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
    "%51.0": ["süre artışı"], "%30.4": ["oturum", "1 |"],
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
