"""Tag'li ön-kayıt belgesi YALNIZCA sonuna eklenerek ("Sapmalar") değişmiş mi? Gövdesi donmuştur.

    .venv\\Scripts\\python scripts/check_prereg_frozen.py            # çıkış kodu 0 = gövde tag'dekiyle aynı

`foton-v02-prereg` tag'indeki `PREREGISTRATION_V02.md` ile çalışma ağacındaki dosya karşılaştırılır. İzin
verilen iki değişiklik vardır, ikisi de belgenin KENDİSİNDE önceden yazılıdır:

  1. Dosyanın SONUNA ekleme: "Sapmalar" son bölümdür ve "Ön-kayıttan her sapma ... buraya eklenir; yukarıdaki
     bölümler düzenlenmez" der.
  2. V10'daki tek yer tutucu satırın (`__DOLDURULACAK: FOTON config SHA-256__`) doldurulması: V10 "SHA-256'sı
     bu bölüme yazılır" der ve tag mesajı bunu tag'den sonra değişebilecek TEK alan olarak kaydeder.

Başka her fark (bir sayının, bir eşiğin, bir kuralın sessizce değişmesi) hata verir. Neden bir betik: ön-kaydın
değeri sonuçlardan ÖNCE var olmasından gelir; gövdenin değişmediği git geçmişinden okunabilir ama kimse
okumaz — mekanik denetim okur.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAG = "foton-v02-prereg"
DOC = "PREREGISTRATION_V02.md"
PLACEHOLDER = re.compile(r"^    __DOLDURULACAK: FOTON config SHA-256__.*$", re.M)
FILLED = re.compile(r"^    FOTON preset SHA-256 \(`experiments/foton-v02/preset_hashes\.json`\): [0-9a-f]{64}.*$", re.M)
TOKEN = "    <<V10 FOTON CONFIG SHA-256 SATIRI>>"


def normalize(text: str) -> str:
    """V10'un tek yer tutucu satırını (boş ya da dolu hâliyle) ortak bir belirtece çevirir."""
    n_ph, n_fl = len(PLACEHOLDER.findall(text)), len(FILLED.findall(text))
    if n_ph + n_fl != 1:
        raise ValueError(f"V10 yer tutucu satırı tam BİR kez olmalı (boş: {n_ph}, dolu: {n_fl})")
    return FILLED.sub(TOKEN, PLACEHOLDER.sub(TOKEN, text))


def check(tagged: str, current: str) -> list[str]:
    """Hata listesi (boş = temiz). `tagged` ve `current` dosya metinleridir."""
    try:
        t, c = normalize(tagged).rstrip("\n"), normalize(current)
    except ValueError as e:
        return [str(e)]
    if c.startswith(t):
        return []
    # ilk farkın yerini göster
    tl, cl = t.split("\n"), c.split("\n")
    i = next((k for k, (a, b) in enumerate(zip(tl, cl)) if a != b), min(len(tl), len(cl)))
    got = cl[i][:160] if i < len(cl) else "<dosya burada bitiyor>"
    return [f"tag'li gövde DEĞİŞMİŞ — ilk fark satır {i + 1}:\n   tag : {tl[i][:160] if i < len(tl) else '<son>'}\n   şimdi: {got}"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", default=os.path.join(ROOT, DOC), help="denetlenecek dosya (mutasyon sınavı bozulmuş bir kopya verir)")
    args = ap.parse_args()
    p = subprocess.run(["git", "show", f"{TAG}:{DOC}"], cwd=ROOT, capture_output=True)
    if p.returncode != 0:
        sys.exit(f"tag okunamadı: {TAG}")
    tagged = p.stdout.decode("utf-8")
    with open(args.prereg, encoding="utf-8") as f:
        current = f.read()
    bad = check(tagged, current)
    for b in bad:
        print("✗", b)
    if bad:
        sys.exit(1)
    added = len(normalize(current)) - len(normalize(tagged).rstrip("\n"))
    print(f"SONUÇ: gövde `{TAG}` tag'indekiyle aynı; sona eklenen {added} karakter (Sapmalar).")


if __name__ == "__main__":
    main()
