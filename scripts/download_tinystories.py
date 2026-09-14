"""TinyStories veri setini HuggingFace'ten indirir (yalnızca standart kütüphane).

Neden TinyStories?  Küçük çocuk hikâyelerinden oluşan, dar kelime dağarcıklı sentetik bir
veri seti. ~10-30M parametreli modeller bile birkaç on dakikalık eğitimde tutarlı İngilizce
hikâyeler üretebiliyor -> mimari hatalarını hızlıca görmek için ideal bir "test tezgâhı".
V2 sürümü (yalnızca GPT-4 ile üretilmiş) V1'den daha temiz olduğu için onu kullanıyoruz.

Kullanım:  python scripts/download_tinystories.py
"""
import os
import sys
import time
import urllib.request

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = ["TinyStoriesV2-GPT4-valid.txt", "TinyStoriesV2-GPT4-train.txt"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TIMEOUT = 60  # sn; timeout olmadan kopan bir bağlantı sonsuza kadar asılı kalabilir


def _fetch(name: str, tmp: str, total: int) -> None:
    # Yarım kalmış indirmeye kaldığı yerden devam et (HTTP Range)
    start = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(BASE + name, headers={"Range": f"bytes={start}-"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        if start > 0 and r.status != 206:  # sunucu Range'i yok saydıysa baştan yaz
            start = 0
        with open(tmp, "ab" if start else "wb") as f:
            done, last = start, -1
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                pct = int(100 * done / total) if total else 0
                if pct // 10 != last:
                    print(f"{name}: {done / 1e6:,.0f}/{total / 1e6:,.0f} MB ({pct}%)", flush=True)
                    last = pct // 10


def download(name: str) -> None:
    dst = os.path.join(OUT_DIR, name)
    tmp = dst + ".part"
    req = urllib.request.Request(BASE + name, method="HEAD")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        total = int(r.headers.get("Content-Length", 0))
    if os.path.exists(dst) and os.path.getsize(dst) == total:
        print(f"[skip] {name} zaten var ({total / 1e6:.0f} MB)")
        return
    for attempt in range(1, 21):
        try:
            _fetch(name, tmp, total)
            break
        except OSError as e:  # timeout, bağlantı kopması vb. -> bekle ve kaldığı yerden sürdür
            print(f"[yeniden deneme {attempt}] {type(e).__name__}: {e}", flush=True)
            time.sleep(min(30, 2 * attempt))
    if os.path.getsize(tmp) != total:
        sys.exit(f"{name}: boyut uyuşmuyor ({os.path.getsize(tmp)} != {total})")
    os.replace(tmp, dst)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for n in FILES:
        download(n)
    print("tamam", file=sys.stderr)
