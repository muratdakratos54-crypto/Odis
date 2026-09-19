"""Koşu defteri: her koşu DENEMESİ için tek satır — PREREGISTRATION_V02 V11'in zorunlu kıldığı artefakt.

    .venv\\Scripts\\python scripts/run_ledger.py --reconstruct checkpoints/foton-v02 --report
    .venv\\Scripts\\python scripts/run_ledger.py --report
    .venv\\Scripts\\python scripts/run_ledger.py --record --run x-s11 --status iptal --reason "elle durduruldu"

Neden ayrı bir dosya ve neden SABİT bir yol: V11 "koşu defterindeki toplam/iptal/çöken koşu sayısı"nı
raporlamayı zorunlu kılıyor ama defterin kendisini hiçbir yerde tanımlamıyordu — ne dosya adı, ne şema,
ne de "ne zaman yazılır". Sonuç: defter YOKTU ve iptal edilen bir koşu diskte hiç iz bırakmıyordu.
Kuyruk yarım kalan koşuyu `<ad>.yarim-<n>` diye kenara alıyor; o klasör elle silindiğinde deneme tamamen
kaybolur, yani tam olarak sayılması istenen şey sayılamaz hale gelir. "5 koşu koştuk" demek için 7
denemeden 2'sini anmamak (seçici raporlama) ön-kaydın engellemek için yazıldığı şeydir; defteri
tanımsız bırakmak o engeli kâğıt üstünde bırakıyordu.

İkinci ve daha sinsi boşluk: SÜRDÜRÜLEN koşu. `src/train.py --resume true` checkpoint'ten devam eder,
`metrics.jsonl`'ı checkpoint adımına kadar kırpıp kaldığı yerden yazar ve bitince kaydı düz bir
"bitti"den AYIRT EDİLEMEZ. Oysa böyle bir koşunun duvar saati iki oturuma yayılır (arada bekleme,
soğuma, başka iş) ve V10(ii) gereği duvar saati ölçütüne GİRMEZ. Bu yüzden `sürdürüldü` ayrı bir
durumdur ve yalnızca deftere bakarak saptanabilir olmalıdır.

Sürdürmenin diskteki iki imzası (ikisi de aranıyor, biri yetse de):
  1. `time` alanı SÜREÇ başlangıcına göre sayılır -> yeni oturumda geriye atlar;
  2. `train_time` checkpoint'ten devam eder, `time` sıfırdan -> tek oturumda daima
     `time >= train_time` iken sürdürülen koşuda ilk satırlarda `time < train_time` olur.
(2) tek başına da yeterli: kırpma ilk oturumdan tek satır bile bırakmasa (1) görünmez olurdu.

Durum sözlüğü kapalıdır (`STATUSES`): `append()` sözlükte olmayan bir durumu REDDEDER. Serbest metinli
bir durum alanı ilk acelede "ok"/"fail" olur ve iptal ile çökme ayrımı — V11'in istediği ayrımın
tamamı — kaybolur.

Kayıtlar iki yoldan gelir ve `source` alanı hangisi olduğunu söyler: `kuyruk` (koşu biterken
`scripts/run_queue.py` yazdı) ve `yeniden-kurulum` (bu betik diskten çıkardı). Ayrım gereklidir çünkü
yeniden kurulum bir ÇIKARIMDIR: kenara alınmış bir klasörün çökme mi iptal mi olduğu diskte yazılı
değildir, `evidence` alanı da tam olarak hangi kanıta dayanıldığını ("DONE", "train.log-oom",
"kanit-yok", ...) bu yüzden taşır.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Yol SABİT: defterin yerini komut satırına bırakmak "hangi defter?" sorusunu doğurur ve V11'in
# istediği sayıları yeniden kurulamaz yapar. `--ledger` yalnızca testler ve prova için var.
LEDGER = os.path.join(ROOT, "experiments", "foton-v02-prereg", "run_ledger.jsonl")

SCHEMA = 1
# Kapalı durum sözlüğü. `sürdürüldü`, `bitti`nin ALTINDA değil YANINDA: bitmiş ama iki oturuma yayılmış
# koşu duvar saati ölçütüne girmez (V10-ii), o yüzden "bitti" diye sayılması yanlış olur.
STATUSES = ("bitti", "sürdürüldü", "çöktü", "iptal", "oom", "nan")
# "Çöken koşu" V11'de tek sayı olarak isteniyor; oom ve nan ayrı ayrı da tutulur çünkü sebebi farklı
# (biri VRAM bütçesi, biri sayısal kararsızlık) ve düzeltmesi farklı.
CRASHED = ("çöktü", "oom", "nan")
FIELDS = ("schema", "logged_at", "source", "run", "dir", "arm", "seed", "block", "status", "resumed",
          "sessions", "started", "ended", "steps", "git_head", "dirty", "src_sha256", "config_sha256",
          "evidence", "reason")

# Windows'ta Ctrl-C ile öldürülen çocuk süreç bu çıkış kodunu verir (STATUS_CONTROL_C_EXIT, 0xC000013A);
# subprocess kodu işaretsiz de döndürebilir. Bu kod ÇÖKME DEĞİL iptaldir ve ayrımı ancak burada kurabiliriz.
CTRL_C_CODES = (3221225786, -1073741510)


# --- defterin kendisi: ekleme ve okuma ---------------------------------------------------------------

def blank(**kw) -> dict:
    """Tüm alanları tanımlı, sırası sabit bir kayıt. Alan sırası sabit ki defter elle de okunabilsin
    (satırlar `git diff`te hizalı kalır) ve eksik alan sessizce "yok" sayılmasın."""
    rec = {k: None for k in FIELDS}
    rec.update(schema=SCHEMA, source="yeniden-kurulum", resumed=False, sessions=1, evidence="", reason="")
    for k, v in kw.items():
        if k not in FIELDS:
            raise ValueError(f"koşu defterinde '{k}' diye bir alan yok (şema {SCHEMA}): {sorted(FIELDS)}")
        rec[k] = v
    return rec


def append(rec: dict, path: str = LEDGER) -> dict:
    """Tek satır ekler (append-only) ve yazılan kaydı döndürür.

    Neden append-only: defter, kendisini ezebilen bir süreç tarafından yazılıyor. Yeniden yazma
    (oku-değiştir-yaz) bir çökmede defteri kısaltır, yani kaybı gizleyen şey kaybın kaydı olur.
    """
    rec = blank(**rec)                        # bilinmeyen alan ValueError: sessizce kaybolan alan olmasın
    if not rec["run"]:
        raise ValueError("koşu defteri kaydında 'run' zorunlu")
    if rec["status"] not in STATUSES:
        raise ValueError(f"bilinmeyen durum {rec['status']!r}; sözlük kapalı: {list(STATUSES)}")
    rec["schema"] = SCHEMA
    rec["logged_at"] = iso(time.time())
    if rec["dir"] is None:
        rec["dir"] = rec["run"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # Önceki süreç satırın ortasında öldüyse dosya "\n" ile bitmez. Kendi satırımızı doğrudan eklemek
    # iki kaydı tek bozuk satırda birleştirirdi: bozuk satır bozuk kalsın, bizimki sağlam olsun.
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_nl = f.read(1) != b"\n"
    else:
        needs_nl = False
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(("\n" if needs_nl else "") + json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _parse_line(line: str) -> dict | None:
    """Tek defter satırı -> kayıt, ya da çözülemiyorsa None.

    `read` ile `damaged` bu TEK tanımı paylaşmak zorunda: iki ayrı çözümleme ölçütü yazılırsa bir satır
    ikisinin de dışında kalabilir ve "kayıt 11 satır | bozuk satır 0" derken 12 satır olur — raporun
    tamlık iddiası tam olarak bu bölümlemeye dayanıyor.

    Sözlük OLMAYAN geçerli JSON (yarım yazılmış satırın sonu, ör. `123` ya da `null`) da bozuktur:
    `json.loads` ona hata vermez, ama `rec.get(...)` diye okunduğunda rapor AttributeError ile çöker.
    """
    try:
        rec = json.loads(line)
    except ValueError:
        return None
    return rec if isinstance(rec, dict) else None


def read(path: str = LEDGER) -> list[dict]:
    """Defteri okur; ÇÖZÜLEMEYEN satırları atlar (bkz. `damaged`). Dosya yoksa boş liste."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rec = _parse_line(line)
                if rec is not None:
                    out.append(rec)
    return out


def damaged(path: str = LEDGER) -> list[tuple[int, str]]:
    """Çözülemeyen satırlar [(satır no, ilk 80 karakter)].

    Raporun bunu YAZMASI şart: yarım yazılmış tek bir satır bir çökme kaydını yok edebilir ve
    "bozuk satır 0" satırı olmadan "toplam 12" sayısının tam olduğu söylenemez.
    """
    if not os.path.exists(path):
        return []
    bad = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if line.strip() and _parse_line(line) is None:
                bad.append((i, line.strip()[:80]))
    return bad


SET_ASIDE_SUFFIXES = (".yarim-", ".blok-iptal-")     # `run_queue.py`'nin kenara alma ekleri: <ad>.yarim-<n>, <ad>.blok-iptal-<n>


def base_run_name(dirname: str) -> str:
    """Kenara alınmış bir klasör adından KOŞU adını verir (`d-s11.blok-iptal-1` -> `d-s11`).

    `.blok-iptal-` eki V11a(iii) ile geldi (kesilen bloğun TAMAMI yeniden koşulur; bitmiş koşuları da kenara
    alınır). Bu ek tanınmazken ölçüldü: yeniden kurulum aynı denemeyi `d-s11.blok-iptal-1` diye YENİ bir koşu
    adıyla ikinci kez yazıyordu (2 blokluk bir senaryoda toplam 13 -> 16 satır, koşu adı 8 -> 11) — V11'in
    zorunlu kıldığı sayılar şişiyordu.
    """
    for suf in SET_ASIDE_SUFFIXES:
        head, sep, tail = dirname.rpartition(suf)
        if sep and tail.isdigit():
            return base_run_name(head)
    return dirname


def is_set_aside(dirname: str) -> bool:
    return base_run_name(dirname) != dirname


def attempt(rec: dict) -> tuple:
    """Bir DENEMENİN kimliği: (koşu adı, başlangıç anı). Klasör adı kimliğe GİRMEZ.

    Gerekçe ölçümle: `run_queue.py` yarım kalmış bir koşuyu `<ad>.yarim-<n>` diye kenara alıyor ve
    yeniden adlandırma mtime'ı KORUYOR. Klasör adı kimliğe girerse aynı deneme iki ayrı satır olur —
    kuyruk kancası yeniden adlandırmadan ÖNCE yazar (`dir="m-s11"`), yeniden kurulum SONRA görür
    (`dir="m-s11.yarim-1"`). Ölçülen sonuç: tek denemede `TOPLAM deneme 2 | ÇÖKEN 2`. Bunlar V11'in
    zorunlu kıldığı sayılar, yani defterin var olma sebebi olan sayılar şişiyordu.
    """
    return (rec.get("run"), rec.get("started"))


def key(rec: dict) -> tuple:
    """Deneme kimliği + durum. Aynı denemenin DURUM DEĞİŞİKLİĞİ yeni satır hak eder (V10(iii)):
    bir klasör ilk taramada "çöktü", sürdürülüp bitince "sürdürüldü" olur ve ikisi de deftere girer.
    Durum aynı kaldığı sürece (ör. koşan bir klasörün tekrar taranması) yeni satır eklenmez."""
    return attempt(rec) + (rec.get("status"),)


def append_missing(records: list[dict], path: str = LEDGER) -> list[dict]:
    """Defterde olmayan kayıtları ekler; yeniden kurulumu tekrar tekrar koşmak defteri şişirmesin.

    KUYRUK SATIRI OTORİTERDİR. Bir deneme için `source="kuyruk"` satırı varsa, yeniden kurulum o
    denemeye dokunmaz. Neden: kuyruk sürecin ÇIKIŞ KODUNU biliyor, yeniden kurulum bilmiyor. Aynı
    çöken koşuda kuyruk `çöktü`/`cikis-kodu-1`, yeniden kurulum ise `iptal`/`kanit-yok` çıkarıyor —
    yani `status`'u kimliğe koymak (V10(iii) için doğru olan) bu durumda çelişen bir İKİNCİ satır
    üretiyordu. Daha az bilgiye sahip taraf, daha çok bilgiye sahip tarafın kaydını bozmaz.

    Bu, gerçek durum değişikliğini engellemez: sürdürülüp biten bir koşu için kuyruk zaten kendi
    yeni satırını yazar (yeni deneme, yeni `started`).
    """
    existing = read(path)
    have = {key(r) for r in existing}
    from_queue = {attempt(r) for r in existing if r.get("source") == "kuyruk"}
    out = []
    for rec in records:
        if attempt(rec) in from_queue or key(rec) in have:
            continue
        out.append(append(rec, path))
        have.add(key(rec))
    return out


# --- diskten yeniden kurulum -------------------------------------------------------------------------

def iso(ts: float | None) -> str | None:
    """YEREL saat, ISO benzeri. Neden yerel: `queue.log` ve `train.log` yerel saat yazıyor; defteri
    UTC'ye çevirmek denetim sırasında satırların yan yana konmasını gereksizce zorlaştırır."""
    return None if ts is None else time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _sha256(path: str) -> str | None:
    """Dosyanın SHA-256'sı. `config.json` için V11 zorunlu kılıyor: iki kolun aynı ayarla koştuğu
    ancak bu sayının eşitliğiyle gösterilebilir."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def metric_rows(path: str) -> list[dict]:
    """`metrics.jsonl` satırları; çözülemeyen satırlar atlanır (çöken koşunun yarım son satırı —
    `run_queue.finished()` de aynı toleransı gösteriyor)."""
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    except UnicodeDecodeError:
        return []
    return out


def sessions_of(rows: list[dict]) -> tuple[int, str]:
    """(oturum sayısı, imza). Modül docstring'indeki iki imza:

    1. `time` süreç başlangıcına göre -> yeni oturumda geriye atlar;
    2. tek oturumda `time >= train_time` (time eval ve kayıt süresini de içerir), sürdürmede tersine döner.
    """
    n, sig = 1, ""
    prev = None
    for r in rows:
        t, tt = r.get("time"), r.get("train_time")
        if t is None:
            continue
        if prev is not None and t < prev - 1e-9:      # (1) saat geriye atladı: yeni süreç
            n += 1
            sig = "time-geri-atladi"
        elif tt is not None and t < tt - 1e-6:        # (2) kırpma (1)'i görünmez kılmış olabilir
            if n == 1:
                n = 2
            sig = sig or "time<train_time"
        prev = t
    return n, sig


def has_nan(rows: list[dict]) -> bool:
    """Kayıtlarda NaN var mı. NaN'lı bir koşu "bitti" sayılırsa zehirli bir eğri karar hesabına girer;
    `json` NaN'ı olduğu gibi yazıp okuduğu için diskte saptanabilir (v != v yalnızca NaN'da doğru)."""
    return any(isinstance(v, float) and v != v
               for r in rows for k, v in r.items() if k in ("loss", "ce", "val_loss", "grad_norm"))


def log_tail(path: str, nbytes: int = 65536) -> str:
    """`train.log`'un son `nbytes` baytı. Neden sadece son: çökme izi (traceback, OOM) dosyanın
    SONUNDADIR; 4375 adımlık bir koşunun logu megabaytlarca ve tamamını okumak boşa iş."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - nbytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def crash_kind(tail: str) -> str | None:
    """Log kuyruğundan çökme türü: 'oom' | 'çöktü' | None. OOM ayrı tutulur çünkü ön-kaydın VRAM
    bütçesiyle ilgilidir (`vram_limit_frac`), rastgele bir hata değildir."""
    low = tail.lower()
    if "out of memory" in low or "outofmemoryerror" in low:
        return "oom"
    if "traceback (most recent call last)" in low:
        return "çöktü"
    return None


def _arm_seed(prov: dict | None, cfg: dict | None, run: str) -> tuple[str | None, int | None, bool]:
    """(kol, seed, koşu_adından_çıkarıldı_mı). Önce provenance/config (yazılı kanıt), olmazsa koşu adı."""
    preset = seed = None
    args = (prov or {}).get("args") or []
    for flag, target in (("--preset", "preset"), ("--seed", "seed")):
        if flag in args:
            i = args.index(flag) + 1
            if i < len(args):
                if target == "preset":
                    preset = args[i]
                else:
                    seed = int(args[i]) if str(args[i]).lstrip("-").isdigit() else None
    if cfg and isinstance(cfg.get("train"), dict) and cfg["train"].get("seed") is not None:
        seed = cfg["train"]["seed"]
    arm = preset.replace("foton-", "") if preset else None
    if arm is None or seed is None:
        # Son çare: `<önek->[dmhx]-s<seed>` kalıbı. Çıkarım olduğu `evidence`/`reason`'a yazılır.
        m = re.search(r"(?:^|-)([dmhx])-s(\d+)$", run)
        if m:
            arm, seed = arm or m.group(1), seed if seed is not None else int(m.group(2))
            return arm, seed, True
    return arm, seed, False


def scan_run_dir(run_dir: str, block: str | None = None, exit_code: int | None = None,
                 running_grace_sec: float = 900.0) -> dict | None:
    """Bir koşu klasöründen kayıt çıkarır. KOŞMAKTA olduğu düşünülen klasör için None döner.

    "Koşuyor" koruması şart: bitmemiş ve çökme izi olmayan bir klasör, o an eğitim yazarken
    taranırsa "iptal" diye deftere girer ve defter kendi kendine yanlış bir iptal uydurur.
    Ölçüt: `metrics.jsonl` son `running_grace_sec` saniye içinde yazılmışsa koşuyor sayılır.
    Kenara alınmış (`.yarim-`) klasörde bu koruma gereksizdir — kuyruk onu zaten bırakmıştır.

    Durum önceliği (her biri `evidence` alanına yazılır):
      NaN > OOM > bitmiş(DONE/end) > traceback > çıkış kodu > kanıt yok (-> iptal)
    NaN en önde: NaN'lı ama "bitmiş" bir koşu karar hesabına girmemeli.
    """
    d = os.path.basename(os.path.normpath(run_dir))
    run = base_run_name(d)
    set_aside = is_set_aside(d)
    if not os.path.isdir(run_dir):
        # Klasör hiç oluşmamış (süreç açılışta öldü) ya da sonradan silinmiş. "Kanıt yok"tan AYRI bir
        # kanıt: klasör yoksa koşu tek satır bile yazamamıştır, varsa yazdı ama bitmedi demektir.
        arm, seed, _ = _arm_seed(None, None, run)
        return blank(run=run, dir=d, arm=arm, seed=seed, block=block, status="iptal",
                     evidence="klasor-yok", reason="koşu klasörü diskte yok: deneme hiçbir iz bırakmadı")
    prov = _json(os.path.join(run_dir, "provenance.json"))
    cfg = _json(os.path.join(run_dir, "config.json"))
    rows = metric_rows(os.path.join(run_dir, "metrics.jsonl"))
    tail = log_tail(os.path.join(run_dir, "train.log"))
    done = os.path.exists(os.path.join(run_dir, "DONE"))
    ended_row = any(r.get("type") == "end" for r in rows)
    m_mtime = _mtime(os.path.join(run_dir, "metrics.jsonl"))

    # "Koşuyor" ölçütü klasördeki EN YENİ dosyaya bakar, yalnızca metrics.jsonl'a değil: kuyruk
    # provenance.json'ı yazdıktan sonra `src/train.py` veri yükleyip derleyene kadar metrics.jsonl
    # HENÜZ YOKTUR (ölçülen açılış: provenance 21:43:58 -> ilk metrics satırı ~21:44:05, r-d-s11).
    # O pencerede yalnızca metrics'e bakılsa canlı koşu "kanıt yok -> iptal" diye deftere girer ve
    # defter, ekleme-yalnızca olduğu için kendi uydurduğu iptali bir daha silemez.
    live_mtime = max((t for t in (m_mtime, _mtime(os.path.join(run_dir, "provenance.json")),
                                  _mtime(os.path.join(run_dir, "config.json"))) if t is not None),
                     default=None)
    if not (done or ended_row) and not set_aside and crash_kind(tail) is None \
            and live_mtime is not None and time.time() - live_mtime < running_grace_sec:
        return None

    sessions, sig = sessions_of(rows)
    resumed = sessions > 1
    notes: list[str] = []
    if set_aside:
        notes.append(f"kuyruk yarım kalan koşuyu {d} olarak kenara aldı")

    kind = crash_kind(tail)
    if has_nan(rows):
        status, evidence = "nan", "metrics-nan"
        notes.append("metrics.jsonl'da NaN var: eğri karar hesabına giremez")
    elif kind == "oom":
        status, evidence = "oom", "train.log-oom"
    elif done or ended_row:
        status = "sürdürüldü" if resumed else "bitti"
        evidence = "DONE" if done else "metrics-end"
    elif kind == "çöktü":
        status, evidence = "çöktü", "train.log-traceback"
    elif exit_code not in (None, 0):
        # Kuyruk çıkış kodunu biliyor: Ctrl-C ile öldürülen süreç ÇÖKMEDİ, iptal edildi.
        status = "iptal" if exit_code in CTRL_C_CODES else "çöktü"
        evidence = f"cikis-kodu-{exit_code}"
        notes.append("kullanıcı kesmesi (Ctrl-C) çıkış kodu" if status == "iptal"
                     else f"çıkış kodu {exit_code}, log'da traceback yok")
    else:
        status, evidence = "iptal", "kanit-yok"
        notes.append("train.log'da çökme izi yok: iptal sayıldı, ama diskte İPTAL ile ÇÖKME "
                     "ayrımının kanıtı yoktur")

    if resumed:
        notes.append(f"{sessions} oturum ({sig}): duvar saati tek oturuma ait DEĞİL, "
                     "V10(ii) gereği duvar saati ölçütüne giremez")
    if block is None:
        notes.append("blok kimliği diskte yazılı değil (kuyruk dosyasındaki _block alanından gelir)")

    arm, seed, from_name = _arm_seed(prov, cfg, run)
    if from_name:
        notes.append("kol/seed koşu adından çıkarıldı (provenance.json'da yok)")
    steps = max((r.get("step", 0) for r in rows), default=None)
    started = _mtime(os.path.join(run_dir, "provenance.json")) or \
        _mtime(os.path.join(run_dir, "config.json")) or m_mtime
    ended = _mtime(os.path.join(run_dir, "DONE")) if done else m_mtime
    return blank(
        source="yeniden-kurulum", run=run, dir=d, arm=arm, seed=seed, block=block, status=status,
        resumed=resumed, sessions=sessions, started=iso(started), ended=iso(ended), steps=steps,
        git_head=(prov or {}).get("git_head"), dirty=bool((prov or {}).get("git_status")) if prov else None,
        src_sha256=(prov or {}).get("src_sha256"),
        config_sha256=_sha256(os.path.join(run_dir, "config.json")),
        evidence=evidence, reason="; ".join(notes))


def reconstruct(ckpt_dir: str, blocks: dict[str, str] | None = None,
                running_grace_sec: float = 900.0) -> tuple[list[dict], list[str]]:
    """(kayıtlar, koşuyor sayılıp atlanan klasörler). Kayıtlar başlangıç zamanına göre sıralı:
    defter kronolojik okunabilsin, çünkü V10'un blok sırası iddiası sıraya bakılarak denetlenir."""
    recs, skipped = [], []
    for name in sorted(os.listdir(ckpt_dir)):
        run_dir = os.path.join(ckpt_dir, name)
        if not os.path.isdir(run_dir):
            continue
        block = (blocks or {}).get(base_run_name(name))
        rec = scan_run_dir(run_dir, block=block, running_grace_sec=running_grace_sec)
        (recs.append(rec) if rec is not None else skipped.append(name))
    recs.sort(key=lambda r: (r["started"] or "", r["dir"]))
    return recs, skipped


def blocks_from_queue(path: str) -> dict[str, str]:
    """Kuyruk dosyasından {koşu adı: blok}. Blok diskte yazılı olmadığı için tek kaynağı budur
    (`scripts/make_campaign_queue.py` her koşuya `_block` yazar)."""
    q = _json(path) or {}
    return {r["name"]: r["_block"] for r in q.get("runs", []) if r.get("_block")}


# --- rapor -------------------------------------------------------------------------------------------

def counts(records: list[dict]) -> dict:
    """V11'in istediği sayılar. `toplam` DENEME sayısıdır, koşu sayısı değil: aynı koşu adı bir
    iptal + bir bitti olarak iki satır olabilir ve V10(iii) tam olarak bunu istiyor."""
    per = {s: sum(1 for r in records if r.get("status") == s) for s in STATUSES}
    # V11a(iii): kesilen bloğun BİTMİŞ koşuları da kenara alınır ve o denemeye `evidence = "blok-iptal"` taşıyan
    # bir durum-değişikliği satırı eklenir (defter append-only: eski "bitti" satırı geri alınamaz). Öyle bir
    # deneme duvar saati ölçütüne GİREMEZ — bloğu kesintisiz koşmadı. Bu dışlama yokken ölçüldü: 2 blokluk,
    # tek çökmeli bir senaryoda "uygun" 10 okunuyordu, doğrusu 8.
    cancelled = {attempt(r) for r in records if r.get("evidence") == "blok-iptal"}
    admissible = [r for r in records                      # duvar saati ölçütüne girebilen satırlar
                  if r.get("status") == "bitti" and not r.get("resumed") and r.get("block")
                  and attempt(r) not in cancelled]
    return {"toplam": len(records), **per, "çöken": sum(per[s] for s in CRASHED),
            "sürdürülen": sum(1 for r in records if r.get("resumed")),
            "blok_bilinmeyen": sum(1 for r in records if not r.get("block")),
            "duvar_saati_uygun": len(admissible), "blok_iptal": len(cancelled),
            "koşu_adı_sayısı": len({r.get("run") for r in records})}


def print_report(path: str = LEDGER) -> dict:
    recs = read(path)
    bad = damaged(path)
    c = counts(recs)
    print(f"## Koşu defteri — {os.path.relpath(path, ROOT) if path.startswith(ROOT) else path}")
    print(f"kayıt {len(recs)} satır | bozuk satır {len(bad)}"
          + ("  <- YARIM YAZILMIŞ SATIR VAR, sayılar eksik olabilir" if bad else ""))
    for i, text in bad:
        print(f"  [bozuk] satır {i}: {text}")
    # V11'in birebir istediği üç sayı, raporun en başında:
    print(f"\nTOPLAM deneme {c['toplam']} | İPTAL {c['iptal']} | ÇÖKEN {c['çöken']} "
          f"(çöktü {c['çöktü']} + oom {c['oom']} + nan {c['nan']})")
    print("durum dağılımı: " + ", ".join(f"{s} {c[s]}" for s in STATUSES))
    print(f"ayrı koşu adı {c['koşu_adı_sayısı']} | sürdürülen {c['sürdürülen']} "
          f"| blok kimliği bilinmeyen {c['blok_bilinmeyen']}")
    print(f"duvar saati karşılaştırmasına UYGUN satır {c['duvar_saati_uygun']} "
          f"(ölçüt: durum 'bitti' + tek oturum + blok kimliği belli, V10-ii/iv)")
    if recs:
        print(f"\n{'koşu':22s} {'kol':4s} {'seed':>5s} {'blok':6s} {'durum':11s} {'adım':>6s} "
              f"{'başlangıç':19s} {'kanıt':22s} kaynak")
        for r in recs:
            print(f"{(r.get('dir') or r.get('run') or ''):22s} {(r.get('arm') or '-'):4s} "
                  f"{(r.get('seed') if r.get('seed') is not None else '-')!s:>5s} "
                  f"{(r.get('block') or '-'):6s} {(r.get('status') or '-'):11s} "
                  f"{(r.get('steps') if r.get('steps') is not None else '-')!s:>6s} "
                  f"{(r.get('started') or '-'):19s} {(r.get('evidence') or '-'):22s} {r.get('source') or '-'}")
    # Gerekçeler tabloya sığmaz. Aynı gerekçe onlarca satırda tekrarlanabildiği için (ör. "blok kimliği
    # diskte yazılı değil") gerekçeye göre GRUPLANIR: tekrarlı liste raporu okunmaz yapıp asıl
    # gerekçeleri (iptal/çökme) gözden kaçırtır.
    groups: dict[str, list[str]] = {}
    for r in recs:
        if r.get("reason"):
            groups.setdefault(r["reason"], []).append(r.get("dir") or r.get("run") or "?")
    if groups:
        print("\ngerekçeler:")
        for reason, names in groups.items():
            who = names[0] if len(names) == 1 else f"{len(names)} satır ({', '.join(names)})"
            print(f"  - {who}: {reason}")
    return c


# --- kuyruğun ve elin kullandığı uç ------------------------------------------------------------------

def record_run_dir(run_dir: str, block: str | None = None, exit_code: int | None = None,
                   reason: str | None = None, source: str = "kuyruk",
                   only_if_new_attempt: bool = False, path: str = LEDGER) -> dict | None:
    """Koşu klasörünü tarayıp tek satır yazar — `scripts/run_queue.py`'nin kancası bunu çağırır.

    `running_grace_sec=0`: kuyruk kendi başlattığı koşu bittikten SONRA yazıyor, "koşuyor" koruması
    burada yalnızca kaydı engellerdi (ve grace=0 ile `scan_run_dir` asla None dönmez). Klasör hiç
    oluşmamışsa bile satır yazılır: izi olmayan denemeyi saymamak bu defterin var olma sebebine aykırı.

    `only_if_new_attempt`: aynı DENEME iki kez sayılmasın. Kuyruk koşu bitince bir satır yazar; bir
    sonraki kuyruk aynı klasörü `<ad>.yarim-<n>` diye kenara alırken İKİNCİ kez görür. Kenara alma
    kancasının işi yalnızca kaydı OLMAYAN denemeyi yakalamaktır (kuyruk ya da makine önceki denemede
    öldüyse kimse yazmamıştır). Deneme kimliği (koşu adı + başlangıç anı) klasör adı değişse de
    aynı kalır: `provenance.json` koşu başlarken yazılıp bir daha dokunulmuyor, yeniden adlandırma
    da mtime'ı korur. Kayıt varsa None döner.
    """
    rec = scan_run_dir(run_dir, block=block, exit_code=exit_code, running_grace_sec=0.0)
    if only_if_new_attempt and any(attempt(r) == attempt(rec) for r in read(path)):
        return None                      # `attempt()` ile tek kimlik: iki tanım ayrışmasın
    rec["source"] = source
    if reason:
        rec["reason"] = f"{reason}; {rec['reason']}" if rec["reason"] else reason
    return append(rec, path)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def record_manual(run: str, status: str, reason: str, arm: str | None = None, seed: int | None = None,
                  block: str | None = None, steps: int | None = None, path: str = LEDGER) -> dict:
    """Elle kayıt: diskte iz bırakmayan denemeler (kuyruk başlamadan vazgeçilen koşu, makine yeniden
    başlatması, elle öldürülen süreç) için. Kod izi CANLI ağaçtan alınır — `run_queue.code_hash`
    ile AYNI tanım, çünkü iki ayrı sha256 tanımı defteri provenance.json ile karşılaştırılamaz yapar."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run_queue                                      # tembel import: run_queue bu modülü import ediyor

    status_txt = _git("status", "--porcelain")
    return append(blank(source="el", run=run, dir=run, arm=arm, seed=seed, block=block, status=status,
                        steps=steps, started=iso(time.time()), ended=iso(time.time()),
                        git_head=_git("rev-parse", "HEAD"), dirty=bool(status_txt),
                        src_sha256=run_queue.code_hash(ROOT), evidence="el-kaydi", reason=reason), path)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows konsolu cp1252 olabilir
    # stderr de şart: aşağıdaki `p.error(...)` gerekçeleri Türkçe ve argparse onları stderr'e yazıyor.
    # Varsayılan cp1252/backslashreplace ile "iptal kaydı gerekçesiz yazılamaz" ekrana
    # "iptal kaydı gerek?esiz yazılamaz" diye düşüyor — yani kuralı ihlal eden kişi kuralı okuyamıyor.
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="koşu defteri: ekleme, diskten yeniden kurulum, rapor")
    p.add_argument("--ledger", default=LEDGER, help="defter yolu (varsayılan SABİT yol; testler için)")
    p.add_argument("--reconstruct", metavar="DIZIN",
                   help="checkpoint dizinini tarayıp eksik kayıtları deftere ekler")
    p.add_argument("--queue", help="kuyruk dosyası: blok kimlikleri (_block) buradan okunur")
    p.add_argument("--grace", type=float, default=900.0,
                   help="bitmemiş ve son N saniyede yazılmış koşu KOŞUYOR sayılır, deftere yazılmaz")
    p.add_argument("--report", action="store_true", help="V11'in istediği sayıları yazar")
    p.add_argument("--dry-run", action="store_true", help="deftere yazmaz, yazacağını gösterir")
    p.add_argument("--record", action="store_true", help="elle tek kayıt ekle (--run ve --status ile)")
    p.add_argument("--run", help="koşu adı")
    p.add_argument("--status", choices=list(STATUSES))
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    p.add_argument("--block")
    p.add_argument("--steps", type=int)
    p.add_argument("--reason", default="", help="serbest metin gerekçe (iptalde ZORUNLU sayılmalı)")
    a = p.parse_args()

    if a.record:
        if not (a.run and a.status):
            p.error("--record için --run ve --status gerekli")
        if a.status == "iptal" and not a.reason:
            p.error("iptal kaydı gerekçesiz yazılamaz: --reason verin")
        rec = record_manual(a.run, a.status, a.reason, a.arm, a.seed, a.block, a.steps, a.ledger)
        print(f"eklendi: {rec['run']} [{rec['status']}] -> {a.ledger}")

    if a.reconstruct:
        blocks = blocks_from_queue(a.queue) if a.queue else {}
        recs, skipped = reconstruct(a.reconstruct, blocks, a.grace)
        print(f"{a.reconstruct}: {len(recs)} koşu klasöründen kayıt çıkarıldı"
              + (f", {len(skipped)} klasör KOŞUYOR sayılıp atlandı ({', '.join(skipped)})" if skipped else ""))
        if a.dry_run:
            for r in recs:
                print(f"  [prova] {r['dir']:24s} {r['status']:11s} {r['evidence']}")
        else:
            added = append_missing(recs, a.ledger)
            print(f"deftere {len(added)} yeni satır eklendi, {len(recs) - len(added)} satır zaten vardı "
                  f"-> {a.ledger}")

    if a.report or not (a.record or a.reconstruct):
        print()
        print_report(a.ledger)


if __name__ == "__main__":
    main()
