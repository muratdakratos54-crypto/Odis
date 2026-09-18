"""Ön-kaydın SHA-256 çivilerini FİİLEN doğrulanabilir yapan betik: manifest üret / manifesti doğrula.

    .venv\\Scripts\\python scripts/verify_pins.py --generate
    .venv\\Scripts\\python scripts/verify_pins.py --verify

Neden var: denetim ön-kayıtta iki boşluk buldu.

1. `PREREGISTRATION_V02.md` 8 hash çiviliyor ve "bu hash'ler temiz klonda her platformda doğrulanır"
   diyor. Bu YANLIŞ. `.gitattributes` satır sonu normalizasyonunu gerçekten kapatıyor, ama
   `.gitignore` `data/*` ve `checkpoints/*` satırlarını taşıyor: temiz klonda o dosyaların HİÇBİRİ
   yok. Doğrulanamayan şey hash'in platform kararlılığı değil, hash'lenecek dosyanın VARLIĞIYDI.
2. "Dondurulmuş korpus ve baseline çivileri" başlığı baseline çivisi vadediyor, blokta yalnızca
   korpus var. Dondurulmuş hedefin (`L*`) kanıt zinciri — koşu başına `metrics.jsonl`,
   `config.json`, `provenance.json` — ne commit'li ne hash'li. v01 bunları çivilemişti (EK 1 B17);
   v02 hiçbirini. Ayrıca `tokenizer.vocab` çivisiz kalmış.

Bu betiğin ürettiği manifest, `experiments/foton-v02-prereg/pins.json`, çivilenen dosyalar git'te
OLMADIĞI için **commit'lenen artefaktın kendisidir**: 39 dosyanın yolu, baytı, SHA-256'sı ve her
birinin NEYİN kanıtı olduğu (kanıt notu olmadan bir hash listesi "39 dosya aynı kaldı" der,
"hedef şu eğriden türedi" demez).

ÜÇÜNCÜ BİR KİŞİ NEYİ DOĞRULAYABİLİR
  * Manifestin kendisini: hangi commit'te hangi içerikle durduğunu ve sonradan değişmediğini
    (manifest git'te, çivilediği dosyalar değil).
  * Dosyaları bizden AYRICA aldığında, aldığı baytların manifeste uyduğunu — `--verify` tam olarak
    bunu yapar, dosya dosya PASS/FAIL.
  * Eğrilerin yanında aynı koşunun `config.json` ve `provenance.json`'unun da çivilendiğini: "hangi
    kod, hangi config, hangi eğri" üçlüsü birbirinden koparılamaz. Bir eğri değiştirilip config'i
    eskisiyle bırakılamaz, ikisi de aynı manifestte.
  * Manifestteki korpus hash'lerinin ön-kayıttaki 8 satırla BİREBİR aynı olduğunu: `--verify` bunu
    da denetler, yani manifest ön-kayıtta yazılı çivilerin üstüne sessizce yenilerini koyamaz.

NEYİ DOĞRULAYAMAZ — dipnot değil, burada
  * Dosyalar elinde yoksa HİÇBİR ŞEY. Manifest dosyanın yerine geçmez, ona işaret eder. Temiz klon
    + `--verify` = 39 satır "EKSİK". Bu bir arıza değil, iddianın gerçek sınırı.
  * Manifest, ÜRETİLDİĞİ ANA GÖRE bütünlük kanıtlar. Dosyaların herhangi bir FOTON koşusundan ÖNCE
    üretildiğini KANITLAMAZ. Güvenilen bir zaman damgası yok: commit tarihini de, bu manifesti de
    depo sahibi yazıyor, ikisi de sonradan yazılabilir. "Korpus önceden donduruldu" iddiası bu
    manifestten ÇIKMAZ; o iddia ön-kaydın commit sırasına ve dürüstlüğüne dayanır.
  * Dosyaların bize ulaşma yolunu. Bizden alınan bir kopyanın manifeste uyması, o kopyanın
    manifest yazılmadan önceki hâli olduğunu göstermez — yalnızca ikisinin tutarlı olduğunu.

NEDEN DOSYALARI COMMIT'LEMEKTEN ZAYIF
Baytlar git'te olsa içerik-adresli geçmişe girerlerdi: sonraki her değişiklik, klonu olan herkeste
görünür olurdu ve doğrulama için bizden bir şey almak gerekmezdi. Manifest bu güveni dosyaları
elinde tutan tarafa (bize) geri taşır. Korpus için pratik bir seçenek yok (612 MB). Ama kanıt
zinciri için var: 10 koşunun `metrics.jsonl` + `config.json` + `provenance.json` toplamı ~2.8 MB ve
`experiments/` gitignore'da değil. Bu betik o kararı vermiyor; sınırı yazıyor ki karar bilinçli
olsun. Güçlendirmenin diğer yolu, bu manifestin hash'ini dışarıya (üçüncü tarafa) tarih damgasıyla
yayınlamak olurdu; yapılmadı.

ÇİVİLENMEYENLER, gerekçesiyle
  * `best.pt` / `last.pt` — 20 dosya, ölçülen 10.74 GB; her `--verify` dakikalarca sürerdi. `L*`
    ağırlıklardan değil val EĞRİSİNDEN türüyor, o yüzden kanıt zinciri eğri + config + köken
    üçlüsü. Ağırlıklara dayanan bir iddia ortaya çıkarsa kendi çivi bloğunu gerektirir (v01 bunu
    `experiments/*/checksums.txt` ile yapmıştı).
  * `data/foton/_raw/` — yukarı akış shard'ları, ölçülen 2.93 GB, yeniden dağıtılamaz; lisans dökümü
    `corpus_report.json`'da.
  * `build.log`, `train.log`, `DONE` — kurulum/eğitim kayıtlarının insan okunur kopyaları ve bitiş
    bayrağı; ön-kaydın hiçbir sayısı bunlardan okunmuyor.

ÜÇ KORUMA — üçü de "sessizce" kelimesine karşı
  * `--generate`, mevcut manifestteki bir hash'i DEĞİŞTİRECEKSE ya da bir çiviyi DÜŞÜRECEKSE
    reddeder (`--force` gerekir): yeniden çivileme sessizce olmaz, ön-kayıt değişikliğidir.
  * `--verify`, diskte çivilenmemiş bir baseline koşusu görürse başarısız olur: manifest 10 koşuyu
    kanıt ilan ediyor, 11. koşu kanıt zincirini sessizce BÜYÜTEMEZ.
  * `--verify`, manifest `expected_pins()` kümesinin tamamını kapsamıyorsa başarısız olur: kanıt
    zinciri sessizce KÜÇÜLEMEZ. Hash'lerin tutması yetmez, kaç tanesinin tuttuğu da önemli — 38
    satırlık bir manifest 38 PASS ile "doğrulandı" görünürdü.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST = "experiments/foton-v02-prereg/pins.json"
PREREG = "PREREGISTRATION_V02.md"
DATA_DIR = "data/foton"
RUNS_DIR = "checkpoints/foton-v02"
SCHEMA = "foton-v02-pins/1"

SEEDS = (11, 22, 33, 44, 55)                  # ön-kaydın 5 karar seed'i
ARMS = ("m", "d")                             # m = foton-m (en iyi baseline), d = foton-d (bağlam kolu)
RUN_FILES = ("metrics.jsonl", "config.json", "provenance.json")
STEP80 = 3500                                 # referans bütçenin %80'i (4375 × 0.8); L* ve L_aux bu eval noktasından okunur

# Drift denetiminin bildirmeyeceği, kasıtlı olarak çivilenmeyen dosyalar (gerekçeleri docstring'de).
CORPUS_EXCLUDE = frozenset({"build.log"})
RUN_EXCLUDE = frozenset({"best.pt", "last.pt", "train.log", "DONE"})
# Koşu klasörü adı: `baseline-<kol>-s<seed>` (ilk küme) ya da `r-<kol>-s<seed>` (maskeli ölçütle
# yeniden koşulan ikinci küme — bkz. Sapmalar/S2). İkisi de tanınmak zorunda, yoksa doğrulama
# kapsamı sessizce boşalır.
RUN_DIR_RE = re.compile(r"^(?:baseline|r)-[a-z]+-s\d+$")
# Ayırıcı `\s+` DEĞİL `[ \t]+`: `\s` satır sonunu da yerdi, yani tek başına duran 64 hex'lik bir
# satır (ör. bir provenance src_sha256'sı — onlar da 64 hex) bir SONRAKİ satırın ilk kelimesiyle
# eşleşip uydurma bir çivi üretirdi. O uydurma yol manifestte bulunmayacağı için `--verify`
# sebepsiz FAIL verirdi. Ön-kayıt hâlâ düzenleniyor; bu tuzak bugün patlamıyor diye durmamalı.
PREREG_PIN_RE = re.compile(r"^([0-9a-f]{64})[ \t]+\*?(\S+)$", re.MULTILINE)

# Korpus çivileri: dosya -> NEYİN kanıtı. Not, "ne bu dosya" değil "ön-kaydın hangi iddiası buna
# dayanıyor" sorusunu cevaplıyor; çivi kırıldığında neyin geçersizleştiği okunabilsin.
CORPUS: dict[str, str] = {
    "meta.json": "korpus kimliği: alan başına token sayıları, vocab_size 32768, Tier A beyanı — "
                 "L*'in ölçüldüğü val pencerelerinin TANIMI bu dosyada",
    "tokenizer.model": "32768 parçalı SentencePiece modeli; .bin'lerdeki id'lerin bir anlamı yalnızca "
                       "bu dosyayla var, yani val loss'un birimi buna bağlı",
    "tokenizer.vocab": "parça↔id tablosunun insan okunur kopyası; denetimde ÇİVİSİZ olduğu bulundu. "
                       "Aynı eğitimden çıkar ama ayrı çivilenir: tokenizer kimliği tek dosyaya "
                       "bağlı kalmasın",
    "nl_train.bin": "NL alanı eğitim akışı (100M token, uint16); F_base'in paydası bu akıştan ölçüldü",
    "nl_val.bin": "NL alanı val akışı (2M token); L*'in üçte biri bu dosyadan gelen pencerelerden",
    "code_train.bin": "kod alanı eğitim akışı (100M token, uint16)",
    "code_val.bin": "kod alanı val akışı (2M token); alanlar arası 0.9 nat farkın bir ucu burada",
    "math_train.bin": "matematik alanı eğitim akışı (100M token, uint16)",
    "math_val.bin": "matematik alanı val akışı (2M token); alan başına val loss'un en düşük kolu",
}


def _run_note(fname: str, arm: str, seed: int) -> str:
    """Koşu dosyasının kanıt notu: hangi hedef, hangi kol, hangi seed."""
    preset = "foton-m" if arm == "m" else "foton-d"
    if fname == "metrics.jsonl":
        if arm == "m":
            return (f"L*'in türetildiği val eğrisi ({preset}, seed {seed}): hedef, adım {STEP80} "
                    "(referans bütçenin %80'i) val_loss'undan doğrudan okunuyor, interpolasyon yok")
        return (f"L_aux'un türetildiği val eğrisi ({preset}, seed {seed}) ve D'nin tam bütçede "
                "L*'e ULAŞMADIĞININ kaydı — sansür iddiası bu dosyaya dayanıyor")
    if fname == "config.json":
        return (f"eğriyi üreten tam config ({preset}, seed {seed}): eval_interval 250 ve "
                "batch_size × eval_iters = 1280 buradan doğrulanır; L* ölçülen PENCERE KÜMESİNE bağlı")
    return (f"koşunun kod ve donanım kimliği ({preset}, seed {seed}): src_sha256 + git_head — "
            "eğrinin hangi kodla üretildiği yalnızca bu dosyada yazılı")


def run_prefix(root: str, runs_dir: str) -> str:
    """Koşu klasörlerinin öneki, DİSKTEN çıkarılır: `baseline` ya da `r`.

    Neden sabit değil: `L*` maskeli ölçütle yeniden koşulan ikinci kümeden türetildi
    (Sapmalar/S2) ve o kümenin klasörleri `r-<kol>-s<seed>`, ilk kümenin `baseline-...`.
    Öneki modül sabiti yapmak, yanlış kümeye bakıldığında `expected_pins()`'i BOŞ döndürür ve
    doğrulama "0 çivi, hepsi geçti" diye sessizce başarılı olur — kapsamın var olma sebebini
    yok eden tam o sessizlik. Diskte hangi önek varsa o kullanılır; ikisi de varsa `r` kazanır
    (kanıt zinciri ikinci kümedir) ve bu durum `--verify` çıktısında yazılır.
    """
    d = abspath(root, runs_dir)
    if not os.path.isdir(d):
        return "r"
    names = set(os.listdir(d))
    for pref in ("r", "baseline"):
        if any(n.startswith(f"{pref}-") for n in names):
            return pref
    return "r"


def expected_pins(data_dir: str = DATA_DIR, runs_dir: str = RUNS_DIR,
                  prefix: str = "r") -> list[tuple[str, str]]:
    """Çivilenmesi GEREKEN (yol, kanıt notu) listesi; yollar depo köküne göre ve daima '/' ile.

    Liste KODDA sabit, glob değil: eksik bir koşu "9 dosya çiviledim" diye sessizce geçmesin,
    fazladan bir koşu da drift olarak görünsün. Windows'ta üretilen manifest Linux'ta da aynı
    yolları taşısın diye ayırıcı her platformda '/'.
    """
    pins = [(f"{data_dir}/{name}", note) for name, note in CORPUS.items()]
    for arm in ARMS:
        for seed in SEEDS:
            run = f"{runs_dir}/{prefix}-{arm}-s{seed}"
            for fname in RUN_FILES:
                pins.append((f"{run}/{fname}", _run_note(fname, arm, seed)))
    return pins


def abspath(root: str, rel: str) -> str:
    """Manifest yolunu ('/' ile) bu platformun yoluna çevirir."""
    return os.path.join(root, *rel.split("/"))


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    with open(path, "rb") as f:
        digest = getattr(hashlib, "file_digest", None)
        if digest is not None:                 # 3.11+: 200 MB'lık .bin'ler için C tarafında okur
            return digest(f, "sha256").hexdigest()
        h = hashlib.sha256()
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
        return h.hexdigest()


def val_at(path: str, step: int) -> float | None:
    """`metrics.jsonl`'de verilen adımın val_loss'u; yoksa None.

    Kanıt notuna o eğrinin TAŞIDIĞI sayıyı da yazmak için: çivi "hangi eğri" demekle kalmasın,
    "bu eğri adım 3500'de 2.903284 diyor" desin. Böylece manifest, L*'in yeniden türetilmesinde
    beklenen değeri de tutar.
    """
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or '"val"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "val" and rec.get("step") == step and "val_loss" in rec:
                    return float(rec["val_loss"])
    except OSError:
        return None
    return None


def git_head(root: str) -> str | None:
    """Manifestin üretildiği commit. Zaman damgası DEĞİL: yalnızca "hangi ağaçtan bakılmış" bilgisi."""
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def build_manifest(root: str, data_dir: str, runs_dir: str) -> dict:
    """Diskteki dosyalardan manifest kurar. Eksik dosyalar `missing` listesine yazılır, atlanmaz."""
    files, missing = [], []
    for rel, note in expected_pins(data_dir, runs_dir, run_prefix(root, runs_dir)):
        ap = abspath(root, rel)
        if not os.path.isfile(ap):
            missing.append(rel)
            continue
        if rel.endswith("metrics.jsonl"):
            v = val_at(ap, STEP80)
            if v is not None:
                note = f"{note} [adım {STEP80} val_loss = {v:.6f}]"
        files.append({"path": rel, "bytes": os.path.getsize(ap), "sha256": sha256_of(ap), "note": note})
    return {
        "schema": SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_git_head": git_head(root),
        "generated_on": f"{platform.platform()} / python {platform.python_version()}",
        "claim": "Bu manifest, listelenen dosyaların ÜRETİLDİĞİ ANDAKİ baytlarına göre bütünlük "
                 "kanıtlar. Dosyaların herhangi bir FOTON koşusundan ÖNCE var olduğunu KANITLAMAZ; "
                 "güvenilen bir zaman damgası yoktur. Dosyalar git'te değil (.gitignore: data/*, "
                 "checkpoints/*), bu yüzden commit'lenen kanıt manifestin KENDİSİDİR ve doğrulama "
                 "dosyaları ayrıca almayı gerektirir.",
        "verify_with": "python scripts/verify_pins.py --verify",
        "counts": {"pinned": len(files), "missing": len(missing), "expected": len(files) + len(missing)},
        "missing": missing,
        "files": files,
    }


def prereg_pins(path: str) -> dict[str, str]:
    """Ön-kayıttaki '<sha256>  <yol>' satırları. Manifestin o bloktan sapmasını yakalamak için."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return {}
    return {p: h for h, p in PREREG_PIN_RE.findall(text)}


def disk_pinnable(root: str, data_dir: str, runs_dir: str) -> list[str]:
    """Diskte çivilenmesi BEKLENEN dosyalar (drift denetimi için): korpus + baseline koşu üçlüleri."""
    found = []
    d = abspath(root, data_dir)
    if os.path.isdir(d):
        found += [f"{data_dir}/{n}" for n in sorted(os.listdir(d))
                  if os.path.isfile(os.path.join(d, n)) and n not in CORPUS_EXCLUDE]
    r = abspath(root, runs_dir)
    if os.path.isdir(r):
        for run in sorted(os.listdir(r)):
            rd = os.path.join(r, run)
            if not os.path.isdir(rd) or not RUN_DIR_RE.match(run):
                continue
            found += [f"{runs_dir}/{run}/{n}" for n in sorted(os.listdir(rd))
                      if os.path.isfile(os.path.join(rd, n)) and n not in RUN_EXCLUDE]
    return found


def do_generate(root: str = ROOT, manifest: str = MANIFEST, data_dir: str = DATA_DIR,
                runs_dir: str = RUNS_DIR, prereg: str = PREREG, force: bool = False) -> int:
    """Manifesti diskten üretir. Mevcut bir çiviyi DEĞİŞTİRECEKSE --force olmadan reddeder."""
    man = build_manifest(root, data_dir, runs_dir)
    out = abspath(root, manifest)
    print(f"== ÇİVİ MANİFESTİ ÜRETİLİYOR -> {manifest} ==")
    print(f"beklenen {man['counts']['expected']} dosya: {man['counts']['pinned']} çivilendi, "
          f"{man['counts']['missing']} eksik")

    old = {}
    if os.path.isfile(out):
        try:
            with open(out, encoding="utf-8") as f:
                old = {e["path"]: e["sha256"] for e in json.load(f).get("files", [])}
        except (OSError, ValueError, KeyError, TypeError):
            print("  UYARI: mevcut manifest okunamadı, yeniden çivileme denetimi yapılamıyor")
    repinned = [e["path"] for e in man["files"]
                if e["path"] in old and old[e["path"]] != e["sha256"]]
    dropped = [p for p in old if p not in {e["path"] for e in man["files"]}]

    if man["missing"]:
        # Eksik dosya manifesti sessizce küçültür: ön-kayıt 39 dosyayı kanıt ilan ediyor, 37'sini değil.
        print("  EKSİK dosyalar (manifest bunları 'missing' olarak taşıyor, çivilemiyor):")
        for p in man["missing"]:
            print(f"    - {p}")
    if repinned and not force:
        print(f"  REDDEDİLDİ: {len(repinned)} dosyanın hash'i mevcut manifestten FARKLI. Yeniden "
              "çivileme sessizce olamaz; ön-kayıt değişikliğidir (--force ile yazılır):")
        for p in repinned:
            print(f"    ~ {p}")
        return 1
    if dropped and not force:
        print(f"  REDDEDİLDİ: mevcut manifestte olan {len(dropped)} çivi yeni listede yok "
              "(kanıt zinciri küçülüyor; --force ile yazılır):")
        for p in dropped:
            print(f"    - {p}")
        return 1
    if repinned or dropped:
        print(f"  --force: {len(repinned)} çivi değişti, {len(dropped)} çivi düştü")

    pp = prereg_pins(abspath(root, prereg)) if prereg else {}
    if pp:
        cur = {e["path"]: e["sha256"] for e in man["files"]}
        diff = [p for p, h in pp.items() if p in cur and cur[p] != h]
        print(f"ön-kayıt bloğu: {len(pp)} çivi okundu, {len(diff)} tanesi diskle UYUŞMUYOR")
        for p in diff:
            print(f"    ~ {p}: ön-kayıt {pp[p][:12]}… / disk {cur[p][:12]}…")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"-> {manifest} yazıldı ({os.path.getsize(out):,} B)")
    print("NOT: bu manifest, üretildiği ana göre bütünlük kanıtlar; dosyaların bir FOTON koşusundan "
          "ÖNCE üretildiğini kanıtlamaz.")
    return 0


def do_verify(root: str = ROOT, manifest: str = MANIFEST, data_dir: str = DATA_DIR,
              runs_dir: str = RUNS_DIR, prereg: str = PREREG) -> int:
    """Manifesti yeniden hash'ler; dosya dosya PASS/FAIL basar, tek bir sapmada exit 1."""
    mpath = abspath(root, manifest)
    try:
        with open(mpath, encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError) as e:
        print(f"MANİFEST OKUNAMADI: {manifest}: {e}")
        print("Çivi doğrulaması mümkün değil: commit'lenen kanıt manifestin kendisidir.")
        return 1

    entries = man.get("files", [])
    print(f"== ÇİVİ DOĞRULAMA: {manifest} ==")
    print(f"manifest: {len(entries)} dosya, üretim {man.get('generated_at_utc', '?')}, "
          f"commit {str(man.get('generated_git_head'))[:12]}")
    if man.get("missing"):
        print(f"manifest ÜRETİLİRKEN eksik olan {len(man['missing'])} dosya var: "
              f"{', '.join(man['missing'][:3])}{' …' if len(man['missing']) > 3 else ''}")

    ok, fail, gone = [], [], []
    for e in entries:
        rel, ap = e["path"], abspath(root, e["path"])
        if not os.path.isfile(ap):
            gone.append(rel)
            print(f"  EKSİK {rel:<54s} dosya yok")
            continue
        size, digest = os.path.getsize(ap), sha256_of(ap)
        if digest == e["sha256"] and size == e["bytes"]:
            ok.append(rel)
            print(f"  PASS  {rel:<54s} {size:>13,} B")
        else:
            fail.append(rel)
            why = (f"hash farklı (çivi {e['sha256'][:12]}… / disk {digest[:12]}…)"
                   if digest != e["sha256"] else "")
            if size != e["bytes"]:
                why = (why + " ve " if why else "") + f"boyut farklı (çivi {e['bytes']:,} / disk {size:,})"
            print(f"  FAIL  {rel:<54s} {why}")

    # Drift: manifest 10 koşuyu kanıt ilan ediyor; diskte çivilenmemiş bir koşu varsa kanıt zinciri
    # sessizce büyümüş olur ve "kanıt bu 39 dosyadır" cümlesi artık doğru değildir.
    pinned = {e["path"] for e in entries}
    unpinned = [p for p in disk_pinnable(root, data_dir, runs_dir) if p not in pinned]
    if unpinned:
        print(f"  ÇİVİSİZ: diskte çivilenmemiş {len(unpinned)} dosya var "
              "(manifest güncel değil; yeniden çivileme ön-kayıt değişikliğidir):")
        for p in unpinned:
            print(f"    + {p}")

    # Kapsam: manifest, çivilenmesi GEREKEN kümenin TAMAMINI taşımak zorunda. `--generate` eksik bir
    # dosyayı `missing` diye kaydedip 0 dönebiliyor (üretim anında disk hazır olmayabilir), ama
    # doğrulamanın o hoşgörüyü taşıması ölümcül: 38 satırlık manifest, o 38'i de tutuyorsa "bütün
    # çiviler tuttu" diye GEÇİYORDU ve ön-kaydın "kanıt bu 39 dosyadır" cümlesi sessizce yanlışa
    # dönüyordu. Ölçüldü: math_val.bin diskte yokken üretilen 38 çivilik manifest `--verify`'dan
    # exit 0 alıyordu. Manifestten elle satır düşürmek (+ dosyayı silmek) de aynı boşluktan
    # geçiyordu; ikisini de kapatan tek denetim bu, çünkü `expected_pins()` kümeyi KODDA ilan ediyor
    # ve manifestin kendi `missing`/`counts` alanlarına güvenmiyor — o alanları da düzenleyebilirsiniz.
    uncovered = [rel for rel, _ in expected_pins(data_dir, runs_dir, run_prefix(root, runs_dir))
                 if rel not in pinned]
    if uncovered:
        print(f"  KAPSAM DIŞI: çivilenmesi gereken {len(uncovered)} dosya manifestte HİÇ YOK "
              "(bu bir hash kırılması değil, kanıt kümesinin eksik olması):")
        for p in uncovered:
            print(f"    ! {p}")

    pp = prereg_pins(abspath(root, prereg)) if prereg else {}
    cur = {e["path"]: e["sha256"] for e in entries}
    pre_bad = [p for p, h in pp.items() if p not in cur or cur[p] != h]
    if pp:
        print(f"ön-kayıt çapraz denetimi: {len(pp)} çivi, {len(pp) - len(pre_bad)} tanesi manifestle aynı")
        for p in pre_bad:
            print(f"  FAIL  ön-kayıt çivisi manifestle uyuşmuyor: {p}")
    else:
        print(f"ön-kayıt çapraz denetimi ATLANDI ({prereg or 'kapalı'}: çivi bloğu okunamadı)")

    bad = len(fail) + len(gone) + len(unpinned) + len(uncovered) + len(pre_bad)
    print(f"\n{len(entries)} çivi: {len(ok)} PASS, {len(fail)} FAIL, {len(gone)} EKSİK, "
          f"{len(unpinned)} ÇİVİSİZ, {len(uncovered)} KAPSAM DIŞI, {len(pre_bad)} ön-kayıt sapması")
    if bad:
        print("SONUÇ: DOĞRULAMA BAŞARISIZ")
        if gone and len(gone) == len(entries):
            print("Bütün dosyalar eksik: temiz klonda beklenen sonuç budur. Çivilenen dosyalar git'te "
                  "değil (.gitignore: data/*, checkpoints/*); doğrulama için dosyaları ayrıca almak "
                  "gerekir. Manifest dosyanın yerine geçmez, ona işaret eder.")
        return 1
    print("SONUÇ: bütün çiviler tuttu. Bu, dosyaların manifestin ÜRETİLDİĞİ ANDAKİ baytlarla aynı "
          "olduğunu gösterir — bir FOTON koşusundan önce üretildiklerini GÖSTERMEZ.")
    return 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Ön-kaydın SHA-256 çivilerini üret / doğrula")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true", help="diskteki dosyalardan manifesti yaz")
    g.add_argument("--verify", action="store_true", help="manifesti yeniden hash'leyip doğrula")
    p.add_argument("--root", default=ROOT, help="depo kökü (testler tmp_path verir)")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--data", default=DATA_DIR)
    p.add_argument("--runs", default=RUNS_DIR)
    p.add_argument("--prereg", default=PREREG,
                   help="korpus çivileri bu dosyayla karşılaştırılır; boş verilirse atlanır")
    p.add_argument("--force", action="store_true",
                   help="--generate: değişmiş hash'lerin üstüne yaz (YENİDEN ÇİVİLEME)")
    a = p.parse_args()
    root = os.path.abspath(a.root)
    rc = (do_generate(root, a.manifest, a.data, a.runs, a.prereg, a.force) if a.generate
          else do_verify(root, a.manifest, a.data, a.runs, a.prereg))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
