"""Eğitim koşularını sırayla çalıştırır (tek GPU): kesilirse kaldığı yerden, biten koşuları atlayarak devam eder.

    .venv\\Scripts\\python scripts/run_queue.py queue.json

queue.json:
    {"out_dir": "checkpoints/v2", "cwd": ".", "python": ".venv/Scripts/python.exe", "env": {"AD": "değer"},
     "runs": [{"name": "dense-s1337", "args": ["--preset", "dense", "--seed", "1337"]}, ...]}

* Biten koşu atlanır: klasörde DONE işareti ya da metrics.jsonl'da "end" kaydı var.
* BLOKSUZ koşu (`_block` anahtarı yok) — eski davranış: yarım kalmış koşu SİLİNMEZ, <ad>.yarim-<n> adıyla kenara
  alınıp baştan başlatılır: resume, eğitim duvar saatinin checkpoint'ten sonraki kısmını kaybetmese de araya giren
  bekleme ve ısınma ölçümü bozar.
* BLOKLU koşu (`_block` anahtarı var; `scripts/make_campaign_queue.py` yazar) — ön-kayıt V11a(iii): bir seed bloğu
  yarıda kesildiyse (çökme, OOM, makine yeniden başlatma, kuyruğun durdurulması) O BLOĞUN TAMAMI yeniden koşulur.
  Kuyruk BAŞLARKEN kısmen koşulmuş her bloğun diskteki BÜTÜN koşu klasörleri — bitmiş olanlar dahil — silinmeden
  <ad>.blok-iptal-<n> adıyla kenara alınır, her biri koşu defterine yazılır ve blok İLK koşusundan yeniden başlar.
  Neden bitmiş koşu da: V11a(ii)/(iv) duvar saati karşılaştırmasını yalnızca KESİNTİSİZ koşulmuş bir bloğun içinde
  geçerli sayıyor. Kesintiden önce biten D ile kesintiden saatler sonra koşan X aynı girişim penceresini
  paylaşmaz; eski "biteni atla, kalanı koş" davranışı tam olarak V11a'nın kırmak için yazıldığı confound'u
  (koşu-oturumu gürültüsünün tek kola yüklenmesi) sessizce geri getiriyordu. Tamamı bitmiş blok eskisi gibi atlanır.
  Bir bloğun koşuları kuyruk dosyasında BİTİŞİK olmak zorundadır (V11a-i "kesintisiz"); değilse kuyruk başlamaz.
  Karar mantığı SAF bir fonksiyondur (`plan_blocks`): GPU ve alt süreç olmadan sınanır.
* Her koşunun konsol çıktısı <out_dir>/<ad>/train.log, kuyruğun özeti <out_dir>/queue.log.
* Kod izi: her koşunun klasörüne provenance.json yazılır (cwd/src altındaki .py dosyalarının SHA-256'sı, git HEAD ve
  commit'lenmemiş değişiklikler, python/torch/GPU sürümleri). Ek olarak:
    - `session`: BU run_queue sürecinin başlangıç anı (yerel, ISO). Sürecin başlattığı her koşuda AYNIDIR. V11a:
      "duvar saati büyüklüğü verilirken oturum kimliği ZORUNLUDUR" — kimlik buradan gelir.
    - `dirty`: `git status --porcelain` boş değil mi (ham tanım; koşu defterinin `dirty` alanıyla aynı).
    - `code_dirty`: aynı çıktının yalnızca KOD yolları (src/, scripts/, tests/, docs/, PREREGISTRATION*.md).
      Kuyruk her koşudan sonra experiments/ (koşu defteri) ve checkpoints/ altını KENDİSİ değiştirdiği için ham
      `dirty` kampanya boyunca hep True çıkıyor ve hiçbir şey söylemiyordu; `code_dirty` ölçümü ÜRETEN kodun
      commit'li olup olmadığını söyler. git okunamadıysa `None` (bilinmiyor) — temiz SAYILMAZ.
* SIKI kuyruk (`"_strict": true`; karar ve S1 kuyrukları): `code_dirty` False DEĞİLSE kuyruk HİÇ başlamaz (hiçbir
  klasöre dokunmadan çıkar), ve kuyruk başladığından beri src/ özeti değiştiyse sıradaki koşu BAŞLATILMADAN durur.
  Sıkı olmayan kuyruk eskisi gibi yalnızca queue.log'a uyarı yazar. Çıkış kodları: `EXIT_*` sabitleri.
* Koşu defteri: her DENEME için `scripts/run_ledger.py`'nin sabit defterine bir satır eklenir — hem biten/çöken koşu
  için, hem de kenara alınan koşu için (o klasör silinince denemenin tek izi defter olur). Defter yazılamazsa
  kuyruk DURMAZ, yalnızca queue.log'a uyarı düşer: saatler süren bir bloğu bir kayıt satırı yüzünden kaybetmek daha pahalı.
* Windows'ta kuyruk çalıştığı sürece sistemin uykuya geçmesi engellenir (ekran kapanabilir); süreç bitince kalkar.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HALF_SUFFIX = "yarim"             # bloksuz koşunun yarım kalmış denemesi: <ad>.yarim-<n>
BLOCK_SUFFIX = "blok-iptal"       # V11a(iii) gereği iptal edilen blok denemesinin koşuları: <ad>.blok-iptal-<n>

# `code_dirty` tanımı: ölçümü ÜRETEN ya da tanımlayan yollar. Kampanyanın kendi çıktıları (experiments/,
# checkpoints/) bilerek DIŞARIDA: kuyruk onları her koşudan sonra kendisi değiştiriyor.
CODE_DIRS = ("src/", "scripts/", "tests/", "docs/")
CODE_GLOBS = ("PREREGISTRATION*.md",)

# Kuyruğun KENDİ çıkış kodları. Eğitim sürecinin kodlarıyla (1 = hata, 2 = argparse) karışmasın diye yüksek seçildi.
EXIT_STRICT_DIRTY = 90            # sıkı kuyruk: kod commit'siz (ya da git okunamadı) -> hiç başlamadı
EXIT_STRICT_SRC_CHANGED = 91      # sıkı kuyruk: kuyruk sürerken src/ değişti -> sıradaki koşu başlatılmadı
EXIT_BAD_QUEUE = 92               # kuyruk dosyası tutarsız (yinelenen koşu adı, bool olmayan `_strict`, ...)
EXIT_SET_ASIDE_FAILED = 93        # klasör kenara alınamadı (Windows'ta açık dosya kilidi): üzerine yazılmaz


def finished(run_dir: str) -> bool:
    if os.path.exists(os.path.join(run_dir, "DONE")):
        return True
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    if line.strip() and json.loads(line).get("type") == "end":
                        return True
                except ValueError:  # çökmüş bir koşunun yarım yazılmış son satırı
                    continue
    except UnicodeDecodeError:
        return False
    return False


def keep_awake() -> None:
    """Saatler süren kuyrukta boşta kalma uykusu koşuları durdurmasın. SetThreadExecutionState yalnızca bu
    süreç yaşadıkça geçerlidir (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); ayar kalıcı bir değişiklik yapmaz."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


# --- karar mantığı: SAF fonksiyon (GPU, alt süreç ve disk YOK) -----------------------------------------

def plan_blocks(runs: list[dict], finished: dict[str, bool], exists: dict[str, bool]) -> list[dict]:
    """Kuyruğun yapacağı işleri SIRAYLA döndürür; diske dokunmaz.

    `finished` / `exists`: {koşu adı: klasör bitmiş mi / var mı} — KİMLİKLE anahtarlı, paralel liste değil
    (eksik ad KeyError: sessizce "yok" sayılmaz). Dönen işler:

        {"op": "set_aside", "run": ad, "kind": "blok-iptal" | "yarim", "block": blok | None}
        {"op": "skip", "run": ad}
        {"op": "run",  "run": ad}

    Kurallar (ön-kayıt V11a(i) ve V11a(iii)):
      * Blok = aynı `_block` değerini taşıyan koşular. Üyeleri kuyrukta BİTİŞİK olmak zorundadır (V11a-i: "bir
        seed bloğu içindeki dört koşu kesintisiz ve kuyruk dosyasındaki sırayla koşulur"). Araya başka bloğun ya
        da bloksuz bir koşunun girdiği kuyruk ValueError ile REDDEDİLİR: böyle bir kuyruk hiç kesilmeden koşsa
        bile blok kesintisiz koşmamış olur, ama defterde tek blok görünür ve duvar saati ölçütüne girerdi.
        Üreteç (`make_campaign_queue.py`) bunu zaten üretmez; hedef elle düzenlenmiş kuyruk dosyasıdır.
      * KISMEN koşulmuş blok = en az bir koşu klasörü VAR (bitmiş ya da yarım) ama koşuların HEPSİ bitmiş değil.
        Böyle her bloğun var olan BÜTÜN klasörleri kenara alınır ve bloğun BÜTÜN koşuları — bitmiş olanlar
        dahil — kuyruk sırasıyla yeniden koşulur. Kenara almalar planın EN BAŞINDADIR ("kuyruk başlarken"):
        iptal edilmiş bir denemenin klasörü, saatler sonra sıra o bloğa gelene kadar geçerli görünmesin.
      * Tamamı bitmiş blok atlanır; hiç dokunulmamış blok koşulur.
      * `_block` taşımayan koşu eski kuralla: bitmişse atla; klasörü varsa `.yarim-<n>` diye kenara al ve koş.
    """
    names = [r["name"] for r in runs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        # Ad = klasör = kimlik. Yinelenen ad, ikinci koşunun "zaten bitmiş" diye sessizce atlanması demekti.
        raise ValueError(f"kuyrukta yinelenen koşu adı: {dup}")
    for n in names:
        if finished[n] and not exists[n]:
            raise ValueError(f"tutarsız durum: '{n}' bitmiş görünüyor ama klasörü yok")

    blocks: dict[str, list[str]] = {}
    prev = None
    for r in runs:
        b = r.get("_block") or None
        if b is not None and b != prev and b in blocks:
            # V11a(i): blok KESİNTİSİZ koşulur. Daha önce görülmüş bir bloğa geri dönülüyorsa araya başka bir
            # koşu girmiş demektir; kuyruk başlamadan reddedilir (kampanyadan sonra onarılamaz).
            raise ValueError(f"blok '{b}' kuyrukta bitişik değil: '{r['name']}' koşusundan önce araya "
                             f"başka koşu girmiş (V11a-i: blok kesintisiz koşulur)")
        prev = b
        if r.get("_block"):
            blocks.setdefault(r["_block"], []).append(r["name"])
    cancelled =[b for b, members in blocks.items()
                 if any(exists[n] for n in members) and not all(finished[n] for n in members)]

    actions: list[dict] = []
    for b in cancelled:
        for n in blocks[b]:
            if exists[n]:
                actions.append({"op": "set_aside", "run": n, "kind": BLOCK_SUFFIX, "block": b})
    for r in runs:
        n, b = r["name"], (r.get("_block") or None)
        if b in cancelled:
            actions.append({"op": "run", "run": n})       # bitmiş olsa da: bloğun TAMAMI (V11a-iii)
        elif finished[n]:
            actions.append({"op": "skip", "run": n})
        else:
            if exists[n]:
                # İptal edilmemiş bir blokta bitmemiş klasör olamaz (o blok "kısmen koşulmuş" sayılırdı).
                assert b is None, f"plan hatası: '{n}' bloğu ({b}) kısmen koşulmuş ama iptal edilmemiş"
                actions.append({"op": "set_aside", "run": n, "kind": HALF_SUFFIX, "block": None})
            actions.append({"op": "run", "run": n})
    return actions


def set_aside(run_dir: str, kind: str) -> str:
    """Klasörü `<run_dir>.<kind>-<n>` adına TAŞIR ve yeni yolu döndürür; n = ilk boş tam sayı.

    ASLA silmez: kenara alınan klasör o denemenin tek ham kaydıdır (V11a-iii "her ikisi de deftere yazılır";
    defter satırı klasörden çıkarılıyor). Yeniden adlandırma mtime'ı korur, yani koşu defterinin deneme
    kimliği (koşu adı + başlangıç anı) klasör adı değişse de aynı kalır. `os.rename` bilerek: Windows'ta hedef
    varsa hata verir, üzerine YAZMAZ."""
    n = 1
    while os.path.exists(f"{run_dir}.{kind}-{n}"):
        n += 1
    dst = f"{run_dir}.{kind}-{n}"
    os.rename(run_dir, dst)
    return dst


# --- kod izi -------------------------------------------------------------------------------------------

def code_hash(cwd: str) -> str:
    h = hashlib.sha256()
    src = os.path.join(cwd, "src")
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(root, name)
                h.update(os.path.relpath(path, cwd).replace("\\", "/").encode())
                with open(path, "rb") as f:
                    h.update(f.read())
    return h.hexdigest()


def code_dirty_paths(status_lines: list[str]) -> list[str]:
    """`git status --porcelain` satırlarından KOD yollarını (CODE_DIRS / CODE_GLOBS) çıkarır.

    `provenance.json`'daki `git_status` listesinden YENİDEN hesaplanabilir olsun diye aynı satırları alır.
    O liste tüm çıktının `strip()`'inden geldiği için İLK satırın baştaki boşluğu düşmüş olabilir
    (" M src/x.py" -> "M src/x.py"); standart biçimde 3. karakter daima boşluktur, düşmüş biçimde değildir
    (boşlukla başlayan yolu git tırnağa alır) — ayrım buna dayanır. Yeniden adlandırmada ("eski -> yeni")
    iki uç da sayılır: yanılacaksa "kirli" yönünde yanılsın."""
    out = []
    for line in status_lines:
        if not line.strip():
            continue
        body = line[3:] if line[2:3] == " " else line[2:]
        for part in body.split(" -> "):
            path = part.strip().strip('"')                 # git özel karakterli yolu tırnağa alır; ayraç daima "/"
            low = path.casefold()                          # Windows'ta "Tests/" ile "tests/" aynı klasör
            if low.startswith(CODE_DIRS) or any(fnmatch.fnmatchcase(low, g.casefold()) for g in CODE_GLOBS):
                out.append(path)
    return out


def _cmd(args: list[str], cwd: str) -> str | None:
    """Komutun stdout'u (kırpılmış); komut çalışmadıysa ya da sıfırdan farklı kodla çıktıysa None.
    None ile "" ayrımı `code_dirty` için şart: git'in ÇALIŞMAMASI "değişiklik yok" demek değildir."""
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def session_id() -> str:
    """Oturum kimliği: bu sürecin başlangıç anı, yerel saatle (`run_ledger.iso` ile aynı biçim). `execute`
    bunu TEK KEZ çağırır; koşu başına çağrılsaydı kimlik koşudan koşuya değişir, oturumu tanımlamazdı."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def provenance(cwd: str, python: str, session: str, cmd=_cmd) -> dict:
    status = cmd(["git", "status", "--porcelain"], cwd)
    lines = (status or "").splitlines()
    return {
        "src_sha256": code_hash(cwd),
        "git_head": cmd(["git", "rev-parse", "HEAD"], cwd) or "",
        "git_status": lines,
        "dirty": bool(lines),                              # ham tanım: koşu defterinin `dirty` alanıyla aynı
        "code_dirty": None if status is None else bool(code_dirty_paths(lines)),
        "session": session,
        "python": cmd([python, "-c", "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda)"], cwd) or "",
        "gpu": cmd(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], cwd) or "",
        "host": platform.platform(),
    }


def strict_refusal(prov: dict) -> str | None:
    """Sıkı kuyruğun başlamayı REDDETME gerekçesi; başlayabilirse None. Yalnızca `code_dirty is False` geçer:
    None (git okunamadı) temizliğin KANITLANAMADIĞI durumdur ve sıkı katmanda ret sebebidir."""
    if prov["code_dirty"] is False:
        return None
    if prov["code_dirty"] is None:
        return "git durumu okunamadı: kodun commit'li olduğu KANITLANAMIYOR"
    return "commit'lenmemiş kod değişikliği var: " + ", ".join(code_dirty_paths(prov["git_status"]))


# --- koşu defteri kancaları ----------------------------------------------------------------------------

def _ledger_module():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import run_ledger

    return run_ledger


def block_cancel_record(rl, dst_dir: str, run: dict, path: str) -> dict | None:
    """V11a(iii) gereği kenara alınan koşunun defter satırı. `rl` = run_ledger modülü.

    * Koşu adı KUYRUKTAN alınır: `run_ledger.scan_run_dir` yalnızca `.yarim-` ekini tanıyor; `.blok-iptal-`
      ekli klasörde koşu adını klasör adının tamamı sanar ve deneme kimliği (koşu adı + başlangıç) bozulur.
    * BİTMİŞ koşu (diskte `bitti`/`sürdürüldü`) için DURUM DEĞİŞİKLİĞİ satırı yazılır: durum `iptal`, kanıt
      `blok-iptal`. Defterin kuralı bu ("aynı denemenin durum değişikliği yeni satır hak eder"). Diskteki
      durumla yazılsaydı aynı deneme ikinci bir `bitti` satırı olur ve raporun "duvar saati karşılaştırmasına
      uygun" sayısı, V11a(iv)'ün dışarıda bıraktığı koşuyu İKİ KEZ sayardı. Kapalı durum sözlüğünde bloğa özel
      bir durum olmadığı için ayrım `evidence == "blok-iptal"` ile taşınır.
    * BİTMEMİŞ koşu diskteki durumuyla (çöktü/oom/nan/iptal) ve yalnızca defterde kaydı YOKSA yazılır: kuyruk
      çöküşü gördüyse satırı zaten yazmıştır, ikinci satır V11'in ÇÖKEN sayısını şişirirdi
      (`record_run_dir(only_if_new_attempt=True)` ile aynı ilke). Kayıt varsa None döner.
    """
    rec = rl.scan_run_dir(dst_dir, block=run.get("_block"), running_grace_sec=0.0)
    rec["run"] = run["name"]
    rec["source"] = "kuyruk"
    # Kod izi yazılamadan ölmüş denemede (klasör var, provenance.json yok) kol/seed diskten çıkmaz; ek yüzünden
    # koşu adından da çıkmaz. Kuyruk dosyası aynı bilgiyi taşıyor: provenance'taki `args` zaten buradan gelir.
    args = run.get("args") or []
    if rec["arm"] is None:
        rec["arm"] = run.get("_arm")
    if rec["seed"] is None and "--seed" in args[:-1] and str(args[args.index("--seed") + 1]).isdigit():
        rec["seed"] = int(args[args.index("--seed") + 1])
    why = (f"V11a(iii): blok {run.get('_block')} yarıda kesildi, bloğun TAMAMI yeniden koşulacak; "
           f"klasör {os.path.basename(dst_dir)} olarak kenara alındı")
    if rec["status"] in ("bitti", "sürdürüldü"):
        why += (f"; koşu kendi başına '{rec['status']}' ({rec['evidence']}) ama kesintisiz bir blokta "
                "bitmediği için duvar saati ölçütüne GİREMEZ (V11a-iv)")
        rec["status"], rec["evidence"] = "iptal", BLOCK_SUFFIX
    elif any(rl.attempt(r) == rl.attempt(rec) for r in rl.read(path)):
        return None
    rec["reason"] = f"{why}; {rec['reason']}" if rec["reason"] else why
    return rl.append(rec, path)


# --- yürütme -------------------------------------------------------------------------------------------

def launch_subprocess(cmd: list[str], cwd: str, env: dict, log_path: str) -> int:
    with open(log_path, "w", encoding="utf-8") as log:
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def execute(q: dict, label: str, *, ledger_path: str | None, launch=launch_subprocess, cmd=_cmd,
            echo=print) -> int:
    """Kuyruğu yürütür ve SÜRECİN çıkış kodunu döndürür (0 = tamamlandı).

    `launch` ve `cmd` dışarıdan verilebilir ki blok kuralı, sıkı katman ve oturum kimliği GPU'suz sınanabilsin.
    `ledger_path`: None = defterin SABİT yolu (`run_ledger.LEDGER`). Varsayılanı bilerek yok: bir test yolu
    vermeyi unutup gerçek deftere satır yazamasın."""
    cwd = os.path.abspath(q.get("cwd", "."))
    out_dir = os.path.abspath(q["out_dir"])
    # Windows'ta CreateProcess göreli yürütülebilir yolu cwd'ye göre çözmez: mutlak yap
    python = os.path.abspath(os.path.join(cwd, q.get("python", sys.executable)))
    session = session_id()                                # TEK KEZ: bu sürecin başlattığı her koşuda aynı
    runs = q["runs"]
    by_name = {r["name"]: r for r in runs}
    index = {r["name"]: i for i, r in enumerate(runs, 1)}
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "queue.log"), "a", encoding="utf-8") as qlog:

        def note(msg: str) -> None:
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
            echo(line, flush=True)
            qlog.write(line + "\n")
            qlog.flush()

        def tag(name: str) -> str:
            return f"[{index[name]}/{len(runs)}] {name}"

        def ledger(write) -> None:
            """Koşu defterine bir satır ekler. Blok kimliği YALNIZCA kuyruk dosyasında var (`_block`),
            diskte yazılı değil; duvar saati karşılaştırması blok-içi olmak zorunda olduğu için (V10-ii)
            kaydı buradan yazmak tek yol. Hata kuyruğu durdurmaz."""
            try:
                rl = _ledger_module()
                write(rl, ledger_path or rl.LEDGER)
            except Exception as e:                        # defter yazılamadı: koşu kaydı kaybolur, kuyruk sürer
                note(f"[uyarı] koşu defterine yazılamadı: {e!r}")

        strict = q.get("_strict", False)
        if not isinstance(strict, bool):
            note(f"hata: '_strict' true/false olmalı, {strict!r} değil — kuyruk başlatılmadı")
            return EXIT_BAD_QUEUE
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", **q.get("env", {})}
        prov0 = provenance(cwd, python, session, cmd)
        note(f"kuyruk: {label} | oturum {session} | {'SIKI' if strict else 'sıkı değil'} | "
             f"kod {prov0['src_sha256'][:12]} | git {prov0['git_head'][:10]} "
             f"(+{len(prov0['git_status'])} değişiklik, code_dirty={prov0['code_dirty']}) | "
             f"{prov0['python']} | {prov0['gpu']}")
        if strict:
            refusal = strict_refusal(prov0)
            if refusal:
                # Hiçbir klasöre DOKUNMADAN çık: reddedilen bir başlatma blokları kenara almamalı.
                note(f"hata: SIKI kuyruk başlatılmadı — {refusal}. Commit'leyip yeniden başlatın "
                     "(experiments/ ve checkpoints/ altı sayılmaz).")
                return EXIT_STRICT_DIRTY

        try:
            plan = plan_blocks(runs,
                               finished={n: finished(os.path.join(out_dir, n)) for n in by_name},
                               exists={n: os.path.exists(os.path.join(out_dir, n)) for n in by_name})
        except ValueError as e:
            note(f"hata: {e} — kuyruk başlatılmadı")
            return EXIT_BAD_QUEUE
        for b in dict.fromkeys(a["block"] for a in plan if a["op"] == "set_aside" and a["block"]):
            members = [r["name"] for r in runs if r.get("_block") == b]
            note(f"blok {b} YARIDA KESİLMİŞ: V11a(iii) gereği var olan klasörleri kenara alınıp bloğun tamamı "
                 f"({', '.join(members)}) yeniden koşulacak")

        for act in plan:
            run = by_name[act["run"]]
            run_dir = os.path.join(out_dir, run["name"])
            if act["op"] == "skip":
                note(f"{tag(run['name'])}: zaten bitmiş, atlandı")
                continue
            if act["op"] == "set_aside":
                try:
                    dst = set_aside(run_dir, act["kind"])
                except OSError as e:
                    note(f"hata: {run_dir} kenara alınamadı ({e!r}); klasörde açık dosya olabilir — kuyruk durduruldu")
                    return EXIT_SET_ASIDE_FAILED
                if act["kind"] == BLOCK_SUFFIX:
                    note(f"{tag(run['name'])}: blok {act['block']} iptal (V11a-iii), klasör {dst} olarak kenara alındı")
                    ledger(lambda rl, path: block_cancel_record(rl, dst, run, path))
                else:
                    note(f"{tag(run['name'])}: yarım kalmış koşu {dst} olarak kenara alındı")
                    # Yalnızca DEFTERDE KAYDI OLMAYAN deneme için: kaydı olan (kuyruk bitişini gördüğü) deneme
                    # aynı klasör yüzünden ikinci kez sayılmasın. Kaydı olmayan deneme = kuyruk ya da makine
                    # önceki denemede ölmüş demektir ve o denemenin tek izi bu satır olur.
                    ledger(lambda rl, path: rl.record_run_dir(
                        dst, block=run.get("_block"), source="kuyruk", only_if_new_attempt=True, path=path,
                        reason="kuyruk kenara aldı; bu denemenin defterde kaydı yoktu "
                               "(kuyruk ya da makine önceki denemede ölmüş)"))
                continue

            prov = {**provenance(cwd, python, session, cmd), "args": run["args"], "env": q.get("env", {})}
            if prov["src_sha256"] != prov0["src_sha256"]:
                change = f"{prov0['src_sha256'][:12]} -> {prov['src_sha256'][:12]}"
                if strict:
                    # Koşu BAŞLATILMADAN dur: farklı kodla koşan iki kol karşılaştırılamaz ve kampanyadan
                    # sonra onarılamaz. Blok yarıda kaldıysa sonraki başlatma onu V11a(iii) ile baştan koşar.
                    note(f"hata: SIKI kuyruk durduruldu — kuyruk başladığından beri src/ değişti ({change}); "
                         f"{run['name']} başlatılmadı")
                    return EXIT_STRICT_SRC_CHANGED
                note(f"[uyarı] kuyruk başladığından beri src/ değişti: {change}")
            os.makedirs(run_dir)                          # var olan klasörün üzerine ASLA yazılmaz (FileExistsError)
            with open(os.path.join(run_dir, "provenance.json"), "w", encoding="utf-8") as f:
                json.dump(prov, f, indent=2, ensure_ascii=False)
            cmdline = [python, "-m", "src.train", *run["args"], "--out_dir", out_dir, "--run_name", run["name"]]
            note(f"{tag(run['name'])}: başladı: {' '.join(cmdline[3:])}")
            t = time.perf_counter()
            rc = launch(cmdline, cwd, env, os.path.join(run_dir, "train.log"))
            if rc == 0:
                open(os.path.join(run_dir, "DONE"), "w").close()
            note(f"{tag(run['name'])}: bitti kod={rc} ({(time.perf_counter() - t) / 60:.1f} dk)")
            ledger(lambda rl, path: rl.record_run_dir(
                run_dir, block=run.get("_block"), exit_code=rc, source="kuyruk", path=path,
                reason=f"kuyruk çalıştırdı, çıkış kodu {rc}; oturum {session}, code_dirty={prov['code_dirty']}"))
            if rc != 0:
                if run.get("_block"):
                    note(f"blok {run['_block']} yarıda kesildi: sonraki başlatmada bloğun TAMAMI yeniden koşulur (V11a-iii)")
                note("hata: kuyruk durduruldu")
                return rc
        note("kuyruk tamamlandı")
    return 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    keep_awake()
    with open(sys.argv[1], encoding="utf-8") as f:
        q = json.load(f)
    rc = execute(q, sys.argv[1], ledger_path=None)
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
