"""Eğitim koşularını sırayla çalıştırır (tek GPU): kesilirse kaldığı yerden, biten koşuları atlayarak devam eder.

    .venv\\Scripts\\python scripts/run_queue.py queue.json

queue.json:
    {"out_dir": "checkpoints/v2", "cwd": ".", "python": ".venv/Scripts/python.exe", "env": {"AD": "değer"},
     "runs": [{"name": "dense-s1337", "args": ["--preset", "dense", "--seed", "1337"]}, ...]}

* Biten koşu atlanır: klasörde DONE işareti ya da metrics.jsonl'da "end" kaydı var.
* Yarım kalmış koşu SİLİNMEZ, <ad>.yarim-<n> adıyla kenara alınıp baştan başlatılır: resume, eğitim duvar saatinin
  checkpoint'ten sonraki kısmını kaybetmese de araya giren bekleme ve ısınma ölçümü bozar.
* Her koşunun konsol çıktısı <out_dir>/<ad>/train.log, kuyruğun özeti <out_dir>/queue.log.
* Kod izi: her koşunun klasörüne provenance.json yazılır (cwd/src altındaki .py dosyalarının SHA-256'sı, git HEAD ve
  commit'lenmemiş değişiklikler, python/torch/GPU sürümleri). Kuyruk sırasında kod değişirse queue.log uyarır.
* Koşu defteri: her DENEME için `scripts/run_ledger.py`'nin sabit defterine bir satır eklenir — hem biten/çöken koşu
  için, hem de kenara alınan yarım koşu için (o klasör silinince denemenin tek izi defter olur). Defter yazılamazsa
  kuyruk DURMAZ, yalnızca queue.log'a uyarı düşer: saatler süren bir bloğu bir kayıt satırı yüzünden kaybetmek daha pahalı.
* Windows'ta kuyruk çalıştığı sürece sistemin uykuya geçmesi engellenir (ekran kapanabilir); süreç bitince kalkar.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time


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


def _cmd(args: list[str], cwd: str) -> str:
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def provenance(cwd: str, python: str) -> dict:
    return {
        "src_sha256": code_hash(cwd),
        "git_head": _cmd(["git", "rev-parse", "HEAD"], cwd),
        "git_status": _cmd(["git", "status", "--porcelain"], cwd).splitlines(),
        "python": _cmd([python, "-c", "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda)"], cwd),
        "gpu": _cmd(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], cwd),
        "host": platform.platform(),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    keep_awake()
    with open(sys.argv[1], encoding="utf-8") as f:
        q = json.load(f)
    cwd = os.path.abspath(q.get("cwd", "."))
    out_dir = os.path.abspath(q["out_dir"])
    # Windows'ta CreateProcess göreli yürütülebilir yolu cwd'ye göre çözmez: mutlak yap
    python = os.path.abspath(os.path.join(cwd, q.get("python", sys.executable)))
    os.makedirs(out_dir, exist_ok=True)
    qlog = open(os.path.join(out_dir, "queue.log"), "a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        qlog.write(line + "\n")
        qlog.flush()

    def ledger(run_dir: str, run: dict, reason: str, exit_code: int | None = None,
               only_if_new: bool = False) -> None:
        """Koşu defterine bir satır ekler. Blok kimliği YALNIZCA kuyruk dosyasında var (`_block`),
        diskte yazılı değil; duvar saati karşılaştırması blok-içi olmak zorunda olduğu için (V10-ii)
        kaydı buradan yazmak tek yol. Hata kuyruğu durdurmaz."""
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import run_ledger

            run_ledger.record_run_dir(run_dir, block=run.get("_block"), exit_code=exit_code,
                                      reason=reason, source="kuyruk", only_if_new_attempt=only_if_new)
        except Exception as e:                            # defter yazılamadı: koşu kaydı kaybolur, kuyruk sürer
            note(f"[uyarı] koşu defterine yazılamadı ({os.path.basename(run_dir)}): {e!r}")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **q.get("env", {})}
    prov0 = provenance(cwd, python)
    note(f"kuyruk: {sys.argv[1]} | kod {prov0['src_sha256'][:12]} | git {prov0['git_head'][:10]} "
         f"(+{len(prov0['git_status'])} değişiklik) | {prov0['python']} | {prov0['gpu']}")
    for i, run in enumerate(q["runs"], 1):
        run_dir = os.path.join(out_dir, run["name"])
        if finished(run_dir):
            note(f"[{i}/{len(q['runs'])}] {run['name']}: zaten bitmiş, atlandı")
            continue
        if os.path.exists(run_dir):
            n = 1
            while os.path.exists(f"{run_dir}.yarim-{n}"):
                n += 1
            os.replace(run_dir, f"{run_dir}.yarim-{n}")
            note(f"[{i}/{len(q['runs'])}] {run['name']}: yarım kalmış koşu {run_dir}.yarim-{n} olarak kenara alındı")
            # Yalnızca DEFTERDE KAYDI OLMAYAN deneme için: kaydı olan (kuyruk bitişini gördüğü) deneme
            # aynı klasör yüzünden ikinci kez sayılmasın. Kaydı olmayan deneme = kuyruk ya da makine
            # önceki denemede ölmüş demektir ve o denemenin tek izi bu satır olur.
            ledger(f"{run_dir}.yarim-{n}", run, "kuyruk kenara aldı; bu denemenin defterde kaydı yoktu "
                   "(kuyruk ya da makine önceki denemede ölmüş)", only_if_new=True)
        os.makedirs(run_dir)
        prov = {**provenance(cwd, python), "args": run["args"], "env": q.get("env", {})}
        if prov["src_sha256"] != prov0["src_sha256"]:
            note(f"[uyarı] kuyruk başladığından beri src/ değişti: {prov0['src_sha256'][:12]} -> {prov['src_sha256'][:12]}")
        with open(os.path.join(run_dir, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(prov, f, indent=2, ensure_ascii=False)
        cmd = [python, "-m", "src.train", *run["args"], "--out_dir", out_dir, "--run_name", run["name"]]
        note(f"[{i}/{len(q['runs'])}] {run['name']}: başladı: {' '.join(cmd[3:])}")
        t = time.perf_counter()
        with open(os.path.join(run_dir, "train.log"), "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            open(os.path.join(run_dir, "DONE"), "w").close()
        note(f"[{i}/{len(q['runs'])}] {run['name']}: bitti kod={rc} ({(time.perf_counter() - t) / 60:.1f} dk)")
        ledger(run_dir, run, f"kuyruk çalıştırdı, çıkış kodu {rc}", exit_code=rc)
        if rc != 0:
            note("hata: kuyruk durduruldu")
            sys.exit(rc)
    note("kuyruk tamamlandı")


if __name__ == "__main__":
    main()
