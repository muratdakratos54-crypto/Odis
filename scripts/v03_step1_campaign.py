"""v03 adım 1 — kampanya SÜRÜCÜSÜ: seed 101 kuyruğu -> geçerlilik kapısı -> seed 102 + 103 kuyruğu. İnsan / oturum gerektirmez.

    .venv/Scripts/python scripts/v03_step1_campaign.py            (temiz, commit'li ağaçtan; bağımsız süreç olarak)

Sahibin talimatı: 12 koşu seed bloklarıyla koşar (101 -> 102 -> 103). Seed 101'in dört kolu bitince YALNIZCA geçerlilik
kapılarına bakılır (`v03_step1_gate.py`; δ ve δ_D hesaplanmaz). Kapı aşılırsa — ya da dört koşudan biri bütünlük
denetiminden geçmezse — kuyruğun kalanı BAŞLATILMAZ. Aşılmazsa devam edilir; ara sonuç yoktur.

Kuyruğu ikiye bölmenin sebebi budur: `run_queue.py`'ye dokunulmadı; "kapı aşılırsa dur" kararı iki kuyruk arasında, bu
betikte verilir. İki kuyruğun birleşimi onaylı 12 koşuluk kuyruktur (`queues/step1.json`; testle kilitli). Kuyruklar SIKIdır:
`code_dirty = False` her koşudan önce denetlenir. Bu betik depoya HİÇBİR ŞEY yazmaz: günlük, kapı çıktısı ve durum dosyası
`checkpoints/foton-v03-step1/` altındadır (kuyruğun kendi koşu defteri dışında depo değişmez).

Durum dosyası (`campaign_status.json`): süreç BAŞLARKEN `running` (+ pid, başlangıç anı), biterken `completed` |
`queue_a_failed` | `gate_exceeded` | `gate_integrity_error` | `queue_b_failed` | `crashed`. Dosyanın hiç olmaması "bu
sürücü hiç başlamadı" demektir; `running` kalması makinenin / sürecin öldüğünü gösterir (finally çalışmadı) — ikisi
karışmasın diye başlangıç kaydı yazılır. Analiz (`v03_step1_analyze.py`) YALNIZCA `completed`'dan sonra, elle koşulur.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QUEUES = os.path.join("experiments", "foton-v03-step1", "queues")
QUEUE_A, QUEUE_B = os.path.join(QUEUES, "step1_s101.json"), os.path.join(QUEUES, "step1_s102_s103.json")
OUT = os.path.join("checkpoints", "foton-v03-step1")
GATE_OUT, STATUS, LOG = os.path.join(OUT, "gate_s101.json"), os.path.join(OUT, "campaign_status.json"), os.path.join(OUT, "campaign.log")
GATE_EXCEEDED = 3                                          # v03_step1_gate.EXIT_EXCEEDED


def campaign(run, note) -> tuple[str, int]:
    """Sıra ve DURMA kuralı — saf: `run(cmd) -> çıkış kodu` dışarıdan verilir ki GPU'suz sınanabilsin."""
    py = sys.executable
    note(f"1/3 seed 101 kuyruğu: {QUEUE_A}")
    rc = run([py, os.path.join("scripts", "run_queue.py"), QUEUE_A])
    if rc != 0:
        return "queue_a_failed", rc
    note("2/3 seed 101 geçerlilik kapısı (delta HESAPLANMAZ)")
    rc = run([py, os.path.join("scripts", "v03_step1_gate.py"), "--out", GATE_OUT])
    if rc != 0:                                            # 0 DIŞINDAKİ her kod kuyruğu durdurur (aşıldı / bütünlük / çöktü)
        return ("gate_exceeded" if rc == GATE_EXCEEDED else "gate_integrity_error"), rc
    note(f"3/3 seed 102 + 103 kuyruğu: {QUEUE_B}")
    rc = run([py, os.path.join("scripts", "run_queue.py"), QUEUE_B])
    if rc != 0:
        return "queue_b_failed", rc
    return "completed", 0


def main() -> int:
    # Bağımsız süreçte stdout DOSYAYA yönlenir ve Windows'ta o zaman cp1252 olur: Türkçe karakterli İLK günlük satırında
    # UnicodeEncodeError ile ölürdü — kuyruk hiç başlamadan, durum dosyası da yazılmadan (başlatmadan önce denenerek bulundu).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    for stale in (STATUS, GATE_OUT):                       # bayat bir "completed" ya da eski denemenin kapı çıktısı
        if os.path.exists(stale):                          # bu kampanyanın sonucu sanılmasın
            os.remove(stale)
    with open(STATUS, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"status": "running", "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f)

    def note(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with open(LOG, "a", encoding="utf-8") as f:          # asıl kayıt günlük DOSYASIDIR
            f.write(line + "\n")
        try:
            print(line, flush=True)
        except (OSError, ValueError):                         # konsol yazılamıyor diye 5 GPU saatlik kampanya ölmesin
            pass

    def run(cmd: list[str]) -> int:
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        note(f"    çıkış kodu {rc}: {' '.join(cmd[1:])}")
        return rc

    status, rc = "crashed", 1
    try:
        status, rc = campaign(run, note)
    finally:
        with open(STATUS, "w", encoding="utf-8", newline="\n") as f:          # ÖNCE durum dosyası: günlük yazılamasa da kalsın
            json.dump({"status": status, "rc": rc, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        note(f"KAMPANYA DURUMU: {status} (kod {rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
