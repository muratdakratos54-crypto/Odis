"""v03 adım 1 — kampanya SÜRÜCÜSÜ: seed 101 kuyruğu -> geçerlilik kapısı -> seed 102 + 103 kuyruğu. İnsan / oturum gerektirmez.

    .venv/Scripts/python scripts/v03_step1_campaign.py                          (İLK yol; seed 101'den başlar)
    .venv/Scripts/python scripts/v03_step1_campaign.py --continue-after-s101    (DEVAM yolu; sahibin onayı, 2026-09-21)

İkisi de temiz, commit'li ağaçtan ve bağımsız süreç olarak başlatılır. DİKKAT: seed 101 kapısı aşıldıktan sonra bayraksız
başlatmak hiçbir koşu yapmaz (seed 101 bloğu tamam, kapısı deterministik olarak yine aşılır): devam için BAYRAK şarttır.

Sahibin talimatı: 12 koşu seed bloklarıyla koşar (101 -> 102 -> 103). Seed 101'in dört kolu bitince YALNIZCA geçerlilik
kapılarına bakılır (`v03_step1_gate.py`; δ ve δ_D hesaplanmaz). Kapı aşılırsa — ya da dört koşudan biri bütünlük
denetiminden geçmezse — kuyruğun kalanı BAŞLATILMAZ. Aşılmazsa devam edilir; ara sonuç yoktur.

Kuyruğu ikiye bölmenin sebebi budur: `run_queue.py`'ye dokunulmadı; "kapı aşılırsa dur" kararı iki kuyruk arasında, bu
betikte verilir. İki kuyruğun birleşimi onaylı 12 koşuluk kuyruktur (`queues/step1.json`; testle kilitli). Kuyruklar SIKIdır:
`code_dirty = False` her koşudan önce denetlenir. Bu betik depoya HİÇBİR ŞEY yazmaz: günlük, kapı çıktısı ve durum dosyası
`checkpoints/foton-v03-step1/` altındadır (kuyruğun kendi koşu defteri dışında depo değişmez).

DEVAM YOLU (`--continue-after-s101`; sahibin onayı, 2026-09-21): seed 101 kapısı D bacağında aşıldıktan ve teşhis koşusu
"kod sağlam" dedikten sonra sahibi kalan 8 koşuyu onayladı ve seed başına ön koşul bacağını yürürlükte tuttu: kuyruk
`step1_s102.json` -> seed 102 ön koşul kapısı -> `step1_s103.json` -> seed 103 ön koşul kapısı. İki dosya, onaylı ikinci
kuyruğun (`step1_s102_s103.json`) seed sınırında BÖLÜNMÜŞ hâlidir: aynı 8 koşu, aynı sıra (testle kilitli; kuyruk
kısaltılmadı, yeniden üretilmedi — yeniden üretmek Latin kare dönüşümünü değiştirirdi). Kapılar `--legs precondition`
ile çağrılır: D'ye bakılmaz. Aşılırsa kuyruk durur ve sabahı bekler. Bu yolda seed 101'in kapı çıktısı (`gate_s101.json`)
SİLİNMEZ ve ÜZERİNE YAZILMAZ: o, kapının aşıldığının kaydıdır.
0. ADIM (koşudan önceki incelemenin bulgusu): ilk kuyruktan ÖNCE seed 101 bloğu `--seed 101 --legs precondition` ile yeniden
denetlenir (çıktı `gate_s101_recheck.json`). Seed 102 / 103 kapıları kendi koşularının src özetini ŞU ANKİ ağaçla
karşılaştırır; o koşular o ağaçtan koştuğu için bu, seed 101'e göre totolojiktir. 101 | 102 sınırını ancak bu adım görür:
seed 101'den sonra src/ değişmişse 8 koşu başlamadan durulur (yoksa analiz 12 koşudan SONRA "tek src özeti" diye reddederdi).

Durum dosyası (`campaign_status.json`): süreç BAŞLARKEN `running` (+ pid, başlangıç anı), biterken İLK yolda `completed` |
`queue_a_failed` | `gate_exceeded` | `gate_integrity_error` | `queue_b_failed` | `crashed`; DEVAM yolunda `completed` |
`s101_recheck_failed` | `queue_s102_failed` | `gate_s102_exceeded` | `gate_s102_integrity_error` | `queue_s103_failed` |
`gate_s103_exceeded` | `gate_s103_integrity_error` | `crashed`. İzleyen taraf `status != "running"` koşuluna göre bekler. Dosyanın hiç olmaması "bu
sürücü hiç başlamadı" demektir; `running` kalması makinenin / sürecin öldüğünü gösterir (finally çalışmadı) — ikisi
karışmasın diye başlangıç kaydı yazılır. Analiz (`v03_step1_analyze.py`) YALNIZCA `completed`'dan sonra, elle koşulur.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QUEUES = os.path.join("experiments", "foton-v03-step1", "queues")
QUEUE_A, QUEUE_B = os.path.join(QUEUES, "step1_s101.json"), os.path.join(QUEUES, "step1_s102_s103.json")
QUEUE_S102, QUEUE_S103 = os.path.join(QUEUES, "step1_s102.json"), os.path.join(QUEUES, "step1_s103.json")
OUT = os.path.join("checkpoints", "foton-v03-step1")
GATE_OUT, STATUS, LOG = os.path.join(OUT, "gate_s101.json"), os.path.join(OUT, "campaign_status.json"), os.path.join(OUT, "campaign.log")
GATE_OUT_S = {102: os.path.join(OUT, "gate_s102.json"), 103: os.path.join(OUT, "gate_s103.json")}
RECHECK_OUT = os.path.join(OUT, "gate_s101_recheck.json")
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


def continuation(run, note) -> tuple[str, int]:
    """Sahibin onayladığı DEVAM: seed 102 kuyruğu -> seed 102 ön koşul kapısı -> seed 103 kuyruğu -> seed 103 ön koşul kapısı.
    Saf: `run(cmd) -> çıkış kodu` dışarıdan verilir. Kapılar YALNIZ ön koşulun seed-başı bacağına bakar (`--legs precondition`)."""
    py = sys.executable
    note("0/4 seed 101 bloğunun yeniden denetimi (bütünlük + src sürekliliği; D'ye bakılmaz; gate_s101.json'a DOKUNULMAZ)")
    rc = run([py, os.path.join("scripts", "v03_step1_gate.py"), "--seed", "101", "--legs", "precondition", "--out", RECHECK_OUT])
    if rc != 0:                                            # seed 101'den sonra src/ değişmiş ya da blok bozulmuş: 8 koşu BAŞLAMAZ
        return "s101_recheck_failed", rc
    for i, (seed, queue) in enumerate(((102, QUEUE_S102), (103, QUEUE_S103))):
        note(f"{2 * i + 1}/4 seed {seed} kuyruğu: {queue}")
        rc = run([py, os.path.join("scripts", "run_queue.py"), queue])
        if rc != 0:
            return f"queue_s{seed}_failed", rc
        note(f"{2 * i + 2}/4 seed {seed} ön koşul kapısı (yalnız ön koşulun seed-başı bacağı; delta HESAPLANMAZ)")
        rc = run([py, os.path.join("scripts", "v03_step1_gate.py"), "--seed", str(seed), "--legs", "precondition",
                  "--out", GATE_OUT_S[seed]])
        if rc != 0:                                        # 0 DIŞINDAKİ her kod kuyruğu durdurur; sabahı bekler
            return (f"gate_s{seed}_exceeded" if rc == GATE_EXCEEDED else f"gate_s{seed}_integrity_error"), rc
    return "completed", 0


def main(argv: list[str] | None = None) -> int:
    # Bağımsız süreçte stdout DOSYAYA yönlenir ve Windows'ta o zaman cp1252 olur: Türkçe karakterli İLK günlük satırında
    # UnicodeEncodeError ile ölürdü — kuyruk hiç başlamadan, durum dosyası da yazılmadan (başlatmadan önce denenerek bulundu).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--continue-after-s101", action="store_true",
                    help="sahibin onayladığı devam: seed 102 + 103, her birinin ardından ön koşul kapısı")
    args = ap.parse_args([] if argv is None else argv)
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    # Bayat bir "completed" ya da eski denemenin kapı çıktısı bu kampanyanın sonucu sanılmasın. DEVAM yolunda seed 101'in
    # kapı çıktısı SİLİNMEZ: kapının aşıldığının kaydıdır.
    stale_files = (STATUS, RECHECK_OUT, *GATE_OUT_S.values()) if args.continue_after_s101 else (STATUS, GATE_OUT)
    for stale in stale_files:
        if os.path.exists(stale):
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
    if args.continue_after_s101:
        note("DEVAM (sahibin onayı, 2026-09-21): kalan 8 koşu; seed başına ön koşul bacağı yürürlükte")
    try:
        status, rc = (continuation if args.continue_after_s101 else campaign)(run, note)
    finally:
        with open(STATUS, "w", encoding="utf-8", newline="\n") as f:          # ÖNCE durum dosyası: günlük yazılamasa da kalsın
            json.dump({"status": status, "rc": rc, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        note(f"KAMPANYA DURUMU: {status} (kod {rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
