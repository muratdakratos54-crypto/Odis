"""Sahibin TAKDİR kararlarının defteri: S1'e çıkma / çıkmama seçimi, S1 KOŞULMADAN ÖNCE, zaman damgalı tek satır.

    .venv\\Scripts\\python scripts/owner_decision.py --choice "S1'e çık" \\
        --basis experiments/foton-v02/decision.json --reason "S0 satırı 'belirsiz': B10 gereği tek yükseltme hakkı"
    .venv\\Scripts\\python scripts/owner_decision.py --verify        # S1 koşuları karardan SONRA mı başlamış?

NEDEN VAR (PREREGISTRATION_V02.md, "Her satırdan sonra ne olur"). "nötr" ve "başarısız" satırlarında
ölçek yükseltme hakkını kullanıp kullanmamak sahibin TAKDİRİNDEDİR ve o seçim sonucu değiştirir:
"başarısız"ta hak kullanılmazsa vazgeçme koşulunun tek ölçek yolu açılır, kullanılırsa iki ölçek yolu.
Seçim S1'in sonucunu gördükten SONRA kayda geçirilebilseydi, "S1 iyi çıktı -> 'hakkı kullandım'",
"S1 kötü çıktı -> 'S1 keşifseldi, hak kullanılmadı'" demek mümkün olurdu. Bu yüzden seçim S1
koşulmadan ÖNCE, dayandığı S0 karar dosyasının SHA-256'sıyla birlikte tek satır olarak yazılır;
S1 kuyruğu o satır olmadan ÜRETİLMEZ (`make_campaign_queue.py --scale S1`) ve rapor aşamasında
her S1 koşusunun başlangıcının o satırdan SONRA olduğu denetlenir (`verify_order`).

Defter append-only'dir ve ölçek hakkı için TEK karar satırı kabul eder: ikinci bir satır (fikir
değiştirme) hata verir. `abandon()`'ın `escalation_used` girdisi bu defterden okunur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = "experiments/foton-v02/owner_decisions.jsonl"
CHOICES = ("S1'e çık", "S1'e çıkma")
S1_PREFIX = "s1-"                 # `make_campaign_queue.py --scale S1` koşu adlarına bu öneki koyar
MIN_REASON = 20                   # karakter; "ok" gibi bir gerekçe gerekçe değildir
FMT = "%Y-%m-%dT%H:%M:%S"         # koşu defteriyle AYNI biçim (yerel saat): ikisi yan yana konabilsin


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip() or None
    except Exception:
        return None


def read(log: str = LOG) -> list[dict]:
    if not os.path.exists(log):
        return []
    with open(log, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def record(choice: str, reason: str, basis: str, log: str = LOG, now: float | None = None) -> dict:
    """Ölçek hakkı kararını TEK satır olarak yazar. Gerekçe ve dayanak dosyası ZORUNLU."""
    if choice not in CHOICES:
        raise ValueError(f"seçim {CHOICES} içinden olmalı, verilen: {choice!r}")
    if len((reason or "").strip()) < MIN_REASON:
        raise ValueError(f"gerekçe ZORUNLU (en az {MIN_REASON} karakter): karar gerekçesiyle birlikte loglanır")
    if not os.path.isfile(basis):
        raise ValueError(f"dayanak dosyası yok: {basis} (S0'ın `decide_primary()` çıktısı olmalı)")
    if read(log):
        raise ValueError("ölçek hakkı kararı ZATEN kayıtlı: defter tek karar kabul eder, karar değiştirilemez")
    ts = time.time() if now is None else now
    rec = {"schema": 1, "kind": "ölçek hakkı", "choice": choice, "reason": reason.strip(),
           "ts_local": time.strftime(FMT, time.localtime(ts)),
           "ts_utc": time.strftime(FMT + "Z", time.gmtime(ts)),
           "basis": basis.replace("\\", "/"), "basis_sha256": _sha256(basis), "git_head": _git_head()}
    os.makedirs(os.path.dirname(os.path.abspath(log)) or ".", exist_ok=True)
    with open(log, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def require_s1_authorized(log: str = LOG) -> dict:
    """S1 kuyruğu üretilmeden ÖNCE çağrılır: "S1'e çık" kararı kayıtlı değilse hata verir."""
    recs = [r for r in read(log) if r.get("kind") == "ölçek hakkı"]
    if not recs:
        raise PermissionError(f"S1 kuyruğu üretilemez: sahibin ölçek hakkı kararı kayıtlı değil ({log}). "
                              "Önce: scripts/owner_decision.py --choice \"S1'e çık\" --basis ... --reason ...")
    if recs[0]["choice"] != "S1'e çık":
        raise PermissionError(f"S1 kuyruğu üretilemez: kayıtlı karar {recs[0]['choice']!r} "
                              f"({recs[0]['ts_local']}); karar değiştirilemez")
    return recs[0]


def escalation_used(log: str = LOG) -> bool:
    """`abandon(..., escalation_used=...)` girdisi: kayıtlı karar "S1'e çık" mı. Kayıt yoksa HATA
    (varsayılan bir değer, kararın loglanmadığı durumu gizlerdi)."""
    recs = [r for r in read(log) if r.get("kind") == "ölçek hakkı"]
    if not recs:
        raise ValueError(f"ölçek hakkı kararı kayıtlı değil ({log}): vazgeçme değerlendirmesinden önce loglanmalı")
    return recs[0]["choice"] == "S1'e çık"


def verify_order(ledger_records: list[dict], log: str = LOG) -> list[str]:
    """Her S1 koşusu karar satırından SONRA başlamış olmalı. İhlallerin listesini döndürür (boş = temiz)."""
    s1 = [r for r in ledger_records if str(r.get("run", "")).startswith(S1_PREFIX)]
    if not s1:
        return []
    recs = [r for r in read(log) if r.get("kind") == "ölçek hakkı"]
    if not recs:
        return [f"{len(s1)} S1 koşusu var ama sahibin ölçek hakkı kararı kayıtlı DEĞİL"]
    d = recs[0]
    bad = []
    if d["choice"] != "S1'e çık":
        bad.append(f"kayıtlı karar {d['choice']!r} ama {len(s1)} S1 koşusu var")
    for r in s1:
        if not r.get("started") or r["started"] <= d["ts_local"]:
            bad.append(f"{r.get('run')}: başlangıç {r.get('started')} karar satırından ({d['ts_local']}) ÖNCE ya da bilinmiyor")
    return bad


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--choice", choices=CHOICES)
    p.add_argument("--reason", default="")
    p.add_argument("--basis", help="S0'ın decision.json'u (decide_primary çıktısı)")
    p.add_argument("--log", default=LOG)
    p.add_argument("--verify", action="store_true", help="S1 koşuları karardan SONRA mı başlamış (koşu defterine karşı)")
    p.add_argument("--ledger", default="experiments/foton-v02-prereg/run_ledger.jsonl")
    args = p.parse_args()
    if args.verify:
        with open(args.ledger, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip().startswith("{")]
        bad = verify_order(recs, args.log)
        for b in bad:
            print("  ✗", b)
        print("SONUÇ:", "İHLAL" if bad else "S1 koşularının hepsi karar satırından sonra başlamış (ya da S1 koşusu yok)")
        sys.exit(1 if bad else 0)
    rec = record(args.choice, args.reason, args.basis, args.log)
    print(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
