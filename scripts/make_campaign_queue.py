"""FOTON kampanya kuyruğunu SEED BLOĞU hâlinde (iç içe) üretir — PREREGISTRATION_V02.md V11a şartı.

    .venv\\Scripts\\python scripts/make_campaign_queue.py --arms foton-d,foton-m,foton-h,foton-x \\
        --seeds 11,22,33,44,55 --out experiments/foton-v02-prereg/queues/decision.json

Neden mekanik üretiliyor: sıra elle dizilirse kayar. V11a bağlayıcı olarak şunu istiyor —

    D-s11, M-s11, H-s11, X-s11,  D-s22, M-s22, H-s22, X-s22,  ...

Gerekçe ölçümle (baseline foton-m, 5 koşu, aynı config): medyan tok/s 52.8k–66.6k, koşular arası
CV %9.02, en yavaş/en hızlı 1.263x. Dahası gürültü zaman-korelasyonlu: s11'in son iki diliminde
başlayan yavaşlık s22'nin ilk dört dilimine taşındı (~40 dk'lık girişim penceresi koşu sınırını
aştı). Kol bloğu hâlinde koşmak böyle bir pencerenin TEK kola düşmesine yol açar ve %9 oturum
gürültüsüyle n=5'te duvar saati farkının GA yarı-genişliği ±11.2 puan olur — ölçüt işlevsiz kalır.
Seed bloğu aynı pencereyi dört kolun paylaşmasını sağlar. Maliyeti sıfır.
"""
from __future__ import annotations

import argparse
import sys
import json
import os


S1_SUFFIX = "-s1"                 # S1 preset adları bununla biter (ön-kayıt: "Her satırdan sonra ne olur")
S1_PREFIX = "s1-"                 # ... ve S1 koşu adları bu öneki alır (`owner_decision.verify_order` buna bakar)


def short(arm: str) -> str:
    """Kolun kısa adı: `foton-x` ve `foton-x-s1` -> `x`."""
    a = arm.replace("foton-", "")
    return a[:-len(S1_SUFFIX)] if a.endswith(S1_SUFFIX) else a


def scale_of(arms: list[str]) -> str:
    """Kuyruğun ölçeği KOL ADLARINDAN çıkar; ayrı bir bayrak yok ki S1 preset'leri "S0 kuyruğu" diye
    üretilip sahibin karar kapısı atlanamasın. Karışık ölçek hata."""
    s1 = [a.endswith(S1_SUFFIX) for a in arms]
    if any(s1) and not all(s1):
        raise ValueError(f"bir kuyrukta tek ölçek olur; karışık kollar: {arms}")
    return "S1" if all(s1) and arms else "S0"


def build_runs(arms: list[str], seeds: list[int], tag: str = "", stop_step: int = 0) -> list[dict]:
    """Seed bloğu × Latin kare sırasıyla koşu listesi. Saf fonksiyon: testler ve mutasyon sınavı için."""
    runs = []
    for bi, s in enumerate(seeds):        # DIŞ döngü seed: blok = bir seed'in bütün kolları
        # Blok İÇİ sıra her blokta KAYDIRILIR (Latin kare). Sabit sıra (her blokta D,M,H,X)
        # blok-içi pozisyonu kolla TAM olarak karıştırırdı: D her zaman soğuk GPU'da ilk, X her
        # zaman ısınmış GPU'da son. Bu, V11a'nın blokları arası için kırdığı confound'un aynısını
        # blok içine taşımaktı. Kaydırma bedava: her kol her pozisyonu eşit sayıda görür.
        order = arms[bi % len(arms):] + arms[:bi % len(arms)]
        for a in order:
            pre = S1_PREFIX if a.endswith(S1_SUFFIX) else ""
            name = f"{pre}{tag + '-' if tag else ''}{short(a)}-s{s}"
            cmd = ["--preset", a, "--seed", str(s)]
            if stop_step:
                cmd += ["--stop_step", str(stop_step)]
            runs.append({"name": name, "args": cmd, "_block": f"s{s}", "_arm": short(a)})
    return runs


def positions(runs: list[dict], arms: list[str], seeds: list[int]) -> dict[str, list[int]]:
    pos: dict[str, list[int]] = {short(a): [] for a in arms}
    for s in seeds:
        for i, r in enumerate([r for r in runs if r["_block"] == f"s{s}"]):
            pos[r["_arm"]].append(i)
    return pos


def check_balance(runs: list[dict], arms: list[str], seeds: list[int]) -> None:
    """V11a'nın iki şartını ASSERTION olarak denetler; ihlalde kuyruk ÜRETİLMEZ.

    (1) her kol her blokta tam bir kez;
    (2) Latin kare dengesi: her kol her blok-içi pozisyonu en az ⌊blok/kol⌋ kez görür.

    (2)'nin ilk hâli yalnızca `blok % kol == 0` iken çalışıyordu — yani kampanyanın GERÇEK
    biçiminde (5 seed × 4 kol) hiç çalışmıyordu. Mutasyon sınavı bunu yakaladı: rotasyon satırı
    silinip sıra sabitlendiğinde hiçbir assertion ateşlenmiyordu. Yakalamadığı kanıtlanmamış
    guard, guard değildir.
    """
    shorts = [short(a) for a in arms]
    for s in seeds:
        block = [r["_arm"] for r in runs if r["_block"] == f"s{s}"]
        assert sorted(block) == sorted(shorts), f"blok s{s} kol kümesi bozuk: {block}"
    pos = positions(runs, arms, seeds)
    floor = len(seeds) // len(arms)
    for a in shorts:
        for k in range(len(arms)):
            seen = pos[a].count(k)
            assert seen >= floor, (f"Latin kare dengesi bozuk: {a} kolu {k}. pozisyonu {seen} kez "
                                   f"gördü, en az {floor} olmalı — blok-içi sıra kolla karışıyor")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows konsolu cp1252 olabilir
    p = argparse.ArgumentParser()
    p.add_argument("--arms", required=True, help="preset adları, sırası KORUNUR (ör. foton-d,foton-m,foton-h,foton-x)")
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--out", required=True)
    p.add_argument("--out_dir", default="checkpoints/foton-v02")
    p.add_argument("--stop_step", type=int, default=0, help=">0 ise kısaltılmış (tarama) koşu")
    p.add_argument("--tag", default="", help="koşu adı öneki (ör. screen)")
    p.add_argument("--owner_log", default=None, help="sahibin karar defteri (varsayılan: owner_decision.LOG)")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    scale = scale_of(arms)
    decision = None
    if scale == "S1":
        # Sahibin S1'e çıkma seçimi S1 KOŞULMADAN ÖNCE, zaman damgalı ve gerekçeli tek satır olarak
        # loglanmış olmalı; o satır yoksa S1 kuyruğu ÜRETİLMEZ (seçim S1'in sonucuna göre yapılamasın).
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import owner_decision
        try:
            decision = owner_decision.require_s1_authorized(args.owner_log or owner_decision.LOG)
        except PermissionError as e:
            sys.exit(f"HATA: {e}")
    runs = build_runs(arms, seeds, tag=args.tag, stop_step=args.stop_step)
    check_balance(runs, arms, seeds)
    q = {"out_dir": args.out_dir, "cwd": ".", "python": ".venv/Scripts/python.exe", "runs": runs,
         "_scale": scale, "_owner_decision": decision,
         "_order": "seed-block interleaved + Latin-square counterbalanced (PREREGISTRATION_V02 V11a)",
         "_blocks": [[r["name"] for r in runs if r["_block"] == f"s{s}"] for s in seeds]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(q, f, indent=1, ensure_ascii=False)
    print(f"{len(runs)} koşu, {len(seeds)} seed bloğu x {len(arms)} kol -> {args.out}")
    for i, s in enumerate(seeds):
        block = [r["name"] for r in runs if r["_block"] == f"s{s}"]
        print(f"  blok {i+1} (s{s}): " + ", ".join(block))
    pos = positions(runs, arms, seeds)
    print("kontrol: her blokta her kol tam bir kez; Latin kare dengesi assertion'dan geçti — OK")
    print("kol -> gördüğü blok-içi pozisyonlar:")
    for a in [short(x) for x in arms]:
        print(f"  {a:4s} {pos[a]}  (ortalama pozisyon {sum(pos[a])/len(pos[a]):.2f})")

if __name__ == "__main__":
    main()
