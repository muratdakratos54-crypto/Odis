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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows konsolu cp1252 olabilir
    p = argparse.ArgumentParser()
    p.add_argument("--arms", required=True, help="preset adları, sırası KORUNUR (ör. foton-d,foton-m,foton-h,foton-x)")
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--out", required=True)
    p.add_argument("--out_dir", default="checkpoints/foton-v02")
    p.add_argument("--stop_step", type=int, default=0, help=">0 ise kısaltılmış (tarama) koşu")
    p.add_argument("--tag", default="", help="koşu adı öneki (ör. screen)")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = []
    for bi, s in enumerate(seeds):        # DIŞ döngü seed: blok = bir seed'in dört kolu
        # Blok İÇİ sıra her blokta KAYDIRILIR (Latin kare). Sabit sıra (her blokta D,M,H,X)
        # blok-içi pozisyonu kolla TAM olarak karıştırırdı: D her zaman soğuk GPU'da ilk, X her
        # zaman ısınmış GPU'da son. Bu, V11a'nın blokları arası için kırdığı confound'un aynısını
        # blok içine taşımaktı. Kaydırma bedava: her kol her pozisyonu eşit sayıda görür.
        order = arms[bi % len(arms):] + arms[:bi % len(arms)]
        for a in order:
            short = a.replace("foton-", "")
            name = f"{args.tag + '-' if args.tag else ''}{short}-s{s}"
            cmd = ["--preset", a, "--seed", str(s)]
            if args.stop_step:
                cmd += ["--stop_step", str(args.stop_step)]
            runs.append({"name": name, "args": cmd, "_block": f"s{s}", "_arm": short})
    q = {"out_dir": args.out_dir, "cwd": ".", "python": ".venv/Scripts/python.exe", "runs": runs,
         "_order": "seed-block interleaved + Latin-square counterbalanced (PREREGISTRATION_V02 V11a)",
         "_blocks": [[r["name"] for r in runs if r["_block"] == f"s{s}"] for s in seeds]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(q, f, indent=1, ensure_ascii=False)
    print(f"{len(runs)} koşu, {len(seeds)} seed bloğu x {len(arms)} kol -> {args.out}")
    for i, s in enumerate(seeds):
        block = [r["name"] for r in runs if r["_block"] == f"s{s}"]
        print(f"  blok {i+1} (s{s}): " + ", ".join(block))
    # kontrol 1: her kol her blokta TAM BİR KEZ
    short_arms = [a.replace("foton-", "") for a in arms]
    for s in seeds:
        block = [r["_arm"] for r in runs if r["_block"] == f"s{s}"]
        assert sorted(block) == sorted(short_arms), f"blok s{s} kol kümesi bozuk: {block}"
    # kontrol 2: Latin kare — her kol her blok-içi POZİSYONU eşit sayıda görür (tam bölünüyorsa)
    pos: dict[str, list[int]] = {a: [] for a in short_arms}
    for s in seeds:
        for i, r in enumerate([r for r in runs if r["_block"] == f"s{s}"]):
            pos[r["_arm"]].append(i)
    print("kontrol: her blokta her kol tam bir kez — OK")
    print("Latin kare, kol -> gördüğü blok-içi pozisyonlar:")
    for a in short_arms:
        print(f"  {a:4s} {pos[a]}  (ortalama pozisyon {sum(pos[a])/len(pos[a]):.2f})")
    if len(seeds) % len(arms) == 0:
        for a in short_arms:
            assert sorted(pos[a]) == sorted(range(len(arms)) ) * (len(seeds)//len(arms)), \
                f"{a} pozisyonları dengeli değil: {pos[a]}"
        print("  -> pozisyonlar TAM dengeli (seed sayısı kol sayısının katı)")
    else:
        print(f"  -> seed sayısı ({len(seeds)}) kol sayısının ({len(arms)}) katı DEĞİL: "
              f"pozisyon dengesi kısmi, ortalama pozisyonlar yukarıda raporlanıyor")


if __name__ == "__main__":
    main()
