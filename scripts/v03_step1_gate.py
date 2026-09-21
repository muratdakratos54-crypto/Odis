"""v03 adım 1 — SEED 101 GEÇERLİLİK KAPISI (sahibin talimatı, koşulardan ÖNCE yazıldı). δ ve δ_D HESAPLANMAZ.

    .venv/Scripts/python scripts/v03_step1_gate.py --out checkpoints/foton-v03-step1/gate_s101.json

Seed 101'in dört kolu (X-dom, H-dom, X-iid, D) bitince, seed 102 başlamadan ÖNCE yalnızca iki geçerlilik kapısına bakılır:

  1. ön koşulun SEED-BAŞI bacağı: X-iid-101, aynı seed'li v02-X'ten sapmamış mı (|mean_fixed farkı| ≤ 0.015 nat VE
     |oracle tavanı farkı| ≤ 0.4 puan) — `v03_step1_decision.precondition_seed_leg` (kuralın TEK kopyası);
  2. D akla-yatkınlık denetiminin SEED-BAŞI bacağı: |val(D-101, adım 2000) − 3.1198| ≤ 0.012 nat —
     `v03_step1_decision.d_sanity_seed_leg`.

İşaretli-ORTALAMA bacakları (Ö1, Ö2) üç seed ister; burada bakılmaz, kuyruk bitince bakılır. Kapılardan biri AŞILIRSA
çıkış kodu 3'tür ve kampanya sürücüsü (`v03_step1_campaign.py`) kuyruğun kalanını BAŞLATMAZ; sahibe rapor edilir.

ARA SONUÇ YOK — ne yazılır, ne yazılmaz:
  * Kapı AŞILMADIĞINDA (kuyruk sürecek): çıktı YALNIZCA iki bayraktır (`clean`); hiçbir sayı yazılmaz ve basılmaz.
    Sebebi ölçülü: iki kapının kendi nicelikleri (mean_fixed farkı, D'nin adım-2000 sapması) diskteki sabitlerle
    toplandığında X-iid-101'in `val_atanan`'ını ve D-101'in adım-2000 kaybını verir — X-iid kolunda `val_atanan`
    TANIM GEREĞİ `mean_fixed`'dir. Kuyruk sürerken bu sayılar da görünmesin.
  * Kapı AŞILDIĞINDA (kampanya durur, sahibe rapor gider): iki kapının nicelikleri yazılır — sahibin kararı için gerekli.
    Bu durumda bile X-dom / H-dom'un kaybı, D'nin adım-2188 kaybı, δ, δ_D ve maske teşhisi ne hesaplanır ne yazılır.
  * Hiçbir durumda δ / δ_D hesaplanmaz: X-dom ve H-dom'un kaybı çıktının hiçbir yerinde yoktur. 12 koşu bitmeden
    hiçbir δ hesaplanmayacak.

SEED 102 / 103 KAPILARI (sahibin talimatı, 2026-09-21, kalan 8 koşu onaylanırken): "Seed başına ön koşul bacağı
yürürlükte: 102 veya 103'te aşılırsa kuyruk durur ve sabahı bekler." Bu kapılar `--legs precondition` ile çağrılır ve
YALNIZCA ön koşulun seed-başı bacağına bakar. D akla-yatkınlık bacağına BAKILMAZ: (b) bu turda zaten okunmaz (sahibin kararı)
ve D-102 / D-103'ün değeri teşhistir, kuyruğu durdurmak değil. Dört koşunun bütünlüğü (D dahil) yine denetlenir.

Ayrıca dört koşunun BÜTÜNLÜĞÜ analizle AYNI fonksiyonlarla denetlenir (kimlik, bütçe, LR takvimi, iki yol, atama tablosu,
val penceresi özeti, veri sırası ve başlangıç parmak izi) ve seed-101 koşularının src özeti ŞU ANKİ ağacınkiyle
karşılaştırılır (kuyruk ikiye bölündüğü için `run_queue`'nun kuyruk-içi src denetimi iki kuyruğun ARASINI görmez: seed 102
başka bir src ile koşarsa analiz 12 koşu bittikten sonra karar üretmezdi — 3.3 GPU saat boşa). Bütünlük hatası çıkış kodu
4'tür ve o da kuyruğu durdurur; bu bir ara sonuç değildir, hiçbir kayıp sayısı üretmez.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_queue  # noqa: E402
import v03_step1_analyze as an  # noqa: E402

dec = an.dec
GATE_SEED = 101
LEGS = ("both", "precondition")          # "both": seed 101 kapısı (onaylı ilk hâl) · "precondition": seed 102 / 103 kapıları
EXIT_OK, EXIT_EXCEEDED, EXIT_INTEGRITY = 0, 3, 4
D_NOT_EVALUATED = ("BAKILMADI: seed 102 / 103'te yalnızca ön koşulun seed-başı bacağı yürürlükte (sahibin talimatı, "
                   "2026-09-21); (b) bu turda zaten okunmaz, D-102 / D-103'ün değeri teşhistir")
NOT_COMPUTED = ("X-dom / H-dom'un kaybı, D'nin adım-2188 kaybı, delta, delta_D ve maske teşhisi bu betikte HESAPLANMAZ / "
                "YAZILMAZ: 12 koşu bitmeden hiçbir delta hesaplanmayacak (sahibin talimatı). Kapı AŞILMADIYSA hiçbir sayı "
                "yazılmaz; aşıldıysa yalnızca iki kapının kendi nicelikleri yazılır (bunlar X-iid-101'in mean_fixed'ini ve "
                "D-101'in adım-2000 kaybını sabit bir kaydırmayla taşır — kampanya o durumda zaten durur ve sahibe rapor edilir).")


def not_computed(seed: int, legs: str) -> str:
    """Kaydın kendisi yanlış hatırlatmasın: 'yalnız ön koşul' kipinde D niceliği hiç yazılmaz ve nicelikler O seed'indir."""
    if legs == "both":
        return NOT_COMPUTED
    return ("X-dom / H-dom'un kaybı, D'nin kaybı (hiçbir adımda), delta, delta_D ve maske teşhisi bu betikte HESAPLANMAZ / "
            "YAZILMAZ: 12 koşu bitmeden hiçbir delta hesaplanmayacak (sahibin talimatı). Bu kipte YALNIZCA ön koşulun seed-başı "
            f"bacağına bakılır. Kapı AŞILMADIYSA hiçbir sayı yazılmaz; aşıldıysa yalnızca ön koşulun nicelikleri yazılır (bunlar "
            f"X-iid-{seed}'in mean_fixed'ini sabit bir kaydırmayla taşır — kuyruk o durumda durur ve sabahı bekler).")


def gate(runs_dir: str, v02_dir: str, seed: int = GATE_SEED, tag: str = "v03s1", root: str = ROOT, legs: str = "both") -> dict:
    """Kapının sonucu (+ AŞILDIYSA nicelikleri). `legs`: "both" = ön koşul + D (seed 101); "precondition" = yalnız ön koşul
    (seed 102 / 103). Bütünlük hatası -> ValueError (analizin kendi denetimleri; iki kipte de dört koşunun HEPSİ)."""
    if legs not in LEGS:
        raise ValueError(f"legs {legs!r}, beklenen {LEGS}")
    v02 = {seed: an.read_v02(os.path.join(v02_dir, f"v7-x-s{seed}"), seed)}
    runs = {arm: {seed: an.read_run(os.path.join(runs_dir, f"{tag}-{arm}-s{seed}"), arm, seed, tag, v02[seed]["config"])}
            for arm in an.ARMS}
    an.check_pairing(runs, v02, seeds=(seed,))
    prov = [runs[arm][seed]["provenance"] for arm in an.ARMS]
    src = {str(p["src_sha256"]) for p in prov}
    if not all(p["code_dirty"] is False for p in prov) or len(src) != 1:
        raise ValueError(f"seed {seed}: karar koşularında code_dirty = False ve TEK src özeti zorunlu, istisnasız: {sorted(src)}")
    now = run_queue.code_hash(root)
    if src != {now}:            # kuyruk B başka bir src ile koşarsa analiz 12 koşudan SONRA karar üretmezdi
        raise ValueError(f"seed {seed} koşulduğundan beri src/ değişti ({src.pop()[:12]} -> {now[:12]}): kalan 8 koşu başka "
                         "kodla koşardı ve 12 koşu tek src özeti taşımazdı")
    iid, ref = runs["x-iid"][seed], v02[seed]
    dm, dc = iid["mean_fixed"] - ref["mean_fixed"], iid["ceiling_points"] - ref["ceiling_points"]
    pre_ok = dec.precondition_seed_leg(dm, dc)
    pre: dict = {"clean": bool(pre_ok)}
    if legs == "both":
        d_val = runs["d"][seed]["val_curve"][dec.D_REF_STEP]
        d_ok = dec.d_sanity_seed_leg(d_val)
        dsn: dict | str = {"clean": bool(d_ok)}
        exceeded = not (pre_ok and d_ok)
    else:                       # seed 102 / 103: D'ye BAKILMAZ (sahibin talimatı); kapı yalnız ön koşuldur
        dsn, exceeded = D_NOT_EVALUATED, not pre_ok
    if exceeded:                # kampanya durdu: sahibin kararı için nicelikler gerekli
        pre.update(mean_fixed_diff_nat=dm, ceiling_diff_points=dc, limits=[dec.PRE_MEAN_FIXED_NAT, dec.PRE_CEILING_POINTS])
        if legs == "both":
            dsn.update(deviation_nat_at_ref_step=d_val - dec.D_REF_VAL, reference=[dec.D_REF_STEP, dec.D_REF_VAL, dec.D_REF_TOL])
    which = "yalnızca seed-başı bacaklar" if legs == "both" else "YALNIZCA ön koşulun seed-başı bacağı"
    return {
        "what": f"v03 adım 1 — seed {seed} geçerlilik kapısı ({which}; kural: scripts/v03_step1_decision.py)",
        "seed": seed, "integrity": "dört koşu analizin kimlik / bütçe / LR / iki-yol / atama / eşleme / src denetimlerinden geçti",
        "precondition_seed_leg": pre, "d_sanity_seed_leg": dsn, "exceeded": bool(exceeded), "NOT_computed": not_computed(seed, legs),
    }


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "checkpoints", "foton-v03-step1"))
    ap.add_argument("--v02", default=os.path.join(ROOT, "checkpoints", "foton-v02-tarama"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=GATE_SEED, choices=dec.SEEDS)
    ap.add_argument("--legs", default="both", choices=LEGS)
    args = ap.parse_args(argv)
    try:
        res, rc = gate(args.runs, args.v02, seed=args.seed, legs=args.legs), None
    except Exception as e:      # bozuk / eksik / beklenmedik: kuyruk DURUR ve sebep DOSYAYA yazılır (kapı "geçti" sayılmaz)
        res, rc = {"seed": args.seed, "legs": args.legs, "integrity_error": f"{type(e).__name__}: {e}", "exceeded": None,
                   "NOT_computed": not_computed(args.seed, args.legs)}, EXIT_INTEGRITY
    if rc is None:
        rc = EXIT_EXCEEDED if res["exceeded"] else EXIT_OK
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print({EXIT_OK: "KAPI AŞILMADI: kuyruk sürer, ara sonuç yok", EXIT_EXCEEDED: "KAPI AŞILDI: kuyruk DURUR, sahibe rapor",
           EXIT_INTEGRITY: "BÜTÜNLÜK HATASI: kuyruk DURUR, sahibe rapor"}[rc])
    return rc


if __name__ == "__main__":
    sys.exit(main())
