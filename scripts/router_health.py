"""Router sağlığı: V6 erken uyarı kuralı, V7 kazanan seçimi, V6-G "router çöktü" ölçütü — koşu loglarından.

    .venv\\Scripts\\python scripts/router_health.py --runs checkpoints/foton-v02-tarama --tag v7 \\
        --out experiments/foton-v02/v7_ablation.json

Bu betik val loss'a BAKMAZ ve bakamaz: `load_router_rows()` yalnızca `train` satırlarının `router_*` alanlarını
ve adım numarasını okur. Ön-kayıt V7: kazanan val loss'a göre SEÇİLMEZ (3 bağımsız çekimde beklenen maksimum
0.846 sigma: ölçülen sigma = 0.88'de s̄'ye +0.4 ... +0.7 puan hediye). Seçim ölçütü router sağlığıdır.

ÖN-KAYDIN AÇIK BIRAKTIĞI VE TARAMADAN ÖNCE BURADA SABİTLENEN OKUMALAR ("Sapmalar"da da yazılı):

* "%50 bütçe" = `--stop_step 2188` (4375'in yarısının yukarı yuvarlanmışı), LR schedule TAM bütçe üzerinden:
  tarama koşusu, karar koşusunun izleyeceği yörüngenin ilk yarısının AYNISIDIR (kısaltılmış bir cosine değil).
* Pencere = koşunun SON 500 adımı (25'lik log aralığıyla son 20 `train` satırı).
* V7 (i) `router_dead = 0`: koşulun ÜÇ seed'inin üçünde de, penceredeki HER satırda ölü modül yok.
  (ii) `router_entropy_per_seq` en düşük: pencere ortalamasının seed'ler üzerinden ortalaması.
  (iii) eşitlik (fark < 1e-3): ön-kaydın alfabetik sırası — regularizasyon yok < entropi bonusu < load-balancing aux.
  Hiçbir koşul (i)'i geçemezse (ön-kayıt bu durumu yazmıyor): en az ölü modüllü koşul(lar) arasında (ii), (iii);
  ve bu durum çıktıda `fallback: true` olarak işaretlenir — rapor onu açıkça yazar.
* "Ölü modül" = log aralığındaki yük payı ideal payın (1/3) %10'undan az (`src/train.RouterLog`, `src/moe.py` eşiği).
* V6 erken uyarı (yalnızca tarama tier'ı, yalnızca X kolu; H'de çökme beklenen cevaptır): "ilk 500 adımda
  kalıcı" = 500. adımda biten ARDIŞIK ÜÇ log aralığının (450, 475, 500) üçünde de ihlal. "İlk 500 adımdaki
  herhangi üç ardışık aralık" okuması her koşuda ateşlenirdi: init'te router logit'leri ~0 olduğu için dizi
  başına entropi ilk aralıklarda zorunlu olarak ~1.0'dır (ölçülerek görüldü).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping

CONDITIONS = ("none", "entropy", "lb")           # ön-kaydın eşitlik sırası: regularizasyon yok < entropi < lb
PRESET_OF = {"none": "foton-x", "entropy": "foton-x-ent", "lb": "foton-x-lb"}
RUN_OF = {"none": "x", "entropy": "x-ent", "lb": "x-lb"}
WINDOW_STEPS = 500
TIE_TOL = 1e-3
EARLY_STEPS = (450, 475, 500)
EARLY_MARGINAL_BELOW = 0.1
EARLY_PER_SEQ_ABOVE = 0.9
ROUTER_FIELDS = ("router_entropy_marginal", "router_entropy_per_seq", "router_dead", "router_load_cv",
                 "router_margin", "router_switch_rate", "router_counts", "router_domain_counts")


def load_router_rows(run_dir: str) -> list[dict]:
    """Bir koşunun `train` satırlarından YALNIZCA adım + router alanları. val_loss bilerek OKUNMAZ."""
    rows = []
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "train" and "router_dead" in r:
                # v03 adım 1'in zorlanmış-atama kollarında router ÇALIŞMAZ ve bu alanlar `null` ("uygulanamaz") yazılır:
                # o koşulardan router sağlığı / V7 seçimi OKUNAMAZ — sessizce sayı gibi yorumlanmasın.
                if any(r.get(k) is None for k in ("router_dead", "router_entropy_per_seq")):
                    raise ValueError(f"{run_dir}: zorlanmış-atama kolu — router metrikleri UYGULANAMAZ")
                rows.append({"step": r["step"], **{k: r.get(k) for k in ROUTER_FIELDS}})
    if not rows:
        raise ValueError(f"{run_dir}: router alanı taşıyan train satırı yok (FOTON koşusu değil mi?)")
    return sorted(rows, key=lambda r: r["step"])


def window(rows: list[dict]) -> list[dict]:
    """Koşunun son `WINDOW_STEPS` adımındaki satırlar."""
    last = rows[-1]["step"]
    w = [r for r in rows if r["step"] > last - WINDOW_STEPS]
    if not w:
        raise ValueError("pencere boş")
    return w


def early_warning(rows: list[dict]) -> list[str]:
    """V6 erken uyarı tetikleri (boş liste = temiz). Yalnızca tarama tier'ındaki X koşularına uygulanır."""
    by_step = {r["step"]: r for r in rows}
    missing = [s for s in EARLY_STEPS if s not in by_step]
    if missing:
        raise ValueError(f"erken uyarı için gereken log adımları yok: {missing}")
    trio = [by_step[s] for s in EARLY_STEPS]
    out = []
    if all(r["router_entropy_marginal"] < EARLY_MARGINAL_BELOW for r in trio):
        out.append(f"ÇÖKME: marjinal entropi {EARLY_STEPS} adımlarının üçünde de < {EARLY_MARGINAL_BELOW}")
    if all(r["router_entropy_per_seq"] > EARLY_PER_SEQ_ABOVE for r in trio):
        out.append(f"ÖĞRENMİYOR: dizi başına entropi {EARLY_STEPS} adımlarının üçünde de > {EARLY_PER_SEQ_ABOVE}")
    return out


def summarize(rows: list[dict]) -> dict:
    w = window(rows)
    mean = lambda k: sum(r[k] for r in w) / len(w)      # noqa: E731
    return {"last_step": rows[-1]["step"], "window_rows": len(w), "max_dead": max(r["router_dead"] for r in w),
            "mean_dead": mean("router_dead"), "entropy_per_seq": mean("router_entropy_per_seq"),
            "entropy_marginal": mean("router_entropy_marginal"), "load_cv": mean("router_load_cv"),
            "margin": mean("router_margin"), "final_counts": w[-1]["router_counts"]}


def v7_select(runs: Mapping) -> dict:
    """`runs`: `dict[koşul] -> dict[seed] -> satırlar` (koşul ∈ CONDITIONS). Kazananı ve gerekçesini döner.

    Girdi KİMLİKLE anahtarlıdır (koşul, seed); liste kabul edilmez. Her koşulun seed kümesi AYNI olmalıdır.
    """
    if not isinstance(runs, Mapping) or set(runs) != set(CONDITIONS):
        raise ValueError(f"tam üç koşul gerekir {CONDITIONS}, verilen: {list(runs) if isinstance(runs, Mapping) else type(runs).__name__}")
    seed_sets = {c: set(runs[c]) for c in CONDITIONS}
    if len({frozenset(s) for s in seed_sets.values()}) != 1 or not seed_sets["none"]:
        raise ValueError(f"her koşul AYNI seed kümesinde koşulmalı: {seed_sets}")
    per = {}
    for c in CONDITIONS:
        s = {seed: summarize(rows) for seed, rows in sorted(runs[c].items())}
        n = len(s)
        per[c] = {"seeds": s, "passes_dead_filter": all(v["max_dead"] == 0 for v in s.values()),
                  "mean_dead": sum(v["mean_dead"] for v in s.values()) / n,
                  "entropy_per_seq": sum(v["entropy_per_seq"] for v in s.values()) / n}
    alive = [c for c in CONDITIONS if per[c]["passes_dead_filter"]]
    fallback = not alive
    if fallback:
        least = min(per[c]["mean_dead"] for c in CONDITIONS)
        alive = [c for c in CONDITIONS if per[c]["mean_dead"] <= least + 1e-12]
    best = min(per[c]["entropy_per_seq"] for c in alive)
    tied = [c for c in alive if per[c]["entropy_per_seq"] - best < TIE_TOL]
    winner = tied[0]                                     # CONDITIONS sırası = ön-kaydın eşitlik sırası
    why = (f"(i) router_dead = 0 süzgecini geçenler: {alive if not fallback else 'YOK'}; "
           + (f"YEDEK kural: en az ölü modüllü koşul(lar) {alive}; " if fallback else "")
           + f"(ii) en düşük dizi-başı entropi {best:.4f}; "
           + (f"(iii) eşitlik ({tied}) -> ön-kayıt sırası; " if len(tied) > 1 else "")
           + f"kazanan: {winner} ({PRESET_OF[winner]})")
    return {"winner": winner, "winner_preset": PRESET_OF[winner], "fallback": fallback, "tied": tied,
            "reason": why, "conditions": per, "selection_reads_val_loss": False,
            "rule": {"window_steps": WINDOW_STEPS, "tie_tol": TIE_TOL, "order": list(CONDITIONS)}}


def collapsed(rows: list[dict]) -> dict:
    """V6-G "router çöktü": son 500 adımda `router_dead >= 2` ya da marjinal entropi < 0.1 (pencere ORTALAMASI)."""
    s = summarize(rows)
    return {"collapsed": s["max_dead"] >= 2 or s["entropy_marginal"] < EARLY_MARGINAL_BELOW, **s}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="tarama koşularının dizini")
    ap.add_argument("--tag", default="v7", help="koşu adı öneki (make_campaign_queue --tag)")
    ap.add_argument("--seeds", default="101,102,103")
    ap.add_argument("--out", default="experiments/foton-v02/v7_ablation.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = {c: {s: load_router_rows(os.path.join(args.runs, f"{args.tag}-{RUN_OF[c]}-s{s}")) for s in seeds}
            for c in CONDITIONS}
    res = v7_select(runs)
    res["early_warning"] = {c: {str(s): early_warning(rows) for s, rows in runs[c].items()} for c in CONDITIONS}
    for c in CONDITIONS:
        p = res["conditions"][c]
        print(f"{c:8s} ölü-süzgeci {'GEÇTİ' if p['passes_dead_filter'] else 'kaldı':6s} | ort. ölü {p['mean_dead']:.2f} "
              f"| dizi-başı H {p['entropy_per_seq']:.4f} | erken uyarı {res['early_warning'][c]}")
        for s, v in p["seeds"].items():
            print(f"    s{s}: maks ölü {v['max_dead']} | dizi-başı H {v['entropy_per_seq']:.4f} | marjinal H "
                  f"{v['entropy_marginal']:.4f} | yük CV {v['load_cv']:.3f} | son yük {v['final_counts']}")
    print(res["reason"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("->", args.out)


if __name__ == "__main__":
    main()
