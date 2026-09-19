"""Üç modülün dikkat grafikleri birbirinden NE KADAR farklı — bağlam uzunluğunun fonksiyonu olarak (kombinatorik; koşu yok).

    .venv\\Scripts\\python scripts/topology_overlap.py --out experiments/foton-v02/topology_overlap.json

v03 tasarım notlarındaki hipotezin SAYILABİLEN kısmı: "T=512'de üç grafik de geçmişin yarısını görüyor; hangi yarısı
olduğu ikinci dereceden bir etki. Topoloji, pencere diziye göre küçük olduğunda ayrışır." Bu betik hipotezi SINAMAZ
(kayıp farkını ölçmez); yalnızca grafiklerin kendisini sayar — DONMUŞ desenlerle (`src/attn_patterns.PATTERNS`), T×T
maske kurmadan, ofset kümeleri üzerinden (T=8192'de de ucuz):

  * yoğunluk          : modülün head-ortalaması bağlantısı / tam nedensel bağlantı ((T+1)/2);
  * kapsama           : modülün head'lerinin BİRLEŞİMİNİN gördüğü (i, j) çiftlerinin tam nedensel çiftlere oranı;
  * Jaccard(X, Y)     : iki modülün birleşim grafiklerinin kesişimi / birleşimi (1 = aynı grafik, 0 = ayrık);
  * ortalama erişim   : görülen anahtarların sorguya ortalama uzaklığı (token).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src import attn_patterns as ap  # noqa: E402

MODULES = ("A", "B", "C")


def union_offsets(name: str, T: int) -> np.ndarray:
    """Modülün head'lerinden EN AZ BİRİNİN gördüğü ofsetler (sorgu i, anahtar i - ofset), bool [T]."""
    seen = np.zeros(T, dtype=bool)
    for head, _ in ap.PATTERNS[name].head_groups():
        seen[head.offsets(T)] = True
    return seen


def pair_count(offsets: np.ndarray) -> int:
    """Ofset kümesinin nedensel maske altında ürettiği (i, j) çifti sayısı: ofset k, T - k sorguda kullanılabilir."""
    T = len(offsets)
    return int((T - np.flatnonzero(offsets)).sum())


def analyze(T: int, n_heads: int = ap.N_HEAD_KAMPANYA) -> dict:
    full_pairs = T * (T + 1) // 2
    unions = {m: union_offsets(m, T) for m in MODULES}
    out = {"T": T, "full_causal_avg_connections": (T + 1) / 2, "modules": {}, "pairs": {}}
    for m, u in unions.items():
        avg = ap.PATTERNS[m].avg(T, n_heads)
        k = np.flatnonzero(u)
        out["modules"][m] = {
            "avg_connections_per_token_head": avg,
            "density_vs_full_causal": avg / ((T + 1) / 2),
            "union_coverage_of_causal_pairs": pair_count(u) / full_pairs,
            "mean_reach_tokens": float((k * (T - k)).sum() / (T - k).sum()),
            "max_reach_tokens": int(k.max()),
        }
    for a, b in itertools.combinations(MODULES, 2):
        inter, union = pair_count(unions[a] & unions[b]), pair_count(unions[a] | unions[b])
        out["pairs"][f"{a}-{b}"] = {"jaccard": inter / union,
                                    "share_of_a_also_in_b": inter / pair_count(unions[a]),
                                    "share_of_b_also_in_a": inter / pair_count(unions[b])}
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--T", default="512,2048,4096,8192")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    res = {"what": "donmuş üç desenin grafik örtüşmesi (kombinatorik; kayıp ölçümü DEĞİL)",
           "frozen_spec": {k: getattr(ap, k) for k in ("A_LOCAL_W", "B_DILATION_R", "B_BUDGET_W", "C_LOCAL_W", "C_STRIDE_S", "C_STRIDE_W")},
           "by_T": [analyze(int(t)) for t in args.T.split(",")]}
    for r in res["by_T"]:
        mods = " | ".join(f"{m}: yoğunluk %{100 * v['density_vs_full_causal']:.1f}, kapsama %{100 * v['union_coverage_of_causal_pairs']:.1f}, "
                          f"ort. erişim {v['mean_reach_tokens']:.0f}" for m, v in r["modules"].items())
        jac = ", ".join(f"J({k}) = {v['jaccard']:.2f}" for k, v in r["pairs"].items())
        print(f"T={r['T']:5d} | {mods} | {jac}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("->", args.out)


if __name__ == "__main__":
    main()
