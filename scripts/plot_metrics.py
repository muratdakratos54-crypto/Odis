"""Koşuların metriklerini (checkpoints/<run>/metrics.jsonl) grafiğe döker.

    .venv\\Scripts\\python scripts/plot_metrics.py --runs dense moe moe-noaux --out checkpoints/compare.png

Çizilenler:
  * Val loss (cross-entropy) - adım ve işlenen token sayısına göre
  * MoE koşuları için: yük dengesizliği (katmanların ortalama CV'si) zaman içinde, aux loss,
    ve eğitimin son log aralığında katman x expert yük ısı haritası (1.0 = ideal pay 1/E)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # ekran gerektirmeyen arka uç: sadece dosyaya çiz
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(run_dir: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"train": [], "val": [], "sample": []}
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.setdefault(r["type"], []).append(r)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["dense"])
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out", default="checkpoints/compare.png")
    args = p.parse_args()

    data = {r: load(os.path.join(args.ckpt_dir, r)) for r in args.runs}
    moe_runs = [r for r in args.runs if data[r]["train"] and "load" in data[r]["train"][-1]]
    color = {r: f"C{i}" for i, r in enumerate(args.runs)}  # her koşu tüm grafiklerde aynı renk

    n_rows = 1 + (1 if moe_runs else 0) + len(moe_runs)
    fig = plt.figure(figsize=(13, 4.2 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 2)

    ax = fig.add_subplot(gs[0, 0])
    for r in args.runs:
        v = data[r]["val"]
        ax.plot([x["step"] for x in v], [x["val_loss"] for x in v], marker="o", ms=3, label=r, color=color[r])
    ax.set(title="Val loss (cross-entropy)", xlabel="adım", ylabel="loss")
    ax.set_ylim(top=min(3.0, ax.get_ylim()[1]))
    ax.grid(alpha=0.3)
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    for r in args.runs:
        t = data[r]["train"]
        key = "ce" if t and "ce" in t[0] else "loss"
        ax.plot([x["step"] for x in t], [x[key] for x in t], lw=1, label=r, color=color[r])
    ax.set(title="Train cross-entropy", xlabel="adım", ylabel="loss")
    ax.set_ylim(top=min(3.0, ax.get_ylim()[1]))
    ax.grid(alpha=0.3)
    ax.legend()

    if moe_runs:
        ax = fig.add_subplot(gs[1, 0])
        for r in moe_runs:
            t = data[r]["train"]
            ax.plot([x["step"] for x in t], [x["load_cv"] for x in t], label=r, color=color[r])
        ax.set(title="Expert yük dengesizliği (CV = std/ortalama; 0 = mükemmel denge)", xlabel="adım", ylabel="CV")
        ax.grid(alpha=0.3)
        ax.legend()

        ax = fig.add_subplot(gs[1, 1])
        for r in moe_runs:
            t = data[r]["train"]
            ax.plot([x["step"] for x in t], [x["aux_loss"] for x in t], label=r, color=color[r])
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set(title="Load-balancing aux loss (1.0 = denge; aux_coef=0 olsa da ölçülür)", xlabel="adım")
        ax.grid(alpha=0.3)
        ax.legend()

        for i, r in enumerate(moe_runs):
            load_mat = np.array(data[r]["train"][-1]["load"])  # (MoE katmanı, expert) oranları
            ratio = load_mat * load_mat.shape[1]                # 1.0 = ideal pay
            # Satırları konsol logu ve analyze_experts.py --layer ile aynı *model* katman numarasıyla etiketle
            with open(os.path.join(args.ckpt_dir, r, "config.json")) as f:
                first_dense = json.load(f)["model"].get("first_dense_layers", 0)
            ax = fig.add_subplot(gs[2 + i, :])
            im = ax.imshow(ratio, cmap="RdBu_r", vmin=0, vmax=2, aspect="auto")
            ax.set(title=f"{r}: son log aralığında expert yükü / ideal pay (mavi = az, kırmızı = çok kullanılıyor)",
                   xlabel="expert", ylabel="katman (model indeksi)")
            ax.set_xticks(range(ratio.shape[1]))
            ax.set_yticks(range(ratio.shape[0]), [str(li + first_dense) for li in range(ratio.shape[0])])
            for (li, e), val in np.ndenumerate(ratio):
                # koyu (çok az / çok fazla yüklü) hücrelerde beyaz yazı, açık hücrelerde siyah
                ax.text(e, li, f"{val:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val - 1.0) > 0.6 else "black")
            fig.colorbar(im, ax=ax, shrink=0.8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"kaydedildi: {args.out}")


if __name__ == "__main__":
    main()
