"""En iyi modül, router'ın GÖRDÜĞÜ girdiden öngörülebilir mi? (ön-kayıt V6 "sebep incelenir" — yalnızca TEŞHİS)

    .venv\\Scripts\\python scripts/router_probe_diag.py --run checkpoints/foton-v02-tarama/v7-x-s101 \\
        --out experiments/foton-v02/v7_router_probe_s101.json

Router doğrusaldır ve yalnızca havuzlanmış prefix vektörünü (pozisyon 0..63'ün ham residual ortalaması) görür.
Bu betik o vektörü sabit 1280 val dizisi için checkpoint'ten YENİDEN hesaplar ve üzerinde çapraz doğrulamalı bir
doğrusal sınıflandırıcı (çok sınıflı lojistik regresyon — router'ın kendi fonksiyon ailesi) eğitir:

  * hedef `best`  : dizide en düşük ölçüt-kaybını veren modül (son eval noktasının ham dökümünden);
  * hedef `domain`: alan kimliği — SAĞLAMA: girdi alanı taşıyor mu (taşımıyorsa probun kendisi bozuktur).

Okuma: `best` için CV doğruluğu çoğunluk tabanından ve permütasyon null'ından ayırt edilemiyorsa, bu girdiyle
HİÇBİR doğrusal router en iyi modülü öğrenemez — "öğrenilecek bir şey yok". Ayırt ediliyorsa sinyal vardır ve
router'ın öğrenememesi eğitim dinamiğindendir. Salt-okunur; seçim yapmaz; `src/`'ye dokunmaz.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.config import ModelConfig  # noqa: E402
from src.data_multi import MultiDomainDataset, multi_domain_paths  # noqa: E402
from src.model import GPT  # noqa: E402


def fit_predict(xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor, n_cls: int, l2: float) -> torch.Tensor:
    """Çok sınıflı lojistik regresyon (bias'lı, L2), LBFGS. Girdi standartlaştırılmış olmalı."""
    w = torch.zeros(xtr.shape[1] + 1, n_cls, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")
    xa = torch.cat([xtr, torch.ones(len(xtr), 1, dtype=torch.float64)], 1)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xa @ w, ytr) + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (torch.cat([xte, torch.ones(len(xte), 1, dtype=torch.float64)], 1) @ w).argmax(1)


def cv_accuracy(x: np.ndarray, y: np.ndarray, n_cls: int, folds: int = 5, l2: float = 1e-2, seed: int = 0) -> float:
    """Katmanlanmamış k-katlı CV doğruluğu. Standartlaştırma YALNIZCA eğitim katından öğrenilir (sızıntı yok)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    hit = 0
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
        pred = fit_predict(torch.from_numpy((x[tr] - mu) / sd), torch.from_numpy(y[tr]),
                           torch.from_numpy((x[te] - mu) / sd), n_cls, l2)
        hit += int((pred.numpy() == y[te]).sum())
    return hit / len(y)


def probe(x: np.ndarray, y: np.ndarray, n_cls: int, n_perm: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed + 1)
    acc = cv_accuracy(x, y, n_cls, seed=seed)
    null = [cv_accuracy(x, rng.permutation(y), n_cls, seed=seed) for _ in range(n_perm)]
    return {"cv_accuracy": acc, "majority_baseline": float(np.bincount(y, minlength=n_cls).max() / len(y)),
            "permutation_null_mean": float(np.mean(null)), "permutation_null_max": float(np.max(null)),
            "permutation_p": float((1 + sum(v >= acc for v in null)) / (1 + n_perm)), "n": int(len(y))}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data_dir", default="data/foton")
    ap.add_argument("--pca", type=int, default=64, help="probdan önce girdiyi bu kadar ana bileşene indir (1280 dizi, 512 boyut)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    state = torch.load(os.path.join(args.run, "last.pt"), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["model_cfg"])
    model = GPT(cfg)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    dump_path = sorted(glob.glob(os.path.join(args.run, "eval_raw", "step_*.npz")))[-1]
    with np.load(dump_path) as z:
        dump = {k: z[k] for k in ("seq_masked", "logits", "domain", "choice")}
        dump_step = int(z["step"])
    if dump_step != state.get("step"):
        sys.exit(f"son ham döküm (adım {dump_step}) ile last.pt (adım {state.get('step')}) aynı adımdan değil")

    ds = MultiDomainDataset(multi_domain_paths(os.path.join(ROOT, args.data_dir), "val"), cfg.max_seq_len, "cpu")
    pairs = ds.frozen_val_windows()
    pooled, logits = [], []
    with torch.no_grad():
        for i in range(0, len(pairs), 16):
            x, _ = ds.windows(pairs[i:i + 16])
            h = model.tok_emb(x)
            cos, sin = model.rope_cos[: x.shape[1]], model.rope_sin[: x.shape[1]]
            for blk in model.blocks[: cfg.foton_pre]:
                h, _ = blk(h, cos, sin)
            _, _, lg = model._foton_route(h)
            pooled.append(h[:, : cfg.foton_prefix_k].float().mean(1).numpy())
            logits.append(lg.numpy())
    pooled, logits = np.concatenate(pooled).astype(np.float64), np.concatenate(logits)
    # yeniden hesaplanan router logit'leri koşunun kendi dökümüyle aynı olmalı: aynı diziler, aynı ağırlıklar
    # (döküm GPU/bf16 gövdeden, burası CPU/fp32: küçük sayısal fark beklenir, seçimlerin çoğu aynı kalmalı)
    err = float(np.abs(logits - dump["logits"]).max())
    agree = float((logits.argmax(1) == dump["choice"]).mean())
    if len(pooled) != len(dump["choice"]) or agree < 0.9:
        sys.exit(f"yeniden hesaplanan router dökümle uyuşmuyor: seçim uyumu {agree:.3f}, maks logit farkı {err:.4f}")

    common = pooled.mean(0)
    centered = pooled - common
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    feats = centered @ vt[: args.pca].T
    best = dump["seq_masked"].argmin(1).astype(np.int64)
    dom = dump["domain"].astype(np.int64)
    res = {
        "what": "en iyi modül router'ın girdisinden öngörülebilir mi (teşhis; karar kuralı DEĞİL)",
        "run": args.run, "step": dump_step, "pca_components": args.pca,
        "recomputed_router_vs_dump": {"max_abs_logit_diff": err, "choice_agreement": agree},
        "pooled_input_geometry": {
            "common_mode_norm": float(np.linalg.norm(common)),
            "per_sequence_deviation_norm_mean": float(np.linalg.norm(centered, axis=1).mean()),
            "variance_explained_by_first_components": [float(v) for v in (s[:5] ** 2 / (s ** 2).sum())],
        },
        "probe_best_module": probe(feats, best, 3),
        "probe_domain_sanity": probe(feats, dom, int(dom.max()) + 1, n_perm=5),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("->", args.out)


if __name__ == "__main__":
    main()
