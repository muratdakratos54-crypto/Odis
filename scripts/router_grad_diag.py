"""Router'ın aldığı ÖĞRENME SİNYALİ doğru yönü gösteriyor mu? (ön-kayıt V6 "sebep incelenir" — yalnızca TEŞHİS)

    .venv\\Scripts\\python scripts/router_grad_diag.py --ckpt checkpoints/foton-v02-tarama/v7-x-s101/last.pt \\
        --out experiments/foton-v02/v7_router_grad_diag.json

Donmuş gate: `x_çıkış = x + g·(Modül(x) − x)`, `g = p/p.detach()` (değer 1, gradyan d log p). Router'ın görev
loss'undan aldığı TEK sinyal dizi başına bir skalerdir: `dL/dg_b`. Seçilen modül c için logit gradyanı
`dL/dg_b · (e_c − p)` olduğundan router, `dL/dg_b(m)`'si en NEGATİF olan modüle doğru sürüklenir. Router'ın doğru
şeyi öğrenebilmesi için o modülün, dizide gerçekten en düşük kayıplı modül olması gerekir.

Bu betik onu ÖLÇER: sabit val dizilerinde, her dizi ve her modül m için (bütün diziler m'e ZORLANARAK)
`loss_m(b)` ve `dL/dg_b(m)` hesaplanır (float32, tam aritmetik), sonra:

  * `grad_points_to_best`: argmin_m dL/dg_b(m) == argmin_m loss_m(b) olan dizilerin oranı (şans = 1/3);
  * `corr_grad_advantage`: dL/dg_b(m) ile modülün o dizideki avantajı (loss_m − üç modülün ortalaması) arasındaki
    Spearman korelasyonu (sinyal doğru yöndeyse POZİTİF: daha iyi modül = daha negatif gradyan);
  * `g_mean`, `g_std`: sinyalin modül başına ortalaması ve diziler arası saçılımı;
  * isteğe bağlı `--bf16`: aynı gradyan bf16 autocast altında yeniden hesaplanır ve fp32 ile karşılaştırılır (eğitim
    gate çarpımını bf16'da yapıyor; T×d = 262k terimli indirgemenin yuvarlama hatası sinyali boğuyor mu?).
    CPU çekirdekleri CUDA'nınkilerle aynı değildir: CPU sonucu yalnızca bir GÖSTERGEDİR.

Salt-okunur: checkpoint'i ve val verisini okur, hiçbir şeyi değiştirmez; `src/`'ye dokunmaz (trunk burada, gate
yaprak tensör yapılarak yeniden kurulur ve `GPT.forward(force_module=m)` ile aynı loss'u verdiği doğrulanır).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.config import ModelConfig  # noqa: E402
from src.data_multi import MultiDomainDataset, multi_domain_paths  # noqa: E402
from src.model import GPT  # noqa: E402


def forced_loss_and_gate_grad(model: GPT, idx: torch.Tensor, targets: torch.Tensor, m: int, autocast) -> tuple:
    """Bütün diziler modül m'e zorlanmışken: (dizi başına CE [B], dL/dg [B], batch loss). Gate yaprak tensördür."""
    B, T = idx.shape
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    gate = torch.ones(B, device=idx.device, requires_grad=True)
    with autocast:
        x = model.tok_emb(idx)
        pre = model.cfg.foton_pre
        for blk in model.blocks[:pre]:
            x, _ = blk(x, cos, sin)
        y = model._foton_module(m, x, cos, sin)
        x = x + gate.to(x.dtype)[:, None, None] * (y - x)
        for blk in model.blocks[pre:]:
            x, _ = blk(x, cos, sin)
        logits = model.lm_head(model.norm(x))
    ce_tok = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction="none").view(B, T)
    loss = ce_tok.mean()                              # eğitim loss'unun normalizasyonu: bütün B×T token'ın ortalaması
    loss.backward()
    return ce_tok.mean(1).detach(), gate.grad.detach().clone(), loss.detach()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize(loss: np.ndarray, g: np.ndarray) -> dict:
    """loss, g: [N, 3]."""
    adv = loss - loss.mean(1, keepdims=True)
    centered_g = g - g.mean(1, keepdims=True)
    return {
        "n_sequences": int(len(loss)),
        "chance": 1 / loss.shape[1],
        "grad_points_to_best": float((g.argmin(1) == loss.argmin(1)).mean()),
        "grad_points_to_worst": float((g.argmin(1) == loss.argmax(1)).mean()),
        "corr_grad_advantage_pooled": spearman(g.ravel(), adv.ravel()),
        "corr_grad_advantage_within_sequence": spearman(centered_g.ravel(), adv.ravel()),
        "g_mean_per_module": [float(v) for v in g.mean(0)],
        "g_std_across_sequences_per_module": [float(v) for v in g.std(0)],
        "g_within_sequence_spread": float((g.max(1) - g.min(1)).mean()),
        "loss_mean_per_module": [float(v) for v in loss.mean(0)],
        "loss_within_sequence_spread": float((loss.max(1) - loss.min(1)).mean()),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/foton")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--bf16", type=int, default=0, help="ilk N batch'te gradyanı bf16 autocast altında da hesapla")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["model_cfg"])
    if not cfg.foton:
        sys.exit("FOTON checkpoint'i değil")
    model = GPT(cfg)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected or any(not k.endswith(("attn_mask", "rope_cos", "rope_sin")) for k in missing):
        sys.exit(f"checkpoint modelle uyuşmuyor: eksik {missing}, fazla {unexpected}")
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = MultiDomainDataset(multi_domain_paths(os.path.join(ROOT, args.data_dir), "val"), cfg.max_seq_len, args.device)
    pairs = ds.frozen_val_windows()[: args.batches * args.batch_size]
    nullctx = contextlib.nullcontext()
    bf16ctx = lambda: torch.autocast(device_type=args.device, dtype=torch.bfloat16)  # noqa: E731
    n_mod = len(model.foton_modules)
    loss_all, g_all, g_bf16, dom_all = [], [], [], []
    for bi in range(args.batches):
        x, y = ds.windows(pairs[bi * args.batch_size:(bi + 1) * args.batch_size])
        dom_all.append(ds.last_domain_ids.cpu().numpy())
        if bi == 0:                                   # yeniden kurulan trunk, modelin kendi forward'ıyla aynı loss'u veriyor mu
            with torch.no_grad():                     # SAF CE ile karşılaştır: V7'nin regularizasyonlu koşullarında
                _, _, info = model(x, y, force_module=0)   # modelin `loss`'u entropi bonusu / aux terimini de içerir
            ref = info["ce"]
            _, _, mine = forced_loss_and_gate_grad(model, x, y, 0, nullctx)
            if abs(ref.item() - mine.item()) > 1e-5:
                sys.exit(f"yeniden kurulan trunk modelin forward'ıyla uyuşmuyor: {ref.item()} != {mine.item()}")
        rows_l, rows_g, rows_b = [], [], []
        for m in range(n_mod):
            l, g, _ = forced_loss_and_gate_grad(model, x, y, m, nullctx)
            rows_l.append(l.cpu().numpy()), rows_g.append(g.cpu().numpy())
            if bi < args.bf16:
                _, gb, _ = forced_loss_and_gate_grad(model, x, y, m, bf16ctx())
                rows_b.append(gb.float().cpu().numpy())
        loss_all.append(np.stack(rows_l, 1)), g_all.append(np.stack(rows_g, 1))
        if rows_b:
            g_bf16.append(np.stack(rows_b, 1))
        print(f"batch {bi + 1}/{args.batches} bitti", flush=True)

    loss, g = np.concatenate(loss_all), np.concatenate(g_all)
    res = {"what": "router öğrenme sinyali teşhisi (karar kuralı DEĞİL)", "ckpt": args.ckpt, "step": state.get("step"),
           "device": args.device, "dtype": "float32", **summarize(loss, g)}
    if g_bf16:
        gb = np.concatenate(g_bf16)
        ref = g[: len(gb)]
        res["bf16_check"] = {
            "n_sequences": int(len(gb)), "device": args.device,
            "note": "CPU çekirdekleri CUDA'nınkilerle aynı değildir; yalnızca gösterge" if args.device == "cpu" else "",
            "relative_error_median": float(np.median(np.abs(gb - ref) / (np.abs(ref) + 1e-12))),
            "sign_agreement": float((np.sign(gb) == np.sign(ref)).mean()),
            "corr_with_fp32": float(np.corrcoef(gb.ravel(), ref.ravel())[0, 1]),
            "argmin_agreement": float((gb.argmin(1) == ref.argmin(1)).mean()),
        }
    res["per_sequence"] = {"loss": loss.round(6).tolist(), "dL_dg": g.tolist(), "domain": np.concatenate(dom_all).tolist()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "per_sequence"}, indent=1, ensure_ascii=False))
    print("->", args.out)


if __name__ == "__main__":
    main()
