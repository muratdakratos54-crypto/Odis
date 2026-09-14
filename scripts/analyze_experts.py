"""Eğitilmiş bir MoE checkpoint'inde expert yükünü ve uzmanlaşmayı analiz eder.

    .venv\\Scripts\\python scripts/analyze_experts.py --ckpt checkpoints/moe/best.pt --layer 4

Yük dengesi "her expert'e eşit iş düşüyor mu?" sorusunu yanıtlar; bu betik bir adım ileri gidip
"expert'ler aynı işi mi yapıyor, yoksa gerçekten farklı şeylerde mi uzmanlaşıyor?" diye bakar.

Yöntem: her MoE katmanına bir forward hook takıp (model koduna dokunmadan) val verisindeki her
tokenın hangi k expert'e gittiğini (MoE çıktısındaki stats["topk_idx"]) kaydediyoruz. Sonra her expert için
"lift" = P(expert | token) / P(expert) oranı en yüksek tokenları listeliyoruz: lift 3 ise o token
bu expert'e ortalamanın 3 katı sıklıkla gidiyor demektir.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.checkpoint import load_model  # noqa: E402
from src.data import TokenDataset, dataset_paths  # noqa: E402
from src.moe import load_balance_metrics, load_sparkline  # noqa: E402
from src.tokenizer import load_tokenizer  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/moe/best.pt")
    p.add_argument("--layer", type=int, default=None, help="uzmanlaşma tablosu için katman (varsayılan: ortadaki MoE katmanı)")
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--top", type=int, default=8, help="expert başına gösterilecek token sayısı")
    p.add_argument("--min_count", type=int, default=200, help="lift hesabına girecek bir tokenın en az görülme sayısı")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, ckpt = load_model(args.ckpt, args.device)
    model.eval()
    cfg = model.cfg
    if cfg.n_experts == 0:
        sys.exit("bu checkpoint dense bir model; analiz edilecek expert yok")
    tok = load_tokenizer(ckpt["train_cfg"]["data_dir"])
    moe_layers = [i for i, b in enumerate(model.blocks) if b.is_moe]
    E, k, V = cfg.n_experts, cfg.top_k, cfg.vocab_size

    # (katman, token, expert) sayaçları + her forward'daki token id'leri
    counts = torch.zeros(len(moe_layers), V, E, dtype=torch.long)
    current_ids: dict[str, torch.Tensor] = {}

    def make_hook(slot: int):
        def hook(_module, _inp, output):
            topk_idx = output[1]["topk_idx"].cpu()                 # (N, k) - MoE.forward -> (y, stats)
            ids = current_ids["x"].reshape(-1, 1).expand(-1, k)    # (N, k)
            flat = ids.reshape(-1) * E + topk_idx.reshape(-1)
            counts[slot].view(-1).index_add_(0, flat, torch.ones_like(flat))
        return hook

    for slot, li in enumerate(moe_layers):
        model.blocks[li].ffn.register_forward_hook(make_hook(slot))

    _, val_path = dataset_paths(ckpt["train_cfg"]["data_dir"])
    ds = TokenDataset(val_path, cfg.max_seq_len, args.device)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _ in range(args.batches):
            x, y = ds.get_batch(args.batch_size, g)
            current_ids["x"] = x.cpu()
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else torch.no_grad()
            with ctx:
                model(x, y)

    # ---- 1) Katman başına yük dengesi ----
    load = counts.sum(dim=1)  # (katman, E)
    bal = load_balance_metrics(load)
    n_tok = args.batches * args.batch_size * cfg.max_seq_len
    print(f"{args.ckpt}: {n_tok:,} val tokenı, {len(moe_layers)} MoE katmanı, E={E}, top-{k}")
    print(f"ortalama CV {bal['load_cv']:.3f} | maks {bal['load_max_ratio']:.2f}x | "
          f"min {bal['load_min_ratio']:.2f}x | entropi {bal['load_entropy']:.3f} | ölü expert {bal['dead_experts']}\n")
    for slot, li in enumerate(moe_layers):
        row = bal["load"][slot]
        print(f"katman {li}: |{load_sparkline(row)}|  " + " ".join(f"{f * E:4.2f}" for f in row))

    # ---- 2) Seçilen katmanda expert uzmanlaşması ----
    li = args.layer if args.layer is not None else moe_layers[len(moe_layers) // 2]
    slot = moe_layers.index(li)
    c = counts[slot].double()                     # (V, E)
    tok_total = c.sum(-1, keepdim=True)           # her tokenın atama sayısı (= görülme * k)
    p_e_given_t = c / tok_total.clamp_min(1)
    p_e = c.sum(0) / c.sum()
    lift = p_e_given_t / p_e
    lift[(tok_total.squeeze(-1) / k) < args.min_count] = 0  # nadir tokenlar gürültülü: dışarıda bırak

    def show(i: int) -> str:
        # SentencePiece'te ham parçayı göster: '▁The' = kelime başındaki "The", 'The' = kelime içi,
        # '<0x0A>' = satır sonu byte'ı, '<s>'/'</s>' = BOS/EOS. (decode() baştaki boşluğu yutar.)
        s = tok.sp.id_to_piece(i) if hasattr(tok, "sp") else tok.decode([i])
        return repr(s)[1:-1]

    print(f"\nkatman {li} expert uzmanlaşması (lift = P(expert|token) / P(expert); en az {args.min_count} kez görülen tokenlar):")
    for e in range(E):
        top = torch.topk(lift[:, e], args.top)
        items = ", ".join(f"{show(int(i))}({v:.1f})" for v, i in zip(top.values.tolist(), top.indices.tolist()) if v > 0)
        print(f"  expert {e:2d} [{p_e[e] * E:4.2f}x yük]: {items}")


if __name__ == "__main__":
    main()
