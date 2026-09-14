"""Eğitilmiş bir checkpoint'ten metin üret.

    .venv\\Scripts\\python -m src.sample --ckpt checkpoints/dense/best.pt --prompt "Once upon a time"

Örnekleme parametreleri:
  --temperature : logit'leri böler. 0 = greedy (hep en olası token; tekrara düşmeye meyilli),
                  0.7-0.9 = tutarlı ama çeşitli, >1 = giderek rastgele.
  --top_k       : her adımda yalnızca en olası k token arasından seç (0 = filtre kapalı).
"""
from __future__ import annotations

import argparse
import sys

import torch

from src.checkpoint import load_model
from src.tokenizer import load_tokenizer


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/dense/best.pt")
    p.add_argument("--prompt", default="")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, ckpt = load_model(args.ckpt, device)
    model.eval()
    tok = load_tokenizer(ckpt["train_cfg"]["data_dir"])
    print(f"checkpoint: {args.ckpt} (adım {ckpt['step']}, val_loss {ckpt['val_loss']:.4f})\n")

    # Eğitimde her hikâye BOS ile başladığı için üretime de BOS ile başlıyoruz: model
    # "yeni bir hikâyenin başındayım" bağlamını tanır.
    prompt_ids = [tok.bos_id] + tok.encode(args.prompt)
    x = torch.tensor([prompt_ids] * args.num_samples, device=device)
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad()
    with ctx:
        out = model.generate(x, args.max_new_tokens, args.temperature, args.top_k, eos_id=tok.eos_id)

    for i in range(args.num_samples):
        ids = out[i, 1:].tolist()  # BOS'u at
        if tok.eos_id in ids[len(prompt_ids) - 1:]:
            ids = ids[: ids.index(tok.eos_id, len(prompt_ids) - 1)]  # EOS sonrası çöpü kes
        print(f"--- örnek {i + 1} ---\n{tok.decode(ids)}\n")


if __name__ == "__main__":
    main()
