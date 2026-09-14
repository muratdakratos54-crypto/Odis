"""Bir preset için farklı mikro-batch boyutlarında tepe VRAM ve throughput ölçer (rastgele tokenlarla).

    .venv\\Scripts\\python scripts/benchmark.py --preset dense --batch_sizes 16,32,48,64

Neden?  12 GB VRAM'de belleğin nereye gittiğini bilmek, batch boyutunu tahminle değil ölçümle
seçmemizi sağlar. Kabaca bellek dağılımı (float32 ağırlık + bf16 aktivasyon):
  * ağırlıklar 4 B/param + gradyanlar 4 B/param + AdamW momentleri 8 B/param = 16 B/param
    (30M param -> ~0.5 GB; burada sorun değil)
  * aktivasyonlar: batch x seq x d_model x n_layer ile ölçeklenir -> asıl belirleyici
  * logit'ler: batch x seq x vocab (float32 CE için ayrıca kopya) -> büyük sözlükte çok pahalı
Güvenlik payı: tepe belleğin ~10 GB'ı geçmemesi iyi olur (Windows masaüstü de VRAM kullanır,
parçalanma/fragmentation ve eval sırasında ek tahsisler olur).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import build_configs  # noqa: E402
from src.model import GPT  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--batch_sizes", default="16,32,48,64")
    p.add_argument("--steps", type=int, default=8)
    args, rest = p.parse_known_args()
    mcfg, tcfg = build_configs(rest)

    torch.set_float32_matmul_precision("high")
    # train.py ile aynı koruma: VRAM taşınca sessiz yavaşlama yerine OOM al
    if tcfg.vram_limit_frac > 0:
        torch.cuda.set_per_process_memory_fraction(tcfg.vram_limit_frac)
    model = GPT(mcfg).cuda()
    opt = model.configure_optimizer(1e-4, 0.1, (0.9, 0.95), "cuda")
    print(f"{model.num_params() / 1e6:.1f}M param, seq {mcfg.max_seq_len}, vocab {mcfg.vocab_size}")
    for bs in map(int, args.batch_sizes.split(",")):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randint(0, mcfg.vocab_size, (bs, mcfg.max_seq_len), device="cuda")
            for i in range(args.steps):
                if i == 2:  # ilk adımlar CUDA/allocator ısınması: zamanlamaya katma
                    torch.cuda.synchronize()
                    t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, x)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tok_s = bs * mcfg.max_seq_len * (args.steps - 2) / (time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"batch {bs:3d}: tepe {peak:5.2f} GB | {tok_s / 1e3:6.1f}k tok/s")
        except torch.OutOfMemoryError:
            print(f"batch {bs:3d}: OOM")
            opt.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
