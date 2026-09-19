"""KURULDUĞU HÂLİYLE mimari: dört kolun (D, M, H, X) parametre dökümü — modeller GERÇEKTEN kurulup sayılarak.

    .venv\\Scripts\\python scripts/architecture_as_built.py            # tabloyu basar
    .venv\\Scripts\\python scripts/architecture_as_built.py --json     # docs/ARCHITECTURE_AS_BUILT.md'nin sayıları

`docs/ARCHITECTURE_AS_BUILT.md` bu betiğin çıktısından yazıldı ve `tests/test_architecture_doc.py` belgedeki sayıların
betiğin ürettiği sayılarla AYNI olduğunu sınar: belge koddan koparsa test kırılır. Neden var: v03 tasarım notlarına
"üç modül maske dışında her şeyi paylaşıyor" cümlesi girdi; yanlıştı (her modülün kendi 2 bloğu var). Mimari hakkında
ezberden yazılan cümle, deneyin yorumunu değiştiriyor — sayılar kurulan modelden okunmalı.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.config import PRESETS  # noqa: E402
from src.flops import train_flops_per_token  # noqa: E402
from src.model import FOTON_ARMS, GPT  # noqa: E402

VOCAB = 32768
ARMS = {"D": "foton-d", "M": "foton-m", "H": "foton-h", "X": "foton-x"}


def n(module) -> int:
    return sum(p.numel() for p in module.parameters())


def describe(arm: str) -> dict:
    cfg = replace(PRESETS[ARMS[arm]][0], vocab_size=VOCAB)
    model = GPT(cfg)
    out = {"preset": ARMS[arm], "d_model": cfg.d_model, "n_layer_traversed": cfg.n_layer, "n_head": cfg.n_head,
           "head_dim": cfg.d_model // cfg.n_head, "max_seq_len": cfg.max_seq_len, "vocab_size": cfg.vocab_size,
           "tie_embeddings": cfg.tie_embeddings, "embedding": model.tok_emb.weight.numel(),
           "lm_head_is_embedding": model.lm_head.weight is model.tok_emb.weight, "final_norm": n(model.norm),
           "params_total": model.num_params(), "params_active": model.num_params(active_only=True),
           "flops_train_per_token": train_flops_per_token(cfg)}
    blk = model.blocks[0]
    out["block_dense"] = {"attn_qkv": blk.attn.qkv.weight.numel(), "attn_proj": blk.attn.proj.weight.numel(),
                          "norms": n(blk.attn_norm) + n(blk.ffn_norm),
                          "ffn": n(blk.ffn) if not blk.is_moe else None, "total": n(blk) if not blk.is_moe else None,
                          "has_bias": any(p.dim() == 1 and "norm" not in name for name, p in blk.named_parameters())}
    out["blocks_built"] = len(model.blocks) + (sum(len(m) for m in model.foton_modules) if cfg.foton else 0)
    if cfg.foton:
        mods = model.foton_modules
        out["foton"] = {
            "arm": cfg.foton, "layout_pre_routed_post": [cfg.foton_pre, cfg.foton_routed, cfg.foton_post],
            "shared_blocks": len(model.blocks), "shared_block_params": n(model.blocks),
            "modules": len(mods), "layers_per_module": len(mods[0]),
            "module_patterns": [[b.attn.pattern for b in m] for m in mods],
            "shared_block_patterns": [b.attn.pattern for b in model.blocks],
            "params_per_module": [n(m) for m in mods],
            "modules_share_any_parameter": len({id(p) for m in mods for p in m.parameters()}) != sum(
                len(list(m.parameters())) for m in mods),
            "router_weight_shape": list(model.foton_router.weight.shape), "router_has_bias": model.foton_router.bias is not None,
            "router_params": n(model.foton_router), "prefix_k": cfg.foton_prefix_k, "reg": cfg.foton_reg,
            "connections_counted": model.foton_connection_counts(),
            "arms_table": {k: list(v) for k, v in FOTON_ARMS.items()},
        }
    if cfg.n_experts:
        moe_blocks = [b for b in model.blocks if b.is_moe]
        m0 = moe_blocks[0].ffn
        out["moe"] = {
            "moe_layers": [i for i, b in enumerate(model.blocks) if b.is_moe],
            "dense_layers": [i for i, b in enumerate(model.blocks) if not b.is_moe],
            "n_experts": cfg.n_experts, "top_k": cfg.top_k, "expert_hidden": cfg.expert_hidden,
            "n_shared_experts": cfg.n_shared_experts, "router_gate": cfg.router_gate,
            "norm_topk_prob": cfg.norm_topk_prob, "bias_update_rate": cfg.bias_update_rate,
            "aux_loss_coef": cfg.aux_loss_coef, "z_loss_coef": cfg.z_loss_coef, "moe_impl": cfg.moe_impl,
            "params_per_expert": n(m0.experts[0]), "params_shared_expert": n(m0.shared) if m0.shared is not None else 0,
            "params_router": n(m0.router), "params_moe_block": n(moe_blocks[0]), "params_dense_block": n(model.blocks[0]),
            "active_ffn_hidden_per_token": cfg.top_k * cfg.expert_hidden + cfg.n_shared_experts * cfg.expert_hidden,
        }
    return out


def router_free_flops(arm: str) -> int:
    """Kolun eğitim FLOP/token'ı, router terimi ÇIKARILARAK (src/flops.py router'ı `cfg.foton` doluyken hep sayar)."""
    from src.flops import forward_flops_per_token
    terms = forward_flops_per_token(replace(PRESETS[ARMS[arm]][0], vocab_size=VOCAB))
    return 3 * (sum(terms.values()) - terms["foton_router"])


def data_stream_facts(seed: int = 101, steps: int = 2188, data_dir: str = "data/foton") -> dict | None:
    """Eğitim dizi akışının (yalnızca seed'e bağlı `train_gen`) olguları — model kurmadan, akış yeniden oynatılarak.

    Döner None: veri dizini yoksa (temiz klon). Sayılan: birebir tekrar eden pencere, başka bir pencereyle kısmen örtüşen
    pencere, tekil token kapsaması, mikro-batch (16) ve adım (64) başına alan sayıları."""
    import numpy as np
    import torch

    from src.data_multi import MultiDomainDataset, multi_domain_paths
    root = os.path.join(ROOT, data_dir)
    if not os.path.exists(os.path.join(root, "meta.json")):
        return None
    ds = MultiDomainDataset(multi_domain_paths(root, "train"), 512, "cpu")
    gen = torch.Generator().manual_seed(seed)
    seqs = [p for _ in range(steps) for p in ds.sequence_stream(64, gen)]
    dom = np.array([ds.domain_id(d) for d, _ in seqs])
    starts = np.array([s for _, s in seqs], dtype=np.int64)
    exact = len(seqs) - len(set(seqs))
    overlapping, unique_tokens = 0, 0
    for di in range(len(ds.names)):
        s = np.sort(starts[dom == di])
        gap_prev = np.diff(s, prepend=-10 ** 12)
        gap_next = np.diff(s, append=10 ** 12)
        overlapping += int(((gap_prev < 512) | (gap_next < 512)).sum())
        unique_tokens += int(np.minimum(np.diff(s, append=s[-1] + 512), 512).sum())   # birleşim uzunluğu
    micro = np.stack([np.bincount(dom[i:i + 16], minlength=3) for i in range(0, len(dom), 16)])
    step = np.stack([np.bincount(dom[i:i + 64], minlength=3) for i in range(0, len(dom), 64)])
    corpus = sum(len(ds.data[k]) for k in ds.names)
    return {"seed": seed, "steps": steps, "windows": len(seqs), "domain_names_in_id_order": list(ds.names),
            "domain_weights": [float(w) for w in ds.weights], "exact_repeats": exact,
            "share_of_windows_overlapping_another": overlapping / len(seqs),
            "drawn_tokens_share_of_corpus": len(seqs) * 512 / corpus, "unique_coverage_share_of_corpus": unique_tokens / corpus,
            "per_microbatch_16": {"mean": [float(v) for v in micro.mean(0)], "sd": [float(v) for v in micro.std(0)],
                                  "min": [int(v) for v in micro.min(0)], "max": [int(v) for v in micro.max(0)],
                                  "microbatches": int(len(micro)), "microbatches_missing_a_domain": int((micro == 0).any(1).sum())},
            "per_step_64": {"mean": [float(v) for v in step.mean(0)], "min": [int(v) for v in step.min(0)],
                            "max": [int(v) for v in step.max(0)], "steps_missing_a_domain": int((step == 0).any(1).sum())}}


def val_window_facts(data_dir: str = "data/foton") -> dict | None:
    from src.data_multi import MultiDomainDataset, multi_domain_paths
    root = os.path.join(ROOT, data_dir)
    if not os.path.exists(os.path.join(root, "meta.json")):
        return None
    ds = MultiDomainDataset(multi_domain_paths(root, "val"), 512, "cpu")
    a, b = ds.frozen_val_windows(), ds.frozen_val_windows()
    counts = {name: sum(1 for d, _ in a if d == name) for name in ds.names}
    return {"windows": len(a), "identical_on_every_call": a == b, "per_domain": counts}


def collect() -> dict:
    res = {arm: describe(arm) for arm in ARMS}
    res["router_free_flops_train_per_token"] = {arm: router_free_flops(arm) for arm in ("H", "X")}
    h, x = res["H"], res["X"]
    res["H_vs_X"] = {"same_param_count": h["params_total"] == x["params_total"],
                     "only_difference": "yönlendirilen katmanların maskesi (parametresiz, checkpoint'e girmeyen buffer)",
                     "H_patterns": h["foton"]["module_patterns"], "X_patterns": x["foton"]["module_patterns"]}
    tc = PRESETS["foton-d"][1]
    res["training"] = {"tokens_per_step": tc.batch_size * tc.grad_accum * res["D"]["max_seq_len"], "max_steps": tc.max_steps,
                       "budget_tokens": tc.batch_size * tc.grad_accum * res["D"]["max_seq_len"] * tc.max_steps,
                       "lr": tc.lr, "min_lr": tc.min_lr, "warmup_steps": tc.warmup_steps, "weight_decay": tc.weight_decay,
                       "eval_interval": tc.eval_interval, "eval_sequences": tc.batch_size * tc.eval_iters,
                       "loss_prefix_skip": tc.loss_prefix_skip, "dtype": tc.dtype,
                       "identical_across_arms": all(replace(PRESETS[p][1], run_name="") == replace(tc, run_name="")
                                                    for p in ARMS.values())}
    return res


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--data", action="store_true", help="veri akışı olgularını da hesapla (data/foton gerekir; ~10 sn)")
    args = ap.parse_args()
    res = collect()
    if args.data:
        res["train_stream_screening"] = data_stream_facts(steps=2188)
        res["train_stream_full_budget"] = data_stream_facts(steps=4375)
        res["val_windows"] = val_window_facts()
    if args.json:
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return
    for arm in ARMS:
        r = res[arm]
        print(f"{arm}: toplam {r['params_total']:,} | aktif {r['params_active']:,} | kurulan blok {r['blocks_built']} | "
              f"FLOP/token {r['flops_train_per_token']:,}")
    print(json.dumps({k: res[k] for k in ("H_vs_X", "training")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
