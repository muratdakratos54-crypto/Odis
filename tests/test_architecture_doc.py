"""`docs/ARCHITECTURE_AS_BUILT.md` KURULAN modelle aynı şeyi söylüyor mu.

Belge `scripts/architecture_as_built.py`'nin çıktısından yazıldı; burada modeller yeniden kurulur ve belgedeki sayıların
hâlâ o modellerin sayıları olduğu sınanır. Belge koddan koparsa (ya da kod belgeden) test kırılır.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("architecture_as_built", os.path.join(ROOT, "scripts", "architecture_as_built.py"))
aab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aab)

with open(os.path.join(ROOT, "docs", "ARCHITECTURE_AS_BUILT.md"), encoding="utf-8") as _f:
    DOC = _f.read()


@pytest.fixture(scope="module")
def built():
    return {arm: aab.describe(arm) for arm in aab.ARMS}


def fmt(n: int) -> str:
    return f"{n:,}"


def test_every_parameter_count_in_the_doc_is_the_built_models_count(built):
    d, m, h, x = (built[k] for k in "DMHX")
    numbers = [d["params_total"], m["params_total"], m["params_active"], h["params_total"], h["params_active"],
               d["embedding"], d["block_dense"]["total"], d["block_dense"]["attn_qkv"], d["block_dense"]["attn_proj"],
               d["block_dense"]["ffn"], d["block_dense"]["norms"], m["moe"]["params_moe_block"], m["moe"]["params_per_expert"],
               m["moe"]["params_router"], h["foton"]["params_per_module"][0], h["foton"]["shared_block_params"],
               h["foton"]["router_params"], d["flops_train_per_token"], m["flops_train_per_token"],
               h["flops_train_per_token"], x["flops_train_per_token"]]
    missing = [fmt(v) for v in numbers if fmt(v) not in DOC]
    assert not missing, f"belgede olmayan (ya da değişmiş) sayılar: {missing}"
    assert d["params_total"] == d["params_active"] == d["embedding"] + 8 * d["block_dense"]["total"] + d["final_norm"]
    assert h["params_total"] == x["params_total"] and h["params_active"] == x["params_active"] == d["params_total"] + h["foton"]["router_params"]


def test_the_structural_claims_of_the_doc_hold_in_the_built_models(built):
    m, h, x = built["M"], built["H"], built["X"]
    # "3 + 2 + 3": 6 paylaşılan blok + 3 modül x 2 blok = 12 kurulan blok; paylaşılan bloklarda maske yok
    for arm in (h, x):
        f = arm["foton"]
        assert f["layout_pre_routed_post"] == [3, 2, 3] and f["shared_blocks"] == 6 and arm["blocks_built"] == 12
        assert f["modules"] == 3 and f["layers_per_module"] == 2 and f["shared_block_patterns"] == [None] * 6
        assert f["modules_share_any_parameter"] is False and len(set(f["params_per_module"])) == 1   # modüller AYRI ağırlık
        assert f["router_weight_shape"] == [3, 512] and f["router_has_bias"] is False and f["prefix_k"] == 64 and f["reg"] == "none"
    assert h["foton"]["module_patterns"] == [["A", "A"]] * 3                 # H = (A, A, A)
    assert x["foton"]["module_patterns"] == [["A", "A"], ["B", "B"], ["C", "C"]]
    assert x["foton"]["connections_counted"] == {"M0:A": 128.173828125, "M1:B": 128.5, "M2:C": 128.375}
    for v in ("128.173828125", "128.375", "H = (A, A, A)"):
        assert v in DOC
    # M: blok 0 dense, 1-7 MoE; 16 expert, top-3, sigmoid, 1 paylaşılan, aux 0, z-loss 0.001, aktif FFN 1408
    mo = m["moe"]
    assert mo["dense_layers"] == [0] and mo["moe_layers"] == [1, 2, 3, 4, 5, 6, 7]
    assert (mo["n_experts"], mo["top_k"], mo["expert_hidden"], mo["n_shared_experts"]) == (16, 3, 352, 1)
    assert (mo["router_gate"], mo["aux_loss_coef"], mo["z_loss_coef"], mo["bias_update_rate"]) == ("sigmoid", 0.0, 0.001, 0.001)
    assert mo["active_ffn_hidden_per_token"] == 1408 and mo["norm_topk_prob"] is True and mo["moe_impl"] == "grouped"
    assert all(not built[a]["block_dense"]["has_bias"] and built[a]["lm_head_is_embedding"] for a in "DMHX")


def test_c_is_local_on_heads_0_3_and_strided_on_heads_4_7_as_the_doc_says():
    from src import attn_patterns as ap
    m = ap.PATTERNS["C"].masks(512, 8)
    assert [int(m[h, 511].sum()) for h in range(8)] == [256] * 4 + [128] * 4
    assert sorted(511 - j for j in range(512) if m[7, 511, j])[:3] == [0, 4, 8] and max(511 - j for j in range(512) if m[7, 511, j]) == 508
    assert max(511 - j for j in range(512) if m[0, 511, j]) == 255
    b = ap.PATTERNS["B"].masks(512, 8)
    assert sorted(511 - j for j in range(512) if b[0, 511, j])[:3] == [0, 2, 4] and int(b[0, 511].sum()) == 256
    assert int(ap.PATTERNS["A"].masks(512, 8)[3, 511].sum()) == 150


def test_training_setup_is_identical_across_the_four_arms():
    from dataclasses import replace

    from src.config import PRESETS
    tc = PRESETS["foton-d"][1]
    assert all(replace(PRESETS[p][1], run_name="") == replace(tc, run_name="") for p in aab.ARMS.values())
    tr = {"identical_across_arms": True, "tokens_per_step": tc.batch_size * tc.grad_accum * 512,
          "budget_tokens": tc.batch_size * tc.grad_accum * 512 * tc.max_steps, "eval_sequences": tc.batch_size * tc.eval_iters,
          "loss_prefix_skip": tc.loss_prefix_skip, "lr": tc.lr, "min_lr": tc.min_lr}
    assert tr["identical_across_arms"] is True and tr["tokens_per_step"] == 32768 and tr["budget_tokens"] == 143360000
    assert tr["eval_sequences"] == 1280 and tr["loss_prefix_skip"] == 64 and (tr["lr"], tr["min_lr"]) == (1e-3, 1e-4)
    for v in ("32,768", "143,360,000", "1,280", "64..511"):
        assert v in DOC


def test_router_free_flops_and_the_h_x_gap_are_what_the_doc_says(built):
    free = {arm: aab.router_free_flops(arm) for arm in ("H", "X")}
    assert all(fmt(v) in DOC for v in free.values())
    assert built["X"]["flops_train_per_token"] - free["X"] == 18 == built["H"]["flops_train_per_token"] - free["H"]
    assert built["X"]["flops_train_per_token"] - built["H"]["flops_train_per_token"] == 4008 and "4,008" in DOC
    assert built["D"]["params_total"] == built["X"]["params_active"] - built["X"]["foton"]["router_params"]


def test_data_stream_facts_in_the_doc_are_reproduced_by_replaying_the_stream():
    """Belgedeki veri olguları (tekrar etmeyen pencereler, mikro-batch başına alan sayıları, val pencerelerinin alan dağılımı)
    dizi akışı yeniden oynatılarak üretildi; veri dizini yoksa (temiz klon) atlanır."""
    scr = aab.data_stream_facts(steps=2188)
    if scr is None:
        pytest.skip("data/foton yok")
    full, val = aab.data_stream_facts(steps=4375), aab.val_window_facts()
    assert scr["domain_names_in_id_order"] == ["code", "math", "nl"] and all(abs(w - 1 / 3) < 1e-6 for w in scr["domain_weights"])
    assert (scr["windows"], scr["exact_repeats"], full["windows"], full["exact_repeats"]) == (140032, 25, 280000, 112)
    assert scr["per_microbatch_16"]["microbatches"] == 8752 and scr["per_microbatch_16"]["microbatches_missing_a_domain"] == 40
    assert scr["per_step_64"]["steps_missing_a_domain"] == 0 and (min(scr["per_step_64"]["min"]), max(scr["per_step_64"]["max"])) == (9, 36)
    assert val == {"windows": 1280, "identical_on_every_call": True, "per_domain": {"code": 431, "math": 434, "nl": 415}}
    shown = {"%37.9": scr["share_of_windows_overlapping_another"], "%61.6": full["share_of_windows_overlapping_another"],
             "%23.9": scr["drawn_tokens_share_of_corpus"], "%47.8": full["drawn_tokens_share_of_corpus"],
             "%21.3": scr["unique_coverage_share_of_corpus"], "%38.0": full["unique_coverage_share_of_corpus"]}
    for text, value in shown.items():
        assert text in DOC and text == f"%{100 * value:.1f}", (text, value)
    for v in ("140,032", "8,752", "code 431 / math 434 / nl 415", "code = 0 / math = 1 / nl = 2"):
        assert v in DOC


def test_forced_module_path_skips_the_gate_and_adamw_skips_gradless_parameters():
    """Belgenin 4. bölümündeki iki davranış: `force_module` yolunda router gradyan ALMAZ; gradyanı None olan parametreyi
    AdamW tümüyle atlar (weight decay dahil) — zorlanmış yönlendirmede router init'te bit düzeyinde sabit kalır."""
    import torch
    from dataclasses import replace

    from src.config import PRESETS
    from src.model import GPT
    torch.manual_seed(0)
    model = GPT(replace(PRESETS["tiny-foton"][0], vocab_size=61))
    opt = model.configure_optimizer(1e-2, 0.1, (0.9, 0.95), "cpu")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    idx = torch.randint(0, 61, (4, 32))
    _, loss, info = model(idx, idx, force_module=1)
    loss.backward()
    assert model.foton_router.weight.grad is None and "foton_choice" in info            # router koştu ama gradyan almadı
    assert all(p.grad is None for p in model.foton_modules[0].parameters())
    assert all(p.grad is not None for p in model.foton_modules[1].parameters())
    opt.step()
    after = model.state_dict()
    assert torch.equal(before["foton_router.weight"], after["foton_router.weight"])     # weight decay de UYGULANMADI
    assert all(torch.equal(before[k], after[k]) for k in before if k.startswith(("foton_modules.0.", "foton_modules.2.")))
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("foton_modules.1."))
