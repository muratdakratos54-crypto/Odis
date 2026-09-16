"""GPU'da "grouped" ve "loop" expert yollarının eşdeğerliği: routing, ileri, geri, 1 optimizer adımı.

    .venv\\Scripts\\python scripts/verify_dispatch.py --out experiments/v2-grouped-dispatch/verify_dispatch.json

    .venv\\Scripts\\python scripts/verify_dispatch.py --deterministic --out experiments/v2-grouped-dispatch/verify_dispatch_deterministic.json

Bit bit eşitlik ARANMAZ: GPU'da doğru bir uygulama bile toplama sırası ya da dtype çevirme noktası farklı olduğu için
son bitlerde ayrışabilir; üstelik varsayılan çekirdekler (bellek-verimli attention geri yönü, CUDA index_add_) aynı
kodu iki kez çalıştırınca bile farklı bitler üretebilir ve bu gürültünün büyüklüğü süreçten sürece değişir (eşit skorlu
tokenların routing'i çevrilince büyür). Her ölçüm şu sayılarla verilir:
  * grouped vs loop         : asıl fark (2 grouped koşusunun en kötüsü)
  * loop vs loop (tekrar)   : taban = aynı kodun (--reruns - 1) tekrarının loop'a en kötü farkı (tek örnek yanıltıcıydı)
  * loop bf16 vs loop fp32  : ÖLÇEK - aynı matematiğin yalnızca kayan nokta hassasiyeti değişince oluşan fark
  * negatif kontrol         : grouped yolunun İÇİNDE expert 0 ve 1'in istiflenmiş ağırlıkları yer değiştirilmiş
                              (parametre adları aynı kalır); ölçümün gerçek bir dispatch hatasını ayırt ettiğini gösterir
Ek tanı ("fork32"): gate ve up kolları için satırları float32'den AYRI AYRI çeviren varyant ("loop"un autocast yapısı).

Kabul ölçütleri. Geçmiş, dürüstlük için: ilk sürümdeki mutlak eşikler loop'un kendi tekrarında bile sağlanamadı
(AdamW'nin ilk adımı sıfıra yakın gradyanlarda işaret gibi davranır), bu yüzden göreli ölçütlere geçildi. Bir inceleme
turunda fp32 sınırının bf16 ölçeğinden türetildiği için gevşek kaldığı bulundu ve fp32'ye kendi eşikleri verildi.
  1. teacher-forced: her MoE katmanına loop'un girdisi verilince katman çıktısı rel <= max(3 x taban, 1e-6); taban ve
     grouped değeri de tekrar zarfıyla ölçülür (3 loop tekrarı, 2 grouped çağrısı). (İlk hali sabit 1e-6'ydı, sonra tek
     örnekli taban eklendi; o taban bf16'da bazen 0 çıkıp sınırı gereksiz sıkılaştırdığı için zarfa çevrildi.)
     (Aynı girdiyle routing'in aynı olması yapısal olarak garanti - router dispatch'ten önce - ayrıca raporlanır.)
  2. bf16: fark(grouped, loop) <= max(3 x taban, 0.25 x ölçek)
     fp32: fark(grouped, loop) <= max(10 x taban, FP32_TOL)  (FP32_TOL, fp32 tekrar gürültüsünün ~10 katı)
  3. negatif kontrol >= 10 x fark(grouped, loop) ve >= ölçek   (logits, gradyan, en kötü tensör gradyanı)
--deterministic (tanı, geçme şartı değil): use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG + SDPA math arka ucu ile
tekrar gürültüsü kalkar; grouped ile loop'un bu durumda bit bit aynı olup olmadığı raporlanır.
Modeller: eğitilmiş v1 checkpoint'leri (moe: dengeli; moe-noaux: çökmüş, batch'te boş expert'ler) + yeni başlatılmış
moe. Veri: gerçek val penceresi, eğitimdeki şekil (32 x 512).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.checkpoint import load_model  # noqa: E402
from src.config import PRESETS  # noqa: E402
from src.data import TokenDataset  # noqa: E402
from src.model import GPT  # noqa: E402
from src.moe import MoE, _CastGatherRows  # noqa: E402

METRICS = ("logits", "grads", "grads_worst_tensor", "update", "loss_after_step")
FP32_TOL = {"logits": 1e-6, "grads": 1e-5, "grads_worst_tensor": 1e-5, "update": 1e-4, "loss_after_step": 1e-5}


def set_impl(model: GPT, impl: str) -> None:
    for m in model.modules():
        if isinstance(m, MoE):
            m.impl = impl


@contextmanager
def precision_mode(precision: str):
    """bf16: eğitimdeki ayar (autocast + TF32, train.py set_float32_matmul_precision('high')). fp32: TF32 kapalı."""
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = precision == "bf16"
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext():
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def _grouped_variant(fork32: bool = False, swap01: bool = False):
    """grouped yolun tanı/kontrol kopyaları. fork32: satırlar float32'de toplanıp gate ve up için ayrı ayrı çevrilir.
    swap01: istiflenmiş ağırlıklarda expert 0 ile 1 yer değiştirir (negatif kontrol: dispatch hatası)."""
    def grouped(self, h, token_of, counts):
        dtype = self._expert_dtype(h)
        offs = counts.cumsum(0, dtype=torch.int32)
        E = len(self.experts)
        perm = torch.tensor([1, 0, *range(2, E)] if swap01 else list(range(E)), device=h.device)
        if fork32:
            x32 = h.index_select(0, token_of)
            x_gate, x_up = x32.to(dtype), x32.to(dtype)
        else:
            x_gate, x_up = _CastGatherRows.apply(h, token_of, dtype)
        with torch.autocast(device_type=h.device.type, enabled=False):
            w_gate, w_up, w_down = (torch.stack([getattr(e, n).weight for e in self.experts])[perm].to(dtype).transpose(1, 2)
                                    for n in ("w_gate", "w_up", "w_down"))
            gate = torch._grouped_mm(x_gate, w_gate, offs=offs)
            up = torch._grouped_mm(x_up, w_up, offs=offs)
            return torch._grouped_mm(F.silu(gate) * up, w_down, offs=offs)
    return grouped


VARIANTS = {"fork32": _grouped_variant(fork32=True), "badswap": _grouped_variant(swap01=True)}


def run_once(model: GPT, x, y, precision: str, impl: str, keep_inputs: bool = False, lr: float = 1e-3) -> dict:
    """ileri + geri + 1 AdamW adımı + adım sonrası loss, modelin KOPYASI üzerinde (her çağrı aynı ağırlıklardan).
    impl: loop | grouped | fork32 | badswap. Karşılaştırılacak tensörler CPU'ya alınır (12 GB VRAM'de dört koşunun
    gradyanları + parametreleri sığmaz)."""
    model = copy.deepcopy(model)
    set_impl(model, "loop" if impl == "loop" else "grouped")
    original = MoE._grouped_experts
    if impl in VARIANTS:
        MoE._grouped_experts = VARIANTS[impl]
    try:
        captured = []
        hooks = [m.register_forward_hook(lambda mod, inp, out: captured.append(
            (inp[0].detach() if keep_inputs else None, out[1]["topk_idx"].cpu())))
            for m in model.modules() if isinstance(m, MoE)]
        with precision_mode(precision):
            logits, loss, _ = model(x, y)
        for h in hooks:
            h.remove()
        loss.backward()
        grads = {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None}
        before = {n: p.detach().cpu() for n, p in model.named_parameters()}
        opt = model.configure_optimizer(lr, 0.1, (0.9, 0.95), "cuda")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad(), precision_mode(precision):
            _, loss_after, _ = model(x, y)
    finally:
        MoE._grouped_experts = original
    params = {n: p.detach().cpu() for n, p in model.named_parameters()}
    out = {"logits": logits.detach().float().cpu(), "loss": loss.item(), "loss_after_step": loss_after.item(),
           "grads": grads, "update": {n: params[n] - before[n] for n in params}, "params": params,
           "moe_inputs": [c[0] for c in captured], "topk": [c[1] for c in captured]}
    del model, opt, logits, loss, loss_after
    torch.cuda.empty_cache()
    return out


def rel_l2(a: dict | torch.Tensor, b: dict | torch.Tensor, ref: dict | torch.Tensor) -> float:
    """||a - b|| / ||ref||, sözlükse tüm tensörler tek bir vektör gibi."""
    if isinstance(a, dict):
        num = sum((a[n].double() - b[n].double()).pow(2).sum().item() for n in a)
        den = sum(ref[n].double().pow(2).sum().item() for n in ref)
        return math.sqrt(num / max(den, 1e-300))
    return ((a.double() - b.double()).norm() / ref.double().norm().clamp_min(1e-300)).item()


def tensor_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Tek tensör için ||a - b|| / max(||a||, ||b||): hiç token almamış bir expert'in sıfır gradyanı paydayı sıfırlamasın."""
    a, b = a.double(), b.double()
    den = max(a.norm().item(), b.norm().item())
    return (a - b).norm().item() / den if den > 0 else 0.0


def envelope(cs: list[dict]) -> dict:
    """Birden çok karşılaştırmanın en kötüsü: sayısal metriklerde maks, uyumda min, bit bit/aynılıkta VE."""
    out = {}
    for k in cs[0]:
        vals = [c[k] for c in cs]
        if isinstance(vals[0], bool):
            out[k] = all(vals)
        elif k == "routing_min_token_agreement":
            out[k] = min(vals)
        else:
            out[k] = max(vals)
    return out


def compare(a: dict, b: dict) -> dict:
    """a referans (loop). update: parametre farkı / güncellemenin kendi normu."""
    return {
        "routing_identical_all_layers": all(torch.equal(p, q) for p, q in zip(a["topk"], b["topk"])),
        "routing_min_token_agreement": min((p == q).all(dim=-1).float().mean().item() for p, q in zip(a["topk"], b["topk"])),
        "loss_abs_diff": abs(a["loss"] - b["loss"]),
        "logits": rel_l2(a["logits"], b["logits"], a["logits"]),
        "logits_bitexact": bool(torch.equal(a["logits"], b["logits"])),
        "grads": rel_l2(a["grads"], b["grads"], a["grads"]),
        "grads_worst_tensor": max(tensor_rel(a["grads"][n], b["grads"][n]) for n in a["grads"]),
        "grads_bitexact": all(torch.equal(a["grads"][n], b["grads"][n]) for n in a["grads"]),
        "grad_keys_equal": a["grads"].keys() == b["grads"].keys(),
        "update": rel_l2(a["params"], b["params"], a["update"]),
        "params_max_abs_diff": max((a["params"][n] - b["params"][n]).abs().max().item() for n in a["params"]),
        "params_frac_elements_differ": (sum(int((a["params"][n] != b["params"][n]).sum()) for n in a["params"])
                                        / sum(p.numel() for p in a["params"].values())),
        "params_bitexact": all(torch.equal(a["params"][n], b["params"][n]) for n in a["params"]),
        "loss_after_step": abs(a["loss_after_step"] - b["loss_after_step"]) / abs(a["loss_after_step"] - a["loss"]),
    }


def teacher_forced(model: GPT, ref: dict, precision: str) -> dict:
    """Her MoE katmanına loop koşusunun o katmana verdiği girdiyi aynen ver: routing birebir aynı olmalı.
    (Serbest koşuda derin katmanların girdisi önceki katmanların yuvarlamasıyla az farklı olabilir.)"""
    moes = [m for m in model.modules() if isinstance(m, MoE)]
    same, rels, floors, bit = [], [], [], True
    with torch.no_grad(), precision_mode(precision):
        for m, h_in, topk_ref in zip(moes, ref["moe_inputs"], ref["topk"]):
            m.impl = "loop"
            y_loop, s_loop = m(h_in)
            # Taban zarfı: CUDA index_add_ aynı girdide bile son bitlerde değişebilir ve tek ölçüm sıfır çıkabilir
            floors.append(max(rel_l2(y_loop.float(), m(h_in)[0].float(), y_loop.float()) for _ in range(3)))
            m.impl = "grouped"
            y_grp, s_grp = m(h_in)
            same.append(torch.equal(s_grp["topk_idx"].cpu(), topk_ref) and torch.equal(s_loop["topk_idx"].cpu(), topk_ref))
            rels.append(max(rel_l2(y_loop.float(), y_grp.float(), y_loop.float()),
                            rel_l2(y_loop.float(), m(h_in)[0].float(), y_loop.float())))
            bit &= bool(torch.equal(y_loop, y_grp))
    return {"routing_identical_every_layer": all(same), "layer_output_rel_l2_max": max(rels),
            "layer_output_floor_rel_l2_max": max(floors), "layer_output_bitexact": bit}


def main_deterministic(models: dict, x, y) -> dict:
    """Tanı: deterministik çekirdeklerle (use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG + SDPA math) GPU
    tekrar gürültüsü kalkar; bu durumda grouped ile loop bit bit aynı mı? Geçme şartı DEĞİL, ek kanıt."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.use_deterministic_algorithms(True)
    report = {}
    with sdpa_kernel(SDPBackend.MATH):
        for name, model in models.items():
            model.train()
            for precision in ("bf16", "fp32"):
                l1, l2 = run_once(model, x, y, precision, "loop"), run_once(model, x, y, precision, "loop")
                g, f = run_once(model, x, y, precision, "grouped"), run_once(model, x, y, precision, "fork32")
                r = {"loop_vs_loop": compare(l1, l2), "grouped_vs_loop": compare(l1, g), "fork32_vs_loop": compare(l1, f)}
                report[f"{precision} | {name}"] = r
                print(f"=== deterministik | {precision} | {name}")
                for label, c in r.items():
                    print(f"  {label:16s} bit bit: logits {c['logits_bitexact']!s:5s} grads {c['grads_bitexact']!s:5s} "
                          f"params {c['params_bitexact']!s:5s} | grads rel {c['grads']:.1e} | routing aynı "
                          f"{c['routing_identical_all_layers']}")
    return report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiments/v2-grouped-dispatch/verify_dispatch.json")
    p.add_argument("--data_dir", default="data/sp8k")
    p.add_argument("--reruns", type=int, default=4, help="taban için loop tekrar sayısı (taban = en kötü fark)")
    p.add_argument("--deterministic", action="store_true", help="yalnızca deterministik çekirdek tanısı")
    args = p.parse_args()
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA bağlamı oluşmadan önce

    ds = TokenDataset(os.path.join(args.data_dir, "val.bin"), 512, "cuda")
    x, y = ds.get_batch(32, torch.Generator().manual_seed(12345))
    models = {f"{n} (v1 best.pt)": load_model(os.path.join("checkpoints", n, "best.pt"), "cuda")[0] for n in ("moe", "moe-noaux")}
    torch.manual_seed(1337)
    models["moe (yeni başlatma)"] = GPT(PRESETS["moe"][0]).cuda()

    if args.deterministic:
        report = main_deterministic(models, x, y)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"mode": "deterministic_diagnostic", "results": report}, f, indent=2, ensure_ascii=False)
        print(f"-> {args.out}")
        return

    report, ok = {}, True
    for name, model in models.items():
        model.train()
        runs = {}
        for precision in ("bf16", "fp32"):
            loop1 = run_once(model, x, y, precision, "loop", keep_inputs=True)
            runs[precision] = loop1
            floor = [compare(loop1, run_once(model, x, y, precision, "loop")) for _ in range(args.reruns - 1)]
            grouped = [compare(loop1, run_once(model, x, y, precision, "grouped")) for _ in range(2)]
            # Simetrik sıfır adayı: loop'un kendisi (2 ek tekrar) grouped'la aynı kontrollerden geçirilir. Kontroller
            # GPU tekrar gürültüsüyle sınırlıysa sıfır aday da zaman zaman başarısız olur; grouped ondan sık olmamalı.
            null = [compare(loop1, run_once(model, x, y, precision, "loop")) for _ in range(2)]
            counts = torch.stack([torch.bincount(t.reshape(-1), minlength=16) for t in loop1["topk"]])
            r = {
                "empty_layer_expert_pairs_in_batch": int((counts == 0).sum()),
                "grouped_vs_loop": envelope(grouped),
                "null_candidate_loop_vs_loop": envelope(null),
                "floor_loop_vs_loop": envelope(floor),
                "floor_samples": len(floor),
                "scale_loop_bf16_vs_fp32": compare(runs["fp32"], runs["bf16"]) if precision == "fp32" else None,
                "diagnostic_fork32_vs_loop": compare(loop1, run_once(model, x, y, precision, "fork32")),
                "negative_control_vs_loop": compare(loop1, run_once(model, x, y, precision, "badswap")),
                "teacher_forced": teacher_forced(model, loop1, precision),
            }
            loop1["moe_inputs"] = None  # GPU belleği
            report[f"{precision} | {name}"] = r
        # Ölçek iki hassasiyetin koşularından hesaplanır: bf16 bölümü de aynı ölçeği kullanır
        scale = report[f"fp32 | {name}"]["scale_loop_bf16_vs_fp32"]
        for precision in ("bf16", "fp32"):
            r = report[f"{precision} | {name}"]
            r["scale_loop_bf16_vs_fp32"] = scale
            g, fl, sc, neg = (r[k] for k in ("grouped_vs_loop", "floor_loop_vs_loop", "scale_loop_bf16_vs_fp32",
                                             "negative_control_vs_loop"))
            tf = r["teacher_forced"]
            tf_bound = max(3 * tf["layer_output_floor_rel_l2_max"], 1e-6)
            checks = {f"teacher-forced katman çıktısı {tf['layer_output_rel_l2_max']:.1e} <= {tf_bound:.1e}":
                      tf["layer_output_rel_l2_max"] <= tf_bound,
                      "gradyan anahtarları aynı": g["grad_keys_equal"]}
            r["structural_routing_same_given_same_input"] = tf["routing_identical_every_layer"]
            null_checks = {}
            for m in METRICS:
                bound = max(3 * fl[m], 0.25 * sc[m]) if precision == "bf16" else max(10 * fl[m], FP32_TOL[m])
                checks[f"{m}: {g[m]:.2e} <= {bound:.2e}"] = g[m] <= bound
                null_checks[m] = r["null_candidate_loop_vs_loop"][m] <= bound
            for m in ("logits", "grads", "grads_worst_tensor"):
                checks[f"neg. kontrol {m} ayırt ediyor"] = neg[m] >= max(10 * g[m], sc[m])
            r["checks"] = checks
            r["null_candidate_checks"] = null_checks
            ok &= all(checks.values())
            report[f"{precision} | {name}"] = r
            print(f"\n=== {precision} | {name} | batch'te boş (katman, expert) çifti: {r['empty_layer_expert_pairs_in_batch']}")
            print(f"  {'':26s}{'routing aynı':>13s}{'min uyum':>10s}{'logits':>10s}{'grads':>10s}{'en kötü t.':>11s}"
                  f"{'update':>10s}{'loss+1':>10s}  bit(logit/grad/param)")
            for label, c in (("grouped vs loop", g), ("sıfır aday: loop vs loop", r["null_candidate_loop_vs_loop"]),
                             ("taban: loop vs loop", fl), ("ölçek: loop bf16 vs fp32", sc),
                             ("tanı: fork32 vs loop", r["diagnostic_fork32_vs_loop"]), ("negatif kontrol", neg)):
                print(f"  {label:26s}{c['routing_identical_all_layers']!s:>13s}{c['routing_min_token_agreement']:10.5f}"
                      f"{c['logits']:10.2e}{c['grads']:10.2e}{c['grads_worst_tensor']:11.2e}{c['update']:10.2e}"
                      f"{c['loss_after_step']:10.2e}  {c['logits_bitexact']}/{c['grads_bitexact']}/{c['params_bitexact']}")
            print(f"  teacher-forced: routing her katmanda aynı (yapısal) {tf['routing_identical_every_layer']} | katman çıktısı "
                  f"rel maks {tf['layer_output_rel_l2_max']:.2e} (loop tekrarı {tf['layer_output_floor_rel_l2_max']:.2e}) "
                  f"bit={tf['layer_output_bitexact']}")
            print("  " + " | ".join(f"{k}: {'OK' if v else 'BAŞARISIZ'}" for k, v in checks.items()))
            print("  sıfır aday (loop tekrarı) aynı sınırlarla: " +
                  " | ".join(f"{k}: {'OK' if v else 'BAŞARISIZ'}" for k, v in null_checks.items()))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"all_checks_passed": ok, "results": report}, f, indent=2, ensure_ascii=False)
    print(f"\nTÜM KONTROLLER: {'GEÇTİ' if ok else 'BAŞARISIZ'}  -> {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
