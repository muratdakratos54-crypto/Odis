"""Bir MoE checkpoint'inde router davranışını ölçer: denge, uzmanlaşma ve "sahte denge" testleri.

    .venv\\Scripts\\python scripts/router_analysis.py --ckpt checkpoints/v3/auxfree-s1337/best.pt --out rapor.json

Soru şu: expert'ler yalnızca DENGELİ mi kullanılıyor, yoksa ANLAMLI biçimde mi ayrışmış? Denge tek başına
uzmanlaşma değildir; dengeleyici baskı tokenları semantik uyum yüzünden değil kota doldurmak için başka expert'e
yollayabilir ("sahte denge"). Ölçütler dört gruba ayrılıyor:

1. DENGE (gating'den bağımsız, yalnızca atama sayıları): yük CV, ölü expert, maks/min pay, kullanım entropisi.
2. UZMANLAŞMA
   * NMI = I(token kimliği ; seçilen expert) / H(expert): routing kararının token kimliğiyle ne kadar
     belirlendiği. Dengeli ama rastgele routing'de ~0, dengeli ve token'a göre ayrışmışta yüksek.
   * routing tutarlılığı: bir token türünün geçişlerinin kaçı o türün baskın expert'ine gidiyor (>= min_count
     geçişi olan türler, geçişle ağırlıklı). Kota baskısı aynı tokenı farklı expert'lere dağıtırsa düşer.
   * expert fonksiyon çeşitliliği: aynı girdilere tüm expert'leri uygulayıp çıktıların ikili kosinüs benzerliğinin
     ortalaması; 1 - benzerlik. Expert'ler dengeli ama birbirinin kopyasıysa (hepsi genelci) düşük çıkar.
   * embedding tutarlılığı: her expert'in baskın token kümesinin giriş embedding'lerinde içsel benzerliği eksi
     kümeler arası benzerlik (silhouette benzeri). Kümeler semantik olarak sıkıysa pozitif.
   * lift tablosu (nitel): expert başına en yüksek lift'li tokenlar, P(expert|token)/P(expert).
3. ROUTER GÜVENİ: seçilen expert'in gate değeri, seçim marjı (s1-s2)/s1, router entropisi. Kararlar kıl payıysa
   (marj ~0) routing'i küçük kuvvetler çevirebilir demektir.
4. EŞLEŞME ANLAMLI MI (karşı-olgusal CE, aynı sabit val pencereleri):
   * permütasyon: expert kimlikleri tutarlı bir permütasyonla değiştirilir. Yük dağılımı ve gate değerleri
     aynı kalır, yalnızca token<->expert eşleşmesi bozulur. CE artışı küçükse eşleşme anlamsızdı.
   * rastgele: seçim rastgele expert'lerle değiştirilir (gate değerleri korunur).
   * ikinci-en-iyi: top-k yerine k+1..2k sıraları kullanılır (router'ın kendi ikinci tercihi).
   * (aux-loss-free koşulda) bias'sız: seçim ham skorlarla yapılır. Bias'ı kaldırmak CE'yi DÜŞÜRÜYORSA
     dengeleme baskısı göreve zarar veriyor demektir.
5. DENGELEME KUVVETİNİN ÖLÇÜMÜ
   * aux loss'lu koşullarda: router logit'lerinde aux gradyanının büyüklüğü / görev gradyanının büyüklüğü ve
     iki gradyanın kosinüs hizası. Negatif hiza = dengeleme, görevin istediği yönün tersine çekiyor.
   * aux-loss-free koşulda: bias'ın router'ın kendi tercihini ezme oranı (seçim bias'lı ve bias'sız farklıysa),
     bias aralığı.

Sınırlar: tek val kümesi (eğitimdeki sabit pencereler), top-1 tabanlı uzmanlaşma ölçütleri, embedding tutarlılığı
modelin kendi embedding uzayına göre. Sayılar bu ölçek (16 expert, top-3, token düzeyi routing) için geçerlidir.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import numpy as _np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.checkpoint import load_model  # noqa: E402
from src.data import TokenDataset, dataset_paths  # noqa: E402
from src.model import GPT  # noqa: E402
from src.moe import MoE, load_balance_metrics, load_balancing_loss, router_z_loss  # noqa: E402


# --- router matematiği: MoE.forward ile aynı, ama seçimi karşı-olgusal kiplerde değiştirebiliyor ---------------

def route(moe: MoE, logits: torch.Tensor, mode: str = "learned", gen: torch.Generator | None = None):
    """(scores, sel_scores, topk_idx, gates). mode: learned | random | next | nobias."""
    k, E = moe.top_k, moe.n_experts
    scores = logits.sigmoid() if moe.gate == "sigmoid" else logits.softmax(dim=-1)
    sel = scores + moe.expert_bias if (moe.bias_rate > 0 and mode != "nobias") else scores
    if mode == "random":  # token başına k farklı rastgele expert (tekrarsız), vektörize
        idx = torch.rand(scores.size(0), E, device=logits.device, generator=gen).argsort(dim=-1)[:, :k]
    elif mode == "next":
        if E < 2 * k:  # k tane "sıradaki" expert yok; replika forward token_of'u k'ye göre kurduğu için
            raise ValueError(f"'next' kipi E >= 2k gerektirir (E={E}, k={k}); daha az sütun sessizce yanlış eşleşme verir")
        idx = sel.topk(2 * k, dim=-1).indices[:, k:]
    else:
        idx = sel.topk(k, dim=-1).indices
    top = scores.gather(-1, idx)
    gates = top / top.sum(dim=-1, keepdim=True) if moe.norm_topk_prob and k > 1 else top
    return scores, sel, idx, gates


def moe_forward_variant(mode: str, gen: torch.Generator | None = None, store: list | None = None):
    """MoE.forward'ın kopyası; yalnızca seçim kipi eklenmiş (ve istenirse router logit'lerini biriktirir).
    mode='learned' iken gerçek forward ile bit bit aynı olmalı (tests/test_router_metrics.py ve analiz başında
    assert ile doğrulanır). Router modülü forward'da çağrılmadığı (yalnızca ağırlığı kullanıldığı) için logit'lere
    hook ile ulaşılamıyor; gradyan ölçümü bu yüzden buradan geçiyor."""
    def forward(self, x):
        B, T, C = x.shape
        h = x.reshape(-1, C)
        N, E, k = h.size(0), self.n_experts, self.top_k
        with torch.autocast(device_type=h.device.type, enabled=False):
            logits = F.linear(h.float(), self.router.weight.float())
        if store is not None:
            store.append(logits)
        scores, _, topk_idx, gates = route(self, logits, mode, gen)
        flat_expert = topk_idx.reshape(-1)
        order = flat_expert.argsort(stable=True)
        token_of = order // k
        counts = torch.bincount(flat_expert, minlength=E)
        if self.impl == "loop" or not self._grouped_ok(h):
            outs = [e(c) for e, c in zip(self.experts, h[token_of].split(counts.tolist()))]
            expert_out = torch.cat(outs)
        else:
            expert_out = self._grouped_experts(h, token_of, counts)
        w = gates.reshape(-1)[order].unsqueeze(1).to(expert_out.dtype)
        y = torch.zeros(N, C, dtype=expert_out.dtype, device=h.device)
        y.index_add_(0, token_of, expert_out * w)
        if self.shared is not None:
            y = y + self.shared(h).to(y.dtype)
        if self.bias_rate > 0 and self.training:
            self.bias_counts += counts
        aux_probs = scores if self.gate == "softmax" else scores / scores.sum(dim=-1, keepdim=True)
        return y.view(B, T, C), {"aux_loss": load_balancing_loss(aux_probs, counts), "z_loss": router_z_loss(logits),
                                 "counts": counts.detach(), "topk_idx": topk_idx.detach()}
    return forward


# --- ölçütler --------------------------------------------------------------------------------------------------

def mutual_information(C: torch.Tensor) -> dict:
    """C: (V, E) token türü x expert sayım matrisi. I(T;E), H(E), H(T) ve NMI = I / H(E) (bit)."""
    total = C.sum()
    if total == 0:
        return {"mi_bits": 0.0, "h_expert_bits": 0.0, "nmi": 0.0}
    P = C.double() / total
    pt, pe = P.sum(1, keepdim=True), P.sum(0, keepdim=True)
    nz = P > 0
    mi = (P[nz] * (P[nz] / (pt * pe).expand_as(P)[nz]).log2()).sum().item()
    h_e = -(pe[pe > 0] * pe[pe > 0].log2()).sum().item()
    h_t = -(pt[pt > 0] * pt[pt > 0].log2()).sum().item()
    return {"mi_bits": mi, "h_expert_bits": h_e, "h_token_bits": h_t,
            "nmi": mi / h_e if h_e > 0 else 0.0, "utilization_entropy": h_e / math.log2(C.size(1))}


def routing_consistency(C: torch.Tensor, min_count: int = 50) -> float:
    """Token türünün geçişlerinin kaçı baskın expert'ine gidiyor (>= min_count geçişli türler, geçişle ağırlıklı)."""
    row = C.sum(1)
    keep = row >= min_count
    if keep.sum() == 0:
        return float("nan")
    return (C[keep].max(1).values.sum() / row[keep].sum()).item()


def mi_null(C: torch.Tensor, reps: int = 3, seed: int = 0) -> dict:
    """I(T;E)'nin plug-in tahmincisinin BAĞIMSIZLIK tabanı: aynı satır toplamları ve aynı yük dağılımıyla
    atamalar rastgele yapılsa mi_bits kaç okurdu. Plug-in tahminci sonlu örnekte yukarı yanlıdır ve yanlılık
    token bütçesiyle küçülür; bu taban olmadan mi_bits'in MUTLAK değeri (ve dolayısıyla H(E|T) = H(E) - I)
    farklı --batches ile üretilmiş analizler arasında karşılaştırılamaz."""
    if C.sum() == 0:
        return {"mi_bits_null": float("nan"), "h_cond_null": float("nan")}
    p = (C.sum(0).double() / C.sum()).clamp_min(0)
    p = (p / p.sum()).numpy()
    ns = C.sum(1).numpy()
    rng = _np.random.default_rng(seed)
    mis, hconds = [], []
    for _ in range(reps):
        sim = torch.from_numpy(rng.multinomial(ns, p))
        m = mutual_information(sim)
        mis.append(m["mi_bits"])
        hconds.append(m["h_expert_bits"] - m["mi_bits"])
    return {"mi_bits_null": sum(mis) / reps, "h_cond_null": sum(hconds) / reps}


def consistency_null(C: torch.Tensor, min_count: int = 50, reps: int = 20, seed: int = 0) -> float:
    """`routing_consistency`'nin BAĞIMSIZLIK null'ı: aynı satır toplamları (token türü frekansları) ve aynı
    expert yük dağılımıyla atamalar rastgele yapılsa tutarlılık ne okurdu.

    Neden 1/E değil: (i) sonlu sayımda maksimum yukarı yanlı (n=50 geçişte bile en büyük hücre 1/E'nin
    belirgin üstünde), (ii) yük dağılımı düzgün değilse baskın expert'e düşme şansı artar. Çökmüş bir
    routing'de null yükselir, bu yüzden ham tutarlılığı koşullar arası karşılaştırmak yanıltır: null'a göre
    okunmalı. Katman başına hesaplanır (C: tek katmanın token x top-1 expert sayımları)."""
    row = C.sum(1)
    keep = row >= min_count
    if keep.sum() == 0 or C.sum() == 0:
        return float("nan")
    p = (C.sum(0).double() / C.sum()).clamp_min(0)
    p = (p / p.sum()).numpy()
    ns = row[keep].numpy()
    rng = _np.random.default_rng(seed)
    uniq, inv = _np.unique(ns, return_inverse=True)
    frac = _np.empty(len(uniq))
    for j, n in enumerate(uniq):
        draws = rng.multinomial(int(n), p, size=reps)
        frac[j] = (draws.max(axis=1) / n).mean()
    return float((frac[inv] * ns).sum() / ns.sum())


def expert_output_diversity(moe: MoE, h: torch.Tensor, sample: int = 1024) -> float:
    """1 - (aynı girdilerde expert çıktılarının ortalama ikili kosinüs benzerliği). Expert'ler birbirinin
    kopyasıysa ~0, tamamen farklı işler yapıyorsa 1'e yakın."""
    with torch.no_grad():
        x = h[:sample].float()
        outs = torch.stack([e(x) for e in moe.experts])              # (E, S, d)
        outs = F.normalize(outs, dim=-1)
        sim = torch.einsum("esd,fsd->ef", outs, outs) / outs.size(1)  # ortalama kosinüs (token üzerinden)
        E = sim.size(0)
        off = (sim.sum() - sim.diagonal().sum()) / (E * (E - 1))
    return 1.0 - off.item()


def embedding_coherence(C: torch.Tensor, emb: torch.Tensor, min_count: int = 50, top: int = 64) -> float:
    """Her expert'in baskın token kümesinin embedding'lerinde içsel benzerliği eksi kümeler arası benzerlik."""
    row = C.sum(1)
    keep = row >= min_count
    if keep.sum() < 2:
        return float("nan")
    modal = C[keep].argmax(1)
    ids = torch.nonzero(keep).squeeze(1)
    groups = []
    for e in range(C.size(1)):
        sel = ids[modal == e]
        if sel.numel() >= 2:
            order = C[sel, e].argsort(descending=True)[:top]
            groups.append(F.normalize(emb[sel[order]].float(), dim=-1))
    if len(groups) < 2:
        return float("nan")
    within = sum(((g @ g.T).sum() - g.size(0)) / (g.size(0) * (g.size(0) - 1)) for g in groups) / len(groups)
    between, n = 0.0, 0
    for i, gi in enumerate(groups):
        for gj in groups[i + 1:]:
            between += (gi @ gj.T).mean().item()
            n += 1
    return (within.item() - between / n) if n else float("nan")


def top_lift(C: torch.Tensor, decode, min_count: int = 200, top: int = 8) -> list[list[str]]:
    """Expert başına en yüksek lift'li tokenlar: lift = P(expert|token) / P(expert)."""
    total = C.sum().double()
    pe = C.sum(0).double() / total
    row = C.sum(1)
    out = []
    for e in range(C.size(1)):
        lift = torch.where(row >= min_count, (C[:, e].double() / row.clamp_min(1)) / pe[e].clamp_min(1e-12),
                           torch.zeros_like(row, dtype=torch.float64))
        idx = lift.argsort(descending=True)[:top]
        out.append([f"{decode(int(i))}({lift[i]:.1f})" for i in idx if lift[i] > 0])
    return out


# --- ana akış --------------------------------------------------------------------------------------------------

@torch.no_grad()
def collect(model: GPT, batches: list, device: str) -> dict:
    """Katman başına routing istatistikleri + bir batch'in katman girdileri."""
    moes = [b.ffn for b in model.blocks if b.is_moe]
    V, E, k = model.cfg.vocab_size, moes[0].n_experts, moes[0].top_k
    C = torch.zeros(len(moes), V, E, dtype=torch.long)
    counts_all = torch.zeros(len(moes), E, dtype=torch.long)
    acc = {key: torch.zeros(len(moes), dtype=torch.double) for key in
           ("gate_top1", "margin", "entropy", "bias_override", "sel_gap_raw")}
    cap: dict[int, torch.Tensor] = {}
    hooks = [m.register_forward_pre_hook(lambda mod, inp, i=i: cap.__setitem__(i, inp[0].detach().reshape(-1, inp[0].size(-1))))
             for i, m in enumerate(moes)]
    n_tokens = 0
    first_h = None
    for bi, (x, y) in enumerate(batches):
        with torch.autocast(device, dtype=torch.bfloat16):
            model(x, y)
        ids = x.reshape(-1).cpu()
        n_tokens += ids.numel()
        for i, moe in enumerate(moes):
            # Router forward'da modül olarak çağrılmadığı için logit'i katman girdisinden yeniden hesaplıyoruz
            # (MoE.forward ile aynı işlem: float32'de F.linear).
            logits = F.linear(cap[i].float(), moe.router.weight.float())
            scores, sel, idx, gates = route(moe, logits)
            counts_all[i] += torch.bincount(idx.reshape(-1), minlength=E).cpu()
            C[i].view(-1).index_add_(0, ids * E + idx[:, 0].cpu(), torch.ones_like(ids))
            probs = scores / scores.sum(dim=-1, keepdim=True)
            top2 = sel.topk(2, dim=-1).values
            acc["gate_top1"][i] += gates[:, 0].double().sum().cpu()
            # Marj = seçim skorlarının farkı (mutlak). Oranlı tanım, bias eklenince negatif/1'i aşan skorlarla
            # anlamsızlaşıyordu. Softmax olasılıkları ile sigmoid skorları farklı ölçekte: karşılaştırma
            # softmax koşulları arasında anlamlı, sigmoid koşuluyla değil (bkz. bias ezme oranı ve ΔCE bias'sız).
            acc["margin"][i] += (top2[:, 0] - top2[:, 1]).double().sum().cpu()
            acc["entropy"][i] += (-(probs.clamp_min(1e-12).log() * probs).sum(-1) / math.log(E)).double().sum().cpu()
            acc["sel_gap_raw"][i] += (scores.max(-1).values - scores.topk(2, -1).values[:, 1]).double().sum().cpu()
            if moe.bias_rate > 0:  # bias router'ın kendi tercihini kaç tokende eziyor?
                raw = scores.topk(moe.top_k, dim=-1).indices
                same = (raw.sort(-1).values == idx.sort(-1).values).all(-1)
                acc["bias_override"][i] += (~same).double().sum().cpu()
        if bi == 0:
            first_h = {i: cap[i].clone() for i in cap}
    for h in hooks:
        h.remove()
    return {"C": C, "counts": counts_all, "acc": {k2: (v / n_tokens).tolist() for k2, v in acc.items()},
            "n_tokens": n_tokens, "first_h": first_h}


def counterfactual_ce(model: GPT, batches: list, device: str, seed: int = 0) -> dict:
    """Aynı val pencerelerinde CE: öğrenilmiş routing, permütasyon, rastgele, ikinci-en-iyi, (varsa) bias'sız."""
    real_forward = MoE.forward

    def ce(m: GPT) -> float:
        total = 0.0
        with torch.no_grad():
            for x, y in batches:
                with torch.autocast(device, dtype=torch.bfloat16):
                    _, _, info = m(x, y)
                total += info["ce"].item()
        return total / len(batches)

    out = {}
    try:
        gen = torch.Generator(device=device).manual_seed(seed)
        MoE.forward = moe_forward_variant("learned")
        out["learned_replica"] = ce(model)
        MoE.forward = real_forward
        out["learned"] = ce(model)
        # Replika, gerçek forward'ın kopyası: aynı olmalı. bf16 + CUDA index_add_ tekrar gürültüsü için küçük tolerans.
        out["replica_gap"] = abs(out["learned_replica"] - out["learned"])
        assert out["replica_gap"] < 2e-3, f"replika forward gerçekle uyuşmuyor: {out['replica_gap']:.2e}"
        for mode in ("random", "next"):
            MoE.forward = moe_forward_variant(mode, gen)
            out[mode] = ce(model)
        if any(m.bias_rate > 0 for m in model.modules() if isinstance(m, MoE)):
            MoE.forward = moe_forward_variant("nobias")
            out["nobias"] = ce(model)
        MoE.forward = real_forward
        perms = []
        for p in range(3):  # expert kimliklerini tutarlı permütasyonla değiştir (yük ve gate'ler aynı kalır)
            g = torch.Generator().manual_seed(seed + p)
            bad = copy.deepcopy(model)
            for m in bad.modules():
                if isinstance(m, MoE):
                    order = torch.randperm(m.n_experts, generator=g).tolist()
                    m.experts = nn.ModuleList([m.experts[i] for i in order])
            perms.append(ce(bad))
            del bad
            torch.cuda.empty_cache()
        out["permuted"] = sum(perms) / len(perms)
        out["permuted_runs"] = perms
    finally:
        MoE.forward = real_forward
    # Her kip kendi referansına göre: random/next/nobias replika forward'dan geçiyor, bu yüzden referansları da
    # replika ("learned_replica"); permuted gerçek forward'ı permüte edilmiş bir kopyada çalıştırıyor, referansı
    # gerçek ("learned"). Karışık referans, replica_gap'ten (tolerans 2e-3) küçük etkileri ölçülemez yapıyordu:
    # delta_nobias ~ 1e-4 mertebesinde ve eskiden bu farkın içinde kalıyordu.
    REF = {"random": "learned_replica", "next": "learned_replica", "nobias": "learned_replica", "permuted": "learned"}
    return {**out, **{f"delta_{m}": out[m] - out[REF[m]] for m in REF if m in out}}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--data_dir", default="data/sp8k")
    p.add_argument("--batches", type=int, default=20)  # 20 x 32 x 512 = 327k token: NMI ve lift için yeterli
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--min_count", type=int, default=50)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    model, ckpt = load_model(args.ckpt, device)
    model.eval()
    cfg = model.cfg
    assert cfg.n_experts > 0, "MoE olmayan checkpoint"
    _, val_path = dataset_paths(args.data_dir)
    ds = TokenDataset(val_path, cfg.max_seq_len, device)
    g = torch.Generator().manual_seed(12345)  # eğitimdeki eval ile aynı pencereler
    batches = [ds.get_batch(args.batch_size, g) for _ in range(args.batches)]

    data = collect(model, batches, device)
    C, counts = data["C"], data["counts"]
    bal = load_balance_metrics(counts.to(device))
    moes = [b.ffn for b in model.blocks if b.is_moe]
    from src.tokenizer import load_tokenizer
    tok = load_tokenizer(args.data_dir)

    per_layer = []
    for i, moe in enumerate(moes):
        mi = mutual_information(C[i])
        per_layer.append({
            "layer": i + cfg.first_dense_layers, **mi,
            "consistency": routing_consistency(C[i], args.min_count),
            "consistency_null": consistency_null(C[i], args.min_count),
            **mi_null(C[i]),
            "h_cond_bits": mi["h_expert_bits"] - mi["mi_bits"],
            "expert_diversity": expert_output_diversity(moe, data["first_h"][i]),
            "embedding_coherence": embedding_coherence(C[i], model.tok_emb.weight.detach().cpu(), args.min_count),
            "load_cv": (lambda f: (f.std(unbiased=False) / f.mean()).item())(counts[i].double() / counts[i].sum()),
            "gate_top1": data["acc"]["gate_top1"][i], "margin": data["acc"]["margin"][i],
            "entropy": data["acc"]["entropy"][i], "bias_override": data["acc"]["bias_override"][i],
            "raw_score_gap": data["acc"]["sel_gap_raw"][i],
        })

    def avg(key: str) -> float:
        vals = [r[key] for r in per_layer if not math.isnan(r[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    report = {
        "ckpt": args.ckpt, "step": ckpt.get("step"), "val_loss_ckpt": ckpt.get("val_loss"),
        "config": {k: getattr(cfg, k) for k in ("n_experts", "top_k", "aux_loss_coef", "z_loss_coef",
                                                "router_gate", "bias_update_rate", "moe_impl")},
        "tokens_analyzed": data["n_tokens"],
        "balance": {k: v for k, v in bal.items() if k != "load"},
        "load_per_layer": bal["load"],
        "specialization": {k: avg(k) for k in ("nmi", "mi_bits", "mi_bits_null", "h_cond_bits", "h_cond_null",
                                               "consistency", "consistency_null", "expert_diversity",
                                               "embedding_coherence", "utilization_entropy")},
        "router": {k: avg(k) for k in ("gate_top1", "margin", "entropy", "raw_score_gap", "bias_override")},
        "per_layer": per_layer,
        "counterfactual_ce": counterfactual_ce(model, batches, device),
        "gradient_tension": {},
        "bias": {},
    }
    if cfg.bias_update_rate > 0:
        b = torch.stack([m.expert_bias.detach().cpu() for m in moes])
        report["bias"] = {"range_mean": (b.max(1).values - b.min(1).values).mean().item(),
                          "abs_mean": b.abs().mean().item(), "sd_mean": b.std(dim=1, unbiased=False).mean().item(),
                          "per_layer": b.tolist()}
    lift_layer = min(len(moes) - 1, max(0, 4 - cfg.first_dense_layers))
    report["lift_examples"] = {"layer": lift_layer + cfg.first_dense_layers,
                               "experts": top_lift(C[lift_layer], lambda i: tok.decode([i]).replace("\n", "\\n"))}

    print(f"== {args.ckpt} (adım {report['step']}, {data['n_tokens']:,} token) ==")
    print(f"  config: {report['config']}")
    b = report["balance"]
    print(f"  DENGE: yük CV {b['load_cv']:.3f} | ölü expert {b['dead_experts']} | maks {b['load_max_ratio']:.2f}x "
          f"| min {b['load_min_ratio']:.2f}x | yük entropisi {b['load_entropy']:.4f} (top-k)")
    s = report["specialization"]
    print(f"  UZMANLAŞMA: NMI {s['nmi']:.4f} ({s['mi_bits']:.3f} bit) | tutarlılık {s['consistency']:.3f} "
          f"| null {s['consistency_null']:.3f} | top-1 kullanım entropisi {s['utilization_entropy']:.4f}"
          f"\n              I null (bağımsızlık tabanı) {s['mi_bits_null']:.3f} bit "
          f"| H(expert|token) {s['h_cond_bits']:.3f} bit "
          f"| expert çeşitliliği {s['expert_diversity']:.3f} (null ~1.0) | embedding tutarlılığı {s['embedding_coherence']:.3f}")
    r = report["router"]
    print(f"  ROUTER: top-1 gate {r['gate_top1']:.3f} | seçim marjı {r['margin']:.3f} | entropi {r['entropy']:.3f} "
          f"| ham skor farkı {r['raw_score_gap']:.3f}" + (f" | bias ezme oranı {r['bias_override']:.3f}"
                                                          if cfg.bias_update_rate > 0 else ""))
    c = report["counterfactual_ce"]
    print(f"  KARŞI-OLGUSAL CE (öğrenilmiş {c['learned']:.4f}): permütasyon {c['delta_permuted']:+.4f} | "
          f"rastgele {c['delta_random']:+.4f} | ikinci-en-iyi {c['delta_next']:+.4f}" +
          (f" | bias'sız {c['delta_nobias']:+.4f}" if "delta_nobias" in c else ""))
    if cfg.aux_loss_coef > 0:
        model_grad = load_model(args.ckpt, device)[0]
        model_grad.train()
        report["gradient_tension"] = gradient_tension(model_grad, batches[0], device, cfg.aux_loss_coef)
        gt = report["gradient_tension"]
        print(f"  DENGELEME KUVVETİ: |aux grad| / |görev grad| {gt.get('ratio_mean', float('nan')):.4f} | "
              f"kosinüs hizası {gt.get('cosine_mean', float('nan')):+.4f}")
    if report["bias"]:
        print(f"  BIAS: aralık {report['bias']['range_mean']:.3f} | sd {report['bias']['sd_mean']:.3f} "
              f"(|bias| ortalaması {report['bias']['abs_mean']:.3f} seçime GİRMEZ: topk ortak kaydırmaya duyarsız)")
    print(f"  katman {report['lift_examples']['layer']} lift örnekleri:")
    for e, toks in enumerate(report["lift_examples"]["experts"]):
        print(f"    expert {e:2d}: {' '.join(toks)}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"-> {args.out}")


def gradient_tension(model: GPT, batch, device: str, aux_coef: float) -> dict:
    """Router logit'lerinde dengeleme gradyanı ile görev gradyanının büyüklük oranı ve kosinüs hizası.
    GPT.forward CE ve aux'u detach ederek döndürdüğü için loss'ları burada graftan yeniden kuruyoruz."""
    moes = [b.ffn for b in model.blocks if b.is_moe]
    router_logits: list[torch.Tensor] = []
    stats: list[dict] = []
    hooks = [m.register_forward_hook(lambda mod, inp, out: stats.append(out[1])) for m in moes]
    real_forward = MoE.forward
    x, y = batch
    model.zero_grad(set_to_none=True)
    try:
        MoE.forward = moe_forward_variant("learned", store=router_logits)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits_out, _, _ = model(x, y)
            ce = F.cross_entropy(logits_out.float().view(-1, logits_out.size(-1)), y.reshape(-1))
            aux = torch.stack([s["aux_loss"] for s in stats]).mean()
    finally:
        MoE.forward = real_forward
        for h in hooks:
            h.remove()
    g_task = torch.autograd.grad(ce, router_logits, retain_graph=True, allow_unused=True)
    g_aux = torch.autograd.grad(aux, router_logits, allow_unused=True)
    rows = []
    for gt, ga in zip(g_task, g_aux):
        if gt is None or ga is None:
            continue
        gt, ga = gt.double().flatten(), ga.double().flatten()
        rows.append({"task_norm": gt.norm().item(), "aux_norm": (aux_coef * ga).norm().item(),
                     "ratio": ((aux_coef * ga).norm() / gt.norm().clamp_min(1e-30)).item(),
                     "cosine": (torch.dot(gt, ga) / (gt.norm() * ga.norm()).clamp_min(1e-30)).item()})
    return {"per_layer": rows,
            "ratio_mean": sum(r["ratio"] for r in rows) / len(rows) if rows else float("nan"),
            "cosine_mean": sum(r["cosine"] for r in rows) / len(rows) if rows else float("nan")}


if __name__ == "__main__":
    main()
