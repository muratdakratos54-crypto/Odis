"""Mixture-of-Experts (MoE) FFN katmanı.

Temel fikir
-----------
Dense Transformer'da her token aynı FFN'den geçer. MoE'de E adet FFN ("expert") vardır ve küçük
bir router ağı her token için bunlardan yalnızca k tanesini seçer:
  * Toplam parametre (modelin "bilgi kapasitesi") E ile büyür,
  * Token başına hesap (FLOPs) ise yalnızca seçilen k expert kadardır.
Kimi K2 bu sayede 1T toplam parametreye sahipken token başına yalnızca ~32B parametre çalıştırır
(384 routed expert, top-8, +1 shared expert).

Bu dosyadaki tasarım kararları (DeepSeek-V3 / Kimi K2'den ilham, ev ölçeğine indirgenmiş)
-------------------------------------------------------------------------------------------
1. İnce taneli (fine-grained) expert'ler (DeepSeekMoE "fine-grained segmentation"): dense FFN'in
   (hidden=1408) 4 tam kopyası kadar parametreyi, her kopyayı 4'e bölerek 16 adet küçük
   (hidden=352) expert olarak tutuyoruz. Token başına 3 routed + 1 shared = 4 x 352 = 1408 -> dense
   modelle AYNI aktif hesap; toplam FFN parametresi ise ~4.25x. Kazanç: 16'dan 3 seçmenin
   C(16,3)=560 kombinasyonu var; 4 büyük expert'ten 1 seçmenin sadece 4. Daha çok kombinasyon =
   daha esnek uzmanlaşma.
2. Shared (paylaşılan) expert: her token her zaman ondan da geçer. Dilbilgisi gibi *her* token için
   gereken ortak bilgi bu expert'te toplanır; routed expert'ler aynı bilgiyi tekrar tekrar öğrenmek
   zorunda kalmaz ve gerçekten uzmanlaşabilir (DeepSeekMoE makalesi).
3. Dropless routing (kapasite limiti yok): Switch/GShard, çoklu GPU'da sabit tensör boyutları için
   her expert'e "capacity" koyar ve taşan tokenları düşürür. Tek GPU'da dinamik boyutlarla çalışmak
   kolay olduğundan hiçbir tokenı düşürmüyoruz -> kalite kaybı yok, kod daha basit.
4. Load-balancing auxiliary loss (Switch Transformer) + router z-loss (ST-MoE). Aşağıda ayrıntılı.
   Not: DeepSeek-V3 ve Kimi K2 aux loss yerine "aux-loss-free" bir bias ayarlaması + sigmoid
   gating kullanır; biz burada klasik, iyi anlaşılmış aux loss yöntemini uyguluyoruz.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_balancing_loss(probs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Switch Transformer auxiliary loss (top-k'ye genelleştirilmiş):  L = E * sum_i f_i * P_i

    probs : (N, E) router softmax olasılıkları.
    counts: (E,) expert başına top-k atama sayısı (toplamı N*k). forward'da zaten hesaplandığı için
            yeniden hesaplamıyoruz: GPU'da bincount, çıktı boyunu bulmak için CPU ile senkronize olur.

    f_i : token atamalarının (N*k tanesinin) expert i'ye düşen oranı.  sum f = 1.
          argmax/topk'dan geldiği için TÜREVLENEMEZ (sabit gibi davranır).
    P_i : router'ın expert i'ye verdiği ortalama olasılık.             sum P = 1.  Türevlenebilir.

    Neden işe yarar?  dL/dP_i = E * f_i  -> çok token alan (f_i büyük) expert'in olasılığı daha
    güçlü aşağı itilir. Router, yükü dengelemek için olasılığı az kullanılan expert'lere kaydırır.
    Neden gerekli?  Kendi haline bırakılan router "zengin daha zengin olur" döngüsüne girer: biraz
    daha çok seçilen expert daha çok gradyan alır, daha iyi olur, daha çok seçilir... Sonunda birkaç
    expert her şeyi yapar, diğerleri ölür (expert collapse) -> parametrelerin çoğu boşa gider.

    Değer aralığı: mükemmel denge (f = P = 1/E) -> 1.0 ;  tam çöküş (hep aynı k expert) -> E/k.
    """
    n_experts = probs.size(-1)
    f = counts.to(probs.dtype) / counts.sum()
    P = probs.mean(dim=0)
    return n_experts * torch.sum(f * P)


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """ST-MoE z-loss: mean( logsumexp(logits)^2 ).

    Router logit'lerinin büyümesini cezalandırır. Büyük logit'ler softmax'ı doygunluğa sokar
    (olasılıklar 0/1'e yapışır, gradyan kaybolur) ve düşük hassasiyetli (bf16/fp16) hesapta
    yuvarlama hatalarını büyütür (ST-MoE). Router'ı zaten float32'de çalıştırıyoruz; z-loss logit'leri
    küçük tutarak ek güvence sağlar. Denge sağlamaz; sadece router'ı sayısal olarak sağlıklı tutar.
    """
    return torch.logsumexp(logits, dim=-1).pow(2).mean()


def load_balance_metrics(counts: torch.Tensor) -> dict:
    """counts: (MoE katmanı, E) expert başına atama sayıları -> okunabilir denge metrikleri.

    load_cv        : yük oranlarının std/ortalaması (katman ortalaması). 0 = mükemmel denge.
                     Tam çöküşte (hep aynı k expert) sqrt(E/k - 1) olur; E=16, k=3 -> ~2.08.
    load_max_ratio : en yüklü expert, ideal payın (1/E) kaç katını alıyor (tüm katmanlarda en kötü).
    load_min_ratio : en az yüklü expert'in ideal paya oranı (0 = hiç token almayan expert var).
    load_entropy   : yük dağılımının normalize entropisi, 1 = düzgün. Tam çöküşte (hep aynı k expert)
                     log(k)/log(E) olur (E=16, k=3 -> ~0.40); 0 yalnızca top_k=1'de mümkün.
    dead_experts   : ideal payın %10'undan azını alan (fiilen ölü) expert sayısı, tüm katmanlarda.
    """
    f = counts.double() / counts.sum(dim=-1, keepdim=True).clamp_min(1)
    E = f.size(-1)
    cv = (f - f.mean(-1, keepdim=True)).pow(2).mean(-1).sqrt() / f.mean(-1)
    entropy = -(f * f.clamp_min(1e-12).log()).sum(-1) / math.log(E)
    return {
        "load": f.tolist(),
        "load_cv": cv.mean().item(),
        "load_max_ratio": (f.max(-1).values * E).max().item(),
        "load_min_ratio": (f.min(-1).values * E).min().item(),
        "load_entropy": entropy.mean().item(),
        "dead_experts": int((f < 0.1 / E).sum().item()),
    }


def load_sparkline(fractions: list[float]) -> str:
    """Bir katmanın expert yüklerini tek satırlık çubuk grafiğe çevirir (konsol logu için).
    Her karakter bir expert, yük / ideal pay (1/E) oranının en yakın 0.25'lik seviyesi:
    '▁' ~0.25x, '▄' = ideal pay (1.0x), '█' ~1.9x ve üstü,
    ' ' = ideal payın %10'undan az = ölü expert (load_balance_metrics'teki dead_experts ile aynı eşik)."""
    bars = " ▁▂▃▄▅▆▇█"
    E = len(fractions)
    return "".join(" " if f * E < 0.1 else bars[max(1, min(8, round(f * E * 4)))] for f in fractions)


class MoE(nn.Module):
    """Top-k routed expert'ler + opsiyonel shared expert.

    make_expert / make_shared: expert modülünü üreten fonksiyonlar. Expert'in iç yapısını MoE'den
    ayırıyoruz (dependency injection): expert herhangi bir FFN olabilir; biz SwiGLU veriyoruz.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        make_expert: Callable[[], nn.Module],
        make_shared: Callable[[], nn.Module] | None = None,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts, self.top_k = n_experts, top_k
        self.norm_topk_prob = norm_topk_prob
        # Router = tek bir bias'sız lineer katman: token vektörünün her expert'in "anahtar"
        # vektörüyle iç çarpımı -> o expert'e uygunluk skoru.
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([make_expert() for _ in range(n_experts)])
        self.shared = make_shared() if make_shared is not None else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, T, C = x.shape
        h = x.reshape(-1, C)  # (N, C): routing token bazında; batch/konum yapısı önemsiz
        N, E, k = h.size(0), self.n_experts, self.top_k

        # ---- 1) Router: hangi token hangi expert'lere gidecek? ----
        # Router'ı autocast dışında, float32'de çalıştır: softmax/top-k kararları küçük logit
        # farklarına duyarlı; bf16 yuvarlaması routing'i gürültülü yapar. Maliyeti ihmal edilebilir.
        # Ağırlığı da .float()'a çeviriyoruz ki model bütünüyle bf16'ya dönüştürülse bile çalışsın.
        with torch.autocast(device_type=h.device.type, enabled=False):
            logits = F.linear(h.float(), self.router.weight.float())  # (N, E)
        probs = logits.softmax(dim=-1)
        topk_p, topk_idx = probs.topk(k, dim=-1)  # (N, k)
        # Seçilen k olasılığı toplamı 1 olacak şekilde normalize et: çıktının ölçeği, router'ın
        # olasılık kütlesini kaç expert'e yaydığından bağımsız olsun (DeepSeek/Kimi/Mixtral böyle).
        # k=1'de normalize ETMEYİZ: p/p = 1 sabit olur, türevi 0'dır ve router görev loss'undan hiç
        # gradyan alamaz (yalnızca aux loss'la eğitilir). Switch Transformer bu yüzden ham p kullanır.
        if self.norm_topk_prob and k > 1:
            gates = topk_p / topk_p.sum(dim=-1, keepdim=True)
        else:
            gates = topk_p

        # ---- 2) Dispatch: (token, expert) çiftlerini expert'e göre sırala ----
        # Naif yol "her expert için torch.where(idx == e)" E kez GPU->CPU senkronu gerektirir.
        # Bunun yerine tüm N*k atamayı expert id'sine göre bir kez sıralıyoruz; böylece her
        # expert'in tokenları bitişik bir dilim olur ve dilim boyları katman başına sabit sayıda
        # senkronla alınır (bincount + .tolist()), E'den bağımsız.
        flat_expert = topk_idx.reshape(-1)          # (N*k,)  satır düzeni: [t0s0, t0s1, t1s0, ...]
        order = flat_expert.argsort(stable=True)    # expert'e göre sıralayan permütasyon
        token_of = order // k                       # sıralı her atamanın hangi tokena ait olduğu
        counts = torch.bincount(flat_expert, minlength=E)
        expert_in = h[token_of]                     # (N*k, C) - expert 0'ın tokenları, sonra 1'in...

        # ---- 3) Expert'leri çalıştır (boş dilimler de sorunsuz: (0, C) giriş -> (0, C) çıkış) ----
        outs = [expert(chunk) for expert, chunk in zip(self.experts, expert_in.split(counts.tolist()))]
        expert_out = torch.cat(outs)                # (N*k, C)

        # ---- 4) Combine: her çıktıyı gate ağırlığıyla çarpıp kendi tokenının konumuna topla ----
        w = gates.reshape(-1)[order].unsqueeze(1).to(expert_out.dtype)
        y = torch.zeros(N, C, dtype=expert_out.dtype, device=h.device)
        y.index_add_(0, token_of, expert_out * w)   # görev loss'unun gradyanı gates üzerinden router'a akar
        if self.shared is not None:
            y = y + self.shared(h).to(y.dtype)

        stats = {
            "aux_loss": load_balancing_loss(probs, counts),
            "z_loss": router_z_loss(logits),
            "counts": counts.detach(),               # expert başına atama sayısı (loglama için)
            "topk_idx": topk_idx.detach(),           # routing kararları (analiz betikleri için)
        }
        return y.view(B, T, C), stats
