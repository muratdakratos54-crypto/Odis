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
4. Dengeleme: iki yöntem var, ModelConfig ile seçilir.
   * Klasik: load-balancing auxiliary loss (Switch Transformer) + router z-loss (ST-MoE). Aşağıda ayrıntılı.
   * Aux-loss-free (DeepSeek-V3 / Kimi K2): router softmax yerine expert başına bağımsız sigmoid skoru üretir;
     her expert'in bir "seçim bias'ı" olur ve top-k seçimi skor + bias üzerinden yapılır. Bias gradyanla değil,
     her optimizer adımında yükün ideal paydan sapmasının işaretiyle güncellenir (MoE.update_bias). Bias YALNIZCA
     seçime girer, gate ağırlığına girmez: böylece dengeleme kuvveti görev gradyanını hiç bozmaz. Aux loss'un
     aksine loss'a ek terim koymaz (bkz. README "Deney kaydı: v3-router-ablasyonu").
5. Expert hesabı iki yolla yapılabilir (ModelConfig.moe_impl), matematik ve parametreler aynı:
   * "loop"   : tokenlar expert'e göre sıralanır, her expert kendi dilimini ayrı bir modül çağrısıyla
                işler (E Python çağrısı, dilim boyları için CPU senkronu). v1 deneyleri bununla koştu.
   * "grouped": aynı sıralı tokenlar; tüm expert'lerin ağırlıkları (E, h, d) olarak istiflenip SwiGLU'nun üç matrisi
                için üç gruplu matris çarpımıyla (torch._grouped_mm, MegaBlocks'un "grouped GEMM" yolu) işlenir.
                offs = expert başına dilim sonları; dilim boyları CPU'ya çekilmez.
   Gruplu çarpımın gerçekten tek bir çekirdek olup olmadığı donanıma bağlı: RTX 4070 SUPER'da (SM89)
   PyTorch grup başına ayrı cuBLAS gemm çağırır (ölçüldü: 16 expert x 3 matris = 48 gemm, "loop" ile aynı
   çekirdekler). Kazanç orada Python/autocast yükünden ve gereksiz kopyalardan gelir, çekirdek
   birleşmesinden değil. Ayrıntılar ve ölçümler: README "Deney kaydı: v2-grouped-dispatch".
"""
from __future__ import annotations

import math
import warnings
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


def grouped_mm_supported(dtype: torch.dtype, *dims: int) -> bool:
    """torch._grouped_mm bu dtype ve matris boyutlarını kabul eder mi?

    Çekirdek float32/bf16/fp16 ister ve satır/sütun adımlarının 16 baytın katı olmasını şart koşar
    ("strides should be multiple of 16 bytes"): boyutlar float32'de 4'ün, bf16/fp16'da 8'in katı olmalı.
    Kural torch 2.14'te gerçek davranışla karşılaştırıldı: tests/test_moe.py CPU'da boyutlar 1..20 ve
    fp32/bf16/fp16/fp64 için sürekli denetler; CUDA'da (RTX 4070S) 1..40 bir kez elle denendi. Presetlerimiz
    (d=512, h=352; tiny: 128, 88) bunu sağlar.
    Not: torch.compile altında float32'de gruplu çağrı izlenemez (meta çekirdeği yalnızca bf16/fp16 kabul eder),
    orada graph break olur; bf16 autocast ile sorun yok. compile varsayılan kapalı (Windows'ta triton yok).
    """
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    align = 16 // torch.empty((), dtype=dtype).element_size()
    return all(d % align == 0 for d in dims)


def update_expert_biases(model: nn.Module) -> float:
    """Modeldeki tüm MoE katmanlarının aux-loss-free seçim bias'ını günceller (her optimizer adımından sonra
    bir kez çağrılır). Döndürdüğü değer en geniş bias aralığı (loglama için); dengeleme kapalıysa 0."""
    return max((m.update_bias() for m in model.modules() if isinstance(m, MoE)), default=0.0)


class _CastGatherRows(torch.autograd.Function):
    """(rows, rows_view) = h.to(dtype)[idx]  (idx: her token k kez geçer). İkinci çıktı birincinin görünümüdür.

    Önce cast, sonra gather: N x d satır bf16'ya çevrilip N*k satır bf16 kopyalanır; tersi (N*k satır float32
    kopyala, sonra çevir) iki kat bellek trafiği. İleri yönde iki sıra bit bit aynıdır (cast eleman bazında).

    Neden iki çıktı: satırları gate ve up kolları paylaşır. Tek bir tensör verseydik autograd iki kolun gradyanını
    bu bf16 tensörde, yani bf16'da toplardı; "loop" yolunda autocast her F.linear için ayrı cast yaptığından
    toplam float32'de olur. Her kola ayrı çıktı (kopyasız görünüm) verince iki gradyan ayrı gelir ve burada
    float32'de toplanır; bir tokenın k kopyasının gradyanları da float32'de birikir (index_add_). Böylece geri
    yön "loop" ile aynı işlemleri aynı dtype'ta yapar; ileri yön maliyeti değişmez.
    (İlk sürüm tek çıktı veriyordu: RTX 4070S'te bf16 gradyanları "loop"tan ~1 ulp farklıydı; bkz. README "Deney kaydı: v2-grouped-dispatch".)
    """

    @staticmethod
    def forward(ctx, h: torch.Tensor, idx: torch.Tensor, dtype: torch.dtype):
        ctx.save_for_backward(idx)
        ctx.h_shape, ctx.h_dtype = h.shape, h.dtype
        rows = h.to(dtype).index_select(0, idx)
        return rows, rows.view_as(rows)

    @staticmethod
    def backward(ctx, grad_a: torch.Tensor, grad_b: torch.Tensor):
        (idx,) = ctx.saved_tensors
        g = grad_a.to(ctx.h_dtype)
        # dtype değiştiyse g yeni bir tensördür, yerinde toplanabilir; değişmediyse autograd'ın tensörüne dokunma
        g = g.add_(grad_b) if g.data_ptr() != grad_a.data_ptr() else g + grad_b
        out = torch.zeros(ctx.h_shape, dtype=ctx.h_dtype, device=g.device)
        out.index_add_(0, idx, g)
        return out, None, None


class MoE(nn.Module):
    """Top-k routed expert'ler + opsiyonel shared expert.

    make_expert / make_shared: expert modülünü üreten fonksiyonlar. Expert'in iç yapısını MoE'den
    ayırıyoruz (dependency injection): "loop" yolunda expert herhangi bir FFN olabilir. "grouped" yol
    ağırlıkları istiflediği için expert'in bias'sız SwiGLU (w_gate, w_up, w_down) olmasını ister.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        make_expert: Callable[[], nn.Module],
        make_shared: Callable[[], nn.Module] | None = None,
        norm_topk_prob: bool = True,
        impl: str = "grouped",
        gate: str = "softmax",
        bias_rate: float = 0.0,
    ):
        super().__init__()
        assert 1 <= top_k <= n_experts
        assert impl in ("grouped", "loop"), impl
        assert gate in ("softmax", "sigmoid"), gate
        self.n_experts, self.top_k = n_experts, top_k
        self.norm_topk_prob = norm_topk_prob
        self.impl = impl
        self.gate, self.bias_rate = gate, bias_rate
        # Aux-loss-free dengeleme durumu. expert_bias modelin bir parçasıdır (çıkarımda da seçimi etkiler), bu yüzden
        # checkpoint'e yazılır; bias_counts iki güncelleme arasındaki yükü toplayan geçici sayaç.
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.register_buffer("bias_counts", torch.zeros(n_experts), persistent=False)
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
        # softmax: expert'ler arası olasılık dağılımı (aux loss bunu kullanır).
        # sigmoid: expert başına bağımsız uygunluk skoru (DeepSeek-V3 / Kimi K2).
        probs = logits.sigmoid() if self.gate == "sigmoid" else logits.softmax(dim=-1)
        if self.bias_rate > 0:
            # Seçim skoru = skor + bias; gate değeri ham skordan gelir (bias gradyanı hiç etkilemez)
            topk_idx = (probs + self.expert_bias).topk(k, dim=-1).indices
            topk_p = probs.gather(-1, topk_idx)
        else:
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
        # senkronla alınır (bincount; "loop" yolunda ayrıca .tolist()), E'den bağımsız.
        flat_expert = topk_idx.reshape(-1)          # (N*k,)  satır düzeni: [t0s0, t0s1, t1s0, ...]
        order = flat_expert.argsort(stable=True)    # expert'e göre sıralayan permütasyon
        token_of = order // k                       # sıralı her atamanın hangi tokena ait olduğu
        counts = torch.bincount(flat_expert, minlength=E)

        # ---- 3) Expert'leri çalıştır (boş dilimler de sorunsuz: (0, C) giriş -> (0, C) çıkış) ----
        if self.impl == "loop" or not self._grouped_ok(h):
            expert_in = h[token_of]                 # (N*k, C) - expert 0'ın tokenları, sonra 1'in...
            outs = [expert(chunk) for expert, chunk in zip(self.experts, expert_in.split(counts.tolist()))]
            expert_out = torch.cat(outs)            # (N*k, C)
        else:
            expert_out = self._grouped_experts(h, token_of, counts)

        # ---- 4) Combine: her çıktıyı gate ağırlığıyla çarpıp kendi tokenının konumuna topla ----
        w = gates.reshape(-1)[order].unsqueeze(1).to(expert_out.dtype)
        y = torch.zeros(N, C, dtype=expert_out.dtype, device=h.device)
        y.index_add_(0, token_of, expert_out * w)   # görev loss'unun gradyanı gates üzerinden router'a akar
        if self.shared is not None:
            y = y + self.shared(h).to(y.dtype)

        if self.bias_rate > 0 and self.training:
            self.bias_counts += counts  # adım sonunda update_bias() bunu kullanıp sıfırlar

        # sigmoid gating'de skorlar bir dağılım değil: aux loss (ve loglanan aux değeri) normalize edilmiş
        # skorlarla hesaplanır, böylece iki gating arasında ölçek karşılaştırılabilir kalır.
        aux_probs = probs if self.gate == "softmax" else probs / probs.sum(dim=-1, keepdim=True)
        stats = {
            "aux_loss": load_balancing_loss(aux_probs, counts),
            "z_loss": router_z_loss(logits),
            "counts": counts.detach(),               # expert başına atama sayısı (loglama için)
            "topk_idx": topk_idx.detach(),           # routing kararları (analiz betikleri için)
        }
        return y.view(B, T, C), stats

    @torch.no_grad()
    def update_bias(self) -> float:
        """Aux-loss-free dengeleme adımı (DeepSeek-V3 / Kimi K2): ideal payın üstünde yük alan expert'in seçim
        bias'ı bias_rate kadar düşer, altında kalanın yükselir (işaret adımı). Optimizer'a girmez, gradyan taşımaz.
        Döndürdüğü değer loglama için bias aralığı (maks - min)."""
        total = self.bias_counts.sum()
        if self.bias_rate <= 0 or total == 0:
            return 0.0
        load = self.bias_counts / total
        self.expert_bias -= self.bias_rate * torch.sign(load - 1.0 / self.n_experts)
        self.bias_counts.zero_()
        return (self.expert_bias.max() - self.expert_bias.min()).item()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # v1/v2 checkpoint'lerinde expert_bias yok (aux-loss-free dengeleme kapalıyken eğitildiler): sıfır kabul et
        key = prefix + "expert_bias"
        if key not in state_dict:
            state_dict[key] = torch.zeros_like(self.expert_bias)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _expert_dtype(self, h: torch.Tensor) -> torch.dtype:
        # "loop" yolunda F.linear autocast altında girdiyi ve ağırlığı autocast dtype'ına çevirir
        device = h.device.type
        return torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else h.dtype

    def _grouped_ok(self, h: torch.Tensor) -> bool:
        w = self.experts[0].w_down.weight  # (C, h)
        if grouped_mm_supported(self._expert_dtype(h), *w.shape):
            return True
        if not getattr(self, "_warned", False):
            warnings.warn(f"moe_impl='grouped': {self._expert_dtype(h)} ve boyutlar {tuple(w.shape)} "
                          f"torch._grouped_mm ile desteklenmiyor; aynı matematikteki 'loop' yolu kullanılıyor.")
            self._warned = True
        return False

    def _grouped_experts(self, h: torch.Tensor, token_of: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Tüm expert'lerin SwiGLU'su, expert'e göre sıralı N*k satır üzerinde 3 gruplu matris çarpımıyla.

        _grouped_mm(x, W, offs): x'in [offs[e-1], offs[e]) satırları W[e] ile çarpılır. Satır başına işlem
        "loop" yoluyla aynıdır (aynı ağırlık, aynı sıra); fark yalnızca çarpımların nasıl gruplandığı.
        """
        device = h.device.type
        # autocast'in _grouped_mm için ne yaptığına güvenmeden, "loop" yolunun F.linear'ı gibi elle çeviriyoruz
        dtype = self._expert_dtype(h)
        x_gate, x_up = _CastGatherRows.apply(h, token_of, dtype)  # (N*k, C), aynı bellek
        offs = counts.cumsum(0, dtype=torch.int32)                # expert e'nin dilimi: [offs[e-1], offs[e])
        with torch.autocast(device_type=device, enabled=False):
            # nn.Linear ağırlığı (out, in); _grouped_mm satır vektörünü sağdan çarptığı için (E, in, out) ister
            w_gate, w_up, w_down = (
                torch.stack([getattr(e, name).weight for e in self.experts]).to(dtype).transpose(1, 2)
                for name in ("w_gate", "w_up", "w_down")
            )
            gate = torch._grouped_mm(x_gate, w_gate, offs=offs)  # (N*k, h)
            up = torch._grouped_mm(x_up, w_up, offs=offs)        # (N*k, h)
            return torch._grouped_mm(F.silu(gate) * up, w_down, offs=offs)  # (N*k, C) - SwiGLU.forward ile aynı
