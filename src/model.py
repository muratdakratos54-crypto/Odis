"""Decoder-only Transformer (nanoGPT iskeleti + modern Llama/DeepSeek bileşenleri).

Orijinal GPT-2'ye göre yaptığımız değişiklikler ve nedenleri:
  * Öğrenilen pozisyon embedding'i yerine RoPE  -> göreli konum bilgisi, parametre yok,
    uzun bağlama genişletilebilir (Llama, DeepSeek-V3, Kimi K2 hepsi RoPE kullanır).
  * LayerNorm yerine RMSNorm                    -> daha basit/hızlı, pratikte aynı kalite.
  * GELU FFN yerine SwiGLU                      -> aynı parametre bütçesinde daha düşük loss
    (Shazeer 2020). MoE adımında her "expert" tam olarak bu SwiGLU bloğunun bir kopyası olacak.
  * Bias yok                                    -> bias'lar büyük modellerde katkı sağlamıyor.
  * Pre-norm (norm, alt katmandan ÖNCE)         -> residual yolu "temiz" kalır, derin ağlarda
    gradyan akışı stabil olur; warmup'a daha az duyarlı.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.moe import MoE


class RMSNorm(nn.Module):
    """x / RMS(x) * g.  LayerNorm'dan farkı: ortalamayı çıkarmaz (re-centering yok), bias yok."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kareler ortalamasını float32'de al: bf16'da küçük/büyük değerlerde hassasiyet kaybı olur.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE için cos/sin tabloları, şekil (T, head_dim/2).

    Her boyut çifti (i) kendi frekansıyla döner: freq_i = theta^(-2i/d). Düşük i -> hızlı dönen
    (yakın konumları ayırt eden), yüksek i -> yavaş dönen (uzak bağlamı kodlayan) boyutlar.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """q veya k vektörünü konumuna göre döndürür. x: (B, H, T, hd).

    Neden işe yarar: q_m ile k_n döndürüldükten sonra iç çarpımları yalnızca (m - n) farkına
    bağlı olur -> attention skorları doğal olarak *göreli* konum bilgisi taşır.
    (GPT-NeoX/Llama "rotate_half" düzeni: vektörün ilk yarısı ile ikinci yarısı eşleşir.)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        # q, k, v tek matris çarpımında: 3 ayrı kernel çağrısı yerine 1 (daha az GPU overhead'i)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj.is_residual_proj = True  # init'te özel ölçek (bkz. GPT._init_weights)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # RoPE yalnızca q ve k'ye uygulanır: konum, "kime bakacağım" kararını etkilemeli,
        # taşınan içeriği (v) değil.
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # PyTorch SDPA: T x T attention matrisini bellekte tutmayan (flash/mem-efficient) çekirdek
        # seçer. is_causal=True -> her token yalnızca kendisi ve öncekilere bakar (otoregresif).
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    """FFN(x) = W_down( SiLU(W_gate x) * W_up x ).

    Kapı (gate) kolu, yukarı projeksiyonun hangi boyutlarının geçeceğini veriye bağlı olarak
    seçer -> düz ReLU/GELU'dan daha ifade gücü yüksek. Transformer parametrelerinin ~2/3'ü
    FFN'lerdedir; MoE'nin çoğalttığı kısım da tam olarak burası.
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.w_down.is_residual_proj = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    """Pre-norm Transformer bloğu: x + Attn(Norm(x)), sonra x + FFN(Norm(x)).

    MoE'ye geçişte değişen TEK şey FFN kısmı: attention, normlar ve residual yapı aynı kalır.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.is_moe = cfg.n_experts > 0 and layer_idx >= cfg.first_dense_layers
        if self.is_moe:
            shared_hidden = cfg.expert_hidden * cfg.n_shared_experts
            self.ffn = MoE(
                cfg.d_model, cfg.n_experts, cfg.top_k,
                make_expert=lambda: SwiGLU(cfg.d_model, cfg.expert_hidden),
                # n_shared adet shared expert, tek bir geniş SwiGLU'ya eşdeğer (hepsi her tokena
                # uygulandığı için ayrı tutmanın anlamı yok) - DeepSeek de böyle uygular.
                make_shared=(lambda: SwiGLU(cfg.d_model, shared_hidden)) if shared_hidden > 0 else None,
                norm_topk_prob=cfg.norm_topk_prob,
            )
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin))
        if self.is_moe:
            h, stats = self.ffn(self.ffn_norm(x))
        else:
            h, stats = self.ffn(self.ffn_norm(x)), None
        x = x + self.drop(h)
        return x, stats


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)  # son norm: logit ölçeğini stabil tutar
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # Weight tying: "token -> vektör" ve "vektör -> token" aynı V x d matrisini paylaşır.
            # Çıkış (softmax) tarafı her adımda TÜM satırlar için gradyan alır; giriş embedding'inde
            # ise yalnızca batch'te geçen tokenların satırı güncellenir. Paylaşınca nadir tokenların
            # GİRİŞ vektörleri de çıkış tarafından her adımda eğitilir (Press & Wolf, 2017).
            # Ayrıca V x d parametre tasarrufu: dense preset'te 4.2M (~%14); çok küçük modellerde
            # (tiny preset: ~%72) embedding modelin çoğunluğu olabilir.
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
        # persistent=False: checkpoint'e yazılmaz, config'ten her zaman yeniden hesaplanabilir
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # GPT-2 başlatması: N(0, 0.02). Residual'a yazan projeksiyonlar ek olarak
        # 1/sqrt(2*n_layer) ile küçültülür: her blok residual'a 2 katkı ekler, bu ölçek sayesinde
        # derinlik arttıkça residual stream'in varyansı başlangıçta patlamaz.
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "is_residual_proj", False):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, exclude_embedding: bool = False, active_only: bool = False) -> int:
        """active_only: token başına gerçekten çalışan parametreler. MoE katmanında E expert'ten
        yalnızca top_k tanesi çalışır; kalan (E - k) expert'in parametreleri o token için "uyur"."""
        n = sum(p.numel() for p in self.parameters())  # parameters() paylaşılan ağırlığı 1 kez sayar
        if exclude_embedding:
            n -= self.tok_emb.weight.numel()
        if active_only:
            for block in self.blocks:
                if block.is_moe:
                    per_expert = sum(p.numel() for p in block.ffn.experts[0].parameters())
                    n -= (block.ffn.n_experts - block.ffn.top_k) * per_expert
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Döndürür: (logits, loss, info)
        loss = CE + aux_coef * aux_loss + z_coef * z_loss   (dense modelde sadece CE)
        info = {"ce", ve MoE varsa "aux_loss", "z_loss", "counts" (MoE katmanı x expert)} - hepsi detached.
        Model kalitesini karşılaştırırken info["ce"]'ye bakın: loss, MoE'de ceza terimlerini de içerir.
        """
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"dizi uzunluğu {T} > max_seq_len {self.cfg.max_seq_len}"
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        moe_stats = []
        for block in self.blocks:
            x, stats = block(x, cos, sin)
            if stats is not None:
                moe_stats.append(stats)
        x = self.norm(x)

        if targets is None:
            # Üretimde yalnızca son konumun logit'i gerekir: V x T'lik tensörü boşuna hesaplama
            return self.lm_head(x[:, [-1], :]), None, {}
        logits = self.lm_head(x)
        # Cross-entropy float32'de: bf16 log-softmax büyük sözlükte sayısal olarak hassas değil
        # targets.reshape: veri yükleyicideki y = xy[:, 1:] bitişik olmayan bir görünümdür, view çalışmaz
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        loss, info = ce, {"ce": ce.detach()}
        if moe_stats:
            # Katmanların ORTALAMASI (HF Mixtral/Qwen-MoE gibi; Switch Transformer ve DeepSeek ise
            # katman başına ekler = TOPLAM). Ortalamada loss'taki aux teriminin büyüklüğü ~aux_loss_coef
            # kalır ve log'daki aux değeri derinlikten bağımsız ~1 okunur. Bedeli: her router'ın
            # gördüğü etkin katsayı aux_loss_coef / n_moe (moe preset'inde 0.01/7 ≈ 0.0014), yani MoE
            # katmanı sayısı değişirse router başına dengeleme gücü de değişir. Bu zayıf katsayıyla
            # bile denge sağlandı (bkz. README sonuçları).
            aux = torch.stack([s["aux_loss"] for s in moe_stats]).mean()
            z = torch.stack([s["z_loss"] for s in moe_stats]).mean()
            loss = ce + self.cfg.aux_loss_coef * aux + self.cfg.z_loss_coef * z
            info.update(aux_loss=aux.detach(), z_loss=z.detach(),
                        counts=torch.stack([s["counts"] for s in moe_stats]))
        return logits, loss, info

    def configure_optimizer(self, lr: float, weight_decay: float, betas: tuple[float, float], device_type: str):
        # Weight decay yalnızca 2+ boyutlu tensörlere (matris çarpımı ağırlıkları, embedding).
        # Norm kazançları (1-boyutlu) sıfıra çekilmemeli: onlar ölçek parametresi, "karmaşıklık" değil.
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # fused=True: tüm parametre güncellemesini tek CUDA çekirdeğinde yapar (daha hızlı)
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Otoregresif örnekleme. Basitlik için KV-cache yok: her adımda bağlamı baştan işler.
        (512 token ve ~30M parametrede bu yeterince hızlı; KV-cache ileride eklenebilir.)"""
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)  # greedy
            else:
                logits = logits / temperature  # <1: daha keskin/tutarlı, >1: daha çeşitli/rastgele
                if top_k is not None and top_k > 0:  # top_k <= 0 veya None: filtre kapalı
                    # Olasılığı en yüksek k token dışındakileri ele: "kuyruk"taki saçma tokenları kes
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None:
                # Batch'teki her örnek kendi EOS'una farklı adımda ulaşır; hepsi bitince dur.
                # (EOS sonrası üretilenleri çağıran taraf keser.)
                finished |= next_id.squeeze(1) == eos_id
                if bool(finished.all()):
                    break
        return idx
