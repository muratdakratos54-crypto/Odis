"""Analitik aktif FLOP hesabı: karşılaştırmalarda BİRİNCİL hesap ölçüsü.

Türetim ve varsayımlar adım adım: docs/flops.md. Kısaca:
  * Birim: bir çarpma + bir toplama çifti = 2 FLOP (Kaplan ve ark. 2020, Hoffmann ve ark. 2022 ile aynı).
    d_in -> d_out lineer katman, token başına 2 * d_in * d_out FLOP.
  * "Aktif": token başına gerçekten çalışan hesap. MoE katmanında E expert'ten yalnızca top_k tanesi + shared
    expert + router sayılır; token almayan expert'lerin parametreleri hesaba girmez.
  * Birincil toplam = matris çarpımları (tüm lineer katmanlar, lm_head) + causal attention'ın bağlam terimi
    (q.k skorları ve değerlerin ağırlıklı toplamı). Eleman bazlı işlemler (norm, SiLU, softmax, residual,
    gate çarpımı ...) ayrı ve yaklaşık olarak `secondary_forward_flops_per_token` ile verilir; bütçelere girmez.
  * Eğitim = ileri x 3: her matris çarpımının geri yayılımı iki matris çarpımıdır (girdi ve ağırlık gradyanı).
  * Dispatch (argsort, bincount, gather/scatter kopyaları) 0 FLOP: tamsayı ve bellek işlemleri, aritmetik
    değil. "loop" ile "grouped" yolların FLOP'u bu yüzden tanım gereği aynıdır; farkları yalnızca duvar saatinde.

Profiler (torch.utils.flop_counter) yalnızca ÇAPRAZ KONTROL: formülü olmayan bir çekirdeği (ör. gruplu matris
çarpımı) sıfır sayabilir ve attention'ı causal maskeyi yok sayarak T x T üzerinden sayar (bkz. scripts/flops_report.py).
"""
from __future__ import annotations

from src.config import ModelConfig


def _layer_counts(cfg: ModelConfig) -> tuple[int, int]:
    """(dense katman, MoE katmanı) - model.Block.is_moe ile aynı kural."""
    n_moe = sum(1 for i in range(cfg.n_layer) if cfg.n_experts > 0 and i >= cfg.first_dense_layers)
    return cfg.n_layer - n_moe, n_moe


def forward_flops_per_token(cfg: ModelConfig, seq_len: int | None = None) -> dict[str, int]:
    """Token başına İLERİ yön aktif FLOP'u, bileşen bileşen (birincil tanım). seq_len: eğitim dizisi uzunluğu."""
    d, T, V = cfg.d_model, seq_len or cfg.max_seq_len, cfg.vocab_size
    n_dense, n_moe = _layer_counts(cfg)
    return {
        # q, k, v (tek d -> 3d matrisi) + çıkış projeksiyonu (d -> d): 2 * (3d^2 + d^2)
        "attn_qkv_proj": cfg.n_layer * 2 * 4 * d * d,
        # Causal: t. konum (0'dan) t+1 konuma bakar. Skorlar 2*d*(t+1), değerlerin ağırlıklı toplamı 2*d*(t+1).
        # T uzunluğundaki dizide konum ortalaması: 4*d*(T+1)/2 = 2*d*(T+1)
        "attn_context": cfg.n_layer * 2 * d * (T + 1),
        # SwiGLU: w_gate, w_up (d -> h), w_down (h -> d)
        "ffn_dense": n_dense * 2 * 3 * d * cfg.ffn_hidden,
        "moe_router": n_moe * 2 * d * cfg.n_experts,
        "moe_routed_experts": n_moe * cfg.top_k * 2 * 3 * d * cfg.expert_hidden,
        "moe_shared_experts": n_moe * cfg.n_shared_experts * 2 * 3 * d * cfg.expert_hidden,
        # Tied embedding'de de çıkış projeksiyonu tam bir d -> V matris çarpımıdır (giriş lookup'ı 0 FLOP)
        "lm_head": 2 * d * V,
    }


def secondary_forward_flops_per_token(cfg: ModelConfig, seq_len: int | None = None) -> dict[str, float]:
    """Eleman bazlı işlemlerin YAKLAŞIK ileri yön FLOP'u (1 aritmetik işlem = 1 FLOP). Bütçelere girmez;
    birincil tanımın bunları dışarıda bırakmasının karşılaştırmayı ne kadar etkilediğini göstermek için."""
    d, T, V, nh = cfg.d_model, seq_len or cfg.max_seq_len, cfg.vocab_size, cfg.n_head
    n_dense, n_moe = _layer_counts(cfg)
    E, k, he = cfg.n_experts, cfg.top_k, cfg.expert_hidden
    mean_ctx = (T + 1) / 2
    silu_and_gate = 4  # SiLU: exp, 1+, bölme; çarpı "up" kolu
    return {
        # RMSNorm (kare d, ortalama d, rsqrt 1, çarp d, kazanç d ~ 4d): katman başına 2 + son norm
        "rmsnorm": (2 * cfg.n_layer + 1) * 4 * d,
        "rope_qk": cfg.n_layer * 2 * 3 * d,             # q ve k: 2 çarpma + 1 toplama / eleman
        "attn_scale_softmax": cfg.n_layer * 4 * nh * mean_ctx,  # 1/sqrt(hd) + exp, toplam, bölme
        "residual_add": cfg.n_layer * 2 * d,
        "ffn_dense_act": n_dense * silu_and_gate * cfg.ffn_hidden,
        "moe_expert_act": n_moe * (k + cfg.n_shared_experts) * silu_and_gate * he,
        "moe_router_softmax_topk": n_moe * (3 * E + E + 2 * k),  # softmax, top-k taraması, gate normalizasyonu
        "moe_combine": n_moe * (2 * k * d + cfg.n_shared_experts * d),  # gate x çıktı + index_add, shared ekleme
        "moe_aux_z_loss": n_moe * 3 * E,
        "cross_entropy": 3 * V,
    }


def train_flops_per_token(cfg: ModelConfig, seq_len: int | None = None) -> int:
    """Birincil tanımla eğitim (ileri + geri) FLOP'u / token = 3 x ileri."""
    return 3 * sum(forward_flops_per_token(cfg, seq_len).values())
