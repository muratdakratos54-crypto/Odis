"""Model ve eğitim konfigürasyonları + hazır preset'ler.

Tüm hiperparametreler tek yerde dataclass olarak durur; checkpoint'e dict olarak yazılır,
böylece bir checkpoint'i yüklerken modeli *aynı* mimariyle yeniden kurabiliriz.
Komut satırından her alan ezilebilir:  python -m src.train --preset dense --lr 5e-4
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace


@dataclass
class ModelConfig:
    vocab_size: int = 8192      # veri hazırlarken tokenizer'dan okunur (meta.json)
    max_seq_len: int = 512      # bağlam uzunluğu; TinyStories hikâyeleri ~200-250 token
    d_model: int = 512          # residual stream genişliği
    n_layer: int = 8
    n_head: int = 8             # head_dim = 512/8 = 64 (SDPA çekirdekleri için verimli boyut)
    # SwiGLU gizli boyutu. Klasik FFN 4*d kullanır; SwiGLU'da 3 matris olduğu için parametre
    # sayısını eşitlemek adına 8/3*d ≈ 1365 alınır, GPU verimliliği için 64'ün katına yuvarlanır.
    ffn_hidden: int = 1408
    rope_theta: float = 10000.0
    dropout: float = 0.0        # tek epoch'tan az veri görüyoruz -> overfit yok, dropout gereksiz
    tie_embeddings: bool = True  # giriş embedding'i = çıkış projeksiyonu (bkz. model.py)
    # --- Mixture-of-Experts (ayrıntılar: src/moe.py). n_experts = 0 -> tamamen dense model ---
    n_experts: int = 0          # katman başına routed expert sayısı (E)
    top_k: int = 2              # token başına seçilen routed expert sayısı (k)
    expert_hidden: int = 352    # her expert'in SwiGLU gizli boyutu (ince taneli: dense'in 1/4'ü)
    n_shared_experts: int = 0   # her tokenın her zaman geçtiği expert sayısı
    # İlk katman(lar) dense kalır (DeepSeek-V3/Kimi K2: first_k_dense_replace). Erken katmanlar
    # daha çok token-yerel/genel özellik işler; burada routing'den kazanç az, dengesizlik riski yüksek.
    first_dense_layers: int = 1
    norm_topk_prob: bool = True  # seçilen k gate ağırlığını toplamı 1 olacak şekilde normalize et
    # Router çıktısı: "softmax" (expert'ler arası dağılım, aux loss ile dengelenir) ya da "sigmoid" (expert başına
    # bağımsız skor; DeepSeek-V3 / Kimi K2'nin aux-loss-free yolunda bununla gelir).
    router_gate: str = "softmax"
    # Aux-loss-free dengeleme adım büyüklüğü (gamma). >0 ise her expert'in yalnızca SEÇİMDE kullanılan bir bias'ı
    # olur ve her optimizer adımında yük sapmasının işaretiyle güncellenir (bkz. src/moe.py MoE.update_bias).
    # DeepSeek-V3 eğitiminin çoğunda 0.001 kullandı. aux_loss_coef ile birlikte de kullanılabilir (ikisi bağımsız).
    bias_update_rate: float = 0.0
    # Katsayılar MoE katmanlarının ORTALAMASINA uygulanır (bkz. GPT.forward). Switch Transformer
    # (0.01) ve ST-MoE (0.001) aynı değerleri katman başına kullanıp toplar; bizdeki eşdeğeri
    # router başına coef / n_moe_katman.
    aux_loss_coef: float = 0.01  # load-balancing loss katsayısı
    z_loss_coef: float = 0.001   # router z-loss katsayısı
    # Expert hesabının yolu (bkz. src/moe.py): "grouped" = gruplu matris çarpımı, "loop" = expert başına
    # ayrı çağrı (v1). Matematik ve parametreler aynı; yalnızca hız ve kayan nokta yuvarlaması farklı olabilir.
    moe_impl: str = "grouped"


@dataclass
class TrainConfig:
    run_name: str = "dense"
    data_dir: str = "data/sp8k"
    out_dir: str = "checkpoints"
    # --- batch: efektif batch = batch_size * grad_accum * max_seq_len token ---
    # Ölçüm (scripts/benchmark.py, dense 30M, seq 512, RTX 4070S):
    #   batch 16: 3.3 GB 128k tok/s | 32: 6.2 GB 116k | 48: 9.0 GB 110k | 64: 11.9 GB 45k (!)
    # 64'te VRAM dolunca Windows sürücüsü sistem RAM'ine taşıyor ve hız 2.5x düşüyor.
    # 32 seçimi: MoE (daha çok aktivasyon + expert ağırlıkları) ve eval için ~5 GB pay bırakır.
    batch_size: int = 32        # GPU'ya tek seferde giren mikro-batch (VRAM belirler)
    grad_accum: int = 2         # 2 x 32 x 512 = 32 768 token / optimizer adımı
    max_steps: int = 3000
    # Eğitim bütçesi ve LR schedule GÖRÜLEN TOKEN üzerinden tanımlıdır (bkz. train.lr_at). 0 -> adım
    # sayısından türet: max_steps * token/adım. Farklı hızdaki modelleri aynı süre ya da aynı FLOP'ta
    # karşılaştırırken bütçe token olarak verilir; warmup her modelde aynı token sayısında biter.
    max_tokens: int = 0
    stop_step: int = 0          # >0: bu adımda dur, schedule yine bütçeye göre (kısa doğrulama koşuları için)
    # --- optimizer ---
    lr: float = 1e-3            # küçük modeller büyük LR'ı tolere eder (GPT-3 tablosu: ~6e-4 @125M)
    min_lr: float = 1e-4        # cosine decay sonu = lr/10 (Chinchilla tarzı)
    warmup_steps: int = 200
    warmup_tokens: int = 0      # 0 -> warmup_steps * token/adım
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95         # 0.999 yerine 0.95: LLM eğitiminde loss spike'larına daha dayanıklı
    grad_clip: float = 1.0
    # --- değerlendirme / log ---
    eval_interval: int = 250
    eval_iters: int = 40
    # Raporlanan val loss'ta ATLANACAK ilk pozisyon sayısı (0 = hepsi sayılır).
    #
    # Neden var: FOTON'un router'ı DİZİ düzeyinde tek bir modül seçiyor. Router girdiyi dizinin
    # tamamından okursa token 5'in tahmini token 6..511'den bilgi taşır — nedensellik ihlali, ve
    # ölçüsü küçük değil: log(3)/512 = 0.002146 nat/token tavanı, ölçülen eğimde +0.63 puan bedava
    # "tasarruf". Bu yüzden router yalnızca ilk `k` token'lık NEDENSEL prefix'i görür ve raporlanan
    # val loss `k..T-1` pozisyonları üzerinden hesaplanır: prefix'ten seçilen yol, yalnızca
    # prefix'ten SONRAKİ token'ları etkilediği yerde ölçülür. Dört kolda da aynı maske uygulanır,
    # yoksa FOTON'un val loss'u ile baseline'ların val loss'u aynı nicelik olmaz.
    # `ignore_index=-1` ile uygulanıyor (bkz. src/model.py), yani maskelenen pozisyonlar loss'a
    # hiç girmez; girdi olarak yine görülürler (bağlam kaybı yok).
    loss_prefix_skip: int = 0
    log_interval: int = 25
    sample_tokens: int = 120    # her eval'de kısa bir örnek üret (0 = kapalı)
    # --- sistem ---
    seed: int = 1337
    dtype: str = "bfloat16"     # Ada (RTX 40xx) bf16 destekler -> GradScaler gerekmez
    compile: bool = False       # Windows'ta torch.compile için triton-windows gerekir; varsayılan kapalı
    resume: bool = False
    # false: best.pt / last.pt yazılmaz (kısa doğrulama ve ölçüm koşuları için disk + zaman; resume mümkün olmaz)
    save_checkpoints: bool = True
    # Windows'ta NVIDIA sürücüsü VRAM taşınca OOM vermek yerine sistem RAM'ine taşar ("sysmem
    # fallback") ve eğitim sessizce ~3-20x yavaşlar. PyTorch'un ayırıcısını VRAM'in bu oranıyla
    # sınırlayarak *kendi* büyümemizi açık bir OutOfMemoryError'a çeviriyoruz (0 = sınır yok).
    # En iyi çaba: başka uygulamalar (tarayıcı, oyun) VRAM'i doldurursa taşma bu sınırın altında da
    # olabilir. Kesin çözüm (sistem ayarı, elle yapılır): NVIDIA Denetim Masası -> 3D ayarları ->
    # "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback".
    vram_limit_frac: float = 0.92


# Hazır ayarlar. "tiny" sadece kodun uçtan uca çalıştığını saniyeler içinde doğrulamak için.
PRESETS: dict[str, tuple[ModelConfig, TrainConfig]] = {
    "tiny": (
        ModelConfig(max_seq_len=128, d_model=128, n_layer=2, n_head=4, ffn_hidden=352),
        TrainConfig(run_name="tiny", batch_size=16, grad_accum=1, max_steps=60, warmup_steps=10,
                    eval_interval=30, eval_iters=5, log_interval=10, sample_tokens=40),
    ),
    # ~30M parametre (4.2M embedding + 8 katman x ~3.2M). TinyStories için "tatlı nokta":
    # 12 GB'a rahat sığar ve ~20-30 dakikada tutarlı hikâye üretir.
    "dense": (ModelConfig(), TrainConfig()),
    # Kimi K2 tarzı MoE, dense ile AYNI aktif hesapla (adil karşılaştırma):
    #   katman 0: dense SwiGLU(1408)
    #   katman 1-7: 16 routed expert (352) içinden top-3 + 1 shared expert (352)
    #   aktif FFN = 3*352 + 352 = 1408 = dense ;  toplam FFN parametresi katman başına ~4.2x
    "moe": (
        ModelConfig(n_experts=16, top_k=3, expert_hidden=352, n_shared_experts=1, first_dense_layers=1),
        TrainConfig(run_name="moe"),
    ),
    # Ablasyon: aynı MoE ama load-balancing loss KAPALI -> expert collapse'ı gözlemlemek için.
    # (z-loss açık kalır: o denge sağlamaz, sadece logit'leri küçük tutar; böylece tek fark aux loss.)
    "moe-noaux": (
        ModelConfig(n_experts=16, top_k=3, expert_hidden=352, n_shared_experts=1, first_dense_layers=1,
                    aux_loss_coef=0.0),
        TrainConfig(run_name="moe-noaux"),
    ),
    # Aux loss YOK, dengeleme DeepSeek-V3 / Kimi K2 tarzı: sigmoid gating + seçim bias'ı (router ablasyonu, v3).
    # z-loss diğer MoE presetleriyle aynı kalır ki tek değişken dengeleme yöntemi olsun.
    "moe-auxfree": (
        ModelConfig(n_experts=16, top_k=3, expert_hidden=352, n_shared_experts=1, first_dense_layers=1,
                    aux_loss_coef=0.0, router_gate="sigmoid", bias_update_rate=0.001),
        TrainConfig(run_name="moe-auxfree"),
    ),
    # --- FOTON v0.2 kampanyasi baseline'lari (PREREGISTRATION_V02.md'de pinli) ---
    # vocab_size train.py'de tokenizer'dan okunuyor (32768). Adim basina token 32.768 SABIT
    # (16 x 4 x 512); mikro-batch/grad_accum bolusumu serbest ama batch_size x eval_iters = 1280.
    # max_steps 4375 secildi cunku %80 noktasi TAM adim 3500'e (14. eval) denk gelir -> L* icin
    # interpolasyon kurali gerekmez (v01'de bu kural 0.026 nat oynatiyordu).
    "foton-d": (
        ModelConfig(),
        TrainConfig(run_name="foton-d", data_dir="data/foton", batch_size=16, grad_accum=4,
                    max_steps=4375, eval_interval=250, eval_iters=80, sample_tokens=0,
                    loss_prefix_skip=64),
    ),
    "foton-m": (
        ModelConfig(n_experts=16, top_k=3, expert_hidden=352, n_shared_experts=1,
                    first_dense_layers=1, aux_loss_coef=0.0, router_gate="sigmoid",
                    bias_update_rate=0.001),
        TrainConfig(run_name="foton-m", data_dir="data/foton", batch_size=16, grad_accum=4,
                    max_steps=4375, eval_interval=250, eval_iters=80, sample_tokens=0,
                    loss_prefix_skip=64),
    ),
    "tiny-moe": (
        ModelConfig(max_seq_len=128, d_model=128, n_layer=2, n_head=4, ffn_hidden=352,
                    n_experts=8, top_k=2, expert_hidden=88, n_shared_experts=1, first_dense_layers=1),
        TrainConfig(run_name="tiny-moe", batch_size=16, grad_accum=1, max_steps=60, warmup_steps=10,
                    eval_interval=30, eval_iters=5, log_interval=10, sample_tokens=40),
    ),
}


def _parse_bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "y"):
        return True
    if s.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"bool bekleniyordu: {s}")


def build_configs(argv: list[str] | None = None) -> tuple[ModelConfig, TrainConfig]:
    """--preset seçer, sonra dataclass alanlarının hepsini --alan_adi olarak ezilebilir yapar."""
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="dense", choices=sorted(PRESETS))
    types = {int: int, float: float, str: str, bool: _parse_bool}
    for cls in (ModelConfig, TrainConfig):
        for f in fields(cls):
            ftype = eval(f.type) if isinstance(f.type, str) else f.type  # "int" -> int
            p.add_argument(f"--{f.name}", type=types[ftype], default=None)
    args = p.parse_args(argv)

    mcfg, tcfg = PRESETS[args.preset]
    over = {k: v for k, v in vars(args).items() if v is not None and k != "preset"}
    mcfg = replace(mcfg, **{k: v for k, v in over.items() if k in {f.name for f in fields(ModelConfig)}})
    tcfg = replace(tcfg, **{k: v for k, v in over.items() if k in {f.name for f in fields(TrainConfig)}})
    return mcfg, tcfg
