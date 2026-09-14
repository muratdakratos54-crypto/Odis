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
    # --- optimizer ---
    lr: float = 1e-3            # küçük modeller büyük LR'ı tolere eder (GPT-3 tablosu: ~6e-4 @125M)
    min_lr: float = 1e-4        # cosine decay sonu = lr/10 (Chinchilla tarzı)
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95         # 0.999 yerine 0.95: LLM eğitiminde loss spike'larına daha dayanıklı
    grad_clip: float = 1.0
    # --- değerlendirme / log ---
    eval_interval: int = 250
    eval_iters: int = 40
    log_interval: int = 25
    sample_tokens: int = 120    # her eval'de kısa bir örnek üret (0 = kapalı)
    # --- sistem ---
    seed: int = 1337
    dtype: str = "bfloat16"     # Ada (RTX 40xx) bf16 destekler -> GradScaler gerekmez
    compile: bool = False       # Windows'ta torch.compile için triton-windows gerekir; varsayılan kapalı
    resume: bool = False
    # Windows'ta NVIDIA sürücüsü VRAM taşınca OOM vermek yerine sistem RAM'ine taşar ("sysmem
    # fallback") ve eğitim sessizce ~3-20x yavaşlar. PyTorch'un ayırıcısını VRAM'in bu oranıyla
    # sınırlayarak bunu açık bir OutOfMemoryError'a çeviriyoruz (0 = sınır yok).
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
