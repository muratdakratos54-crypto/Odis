"""Eğitim döngüsü.

Kullanım (proje kökünden):
    .venv\\Scripts\\python -m src.train --preset tiny            # birkaç saniyelik duman testi
    .venv\\Scripts\\python -m src.train --preset dense           # ~30M dense model
    .venv\\Scripts\\python -m src.train --preset dense --resume true

Bir optimizer adımının anatomisi:
    grad_accum kez: batch al -> bf16 autocast altında forward -> loss/grad_accum -> backward
    -> gradyan normunu 1.0'a kırp -> AdamW adımı -> LR'ı schedule'a göre güncelle
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, replace

import torch

from src.checkpoint import save_atomic
from src.config import TrainConfig, build_configs
from src.data import TokenDataset, dataset_paths
from src.model import GPT
from src.tokenizer import load_tokenizer


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Lineer warmup + cosine decay.

    Warmup: başlangıçta Adam'ın moment tahminleri henüz oturmamıştır ve ağırlıklar rastgeledir;
    büyük LR ile ilk adımlar modeli kararsız bir bölgeye fırlatabilir. Cosine decay: sona doğru
    LR'ı düşürmek, loss vadisinin dibine yerleşmeyi sağlar (sabit LR'a göre belirgin kazanç).
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def evaluate(model: GPT, ds: TokenDataset, cfg: TrainConfig, autocast_ctx) -> float:
    model.eval()
    # Her değerlendirmede *aynı* pencereleri kullan (sabit seed): böylece val loss'taki değişim
    # örnekleme gürültüsünden değil, modelin gerçekten iyileşmesinden gelir.
    g = torch.Generator().manual_seed(12345)
    total = 0.0
    for _ in range(cfg.eval_iters):
        x, y = ds.get_batch(cfg.batch_size, g)
        with autocast_ctx:
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / cfg.eval_iters


def main() -> None:
    # Windows konsolu/yönlendirmesi UTF-8 olmayabilir; üretilen metin yüzünden çökmeyelim
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    mcfg, tcfg = build_configs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tcfg.seed)
    # TF32: Ampere+ GPU'larda float32 matris çarpımlarını hafifçe hassasiyetten ödün vererek
    # hızlandırır (autocast dışında kalan işlemler için).
    torch.set_float32_matmul_precision("high")
    if device == "cuda" and tcfg.vram_limit_frac > 0:
        torch.cuda.set_per_process_memory_fraction(tcfg.vram_limit_frac)

    tok = load_tokenizer(tcfg.data_dir)
    mcfg = replace(mcfg, vocab_size=tok.vocab_size)
    train_path, val_path = dataset_paths(tcfg.data_dir)
    train_ds = TokenDataset(train_path, mcfg.max_seq_len, device)
    val_ds = TokenDataset(val_path, mcfg.max_seq_len, device)

    run_dir = os.path.join(tcfg.out_dir, tcfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    last_path, best_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "best.pt")
    if os.path.exists(last_path) and not tcfg.resume:
        sys.exit(f"{last_path} zaten var. Devam için --resume true, yeni deney için farklı --run_name kullan.")

    model = GPT(mcfg).to(device)
    optimizer = model.configure_optimizer(tcfg.lr, tcfg.weight_decay, (tcfg.beta1, tcfg.beta2), device)
    train_gen = torch.Generator().manual_seed(tcfg.seed)
    step, best_val = 0, float("inf")

    if tcfg.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # map_location=cuda RNG durumunu da GPU'ya taşır; CPU Generator'ı CPU ByteTensor ister
        train_gen.set_state(ckpt["train_gen_state"].cpu())
        step, best_val = ckpt["step"], ckpt["best_val"]
        print(f"devam ediliyor: adım {step}, en iyi val {best_val:.4f}")

    # bf16 autocast: matris çarpımları bf16'da (2x daha az aktivasyon belleği, Ada'da çok daha hızlı),
    # ağırlıklar ve optimizer durumu float32'de kalır ("mixed precision").
    # bf16'nın fp16'ya üstünlüğü: float32 ile aynı üs aralığı -> gradyan taşması yok -> GradScaler yok.
    ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[tcfg.dtype]
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=ptdtype)
        if device == "cuda" and ptdtype != torch.float32 else nullcontext()
    )
    train_model = torch.compile(model) if tcfg.compile else model

    tokens_per_step = tcfg.batch_size * tcfg.grad_accum * mcfg.max_seq_len
    print(f"model: {model.num_params() / 1e6:.2f}M parametre "
          f"({model.num_params(exclude_embedding=True) / 1e6:.2f}M embedding hariç)")
    print(f"veri: train {len(train_ds) / 1e6:.1f}M token, val {len(val_ds) / 1e6:.2f}M token")
    print(f"adım başına {tokens_per_step:,} token, toplam {tokens_per_step * tcfg.max_steps / 1e6:.0f}M token")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"model": asdict(mcfg), "train": asdict(tcfg)}, f, indent=2)
    # Yeni koşu: dosyayı sıfırdan aç (checkpoint'e ulaşamadan çökmüş eski bir koşunun kayıtları
    # karışmasın). Resume: kaldığı yerden ekle.
    metrics = open(os.path.join(run_dir, "metrics.jsonl"), "a" if tcfg.resume else "w", encoding="utf-8")

    def log(record: dict) -> None:
        metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
        metrics.flush()

    t_start = t_log = time.time()
    loss_accum, steps_since_log = torch.zeros((), device=device), 0
    skip_eval_once = tcfg.resume and step > 0  # bu adımın eval'i kaydedilmeden önce zaten yapılmıştı

    while True:
        # ---------------- değerlendirme + checkpoint ----------------
        if (step % tcfg.eval_interval == 0 or step == tcfg.max_steps) and not skip_eval_once:
            t_eval = time.time()
            val_loss = evaluate(train_model, val_ds, tcfg, autocast_ctx)
            print(f"[eval] adım {step}: val_loss {val_loss:.4f} (ppl {math.exp(val_loss):.2f})")
            log({"type": "val", "step": step, "val_loss": val_loss, "time": time.time() - t_start})
            if tcfg.sample_tokens > 0:
                model.eval()
                start = torch.tensor([[tok.bos_id]], device=device)
                with autocast_ctx:
                    out = model.generate(start, tcfg.sample_tokens, temperature=0.8, top_k=50, eos_id=tok.eos_id)
                model.train()
                text = tok.decode(out[0, 1:].tolist())
                print(f"[örnek] {text!r}")
                log({"type": "sample", "step": step, "text": text})
            state = {"model": model.state_dict(), "model_cfg": asdict(mcfg), "train_cfg": asdict(tcfg),
                     "step": step, "val_loss": val_loss}
            if val_loss < best_val:
                best_val = val_loss
                save_atomic(state, best_path)
            if step > 0:
                save_atomic({**state, "optimizer": optimizer.state_dict(), "best_val": best_val,
                             "train_gen_state": train_gen.get_state()}, last_path)
            t_log += time.time() - t_eval  # eval/örnek/kayıt süresi tok/s ölçümüne karışmasın
        skip_eval_once = False
        if step == tcfg.max_steps:
            break

        # ---------------- bir optimizer adımı ----------------
        lr = lr_at(step, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(tcfg.grad_accum):
            x, y = train_ds.get_batch(tcfg.batch_size, train_gen)
            with autocast_ctx:
                _, loss = train_model(x, y)
            # Gradyan biriktirme: loss'u grad_accum'a böl ki toplanan gradyan, büyük tek bir
            # batch'in *ortalama* gradyanına eşit olsun (LR'ı batch'ten bağımsız tutar).
            (loss / tcfg.grad_accum).backward()
            loss_accum += loss.detach() / tcfg.grad_accum
        # Gradyan kırpma: nadir "kötü batch"lerde oluşan dev gradyanların modeli bozmasını engeller.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # None: sıfır tensörü tutmaktan daha az bellek
        step += 1
        steps_since_log += 1

        # ---------------- log ----------------
        # .item() GPU'yu CPU ile senkronize eder (yavaş); bu yüzden her adımda değil, aralıklarla.
        if step % tcfg.log_interval == 0 or step == tcfg.max_steps:
            dt = time.time() - t_log
            train_loss = loss_accum.item() / steps_since_log
            tok_s = steps_since_log * tokens_per_step / dt
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            print(f"adım {step:5d} | loss {train_loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm.item():.2f} "
                  f"| {tok_s / 1e3:.1f}k tok/s | tepe bellek {mem:.2f} GB")
            log({"type": "train", "step": step, "loss": train_loss, "lr": lr, "grad_norm": grad_norm.item(),
                 "tok_per_sec": tok_s, "peak_mem_gb": mem, "time": time.time() - t_start})
            loss_accum.zero_()
            steps_since_log, t_log = 0, time.time()

    metrics.close()
    print(f"bitti: {(time.time() - t_start) / 60:.1f} dk, en iyi val_loss {best_val:.4f}")


if __name__ == "__main__":
    main()
