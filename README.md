# moe-llm

Sıfırdan, PyTorch ile yazılmış küçük bir **Mixture-of-Experts (MoE)** dil modeli.
Amaç: öğrenmek ve tek bir tüketici GPU'sunda (RTX 4070 Super, 12 GB VRAM) eğitilebilecek
ölçekte, Kimi K2 / DeepSeek-V3 tarzı MoE fikirlerini denemek.

## Yol haritası

1. Sade decoder-only Transformer (nanoGPT tarzı) — tokenizer, model, eğitim, checkpoint, örnek üretim
2. FFN yerine MoE katmanı — router (top-k), N expert, load-balancing (auxiliary) loss, expert kullanım logları

## Klasör yapısı

```
src/          model, tokenizer, veri ve eğitim kodu
scripts/      veri indirme/hazırlama ve yardımcı betikler
data/         ham metin + token dosyaları (git'e girmez)
checkpoints/  eğitim çıktıları (git'e girmez)
```

## Kurulum (Windows / PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```
