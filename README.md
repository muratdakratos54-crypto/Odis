# moe-llm

Sıfırdan, PyTorch ile yazılmış küçük bir **Mixture-of-Experts (MoE)** dil modeli.
Amaç: öğrenmek ve tek bir tüketici GPU'sunda (RTX 4070 Super, 12 GB VRAM) eğitilebilecek
ölçekte, Kimi K2 / DeepSeek-V3 tarzı MoE fikirlerini denemek.

## Yol haritası

1. **Dense baseline** — decoder-only Transformer (nanoGPT iskeleti + RoPE, RMSNorm, SwiGLU),
   kendi BPE tokenizer'ımız, eğitim döngüsü, checkpoint/resume, örnek üretim
2. **MoE** — FFN yerine router (top-k) + N expert + load-balancing (auxiliary) loss, expert kullanım logları

## Klasör yapısı

```
src/
  config.py      tüm hiperparametreler + preset'ler (tiny, dense, ...)
  tokenizer.py   SentencePiece BPE (varsayılan) / tiktoken GPT-2
  model.py       Transformer: RMSNorm, RoPE, attention, SwiGLU, GPT
  data.py        memmap üzerinden rastgele pencere örnekleyen veri yükleyici
  train.py       eğitim döngüsü (bf16, grad accumulation, cosine LR, eval, checkpoint)
  checkpoint.py  atomik kaydetme, checkpoint'ten model yükleme
  sample.py      metin üretimi
scripts/
  download_tinystories.py   veri setini indir (kaldığı yerden devam eder)
  prepare_data.py           tokenizer eğit + train.bin / val.bin üret
  benchmark.py              batch boyutuna göre VRAM ve throughput ölçümü
tests/           CPU'da çalışan doğruluk testleri (nedensellik, RoPE, init, overfit ...)
data/            ham metin + token dosyaları (git'e girmez)
checkpoints/     eğitim çıktıları: last.pt, best.pt, metrics.jsonl, train.log (git'e girmez)
```

## Kurulum (Windows / PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

> Not: PyPI'daki `torch` Windows'ta CPU-only'dir. `requirements.txt` CUDA build'ini
> (`torch==2.14.0+cu130`) PyTorch index'inden sabitler. Kontrol:
> `.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"`

## Kullanım

```powershell
# 1) Veri: ~2.2 GB indir, 8k'lık BPE tokenizer eğit, ~546M token'a çevir (~3 dk)
.\.venv\Scripts\python scripts\download_tinystories.py
.\.venv\Scripts\python scripts\prepare_data.py

# 2) Testler (CPU, birkaç saniye)
.\.venv\Scripts\python -m pytest -q tests

# 3) Eğitim
.\.venv\Scripts\python -m src.train --preset tiny --run_name smoke   # saniyeler: uçtan uca kontrol
.\.venv\Scripts\python -m src.train --preset dense                   # ~30M param, ~15-20 dk
.\.venv\Scripts\python -m src.train --preset dense --resume true     # yarıda kaldıysa devam
.\.venv\Scripts\python -m src.train --preset dense --run_name lr5e4 --lr 5e-4   # her alan ezilebilir

# 4) Metin üretimi
.\.venv\Scripts\python -m src.sample --ckpt checkpoints\dense\best.pt --prompt "Once upon a time"
```

## Sonuçlar

### Adım 1: Dense baseline (`--preset dense`)

29.9M parametre (8 katman, d=512, 8 head, SwiGLU 1408, V=8192), 3000 adım x 32 768 token
= **98M token** (TinyStories'in ~%18'i), RTX 4070 Super'da **15.5 dakika**, 115k tok/s, tepe VRAM 6.3 GB.

| adım | 0 | 500 | 1000 | 2000 | 3000 |
|---|---|---|---|---|---|
| val loss | 9.12 | 1.90 | 1.65 | 1.44 | **1.354** (ppl 3.87) |

Örnek (`--prompt "Once upon a time, there was a little dragon who"`, temperature 0.8, top-k 50):

> Once upon a time, there was a little dragon who lived in a big cave. This dragon was not a bad
> dragon. He was an original dragon, not like any other dragon. The dragon felt sad and wanted to
> find a new home. One day, the dragon met a little girl. The girl said, "Hello, dragon! I can help
> you find a new home." [...] From that day on, they became the best of friends.

## 12 GB VRAM notları

`scripts/benchmark.py` ile ölçüldü (dense 30M, seq 512):

| mikro-batch | tepe VRAM | throughput |
|---|---|---|
| 16 | 3.3 GB | 128k tok/s |
| 32 | 6.2 GB | 116k tok/s |
| 48 | 9.0 GB | 110k tok/s |
| 64 | 11.9 GB | **45k tok/s** |

Windows'ta VRAM dolduğunda sürücü OOM hatası vermek yerine belleği sistem RAM'ine taşır
("sysmem fallback") ve eğitim sessizce yavaşlar. `train.py` bu yüzden PyTorch'u VRAM'in
%92'siyle sınırlar (`--vram_limit_frac`): kendi belleğimiz taşarsa açık bir OOM hatası alırsınız.
Bu en iyi çaba bir korumadır (başka uygulamalar VRAM'i doldurursa yine taşabilir); kesin çözüm
NVIDIA Denetim Masası → 3D ayarları → "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback".

## Kod incelemesi

Dense kod, eğitimden önce çok ajanlı bir incelemeden geçti (4 boyut: model matematiği, eğitim
döngüsü, veri/tokenizer, platform tuzakları + açıklama yorumlarının doğruluğu; her bulgu ayrı bir
şüpheci tarafından çürütülmeye çalışıldı). Kritik (sonucu bozan) hata çıkmadı; doğrulanan
düzeltmeler: weight tying ve GradScaler açıklamalarının düzeltilmesi, RoPE'yi gerçekten sınayan
testler, veri yükleyicide 1 tokenlık off-by-one, resume uç durumları (optimizer ayarları, metrics
tekrarları, max_steps), fp16'nın reddedilmesi, fsync'li checkpoint, indirme betiğinde kısa okuma.
