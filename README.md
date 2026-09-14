# moe-llm

Sıfırdan, PyTorch ile yazılmış küçük bir **Mixture-of-Experts (MoE)** dil modeli.
Amaç: öğrenmek ve tek bir tüketici GPU'sunda (RTX 4070 Super, 12 GB VRAM) eğitilebilecek
ölçekte, Kimi K2 / DeepSeek-V3 tarzı MoE fikirlerini denemek.

## Yol haritası

1. ✅ **Dense baseline** — decoder-only Transformer (nanoGPT iskeleti + RoPE, RMSNorm, SwiGLU),
   kendi BPE tokenizer'ımız, eğitim döngüsü, checkpoint/resume, örnek üretim
2. ✅ **MoE** — FFN yerine router (top-k) + N expert + shared expert + load-balancing (auxiliary)
   loss + z-loss, expert kullanım logları, aux loss ablasyonu ve uzmanlaşma analizi

## Klasör yapısı

```
src/
  config.py      tüm hiperparametreler + preset'ler (tiny, dense, moe, moe-noaux, tiny-moe)
  tokenizer.py   SentencePiece BPE (varsayılan) / tiktoken GPT-2
  model.py       Transformer: RMSNorm, RoPE, attention, SwiGLU; Block'ta dense FFN veya MoE
  moe.py         MoE: router (top-k), expert dispatch/combine, shared expert, aux loss, z-loss, denge metrikleri
  data.py        memmap üzerinden rastgele pencere örnekleyen veri yükleyici
  train.py       eğitim döngüsü (bf16, grad accumulation, cosine LR, eval, checkpoint, expert logları)
  checkpoint.py  atomik kaydetme, checkpoint'ten model yükleme
  sample.py      metin üretimi
scripts/
  download_tinystories.py   veri setini indir (kaldığı yerden devam eder)
  prepare_data.py           tokenizer eğit + train.bin / val.bin üret
  benchmark.py              batch boyutuna göre VRAM ve throughput ölçümü
  plot_metrics.py           koşuları karşılaştıran grafikler (loss, yük dengesi, ısı haritası)
  analyze_experts.py        eğitilmiş MoE'de expert yükü ve hangi tokenlarda uzmanlaştıkları
tests/           CPU'da çalışan doğruluk testleri (nedensellik, RoPE, init, overfit, MoE dispatch ...)
docs/            README'deki grafikler
experiments/     donmuş deney kayıtları: config, metrics.jsonl, train.log, ortam, checksum'lar
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

.\.venv\Scripts\python -m src.train --preset moe                     # 79M toplam / 30M aktif MoE, ~23 dk
.\.venv\Scripts\python -m src.train --preset moe-noaux               # ablasyon: load-balancing loss kapalı

# 4) Metin üretimi
.\.venv\Scripts\python -m src.sample --ckpt checkpoints\dense\best.pt --prompt "Once upon a time"

# 5) Analiz
.\.venv\Scripts\python scripts\plot_metrics.py --runs dense moe moe-noaux --out docs\dense_vs_moe.png
.\.venv\Scripts\python scripts\analyze_experts.py --ckpt checkpoints\moe\best.pt --layer 4
```

MoE koşularında konsol logu her log aralığında denge özetini, her eval'de de katman başına
expert yük çubuklarını basar (`▄` = ideal pay 1/E, `█` ≈ 1.9x+ aşırı yük, boşluk = ölü expert):

```
adım  3000 | loss 1.3214 | lr 1.00e-04 | grad_norm 0.25 | 81.7k tok/s | tepe bellek 8.92 GB
           ce 1.3111 | aux 1.006 | z 0.25 | yük CV 0.097 | maks 1.70x | min 0.79x | ölü expert 0
[expert] katman 4: |▄▄▄▄▄▄▄▄▄▄▄▄▄▃▄▄| min 0.84x maks 1.09x
```

## Deney kaydı: `v1-dense-moe-baseline` (2026-09-15)

> **Donmuş kayıt.** Bu bölüm referans noktamızdır; sonraki deneyler yeni bir bölüm olarak
> eklenmeli, buradaki sayılar değiştirilmemelidir. Kod + kayıt git tag'inde saklıdır:
> `git checkout v1-dense-moe-baseline`. Ham metrikler, config'ler ve loglar
> [`experiments/v1-dense-moe-baseline/`](experiments/v1-dense-moe-baseline/) altında.

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

### Adım 2: MoE (`--preset moe`) ve aux loss ablasyonu (`--preset moe-noaux`)

Mimari (Kimi K2 / DeepSeek-V3 fikirleri, ev ölçeğinde): katman 0 dense; katman 1-7'de FFN yerine
**16 ince taneli routed expert (SwiGLU 352) içinden top-3 + 1 shared expert**. Token başına aktif
FFN = 4 x 352 = 1408 = dense ile aynı, yani **aynı aktif hesap**: dense 29.89M, MoE 29.95M aktif
parametre (fark router'lar), ama MoE'nin toplamı **79.2M**. Load-balancing aux loss (0.01) +
router z-loss (0.001), dropless routing, router float32'de. Üç koşu da aynı 3000 adım x 32 768 token.

| | dense | **moe** (aux açık) | moe-noaux (aux kapalı) |
|---|---|---|---|
| toplam / aktif parametre | 29.9M / 29.9M | 79.2M / 30.0M | 79.2M / 30.0M |
| val loss @1000 / @2000 | 1.653 / 1.442 | **1.622 / 1.406** | 1.637 / 1.423 |
| **val loss @3000** | 1.354 (ppl 3.87) | **1.317 (ppl 3.73)** | 1.337 (ppl 3.81) |
| toplam süre (eval dahil) / throughput (medyan) | 15.5 dk / 115k tok/s | 23.3 dk / 79k tok/s | 22.1 dk / 81k tok/s |
| tepe VRAM (batch 32) | 6.3 GB | 8.9 GB | 8.9 GB |
| yük CV (0 = denge) | – | **0.097** | 1.206 |
| en çok / en az yüklü expert | – | 1.70x / 0.79x | 4.18x / 0.00x |
| ölü expert (< ideal payın %10'u) | – | **0 / 112** | **49 / 112** |
| aux loss değeri (1.0 = denge) | – | 1.006 | 2.246 |

![dense vs moe](docs/dense_vs_moe.png)

Çıkarımlar:
- **Aynı aktif hesapla MoE daha iyi**: val loss 1.354 → 1.317. Ekstra kapasite (4.25x FFN
  parametresi) token başına hesap artmadan kaliteye dönüşüyor — MoE'nin temel vaadi.
- **Aux loss olmadan expert collapse gerçekten oluyor ve çok erken**: 100. adımda 30, sonunda 49
  expert (%44) ölü; aşırı yüklü expert'ler ideal payın 4 katını alıyor. Bedeli: val loss
  1.317 → 1.337; MoE'nin dense'e karşı kazancının (0.037) yarısından fazlası kayboluyor.
  Çöküş ilk birkaç yüz adımda kilitleniyor ve sonra geri dönmüyor (eğitim log aralığı başına,
  7 MoE katmanı x 16 = 112 expert üzerinden):

  | adım | 100 | 500 | 1000 | 2000 | 3000 |
  |---|---|---|---|---|---|
  | moe — ölü expert / yük CV | 0 / 0.244 | 0 / 0.187 | 0 / 0.150 | 0 / 0.120 | 0 / 0.097 |
  | moe-noaux — ölü expert / yük CV | 30 / 1.021 | 45 / 1.134 | 48 / 1.187 | 50 / 1.205 | 49 / 1.206 |
  | moe-noaux — aux loss değeri | 1.642 | 2.068 | 2.215 | 2.248 | 2.246 |

  Aux loss'lu modelde dengesizlik eğitim boyunca azalmaya devam ediyor; aux'suz modelde ise
  aux değeri (katsayı 0 olsa da ölçülüyor) ~2.25'e tırmanıp orada kalıyor (1.0 = denge,
  E/k = 5.33 = tam çöküş).
- **Uzmanlaşma** (`analyze_experts.py`, katman 4, lift = P(expert|token)/P(expert)):
  aux'lu modelde expert 5 özne/isimler (`▁She ▁He ▁Tim ▁girl ▁Lily`), expert 7 noktalama ve cümle
  sonları (`. ." ! ?" <0x0A>`), expert 13 hikâye açılışları (`▁Once ▁upon </s>`) — lift'ler 4-6.
  Aux'suz modelde aynı katmanda 5 expert hiç çalışmıyor; 2.5-3.9x yüklü "genelci" expert'lerin
  lift'i yalnızca ~1.3-2 (belirgin bir uzmanlık yok).
- **Dikkat — duvar saati**: MoE 1.46x daha yavaş (79k vs 115k tok/s). Bu naif tek-GPU
  uygulamasında (expert başına Python döngüsü, sıralama/gather/scatter) FLOPs eşitliği hıza
  yansımıyor; aynı *süre* verilse dense ~%50 daha fazla token görebilirdi. Adil bir "aynı süre"
  karşılaştırması ve grouped GEMM / `torch.compile` ile hızlandırma sonraki deneyler.

Örnek (MoE, aynı prompt ve seed):

> Once upon a time, there was a little dragon who loved to bake. Every day, he would go out and
> bake delicious cookies. One day, the dragon was walking around when he came across a big room
> with a big, round cake on the front. The dragon was so excited. "Wow!" he said. Suddenly, a
> voice called out from behind him. It was his friend, the butterfly. "What are you doing?"
> "I'm baking a strawberry pie," said the dragon. [...]

### Ortam ve tekrar üretim

- **Donanım/yazılım**: RTX 4070 SUPER 12 GB (sürücü 616.64), Windows 11, Python 3.13.14,
  torch 2.14.0+cu130. Tam `pip freeze`: `experiments/v1-dense-moe-baseline/environment.txt`
- **Veri**: TinyStoriesV2-GPT4, SentencePiece BPE V=8192 (`data/sp8k`), train 545 813 247 token,
  val 5 512 219 token. Her koşu yalnızca 98M token gördü (tek epoch'un ~%18'i).
- **Ortak eğitim ayarları**: seq 512, mikro-batch 32 x grad_accum 2, AdamW (β 0.9/0.95, wd 0.1),
  lr 1e-3 → 1e-4 cosine, 200 adım warmup, grad clip 1.0, bf16 autocast, seed 1337. Val loss:
  sabit seed'li 40 x 32 pencere. Koşu başına tam config: `experiments/.../<koşu>/config.json`
- **Her koşunun eğitildiği kod durumu** (koşular commit'ler arasında başlatıldı):

  | koşu | kod | tag'e göre fark |
  |---|---|---|
  | dense | `f0f8494` | inceleme düzeltmeleri yok: veri penceresi üst sınırı 1 token dar (farklı veri sırası) |
  | moe | `f0f8494` + commit'lenmemiş MoE kodu | yukarıdakine ek olarak MoE inceleme düzeltmeleri yok |
  | moe-noaux | `2b13e7d` + commit'lenmemiş MoE kodu | MoE inceleme düzeltmeleri yok |

  MoE inceleme düzeltmeleri bu konfigürasyonun (top-3, float32 ağırlık) matematiğini değiştirmiyor
  (router `F.linear`'a geçti, `bincount` tekrar kullanılıyor, top-1 kuralı, yorumlar). `dense` ve
  `moe` birebir aynı veri sırasını gördü; `moe-noaux`'unkiler farklı (gürültü düzeyinde fark).
- **Tekrar üretmek**: `git checkout v1-dense-moe-baseline`, ardından Kullanım bölümündeki veri
  hazırlama + `--preset dense|moe|moe-noaux` komutları. GPU çekirdekleri deterministik olmadığından
  birebir aynı sayılar değil, aynı eğilimler beklenmeli.
- **Grafiği arşivden yeniden çizmek** (checkpoint gerekmez; `docs/dense_vs_moe.png` ile bayt bayt aynı):
  `python scripts/plot_metrics.py --runs dense moe moe-noaux --ckpt_dir experiments/v1-dense-moe-baseline`
- **Model ağırlıkları git'te değil** (`checkpoints/<koşu>/best.pt`, `last.pt`; toplam ~2.9 GB).
  Yedeklerin doğrulanması için SHA-256 değerleri: `experiments/v1-dense-moe-baseline/checksums.txt`

## Sonraki adımlar

- **Aux-loss-free dengeleme** (DeepSeek-V3 / Kimi K2): aux loss yerine her expert'e yalnızca
  seçimde kullanılan bir bias ekleyip yüküne göre ayarlamak; + softmax yerine sigmoid gating
- **Hız**: expert döngüsü yerine grouped matmul (`torch._grouped_mm`) veya `torch.compile`
  (Windows'ta triton-windows gerekir); aynı süre bütçesinde dense vs MoE karşılaştırması
- **MLA** (Multi-head Latent Attention, K2/DeepSeek) ve KV-cache; daha uzun bağlam
- **Ölçek**: tüm TinyStories (546M token), daha büyük/çok dilli veri ve tokenizer, daha çok expert

## 12 GB VRAM notları

`scripts/benchmark.py` ile ölçüldü (dense 30M, seq 512):

| mikro-batch | tepe VRAM | throughput |
|---|---|---|
| 16 | 3.3 GB | 128k tok/s |
| 32 | 6.2 GB | 116k tok/s |
| 48 | 9.0 GB | 110k tok/s |
| 64 | 11.9 GB | **45k tok/s** |

MoE (79M toplam, 30M aktif): batch 16 → 4.8 GB, batch 32 → 8.3 GB / 70k tok/s, batch 48 → OOM.
Ağırlık + gradyan + AdamW durumu 79M x 16 B ≈ 1.3 GB (dense: 0.5 GB); kalan farkın büyük kısmı
büyük olasılıkla dispatch sırasında kopyalanan aktivasyonlardan (top-3'te her token 3 kez) geliyor.
Dense ile aynı batch'i (32) kullanabildik.

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

MoE kodu da aynı yöntemle (3 boyut: routing/dispatch, aux loss/metrikler, tasarım/yorum/test
yeterliliği) incelendi. Sonucu bozan hata yine çıkmadı. Doğrulanan düzeltmeler: aux loss katman
ortalamasının router başına etkin katsayıyı coef/n_moe yaptığının doğru açıklanması, top-1'de
normalize edilen gate'in router'ın görev gradyanını kesmesi (k=1'de artık normalize edilmiyor),
tekrarlanan `bincount` senkronu, model bf16'ya çevrilince router çökmesi, sparkline/entropi
açıklamaları, grafik katman etiketleri; aux gradyanının router'a ulaştığını ve top-1 router
gradyanını sınayan testler (her ikisi de mutasyon testiyle doğrulandı).
