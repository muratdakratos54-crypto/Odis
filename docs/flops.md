# Analitik aktif FLOP hesabı: türetim

Bu belge `src/flops.py`'deki formüllerin elle türetimidir. Karşılaştırmalarda **birincil** hesap ölçüsü budur.
Duvar saati ve tok/s **ikincil** ölçüdür. Profiler FLOP'u yalnızca **çapraz kontroldür** (bölüm 7).

Semboller (presetlerdeki değerlerle):

| sembol | anlam | dense | moe / moe-noaux |
|---|---|---|---|
| d | residual genişliği (`d_model`) | 512 | 512 |
| L | katman sayısı | 8 | 8 (katman 0 dense, 1-7 MoE) |
| n_h, d_h | head sayısı, head boyutu (d = n_h d_h) | 8, 64 | 8, 64 |
| T | eğitim dizisi uzunluğu | 512 | 512 |
| V | sözlük | 8192 | 8192 |
| h | dense SwiGLU gizli boyutu | 1408 | 1408 (katman 0) |
| E, k | routed expert sayısı, token başına seçilen | – | 16, 3 |
| h_e | expert gizli boyutu (routed ve shared) | – | 352 |
| s | shared expert sayısı | – | 1 |

## 1. Birim

Bir lineer katman y = W x, W ∈ R^(m×n). Her çıktı y_i = Σ_j W_ij x_j için n çarpma ve n−1 toplama gerekir.
Bunu **çarpma+toplama çifti başına 2 FLOP** sayıyoruz: katman token başına **2·m·n** FLOP.
Kaplan ve ark. (2020, Tablo 1) ile Hoffmann ve ark. (2022, Ek F) aynı birimi kullanır.
Bias yok (modelde hiç bias yok).

**Aktif FLOP**, bir tokenın gerçekten geçtiği hesaptır. MoE katmanında E expert'in yalnızca k tanesi çalışır.
Seçilmeyen E−k expert'in parametreleri o token için hesaba girmez.

## 2. İleri yön, bir blok, token başına

**(a) Attention projeksiyonları.** q, k, v tek bir d → 3d matrisiyle, çıkış d → d:

    2·(3d·d) + 2·(d·d) = 8d²  = 2,097,152

**(b) Attention bağlam terimi (causal).** 0'dan saydığımız t. konumdaki token t+1 konuma bakar.

- Skorlar: her head'de q_t·k_j, j ≤ t. Uzunluğu d_h olan t+1 iç çarpım yapar: 2·d_h·(t+1). n_h head'de toplam **2·d·(t+1)**.
- Değerlerin ağırlıklı toplamı o_t = Σ_j a_tj v_j: her head'de t+1 terim, d_h boyut: 2·d_h·(t+1). Toplam **2·d·(t+1)**.
- t konumunda toplam: 4·d·(t+1). Dizideki konumların ortalaması:

      (1/T) Σ_{t=0}^{T-1} 4d(t+1) = 4d·(T+1)/2 = 2d(T+1) = 525,312

- 1/√d_h ölçekleme, softmax ve RoPE eleman bazlıdır. Birincil tanıma girmez (bölüm 5).
- Kaplan'ın bağlam terimi 2·n_layer·n_ctx·d_attn'dir. Bizim katman toplamımız 2·L·d·(T+1) ile sayısal olarak neredeyse aynı.
  Fark şurada: biz causal maskeyi (ortalama (T+1)/2 konum) ve değer toplamını açıkça sayıyoruz.

**(c) Dense FFN (SwiGLU).** w_gate ve w_up d → h, w_down h → d:

    2·(d·h) + 2·(d·h) + 2·(h·d) = 6dh  = 4,325,376

**(d) MoE FFN.**

    router  (d → E):                         2dE          =    16,384
    routed  (k expert, her biri SwiGLU h_e):  k · 6d·h_e   = 3,244,032
    shared  (s expert):                      s · 6d·h_e   = 1,081,344

Tasarım gereği (k+s)·h_e = 4·352 = 1408 = h. Bu yüzden expert terimleri toplamı dense FFN'in **aynısı**dır (4,325,376).
MoE'nin tek ek matris çarpımı router'dır.

**(e) Çıkış.** lm_head d → V: **2dV = 8,388,608**. Embedding tied olsa da çıkıştaki çarpım tam bir matris çarpımıdır.
Girişteki embedding lookup bir indekslemedir: 0 FLOP.

## 3. İleri toplam

    dense = L·(8d² + 2d(T+1) + 6dh) + 2dV
          = 8·(2,097,152 + 525,312 + 4,325,376) + 8,388,608           = 63,971,328
    moe   = dense + (L−1)·2dE = 63,971,328 + 7·16,384                 = 64,086,016
    oran  = 1.001793

| terim | dense | moe |
|---|---:|---:|
| attention projeksiyonları | 16,777,216 | 16,777,216 |
| attention bağlamı (causal) | 4,202,496 | 4,202,496 |
| dense FFN | 34,603,008 | 4,325,376 |
| router | 0 | 114,688 |
| routed expert'ler | 0 | 22,708,224 |
| shared expert | 0 | 7,569,408 |
| lm_head | 8,388,608 | 8,388,608 |
| **ileri / token** | **63,971,328** | **64,086,016** |

## 4. Geri yön ve eğitim

y = W x için geri yayılım iki matris çarpımıdır:

- ∂L/∂x = Wᵀg: 2mn FLOP
- ∂L/∂W = g xᵀ: 2mn FLOP

Toplam, ileri yönün 2 katıdır.

Attention bağlam teriminde de durum aynı:
- skor çarpımından q ve k gradyanları: 2 × 2d(t+1)
- ağırlıklı toplamdan a ve v gradyanları: 2 × 2d(t+1)

Katman 0 da girdisinin gradyanını hesaplar, çünkü embedding'e gradyan gerekir. Router float32'de çalışır ama FLOP sayısı dtype'tan bağımsızdır.

    eğitim / token = 3 × ileri:   dense 191,913,984   moe 192,258,048

3000 adım × 32,768 token = 98.304M token için:
- dense 1.8866e16 FLOP
- moe 1.8900e16 FLOP

**Eşit FLOP bütçesi:** dense'in moe'nin 3000 adımına eşit FLOP'u görmesi için 98.480M token = **3005.38 adım** gerekir.
Tam adıma yuvarlanır: 3005 adım. Kalan fark −0.013% FLOP.

## 5. Birincil tanıma girmeyenler

| ne | neden dışarıda | büyüklük |
|---|---|---|
| Eleman bazlı işlemler (RMSNorm, RoPE, SiLU×up, softmax, residual, gate × expert çıktısı, CE) | Tanım matris çarpımı + bağlam; bunlar tanıma bağlı yaklaşık sayımlar | `secondary_forward_flops_per_token`: ileri toplamın dense'te %0.32'si, moe'de %0.36'sı. Dahil edilince moe/dense oranı 1.00179 → 1.00219, eşit FLOP bütçesi 3006.6 adım (1.2 adım fark) |
| **Dispatch**: argsort, bincount, gather/scatter kopyaları, dtype çevirme | Aktivasyonlar üzerinde aritmetik yok: tamsayı ve bellek işlemleri | **0 FLOP**. "loop" ile "grouped" yolların FLOP'u bu yüzden tanım gereği aynıdır. Aralarındaki fark yalnızca duvar saatinde, ve duvar saatinin önemli bir kısmı tam da buradadır (README "Deney kaydı: v2-grouped-dispatch") |
| AdamW + gradyan kırpma | Token başına değil, adım başına ve **toplam** parametreyle ölçeklenir | moe'de ~79M × ~12 işlem ≈ 1e9 / adım. Adımın eğitim FLOP'u 6.3e12, oran ~%0.015 |
| Eval | Bütçe eğitim hesabıdır | Eval ayrıca yapılır, zamanlamaya da girmez |
| Bellek-verimli attention'ın geri yönde yeniden hesaplaması | Uygulama ayrıntısı, algoritmanın gerektirdiği hesap değil | Profiler bunu sayar (bölüm 7) |

## 6. Denetim: formül modelle nasıl bağlanıyor

Kontroller `tests/test_flops.py` içinde:
1. Formüldeki matris çarpımı terimlerinin toplamı = 2 × (modülden sayılan aktif lineer ağırlıklar). Beş preset için kontrol ediliyor.
   Aktif ağırlıklar: tüm attention Linear'ları + dense FFN ya da (router + k × bir expert + shared) + lm_head.
2. dense ve moe presetleri arasındaki analitik fark yalnızca router terimi (7 · 2 · 512 · 16).
3. Bağımsız sayaç (FlopCounterMode) ile formülün matris çarpımı terimleri birebir eşleşiyor.
   Küçük dense ve `moe_impl="loop"` modellerde ileri ve ileri+geri (3×) için CPU'da ölçülüyor.

## 7. Çapraz kontrol: profiler

`scripts/flops_report.py` gerçek presetlerde, CUDA'da, bf16 autocast ile bir mikro-batch ileri+geri çalıştırır.
Çıktı `experiments/v2-grouped-dispatch/flops_report.json`'a kaydedilir. Değerler token başınadır, ileri+geri:

| model | profiler toplam | analitik | matris çarpımı: profiler / analitik | attention çekirdekleri: profiler / analitik bağlam |
|---|---:|---:|---|---|
| dense | 208,666,624 | 191,913,984 | 179,306,496 / 179,306,496 | 29,360,128 / 12,607,488 |
| moe, `loop` | 209,010,688 | 192,258,048 | 179,650,560 / 179,650,560 | 29,360,128 / 12,607,488 |
| moe, `grouped` | **140,886,016** | 192,258,048 | **111,525,888** / 179,650,560 | 29,360,128 / 12,607,488 |

- **Matris çarpımları:** dense ve `loop`'ta profiler ile analitik sayım birebir aynı (fark 0).
- **`grouped`: profiler 68,124,672 FLOP/token eksik sayıyor.** Bu tam olarak routed expert'lerin eğitim FLOP'u (3 × 22,708,224).
  Sebebi: torch 2.14'ün sayacında `_grouped_mm` için formül yok, bu çekirdek 0 sayılıyor.
  Profiler'a dayansaydık `grouped`, aynı modelin `loop` yolundan **%32.6 daha ucuz** görünürdü (140.9M'e karşı 209.0M; yalnız
  matris çarpımlarında %37.9). Algoritmik FLOP değişmedi. (Profiler'ın `grouped` sayısı analitik değerin %26.7 altında;
  bu fark daha küçük, çünkü profiler attention'ı fazla sayıyor ve eksik saydığı expert terimini kısmen örtüyor.)
- **Attention:** profiler SDPA'yı causal maskeden bağımsız T×T üzerinden sayıyor (ileri: 4·d·T / katman, bizim 2·d·(T+1)'in ~2 katı).
  Geri yönü de 2.5× sayıyor (bellek-verimli çekirdeğin yeniden hesaplaması). Algoritmik değil, uygulamaya ait sayım.
