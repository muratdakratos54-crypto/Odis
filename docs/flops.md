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

## 8. FOTON v0.2 (V=32768)

Dayanak: `PREREGISTRATION_V02.md` V3 ("Analitik hesap `src/flops.py` ... FOTON için genişletilecek ve genişletme
`docs/flops.md`'de elle türetilip karar koşularından önce commit'lenecek") ve v01 `PREREGISTRATION.md` EK1-B13
(sayılmayan her FOTON işlemi beyan edilir). Dört kol: D = `foton-d`, M = `foton-m`, H = `foton-h`, X = `foton-x`;
hepsinde d = 512, L = 8, n_h = 8, h = 1408, T = 512, V = 32768 (tokenizer'dan okunur).

**Bu bölümdeki tabloların HEPSİ koddan üretilir, elle yazılmış sayı yoktur:**

    .venv\Scripts\python -m src.flops

`tests/test_flops.py::test_docs_foton_tables_are_generated_by_the_code` bu çıktının her satırının aşağıda birebir
durduğunu sınar; kod değişip belge yeniden üretilmezse (ya da burada bir rakam elle değiştirilirse) test düşer.

Ek semboller:

| sembol | anlam | değer |
|---|---|---|
| pre + r + post | katman bölüşümü: paylaşılan ön bloklar + modül başına katman + paylaşılan arka bloklar | 3 + 2 + 3 = L |
| n_m | modül sayısı (router çıkışı) | 3 |
| k | router'ın gördüğü nedensel prefix uzunluğu | 64 |
| c | bir modülün token/head başına ORTALAMA nedensel bağlantı sayısı (T konumun hepsi, head ortalaması) | aşağıda, sayılarak |

### 8.1 Bir dizinin geçtiği katmanlar: 8, kurulan 12 değil

FOTON modeli 12 blok KURAR (6 paylaşılan + 3 modül × 2 katman), ama bir dizi bunların yalnızca 8'inden geçer:
3 paylaşılan ön blok → router'ın seçtiği TEK modülün 2 katmanı → 3 paylaşılan arka blok. Dizi almayan iki modül o
dizi için hiç çalışmaz (`GPT._foton_trunk`: `index_select` ile yalnızca seçilen diziler modüle girer). Aktif FLOP bu
yüzden tam olarak `cfg.n_layer` = 8 katmanı sayar:

- attention projeksiyonları: 8 · 8d², D ile aynı;
- dense SwiGLU: 8 · 6dh, D ile aynı (FOTON kollarında MoE yok: `n_experts = 0`, aksi config'te formül hata verir);
- lm_head: 2dV, D ile aynı.

Üç modülün parametre şekilleri birebir aynı olduğu için bu terimler router'ın seçiminden bağımsızdır.
Denetim: formülün matris çarpımı terimleri = 2 × (lm_head + paylaşılan bloklar + TEK modülün Linear ağırlıkları),
modelin kendisinden sayılarak (`test_foton_matmul_counts_traversed_layers_not_constructed_blocks`).

### 8.2 Yönlendirilen katmanların bağlam terimi: 4·d·c

Paylaşılan 6 blok tam nedenseldir, terimleri bölüm 2(b)'deki gibidir: katman başına 2d(T+1).

Yönlendirilen katmanda t konumundaki sorgu, h head'inde c_{t,h} anahtara bakar (maskenin t. satırındaki izinli
girdi sayısı). Bölüm 2(b)'nin aynısı, t+1 yerine c_{t,h} ile:

- skorlar: head başına 2·d_h·c_{t,h}; değerlerin ağırlıklı toplamı: head başına 2·d_h·c_{t,h};
- head'ler üzerinden toplam 4·d_h·Σ_h c_{t,h} = 4·d·(head ortalaması c_t); konumlar üzerinden ortalama:

      katman başına 4·d·c,      c = (maskedeki izinli girdi sayısı) / (n_h · T)

**c nereden gelir.** Formülden değil, SAYILARAK: `src/attn_patterns.PATTERNS[desen].masks(T, n_head)` — modelin
`attn_mask` buffer'ını kuran AYNI çağrı — ile [n_h, T, T] boolean maske kurulur, izinli girdiler sayılır,
n_h·T'ye bölünür (tam kesir). Kapalı form `avg()` ikinci yol olarak karşılaştırılır; ayrışırlarsa hesap durur.
T ve n_head her zaman açıkça verilir (attn_patterns'in varsayılanları S0 değerleridir; S1'de n_head = 10).
Ön-kayıt V3'ün uyarısı tam olarak budur: tam-nedensel 2d(T+1) terimini bu katmanlara uygulamak c yerine 256.5
sayar ve FOTON'un FLOP'unu fazla gösterir.

| kol | modül | sayılmış c (T=512, n_head=8) | kolda kullanılan c |
|---|---|---:|---:|
| H | `M0:A` | 128.173828 | 128.173828 |
| H | `M1:A` | 128.173828 | 128.173828 |
| H | `M2:A` | 128.173828 | 128.173828 |
| X | `M0:A` | 128.173828 | 128.500000 |
| X | `M1:B` | 128.500000 | 128.500000 |
| X | `M2:C` | 128.375000 | 128.500000 |

**Hangi c, ve neden.** Kol başına TEK sabit: kolun modülleri üzerinden MAKSİMUM.

- H: üç modül de A desenidir, maksimum A'nın kendisidir.
- X: maksimum B'dir; bu, ön-kaydın "ortak hedef 128.50"sidir.

Gerekçe: birincil istatistiğin paydası (`F = N* × train_flops_per_token`) router'ın kontrolünde OLAMAZ (ön-kayıt V3,
"FOTON'a özgü uyarı"). FLOP/token gerçekleşen seçim karışımıyla hesaplansaydı, router en ucuz modüle (A) çökerek
ölçülen tasarrufu büyütebilirdi. Tek sabitle FLOP/token yalnızca config'e bağlıdır; `forward_flops_per_token` ve
`train_flops_per_token` bilerek karışım parametresi ALMAZ. Maksimum seçimi FOTON'un ALEYHİNE muhafazakârdır:
hiçbir karışımın gerçek bağlam FLOP'u bu sabitle hesaplananı aşamaz. Muhafazakârlığın bedeli aşağıdaki
duyarlılık tablosundadır (`foton_flops_realized`; YALNIZCA rapor satırı, birincil istatistiğe girmez):

| X kolu, gerçekleşen karışım | eğitim / token | birincil X değerine göre (%) |
|---|---:|---:|
| hepsi `M0:A` | 265,834,602 | -0.00151 |
| hepsi `M1:B` | 265,838,610 | +0.00000 |
| hepsi `M2:C` | 265,837,074 | -0.00058 |
| eşit karışım (1/3, 1/3, 1/3) | 265,836,762 | -0.00070 |

"hepsi A" satırı H'nin değeriyle birebir aynıdır (iki kolun TEK analitik farkı c'dir). Router'ın seçimle
oynatabileceği en büyük pay bu tablonun ilk satırıdır; birincil hesap onu da X'in aleyhine sabitler.

**Geri yön.** Bölüm 4'teki gibi: skor çarpımından q ve k gradyanları, ağırlıklı toplamdan a ve v gradyanları, her
biri aynı c bağlantı üzerinde. Eğitim = 3 × ileri bu terim için de geçerlidir.

**Yuvarlama.** Terim tam kesirle hesaplanıp en sonda tamsayıya yuvarlanır: `round(r · 4 · d · c)`. d = T = 512'de
sonuç zaten tamsayıdır (tablodaki 525,000 ve 526,336).

**Pin (ikinci kilidin FLOP tarafı).** Ön-kaydın O-X1 bölümünde ESKİ tasarım kaldı (A bitişik 256 / B dilated 256 /
C 128+128; nedensel maske altında 192.25 / 128.50 / 88.31 bağlantı — reddedilen compute confound'un kendisi).
Kampanya T'sinde (512) herhangi bir modülün SAYILMIŞ c'si 128.4 ± 0.4 dışındaysa `src/flops.py` sayı üretmez, hata
verir (hedef ve tolerans `src/model.py`'deki `FOTON_CONN_TARGET` / `FOTON_CONN_TOL`; tek kopya).

### 8.3 Router: dizi başına tek çarpım

Router, ilk k = 64 konumun ortalaması olan TEK d-boyutlu vektörü d → n_m lineer katmandan geçirir: DİZİ başına
2·d·n_m = 3,072 FLOP. Token başına yazılışı T'ye bölünmüş hâlidir: `round(2·d·n_m / T)` = 6 (`foton_router`).
Token başına 2·d·n_m saymak router'ı T kat şişirirdi. Geri yön iki çarpımdır (ağırlık ve girdi gradyanı): 3×.
Denetim: bağımsız sayaçta (`FlopCounterMode`) B dizilik bir batch'in `mm` toplamı tam olarak
B·T·(token başına matris çarpımları) + B·2·d·n_m çıkar
(`test_profiler_foton_linear_terms_and_dense_mask_artefact`).

### 8.4 Dört kol

| terim (ileri / token) | D | M | H | X |
|---|---:|---:|---:|---:|
| `attn_qkv_proj` | 16,777,216 | 16,777,216 | 16,777,216 | 16,777,216 |
| `attn_context` | 4,202,496 | 4,202,496 | 3,151,872 | 3,151,872 |
| `ffn_dense` | 34,603,008 | 4,325,376 | 34,603,008 | 34,603,008 |
| `moe_router` | 0 | 114,688 | 0 | 0 |
| `moe_routed_experts` | 0 | 22,708,224 | 0 | 0 |
| `moe_shared_experts` | 0 | 7,569,408 | 0 | 0 |
| `lm_head` | 33,554,432 | 33,554,432 | 33,554,432 | 33,554,432 |
| `foton_attn_routed` | 0 | 0 | 525,000 | 526,336 |
| `foton_router` | 0 | 0 | 6 | 6 |
| **ileri / token** | **89,137,152** | **89,251,840** | **88,611,534** | **88,612,870** |
| **eğitim / token (3×)** | **267,411,456** | **267,755,520** | **265,834,602** | **265,838,610** |

D ve M sütunları FOTON genişletmesinden ÖNCEKİ değerlerle bit düzeyinde aynıdır: ön-kayıttaki 267.411.456 ve
267.755.520 (`test_frozen_baseline_train_flops_unchanged`; FOTON olmayan bütün presetler için
`test_non_foton_configs_bit_identical`). `foton_*` anahtarları her config'te vardır, FOTON olmayanlarda 0'dır.

| karşılaştırma (eğitim / token) | fark (FLOP) | fark (%) |
|---|---:|---:|
| X − H | +4,008 | +0.0015 |
| H − D | -1,576,854 | -0.5897 |
| X − D | -1,572,846 | -0.5882 |
| H − M | -1,920,918 | -0.7174 |
| X − M | -1,916,910 | -0.7159 |

Okuma:

- X − H = 3 · 2 katman · 4d · (c_X − c_H): iki kol arasında başka hiçbir terim farklı değildir.
- FOTON kolları token başına D'den ve M'den daha AZ analitik FLOP harcar, çünkü yönlendirilen 2 katman 256.5 yerine
  ~128.4 bağlantı sayar. Bu fark birincil istatistiğe doğrudan girer: aynı `N*`'da bile `s_i`, tablonun "X − M"
  satırındaki yüzde kadar pozitif çıkar. Ön-kaydın istediği hesap budur (V3), ama büyüklüğü burada açıkça yazılı
  dursun diye ayrıca belirtiliyor; bölüm 8.5'teki çekirdek notuyla birlikte okunmalıdır.

### 8.5 Birincil tanıma GİRMEYEN FOTON işlemleri (v01 EK1-B13 beyanı)

B13: "FOTON'un sayılmayan her işlemi `docs/flops.md`'de listelenir ve (i) analitik formüle eklenir, ya da
(ii) profiler çapraz kontrolüyle üst sınırı ölçülüp raporlanır." Liste eksiksizdir; `src/model.py`'deki FOTON
yolunun (`_foton_route`, `_foton_trunk`, `forward`'daki regularizer) her işlemi aşağıda bir satırdır.

| işlem (kod) | neden birincil tanımın dışında | büyüklük (ileri / token) |
|---|---|---|
| Prefix ortalaması: `x[:, :k].float().mean(dim=1)` | eleman bazlı; dizi başına bir kez | (k·d + d) / T → `foton_prefix_pool` |
| Router softmax + `argmax` (n_m = 3 logit) | eleman bazlı; dizi başına bir kez | 4·n_m / T → `foton_router_softmax_argmax` |
| Dispatch: `(choice == m).nonzero()`, `index_select`, `index_copy`, `torch.zeros_like`, dtype çevirme | aktivasyon üzerinde aritmetik yok: karşılaştırma, tamsayı ve bellek kopyası (bölüm 5'teki MoE dispatch kuralının aynısı) | **0 FLOP**; bedeli duvar saatindedir (ikincil ölçüt) |
| Gate: `p_top / p_top.detach()` ve `gather` | dizi başına 1 bölme; `gather` indeksleme | 1 / T (`foton_gate_residual` içinde) |
| Gate çarpımı ve fark: `x + gate · (y − x)` | eleman bazlı; modül çıkışında BİR kez (katman başına değil): d çıkarma + d çarpma + d toplama | 3·d → `foton_gate_residual` |
| Boolean maskenin uygulanması (`attn_mask=`) | aritmetik değil, seçim — ama aşağıdaki **çekirdek notuna** bakın | analitik sayımda 0; çekirdek düzeyi üst sınır aşağıda |
| Yönlendirilen katmanlarda softmax / ölçekleme | eleman bazlı; c bağlantı üzerinde (birincil terimle aynı c) | 4·n_h·c / katman → `attn_scale_softmax` |
| V7 regularizer'ı, `foton_reg = "entropy"`: clamp, log, çarpma, toplama | eleman bazlı; dizi başına n_m olasılık üzerinde | 4·n_m / T → `foton_reg` |
| V7 regularizer'ı, `foton_reg = "lb"`: `bincount`, f_m, P_m ortalaması, N·Σ f_m·P_m | `bincount` tamsayı; kalanı dizi başına ≤ 2·n_m işlem | 2·n_m / T → `foton_reg` |
| V7 regularizer'ı, `foton_reg = "none"` (karar koşusunun varsayılanı) | hiçbir işlem yok | 0 |
| Oracle değerlendirmesi (`foton_eval_all`: üç modül de koşar) | eval bütçeye girmez (bölüm 5, "Eval" satırı) | eğitim FLOP'una girmez |
| Dizi almayan modüllerin AdamW güncellemesi | adım başına, toplam parametreyle ölçeklenir (bölüm 5, "AdamW" satırı); FOTON'un toplam parametresi D'den büyük | token başına değil; bölüm 5'teki mertebe |

Yaklaşık sayımlar `secondary_forward_flops_per_token` içindedir (yalnızca FOTON'a özgü ya da FOTON'da değişen satırlar):

| ikincil terim (ileri / token, yaklaşık) | D | M | H | X |
|---|---:|---:|---:|---:|
| `attn_scale_softmax` | 65,664.000 | 65,664.000 | 57,451.125 | 57,472.000 |
| `foton_prefix_pool` | 0.000 | 0.000 | 65.000 | 65.000 |
| `foton_router_softmax_argmax` | 0.000 | 0.000 | 0.023 | 0.023 |
| `foton_gate_residual` | 0.000 | 0.000 | 1,536.002 | 1,536.002 |
| `foton_reg` | 0.000 | 0.000 | 0.000 | 0.000 |
| ikincil toplam / birincil ileri (%) | 0.3103 | 0.3390 | 0.3047 | 0.3047 |

FOTON'a özgü ikincil terimlerin toplamı birincil ilerinin yüz binde 2'sinin altındadır: B13'ün kaygısı
("FOTON kapasitesini oraya koyarsa birincil ölçüt onu görmez") bu mimaride yer bulmuyor — sayılmayan işlemlerin
hiçbiri öğrenilen parametre taşımıyor.

**Çekirdek notu: analitik sayım ALGORİTMİK bağlantı sayısıdır, GPU'nun yaptığı aritmetik değil.**
Yönlendirilen katmanlar `F.scaled_dot_product_attention(..., attn_mask=bool maske)` ile koşar. Keyfi bir boolean
maskede çekirdek T×T skorların TAMAMINI hesaplar, sonra maskeler: o iki katmanda GPU'nun yaptığı aritmetik c ile
değil T ile ölçeklenir (ileri 4·d·T / katman). Paylaşılan katmanlar ve D/M ise `is_causal=True` yolundadır; bu
bölümün bağlam terimi onlar için de algoritmik sayımdır (bölüm 2(b)). Analitik tanım bilerek ALGORİTMİKTİR (bölüm 1:
"bir tokenın gerçekten geçtiği hesap"; maskelenen skorlar çıktıya hiç girmez), ve ön-kayıt V3 bu terimin
"gerçekleşen bağlantı sayısından" hesaplanmasını şart koşar. Ama fark küçük değildir ve beyan edilir:

| yönlendirilen 2 katmanın bağlam terimi (ileri / token) | FLOP |
|---|---:|
| analitik, H (c = 128.17) | 525,000 |
| analitik, X (c = 128.50) | 526,336 |
| D'nin aynı 2 katmanı, tam nedensel 2·2d(T+1) | 1,050,624 |
| maskeli çekirdeğin yoğun T×T hesabı, 2·4dT | 2,097,152 |

B13(ii) anlamında ÜST SINIR — yönlendirilen katmanlar algoritmik c yerine çekirdeğin yoğun T×T'siyle sayılsaydı:

| üst sınır: yönlendirilen katmanlar yoğun T×T sayılırsa | eğitim / token | D'ye göre (%) | M'e göre (%) |
|---|---:|---:|---:|
| H | 270,551,058 | +1.1741 | +1.0441 |
| X | 270,551,058 | +1.1741 | +1.0441 |

Yani "X − M" farkının işareti sayım tanımına bağlıdır: algoritmik sayımda FOTON token başına daha ucuz, çekirdek
düzeyi üst sınırda daha pahalıdır. Birincil ölçüt ön-kayıt gereği algoritmik sayımı kullanır; çekirdeğin fazladan
yaptığı iş duvar saatine (ikincil ölçüt) zaten yansır. Seyrek bir çekirdek (blok-seyrek / FlexAttention) bu farkı
kapatırdı; bu kampanyada yoktur. Hangi okumanın raporda başlığa çıkacağı sahibin kararıdır; burada yalnızca beyan edilir.

### 8.6 Profiler çapraz kontrolü: bilinen iki artefakt (B13'ün 5 puan kuralı için)

B13: "Profiler ile analitik sayım arasındaki fark her iki kol için raporlanır; FOTON'un farkı baseline'ın
farkından 5 puandan fazla büyükse sonuç 'başarılı' ilan edilemez." torch'un sayacı (`FlopCounterMode`) bu
karşılaştırmayı iki yerde bozar; ikisi de bölüm 7'de ölçülmüştür:

1. **SDPA maskeden bağımsız T×T sayılır.** Sayaç `is_causal=True`'yu da boolean maskeyi de yok sayar: D'nin
   tam-nedensel katmanı da FOTON'un ~128 bağlantılı katmanı da 4·d·T (ileri) sayılır, geri yön 2.5×. FOTON'un
   analitik bağlam terimi daha küçük olduğu için profiler − analitik farkı FOTON'da yapısal olarak daha BÜYÜK çıkar.
   CPU'daki denetim bunu gösterir: maskeli SDPA matematik yoluna açılır ve sayaç katman başına tam 4·d·T·T / dizi
   `bmm` sayar (`test_profiler_foton_linear_terms_and_dense_mask_artefact`).
2. **`_grouped_mm` 0 sayılır.** M'in varsayılan `moe_impl="grouped"` yolunda routed expert'lerin FLOP'u sayaçta
   hiç görünmez. Bölüm 7'deki ölçüm (profiler analitik değerin %26.7 ALTINDA) V=8192'dedir; kampanya V'sinde
   lm_head büyüdüğü için yüzde küçülür ve aşağıdaki tablonun M (`grouped`) satırında HESAPLANIR. Bu hâliyle
   "baseline'ın farkı" -19.2 puan, FOTON'unki +6.9 puan okunur ve 5 puan kuralı sayacın eksiğinden dolayı
   tetiklenir — FOTON'un sakladığı bir hesaptan dolayı değil.

Bu yüzden B13 karşılaştırması ya M için `moe_impl="loop"` yoluyla (matris çarpımları sayaçta birebir; bölüm 7),
ya da bileşen bileşen (matris çarpımları ayrı, attention çekirdekleri ayrı) yapılmalıdır. Aşağıdaki tablo bu iki
artefakt altında sayacın NE GÖSTERECEĞİNİN öngörüsüdür — **ölçüm değildir**; varsayımı: matris çarpımları birebir,
SDPA ileri 4·d·T / katman ve geri 2.5× (bölüm 7'de V=8192'de ölçülen 29,360,128 = 8 · 4dT · 3.5):

| profiler − analitik farkı, ÖNGÖRÜ (ölçüm değil) | profiler eğitim / token | analitik | fark (%) |
|---|---:|---:|---:|
| D | 284,164,096 | 267,411,456 | +6.265 |
| M (`loop`) | 284,508,160 | 267,755,520 | +6.257 |
| M (`grouped`; `_grouped_mm` 0 sayılır) | 216,383,488 | 267,755,520 | -19.186 |
| H | 284,164,114 | 265,834,602 | +6.895 |
| X | 284,164,114 | 265,838,610 | +6.893 |

D'ye ve M (`loop`)'a göre öngörülen FOTON − baseline fark açıklığı 1 puanın altındadır ve tamamı 1. artefakttan
gelir; M (`grouped`) satırına göre açıklık 5 puanı kat kat aşar ve tamamı 2. artefakttır. Gerçek ölçüm
`experiments/foton-v02/flops_crosscheck.json`'a yazılacaktır (ön-kayıt "Rapor" tablosu). Karşılaştırmanın hangi
yolla yapılacağına (`loop` yolu mu, bileşen bileşen mi) dair HÜKÜM sahibindir; bu belge yalnızca artefaktları ve
öngörüyü kayda geçirir.

### 8.7 Denetim: FOTON formülü neye bağlı

`tests/test_flops.py`, FOTON bölümü. Beklenen değerler `src/flops.py`'nin yardımcılarından değil, donmuş
spesifikasyondan (A yerel W=150 / B dilated r=2 W=256 / C yarım yerel W=256 + yarım strided s=4 W=256) testin
içinde kaba kuvvetle kurulan maskeden sayılır:

1. D ve M'in eğitim FLOP'u ön-kayıttaki sayılarla aynı; FOTON olmayan bütün presetlerde eski anahtarlar bit düzeyinde aynı.
2. Matris çarpımı terimleri = 2 × (bir dizinin geçtiği Linear ağırlıklar), modelden sayılarak; kurulan 12 blok değil.
3. Yönlendirilen bağlam terimi = `round(2 · 4d · c)`, c kaba kuvvet maskeden; H → A, X → maksimum (B).
   `src/flops.py`'nin saydığı c, modelin `attn_mask` buffer'larından sayılanla (`GPT.foton_connection_counts`) aynı.
4. H < X < D < M; X − H yalnızca `foton_attn_routed`'dan gelir ve 3 · 2 · 4d · (c_X − c_H)'ye eşittir.
5. Seçimden bağımsızlık: birincil fonksiyonlar karışım parametresi almaz; hiçbir karışım birincil değeri aşamaz;
   `foton_flops_realized` köşeleri: hepsi-A = H'nin değeri, hepsi-B = X'in değeri. Bozuk karışım (liste, eksik/fazla
   anahtar, toplam ≠ 1, aralık dışı pay — toplamı 1 olan tek negatif pay dahil —, NaN) hata verir. Paylar modüle
   KİMLİKLE eşlenir: anahtarları başka sırada kurulmuş aynı karışım aynı sonucu verir (konumsal eşleme düşer).
6. Router dizi başına bir çarpımdır (T'ye bölünür); bağımsız sayaçla birebir.
7. S1 biçimli config (d=640, 10 katman = 4+2+4, n_head=10) çalışır ve n_head'i attn_patterns'e taşır.
8. ESKİ O-X1 tasarımının sayıları (192.25 / 88.31) ya da tam-nedensel 256.5 formüle sızarsa test düşer; desenler
   eski tasarıma dönerse `src/flops.py` sayı üretmeyi reddeder.
9. Bu bölümün kendisi: üretilen her tablo belgede TABLO TABLO birebir durur (eksik satır da, güncel satırın yanına
   eklenmiş çelişen bir satır da düşürür; üretilmeyen tablolar açık bir beyaz listededir), ve düz metindeki sayılar
   (katman/blok sayıları, router'ın dizi ve token başına FLOP'u, tablodan anılan değerler, B13 için okunan fark
   puanları) config'ten ve formülden kurulup metinde aranır.
