# MİMARİ — KURULDUĞU HÂLİYLE (FOTON v0.2; dört kol: D, M, H, X)

Bu sayfa tasarım niyetini değil **kodun kurduğu şeyi** anlatır. Sayılar `scripts/architecture_as_built.py --data` ile modeller
GERÇEKTEN kurulup (`vocab_size = 32768`, T = 512) ve veri akışı yeniden oynatılarak üretildi; `tests/test_architecture_doc.py`
bu sayfadaki sayıların betiğin çıktısıyla aynı olduğunu sınar. Bağımsız iki denetçi sayfayı cümle cümle koda karşı doğruladı
(ilk hâlde bir yanlış cümle ve plan için kritik eksikler bulundu; düzeltildi). Kaynak: `src/model.py`, `src/moe.py`,
`src/attn_patterns.py`, `src/config.py` (preset `foton-d / -m / -h / -x`), `src/train.py`, `src/data_multi.py`.

## 1. Dört kolun ortak iskeleti

Decoder-only, pre-norm Transformer. `d_model = 512`, `n_head = 8` (head 64), bir dizinin GEÇTİĞİ derinlik **8 blok**, RoPE
(θ = 10000; yalnızca q ve k'ye), RMSNorm (eps 1e-6), **hiçbir `nn.Linear`'da bias parametresi yok**, dropout 0, giriş
embedding'i = çıkış projeksiyonu (**bağlı**; 16,777,216 parametre, bir kez sayılır), son RMSNorm (512). Bir **dense blok** =
`x + Attn(Norm(x))`, sonra `x + SwiGLU(Norm(x))`:

| parça | parametre |
|---|---|
| attention: `qkv` 512→1536 + `proj` 512→512 | 786,432 + 262,144 |
| SwiGLU, gizli boyut 1408 (3 matris) | 2,162,688 |
| iki RMSNorm | 1,024 |
| **dense blok toplamı** | **3,212,288** |

Init: bütün `nn.Linear` ve embedding N(0, 0.02); residual'a yazan projeksiyonlar (`proj`, SwiGLU çıkışı) ek olarak `(2·8)^-0.5`
ile küçültülür (std 0.005) — dört kolda aynı: FOTON'da da geçilen derinlik 8 alınır, kurulan 12 blok değil. Tam nedensel
attention = SDPA `is_causal=True`.

## 2. D ve M — tam tanım

**D (dense):** 8 dense blok, hepsi tam nedensel (token/head başına ortalama 256.5 bağlantı). Toplam = aktif = **42,476,032**
parametre (16,777,216 embedding + 8 × 3,212,288 + 512). Analitik eğitim FLOP/token: 267,411,456.

**M (MoE):** attention, normlar ve residual yapı D ile aynı; değişen yalnızca FFN. **Blok 0 dense** (SwiGLU 1408); **blok 1–7
MoE**: 16 yönlendirilen expert (her biri SwiGLU, gizli 352 = 540,672 parametre) + her token'ın geçtiği **1 paylaşılan expert**
(gizli 352) + router `Linear(512→16)` (bias'sız, 8,192). **Token başına top-3**, dropless (kapasite sınırı / düşen token yok);
router skoru **sigmoid** (expert başına bağımsız), float32'de ve autocast dışında; seçilen 3 skor toplamı 1 olacak şekilde
normalize edilir. Dengeleme **aux-loss'suz**: MoE katmanı başına 16 elemanlı bir seçim bias'ı (parametre DEĞİL: gradyansız,
checkpoint'e yazılan buffer; yalnızca top-k SEÇİMİNE eklenir) her optimizer adımından sonra yük sapmasının işaretiyle 0.001
güncellenir; `aux_loss_coef = 0`. **z-loss 0.001 AÇIK:** M'in eğitim loss'u = CE + 0.001 · z, z = 7 MoE katmanının ORTALAMASI,
katmanda `mean(logsumexp(ham logit)²)`; raporlanan val loss saf CE. Token başına aktif FFN genişliği 3 × 352 + 352 = **1408 =
D'ninki**. MoE bloğu 10,249,216 parametre. Toplam **91,734,528**, aktif **42,533,376**. FLOP/token: 267,755,520.
Expert hesabı `moe_impl = "grouped"`.

## 3. H ve X — katman düzeni: "3 + 2 + 3" tam olarak ne

Bir dizi sırasıyla şunlardan geçer (toplam 8 blok, D ile aynı derinlik):

```
embedding → [S0, S1, S2] → ROUTER → seçilen modül m'in [m.0, m.1] blokları → [S3, S4, S5] → son norm → LM head
             paylaşılan,              üç modülden YALNIZCA BİRİ,               paylaşılan,
             tam nedensel             maskeli attention                       tam nedensel
```

* **Kurulan blok sayısı 12'dir:** 6 paylaşılan (S0–S5) + 3 modül × 2 blok. Dizi başına 8'i çalışır.
* **Paylaşılan olanlar:** embedding / LM head (16,777,216), 6 tam-nedensel dense blok (19,273,728), son norm (512), router (1,536).
* **Her modülün KENDİ ağırlıkları vardır:** modül = 2 tam dense blok — kendi `qkv`, `proj`, SwiGLU(1408) ve normları —
  **6,424,576 parametre**; üç modül arasında (ve modüllerle paylaşılan bloklar arasında) paylaşılan TEK parametre yoktur
  (kurulan modelde sınandı). Modüller yalnızca maskeyle değil, attention VE FFN ağırlıklarıyla da ayrıdır; bağımsız init
  çekimleridir. (`src/model.py`'nin başlığındaki "Modüller aynı `Block`'tur; TEK fark attention maskesi" cümlesi aynı SINIFI
  kastediyordu ama "ağırlıklar paylaşılıyor" diye okunabiliyordu; düzeltildi.)
* Toplam **55,326,720**; dizi başına aktif **42,477,568** (= D + router'ın 1,536'sı). FLOP/token: H 265,834,602 · X 265,838,610.
  Bunun router terimi ileri 6 / eğitim 18 FLOP'tur ve `src/flops.py` onu `foton` doluyken HER ZAMAN sayar; router'sız karşılıklar
  H 265,834,584 · X 265,838,592. Yönlendirilen katmanların bağlam terimi kolun EN BÜYÜK sayılmış bağlantısıyla (X 128.5,
  H 128.17) sabit sayıldığı için FLOP hangi dizinin hangi modüle gittiğine BAĞLI DEĞİLDİR; H, X'ten 4,008 FLOP/token azdır.
* **Modül katmanlarının attention grafiği** (`src/attn_patterns.py`; [8, 512, 512] boolean maske, parametre değil, checkpoint'e
  girmez; token/head başına bağlantı maske kurulup SAYILARAK):

| modül | 8 head'in deseni | sayılmış bağlantı | en uzak erişim |
|---|---|---|---|
| A | 8 head: yerel pencere, son 150 pozisyon | 128.173828125 | 149 |
| B | 8 head: dilated, adım 2, 256 pozisyon (ofset 0, 2, …, 510) | 128.5 | 510 |
| C | head 0–3: yerel 256; head 4–7: strided, adım 4 (ofset 0, 4, …, 508; T = 512'de 128 pozisyona doyar) | 128.375 | 255 / 508 |

  Modülün iki bloğu da AYNI deseni kullanır. Paylaşılan 6 blokta maske yoktur (256.5 bağlantı).
* **H'nin X'ten TEK farkı modül maskeleridir:** X = (A, B, C); **H = (A, A, A)** — üç modül de yerel-150 desenini kullanır ama
  yine üç AYRI ağırlık kümesidir. Parametre sayıları, şekilleri ve kurulum sırası aynıdır; maskeler RNG tüketmediği için aynı
  seed'de H ve X **bit düzeyinde aynı başlangıç ağırlıklarından** başlar. (Init sırası embedding → bloklar → modüller → router →
  bağlı embedding'in LM head üzerinden YENİDEN çekimi olduğundan, router'ı kurmayan bir kol aynı seed'de aynı embedding'den
  başlamaz.) H, "ayrı ağırlık" etkisini "farklı grafik" etkisinden ayıran koldur. **H v02'de HİÇ KOŞULMADI**; eldeki bütün FOTON
  verisi X'tir.

## 4. Router — girdi, çıktı, gradyan, zorlama

* **Girdi:** S2'nin çıkışındaki HAM residual akış (normalizasyon yok), **pozisyon 0..63'ün** eşit ağırlıklı ortalaması → 512
  boyutlu tek vektör (dizi başına). Pozisyon 64 ve sonrası router'a hiç girmez (nedensellik; testle kilitli). Ölçülen geometri
  (adım 2188, 9 tarama koşusu; teste bağlı DEĞİL, kaynak `router_diag/*_probe.json`): ortak bileşenin normu 18.4–33.5 ve seed'e
  bağlı, dizi-başı sapmanın ortalama normu 18.8–21.6; merkezlenmiş vektörün ilk 64 ana bileşeni üzerinde bias'lı doğrusal prob
  ALANI %96.6–97.7 okur (router'ın kendisi bias'sızdır ve ham vektörü görür).
* **İşlem:** `Linear(512→3)`, bias yok, sıcaklık 1, gürültü yok; float32, autocast dışında; softmax → **argmax** (eşitlikte küçük
  indeks). **Dizi başına TEK karar**: seçilen modül o dizinin 512 pozisyonunun tamamına, iki yönlendirilen blokta da uygulanır.
  Token ya da katman başına yönlendirme YOKTUR.
* **Çıktı / gate:** `x_çıkış = x + g · (Modül(x) − x)`, `g = p_seçilen / p_seçilen.detach()` — ileri yönde **g ≡ 1** (modül çıktısı
  ölçeklenmez), geri yönde router'a `dL/dg · ∂log p` gradyanı akar: router'ın görev loss'undan aldığı TEK sinyal dizi başına bir
  skalerdir. V7'nin iki regularizasyon koşulu yalnızca taramada denendi; `foton-h` / `foton-x` preset'lerinde ek terim YOKTUR.
* **Çalışmayan modül / gradyansız parametre:** bir mikro-batch'te dizi almayan modül o mikro-batch'te çalışmaz. Gradyanlar
  optimizer adımının 4 mikro-batch'i boyunca birikir; bir parametrenin gradyanı dördünde de `None` kalırsa AdamW onu o adımda
  TÜMÜYLE atlar (weight decay dahil; ağırlık bit düzeyinde değişmez — sınandı).
* **Dışarıdan zorlama — v02 kollarında YOK, v03 adım 1'de EKLENDİ (2026-09-20):** v02'nin dört kolunda eğitim ve eval döngüsü
  modeli `model(x, y)` diye çağırır; modele alan kimliği ya da dizi başına bir atama GEÇİRİLMEZ. v03 adım 1'in üç kolunda
  (`foton-x-dom`, `foton-h-dom`, `foton-x-iid`; `ModelConfig.foton_assign` = `"domain"` | `"iid"`) atama dışarıdan, dizi başına
  geçirilir (`model(x, y, assign=...)`): router modelde KALIR ama çalışmaz ve gradyan almaz, gate yoktur (çıkış =
  Modül_atama(x)); `val_loss` EĞİTİMDEKİ atama kuralıyla hesaplanır (X-iid'de beklenen değer = üç modülün ortalaması) ve
  router'ın seçimiyle hesaplanan sayı `val_loss_router_UNUSED` adını alır; atamasız çağrı HATA verir (sessizce router'a düşmez).
  Ayrıntı: `experiments/foton-v03-step1/decision_rule.md`. Aşağıdaki cümleler v02 kolları içindir. Alan kimliği her batch için `dataset.last_domain_ids`'te hazırdır ama yalnızca eğitimde router
  loglarına ve eval'de `eval_raw`'ın `domain` alanına yazılır. `forward(..., force_module=m)` bütün batch'i TEK modüle zorlar,
  gate'i ATLAR (router yine koşar ve loglanır ama gradyan almaz → init'te bit düzeyinde sabit kalır; `router_*` logları o zaman
  eğitilmemiş router'ı anlatır) ve yalnızca testlerde ve `scripts/router_grad_diag.py`'de çağrılır. **Oracle eğrileri ve `eval_raw`
  oradan DEĞİL, ayrı bir yoldan gelir:** `GPT.foton_eval_all` router'ı bir kez koşturur, sonra aynı batch'i üç modülün ÜÇÜNDEN de geçirir.

## 5. Eğitim, veri ve ölçüm — dört kolda aynı

* **Optimizer:** AdamW (β 0.9 / 0.95), weight decay 0.1 yalnızca ≥ 2 boyutlu tensörlere (router ağırlığı ve embedding dahil; normlar
  hariç), tepe LR 1e-3, 200 adım doğrusal ısınma, görülen token üzerinden cosine → 1e-4, global gradyan kırpma 1.0, bf16 autocast.
  Adım başına **32,768 token** (16 dizi × 4 biriktirme × 512); bütçe **4,375 adım = 143,360,000 token**; V7 taraması adım 2188'de
  durur (71.7M token), LR takvimi tam bütçe üzerindendir (adım 2188'de LR 5.84e-4: cosine inmemiştir).
* **Veri:** üç alan, alan kimliği **code = 0 / math = 1 / nl = 2**, alan başına 100M eğitim token'ı. Her dizi TEK alandan, rastgele
  başlangıçlı 512 token'lık penceredir; veride doküman sınırı / EOS token'ı YOKTUR. Alan, dizi başına BAĞIMSIZ multinomial çekimdir
  (ağırlık 1/3; katmanlı DEĞİL): 16'lık mikro-batch'te alan başına 5.3 ± 1.9 dizi (0–15; tarama bütçesindeki 8,752 mikro-batch'in
  40'ında bir alan hiç yok), 64'lük adımda ~21.3 (9–36; hiçbir adımda alan eksik değil).
* **Epoch YOKTUR ve eğitimde dizinin kalıcı kimliği YOKTUR:** bir dizi = o an çekilen (alan, başlangıç) çiftidir; başlangıçlar
  yerine koyarak çekilir. Tarama bütçesinde 140,032 çekimin yalnızca **25**'i birebir tekrardır (tam bütçede 280,000'de 112), ama
  pencerelerin %37.9'u (tam bütçede %61.6) başka bir pencereyle KISMEN örtüşür; çekilen token korpusun %23.9'u (tam bütçe %47.8),
  tekil kapsama %21.3 (%38.0). Yani eğitimde "dizi başına sabit" bir atama, her çekimde bağımsız atamadan ayırt edilemez. Kalıcı
  kimliği olan tek küme 1,280 val penceresidir.
* **Veri sırası yalnızca seed'e bağlıdır:** diziler modelden bağımsız ayrı bir generator'dan (`train_gen`) çekilir; aynı seed'li
  bütün kollar aynı dizileri aynı sırada görür (üç V7 koşulunda sınandı). O generator'dan yapılacak her ek çekim sırayı kaydırır.
* **Ölçüm:** eğitim loss'u 512 pozisyonun tamamındaki CE'dir; **raporlanan `val_loss` pozisyon 64..511** üzerindendir, her eval'de
  AYNI sabit 1,280 val penceresinde (16 × 80; code 431 / math 434 / nl 415), adım 0'da, 250 adımda bir ve durma adımında. **FOTON
  kollarında `val_loss` (ve `best.pt` seçimi) ROUTER'ın kendi argmax seçimiyle hesaplanır**; eval yolunda da dışarıdan atama yoktur.
  Her eval ayrıca `eval_raw/step_*.npz`'ye şunları yazar: `ce_tok` [1280, 3, 512] float16, `seq_masked` [1280, 3] (pozisyon 64..511
  ortalaması), router'ın `choice` / `probs` / `logits`'i ve `domain` — yani HERHANGİ bir dizi→modül atamasının val loss'u ve oracle
  tavanı bu dökümden sonradan hesaplanabilir (gate ileri yönde 1 olduğu için değerler aynıdır).
* **D / M referansıyla karşılaştırma:** dondurulmuş D ve M koşuları seed 11–55 ile ve tam bütçeyle koştu; tarama seed'leri 101–103'tür
  (veri akışı eşli DEĞİL) ve D / M'de adım 2188'de eval yoktur — son ortak eval noktası adım 2000'dir (65,536,000 token).
