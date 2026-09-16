# FOTON v0.1 — deney ön-kaydı

Bu dosya FOTON kodu yazılmadan önce sabitlenmiştir ve `foton-v01-prereg` tag'iyle dondurulmuştur.
**Bu tag'den sonra hedef direği oynatılmaz.**

Neden bu dosya var: v3 (router ablasyonu) denetimi, ön-kayıt iddiamın artefaktla doğrulanamadığını gösterdi —
ölçüt belgesi koşular başlamadan değil ilk koşu sürerken yazılmıştı ve bunu ancak `provenance.json`'ların git
durumundan geriye dönük tespit edebildim. Dersi burada uyguluyorum: **ölçütler, hedef ve karar kuralı koda ve
koşulara önce yazılıyor ve commit'leniyor.**

Değiştirilemez bölümler: "Hedef L*", "Birincil ölçüt" … "Kilit cümle" ve "Operasyonel tanımlar".
Sonradan ortaya çıkan her sapma dosyanın **sonundaki "Sapmalar" bölümüne eklenir**, yukarıdaki bölümler
düzenlenerek değil.

---

## HEDEF L* (sayı olarak sabitlendi)

    L* = 1.3592   (validation loss, nat/token)

FOTON sonuçları görülmeden, yalnızca dondurulmuş baseline eğrisinden hesaplandı.

**Nasıl hesaplandı.** "En iyi baseline'ın referans eğitim bütçesinin %80'inde ulaştığı ortalama validation loss":

| adım | değer |
|---|---|
| Referans eğitim bütçesi | **98.304.000 token** = 3000 adım × 32.768 token/adım |
| En iyi baseline | **`moe-auxfree`** (v3 koşulu (d): sigmoid gating + seçim bias'ı γ=0.001, aux 0) — referans bütçede en düşük val loss: 1.3196 |
| %80 noktası | 78.643.200 token = **adım 2400** |
| Adım 2400 bir eval noktası değil | eval'ler 250 adımda bir: komşular adım 2250 ve 2500 |
| İnterpolasyon | token cinsinden **lineer**, ağırlık (2400−2250)/(2500−2250) = **0.6** |

Seed başına girdiler ve sonuç:

| seed | adım 2250 | adım 2500 | adım 2400 (interp) |
|---|---|---|---|
| 1337 | 1.374582 | 1.348885 | 1.359164 |
| 1338 | 1.374660 | 1.349409 | 1.359509 |
| 1339 | 1.374894 | 1.348438 | 1.359021 |
| **ortalama** | | | **1.359231 → L\* = 1.3592** |

Seed'ler arası sd = 0.00025, yani hedef iyi belirlenmiş.

**İnterpolasyon kuralı niye kritik.** Alternatifler hedefi ciddi oynatıyor: adım 2250'ye yuvarlamak 1.3747,
adım 2500'e yuvarlamak 1.3489 verirdi — 0.026 nat, yani v2'de ölçülen dense–MoE farkının (0.034 nat)
mertebesinde. log(token) interpolasyonu 1.3589 verir (0.0003 fark, önemsiz). Kural yukarıda sabitlendi ve
değiştirilemez.

**Dondurulmuş baseline eğrisi (SHA-256, `metrics.jsonl`):**

```
417d7af4d8f2ac2601d940167eb4e1b45e9ddf5f4ab776e2608b06f069cf86ce  experiments/v3-router-ablation/runs/auxfree-s1337/metrics.jsonl
3077d6fb053411796ade1d865e37f97b79cf155532de368062e8bf663b5c9aa2  experiments/v3-router-ablation/runs/auxfree-s1338/metrics.jsonl
7adcb7500e021a81f9a4558f26640488fd3393abf64c2a7a9caf3efa0bd28e17  experiments/v3-router-ablation/runs/auxfree-s1339/metrics.jsonl
```

**Baseline'ın L*'e ulaşma maliyeti (karşılaştırmanın 0 noktası):**

    F_base(S0) = 78.643.200 token × 192.258.048 FLOP/token = 1.511979e16 aktif FLOP

**Çapraz kontroller (L*'in makul olduğunu göstermek için, karar için değil).** Aynı L* hedefine:
auxfree bütçenin %80.0'ında (tanım gereği), aux 0.1 %80.2'sinde, moe-tok %81.1'inde, moe-noaux %86.9'unda,
dense %98.7'sinde ulaşıyor. Yani L* dense baseline'ının referans bütçede *ancak* ulaştığı bir seviyedir ve
MoE baseline'ı bu hedefte dense'ten %23.2 daha az FLOP harcıyor. **FOTON'un yenmesi gereken şey dense değil,
bu MoE baseline'ıdır.**

---

## BİRİNCİL ÖLÇÜT

Önceden belirlenmiş validation loss hedefi L*'e ulaşmak için gereken aktif FLOP. L*, dondurulmuş baseline
öğrenme eğrisinden belirlenir: en iyi baseline'ın referans eğitim bütçesinin %80'inde ulaştığı ortalama
validation loss. L* FOTON sonuçları GÖRÜLMEDEN hesaplanır ve bu dosyaya sayı olarak yazılır.
→ **Hesaplandı ve yukarıda yazıldı: L\* = 1.3592.**

## İKİNCİL ÖLÇÜT

Aynı hedefe ulaşmak için gereken duvar saati süresi.

## KARAR TABLOSU

```
  >= %10 tasarruf + quality/sec pozitif ....... güçlü başarı
  >= %5 tasarruf, %95 GA tamamen > 0 .......... başarılı
  +%2 – +%5, veya GA sıfırı kesiyor ........... belirsiz
  -%2 – +%2 ................................... nötr
  < -%2, GA çoğunlukla negatif ................ başarısız
```

**Zorunlu:** sonuç raporunda bu satırlardan birinin adı **birebir** yazılır. "İyi görünüyor",
"umut verici", "kısmen başarılı" gibi yorumlar geçerli sonuç beyanı değildir. Kesin uygulama: O5, O10.

## SEED POLİTİKASI

Karar koşuları 5 eşleştirilmiş seed: {11, 22, 33, 44, 55}. Tarama/debug koşuları 3 seed. Aynı seed = aynı
veri sırası, batch sırası, dropout RNG politikası, eval örnekleri. Mimariler farklı olduğu için weight init
birebir aynı olamaz; mümkün olduğu kadar paired design.
Eşleştirilmiş iyileşmenin standart sapması > 5 puan ise sonuç otomatik "yüksek varyans / belirsiz",
seed 10'a çıkar. Birincil kriter ham SD değil, güven aralığıdır.

## ÇELİŞKİ KURALLARI

FLOP kazancı var, duvar saati kaybı var
→ mimari başarı / sistem başarısızlığı. Fikir korunur, kernel üzerinde çalışılır.

FLOP kaybı var, duvar saati kazancı var
→ algoritmik olarak daha kötü. FOTON'un iddiasını desteklemez.

## BELİRSİZ SONUÇ

Yalnızca BİR yükseltme hakkı: bir üst ölçek basamağı + 5 seed. Basamaklar arası trend artıyorsa devam,
düzse veya azalıyorsa ölçek bizi kurtarmıyor demektir.

## VAZGEÇME KOŞULU

İki ardışık model ölçeğinde ortalama hesap iyileşmesi <= 0, %95 GA üst sınırı < +%2, VE oracle-routing
kazancı < %2 ise sequence-level top-1 FOTON hipotezi terk edilir.

## KİLİT CÜMLE

FOTON v0.1 başarılı sayılacaksa, beş eşleştirilmiş seed üzerinde önceden belirlenmiş validation-loss
hedefine ulaşmak için gereken aktif FLOP'ta en az %5 ortalama azalma göstermeli ve eşleştirilmiş %95 güven
aralığının alt sınırı sıfırın üzerinde olmalıdır. %2-5 arası veya sıfırı kesen güven aralığı belirsiz sayılır
ve yalnızca bir üst ölçek basamağında tekrarlanır. İki ardışık ölçekte ortalama kazanç <= 0, %95 GA üst
sınırı < +%2 ve oracle-routing kazancı < %2 ise hipotez terk edilir.

---

## OPERASYONEL TANIMLAR (yukarıdakileri uygular, değiştirmez)

Bunlar ölçülebilirlik için zorunlu. Her biri, tanımsız bırakılsa sonradan hedef direğini oynatmaya izin
verecek bir boşluğu kapatıyor.

### O1. Baseline kimliği ve eşleştirme

Baseline = `moe-auxfree` preset'i (`src/config.py`), yukarıda config'i ve eğrisi hash'lenmiş koşu.
`L*` **bir daha hesaplanmaz**; S0'da sabit 1.3592'dir.

Ama dondurulmuş baseline {1337, 1338, 1339} seed'leriyle koştu, karar koşuları ise {11, 22, 33, 44, 55}
kullanıyor. Eşleştirmenin tanımlı olması için: **baseline, aynı config ile {11, 22, 33, 44, 55} seed'lerinde
yeniden koşulur** ve eşleştirme FOTON(seed s) ↔ baseline(seed s) olur. Yeni baseline koşuları L*'i
değiştirmez, yalnızca eşleştirilmiş FLOP karşılaştırmasının payını verir.

Sabit kalan her şey (baseline `config.json`'undan birebir):

| ayar | değer |
|---|---|
| veri / tokenizer | `data/sp8k`, SentencePiece 8k, train 545.813.247 / val 5.512.219 token |
| dizi uzunluğu, tokens/adım | 512, **32.768** (batch 32 × grad_accum 2 × 512) |
| bütçe | `max_steps` 3000 (eşdeğeri `max_tokens` 98.304.000) |
| LR | token tabanlı cosine (`src/train.lr_at`), lr 1e-3 → min_lr 1e-4, warmup 200 adım eşdeğeri token |
| optimizer | AdamW, β = (0.9, 0.95), weight_decay 0.1, grad_clip 1.0 |
| eval | `evaluate()`, `eval_interval` 250, `eval_iters` 40, **sabit generator seed 12345** |

Sabit eval seed'i önemli: tüm koşular ve tüm seed'ler **aynı val pencerelerinde** ölçülür, yani val loss
farkları eval örneklemesinden gelmez.

### O2. "Aktif FLOP" nasıl sayılır

`docs/flops.md`'deki analitik hesap, `src/flops.train_flops_per_token(config)`. Profiler yalnızca çapraz
kontrol; karar sayısı **asla** profiler'dan alınmaz (v2'de profiler `_grouped_mm`'i %32.6 eksik saymıştı).

    F(koşu) = N* × train_flops_per_token(koşunun config'i)

FOTON'un modül seçimi aktif FLOP'u değiştiriyorsa, `src/flops.py` FOTON için genişletilir ve genişletme
`docs/flops.md`'de elle türetilerek yazılır; türetim koşulardan önce commit'lenir.

### O3. `N*` = L*'e ulaşan token sayısı

Eval ızgarası **en az 250 adımda bir** olmalı (daha sık olabilir). `N*`, val loss eğrisinin L*'i
**ilk kez aşağı yönde kestiği** yerde, token cinsinden lineer interpolasyonla bulunur:

    N* = x0 + (y0 − L*) / (y0 − y1) × (x1 − x0),  y0 >= L* >= y1 olan ilk komşu eval çifti

**Üst sınır ve sansür.** Bir koşu referans bütçenin **2 katına** kadar koşturulabilir. L*'e hâlâ
ulaşmıyorsa `N*` = 2 × referans bütçe olarak **sansürlü** kaydedilir (bu, FOTON'un aleyhine yanlı bir
ikamedir; kasıtlı, çünkü kanıtlanmaya çalışılan iddia FOTON'un lehinedir). Sansür varsa sonuç raporunda
kaç seed'in sansürlendiği açıkça yazılır ve sonuç "güçlü başarı" veya "başarılı" ilan **edilemez**.

### O4. Birincil istatistik

Seed başına tasarruf, **yüzde puan** olarak:

    s_i = 100 × (1 − F_foton,i / F_base,i)

Raporlanan: ortalama s̄, örneklem sd, ve eşleştirilmiş **%95 GA = s̄ ± t(n−1) × sd/√n** (n=5 için
t(4) = 2.776445; n=10 için t(9) = 2.262157). "Tasarruf %X" ifadesi daima s̄'yi kastediyor.
Seed politikasındaki "5 puan" eşiği **sd(s_i) > 5** demektir.

**Tek birincil test.** Karar yalnızca bu tek eşleştirilmiş test üzerinden verilir; ayrıntı ve test
sayıları O4b'de, hesabın kodu ve regresyon testleri O4c'de.

**Hesabın uygulaması bağlayıcıdır:** `s_i` listesi `src/stats.paired(...)` ile (seed kimliğine göre)
eşleştirilir ve karar `src/stats.decide(...)` ile verilir. Bu fonksiyonlar satır adını döndürür ve
`tests/test_stats.py` ile regresyona bağlıdır (O4c). Elle hesap ya da başka bir uygulama kullanılamaz.

### O4b. Test AİLELERİ ve önceden sayılmış test sayısı

v3 denetiminin en önemli bulgusu şuydu: 151 eşleştirilmiş test yapılmış, 82'si nominal p ≤ .05 çıkmış,
**Holm düzeltmesinden yalnızca 9'u geçmişti** — ve hiçbiri kalite/uzmanlaşma sonucu değildi. FOTON'da aynı
tuzağa düşmemek için testler aileye ayrılıyor ve sayısı **şimdi** sabitleniyor.

**Aile 1 — BİRİNCİL (karar veren). Ölçek başına 1 test.**

| # | test | düzeltme |
|---|---|---|
| P1 | eşleştirilmiş FLOP tasarrufu `s̄`, H0: s̄ = 0 (O4) | yok — tek ve önceden belirli test |

**Karar SADECE P1'e göre verilir.** Tek test olduğu için çoklu karşılaştırma düzeltmesi gerekmez;
bu, "düzeltmeden kaçmak" değil, düzeltmeye ihtiyaç duymayacak kadar az test yapmaktır.

**Aile 2 — İKİNCİL (önceden belirli, tek başına karar vermez). Ölçek başına 2 test.**

| # | test | nereye girer |
|---|---|---|
| S1 | duvar saati tasarrufu `w̄` (quality/sec, O6) | yalnızca "güçlü başarı" satırının ek şartı ve çelişki kuralları |
| S2 | oracle-routing tasarrufu `s̄^oracle` (O8) | yalnızca vazgeçme koşulunun üçüncü şartı |

**Aile 3 — TEŞHİS. Ölçek başına tam 10 test, kendi içinde Holm (FWER .05) + BH (FDR .05) raporlanır.**

| # | ölçüt |
|---|---|
| D1 | routing regret: CE(öğrenilmiş routing) − CE(oracle routing), nat |
| D2 | oracle gap: `s̄^oracle − s̄`, puan |
| D3 | I(girdi sınıfı ; seçilen modül), bit |
| D4 | routing entropisi H(modül)/log M |
| D5 | H(modül \| girdi sınıfı), bit — routing'in bağlama bağlı kısmı |
| D6 | JS uzaklığı: modül çıktı dağılımları arası ortalama ikili Jensen-Shannon |
| D7 | modül yük CV |
| D8 | ölü modül sayısı |
| D9 | korelasyon: routing kararı ↔ dizi uzunluğu |
| D10 | korelasyon: routing kararı ↔ girdi alanı/türü |

**Önceden sayılmış toplam: 1 + 2 + 10 = 13 test/ölçek, en çok 2 ölçek → 26 test.**

Bağlayıcı kurallar:

1. **Hiçbir teşhis ölçütü tek başına başarı iddiası doğuramaz.** Teşhis ailesi yalnızca P1'in sonucunu
   **YORUMLAMAK** için kullanılır: neden kazandı / neden kazanmadı. Karar tablosunun hiçbir satırını
   tetikleyemez, hiçbir satırı değiştiremez.
2. Bu listede olmayan her ölçüm **keşifseldir**: ayrı başlık altında, "keşifsel" etiketiyle raporlanır,
   düzeltilmiş aileye **girmez** ve hiçbir iddiayı desteklemek için kullanılamaz.
3. Teşhis ailesinde bir ölçüt ölçülemezse (ör. FOTON'un mimarisi o ölçütü tanımsız bırakırsa) aile küçülür,
   yerine yeni ölçüt **eklenmez**; eksilme "Sapmalar" bölümüne yazılır.
4. Her iki ölçekte de aynı 13 test yapılır; ölçekler arasında **havuzlama yoktur**, her ölçek kendi
   kararını verir (O7'deki trend kuralı bu kararların karşılaştırılmasıdır, yeni bir test değildir).
5. Raporda test sayısı açıkça yazılır: "yapılan test: 13 (1 birincil + 2 ikincil + 10 teşhis), keşifsel: k".

### O4c. Eşleştirme hatası: kararın geçerliliğini bozan bilinen tuzak

Karar kuralının tamamı **eşleştirilmiş güven aralığının işaretine** bakıyor. v3 denetimi bu hesabı yapan
kodda iki hata buldu; ikisi de kararı sessizce geçersiz kılardı:

1. **Eşleştirme seed KİMLİĞİNE göre değil liste KONUMUNA göre yapılıyordu.** Bir koşulda tek bir analiz
   eksik olsa, örneğin FOTON'un s22'si baseline'ın s11'iyle eşleşiyordu — fark artık eşleştirilmiş değil,
   ama GA eşleştirilmiş gibi (dar) hesaplanıyordu.
2. **t kritik değeri n'den bağımsız sabitti** (t(2) = 4.303). n = 2'ye düşen bir karşılaştırmada gerçek
   değer t(1) = 12.706 olduğu için GA gerçeğin ~3'te birine daralıyordu.

**Bağlayıcı kural:** FOTON'un karar istatistiği `src/stats.py`'deki `paired()` ve `decide()` ile
hesaplanır; başka bir uygulama kullanılamaz. Bu iki hata `tests/test_stats.py`'de regresyona bağlandı
(`test_paired_matches_by_seed_identity_not_position`, `test_paired_ci_uses_t_of_n_minus_one`) ve karar
tablosunun her satırı ile sınır durumları da orada sınanıyor. Karar raporu yazılmadan önce bu testlerin
geçtiği doğrulanır. `decide()` satır adını döndürür; rapor o adı birebir kullanır.

### O5. Karar tablosunun kesin uygulaması (öncelik sırası)

Tablo satırları sırayla değerlendirilir; ilk tutan satır sonucu verir:

| # | koşul | satır adı |
|---|---|---|
| 0 | sd(s_i) > 5 puan | **yüksek varyans / belirsiz** (→ 10 seed) |
| 1 | s̄ ≥ 10 **ve** GA alt sınırı > 0 **ve** quality/sec ortalaması > 0 | **güçlü başarı** |
| 2 | s̄ ≥ 5 **ve** GA alt sınırı > 0 | **başarılı** |
| 3 | s̄ > +2 **ve** (GA sıfırı kesiyor **veya** s̄ < 5) | **belirsiz** |
| 4 | −2 ≤ s̄ ≤ +2 | **nötr** |
| 5 | s̄ < −2 | **başarısız** |

Disambiguasyon notu (yazarın kararı, kullanıcının tablosunu daraltmaz): satır 3 yalnızca **pozitif**
ortalamalar için geçerlidir; s̄ < −2 iken GA sıfırı kesse bile sonuç "başarısız"tır (satır 5) ve raporda
GA'nın sıfırı kestiği ayrıca yazılır. s̄ ≥ 10 ama quality/sec negatifse sonuç satır 2'ye düşer ve
"Çelişki kuralları"ndaki **mimari başarı / sistem başarısızlığı** etiketi eklenir.

### O6. `quality/sec` ve duvar saati

    w_i = 100 × (1 − T_foton,i / T_base,i)      T = L*'e ulaşmak için geçen eğitim süresi (saniye)

`T`, `N*`'e karşılık gelen süredir: `src/train.py`'nin `train_time`'ı (senkronize, eval süresi hariç)
üzerinden, `N*` tokenına lineer interpolasyonla. "quality/sec pozitif" = **ortalama w_i > 0**.

**Ölçüm protokolü (v3 dersi: tok/s'de koşu-oturumu etkisi vardı).** Duvar saati karşılaştırması yalnızca
FOTON ve baseline koşuları **aynı oturumda, dönüşümlü** (baseline s11, FOTON s11, baseline s22, FOTON s22, …)
ve başka GPU yükü olmadan koştuğunda geçerlidir. Farklı oturumlardaki süreler karşılaştırılmaz.
S1'de VRAM için mikro-batch küçültülüp grad_accum büyütülebilir (tokens/adım sabit kalmak şartıyla); bu
hedef direğini oynatmak değildir ama duvar saati karşılaştırması yine aynı-oturum kuralına tabidir.

### O7. Ölçek basamakları (önceden isimlendirildi)

| basamak | d_model | n_layer | n_head | ffn_hidden | expert_hidden | aktif param | referans bütçe | S0'a göre hesap |
|---|---|---|---|---|---|---|---|---|
| **S0** (mevcut) | 512 | 8 | 8 | 1408 | 352 | 30.0M | 98.304.000 token (3000 adım) | 1.00× |
| **S1** (tek yükseltme hakkı) | 640 | 10 | 10 | 1760 | 440 | 55.5M | 182.255.616 token (5562 adım) | 3.40× |
| **S2** (yalnızca S1'de sürerse) | 768 | 12 | 12 | 2112 | 528 | 93.1M | 305.725.440 token (9330 adım) | 9.50× |

Bütçe kuralı: her basamakta token/aktif-parametre oranı S0 ile aynı tutulur (S0: 3.282). MoE tarafında
`n_experts=16, top_k=3, n_shared_experts=1, first_dense_layers=1` sabit.
Maliyet uyarısı: S1'de 10 karar koşusu (5 seed × 2 mimari) bu GPU'da ~11.6 saat, S2'de ~32.5 saat.

**L\*(S1)** aynı kuralla S1 baseline'ının eğrisinden hesaplanır (S1 baseline'ın %80 noktası, lineer
interpolasyon, 5 seed ortalaması) ve **FOTON'un S1 sonuçlarına bakılmadan** bu dosyanın "Sapmalar"
bölümüne sayı olarak yazılır.

### O8. Oracle-routing kazancı

FOTON'un eğitilmiş checkpoint'lerinde, eval sırasında router'ın sequence-level seçimi yerine **o dizi için
kaybı en küçük yapan modül** seçilir (bütün modüller çalıştırılarak). Bu, aynı eval ızgarasında bir
"oracle eğrisi" verir. Oracle-routing kazancı = **birincil ölçütün oracle eğrisiyle hesaplanmış hâli**
(`s_i^oracle`, aynı formül, FLOP payı yine FOTON'un gerçek eğitim FLOP'u — oracle yalnızca eval seçimini
değiştirir).

Sorduğu soru: *router mükemmel olsaydı mimari yine de kazandırır mıydı?* Kazandırmıyorsa (< %2) sorun
router değil mimaridir ve vazgeçme koşulunun üçüncü şartı sağlanır.

### O9. KALİBRASYON: gözlenen gürültü ve tasarımın gücü

Tag'den **önce**, yalnızca dondurulmuş baseline koşularının varyansından yapıldı; hiçbir FOTON verisi
kullanılmadı. Tam çıktı: `experiments/foton-v01-prereg/calibration.log`, üreten betik
`scripts/power_analysis.py` (yeniden koşulabilir).

Sebep: eğer baseline gürültüsü zaten ±%6 mertebesindeyse, +%5 eşiği ve 5 seed garantili "belirsiz" üretir.
Bunu tag'den sonra öğrenmek istemedik.

**(1) Gözlenen seed varyansı (7 koşul × 3 seed = 21 koşu, hepsi S0).**

| koşul | val loss sd (nat) | N* sd (%) |
|---|---|---|
| auxfree | 0.0002 | 0.10% |
| aux 0.1 | 0.0011 | 0.60% |
| aux 0.01 | 0.0023 | 0.53% |
| aux 0.001 | 0.0013 | 0.98% |
| aux 1.0 | 0.0030 | 1.80% |
| aux 0 | 0.0019 | 1.28% |
| dense | 0.0009 | 0.73% |
| **havuzlanmış (df 14)** | **0.00175** | **1.00%** |

Eğri eğimi L* civarında d(val)/d(ln token) = **−0.2439 nat**, yani %1 hesap farkı ≈ 0.00243 nat. Buna göre
val loss sd'si 0.00175 nat ≙ **%0.72** hesap cinsinden; doğrudan ölçülen %1.00 ile aynı mertebede (tutarlı).

→ **Tek koşu gürültüsü ±%1.00 (1 sd), korkulan ±%6 değil.**

**(2) Eşleştirilmiş farkın sd'si** (21 koşul çifti, her biri 3 seed): havuzlanmış **1.37 puan** (aynı
mimari, df 30) ve **1.15 puan** (mimari-ötesi = dense↔MoE çiftleri, df 12). Küçük farklı çiftlerde 0.5–1.1,
en gürültülü çiftte 2.5 puan.

**(3) Güç (Monte Carlo, 200.000 tekrar, ön-kayıttaki BİLEŞİK kuralla: s̄ ≥ 5 VE GA alt sınırı > 0).**

| gerçek etki | n=5, sd=1.15 | n=10, sd=1.15 | n=5, sd=3.0 | n=5, sd=6.0 |
|---|---|---|---|---|
| 0% | %0 başarılı (100% nötr) | %0 | %0 | %1.0 (tripwire %59) |
| +2% | %0 (%50 belirsiz) | %0 | %1.2 | %4.6 |
| **+5%** | **%49.8** (%50.2 belirsiz) | **%49.9** | %47.3 | %18.7 |
| +7% | **%100** | %100 | %90.1 | %29.7 |
| +10% | %100 | %100 | %97.5 | %38.9 |

n=5'te %95 GA yarı-genişliği **1.43 puan**, yani "GA alt sınırı > 0" şartı s̄ > ~1.4 puandan itibaren
sağlanıyor: **gürültü, +%5 eşiği için bağlayıcı değil.**

**(4) Eşleştirilmiş tasarım varyansı düşürüyor mu? HAYIR.** Ölçülen paired/unpaired kazancı **0.90×**
(aynı mimari) ve **0.82×** (mimari-ötesi) — yani 1'in altında, ölçülebilir kazanç yok. Nedeni rastgele blok
ayrışımıyla doğrulandı: koşullar arası seed korelasyonu ortalama **ρ = +0.07**; seed'in açıkladığı varyans
payı %16, ama saf gürültüde beklenen pay 2/14 = **%14** — yani **ayırt edilemiyor**. Aynı veri sırasını
paylaşmak 3000 adım sonunda ortak bir sapma bırakmıyor; eğitim kaotik biçimde ayrışıyor (v2'de de
gözlenmişti). Eşleştirme protokol hijyeni için korunuyor (aynı veri sırası, aynı val pencereleri) ama
**varyans düşürmesine güvenilmiyor**; yukarıdaki güç hesabı da bu kazancı hiç varsaymıyor.

**KARAR (kullanıcının 3. sorusu: eşiği yükselt mi, seed'i artır mı?) → İkisi de yapılmıyor.** Gerekçe:

- **Eşik yükseltilmiyor:** gözlenen gürültü eşiği yutmuyor (1.4 puanlık GA'ya karşı 5 puanlık eşik).
  Eşiği gürültüye göre yükseltmenin bir gerekçesi yok; yükseltmek iddiayı daha güvenilir yapmaz, sadece
  daha zor yapar.
- **Seed artırılmıyor:** n=10, +%5'teki gücü %49.8'den %49.9'a çıkarıyor — yani **hiç değiştirmiyor.**
  Bağlayıcı olan gürültü değil, `s̄ ≥ 5` **nokta-tahmin** şartı: gerçek etki tam eşikte iken
  P(s̄ ≥ 5) ≈ %50'dir ve bu seed sayısından bağımsızdır. Daha fazla seed, zaten bağlayıcı olmayan
  GA şartını iyileştirir.
- **Tasarımın gerçek sınırı şu ve şimdi kaydediliyor:** güvenilir biçimde saptanabilen en küçük etki
  **≈ +%7** (n=5'te ~%100 güç). +%5 ile +%7 arası bir gerçek etki, kayda değer olasılıkla "belirsiz"
  çıkacaktır — bu tasarım hatası değil, eşiğin tanımının sonucudur ve tek ölçek yükseltme hakkı
  tam bu durum içindir.
- **Tripwire yerinde:** örneklem sd > 5 puan, gözlenen gürültünün ~4 katı. sd gerçekten 5–6 puan olsaydı
  tripwire %40–60 oranında ateşlenir ve 10 seed'e çıkılırdı; yani kullanıcının korktuğu senaryoda tasarım
  sessizce yanlış cevap vermiyor, "yüksek varyans" diyor.

**Not:** bu kalibrasyon S0 baseline'ının gürültüsünü ölçüyor. FOTON farklı bir mimari olduğu için sd'nin
daha büyük çıkması beklenir; güç tablosu bu yüzden sd = 3.0 ve 6.0 satırlarını da içeriyor ve
S1'de kalibrasyon **aynı betikle yenilenir** (S1 baseline'ının 5 seed'i koşulduktan sonra, FOTON'a
bakılmadan).

### O10. Tag'den sonra nesi değişebilir, nesi değişemez

**Değişemez:** L* = 1.3592, baseline kimliği ve eğrisi (O1), referans bütçe, birincil ölçüt ve istatistik
(O2–O4), test aileleri ve önceden sayılmış test sayısı (O4b), karar istatistiğinin uygulaması ve
regresyon testleri (O4c), karar tablosu ve eşikleri (O5), quality/sec tanımı ve ölçüm protokolü (O6),
seed kümeleri, ölçek basamakları ve bütçeleri (O7), oracle tanımı (O8), kalibrasyonun sonucu (O9).

**Değişebilir (uygulama):** FOTON'un kendi mimari ayrıntıları, kernel/uygulama optimizasyonları, VRAM için
mikro-batch/grad_accum bölüşümü (tokens/adım sabit), log/eval sıklığının **artırılması**, kod refactor'ları.

**Zorunlu raporlama.** Sonuç raporunda karar tablosundaki **satır adı birebir yazılır**: "güçlü başarı",
"başarılı", "belirsiz", "nötr", "başarısız" veya "yüksek varyans / belirsiz". "İyi görünüyor", "umut verici",
"kısmen başarılı" gibi ifadeler geçerli sonuç beyanı değildir. Raporda ayrıca s̄, sd, %95 GA, n, sansürlü
seed sayısı, w̄ (quality/sec) ve varsa çelişki kuralı etiketi verilir.

---

## Sapmalar

(Boş. Ön-kayıttan her sapma, gerekçesi ve fark edildiği tarihle buraya eklenir; yukarıdaki bölümler
düzenlenmez.)
