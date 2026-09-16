# FOTON v0.1 — deney ön-kaydı

Bu dosya FOTON kodu yazılmadan önce sabitlenmiştir ve `foton-v01-prereg` tag'iyle dondurulmuştur.
**Bu tag'den sonra hedef direği oynatılmaz.**

> **EK 1 (amendment, r2) var — dosyanın sonuna bakın.** r1 tag'i atıldıktan sonra, ondan önce başlatılmış
> bağımsız denetim 37 boşluk buldu (5 kritik) ve hepsi **hiçbir FOTON verisi üretilmeden** kapatıldı.
> L* değişmedi (üç bağımsız hesapta teyit edildi). Çelişki hâlinde EK 1 geçerlidir.

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

## EK 1 (amendment, r2) — 2026-09-16, `foton-v01-prereg-r2`

**Bu ek, HİÇBİR FOTON VERİSİ ÜRETİLMEDEN ÖNCE yazıldı.** FOTON kodu hâlâ yazılmamıştır; hiçbir FOTON
koşusu başlatılmamıştır. Dolayısıyla ön-kaydın bağlayıcılığı bozulmamıştır: burada kapatılan boşlukların
hiçbiri FOTON sonuçlarına bakılarak seçilmemiştir.

`foton-v01-prereg` (r1) tag'i **silinmedi**; ikisi de duruyor ve r1, bu eke ihtiyaç duyulduğunu gösteren
kaydın kendisidir. **Çelişki hâlinde bu ek geçerlidir.** Yukarıdaki donmuş bölümler düzenlenmemiştir.

**Neden:** r1 tag'i atıldıktan sonra, tag'den önce başlatılmış olan bağımsız denetim tamamlandı (76 ajan;
3 ajan L*'i sıfırdan yeniden hesapladı, 4 ajan "kötü niyetli deneyci" rolünde dosyanın lafzına uyarak
sahte başarı ilan etme yollarını aradı, her bulgu ayrıca çürütülmeye çalışıldı). **37 boşluk doğrulandı
(5 kritik, 17 önemli, 15 küçük); 60 sayı/iddia doğru bulundu.**

### A. L* DEĞİŞMİYOR — üç bağımsız hesapta teyit edildi

Üç ajan, tanımı farklı okumalarla sıfırdan hesapladı: **3/3 sonuç L\* = 1.3592**, per-seed girdileri
(1.359164 / 1.359509 / 1.359021), sd'si (0.00025), interpolasyon ağırlığı (0.6), `F_base` (1.511979e16) ve
çapraz kontrol yüzdeleri (%80.0 / %80.2 / %81.1 / %86.9 / %98.7) dahil birebir uyuştu. Üç `metrics.jsonl`
hash'i de bağımsız olarak yeniden hesaplanıp doğrulandı. **Hedef direği yerinde kalıyor.**

**A1. Aday havuzu ve dışlamalar (r1'de belgelenmemişti, şimdi belgeleniyor).**

| aday | referans bütçede val loss | L* olsaydı | karar |
|---|---|---|---|
| `moe-auxfree` (v3 d), 3 seed | **1.3195** | **1.3592** | **seçildi** |
| `aux 0.1` (v3 e), 3 seed | 1.3207 | 1.3600 | dışlandı: daha yüksek val loss |
| `moe` aux 0.01 (v3 a), 3 seed | 1.3231 | 1.3627 | dışlandı: daha yüksek val loss |
| v1 `moe`, **1 seed** | 1.3172 | 1.3561 | **dışlandı:** (i) n=1, "ortalama" tanımsız; (ii) tekrar üretilemez — aynı config ve aynı seed bugünkü kodla 1.3235 veriyor, çünkü v1'de `sample_tokens=120` eval'lerde üretim yapıp global RNG'yi tüketiyor ve sonraki veri sırasını kaydırıyor; (iii) eğrisi `tokens` alanı taşımıyor |
| v2 `dense-time`, 3 seed | 1.3068 | 1.3425 | **dışlandı:** bütçesi farklı (135.626.752 token). Tanım "REFERANS bütçenin %80'i" diyor; bu kol duvar-saati kontrolüdür |

Seçilen kol, uygun havuz içinde **en düşük L*'i** yani **en zor hedefi** veriyor — yani seçim FOTON'un
lehine değil aleyhine yanlıdır. Bu kasıtlıdır.

**A2. "En iyi" bir nokta tahminidir.** (d) ile (e) ve (a) arasındaki fark v3'ün kendi eşleştirilmiş
GA'sında anlamlı değil ((a)−(d) = +0.00349 GA [−0.00045, +0.00743]; (e)−(d) = +0.00075 GA [−0.00244,
+0.00394]). Ayrıca (d), v3'ün taksonomisinde bir **ablasyon kolu**dur ve v3 kaydı "(d)'nin sonuçları
dengeleme yöntemine atfedilemez" der. Burada (d) bir nedensel iddia için değil, **uygun havuzdaki en güçlü
yapılandırma** olduğu için kullanılıyor; hedefin kaynağı olması nedensel bir yorum taşımaz.

**A3. Sayı düzeltmeleri (donmuş metinde yanlış yazılmış).**

| yer | r1'de yazan | doğrusu | etki |
|---|---|---|---|
| satır 28 | en düşük val loss **1.3196** | **1.3195** (üç seed ortalaması 1.319543) | L*'i etkilemiyor |
| satır 64 | MoE baseline dense'ten **%23.2 daha az** FLOP harcıyor | **%18.8 daha az** (O4 formülüyle). %23.2, "dense baseline'dan %23.2 daha FAZLA harcıyor"un karşılığıdır | aşağıya bak, B7 |

### B. KAPATILAN BOŞLUKLAR

Aşağıdaki maddeler r1'in donmuş kurallarını **daraltır** (asla genişletmez) ve hepsi FOTON verisi
görülmeden yazıldı.

**B1 (KRİTİK). Eval penceresi sayısı sabitlendi: `batch_size × eval_iters = 1280`.**
`evaluate()` val pencerelerini `eval_iters` kez `batch_size` pencere çekerek üretiyor (`src/train.py:70-74`),
yani pencere KÜMESİ mikro-batch'e bağlı. r1'in O6/O10'u "VRAM için mikro-batch küçültülebilir" diyordu ve bu
tek başına ölçülen val loss'u değiştiriyor. Dondurulmuş checkpoint'te (`checkpoints/v3/auxfree-s1337/last.pt`,
adım 3000) ölçtüm:

| batch_size | eval_iters | pencere | val loss | fark | ≈ puan |
|---|---|---|---|---|---|
| 32 | 40 | 1280 | 1.319447 | — | — |
| 16 | 40 | 640 | 1.310783 | **−0.008664** | **−3.55** |
| 8 | 40 | 320 | 1.314114 | −0.005333 | −2.19 |
| 64 | 40 | 2560 | 1.315590 | −0.003857 | −1.58 |
| **16** | **80** | **1280** | **1.319441** | −0.000006 | −0.00 |

Yani mikro-batch'i 16'ya düşürmek modeli hiç değiştirmeden **3.55 puan** bedava tasarruf üretiyordu —
eşiğin %71'i. Son satır çözümü veriyor: **bağlayıcı olan pencere SAYISIDIR, mikro-batch değil.**

**Kural:** karar koşularının ikisi de `batch_size × eval_iters = 1280` ve eval generator seed 12345
kullanır. VRAM mikro-batch'i 16'ya zorlarsa `eval_iters` 80 olur (grad_accum da 4 olur, tokens/adım 32.768
sabit kalır). Bu çarpım **değişmez**; başka bir değer kullanılırsa baseline aynı değerle yeniden koşulur ve
sapma bu bölüme yazılır.

**B2 (KRİTİK). Koşu defteri zorunlu; iptal edilen koşu gizlenemez.**
r1 yalnızca "biten ama L*'e ulaşmayan" koşuyu kapsıyordu; çöken, NaN'a giden, OOM alan ya da "hata buldum"
denip atılan koşu için kural yoktu — yani kötü bir seed sessizce yeniden koşulabilirdi.

**Kural:** `experiments/foton-v01/run_ledger.jsonl` tutulur ve **başlatılan her koşu** oraya yazılır:
koşu adı, seed, kol (FOTON/baseline), başlangıç zamanı, `git_head`, `src_sha256`, config'in SHA-256'sı,
sonuç (`bitti` / `çöktü` / `iptal` + gerekçe). Karar, seed başına **ilk BİTEN** koşuyu kullanır.
Bir karar seed'i yalnızca **donanım/altyapı kaynaklı sert arıza** (OOM, sürücü hatası, NaN) hâlinde
yeniden koşulabilir; "implementasyon hatası bulundu" gerekçesiyle yeniden koşmak, **bütün 5 karar seed'inin**
yeniden koşulmasını gerektirir ve sapma olarak yazılır. Rapor, defterdeki koşu sayısını verir.

**B3 (KRİTİK). İkinci beş seed şimdi isimlendirildi: {66, 77, 88, 99, 110}.**
r1, sd > 5 puanda "seed 10'a çıkar" diyordu ama ikinci beşi hiç adlandırmamıştı; sonuca bakarak seed
seçmek mümkündü. Tripwire ateşlenirse **tam olarak bu beş seed** eklenir, başkası değil. n=10'da karar
tablosu aynen yeniden uygulanır (eşikler değişmez, t(9) = 2.262157). İkinci artırma hakkı **yoktur**.

**B4 (KRİTİK). Tarama ayrı seed kümesinde: {101, 102, 103}. FOTON v0.1 TEK mimaridir.**
r1, denenebilecek FOTON varyant sayısını hiç sınırlamıyordu; O10 "mimari ayrıntılar değişebilir" diyordu.
5 varyanttan en iyisini 3 seed'de seçmek `s̄`'yi ~0.6–1 puan şişirir (dondurulmuş kollarda ölçülen
sd(s_i) 0.5–1.9 puan üzerinden).

**Kural:** (i) tarama/debug koşuları **yalnızca** {101, 102, 103} seed'lerinde yapılır; karar seed'leri
{11,22,33,44,55} ve {66,77,88,99,110} tarama için **kullanılamaz**. (ii) Karar koşularından önce FOTON'un
mimarisi `src/config.py`'de bir preset olarak sabitlenir ve **config'in SHA-256'sı bu bölüme yazılır**;
karar koşuları o config ile yapılır. (iii) Karar aşamasına **tek varyant** girer. Birden fazla varyant
karar seed'lerinde koşulursa, birincil sonuç **ilk ilan edilenin**dir; diğerleri "keşifsel" etiketiyle
raporlanır ve hiçbir iddiayı desteklemez. (iv) Rapor, taranan varyant sayısını ve tablosunu verir.

**B5 (KRİTİK→önemli). Eğitim hiperparametreleri iki kolda AYNI.**
r1'in O1 tablosu "baseline'ın config'inden birebir" diye çerçevelenmişti, yani FOTON'u bağlamıyordu.
Ölçüldü: yalnızca program uzunluğunu 1.38× değiştirmek aynı modelde 4.27 puan oynatıyor.

**Kural:** lr, min_lr, warmup, β, weight_decay, grad_clip, dtype/autocast, dizi uzunluğu, tokens/adım,
`max_steps`/`max_tokens`, `eval_interval` ve `batch_size × eval_iters` **iki kolda aynıdır** ve O1
tablosundaki değerlerdir. FOTON için farklı bir değer gerekiyorsa **baseline da aynı değerle yeniden
koşulur**; tek kolda değiştirmek sapmadır ve kararı geçersiz kılar.

**B6. Eval ızgarası iki kolda aynı: `eval_interval = 250`.**
r1'in O3'ü "daha sık olabilir", O10'u "eval sıklığının artırılması değişebilir" diyordu; ızgaraların eşit
olması istenmiyordu. Val eğrisi dışbükey azalan olduğu için lineer interpolasyon `N*`'ı sistematik olarak
fazla tahmin ediyor ve yanlılık ızgara adımıyla büyüyor: Richardson tahminiyle ölçülen yanlılık
h=250'de ortalama +%0.164, maksimum +%0.280. Asimetrik ızgara FOTON'a bedava ~+0.12 puan verirdi.
**Kural:** karar koşularında iki kol da 250 adım ızgarası kullanır. Daha sık eval isteniyorsa **iki kolda
birlikte** değişir.

**B7. Tasarruf konvansiyonu tek: `s = 100 × (1 − F_foton/F_base)`.**
r1'in kendi L* bölümü (satır 64) ters paydayı kullanmıştı (%23.2 yerine %18.8), yani dosya iki
konvansiyonu birlikte içeriyordu. Bu, eşik sınırında karar değiştirir: `F_foton/F_base = 0.909` iken
O4 formülü 9.10 puan ("belirsiz") verirken ters payda 10.01 puan ("güçlü başarı") verir.
**Kural:** raporda geçen her tasarruf sayısı O4 formülüyle hesaplanır; ters payda kullanılamaz.
`src/stats.py` bu formülü uygulayan tek yerdir.

**B8. Sansür kuralının YÖNÜ düzeltildi ve tripwire'dan ayrıldı.**
r1, `N* = 2 × bütçe` ikamesini "FOTON'un aleyhine yanlı" diye tanımlıyordu; **bu yanlıştı.** Koşu 2×
bütçede L*'e ulaşmadıysa gerçek `N*` 2× bütçeden büyüktür (ya da tanımsız), dolayısıyla 2× ikame etmek
`F_foton`'u gerçeğinden KÜÇÜK, `s_i`'yi gerçeğinden BÜYÜK gösterir — yani **FOTON'un LEHİNE** yanlıdır.
Ayrıca sansürlü bir seed `s_i` ≈ −150 puan verdiği için örneklem sd'sini şişirip tripwire'ı ateşler ve
"hedefe ulaşamıyor" sonucu "varyans yüksek, daha çok seed lazım" gibi görünürdü.

**Kural:** (i) sansürlü seed'in `s_i`'si 2× ikamesiyle hesaplanır ve bunun FOTON'un lehine yanlı olduğu
raporda yazılır; (ii) **tripwire yalnızca sansürsüz seed'lerin sd'siyle** değerlendirilir; (iii) sansürlü
seed sayısı ≥ 1 ise sonuç "başarılı"/"güçlü başarı" olamaz (r1'de de böyleydi) ve sansürlü seed sayısı
≥ 2 ise sonuç doğrudan **"başarısız"**tır.

Aritmetik not: sansürlü bir seed'in `s_i`'si tanım gereği 100 × (1 − 2/0.8) = **−150 puan**dır, yani n=5'te
tek bir sansür ortalamayı ≈ −25 puana çeker ve satır zaten "başarısız" olur. (iii)'ün "başarıyı engelleme"
kısmı bu yüzden kemer + askıdır; işlevsel olan kısım (ii)'dir, çünkü o olmadan sonuç "hedefe ulaşamıyor"
yerine "varyans yüksek, daha çok seed lazım" gibi okunuyordu. `src/stats.decide` üçünü de uyguluyor ve
`tests/test_stats.py` bu okumayı regresyona bağlıyor.

**B9. 2× bütçe koşusunun LR programı.**
`lr_at` cosine'i koşunun kendi bütçesi üzerinden decay ettiği için `max_tokens`'ı 2×'e çıkarmak aynı
token sayısında LR'ı yükseltir, yani 1× ve 2× koşular aynı eğri değildir.
**Kural:** uzatma, referans bütçe üzerinden tanımlı programla yapılır: program 1× bütçede biter, sonrası
`min_lr` sabitinde devam eder. 2× koşu **yalnızca sansür belirlemek** için kullanılır; `N*` her zaman
1× bütçeli koşunun eğrisinden okunur (o eğri L*'i kesmişse).

**B10. Karar tablosunun her satırı için sonraki adım.**
r1, 6 satırın 4'ü için sonraki adımı tanımsız bırakmıştı; "nötr"/"başarısız" çıkınca kural boşluğu
yazarın lehine doldurulabilirdi. Ayrıca **vazgeçme koşulu aritmetik olarak erişilemezdi**: iki ölçek
sonucuna sahip olmak için S1'e yükselmek, yükseltme hakkı ise yalnızca "belirsiz" (yani s̄ > +2) satırından
doğuyordu — dolayısıyla S0'da "başarısız" çıksa bile "iki ardışık ölçek" şartı asla sağlanamıyordu.

| satır | sonraki adım |
|---|---|
| güçlü başarı | FOTON v0.2'ye devam; S1 isteğe bağlı (yükseltme hakkı harcanmaz) |
| başarılı | FOTON v0.2'ye devam; S1 isteğe bağlı |
| belirsiz | **tek yükseltme hakkı kullanılır:** S1 + 5 seed |
| yüksek varyans / belirsiz | 10 seed'e çıkılır (B3); sonra tablo yeniden uygulanır. Yükseltme hakkı harcanmaz |
| nötr | **yükseltme hakkı kullanılabilir** (S1 + 5 seed) — trendi görmek için. Kullanılmazsa S0 sonucu nihaidir ve vazgeçme koşulu S0 + (varsa) S1 ile değerlendirilir |
| başarısız | S0 sonucu bir "ölçek sonucu" sayılır; yükseltme hakkı kullanılabilir. Kullanılmazsa hipotez **terk edilir** (aşağıya bak) |

**Vazgeçme koşulunun erişilebilir hâli:** "iki ardışık ölçek" şartı, yükseltme hakkının kullanıldığı
durumlarda S0+S1 ile sağlanır. Yükseltme hakkı kullanılmazsa ve S0'da satır **"başarısız"** ise hipotez
tek ölçekle terk edilir (oracle şartı yine uygulanır). "Nötr"de yükseltme kullanılmazsa sonuç
"sonuçsuz" olarak kaydedilir, hipotez ne doğrulanmış ne terk edilmiş sayılır — ve bu **bir devam gerekçesi
değildir.**

**B11. Oracle'ın NULL'ı zorunlu.**
Oracle, ölçülen niceliğin kendisi üzerinde seçim yaptığı için modüller istatistiksel olarak aynı olsa bile
eğriyi aşağı çeker; %2 eşiği saf seçim gürültüsünün altında kalabilir.
**Kural:** oracle kazancının yanında **null'ı** raporlanır: aynı oracle prosedürü, modül çıktıları
birbirine **permüte edilerek** (yani seçimin bilgi taşımadığı durumda) hesaplanır. Vazgeçme koşulundaki
"oracle kazancı < %2" şartı, **null'ın üstündeki fazlaya** uygulanır: `s̄^oracle − s̄^oracle_null < %2`.

**B12. Oracle'ın granülaritesi ve kaybı önceden sabit.**
r1'de "o dizi için kaybı en küçük yapan modül" hesaplanabilir değildi (dizi başına kayıp üretilmiyor,
granülarite ve kayıp türü belirsiz).
**Kural:** oracle **dizi düzeyinde** ve **saf CE** ile tanımlıdır: her val dizisi için, FOTON'un bütün
modülleri tek tek çalıştırılır (diğer katmanlar öğrenilmiş routing'i kullanır), o dizinin token-ortalamalı
CE'si en küçük olan modül seçilir. Katman-başına açgözlü ya da tüm-model-tek-modül varyantları
**kullanılamaz**. Uygulama karar koşularından önce commit'lenir.

**B13. FLOP tanımının kapsamı: sayılmayan işlemler beyan edilir.**
Birincil FLOP tanımı eleman bazlı işlemleri (RMSNorm, RoPE, SiLU, softmax, residual, CE) ve dispatch'i
(argsort, bincount, gather/scatter) **0 sayıyor** (`docs/flops.md`). FOTON kapasitesini oraya koyarsa
birincil ölçüt onu görmez.
**Kural:** karar koşularından önce, FOTON'un **sayılmayan** her işlemi `docs/flops.md`'de listelenir ve
(i) analitik formüle eklenir, ya da (ii) profiler çapraz kontrolüyle üst sınırı ölçülüp raporlanır.
Profiler ile analitik sayım arasındaki fark **her iki kol için** raporlanır; FOTON'un farkı baseline'ın
farkından 5 puandan fazla büyükse sonuç "başarılı" ilan **edilemez** ve fark açıklanır.

**B14. `N*` kesişimi sürdürülmüş olmalı.**
r1, ilk aşağı kesişimi alıyordu ama kesişimden sonra eğrinin L* altında kalmasını istemiyordu; gürültülü
bir eğride tek bir aşağı sıçrama `N*`'ı erkene çekerdi.
**Kural:** `N*`, L*'in ilk aşağı kesişimidir **ve** kesişimden sonraki eval noktasının da ≤ L* olması
gerekir. Değilse sonraki kesişime bakılır. (B1 ve B6 gürültüyü zaten sabitliyor.)

**B15. Eğri kesişmesi kontrolü.**
Birincil ölçüt tek noktaya bakıyor; eğriler kesişirse ön-kayıt bunu görmez.
**Kural:** rapor iki kolun **tam val eğrisini** verir ve eğrilerin L* dışında kesişip kesişmediğini
söyler. Kesişme varsa bu, sonucun yanına **zorunlu bir uyarı** olarak yazılır (satır adını değiştirmez).

**B16. `quality/sec` için GA ve asimetri uyarısı.**
r1, `w̄` için güven aralığı istemiyordu ve baseline'ın (auxfree) v3'ün **en yavaş** kolu olduğunu
anmıyordu (83.68k tok/s; aux 0.01'in %2.3, aux'suz kolun %3.8 altında). Yani `w̄ > 0` şartı FOTON'un
kendi hız katkısı olmadan sağlanabilir.
**Kural:** `w̄` da `src/stats.paired` ile eşleştirilmiş %95 GA'sıyla raporlanır. Baseline'ın yapısal
yavaşlığı raporda anılır. Çelişki kuralı etiketi (`mimari başarı / sistem başarısızlığı`) **`w̄ ≤ 0`
olan her durumda** eklenir — r1'de yalnızca s̄ ≥ 10 durumunda ekleniyordu.

**B17. Baseline kimliği hash'lerle çivilendi.**
r1 "config'i ve eğrisi hash'lenmiş" diyordu ama dosyada yalnızca üç `metrics.jsonl` hash'i vardı;
`src/config.py`'deki preset tag'den sonra düzenlenebilirdi.

```
dae23220efd33a4522984ed6fd79cccdae3e3777b0a67cc3c68eb6c11ac11e42  runs/auxfree-s1337/config.json
abcf671a670a7700b87aa1e094b119ee62af636c7fc5084ca1dac7058955c48d  runs/auxfree-s1338/config.json
51fe6077510ea95de8e5636519b3b713856c955bd21e2cba97e0910c2dcad603  runs/auxfree-s1339/config.json
```
(yollar `experiments/v3-router-ablation/` altında)

Baseline'ı üreten kodun `src_sha256`'sı: `8552cc82ec52877ea4745dc1646dad60b7826fe58f5bbf2267a32e9ab134680c`
(= `v3-router-ablation` tag'inin ağacındaki `src/`; doğrulaması o tag'in açıklamasında).
**Kural:** baseline'ın yeni seed koşuları, config'i bu hash'lerdeki `model` bloğuyla **aynı** olacak
şekilde yapılır. `PRESETS["moe-auxfree"]` değişirse yeni koşular geçersizdir.

**B18. Zorunlu raporlama listesi genişletildi.**
r1'in yedi kalemi yukarıdaki sömürülerin hiçbirini görünür kılmıyordu. Rapor **ek olarak** şunları verir:
taranan FOTON varyantı sayısı ve tablosu; koşu defterindeki toplam koşu sayısı ve iptal/çöken sayısı;
her iki kolun `config.json` SHA-256'sı ve `git_head`/`src_sha256`'sı; `batch_size × eval_iters` çarpımı;
`eval_interval`; her iki kolun tam val eğrisi; sansürlü seed sayısı; `w̄` ve GA'sı; profiler-analitik FLOP
farkı (iki kol); oracle kazancı ve null'ı; ve yapılan test sayısı (O4b: "13 (1+2+10), keşifsel: k").

### C. Bu ekten sonra da açık kalan, kabul edilen sınırlar

1. **Eşik sınırındaki güç.** Gerçek etki tam +%5 iken karar ~%50 "belirsiz" verir (O9). Bu, eşiğin
   tanımının sonucudur; kapatılmadı.
2. **Baseline bir ablasyon kolu.** (A2) Havuzdaki en güçlü yapılandırma olduğu için seçildi; nedensel bir
   iddia taşımıyor.
3. **n=5 ve tek veri seti.** FOTON'un sd'si dondurulmuş kolların sd'sinden büyük çıkabilir; O9'un kötümser
   satırları bunun içindir.
4. **Ölçek basamakları arasında havuzlama yok**, trend iki noktadan okunuyor (S0, S1).

---

## Sapmalar

(Boş. Ön-kayıttan her sapma, gerekçesi ve fark edildiği tarihle buraya eklenir; yukarıdaki bölümler
düzenlenmez. EK 1 bir sapma değildir: FOTON verisi üretilmeden önce yazılmış bir amendment'tır.)
