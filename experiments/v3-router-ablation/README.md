# v3 — Router dengeleme ablasyonu (router davranışı bir regularizer artefaktı mı?)

**Soru.** MoE router'ının gözlenen davranışı (dengeli yük, uzmanlaşmış expert'ler) gerçekten öğrenilmiş bir
eşleşme mi, yoksa load-balancing regularizer'ının ürettiği bir görüntü mü? İki alt soru:

1. Aux loss'u güçlendirmek utilization'ı eşitlerken uzmanlaşmayı bozuyor mu?
2. **Sahte denge** var mı — tokenlar semantik uyum yüzünden değil, denge zorlaması yüzünden mi expert değiştiriyor?

**Kısa cevap.** (1) Bozmuyor; ama optimumun ötesine geçmek ölçülebilir biçimde zarar veriyor ve bunu
**gate dağılımını yayvanlaştırarak** yapıyor. Zararın "uzmanlaşmanın türü değişiyor (sözlükleşme)" şeklindeki
açıklaması yönü tutarlı ama çoklu karşılaştırma düzeltmesinden geçmiyor. (2) Yok: ön-kayıtlı örüntü hiçbir
koşulda tamamlanmıyor ve dengeli kolların hepsinde token↔expert eşleşmesi gerçek iş yapıyor.

**Değişen.** Dengeleme yöntemi ve gücü; tokenizer, attention, optimizer, model boyutu, veri, token bütçesi ve
LR programı v2 ile birebir aynı. Model: 7 MoE katmanı × 16 expert, top-k = 3 (112 expert), paylaşılan expert
açık, grouped dispatch, 3000 adım (token tabanlı program), 3 seed (1337/1338/1339).

> **Tasarım kusuru, baştan söylüyorum:** (d) kolu (a)'ya göre **aynı anda üç şeyi** değiştiriyor — gate tipi
> (softmax→sigmoid), aux katsayısı (0.01→0) ve seçim bias'ı (0→0.001). 2×2'nin "softmax + bias" ve
> "sigmoid + aux" hücreleri koşulmadı. Bu yüzden (d)'nin sonuçları **dengeleme yöntemine atfedilemez**;
> gate tipinin etkisi bu deneyde hiçbir yerde izole edilmemiştir. Kodda ikisi bağımsız ayarlanabilir
> (`--router_gate`, `--bias_update_rate`), yalnızca koşulmamış.

## Koşullar

| kod | koşul | nasıl | koşular |
|---|---|---|---|
| (a) | aux 0.01 (normal) | Switch aux, katsayı 0.01 + z-loss 0.001 | **v2'den yeniden kullanıldı** (`checkpoints/v2/moe-tok-s*`) |
| (b) | aux 0.001 (zayıf) | katsayı 10× düşük | yeni |
| (c) | aux yok | katsayı 0 (z-loss duruyor) | **v2'den yeniden kullanıldı** (`checkpoints/v2/moe-noaux-tok-s*`) |
| (d) | aux'suz dengeleme | DeepSeek-V3 / Kimi K2 tarzı: sigmoid gating + expert başına seçim bias'ı (γ = 0.001), bias gate değerine girmez | yeni |
| (e) | aux 0.1 (güçlü) | katsayı 10× yüksek | yeni |
| (f) | aux 1.0 (çok güçlü) | katsayı 100× yüksek | yeni, **(e) görüldükten sonra eklendi** |

(a) ve (c), v3 kodundan **farklı bir kodla** eğitildi (`src_sha256` 970c99d6… vs 8552cc82…) ve **farklı bir
koşu oturumunda** koştu. Eşdeğerlik ayrıca sınandı: bugünkü v3 koduyla (a)'nın ilk 100 adımı yeniden koşuldu
ve kayıtla farkı (7.2e-04), aynı kodun kendi tekrar zarfının (1.33e-03) **0.54 katı** çıktı — yani kod
farkının etkisi GPU tekrar gürültüsünden ayırt edilemiyor (`checks/`). Oturum farkı ise tok/s'yi karıştırıyor
(aşağıda).

## Ön-kayıt: ne gerçekten önceden sabitlendi (düzeltilmiş)

Kaydın ilk hâlinde "ölçüt listesi ve sahte denge örüntüsü koşular başlamadan yazıldı" yazıyordu.
**Bu yanlıştı** ve denetim artefaktla çürüttü. Doğrusu:

| ne | ne zaman | kanıt |
|---|---|---|
| Koşul ve seed planı (b/d/e × 3 seed) | ilk koşudan **önce** | `queues/router_ablation.json` (10:14:41), ilk koşu 10:14:44'te başladı |
| Ölçüt listesi + sahte denge örüntüsü (`docs/router_metrics.md` §1–6) | **ilk koşu sürerken** (10:14:44–10:35:17) | `runs/aux0.001-s1337/provenance.json`'un git durumunda dosya YOK; `runs/aux0.1-s1337/provenance.json`'da VAR |
| (f) aux 1.0 kolu | (e)'nin analizi görüldükten **sonra** | `queues/router_aux1.json` (13:28) > (e) analizleri (13:22–13:23) |
| `h_cond_bits` = H(expert\|token) ölçütü | (f)'nin sonucu açıklanmak zorunda kalınca | mevcut per-katman veriden türetildi, yeni koşu gerekmedi |
| Sahte denge eşiklerinin sayısal hâli | karar anında | `report_numbers.py` |
| §7 "sözlükleşme" bozulma kipi | (f)'de bulundu | `docs/router_metrics.md` §7, etiketli |

Yani §1–6 bütün analizlerden (ilk analiz 13:20) ve 12 koşunun 11'inden önce sabitlendi ve tek bir sayı
içermiyor — ama "koşular başlamadan" yazılmadı ve o sırada (b)-s1337'nin denge sayıları akıyordu.
**Sonraki adımlarda:** ön-kayıt dosyası koşudan önce commit'lenecek.

## İstatistiksel durum (bunu okumadan tabloya bakılmamalı)

n = 3 seed ile toplam **151 eşleştirilmiş test** yapılıyor (18 ölçüt × 5 koşul + seri içi çiftler).
82'si nominal p ≤ .05. Düzeltme altında:

| düzeltme | ayakta kalan | hangileri |
|---|---|---|
| yok (nominal) | 82 | — |
| **BH** (FDR .05) | 60 | val loss etkileri, gate yayvanlığı, mi_bits'in çöküş tarafı, (d)'nin permütasyon farkı |
| **Holm** (FWER .05) | **9** | **yalnızca denge ölçütleri** (yük CV / yük entropisi / top-1 kullanım entropisi / mi_bits null) |

Yani: **FWER kontrolü altında yalnızca çok büyük denge etkileri ayakta kalıyor.** val loss ve uzmanlaşma
etkilerinin hiçbiri Holm'dan geçmiyor. Aşağıdaki tablolarda **H** = Holm'da ayakta, **B** = BH'de ayakta,
işaretsiz = yalnız nominal. Yıldızsız okunması gereken sayılar bunlar.

## Sonuçlar (3 seed, ortalama ± sd)

| ölçüt | (a) aux 0.01 | (b) 0.001 | (c) aux yok | (d) aux'suz bias | (e) 0.1 | (f) 1.0 |
|---|---|---|---|---|---|---|
| val loss | 1.3231 ±0.0023 | 1.3334 ±0.0013 | 1.3383 ±0.0019 | 1.3195 ±0.0002 | **1.3207** ±0.0011 | 1.3339 ±0.0030 |
| ppl | 3.755 ±0.009 | 3.794 ±0.005 | 3.812 ±0.007 | 3.742 ±0.001 | 3.746 ±0.004 | 3.796 ±0.011 |
| tok/s (bin) | 86 ±1 | 85 ±1 | 87 ±0 | 84 ±0 | 85 ±0 | 85 ±1 |
| yük CV | 0.103 ±0.007 | 0.813 ±0.014 | 1.216 ±0.026 | 0.022 ±0.000 | 0.028 ±0.002 | **0.019** ±0.003 |
| ölü expert (/112) | 0.0 | 5.3 ±1.2 | **51.0** ±3.6 | 0.0 | 0.0 | 0.0 |
| maks pay | 1.47 ±0.02 | 3.38 ±0.09 | 4.40 ±0.33 | 1.08 ±0.02 | 1.11 ±0.05 | **1.05** ±0.01 |
| yük entropisi (top-k) | 0.9977 | 0.8791 | 0.7103 | 0.9999 | 0.9998 | 0.9999 |
| mi_bits | 2.429 ±0.023 | 1.722 ±0.051 | 1.491 ±0.061 | 2.392 ±0.048 | 2.494 ±0.020 | 2.622 ±0.030 |
| mi_bits null (taban) | 0.136 | 0.092 | 0.074 | 0.138 | 0.138 | 0.137 |
| H(expert\|token) bit | 1.459 ±0.033 | 1.279 ±0.062 | 1.187 ±0.024 | 1.547 ±0.048 | 1.441 ±0.019 | 1.307 ±0.026 |
| tutarlılık | 0.615 ±0.006 | 0.639 ±0.020 | 0.658 ±0.008 | 0.602 ±0.011 | 0.620 ±0.003 | 0.651 ±0.013 |
| tutarlılık null | 0.120 | 0.237 | 0.287 | 0.108 | 0.109 | 0.111 |
| **tutarlılık − null** | **0.495** | 0.401 | **0.371** | 0.494 | 0.511 | **0.540** |
| top-1 kullanım entropisi | 0.9721 | 0.7502 | 0.6695 | 0.9845 | 0.9838 | 0.9824 |
| seçim marjı | 0.171 ±0.004 | 0.176 ±0.002 | 0.254 ±0.007 | 0.144 ±0.001 | 0.165 ±0.002 | 0.124 ±0.005 |
| top-1 gate | 0.553 | 0.570 | 0.587 | 0.526 | 0.549 | 0.517 |
| ΔCE permütasyon | 1.170 ±0.031 | (geçersiz) | (geçersiz) | 1.065 ±0.043 | 1.201 ±0.052 | 1.129 ±0.027 |
| ΔCE rastgele | 0.916 ±0.018 | 0.857 ±0.079 | 1.033 ±0.110 | 0.817 ±0.060 | 0.932 ±0.029 | 0.902 ±0.023 |
| ΔCE ikinci-en-iyi | 1.221 ±0.031 | 1.195 ±0.123 | 1.193 ±0.122 | 1.105 ±0.100 | 1.254 ±0.055 | 1.194 ±0.030 |
| ΔCE bias'sız | — | — | — | +0.000206 | — | — |
| aux/görev gradyan oranı | 0.0004 | 0.0003 | — | — | 0.0020 | 0.0120 |
| bias ezme oranı | — | — | — | 0.186 ±0.003 | — | — |

("geçersiz" = permütasyon karşı-olgusalı ölü expert olan kollarda okunamaz; gerekçe aşağıda.
Analiz koşul başına seed başına 327.680 sabit val tokenı üzerinde.)

| (a)'ya göre eşleştirilmiş fark (%95 GA, t(2)) | (b) | (c) | (d) | (e) | (f) |
|---|---|---|---|---|---|
| val loss | +0.0103 [+0.002, +0.019] | +0.0152 [+0.006, +0.024] **B** | −0.0035 [−0.010, +0.003] | −0.0023 [−0.011, +0.006] | +0.0108 [−0.000, +0.022] |
| yük CV | +0.7101 **H** | +1.1129 **H** | −0.0811 **B** | −0.0751 **B** | −0.0840 **B** |
| ölü expert | +5.33 [+2.5, +8.2] **B** | +51.0 [+42.0, +60.0] **B** | 0 | 0 | 0 |
| mi_bits | −0.7079 [−0.83, −0.59] **B** | −0.9388 [−1.12, −0.76] **B** | −0.0378 [−0.19, +0.11] | +0.0650 [−0.01, +0.14] | +0.1930 [+0.06, +0.32] |
| H(expert\|token) | −0.1796 [−0.30, −0.06] | −0.2717 [−0.39, −0.16] **B** | +0.0876 [−0.07, +0.25] | −0.0182 [−0.10, +0.06] | −0.1518 [−0.30, −0.01] |
| tutarlılık | +0.0234 [−0.01, +0.06] | +0.0430 [+0.03, +0.06] **B** | −0.0131 [−0.04, +0.01] | +0.0047 [−0.02, +0.03] | +0.0357 [+0.00, +0.07] |
| tutarlılık null | +0.1173 **B** | +0.1672 **B** | −0.0121 | −0.0108 | −0.0090 |
| ΔCE permütasyon | — | — | −0.1050 [−0.16, −0.06] **B** | +0.0311 [−0.06, +0.12] | −0.0415 [−0.12, +0.03] |
| top-1 gate | +0.0163 **B** | +0.0340 **B** | −0.0271 **B** | −0.0041 | −0.0365 **B** |
| seçim marjı | +0.0055 | +0.0831 **B** | −0.0266 **B** | −0.0052 | −0.0462 **B** |

## Soru 1: aux'u güçlendirmek uzmanlaşmayı bozuyor mu?

**Bozmuyor. Ama optimumun ötesi zarar veriyor ve zarar gate dağılımından geliyor.**

Kaydın ilk hâlinde bütün testler (a)'ya karşıydı; oysa "optimumdan kötüleşme" iddiası (a)'ya karşı test
edilemez, çünkü (a) optimum değil. Serinin kendi içinde, **en iyi val loss'lu kola (e, aux 0.1) karşı**:

| aux | yük CV | seçim marjı | mi_bits | H(E\|token) | val loss | (e)'ye göre val loss |
|---|---|---|---|---|---|---|
| 0 | 1.216 | 0.254 | 1.491 | 1.187 | 1.3383 | +0.0176, p = .0039 **B** |
| 0.001 | 0.813 | 0.176 | 1.722 | 1.279 | 1.3334 | +0.0127, p = .0016 **B** |
| 0.01 | 0.103 | 0.171 | 2.429 | **1.459** | 1.3231 | +0.0023, p = .357 (ayrışmıyor) |
| 0.1 | 0.028 | 0.165 | 2.494 | 1.441 | **1.3207** | referans |
| 1.0 | **0.019** | 0.124 | **2.622** | 1.307 | 1.3339 | +0.0132, p = .0107 **B** |

Yani U eğrisi **şekil olarak desteklenir** (iki kol da BH'de ayakta) ama **dibi bir nokta değil plato**:
aux 0.01 ile 0.1 ayrıştırılamıyor. Yük CV monoton düştüğü için dengenin kendisi val loss'u açıklayamıyor.

aux 1.0'ın optimuma göre neyi değiştirdiği, düzeltme durumuyla birlikte:

| ölçüt | fark | p | durum |
|---|---|---|---|
| val loss | +0.0132 | .0107 | **BH'de ayakta** |
| seçim marjı (gate yayvanlaştı) | −0.0410 | .0019 | **BH'de ayakta** |
| mi_bits (token kimliği bilgisi) | +0.1280 | .0230 | yalnız nominal |
| H(expert\|token) (bağlam bağımlılığı) | −0.1336 | .0306 | yalnız nominal |
| ΔCE permütasyon | −0.0726 | .0476 | yalnız nominal |
| tutarlılık | +0.0310 | .0720 | anlamsız |

**Düzeltmeden geçen iki kalem:** kalite kötüleşiyor ve gate yayvanlaşıyor. İkincisi mekanizmayı doğrudan
veriyor: Switch aux'u `coef·E·Σ f_e·P_e`, atama sayılarının değil router'ın ortalama **olasılıklarının**
fonksiyonu, yani tepe yapmış router dağılımını doğrudan cezalandırıyor — marj seride monoton düşüyor
(0.254 → 0.124). Bu, ölçüt tercihine bağlı olmayan, formülden okunabilen bir sonuç.

**Düzeltmeden geçmeyen açıklama (sözlükleşme).** Bilgi, bağlamdan token kimliğine kayıyor gibi görünüyor:
mi_bits ve null üstü tutarlılık yükseliyor, H(expert|token) düşüyor, yönü 3 seed'in hepsinde aynı. Ama:

1. Her bir test yalnız nominal (p = .023–.048), 151 testlik havuzda düzeltmeden geçmiyor.
2. **mi_bits ↑ ile H(E|token) ↓ aynı ölçümün iki yüzü, iki kanıt değil:** `h_cond = H(expert) − mi_bits` ve
   dengeli kollarda H(expert) = 3.923 ± 0.023 bit (neredeyse sabit), dolayısıyla
   r(mi_bits, H(E|token)) = **−0.97**. Kaydın ilk hâli bunları birbirini destekleyen iki bulgu gibi sunuyordu;
   yanlıştı. Aynı sebeple `docs/router_metrics.md` §2'ye yazdığım eski güvenlik kuralı ("ikisi de yüksekse
   bağlam duyarlılığıdır") bu rejimde **cebirsel olarak imkânsız** — düzeltildi.
3. Daha sade bir rakip açıklama var ve veriyle en az o kadar uyumlu: aux 1.0 gate'i yayvanlaştırdığı için
   çıktı 3 expert'in daha eşit karışımı oluyor ve kalite doğrudan bundan düşüyor; bağlam duyarlılığı
   hikâyesine gerek yok. İkisini ayıracak ölçüm bu deneyde yok.

Val loss'u hangi ölçütün açıkladığına dair korelasyonlar kayıtta duruyor (`report_numbers.log` §6) ama
**bağımsız gözlem birimi koşul, koşu değil**: gerçek serbestlik derecesi 4 / 2 / **1**. "İşaret döndü"
ifadesi üç koşul ortalamasına dayanıyor (df = 1) ve üç noktada her monoton ilişki |r| ≈ 1 verir.
Bu yüzden korelasyon tablosu **ölçüt seçimi için uyarıdır, bulgu değildir**: hiçbir ölçüt tek başına hedef
alınamaz, çünkü işareti rejime bağlıdır.

## Soru 2: sahte denge var mı?

**Yok.** Ön-kayıtlı dört maddelik örüntü (eşikleri `report_numbers.py`'de) hiçbir koşulda tamamlanmıyor:

| koşul | tutan maddeler | okuma |
|---|---|---|
| (b) aux 0.001 | 0/4 | denge yok, uzmanlaşma da düşük |
| (c) aux yok | 0/4 | aynı, daha şiddetli |
| (d) aux'suz bias | 1, 3 (2/4) | işaretin göründüğü tek koşul; aşağıya bak |
| (e) aux 0.1 | 1 (1/4) | gerçek denge |
| (f) aux 1.0 | 1, 4 (2/4) | 2. madde **ters** yönde (uzmanlaşma arttı) → sahte denge değil |

Belirleyici kanıt, **dengeli dört kolda** expert kimliklerini tutarlı bir permütasyonla değiştirmenin
1.065–1.201 nat'a mal olması (kaybı 1.81–1.92 katına çıkarıyor). Ölçütün **sıfır noktası** artık sabit:
expert'ler birbirinin kopyasıysa ΔCE permütasyon ≈ 0 (`tests/test_router_stats.py`).

**Permütasyon testinin geçerlilik koşulu (denetimde bulundu).** Permütasyon yalnızca yük *histogramını*
sabit tutar; modül başına yük permüte olur. (c)'de 112 expert'in 51'i ölü olduğu için rastgele bir
permütasyon tokenların **~%40'ını** ideal payın %1'inden azını görmüş, yani fiilen eğitilmemiş modüllere
yönlendiriyor — o koldaki 1.161 nat "eşleşmenin değeri" değil, "eğitilmemiş modüle yönlendirmenin maliyeti".
(a)'da bu oran %0. Bu yüzden (b) ve (c) için permütasyon sayısı tablodan çıkarıldı.
**Sonuç bundan zarar görmüyor:** iddia dengeli kollar üzerine kuruluydu ve orada karışıklık yok.

**(d) hakkında çekince.** İki madde tutuyor: ΔCE permütasyon (a)'dan anlamlı düşük (−0.105, BH'de ayakta) ve
tokenların %18.6'sının top-3 kümesini bias belirliyor. Ama bias'ı çıkarmak CE'yi ne bozuyor ne düzeltiyor
(+0.000206 nat — tekrar zarfının 13 katı, yani ölçülebilir, ama permütasyon maliyetinin **5179'da biri**):
baskı, seçimin fiilen önemsiz olduğu bölgede çalışıyor. Ve rakip açıklama duruyor: (d)'nin top-1 gate'i de
anlamlı düşük (−0.027, BH'de ayakta), yani çıktı 3 expert'in daha eşit karışımı ve her seçim bozması daha az
etkiliyor. **Ölçütlerim "zayıf eşleşme" ile "yayvan gate" arasını ayırmıyor.** Ayrıca (d) üç şeyi birlikte
değiştirdiği için bunların hangisinden geldiği de bilinmiyor.

## Çöküş tarafı (aux yok)

Regularizer'ı kaldırmak uzmanlaşmayı açığa çıkarmıyor, yok ediyor: 112 expert'in 51'i ölü (%45.5), yük CV
1.216 = tam çöküş değerinin (√(E/k−1) = 2.082) %58.4'ü, mi_bits 2.429 → 1.491 (BH'de ayakta), val loss
+0.0152 (BH'de ayakta).

**Bir düzeltme:** kaydın ilk hâli (c)'de `tutarlılık`ın (a)'dan yüksek çıkmasını "artefakt, null'ı 1/E değil
1/E_canlı" diye geçiştiriyordu. Sayı yanlıştı ve düzeltme hiçbir yerde uygulanmamıştı. Artık null
hesaplanıyor (aynı satır toplamları + aynı yük dağılımıyla bağımsız atama): (c)'nin null'ı **0.287**,
(a)'nın **0.120**. Null üstü fazlada sıralama tersine dönüyor: (c) 0.371, (a) 0.495, (f) 0.540 — yani
aux'suz kolda tutarlılık *daha düşük*. Ham `consistency` koşullar arası karşılaştırılamaz;
`tests/test_router_stats.py` bunu sentetik veride gösteriyor (dengesiz + rastgele routing'de ham 0.71,
null 0.70).

## Uzmanlaşmanın niteliği (katman 4, seed 1337, lift = P(expert\|token)/P(expert))

```
(a) aux 0.01   e0: park every got on home he      e1: red felt became too were again
               e2: smiled laughed I together ,    e3: best things time new more other
               e4: Yes friend room me you         e5: Lily So then Tim She girl

(c) aux yok    e0: always he time she then Tim    e1: named red sun lived
               e3: Yes sorry you know this Tim    e2: (boş)  e4: (boş)  e5: (boş)

(d) aux'suz    e0: he they came she who was       e1: red pretty laughed sad long
               e2: ." They !" ?"                  e3: You I best One learned
               e4: too Yes every is do            e5: A So then Tim Lily

(f) aux 1.0    e0: started on is are him it       e1: time upon little pretty small
               e2: ran we then You other          e3: best again Yes more day excited
               e4: would So can told asked        e5: She He Lily got Tim Tom
```

Dengeli kollarda expert'ler okunabilir kümeler alıyor: özel adlar (a-e5, d-e5, f-e5), duygu/sıfat (a-e1, d-e1),
noktalama-diyalog (d-e2), kip fiilleri (f-e4). (c)'de üç expert tamamen boş. **Bu nitel tablo kanıt değil
örnektir** ve sözlükleşmeyi ayırt edemez: dengeli dört kolun hepsinde okunabilir kümeler var.

## Denetim: bu kayıt neyi değiştirdi

Yazıldıktan sonra 76 ajanlı bir denetimden geçti (her boyut ayrı denetçi, her bulgu ayrıca çürütülmeye
çalışıldı): **47 bulgu doğrulandı** (0 kritik, 13 önemli), 102 sayı/iddia doğru bulundu. Sonucu değiştirenler:

1. **"(d) en iyi dengeyi alıyor" yanlıştı** — en düşük yük CV ve en düşük maks pay (f)'de. Simetrik dürüstlük
   için: (f)'nin bu üstünlüğü de anlamlı değil (eşleştirilmiş (d)−(f): yük CV +0.0029, p = .232;
   maks pay +0.027, p = .139). İkisi de mükemmele yakın dengede; üçlü süpürme ("hem… hem… hem…") bir
   ödünleşmeyi tahakküm gibi sunuyordu.
2. **Ön-kayıt iddiası yanlıştı** (yukarıda düzeltildi, provenance kanıtıyla).
3. **Çoklu karşılaştırmaya tek kelime yoktu** — 151 test, Holm'da 9 ayakta.
4. **U eğrisinin tanımlayıcı testi hiç yapılmamıştı** — bütün testler (a)'ya karşıydı.
5. **mi_bits ↑ / H(E|token) ↓ aynı ölçüm** (r = −0.97), iki kanıt değil.
6. **tok/s farkı oturum etkisi** — anlamlı çıkan farkların hepsi v2↔v3 koşu oturumu sınırını geçiyor.
7. **Permütasyon testi (c)'de karışık** — ölü expert'ler yüzünden.
8. **`consistency` null'ı yanlış yazılmıştı** ve uygulanmamıştı; **`expert_diversity`'nin null'ı tavan**
   (eğitilmemiş expert'ler 1.000, eğitim bu ölçütü düşürüyor) — yani yüksek değeri hiçbir zaman kanıt değil.
9. **`|kosinüs| ≤ 0.0042` hiçbir düzeyde geçerli üst sınır değildi** — koşu düzeyinde 0.0060, katman
   düzeyinde 0.0195. Sonuç ("neredeyse dik") 0.0195'te de ayakta ama sayı yanlıştı.
10. **Gerçek kod hatası:** `next` karşı-olgusal kipi E < 2k'da sessizce yanlış eşleşme üretiyordu
    (bu deneyde tetiklenmiyor); `paired()` seed yerine liste konumuna göre eşliyor ve t kritik değerini
    sabit alıyordu; `delta_nobias` replika ile gerçek forward'ı karşılaştırıyordu (etki 2e-4, tolerans 2e-3).
11. **`checks/` içindeki "v2 kodu src hash"i hiçbir artefaktla eşleşmiyordu** ve v2 kaynağı kurtarılamaz
    olduğu için parmak izi eşitliğinin bir tarafı bağımsız doğrulanamıyor — yerine yörünge yeniden
    üretimi eklendi.

Düzeltilen araçların testleri: `tests/test_router_stats.py` (15 test). Ayrıca 18 checkpoint **iki kez**
analiz edildi: ilk koşu `analysis_ilk_kosu/`, kanonik koşu `analysis/`. Tekrar zarfı (GPU gürültüsü)
`load_cv` 1.2e-06, `mi_bits` 3.8e-06, `ΔCE permütasyon` 5.6e-06 — raporlanan bütün etkiler bunun en az
10 katı üstünde.

## Sınırlar

- **FWER altında yalnızca denge etkileri ayakta.** Kalite ve uzmanlaşma sonuçları FDR düzeyinde geçerli;
  sözlükleşme açıklaması yalnız nominal. n = 3 seed bu deney için az.
- **(d) üç şeyi birlikte değiştiriyor**; gate tipi hiçbir yerde izole edilmemiş.
- **tok/s karşılaştırılamaz** (oturum karışıklığı). Söylenebilecek tek şey: bütün kollar 84–87 bin tok/s.
- **Gradyan çekişmesi** tek batch ve eğitimin sonunda ölçüldü, eğitim boyunca ortalama değil.
- **Uzmanlaşma ölçütlerinin tamamı top-1 seçimden**; top-k'nın diğer iki seçimi görülmüyor.
- `h_cond_bits` ve `mi_bits`'in **mutlak** değerleri token bütçesine bağlı (plug-in yanlılığı), bu yüzden
  farklı `--batches` ile üretilmiş analizler karşılaştırılamaz. Kollar arası karşılaştırma geçerli çünkü
  bütçe aynı ve taban (`mi_bits_null` ≈ 0.137 bit) sinyalin ~%6'sı.
- **Tek ölçek, tek veri seti, token düzeyi routing.** Kırılma noktasının yeri (plato 0.01–0.1, kötüleşme
  1.0'da) bu ölçeğe ait.

## FOTON'a taşınmaz

Taşınabilir olan **prensip**:

1. Denge ile uzmanlaşma zorunlu bir ödünleşim değil; bu rejimdeki gerçek risk çöküştü, aşırı regularizasyon
   değil. Ama aşırı regularizasyonun da bir bedeli var ve bedel **gate dağılımının yayvanlaşmasından** geliyor.
2. Denge ölçütleri (yük CV, ölü expert) bir hedef olarak yeterli değil: dengeli kollar arasında yük CV
   val loss ile ilişkisiz. Bir dengeleme ayarını "en dengeli" olduğu için seçmek, bu deneyde kaliteyi
   seçmekle aynı şey değildi.
3. Ölçüt seçerken **null'ı olan ölçüt** kullan: `consistency` ve `mi_bits` kendi bağımsızlık tabanlarına
   göre okunmalı, `expert_diversity`'nin null'ı tavan olduğu için uzmanlaşma kanıtı olamaz, ve token↔expert
   eşleşmesinin değerini ölçen tek sağlam araç **permütasyon karşı-olgusalı** (ölü expert yoksa).
4. Kayba müdahale eden dengeleme (aux) gate dağılımını da deforme ediyor; seçime müdahale eden (bias) etmiyor.
   **Bu bir hipotez, bulgu değil** — (d) kolu gate tipini de değiştirdiği için nedensel olarak ayrışmıyor.

Katsayılar (0.01 / 0.1 / 1.0 / γ = 0.001) **taşınmıyor**. FOTON'un üç modülü sequence düzeyinde top-1
çalışıyor; token düzeyinde 16 expert ile aynı rejim değil. FOTON kendi router regularizasyon ablasyonunu
ayrıca isteyecek — ve o ablasyonda eksik 2×2 hücreleri baştan koşulmalı.

## Dosyalar

| yol | ne |
|---|---|
| `summary.json` | tüm tablo + (a)'ya göre farklar + seri içi farklar + çoklu karşılaştırma |
| `analysis/{koşul}-s{seed}.json` | koşu başına tam router analizi (18 dosya), per-katman ve null'lar dahil |
| `analysis_ilk_kosu/` | aynı 18 analizin ilk koşusu — tekrar zarfı kanıtı |
| `reanalyze.log` | kanonik analiz koşusunun tam logu (18 checkpoint) |
| `runs/{koşu}/` | 12 yeni koşunun `config.json`, `metrics.jsonl`, `provenance.json`, `train.log` |
| `report_numbers.py`, `report_numbers.log` | metinde geçen türev sayıların artefaktlardan yeniden hesabı |
| `queues/router_ablation.{json,log}`, `queues/router_aux1.{json,log}` | koşu kuyrukları ve logları |
| `checks/` | (a)/(c)'nin yeniden kullanılabilirliği: yörünge yeniden üretimi + CPU parmak izi (ve sınırları) |
| `environment.txt`, `checksums.txt` | ortam ve checkpoint SHA-256'ları (ağırlıklar git'te değil) |
| `../../docs/router_metrics.md` | ölçüt tanımları, yorum tuzakları, iki bozulma kipi |
| `../../docs/v3_router.png` | 5 panel: denge ↔ uzmanlaşma, val loss ↔ H(expert\|token), eğitim boyunca CV |
| `../../tests/test_router.py`, `test_router_metrics.py`, `test_router_stats.py` | 37 test |

Yeniden üretmek için:

```
.venv\Scripts\python scripts\run_queue.py experiments\v3-router-ablation\queues\router_ablation.json
.venv\Scripts\python scripts\run_queue.py experiments\v3-router-ablation\queues\router_aux1.json
.venv\Scripts\python scripts\compare_router.py --analyze --analysis <yeni_dizin> --out <yeni>.json --plot <yeni>.png
.venv\Scripts\python experiments\v3-router-ablation\report_numbers.py
```

(`--analysis`/`--out`/`--plot`'a yeni yollar verin; varsayılanlar bu dondurulmuş kaydın üstüne yazar.)
12 koşu × ~20.5 dk ≈ 4.1 saat eğitim + 18 checkpoint × 2 analiz.
