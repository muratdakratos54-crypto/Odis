# 07 — FOTON v0.2 SONUÇ RAPORU

Tarih: 2026-09-19. Ön-kayıt: `PREREGISTRATION_V02.md`, tag `foton-v02-prereg` (gövde değişmedi; Sapmalar S1–S11 sonda).
Bu rapordaki sayılar bölüm 9'daki artifact'lardan gelir; tek istisna duman testi satırıdır (bölüm 2: artifact'ı yok).
Bu rapor, bağımsız bir doğrulama turundan (kör yeniden hesap + sayı denetimi + düşmanca hakem) geçirilip düzeltilmiştir.

## 0. SONUÇ

**Ön-kaydın V6 yolu: "router bu tasarımda eğitilemedi". Karar kampanyası KOŞULMADI; karar tablosundan bir satır adı
ÇIKMADI** (`decide_primary()` hiç çağrılmadı; "başarılı / başarısız / belirsiz" satırlarının hiçbiri bu deneyin sonucu
değildir). Bu, FOTON aleyhine bir sonuçtur ve öyle yazılıyor. Ön-kayıt gereği hipotez ne doğrulanmış ne terk edilmiş
sayılır: vazgeçme koşulu (`abandon()`) değerlendirilemedi (bölüm 8).

"Router bu tasarımda eğitilemedi" ön-kaydın V6 PROSEDÜR etiketidir ve gözlem olarak doğrudur: dizi-başı router entropisi
dokuz tarama koşusunun hiçbirinde, hiçbir log aralığında 0.98'in altına inmedi. Etiket, izinli 3 çare çevriminin
1.'sinden sonra, sahibin kapı kararıyla uygulandı (Sapma-S10 / S11). Sebep incelemesi V6'nın varsaydığı mekanizmayı
("top-1 gate'in klasik gradyan sorunu") DESTEKLEMİYOR: gate'in sinyali sağlamdı; eksik olan, öğrenilecek kararlı bir hedefti.

**İddianın kapsamı — bundan geniş bir iddia KURULMUYOR:** *bu tasarımda* (dizi başına prefix router, k = 64; straight-through
top-1 gate; 3 paylaşılan + 2 yönlendirilen + 3 paylaşılan katman; T = 512'de ~128.4 bağlantıya eşitlenmiş üç dikkat
grafiği), *bu bütçede* (42.5M aktif / 55.3M toplam parametre; tarama tier'ı, 143M token'lık bütçenin ilk yarısı; 3 seed ×
3 regularizasyon koşulu), **prefix'ten tanınabilir bir yönlendirme öğrenilemedi.**

Sonucun daha güçlü ifadesi bölüm 5'teki oracle tavanından gelir (KEŞİFSEL; ön-kayıtlı test değil): **bu koşuların ürettiği
modüllerle (rastgeleye yakın bir yönlendirmeyle eğitilmiş, %50 bütçe) eşiğe yaklaşan bir kazanç yoktu, ve prefix'ten
öğrenilebilir kısmı ~0'dı.** Her dizide en iyi modülü geriye dönük MÜKEMMEL seçen bir router'ın en iyi sabit-tek-modüle
üstünlüğü **1.34 puan**dır (eğim ve taban seçimine göre 0.95–2.17; başarı eşiği **+5**); bunun seçim gürültüsünü aşan
kısmı en çok **0.28 puan**; alan düzeyinde öğrenilebilir kısmı **~0.04 puan**. Tavan, kararlı bir yönlendirmeyle BİRLİKTE
eğitilen modüllerin ne kadar ayrışacağını SINIRLAMAZ (bölüm 5). Kilit cümle: **sinyal var, hedef yok** — prefix alanı %97
doğrulukla taşıyor, en iyi modülü şans düzeyinde (%33–42) taşıyor.

## 1. Önce okunması gereken iki eşitsizlik (kolların karşılaştırılabilirliği)

**(a) FLOP asimetrisi — iki sayım ters yöndedir.** Analitik aktif FLOP/token (birincil ölçütün paydası, ön-kayıt gereği):
D 267,411,456 · M 267,755,520 · H 265,834,602 · X 265,838,610. Analitik sayımda FOTON, M'den **−%0.72** KÜÇÜKTÜR
(maskelenen (i, j) çiftleri sayılmaz). Ama koşan çekirdek (yoğun SDPA + boolean maske) T×T skorun tamamını üretir:
çekirdek düzeyi ÜST SINIRDA (yönlendirilen 2 katman yoğun T×T sayılırsa; `docs/flops.md` §8.5) FOTON, M'den **+%1.04**
BÜYÜK olurdu. Birincisi FOTON'un lehine, ikincisi aleyhinedir; iki sayım arasındaki fark en çok ~1.8 puandır (1.76) ve
başarı eşiğinin üçte biri (1.67) mertebesindedir. Karar koşuları koşulmadığı için bu asimetri hiçbir karar satırını
etkilemedi; v03'te payda tanımı koşulardan ÖNCE tek sayıya bağlanmalıdır. O-X1 mikrobenchmark'ı: maskeli yol, nedensel
hızlı yola göre ~%6 yavaş (A / B / C = 603.6k / 602.2k / 604.2k tok/s, tam nedensel 641.0k); üç modülde eşit (%0.33 içinde).
Sapma-S9'un 07'ye bıraktığı bileşen-bileşen profiler karşılaştırması YAPILMADI: v01 B13'ün 5 puan kuralı yalnızca
"başarılı" satırını korur, satır üretilmediği için tetiklenmedi; S9'daki sayaç artefaktı (M ~−%19, FOTON ~+%7) v03'e
açık iş olarak taşınır.

**(b) Kalan bağlantı eşitsizliği FOTON'un ALEYHİNEDİR.** Yönlendirilen iki katmanda modüller token/head başına
128.17 / 128.50 / 128.38 bağlantı görür (nedensel maske altında T = 512'de maske kurulup SAYILARAK; hedef 128.4 ± 0.4);
D ve M'in her katmanı 256.5 görür. Üç modül birbirine eşitlendi, baseline'a değil.

## 2. Ne koşuldu, ne koşulmadı

| tier | planlanan | koşulan | not |
|---|---|---|---|
| duman testi (kol X, preset `foton-x`, birkaç yüz adım) | — | 1 | 06/1 sırasında; **KAYITSIZ**: çıktısı saklanmadı, koşu defterinde yok, adım sayısı artifact'tan doğrulanamıyor |
| V7 router regularizasyon taraması (kol X, %50 bütçe) | 9 | **9** | 3 koşul × seed {101, 102, 103}; 2.4 GPU saat; tek oturum `2026-09-19T02:46:37` |
| karar kampanyası (D, M, H, X × seed {11..55}) | 20 | **0** | V6 + sahibin kararı (aşağıda) |
| zorunlu 2× koşular, S1 ölçeği, seed yükseltmesi | koşula bağlı | **0** | karar satırı olmadığı için tetiklenmedi |

Taranan FOTON varyantı sayısı: **3** (V7'nin üç koşulu; hepsi raporlanıyor, bölüm 3). Koşu defteri
(`run_ledger.py --report`): toplam 32 deneme (+1 deftere yazılmamış duman testi), iptal 2, çöken 0 (çöktü 0 + OOM 0 +
NaN 0), sürdürülen 0; iki iptal baseline yeniden koşumuna aittir (Sapma-S2), tarama koşularının dokuzu da ilk denemede
bitti. H kolu HİÇ koşulmadı: X ile H'yi ayıran birincil-2 (heterojenliğin katkısı) hakkında bu deney bir şey söylemez.

**Sahibin kapı kararı (2026-09-19; kapı raporundan SONRA, tarama val loss'u görülmeden, oracle tavanı hesaplanmadan
ÖNCE): (A) — kampanya koşulmayacak.** Gerekçesi, kendi cümlesiyle: *"ön-kaydı, cevap hoşumuza gitmediğinde bile bağlasın
diye kurduk. İlk olumsuz sonuçta saparsak geçen haftanın tamamı tiyatro olur."* (B) kampanyayı yine de koşmak sahibince
reddedildi: *"V6-G gereği çıkacak 'başarı' satırı zaten yönlendirme iddiası olarak kurulamaz. 19 GPU saati, kullanılması
yasak bir sonuç için harcanır."* (C) donmuş bir düğmeye dokunan çare reddedildi: *"kendi teşhisin eksik olanın router
kapasitesi değil modüllerin ayrışması olduğunu söylüyor."* Kayıt: Sapma-S11.
*Raporun notu:* (A)'nın bağlayıcı dayanağı V6'nın kendisidir ("karar koşuları koşulmaz"). V6-G'nin mekanik ölçütleri
(`router_dead ≥ 2` / son 500 adımda marjinal entropi < 0.1 / `D3 = I(alan; seçilen modül) < 0.05 bit`) karar koşusu
olmadığı için DEĞERLENDİRİLMEDİ; tarama verisinde çökme yok ve I(alan; seçim) adım 2188'de 0.025–0.215 bit (9 koşunun
6'sında ≥ 0.05): V6-G'nin harfiyen tetikleneceği bu veriyle gösterilemez.

## 3. V7 taraması — üç koşulun üçü de

Mekanik seçim (`v7_ablation.json`; val loss'a BAKMAZ): üç koşul da `router_dead = 0` süzgecini geçti; dizi-başı entropinin
pencere ortalaması (seed ortalaması) **0.9952** (regularizasyon yok) / **0.9972** (entropi bonusu) / **0.9980**
(load-balancing aux) → kazanan: regularizasyon yok. Ama **V6'nın ters-arıza uyarısı** (`router_entropy_per_seq > 0.9`,
adım 450 / 475 / 500'ün üçünde de) **dokuz koşunun dokuzunda ateşlendi**; entropi hiçbir koşuda, hiçbir log aralığında
0.98'in altına inmedi (en düşük 0.9890), yani uyarı Sapma-S4'teki okuma seçimine duyarlı değildir. Çökme YOK (ölü modül 0,
marjinal entropi ≥ 0.91): arıza, ön-kaydın "daha olası" dediği ters arızadır — router kararsız (softmax'ın en büyük
olasılığı 0.35–0.37), seçimi en iyi modülle ilişkisiz.

| koşu | erken uyarı | Hseq (pencere) | Hmarj (pencere) | maks ölü | router = en iyi modül (şans 0.33) | aynı dizinin modül değiştirmesi (rastgele 0.67) | I(alan; seçim) bit @2188 | val @2188 | `code_dirty` |
|---|---|---|---|---|---|---|---|---|---|
| v7-x-s101 | ATEŞLENDİ | 0.9955 | 0.9920 | 0 | 0.362 | 0.692 | 0.043 | 3.1254 | False |
| v7-x-ent-s101 | ATEŞLENDİ | 0.9973 | 0.9934 | 0 | 0.320 | 0.682 | 0.102 | 3.1260 | **True** |
| v7-x-lb-s101 | ATEŞLENDİ | 0.9976 | 0.9993 | 0 | 0.359 | 0.699 | 0.135 | 3.1215 | **True** |
| v7-x-s102 | ATEŞLENDİ | 0.9944 | 0.9862 | 0 | 0.338 | 0.741 | 0.039 | 3.1334 | False |
| v7-x-ent-s102 | ATEŞLENDİ | 0.9956 | 0.9873 | 0 | 0.344 | 0.582 | 0.110 | 3.1310 | False |
| v7-x-lb-s102 | ATEŞLENDİ | 0.9974 | 0.9986 | 0 | 0.361 | 0.710 | 0.215 | 3.1382 | False |
| v7-x-s103 | ATEŞLENDİ | 0.9956 | 0.9915 | 0 | 0.372 | 0.626 | 0.129 | 3.1365 | False |
| v7-x-ent-s103 | ATEŞLENDİ | 0.9986 | 0.9948 | 0 | 0.348 | 0.577 | 0.171 | 3.1339 | False |
| v7-x-lb-s103 | ATEŞLENDİ | 0.9990 | 0.9996 | 0 | 0.323 | 0.675 | 0.025 | 3.1363 | False |

Val loss (pozisyon 64..511, adım 2188) burada yalnızca V11'in "üç koşulun hepsinin tarama sonucu raporlanır" şartı için
duruyor; üç koşul arasında ayırt edilebilir bir fark yok (aynı seed'de koşullar arası fark 0.0045 / 0.0073 / 0.0026 nat;
koşul içinde seed'ler arası sd 0.004–0.009).

## 4. Sebep incelemesi (V6: "sebep incelenir") — uygulama hatası BULUNMADI

Bu bölümdeki ve bölüm 5'teki ölçümlerin hiçbiri ön-kaydın D1–D11 listesinde değildir: V5 gereği hepsi **KEŞİFSEL**dir.
Kilit sayılar, dokuz koşunun aralığıyla (`router_diag/`, `v7_router_diag.json`):

1. **Gate sinyali doğru.** Router'ın görev loss'undan aldığı tek sinyal dizi başına `dL/dg`'dir. Her dizi her modüle
   zorlanarak fp32'de ölçüldü (128 val dizisi/koşu): en negatif `dL/dg`, dizilerin **%77–95**'inde GERÇEKTEN en düşük
   kayıplı modülü gösteriyor (şans %33). Ölçüm betiği sonlu farkla sınandı.
2. **bf16 sebep değil.** Eğitim gate çarpımını bf16'da yapıyor; CUDA'da bf16 gradyanı fp32 ile 0.9999 korele, argmin uyumu
   %99.2 (bir koşuda ölçüldü: `v7-x-s101`).
3. **Sinyal var, hedef yok.** Router'ın GÖRDÜĞÜ girdi (havuzlanmış prefix vektörü; PCA-64 + L2 doğrusal prob, 5 katlı
   çapraz doğrulama, n = 1280, adım 2188): **alan %97** (96.6–97.7); **en iyi modül %33–42** — 9 koşunun 8'inde
   çoğunluk-sınıfı tabanının (%35–44) ALTINDA. 20 permütasyonluk p değerleri 0.048–0.81 (düzeltilmemiş); iki koşuda
   (ikisi de seed 103) prob bütün permütasyonların üstünde (p = 1/21, erişilebilir en küçük değer), fark ≤ 3 puan. Bu
   modüllerle ve bu girdiyle doğrusal bir prob en iyi modülü işe yarar düzeyde öngöremedi; router'ın init'i, sıcaklığı ya
   da gürültüsü BU hedefi öngörülebilir kılmaz. Bu düğmelerin eğitimin gidişini (dolayısıyla modülleri ve hedefi)
   değiştirip değiştirmeyeceği ÖLÇÜLMEDİ.
4. **Modüller arası fark küçük ve kararsız.** Dizi içinde en iyi ile en kötü modül arasındaki fark **~0.010 nat**
   (0.0090–0.0118); modül ORTALAMALARI arasındaki fark 0.0006–0.0021 nat; en iyi modül alandan öngörülemiyor
   (I(alan; en iyi modül) = 0.006–0.016 bit; bağımsızlık altındaki tahmin yanlılığı 0.002). Aynı sabit val dizisinin en iyi
   modülünün ardışık eval noktalarında AYNI kalma oranı adım 1000'e kadar şans düzeyinde (0.34), adım 2000'de 0.39–0.43
   (eşit 250 adımlık aralıklar); son nokta 0.43–0.47'dir ama aralığı 188 adımdır ve öncekilerle bire bir karşılaştırılamaz.
   **Koşular arasında tekrarlanan bir tercih de yok:** aynı dizinin en iyi modülü iki farklı koşuda 0.338 oranında
   örtüşüyor, bağımsızlık altında beklenen 0.338 (36 koşu çifti; farklı seed'li 27 çiftte 0.336 / 0.338).
5. **Seçim kararsız ve en iyi modülle ilişkisiz.** Router'ın seçimi en iyi modülle 0.32–0.37 oranında örtüşüyor (şans
   0.33); sabit val dizilerinin %58–74'ü ardışık eval noktaları arasında modül değiştiriyor (rastgele seçimde 0.67; dokuz
   koşunun ortalaması her eval noktasında 0.64–0.70). Seçim herhangi bir ANDA alanla ilişkilidir (I(alan; seçim) eval
   noktaları arasında 0.007–0.34 bit oynuyor; adım 2188'de 0.025–0.215) ve anlık yük dengesiz olabilir, ama bu ilişki
   zamanda tutunmuyor: "rastgele" değil, "kararsız".

## 5. ORACLE TAVANI — kazanılacak ne vardı? (`oracle_ceiling.json`; mevcut veriden, yeni koşu yok; KEŞİFSEL)

Ham dökümler sabit 1280 val dizisinin her biri için üç modülün ölçüt-pozisyonu CE'sini taşır; her dökümün oracle ve
routed ortalaması koşunun logladığı eğrilerle sağlandı; bütün tablo bağımsız bir kör yeniden hesapla birebir doğrulandı.
Adım 2188'de (bütçenin %50'si). **"9 koşu" = 3 seed × 3 koşul: aynı seed'in koşuları init'i ve veri sırasını paylaşır,
bağımsız tekrar sayısı 3'tür; [min, maks] betimleyicidir, güven aralığı DEĞİLDİR.**

| büyüklük | nat (ort [min, maks]) | puan, L* yerel eğimi −0.3346 (ön-kayıt V9) | puan, geç sekant −0.2385 (V9 duyarlılık ucu) | puan, koşunun kendi yerel eğimi (−0.44…−0.49) |
|---|---|---|---|---|
| **tavan: mükemmel dizi-düzeyi seçim − en iyi sabit tek modül** | 0.0045 [0.0037, 0.0053] | **1.34 [1.10, 1.58]** | 1.87 [1.54, 2.21] | 0.95 [0.81, 1.11] |
| aynı, seçimsiz tabanla (üç modülün ortalaması) | 0.0052 [0.0044, 0.0059] | 1.55 [1.32, 1.73] | 2.17 [1.85, 2.42] | 1.10 [0.95, 1.22] |
| aynı, router'ın kendi seçimine karşı | 0.0050 [0.0041, 0.0056] | 1.49 [1.23, 1.65] | 2.09 [1.72, 2.30] | 1.06 [0.88, 1.18] |
| tavanın seçim gürültüsünü aşan kısmı (B11 null'ı − oracle) | 0.0009 [0.0007, 0.0012] | **0.28 [0.22, 0.36]** | 0.39 [0.30, 0.51] | 0.20 [0.15, 0.26] |
| alan düzeyinde en iyi politika − en iyi sabit tek modül (örneklem içi) | 0.00015 [0, 0.0006] | **0.04 [0, 0.17]** | 0.06 [0, 0.24] | 0.03 [0, 0.12] |
| **başarı eşiği** (ön-kayıt KARAR TABLOSU) | 0.0172 | **+5** | +5 (0.0122 nat) | — |
| vazgeçme koşulunun oracle eşiği (null'ın üstünde) | — | +2 | +2 | — |

**Bu tablo neyi neyle karşılaştırıyor.** +5 eşiği, `L*`'e ulaşmak için gereken aktif FLOP'ta **M'e karşı**, `N*` tabanlı
tasarruftur. Tavan ise X'in KENDİ sabit-modül tabanına karşı, nat'tan eğimle çevrilmiş bir VEKİLDİR. Yan yana konmaları
ölçek içindir; aynı nicelik değildirler. Aynı şekilde +2 satırı vazgeçme koşulunun oracle bacağını SAĞLAMAZ: o bacak
karar koşularının eğrilerinden, `N*` tabanlı hesaplanır ve hesaplanamadı (bölüm 7).

**Yöntem seçimleri — açıkça.** (i) En iyi sabit modül aynı val dizilerinde GERİYE DÖNÜK seçildi (olabilecek en güçlü
taban; tavanı küçültür); seçilen modülün kimliği eval noktaları ve koşular arasında değişiyor (adım 2188'de dokuz koşuda
M0 ×7, M2 ×2; adım 2000'de M0 ×5, M2 ×4) — gerçek bir sabit mimari bu seçimi yapamazdı; seçimsiz taban 1.55'tir.
(ii) B11 null'ı, Sapma-S4'ün "tartışılabilir" diye işaretlediği pozisyon-başı permütasyon okumasıdır (tek sabit
permütasyon; belirsizliği hesaplanmadı); "null'ı aşan kısım" yapısal pay için bir ÜST sınırdır. Seçim gürültüsü okumasının
null'a dayanmayan bağımsız desteği: koşular arası en iyi-modül örtüşmesi şans düzeyindedir (bölüm 4, madde 4).
(iii) L* eğimi (adım ~3500) adım 2188'deki bir nat farkına uygulanıyor; bu, nat tavanının `L*`'e kadar değişmeyeceğini
varsayar — bu yüzden üç eğim birlikte verildi. (iv) Koşu başına bootstrap %95 aralığı (ör. `v7-x-s101`: 1.42
[1.32, 1.53]) yalnızca val dizisi örneklemesini kapsar (model sabit); seed'ler arası yayılım bundan büyüktür: seed
ortalamaları 1.38 / 1.47 / 1.18 puan (sd 0.15). (v) İkiye-bölmeli alan politikasının (modül bir yarıda seçilir, öbür
yarıda ölçülür) en iyi sabit modüle göre farkı 9 koşuda −0.0006 … +0.0002 nat'tır (ortalama −0.00005 nat ≈ 0.02 puan;
5 koşuda kötü, 4 koşuda iyi; en büyük kazanç `v7-x-ent-s102`'de ≈ 0.19 puan): alan düzeyinde ölçülebilir büyüklükte bir
kazanç yok.

**Okuma.** Tavan eşiğin ~dörtte biri (duyarlılık uçlarında beşte biri ile yarısı arası); tavanın büyük kısmı seçim
gürültüsüdür (üç modül istatistiksel olarak aynı olsa bile "her dizide en küçüğü seç" eğriyi aşağı çeker); gürültünün
üstünde kalan kısım en çok 0.3 puan. **Seyir:** nat cinsinden tavan küçülüyor (dokuz koşunun ortalaması: adım 250 →
0.0085, 500 → 0.0068, 1000 → 0.0065, 1500 → 0.0055, 2000 → 0.0047, 2188 → 0.0045; null'ı aşan kısım 0.0059 → 0.0009).
Ama val eğrisi tavandan hızlı düzleştiği için PUAN cinsinden (her eval noktasının kendi yerel eğimiyle) seyir yataydır:
tavan adım 500 → 0.81, 1000 → 0.60, 1500 → 0.93, 2000 → 0.99, 2188 → 0.95 puan; null'ı aşan kısım adım 750'den beri
~0.2 puan (0.26 → 0.16 → 0.21 → 0.21 → 0.21 → 0.21 → 0.20). İki birimde de büyüme işareti yok; puan cinsinden küçülme de yok.

**Tavanın SINIRLADIĞI ve SINIRLAMADIĞI şey — bu ayrım iddianın parçasıdır.** Tavan, rastgeleye yakın bir yönlendirmeyle
eğitilmiş modüller içindir. "Daha iyi bir ROUTER, bu modüllerle, en çok ne kazanırdı" sorusunu sınırlar: +1.3 puan, o da
hedefleri görerek. "Kararlı bir yönlendirmeyle BİRLİKTE eğitilen modüller ne kadar ayrışırdı" sorusunu sınırlamaz
(tavuk-yumurta: seçim kararsız olduğu için modüller her türden veriyi gördü). Eldeki iki gözlem — modüllerin ağırlıkta da
ayrı olması (her modül kendi 2 bloğuna sahiptir) ve alan düzeyinde en iyi politikanın en iyi sabit modülü geçememesi —
yalnızca şunu gösterir: kararlı bir yönlendirme OLMADAN, grafik farkı tek başına alan düzeyinde ya da koşular arasında
tekrarlanan bir tercih üretmedi. Birlikte-eğitim yolunu sınırlamazlar; o yol `v03_design_notes.md`'deki R2 ile sınanır.
Sonucun ifadesi "router öğrenemedi" ile bitmez: **bu modüllerle, mükemmel seçimde bile kazanılabilecek olan eşiğin
~dörtte biriydi ve prefix'ten öğrenilebilir kısmı ölçülemeyecek kadar küçüktü.**

**Ölçek için eşit token bağlam sayısı (V11-ii; `screening_context.json`; KARAR DEĞİL).** Adım 2000'de val loss: D 3.1198
(n = 5), M 3.0682 (n = 5), X 3.1701 (n = 3; regularizasyon yok). X, eşit token'da D'nin **0.050 nat**, M'in **0.102 nat**
GERİSİNDEDİR. Nat farkı adımla küçülüyor (D'ye: adım 500'de 0.149, 1000'de 0.089, 1500'de 0.064), ama bu düzleşen eğrinin
etkisidir: birincil ölçütün biriminde fark KAPANMIYOR — X, aynı kayba ulaşmak için adım 750'den 2188'e kadar D'nin
**~1.13 katı**, M'in **~1.21–1.26 katı** token harcıyor ve oran yatay (D'ye karşı 1.128 → 1.125; M'e karşı 1.206 → 1.262).
Aynı adımdaki oracle tavanı 0.0047 nat'tır: mükemmel yönlendirme D'ye olan açığın ~%9'unu, M'e olan açığın ~%5'ini
kapatırdı. Sınırlar: V11-ii yalnızca D için tanımlıdır, M karşılaştırması keşifseldir; seed'ler eşleştirilmemiş (tarama
101–103, baseline 11–55), n = 3'e karşı 5, yalnızca bütçenin ilk yarısı, payda farkı (bölüm 1a) hesaba katılmadı. Bu
sayılar "D'yi / M'i yenemedik" cümlesi kurmaz — kampanya koşulmadı — yalnızca tavanın yanına ölçek koyar: açık, tavandan
bir mertebe büyüktür.

## 6. Yöntem kayıtları

* **`code_dirty = True` olan iki tarama koşusu: `v7-x-ent-s101` ve `v7-x-lb-s101`.** Başladıkları anda çalışma ağacında
  henüz commit'lenmemiş TEŞHİS betikleri vardı (`scripts/router_diag.py`, `router_grad_diag.py` ve testleri); ölçümü üreten
  kod aynıdır: `src/` özeti dokuz koşuda da `f94e548d211a`. Kuyruk sürerken commit atıldığı için dokuz koşu DÖRT farklı
  `git_head` taşır (`90a362a` ×1, `2cfaa26` ×2, `680abd8` ×1, `f3e6a10` ×5). Tarama tier'ı SIKI değildi ve bu koşuların
  hiçbiri karar verisi değildir. **Kural (v03'e de taşınır): karar koşularında `code_dirty = False` ZORUNLUDUR,
  istisnasız.** Doğrulama turu mekanizmada bir BOŞLUK buldu: SIKI kuyruk `code_dirty`'yi yalnızca kuyruk BAŞINDA
  denetliyor, koşular arasında yalnızca `src/` özetine bakıyordu — yani taramadaki olay SIKI kuyrukta da yaşanırdı.
  Boşluk bu raporla birlikte kapatıldı: `run_queue.py` artık SIKI kuyrukta `code_dirty`'yi HER koşudan önce yeniden
  denetler ve kirlenmişse sıradaki koşuyu başlatmaz (test + mutant Y11). Açık kalan: `make_campaign_queue.py` sıkılığı
  yalnızca `--strict` verilince yazar; karar/S1 kuyruğunda zorunlu kılınması v03'e bırakıldı. Ders: kuyruk koşarken depoda
  teşhis betiği bile yazılmamalı.
* **Kapı raporu val loss'a kördü.** Tarama koşularının val loss'u birincil ölçütün kendisidir; "kampanya koşulsun mu"
  kararı ona bakılmadan verildi (`gate_report_06.md` val loss içermez; sahibin kararı o rapora dayanır). Bu rapordaki val
  loss sayıları (bölüm 3, 5) ve oracle tavanı karardan SONRA eklendi. V7 seçim betiği val loss'u okuyamaz (mutant H05).
* **Erken uyarıda koşular durdurulmadı** (Sapma-S10; sahibi onayladı): tam eğriler sebep incelemesinin ve oracle tavanının
  verisidir. Bu tarama V6'nın 3 çare çevriminin 1.'si sayıldı; 2. ve 3. çevrim koşulmadı, çünkü Sapma-S4'ün sınırı içinde
  (hata düzeltmesi ya da V7 koşulu) meşru bir içerikleri kalmamıştı.
* **Guard'lar:** 06 mutasyon sınavı (`mutation_log_06.md`) ve bu raporun sayılarını üreten teşhis betiklerinin sınavı
  (`mutation_log_07.md`) hayatta kalan mutant olmadan geçti; tag'li sınav 136 mutant (donmuş kayıt). Testlerin tamamı
  geçiyor; `check_prereg_frozen` (gövde tag'dekiyle aynı), `check_prereg_numbers` (90 kontrol), `verify_pins` (39 çivi) temiz.
* **ZORUNLU DOĞRULAMA 1–3** (sayılmış bağlantı, router nedenselliği, modül başına grafik): `gate_report_06.md` bölüm 1–2.
  VRAM tepesi ölçüldü: D 6.48 / M 8.03 / H 6.77 / X 6.81 GB; X'in gerçek koşu tepesi 6.80 GB.

## 7. V11 zorunlu raporlama maddelerinin durumu

| madde | durum |
|---|---|
| karar satırı, `s̄`, GA, `ub95`, birincil-2, Holm (`decision.json`); `savings.json` (ve içindeki B15 eğri kesişim bayrağı); `boundary_diagnostic.json`; `walltime.json`; `session_variance.json`; `flops_crosscheck.json`; `diagnostics.json` (D1–D11) | **ÜRETİLMEDİ** — karar koşusu yok; uydurulmadı |
| vazgeçme kararı (`abandon()`) | **ÇAĞRILMADI** — `abandon()` ölçek sonuçları ister; deney karar tablosuna gelmeden V6 yolundan bitti |
| sahibin ölçek hakkı seçimi (`owner_decisions.jsonl`) | **YOK** — S1 hakkı doğmadı; sahibin KAPI kararı Sapma-S11'de |
| oracle kazancı ve null'ı (`oracle.json`, `N*` tabanlı) | **HESAPLANAMAZ** — eğriler `L*`'e ulaşmıyor (%50 bütçe); yerine nat tabanlı `oracle_ceiling.json` (bölüm 5), KEŞİFSEL teşhis olarak |
| taranan FOTON varyantı sayısı ve tablosu | **3**; `v7_ablation.json`, bölüm 3 |
| koşu defteri | 32 deneme / 2 iptal / 0 çöken / 0 sürdürülen (+1 kayıtsız duman testi); bölüm 2 |
| kol başına `config.json` SHA-256, `git_head`, `src_sha256` (`pins.json`) | 39 çivi temiz (korpus + baseline'lar); tarama koşuları ÇİVİLİ DEĞİL — `src_sha256` dokuzunda aynı, `git_head` dört farklı commit (bölüm 6) |
| kol başına tam val eğrisi + alan başına val loss | val eğrileri: tarama koşularının `metrics.jsonl`'ı; alan başına val loss ÜRETİLMEDİ (`last.pt`'ler duruyor) |
| `batch_size × eval_iters`; `eval_interval` | 16 × 80 = 1280 dizi; 250 adım (+ durma adımı 2188) |
| modül başına tok/s (O-X1) | `ox1_microbench.json`; bölüm 1a |
| router logları (V6'nın alanları) | tarama koşularının `metrics.jsonl` `train` satırları |
| ulaşılabilirlik sınavı (V12) | `experiments/foton-v02-prereg/reachability.log` (tag öncesi; değişmedi) |
| D'ye karşı bağlam sayıları | (i) D'nin tam bütçe sonu val loss'u 2.8424 (2.8367–2.8455), `L*` = 2.8358'e eksiği **+0.0066 nat** (sansürlü; dondurulmuş baseline'lardan); (ii) eşit token: bölüm 5 sonu; (iii) verilemez — X `L_aux` = 2.8929'a ulaşmadı (%50 bütçe, en düşük val 3.12) |
| yapılan test sayısı | **ön-kayıtlı testlerin (2+5+11) hiçbiri yapılmadı; keşifsel: 7** ölçüm ailesi (dL/dg sinyali; doğrusal prob; en iyi modülün durağanlığı ve koşular arası örtüşmesi; nat tabanlı oracle tavanı; alan politikası; M'e karşı eşit token; grafik örtüşmesi). Karar için p-değeri kullanılmadı; prob artifact'ları 20 permütasyonluk, düzeltilmemiş p taşır (0.048–0.81) |

## 8. KURULMAYAN iddialar

* **"Hipotez terk edildi / vazgeçme koşulu sağlandı"** — `abandon()` çağrılamadı; ön-kayıtlı oracle kazancı (`N*` tabanlı,
  karar koşularından) hesaplanamadı. Bölüm 5'teki +2 karşılaştırması nat tabanlı bir vekildir (tarama seed'leri, %50 bütçe)
  ve vazgeçme koşulunun oracle bacağını SAĞLAMAZ. Hipotez ne doğrulanmış ne terk edilmiş sayılır.
* **"Dizi düzeyi yönlendirme çürütüldü"** — ön-kaydın kapsam daralması gereği kurulamaz: null bir sonuç en fazla "ilk 64
  token'dan tanınabilen yönlendirme bu bütçede kazanç getirmedi" diye okunabilir. Tam-dizi oracle'ı da yalnızca BU
  modüller için bir sınırdır.
* "FOTON başarısız" (karar tablosu satırı olarak) — satır üretilmedi. "FOTON, M'den / D'den kötü" — eşit token bağlam sayısı
  bunu düşündürür ama eşleştirilmemiş, yarım bütçeli tarama verisidir; karar değildir.
* "Heterojen dikkat grafikleri işe yaramaz" — H kolu koşulmadı; T = 512'de, 8 katmanın 2'sinde, head başına ~%50
  yoğunlukta sınandı.
* "Prefix router'lar öğrenemez" — router'ın sinyali sağlamdı; öğrenecek kararlı bir hedef yoktu.
* "Tam bütçede de ayrışma olmazdı" — ölçülmedi. Eldeki eğilim büyüme göstermiyor (tavan nat'ta küçülüyor, puanda yatay;
  null'ı aşan kısım puanda ~0.2'de yatay), ama en iyi modülün kararlılığı eşit aralıklı ölçümde şansın üstüne çıkıyordu
  (0.34 → 0.41, adım 2000).
* Bağlam uzunluğu, ağırlık düzeyinde ayrışma ve token/katman düzeyinde yönlendirme hakkındaki her şey HİPOTEZDİR:
  `v03_design_notes.md`. v03 ön-kaydı yazılmadı.

## 9. Artifact'lar

`experiments/foton-v02/`: `gate_report_06.md` · `v7_ablation.json` · `v7_router_diag.json` · `router_diag/` (koşu başına
`_grad.json`, `_probe.json`; `v7-x-s101_grad_cuda_bf16.json`) · `oracle_ceiling.json` · `screening_context.json` ·
`topology_overlap.json` · `ox1_microbench.json` · `vram_probe.json` · `mutation_log_06.md` · `mutation_log_07.md` ·
`queues/` (koşulan `v7_screen.json`; KOŞULMAYAN `decision_strict.json`) · `v03_design_notes.md`.
`experiments/foton-v02-prereg/run_ledger.jsonl` (koşu defteri). Ham koşular: `checkpoints/foton-v02-tarama/` (git dışında;
koşu başına `metrics.jsonl`, `config.json`, `provenance.json`, `eval_raw/step_*.npz`, `last.pt`).
Betikler: `scripts/router_health.py`, `router_diag.py`, `router_grad_diag.py`, `router_probe_diag.py`, `oracle_ceiling.py`,
`screening_context.py`, `topology_overlap.py` (hepsi testli; son altısı `mutation_guards.py --suite 07` sınavında).
V10 yer tutucusu DOLDURULMADI: karar koşusu olmadığı için preset hash'i yazılmadı (hesaplanan toplam
`fb9cd95a8276293e…`, bilgi için).
