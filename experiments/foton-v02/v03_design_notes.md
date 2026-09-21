# v03 TASARIM NOTLARI — hipotez ve tartışılacak yönler (ön-kayıt DEĞİL; hiçbir şey karara bağlanmadı)

Tarih: 2026-09-19. Bu dosya `foton-v02-prereg` tag'li belgenin parçası değildir ve ona dokunmaz. Amacı, v02'nin
sonucundan (bkz. `REPORT_07.md`) sonra v03 ön-kaydı YAZILMADAN ÖNCE tartışılacak şeyleri kayda geçirmektir. Aşağıdaki
hipotez **deney sahibinin hipotezidir** ve burada *test edilecek bir iddia* olarak yazılmıştır; doğru kabul edilmemiştir.
"Ölçülen" diye işaretlenmeyen hiçbir cümle bir bulgu değildir.

## 1. Sahibin hipotezi (H-bağlam)

> T = 512'de tam nedensel dikkat token başına ortalama 256.5 bağlantı görür. Bizim üç modülümüz 128.4'te, yani yoğunluk
> ~%50. Bu yoğunlukta üç grafik de "geçmişin yarısını görüyor"; hangi yarısı olduğu ikinci dereceden bir etkidir. Ölçülen
> ~0.010 nat'lık modüller arası fark bunun sayısal karşılığı olabilir. Dikkat topolojisi, pencere diziye göre KÜÇÜK
> olduğunda ayrışır: T = 8192'de W = 150 yerel (%3.7 yoğunluk), r = 2 dilated ve s = 4 strided bambaşka yerlere bakar.
> **Hipotez: deney, topolojinin fark yaratamayacağı bir bağlam uzunluğunda koşuldu.**

**Test edilebilir biçimi.** Aynı üç desen ailesiyle, modüller arası dizi-içi kayıp yayılımı (v02'de ~0.010 nat) ve oracle
tavanı (v02'de 1.34 puan; null'ı aşan kısmı 0.28 puan) bağlam uzunluğuyla BÜYÜR: T ≥ 4096'da tavanın null'ı aşan kısmı
başarı eşiğiyle (+5) aynı mertebeye gelir ve en iyi modül prefix'ten (ya da alandan) şansın üstünde öngörülebilir olur.
**Çürütülme koşulu:** T ≥ 4096'da da yayılım ~0.01 nat mertebesinde kalır ve en iyi modül öngörülemez. Bu ölçüm router
GEREKTİRMEZ, bu yüzden tam bir yönlendirme deneyinden daha ucuz olması beklenir (maliyeti ÖLÇÜLMEDİ). Ama "v02'nin
betikleri yeter" DEĞİLDİR; ön koşulları: (a) v02 ile karşılaştırılabilir tek tasarım TEK gövdeli, modüllerin
değerlendirmede zorlandığı modeldir (v02'nin yayılımı, tavanı ve null'ı `foton_eval_all`'dan, paylaşılan gövde üzerinden
gelir); ayrı eğitilmiş üç sabit-modül modelde dizi başına fark seed ve gövde varyansını da taşır ve ancak aynı maskeli /
farklı seed'li bir kontrolle (H tipi null) okunabilir; (b) rastgele ya da sabit atamayla EĞİTİM yolu kodda yok
(`force_module` yalnızca değerlendirmede var); (c) uzun bağlam için yeni korpus, yeniden eşitlenmiş desenler ve muhtemelen
yeni bir çekirdek gerekir (Yön 1). Öneri olarak: v03'ün İLK adımı yönlendirme değil, bu "kazanılacak bir şey var mı"
ölçümü olsun (`scripts/oracle_ceiling.py`, `router_probe_diag.py` o ölçümün analiz tarafıdır).

### Hipotezin SAYILABİLEN kısmı — ölçüldü (`topology_overlap.json`; kombinatorik, kayıp ölçümü değil)

Donmuş üç desen (A yerel W=150; B dilated r=2, W=256; C yarı yerel W=256 + yarı strided s=4, W=256). **Yoğunluk = head
ORTALAMASI** (token/head başına bağlantı ÷ tam nedensel bağlantı); **kapsama, Jaccard ve erişim = modülün head'lerinin
BİRLEŞİMİ** (en az bir head'in gördüğü (i, j) çiftleri). A ve B'de bütün head'ler aynı olduğu için yoğunluk = kapsama;
C'de iki head grubu farklı yerlere baktığı için ayrışır:

| T | yoğunluk A / B / C (head ortalaması) | C'nin birleşim kapsaması | ortalama erişim A / B / C (token) | Jaccard A–B / A–C / B–C |
|---|---|---|---|---|
| 512 | %50.0 / %50.1 / %50.0 | %81.3 | 70 / 170 / 131 | 0.33 / 0.61 / 0.50 |
| 2048 | %14.1 / %21.9 / %21.1 | %36.3 | 74 / 243 / 295 | 0.24 / 0.39 / 0.41 |
| 4096 | %7.2 / %11.7 / %11.5 | %20.0 | 74 / 249 / 323 | 0.23 / 0.36 / 0.39 |
| 8192 | %3.6 / %6.1 / %6.0 | %10.5 | 74 / 252 / 335 | 0.23 / 0.35 / 0.38 |

Bu sayıların hipoteze söylediği üç şey — üçü de hipotezi inceltir, hiçbiri onu sınamaz:

1. **Yoğunluk kısmı HEAD başına doğru:** T = 512'de her head tam nedensel grafiğin ~%50'sini görür; T = 8192'de %3.6–6.1
   (sahibin "%3.7"si A için %3.6 olarak sayıldı). Ama MODÜL düzeyinde "üç grafik de geçmişin yarısını görüyor" C için
   tutmuyor: C'nin head birleşimi T = 512'de nedensel çiftlerin **%81**'ini görür — ve buna rağmen C'nin ortalama kaybı
   A ve B'ninkinden ayırt edilemiyor (modül ortalamaları arasındaki fark 0.0006–0.0021 nat). Bu, hipotezle doğrudan ilgili
   ölçülmüş bir sayıdır: T = 512'de daha fazla geçmiş görmek de fark yaratmadı.
2. **"Hangi yarısı" kısmı sayılarla tam örtüşmüyor:** grafikler T = 512'de ZATEN epeyce farklıdır (A–B Jaccard 0.33: gördükleri
   çiftlerin üçte biri ortak). Pencereler SABİT kaldığı için T büyüdükçe örtüşme de düşer ama yoğunluktan çok daha az:
   Jaccard −%24 … −%44 (0.33 → 0.23; 0.61 → 0.35; 0.50 → 0.38), yoğunluk ise 8–14 kat. Yani T = 512'de ayrışma olmamasının
   sebebi "grafikler aynıydı" olamaz; olsa olsa "bu yoğunlukta hangi yarıyı gördüğün fark etmiyor"dur. Hipotezin test
   edilecek çekirdeği budur: T ile asıl değişen şey örtüşme değil, YOĞUNLUKTUR.
3. **Donmuş pencereler T = 512'ye göre eşitlenmişti.** Aynı pencerelerle T = 8192'de A 148.6, B 248.0, C 246.0 bağlantı görür (T = 4096'da 147.3 / 240.1 / 236.1):
   modüller arası compute eşitliği (v02'nin ilk tasarımında gözden kaçan confound'un kendisi) BOZULUR. Uzun bağlamda
   desen parametreleri yeniden türetilmek ve maske kurulup SAYILARAK doğrulanmak zorundadır (v02 ZORUNLU DOĞRULAMA 1).

### Rakip açıklamalar (asistanın eklediği; aynı ölçüde HİPOTEZ) — v03 bunları birbirinden ayırabilmeli

* **R1 — paylaşılan tam-nedensel katmanlar farkı siliyor.** v02'de 8 katmanın 6'sı tam nedensel ve paylaşılır; yalnızca 2
  katman seyrek ve yönlendirilir. Modülün göremediği bağlantıyı önceki/sonraki 3 tam-nedensel katman taşıyabilir. Bu
  doğruysa bağlam uzunluğu tek başına yetmez: yönlendirilen katman payı da artmalıdır. Ayırt edici ölçüm: aynı T'de
  yönlendirilen katman sayısını değiştirip (2 → 4 → 6) modüller arası yayılımı ölçmek.
* **R2 — tavuk-yumurta.** v02'de seçim fiilen rastgeleydi; modüller her türden veriyi gördü ve genelci kaldı. Ayrışma
  ancak kararlı bir atamayla birlikte eğitimde ortaya çıkıyor olabilir. Ayırt edici ölçüm: alan-etiketiyle SABİT atama
  (router yok; kod → A, matematik → B, NL → C ve permütasyonları) ile eğitip oracle tavanını yeniden ölçmek. Tavan yine
  ~0 ise sorun yönlendirme değildir.
* **R3 — ölçek.** 42.5M aktif parametre ve 72M token'da (tarama) model henüz uzun menzilli yapıdan az yararlanıyor
  olabilir; ayrışma bütçeyle gelebilir. v02'nin eldeki eğilimi büyüme GÖSTERMİYOR (tavan nat cinsinden küçüldü, 0.0085 →
  0.0045; puan cinsinden — her eval noktasının kendi yerel eğimiyle — ~0.6–1.0'da yatay; null'ı aşan kısım ~0.2 puanda
  yatay), ama en iyi modülün kararlılığı şansın üstüne çıkıyordu (eşit 250 adımlık aralıklarda 0.34 → 0.41; son nokta
  0.45'tir ama aralığı 188 adımdır).

## 2. v03 için üç yön — TARTIŞILACAK, şimdi karara bağlanmıyor

**Yön 1 — Uzun bağlam (T ≥ 4096).** Topolojiye gerçekten alan açar; Odis'in uzun bağlam ihtiyacıyla aynı yöne bakar.
Tartışılacaklar: (i) 12 GB'ta T ≥ 4096'nın yoğun SDPA + boolean maske ile maliyeti ÖLÇÜLMEDİ; v02'de çekirdek maskelenen çiftleri de
hesaplıyordu, yani maliyet T² ile büyür — muhtemelen gerçek bir seyrek çekirdek (blok-seyrek / FlexAttention türü) gerekir;
o zaman FLOP muhasebesindeki −%0.72 / +%1.04 asimetrisi de kapanabilir (analitik sayım = koşan çekirdek); (ii) korpusta doküman sınırı YOK (Sapma-S6) ve
pencereler 512'lik kesitlerdi — uzun bağlam için korpusun yeniden kurulması gerekir (TRIPWIRE: yeni deney) ve uzun
dokümanların alan dağılımı bugünkünden farklı olacaktır; (iii) desen parametreleri yeni T'de yeniden eşitlenmelidir
(yukarıda madde 3).

**Yön 2 — Modülleri AĞIRLIK düzeyinde ayrıştırmak.** *Düzeltme (ölçüldü, koddan):* sahibin notundaki "üç modül attention
maskesi dışında her şeyi paylaşıyor" cümlesi v02'nin uygulaması için DOĞRU DEĞİLDİR. v02'de her modül kendi 2 bloğuna
sahiptir (kendi attention VE FFN ağırlıkları, bağımsız init çekimleri): toplam 55.3M parametrenin 42.5M'i aktiftir, yani
modül başına ~6.4M ayrı parametre vardır; paylaşılan şey 6 tam-nedensel blok, embedding ve LM head'dir. Dolayısıyla
~0.010 nat, "ağırlıklar paylaşıldığı için" küçük değildir: ağırlıklar ZATEN ayrıyken ve grafikler farklıyken ölçülmüştür.
Bu yönün geçerli kalan biçimi şunlar olabilir: (a) yönlendirilen PAYI büyütmek (8 katmanın 2'si yerine daha fazlası; R1
ile aynı deney), (b) modül kapasitesini büyütmek, (c) H kolunu (aynı maske, ayrı ağırlık) gerçekten koşmak — v02'de H hiç
koşulmadığı için 0.010 nat'ın ne kadarının maske, ne kadarının ayrı ağırlık/init farkından geldiği BİLİNMİYOR.

**Yön 3 — Dizi başına prefix yerine token/katman düzeyinde yönlendirme.** Tartışılacaklar: (i) v02'de dizi-düzeyi seçim
bilerek yapılmıştı — token düzeyinde dikkat grafiği seçmek KV düzenini ve nedenselliği karmaşıklaştırır (her token başka
bir anahtar kümesine bakar; v02'nin "router geleceği göremez" kilidinin karşılığı yeniden kurulmalıdır); (ii) M kolu zaten
token düzeyinde yönlendirilen bir MoE'dir ve v02'de FOTON'dan eşit token'da ~0.10 nat öndedir (tarama verisi,
eşleştirilmemiş); M, D'nin de önündedir, ama bunun ne kadarının yönlendirmeden ne kadarının ek parametreden geldiği v02'de
ayrıştırılmadı — soru, token düzeyi seçimin dikkat GRAFİĞİNDE bir şey kazandırıp kazandırmayacağıdır; (iii)
HAM token-düzeyi minimum mevcut dökümlerden hesaplanabilir (`ce_tok` [dizi, modül, pozisyon] diskte, float16) — ama ucuz
bir ön-sınav DEĞİLDİR; ölçüldü (`oracle_ceiling.json`, `token_level_raw`): ham "her token'da en iyi modül" farkı 0.063–0.077
nat çıkar (dizi-düzeyi tavanın ~15 katı; token başına modül yayılımı 0.13–0.16 nat) ve neredeyse tamamı seçim gürültüsüdür. v02'nin B11 null'ı
token düzeyinde DEJENEREDİR (pozisyon-başı permütasyon pozisyon-başı minimumu değiştirmez: null ≡ oracle, "null'ı aşan
kısım" kurgu gereği 0). Geçerli bir null aynı maskeli ama ayrı ağırlıklı modüller (H tipi kontrol) ya da başka bir gürültü
modeli ister ve eldeki veride yoktur. Sayı bir sınır da değildir: dökümdeki kayıplar dizinin TAMAMI tek modüle zorlanarak
ölçüldü; dizi ortasında modül değiştirmek sonraki bütün token'ların gizli durumunu değiştirir — gerçekleştirilemez bir
karşı-olgusaldır. Ham sayı tek başına raporlanmamalıdır.

## 3. v02'den v03'e taşınması önerilen yöntem dersleri (tartışmaya açık)

* **Önce tavan, sonra router.** Bir yönlendirme deneyine GPU harcamadan önce "mükemmel seçim ne kazanırdı" ölçülsün ve
  başarı eşiğiyle karşılaştırılsın; tavan eşiğin altındaysa router'ı eğitmenin anlamı yoktur.
* Payda tanımı tek sayıya bağlansın (analitik mi, çekirdek düzeyi mi) — v02'de ikisi ters yöndeydi (−%0.72 / +%1.04).
* Karar koşularında `code_dirty = False` zorunlu, istisnasız; tarama kuyrukları da SIKI üretilsin.
* Kapı raporu birincil ölçüte kör kalsın (v02'de öyle yapıldı).
* Erken uyarı kuralı "durdur" yerine "işaretle ve tam eğriyi bitir" desin: teşhisi üreten veri tam eğridir (Sapma-S10).
* Doküman sınırı olmayan korpusta D10/D11 türü teşhisler baştan tanımsızdır (Sapma-S6); korpus kurulurken sınansın.
* SIKI kuyruk `code_dirty`'yi artık her koşudan önce denetliyor (v02'de yalnızca kuyruk başında denetliyordu; 07'de
  kapatıldı). Açık kalan: `make_campaign_queue.py` karar/S1 kuyruğunda `_strict`'i ZORUNLU kılsın (v02'de `--strict` isteğe
  bağlıydı) ve tarama kuyrukları da SIKI üretilsin.
* **Gürültülü eğrilerde TEK NOKTA kontrolü kırılgandır; akla-yatkınlık kapıları EĞRİ BOYUNCA tanımlansın.** Yakın kaçış
  (v03 adım 1, 2026-09-20): D akla-yatkınlık kapısı atanmış kontrol noktasında (adım 2000) sınırı yalnızca %1.1 aştı
  (+0.012132 / 0.012), oysa aynı koşu adım 1500'de z = +9.6 ile referans ailesinin tamamen dışındaydı. Tek noktalı denetim
  bu anomaliyi neredeyse kaçırıyordu ve kirli veriyle (b) kapısı okunabilirdi.
* **Gürültü zarfı AYNI KODLA koşmuş çiftten ölçülür.** (v03 adım 1, 2026-09-20.) Dense teşhis koşusunun ölçütünü ilk
  yazarken gürültü çapası, iki FARKLI kod sürümüyle koşmuş bir çiftten alınmıştı ve zarfı olduğundan dar gösteriyordu;
  diskte aynı kodla koşmuş bir tekrar çifti (`r-m-s11`) vardı ve zarfı daha genişti. Koşudan önceki inceleme yakaladı.
* **Beş seed'lik referans ailesi kuyruğu dar gösteriyor.** Gelecekteki akla-yatkınlık toleransları daha çok seed'den ya da
  ölçülmüş kuyruğu içeren bir zarftan türetilsin. D-101 bunun kanıtı: kod sağlamken (teşhis koşusu), yalnızca seed çekimiyle
  5 seed'lik ailenin aralığının tamamen dışına düştü (adım 1500'de z = +9.6).
* **(b) sorusu — X-dom eşli D'yi geçiyor mu — cevapsız kaldı.** Doğru kalibre edilmiş D toleransı ve genişletilmiş D
  referansıyla AYRI bir ön-kayıtta sorulacak. Kayıp değil, ertelendi. (Sahibin kararı: toleransın dar kalibre olmuş olması,
  kuralı veriyi gördükten sonra esnetme hakkı vermez; v03 adım 1'de (b) okunmaz.)
* Rapor yazıldıktan sonra bağımsız bir doğrulama turu (kör yeniden hesap + sayı denetimi + düşmanca hakem) koşulsun: v02'nin
  07 raporunda bu tur bir aşırı iddiayı ("öğrenilecek bir şey yoktu"), birime bağlı bir eğilim cümlesini ("tavan küçülüyor")
  ve FOTON lehine yanıltıcı bir parantezi ("fark kapanıyor") yakaladı.

---

**Kayıt (2026-09-19, sahibinden):** Yön 2'deki düzeltme doğrudur; "üç modül maske dışında her şeyi paylaşıyor" cümlesi
sahibinin kendi hatasıydı ve öyle kayda geçmesini istedi. Kurulduğu hâliyle mimari: `docs/ARCHITECTURE_AS_BUILT.md`.
Bölüm 3'teki "açık kalan" madde kapatıldı: `make_campaign_queue.py`'de sıkılık artık VARSAYILAN, karar / S1 kuyruğunda
`--loose` reddedilir. Sahibinin şartı: `ARCHITECTURE_AS_BUILT.md` okunup onaylanmadan bu dosyaya tasarım eklenmez.

**Kayıt (2026-09-20):** (1) HATA KAYDI, ortak: `src/model.py` başlığındaki "Modüller aynı `Block`'tur; TEK fark attention
maskesi" yorumu (asistanın yazdığı) yanıltıcıydı VE sahibi onu doğrulamadan olgu diye söyledi. İkisi de hatadır; yorum
düzeltildi, mimari artık kurulan modelden sayılıyor (`docs/ARCHITECTURE_AS_BUILT.md`, onaylandı). (2) M toplam 91.7M, FOTON
55.3M parametre; aktif eşit (~42.5M), toplam kapasite M'nin %60'ı. X'in M'nin gerisinde kalmasının bir kısmı bu olabilir;
v03'te M karşılaştırması bu farkı ya kapatmalı (modül sayısı / boyutu) ya da açıkça taşımalı. Karar yok, kayıt var.
(3) X-rand KALDIRILDI (eğitimde pencereler tekrar etmediği için "kararlı rastgele" = "rastgele"); yerine X-iid. Adım 1'in
kolları ve karar kuralı: `experiments/foton-v03-step1/decision_rule.md`.
