# FOTON v0.2 — deney ön-kaydı

Bu dosya FOTON kodu yazılmadan, **tek bir FOTON koşusu başlatılmadan** önce yazılmıştır.
`foton-v02-prereg` tag'i atıldığı anda dondurulur ve **o tag'den sonra hedef direği oynatılmaz.**

**Tag'den önce bu dosya bağlayıcı DEĞİLDİR ve bunu iddia etmiyor.** Denetim, ilk hâlin "…dondurulmuştur"
diye geçmiş zamanda yazıldığını ve o anda ne tag'in ne de dosyanın git'te olduğunu buldu; yani
belgenin tek değer kaynağı olan "sonuçlardan önce vardı" iddiası, belgenin kendi cümlesiyle
doğrulanamıyordu. Düzeltilmiş hâli: bağlayıcılık **tag'in varlığından** gelir, bu cümleden değil.
Tag mesajı, tag'lendiği commit'i ve `pins.json`'ın hash'ini içerir.

**Dondurulmuş olanın kapsamı.** Bu ön-kayıt **prefix-tanınabilir, dizi düzeyi, top-1** yönlendirmeyi
test eder (bkz. MİMARİ → Router). Daha genel "dizi-uyarlamalı yönlendirme" iddiası bu deneyin
kapsamında değildir ve buradan çıkan bir null onu çürütmez.

## Neden v02 var: v01 taşınamadı

`PREREGISTRATION.md` (v01, tag'ler `foton-v01-prereg` ve `-r2`) `L*`'i TinyStories korpusuna ve
`data/sp8k` tokenizer'ına çivilemişti. FOTON'un temel sorusu sequence düzeyinde **alan** yönlendirmesi
olduğu için tek alanlı bir korpusta sorulamaz: router'a yönlendirecek bir şey yoktur. Korpus değişince
dağılım değişir, dağılım değişince `L*` geçersizleşir, `L*` geçersizleşince ön-kayıt taşınamaz.
Bu yüzden v01 **dokunulmadan** duruyor ve bu **yeni bir deneydir**.

**v01'den aynen devralınan (yeniden türetilmeyen):** karar tablosu ve eşikleri, seed politikası,
çelişki kuralları, ölçek yükseltme hakkı, vazgeçme koşulu, kilit cümle, test aileleri disiplini
(O4b), karar istatistiğinin kod olması ve regresyon testleri (O4c → `src/stats.py`,
`tests/test_stats.py`), güç kalibrasyonu yöntemi (`scripts/power_analysis.py`), raporlama zorunluluğu.

**v02'de yeniden türetilen:** korpus, tokenizer, baseline, `L*`, `F_base`, bütçe, kalibrasyon.

---

## TEMEL SORU

**Aynı gerçek hesap bütçesinde, girdiye göre farklı hesaplama yolu kullanmak sabit
Transformer/MoE'den daha iyi sonuç veriyor mu?**

Alan korelasyonu **başarı ölçütü DEĞİLDİR**, yalnızca teşhistir (aile 3, D9/D10).

---

## KORPUS (dondurulmuş)

`data/foton`, 300.000.000 eğitim tokeni + 6.000.000 val tokeni, üç alan.

| alan | train token | val token | bayt/token | metin | kaynak(lar) |
|---|---|---|---|---|---|
| NL | 100.000.000 | 2.000.000 | 4.26 | 426 MB | caselaw 51.000.000 + hansard 51.000.000 |
| kod | 100.000.000 | 2.000.000 | 3.27 | 327 MB | stackv2_edu |
| matematik | 100.000.000 | 2.000.000 | 3.03 | 303 MB | arxiv_papers |

### Eşitleme TOKEN ekseninde, bayt ekseninde değil — ve bu bir tercih

Alanlar **token sayısında** eşitlendi (her biri tam 100.000.000), **bayt sayısında değil**.
Bayt/token alanlar arasında farklı olduğu için (NL 4.26, kod 3.27, matematik 3.03) bu, metin olarak
NL 426 MB, kod 327 MB, matematik 303 MB demektir — yani NL, metin hacmi olarak matematikten %41 fazla.

Gerekçe: model **token görür**, bayt görmez; eğitim bütçesi, LR programı, FLOP muhasebesi ve `N*`
hepsi token cinsindendir. Dolayısıyla eşitlenecek doğru eksen token'dır. **Ama bu bir tercihtir ve
kayıtta durması gerekir:** bayt ekseninde eşitlenmiş bir korpus farklı bir deney olurdu ve alan
başına "bilgi miktarı" hangi eksende ölçülürse ölçülsün tartışmalıdır.

### Oranlar eşit, ve bu bilinçli bir nötrlük

Alanlar arası oran (1:1:1) ve NL içindeki caselaw/hansard oranı (1:1) **eşit** tutuldu.
Gerekçe register çeşitliliğinden önce şudur: caselaw'u "uzun mesafe atıf yapısı X kolunun dilated
modülüne daha çok yarar" diye ağırlıklandırmak, **korpusu kanıtlanmak istenen sonuca doğru eğmek**
olurdu ve sonucu savunurken "veriyi mimarine göre mi seçtin" sorusuna açık kapı bırakırdı.
Eşit oran nötr seçimdir. Bu eşitlik kaynak başına *token* seviyesinde kırpmayla sağlanıyor
(`scripts/prepare_foton_corpus.py`); ilk uygulamada caselaw'un büyük dokümanları (medyan 10.6 KB)
quota'yı aşıp NL'i neredeyse tamamen caselaw'a çeviriyordu, düzeltildi.

### Alan-homojen packing YAPISAL olarak garanti

Alan başına ayrı `.bin`: bir pencere bir memmap'in içinden alınır ve alanlar farklı memmap'lerdedir,
yani **pencere alan sınırını geçemez** (`src/data_multi.py`). Doküman sınırları EOS ile işaretli;
pencere doküman sınırını geçebilir (standart, ve her kol için aynı). **Alan etiketi modele verilmez** —
router alanı içerikten çıkarmak zorundadır; etiket yalnızca teşhis ve alan başına val loss için tutulur.

### Lisans: %100 Tier A, doküman başına doğrulanmış

Her doküman kendi lisansını taşımak zorundadır; taşımıyorsa atılır. Kabul edilen: public domain,
CC0, CC BY (share-alike YOK), Open Parliament/Government Licence, Blue Oak Council permissive kod
lisansları. Reddedilen: BY-SA, GFDL/copyleft, NC, ND, lisansı olmayan.

| kaynak | alan | Tier A payı | dağılım | atılan |
|---|---|---|---|---|
| caselaw | NL | %100.00 | Public Domain %100 | — |
| uk_hansard | NL | %100.00 | OPL/OGL %100 | — |
| stackv2_edu | kod | %99.36 | permissive SPDX | %0.64 lisanssız |
| arxiv_papers | matematik | %91.45 | CC BY %74.14 + PD/CC0 %17.31 | %8.55 BY-SA |

Tam rapor: `experiments/foton-v02-prereg/corpus_report.json`. Atılan paylar kasten raporlanıyor:
BY-SA'ya dönmek isteyen biri ne kazanacağını bilir (matematikte +%8.55, kodda +%0.64, NL'de sıfır).

### FineWeb ailesi neden dışlandı (karar kayıtta dursun)

`FineWeb`, `FineWeb-2`, `FineWeb-Edu`, `FineMath` **ODC-By 1.0** altında yayınlanıyor ve dataset
kartlarının kendi ifadesi şudur: *"The dataset is released under the Open Data Commons Attribution
License (ODC-By) v1.0. The use of this dataset is also subject to CommonCrawl's Terms of Use."*

ODC-By **veritabanı derlemesini** lisanslar, altındaki web sayfalarının telif hakkını lisanslamaz.
İçerik taranmış web metnidir ve o içeriğin telifi hiç lisanslanmamıştır. Ticari kullanım riski
dataset lisansından değil içerikten gelir; ODC-By onu kapatmaz. Bu yüzden bu aile, "modelin ticari
olarak sorunlu hale gelmemesi" şartını **karşılamaz** ve korpusa alınmadı. Common Pile v0.1 farklıdır:
her doküman fiilen açık lisanslı ya da kamu malıdır.

*(Bu hukuki tavsiye değildir. Lisans metinlerinin ne dediği yazılıdır; model ağırlıklarının türev
eser sayılıp sayılmayacağı hiçbir lisansta çözülmüş değildir.)*

### `cccc` neden dışlandı: ölçülemezlik

`common-pile/cccc` (Creative Commons Common Crawl) modern web prose için en güçlü adaydı ve hattı
NC/ND lisanslı sayfaları zaten çıkarıyor. Ama **doküman başına lisans alanı yok**: dataset kartı
*"Per-document license information is available in the `license` entry of the `metadata` field"*
diyor, artefaktta o alan boştur (`license: None`; metadata yalnızca `content_type, provenance, url,
warc_*` taşıyor; ham ve `_filtered` sürümlerin ikisinde de aynı). BY ile BY-SA ayrımı yapılamadığı
için bu kaynak Tier A süzgecini **destekleyemez** ve dışlandı.

Elde olanla yapılan çıkarım kayda geçiyor: bir shard'ın %68.7'si açık erişim akademik dergi
(MDPI %16.1, BioMedCentral %15.2, PLOS %9.0, Frontiers %8.6 …) ve bu yayıncıların politikası CC BY;
bilinen BY-SA alan adları %1.4. Yani CC BY payı muhtemelen %85+, **ama bu yayıncı politikasından
çıkarımdır, ölçüm değildir.** Ayrıca akademik ağırlık `cccc`'yi NL alanı olarak da uygunsuz yapardı:
matematik alanı da akademik olduğu için iki alan aynı register'a çökerdi.

### `wikiteam` neden dışlandı: içerik ölçümü

İlk NL adayı `wikiteam_filtered`'dı ve lisansı uygundu (doküman başına taşıyor). **İçeriği ölçülünce
reddedildi:** namespace-0 (gerçek makale) baytların %68.1'i, ama bayt ağırlıklı düzyazı payı
**%27.7 ortalama / %19.2 medyan**; baytların yarısı ≥2.765 B dokümanlardan geliyor ve rastgele beş
örnekten ikisi %0 düzyazı (oyun karakteri listeleri, yama notları). Medyan doküman 572 bayt. Bu alan
"doğal dil" değil madde listesi olurdu ve yapısal olarak koda benzediği için routing deneyini bozardı.

Yerine seçilen caselaw (düzyazı %79.7) ve hansard (%93.5) ölçümle seçildi.

---

## TRIPWIRE — korpus değişikliği yeni bir deneydir

Bu ön-kayıt `L*`'i belirli bir korpus dağılımına çiviliyor. Korpusun **herhangi bir alanının kaynağı,
oranı ya da eşitleme ekseni değişirse** dağılım değişir; dağılım değişince `L*` geçersizleşir;
`L*` geçersizleşince bu ön-kayıt **taşınamaz**. Böyle bir değişiklik "yeniden eğitim" değil
**yeni bir deneydir** ve kendi ön-kaydını, kendi baseline'ını ve kendi `L*`'ini gerektirir —
v01'den v02'ye geçerken uygulanan mantığın aynısı.

Bu yüzden korpus **%100 Tier A** kurulmuştur ve bu bir başlangıç koşulu değil **duran kısıttır**.
Share-alike, NC, ND ya da lisansı doküman başına doğrulanamayan veri eklenmesi ön-kayıt ihlalidir.

**H100'e çıkmak ya da koşu maliyetinin anlamlı hale gelmesi bu kısıtı GEVŞETMEZ** — zaten en kısıtlı
hâlde başlıyoruz, gevşetilecek bir şey yok. Ters yön de kapalı: "küçükken yeniden eğitim ucuzdu, o
yüzden BY-SA ekleyelim" argümanı bu ön-kayıtta geçersizdir.

---

## MİMARİ (tasarım BURADA donuyor; H'nin sonucuna göre X değişmez)

```
Embedding -> 3 paylaşılan blok -> prefix router -> {A, B, C} (2 katman) -> 3 paylaşılan blok -> LM head
```

Router **DİZİ düzeyinde, TOP-1**. Top-2 kullanılmaz: kullanılırsa routing ile fusion aynı anda
test edilmiş olur ve hangisinin işe yaradığı ayrışmaz.

**Dört koşu**, hepsi aynı veri, aynı seed seti, aynı token bütçesi, aynı eval checkpoint'leri,
aynı LR politikası:

| kol | ne |
|---|---|
| **D** | dense baseline (`foton-d`) |
| **M** | token-MoE baseline (`foton-m`: 16 expert, top-3, shared expert, aux 0 + sigmoid + seçim bias'ı 0.001) |
| **H** | homojen FOTON: A=B=C = **aynı** desen (A), **farklı ağırlık** |
| **X** | heterojen FOTON: aşağıdaki üç modül |

### ⚠ SONUÇ TABLOSUNUN BAŞINA YAZILACAK: kalan eşitsizliğin YÖNÜ

**FOTON modülleri, D ve M'in gördüğü attention bağlantısının YARISINI görüyor: ~128.4'e karşı
256.50 (token başına, nedensel ortalama).** Bu eşitsizlik **FOTON'un ALEYHİNEDİR** ve dipnot değil,
her sonuç tablosunun başında yer alacak. İki yönlü okunur ve iki yönü de baştan yazılıdır:

* FOTON buna rağmen **kazanırsa**, sonuç olduğundan daha güçlüdür: yarım bağlantıyla kazanmıştır.
* FOTON **kaybederse**, okuyan kişi yarım bağlantıyla yarıştığını bilmek zorundadır; "heterojen
  yönlendirme işe yaramadı" sonucu bu bilgi olmadan çıkarılamaz.

Neden eşitlenemedi (aritmetik, `src/attn_patterns.py` ile kod olarak doğrulanmış): `T = 512`'de
nedensel maskeyle bir dilated desen rate 2'de **en çok 128.50** bağlantıya ulaşır (pencere ne kadar
büyütülürse büyütülsün: pozisyon `i` için `i/2 + 1` üst sınırı var). Yani üç modülü D/M'in 256.50'sinde
eşitlemek **imkânsızdır**; ortak hedef en kısıtlı modülün tavanı olmak zorundaydı.

### Kol X modülleri — hesap eşitliği buradan gelir

Ortak hedef **128.50** bağlantı/token/head. Parametre şekilleri üç modülde BİREBİR aynı; tek fark
attention maskesi.

| modül | attention grafiği | ortalama bağlantı | en uzak erişim |
|---|---|---|---|
| A | **yerel**, pencere W=150 | **128.17** | 150 pozisyon geri |
| B | **dilated**, rate r=2, W=256 | **128.50** | 511 pozisyon geri |
| C | head'ler ikiye bölünür: yarısı **yerel** W=256, yarısı **strided** s=4, W=256 | **128.38** | 256 / 512 |
| — | *(karşılaştırma)* D ve M: tam nedensel | *256.50* | *511* |

Modüller arası kalan fark **en çok 0.33 bağlantı (%0.26)** → attention FLOP farkı ~0.02 puan.
Bu, birincil ölçütün "seçimden bağımsız" olması için gereken şeydir: router'ın en ucuz modüle
çökerek ölçülen tasarrufu şişirmesi böylece engellenmiştir. (Eşitlenmemiş hâlde A/C oranı 2.18×
olurdu ve C'ye çöken bir router +0.9 puan bedava tasarruf gösterirdi.)

Referanslar: Longformer'ın dilated sliding-window motivasyonu, Sparse Transformer'ın local+strided
yapısı, BigBird'ün local+random+global karışımı.

**Eşitlenen eksenler:** `d_model`, `n_head`, `head_dim`, `ffn_hidden`, modül başına katman sayısı (2),
parametre sayısı (birebir: şekiller aynı), Q/K/V boyutları ve **token başına ortalama nedensel
attention bağlantı sayısı (128.50 ± 0.33)**.

**Bilerek değiştirilen eksen:** attention grafiğinin kendisi.

### Router: NEDENSEL prefix — ve bu bir KAPSAM DARALMASIDIR

Router, dizinin **ilk `k = 64` token'ının** gizli durumundan okur (katman 3'ün girdisi, o 64 pozisyon
üzerinde ortalama havuzlama), doğrusal katman → 3 logit → top-1. Gate `k=1` olduğu için ham `p`
normalize **edilmez** (`p/p = 1` sabitinin türevi sıfırdır ve router görev loss'undan hiç gradyan
alamaz — bkz. `src/moe.py`'deki aynı gerekçe), ama residual dalını zayıflatmaması için
**straight-through** uygulanır: değer `p/p.detach()` = 1.0, gradyan `∂log p`. Böylece router CE'den
gradyan alır ve yönlendirilen dal init'te 1/3'e kısılmaz.

**Neden prefix.** Router diziyi BÜTÜN görürse token 5'in tahmini token 6..511'den bilgi taşır:
nedensellik ihlali. Ölçüsü küçük değil — tavan `log 3 / 512 = 0.002146` nat/token, ölçülen eğimde
**+0.64 puan** bedava "tasarruf", yani GA'yı sıfırdan ayıran 1.09 puanlık çubuğun **%59'u**. M bundan
etkilenmez (token başına routing, nedensel). Dahası ön-kaydın kendisi bu işlemi zaten hile sayıyor:
v01 B12 **oracle**'ı "dizinin tamamına bakıp en iyi modülü seç" diye tanımlıyor ve teşhis kefesine
koyuyor; dizi-havuzlu öğrenilmiş bir router onun zayıf bir versiyonudur ve **karar** kefesinde olurdu.

Buna bağlı olarak **raporlanan val loss dört kolda da pozisyon `64..511` üzerinden** hesaplanır
(`TrainConfig.loss_prefix_skip = 64`). Prefix'ten seçilen yol, yalnızca prefix'ten SONRAKİ token'larda
ölçülür. Maske dört kola da uygulanır, yoksa FOTON'un val loss'u ile baseline'ların val loss'u aynı
nicelik olmaz ve `L*` paylaşılan bir hedef olmaz.

**KAPSAM DARALMASI, açıkça.** Router yalnızca ilk 64 token'ı gördüğü için, "bu dizide uzun menzilli
bağımlılık var" ya da "bu dizinin sonu koda dönüyor" gibi eksenleri **öğrenemez**; yalnızca
prefix'ten tanınabilir olanı öğrenebilir. Dolayısıyla FOTON v0.2'nin test ettiği şey:

> **prefix-tanınabilir yönlendirme** — "dizi-uyarlamalı yönlendirme" DEĞİL.

Bu daralma sessiz değil, kayıtlıdır. Bir null sonuç "dizi düzeyi yönlendirme çürütüldü" diye
okunamaz; en fazla "ilk 64 token'dan tanınabilen yönlendirme bu bütçede kazanç getirmedi" diye
okunabilir. Prefix'in kendisinden tanınamayan bir yönlendirme sinyali varsa, onu ölçmek **başka bir
deneydir** ve nedenselliği başka bir yolla (ör. blok düzeyi yönlendirme) çözmek zorundadır.

**Eğitim loss'u maskelenMEZ** (dört kolda da tüm pozisyonlar), ki eğitim hedefi kollar arasında
birebir aynı kalsın. Bunun bedeli aşağıda eşitlenemeyen eksen (6) olarak yazılıdır.

### Paylaşılan bloklar TAM nedensel attention kullanır — ve bu ölçtüğümüz şeyi SEYRELTİR

Paylaşılan 6 blok (3 önde, 3 arkada) D ve M'in bloklarıyla **birebir aynıdır**: tam nedensel
attention. Gerekçe: FOTON'un baseline'lardan farkı böylece YALNIZCA yönlendirilen 2 katmanda kalır,
karşılaştırma o eksende temiz olur.

Bedeli açıkça yazılıyor: router'dan önce 3 tam-nedensel blok geçtiği için **her token'ın alım alanı
router'a gelmeden önce zaten globaldir**. Dolayısıyla A/B/C'nin grafik farkı 8 katmanın yalnızca
2'sinde yaşıyor ve o katmanlarda attention-bağlam terimi ileri FLOP'un küçük bir payıdır. Yani
manipüle edilen değişken **seyreltilmiştir**: bu tasarımda ölçülen etki, aynı desenlerin bütün
katmanlarda uygulandığı bir tasarımda ölçülecek etkiden küçüktür. Bir null sonuç bu yüzden
"sparse attention grafikleri işe yaramaz" demez.

**Eşitlenemeyen eksenler — gizlenmiyor, raporlanıyor:**

1. **FOTON'un yönlendirilen katmanları PENCERELİ, D/M'in katmanları TAM nedensel** (128.4 vs 256.50).
   En büyük tek fark ve **FOTON'un aleyhine**. Yukarıda tablonun başına yazılan madde bu.
2. **Katman başına etkili alım alanı.** Dilated modül aynı bağlantı sayısıyla daha uzağa erişir;
   tasarımın amacı tam olarak bu, ama "eşit" değil. B 511 pozisyon geriye, A 150'ye erişiyor.
3. **L katman boyunca erişilen ayrık pozisyon sayısı** modüller arasında farklı.
4. **Toplam parametre ve optimizer state.** Aktif FLOP tek modüllü bir modelle aynı, ama FOTON
   yönlendirilen katmanların ağırlığını **üç kez** taşır: VRAM, AdamW state ve `weight_decay`
   etkisi farklı. Ayrıca 64 dizinin 5'inden gradyan alan bir modül, 40'ından alanla aynı büyüklükte
   Adam adımı atar (per-parametre normalizasyon), yani gradyanı gürültülüdür.
5. **Modül başına etkin veri.** Her modül token bütçesinin ~1/3'ünü görür. Sabit bütçede bu
   D/M'e karşı gerçek bir dezavantajdır ve **FOTON'un aleyhinedir**; bu yüzden bir null sonuç
   bilgilendirici değildir ve vazgeçme koşulu tek başına buna dayanamaz.
6. **Eğitim-zamanı prefix sızıntısı.** Eğitim loss'u maskelenmediği için ilk 64 pozisyonun
   tahmini, o pozisyonların KENDİ prefix'inden seçilmiş bir yolla yapılır. Ölçüt bu pozisyonları
   saymadığı için raporlanan sayı temizdir; etki yalnızca dolaylıdır (gradyan üzerinden).
   Tavan: `log 3` nat / 64 token = 0.0172 nat/token, ve o pozisyonlar ölçüte girmiyor.
7. **Router'ın adım başına karar sayısı 64, M'in 32.768.** Aynı LR ile eğitilen router'ın gördüğü
   gradyan sinyali ~500 kat seyrektir. Bu "adillik" değil, FOTON'un aleyhine bir kısıttır ve
   V2 optimizer ayarlarını eşit tuttuğu için düzeltilmiyor; raporlanıyor.
8. **Duvar saati.** Bitişik pencere cache-dostu, dilated/strided gather değil: eşit FLOP'ta modül
   süreleri farklı olacaktır. Modül başına tok/s ayrı ölçülüp raporlanacak (aşağıda O-X1).
9. **Maske kurma maliyeti** desene göre değişir.
10. **C'de head'ler iki desene bölünüyor**, A ve B'de tek desen: head düzeyinde homojenlik farklı.
    C'nin yarım head'leri 192.25, yarısı 64.50 bağlantı görüyor; ortalaması eşitlenmiş, dağılımı değil.

### Kapsam dışı

**Looped/recurrent blok bu deneye girmiyor.** O, "aynı hesap derinliğinde parametre paylaşımı
faydalı mı" sorusunu soran ayrı bir deneydir ve kendi ön-kaydını gerektirir.

**Tek modüllü pencereli kol (router'sız) bu deneye girmiyor** — ve bu bir eksiklik olarak kayda
geçiyor. "Dilated-256 tek başına tam nedenselden iyi mi" sorusu ayrı bir kolu gerektirir; o kol
olmadığı için X'in router'ı B'ye ÇÖKERSE sonuç bir yönlendirme bulgusu değil sabit-mimari
bulgusudur. Bunu karar yoluna bağlayan kural aşağıda (V6-G) yazılıdır.

### Sıra

**H, X'ten ÖNCE koşulur** (router hatası varsa GPU yakmamak için). **Ama X'in tasarımı H'nin
sonucuna göre değiştirilmez** — ikisinin de tasarımı bu dosyada donmuştur ve `src/config.py`
preset'lerinin SHA-256'sı karar koşularından önce V10'a yazılır. Dürüstlük notu: bu hash, config'in
**hash'lendikten sonra** değişmediğini kanıtlar; H'nin sonucunun onu etkilemediğini kanıtlamaz. O
kısmın güvencesi bu metindeki donmuş spesifikasyondur (desen parametreleri, `k`, katman bölüşümü,
gate işlemi — hepsi yukarıda sayısal olarak yazılı), yani config'in bu metinden TÜRETİLEBİLİR olması.

### Kol H'nin rolü — ve H çökerse ne demektir

H, "üç ayrı ağırlık kümesi + routing" etkisini izole eder. Üç modül eşit FLOP'lu ve her dizi tam bir
modülden geçtiği için **FOTON'un aktif FLOP'u tek modüllü bir modelle aynıdır**: iddia "daha az FLOP"
değil, *"aynı aktif FLOP'ta, üç farklı attention grafiği + dizi başına yönlendirme, tek sabit
grafikten iyidir"*. Dolayısıyla **X'in yenmesi gereken şey H'dir**, yalnızca D/M değil; aksi hâlde
ölçtüğümüz şey heterojenlik değil kapasite olur. Bu yüzden X↔H **birincil-2**'dir (aşağıda).

**H'de router'ın ÇÖKMESİ beklenen bir sonuçtur, arıza değil — baştan yazılıyor.** A=B=C olduğunda
üç modülü a priori ayıran hiçbir şey yoktur; sabit token bütçesinde loss'u en küçükleyen yapılanma
her diziyi TEK modüle yönlendirip bütün veri bütçesini ona harcamaktır. Yani H'nin "doğru" cevabı
çökmektir ve V6'nın erken uyarı kuralı **H için bir arıza ölçütü değildir** (V6'da bu ayrım yazılı).
H'nin çökmesi hâlinde X↔H karşılaştırması hâlâ geçerlidir: H o zaman "router'lı ama tek etkin
modüllü" bir kol olur, ki heterojenliğin karşısına konacak doğru kontrol tam olarak odur.

---

## HEDEF L*

    L* = 2.8358   (validation loss, nat/token, pozisyon 64..511 üzerinden)

**Nasıl hesaplanır (kural, sonuç görülmeden yazıldı).** En iyi baseline'ın referans eğitim
bütçesinin %80'inde ulaştığı ortalama validation loss:

| adım | değer |
|---|---|
| Referans eğitim bütçesi | **143.360.000 token** = 4375 adım × 32.768 token/adım |
| %80 noktası | 114.688.000 token = **adım 3500** |
| Adım 3500 bir eval noktası **mı?** | **EVET** — `eval_interval` 250, 3500 = 14 × 250 |
| İnterpolasyon (`L*` için) | **GEREKMİYOR** (ama `N*` için gerekiyor, bkz. V3) |
| Ölçüt | maskeli val loss, `loss_prefix_skip = 64`, 1280 sabit pencere |
| Ortalama | 5 karar seed'i {11, 22, 33, 44, 55} |
| En iyi baseline | `foton-m` (referans bütçede en düşük val loss; D'ye karşı doğrulandı, aşağı bak) |

**Ölçülen değerler (5 seed, adım 3500):** 2.834774 / 2.834116 / 2.833520 / 2.835795 / 2.840871
→ ortalama **2.835815**, sd 0.002949 → **L\* = 2.8358**.
Son adımda (4375) ortalama 2.783386. Doğrulama: `N*` 5 seed ortalaması bütçenin **%80.058**'i —
tanım gereği %80 olması gerekiyordu, yani bütçe seçimi çalıştı.
Koşular: `checkpoints/foton-v02r/r-m-s*`, git `28757aa`.
Tam kayıt: `experiments/foton-v02-prereg/lstar.json`, `baseline_reference.json`, `domains/`.

**Alan başına val loss** (son adım, alan başına 1280 pencere, **aynı maske**):

| alan | M (5 seed) | D (5 seed) | M − D |
|---|---|---|---|
| matematik | 2.285–2.294 | 2.331–2.342 | −0.048 |
| kod | 2.845–2.860 | 2.905–2.912 | −0.060 |
| NL | 3.214–3.221 | 3.290–3.300 | −0.075 |

Alanlar arası fark **0.93 nat** (NL − matematik), yani router'ın alanları içerikten ayırt
edebileceği kadar farklılar — bu teşhistir, ölçüt değil. M üç alanın hepsinde D'den iyi, yani
"en iyi baseline" seçimi tek bir alandan gelmiyor.

**Dense bu bütçede `L*`'e ULAŞMIYOR — kayda geçiyor.** `foton-d`, 5 seed, tam bütçe sonunda
ortalama val **2.842379** (2.8367–2.8455); `L*` 2.8358. Eksik **+0.006579 nat**. Yani D'nin `N*`'ı
**sansürlüdür** — sansür tanımına dair sapma "Sapmalar / S1"de yazılı (2× koşu yapılmadı).
Bunun üç sonucu ön-kayda yazılıyor:

1. **"En iyi baseline" seçimi doğrulandı, tahmin değil:** eşit token bütçesinde M − D = **−0.0590 nat**,
   %95 GA [−0.0634, −0.0546], p < 1e-5, 5 eşleştirilmiş seed. TinyStories'teki fark −0.0344'tü;
   bu korpusta boşluk **1.7 kat** daha büyük. `L*` ve payda M'den türer, doğru olan da bu:
   FOTON'un aşmak zorunda olduğu çubuk **daha iyi** baseline'dır.
2. **D, karar karşılaştırmasının paydası değildir**; bağlam kolu olarak koşar ve raporlanır.
   Kampanyanın birincil karşılaştırmaları X↔**M** (birincil-1) ve X↔**H** (birincil-2)'dir.
   Bu ifade **S0'a özgüdür**: S1'de en iyi baseline yeniden belirlenir (V8b) ve orada D kazanırsa
   payda D olur.
3. **D'ye karşı duvar saati sayısı `L*`'te tanımsızdır** — süre bir hedefe ulaşma süresidir,
   D o hedefe ulaşmıyor. Bu yüzden aşağıdaki yardımcı hedef **şimdi**, sonuçlar görülmeden pinlenir:

       L_aux = 2.8929   (dense'in KENDİ %80 noktası, maskeli ölçütte; her iki baseline de ulaşıyor)

   `L_aux` **karar hedefi değildir** ve karar tablosuna girmez. İzinli kullanımları V11'de tam
   liste hâlinde sayılıdır. `L*`'e karşı verilen kararla `L_aux`'a karşı bir sayının çelişmesi
   hâlinde **`L*` geçerlidir**; `L_aux` sonuç beyanı doğurmaz.

**`L*` bu koşulardan donduruldu; kampanyanın DUVAR SAATİ karşılaştırması (`w̄`) AYRI koşular
kullanır — `s_i`'nin paydası ise bu dondurulmuş koşulardan gelir ve V4'te pinlidir.**
Gerekçe **yalnızca duvar saatidir**: kampanya koşuları iç içe ve Latin kare düzeninde koşulmak
zorundadır (V11a), bu koşular ise iki kollu bir kuyrukta koştu.

İlk hâl ikinci bir gerekçe daha yazıyordu — "bağımsızlık: `L*` ile `F_base,i` aynı koşulardan
gelirse hedef ile ona karşı ölçülen pay ilişkili olur" — ve o gerekçe **yanlıştı**, kayda geçiyor:
`L*`'i üreten koşular karar seed'lerinin ta kendisini ({11,…,55}), aynı preset'i ve aynı korpusu
kullanıyor. "Ayrı koşular" bit-birebir aynı konfigürasyonun yeniden koşumudur; bağımsızlık diye bir
şey yoktur. Gerçek durum: `F_base,i` bu dondurulmuş eğrilerden okunur (V4), `L*` de aynı eğrilerden
gelir, yani hedef ile payda **ilişkilidir** ve bu kabul edilmiştir. Bunun yönü FOTON'un aleyhine
değil lehine de değil; `s_i`'nin gürültüsünü bir miktar azaltır (aynı koşu hem hedefi hem paydayı
verdiği için o koşunun sapması kısmen sadeleşir) ve V9'un ölçtüğü sd tam olarak bu düzenekte
ölçülmüştür, yani kalibrasyon tutarlıdır.

**Bütçe bu yüzden 4375 adım seçildi:** %80 noktası tam bir eval noktasına düşsün ve `L*` **türetilmiş
değil doğrudan ölçülmüş** bir sayı olsun. v01'de interpolasyon kuralı hedefi 0.026 nat oynatabiliyordu
(2250'ye yuvarlama 1.3747, 2500'e 1.3489) ve bir amendment gerektirmişti; `L*` için o yargı kararı
**yok**. (`N*` için interpolasyon gerekiyor ve kuralı V3'te pinli.) Yan etki:
token/aktif-parametre **3.371** (M'in aktif parametresine göre; dense konvansiyonunda 3.375,
v01'de 3.282 ve o sayı router'ı İÇEREN konvansiyondan) — operasyon noktası, ölçekleme iddiası değil.

**Referans sayı (karşılaştırmanın paydası DEĞİL):**

    F_base(referans) = 0.8 × 143.360.000 × 267.755.520 = 3.070835e16 aktif FLOP

`s_i`'nin paydası **seed başına** `F_base,i`'dir (V4). Ölçülen ortalaması **3.073053e16**, yani
referans skalerden **%+0.072** yukarıda — çünkü M'in gerçek `N*` ortalaması bütçenin tanımsal
%80'i değil %80.058'i. Seed başına değerler `baseline_reference.json`'da.

**Dondurulmuş korpus ve baseline çivileri (SHA-256):**

```
b0f6570240ac87a5593706303a5e0ddfbe71529fa2764e497a7ff2e7e8f7249c  data/foton/meta.json
0f85409c0b57c64a343963fbde0f18d80469893ccb0e3f7d73548652119de8af  data/foton/tokenizer.model
0165ed7c88c537ef7a550fbc7a3da704e6083a21644b578358a0d7d01ba9ebdd  data/foton/nl_train.bin
1f4310a45eae1c1d93df972be9430d016fb85ae705717330e6448be3ad6d2886  data/foton/nl_val.bin
6abe883ccb5a3235453a2bee4d7a6b063dc9f24d43513c792af72d5dad0193e0  data/foton/code_train.bin
0c86245449d4a6763068f68fe0c9a9699cf5724a87103c620508ba8872ce24cf  data/foton/code_val.bin
1aa49553b9d6c4db6e158d94735b2f0d37bf9f7f95fc06a8e89ac3877f179c3d  data/foton/math_train.bin
e2fce664e2df6380308b3ac77a1c1d66fe8273efc492c732bd82c0e28de9feb8  data/foton/math_val.bin
```

`.gitattributes` satır sonu normalizasyonunu kapatıyor, yani **bu dosyalar git'e girmiş olsaydı**
hash'ler temiz klonda her platformda doğrulanırdı. **Girmemişlerdir** ve bu fark önemli:

**Neyin doğrulanabildiği, neyin doğrulanamadığı — fazla iddia edilmiyor.** `.gitignore` `data/*` ve
`checkpoints/*` içeriyor, yani **temiz bir klonda korpus da checkpoint'ler de YOKTUR.** İlk hâl
"bu hash'ler temiz klonda her platformda doğrulanır" diyordu; doğru değildi, hash'ler yalnızca bu
makinede doğrulanabilir. Kapatma girişimi:

* **`experiments/foton-v02-prereg/pins.json`** commit'lenen artifact'tır ve **39 dosyayı** kapsar:
  korpusun 9'u (6 `.bin` + `meta.json` + `tokenizer.model` + `tokenizer.vocab`) ve baseline kanıt
  zincirinin 30'u (10 koşu × `metrics.jsonl` + `config.json` + `provenance.json`). `L*`'in türediği
  eğriler artık hash'li; ilk hâlde "baseline çivileri" başlığı vardı ve altında yalnızca korpus
  hash'leri duruyordu.
* `scripts/verify_pins.py --verify` hepsini yeniden hash'ler ve tek bir uyuşmazlıkta sıfırdan farklı
  çıkar. Son koşum: **39 çivi, 39 PASS.**
* **Üçüncü bir tarafın doğrulayamadığı şey:** dosyaların manifest ÜRETİLDİĞİ ANDAKİ baytlarla aynı
  olduğu gösterilir; **bir FOTON koşusundan önce üretildikleri GÖSTERİLMEZ.** O güvence yalnızca
  tag'in zaman damgasından ve tag öncesi git geçmişinden gelir. Ayrıca korpus kurucu betiğin girdileri
  (`data/foton/_raw`) hash'li değildir, yani korpusun `prepare_foton_corpus.py`'den yeniden
  üretilebilirliği bu manifestle kanıtlanmaz.

---

## BİRİNCİL ÖLÇÜT — İKİ uç nokta, ikisi de önceden belirli

`L*` hedefine ulaşmak için gereken **aktif FLOP**. Karar SADECE buna göre verilir. Ölçüt bir tane,
ama **karşılaştırma iki tane**, çünkü bu deneyin sorusu tek bir karşılaştırmayla cevaplanmıyor:

| | karşılaştırma | cevapladığı soru |
|---|---|---|
| **birincil-1** | **X ↔ M** | girdiye göre farklı hesaplama yolu, sabit MoE'yi yeniyor mu? |
| **birincil-2** | **X ↔ H** | **heterojenlik** katkı yapıyor mu (yoksa kazanç yalnızca router + 3 ağırlık kümesinden mi geliyor)? |

İkisine **Holm düzeltmesi** uygulanır (aile büyüklüğü 2, `src/stats.multiplicity(..., family_size=2)`).
Bu, ön-kaydın ilk hâlindeki "1 birincil test, düzeltmesiz çünkü tek" ifadesinin yerini alır: o ifade
üç eşleştirilmiş karşılaştırmayı (X↔M, H↔M, X↔H) tanımlayıp birini sayıyordu ve en iyi satırı veren
karşılaştırmayı sonradan seçme kapısını açık bırakıyordu.

**H ↔ M üçüncü bir birincil DEĞİLDİR**, bağlam olarak raporlanır. Gerekçe: H'nin M'i yenmesi FOTON'un
iddiasını desteklemez — o, "router + üç ağırlık kümesi" etkisidir, heterojenlik değil. H'nin M'i
yenip X'in H'yi yenemediği durum açıkça yazılacaktır: **"yönlendirme kazandırdı, heterojenlik
kazandırmadı."**

### Birincil-2 nasıl puanlanır — karar tablosuyla DEĞİL

Karar tablosunun eşikleri (+%2 / +%5 / +%10) **tasarruf** için kalibre edildi ve birincil-1'e aittir.
Birincil-2 bir **fark kontrastıdır** (`s̄(X) − s̄(H)`) ve aynı eşiklerle puanlanamaz: `decide()`'a
verildiğinde +1.88 puanlık bir gerçek heterojenlik katkısı "nötr" satırına düşerdi (çünkü
+1.88 < +%2), yani deneyin asıl sorusunun olumlu cevabı "etki yok" diye etiketlenirdi.

Birincil-2'nin puanlaması şudur — **DÖRT durum, birbirini dışlayan ve tüketen.** `lo` ve `hi`,
`s̄(X) − s̄(H)`'ın **%97.5 GA'sının** sınırları (Holm'un ilk adımı, α = .025; uygulama
`src/stats.contribution()`), MDE = 1.88 puan:

| durum | ölçüt | nasıl yazılır |
|---|---|---|
| **katkı var, büyük** | `lo > 0` ve `hi ≥ MDE` | "heterojenlik ölçülebilir katkı yaptı: +X.X puan, GA [a, b]" |
| **katkı var, küçük** | `lo > 0` ve `hi < MDE` | "heterojenlik katkı yaptı ama katkı MDE'nin altında: +X.X puan" |
| **katkı MDE kadar değil** | `lo ≤ 0` ve `hi < MDE` | "heterojenliğin katkısı MDE kadar büyük DEĞİL; sıfır ya da negatif olabilir" (`hi < 0` ise ayrıca "katkı NEGATİF" notu düşülür) |
| **ölçemedik** | `lo ≤ 0` ve `hi ≥ MDE` | **"yeterince güçlü ölçemedik"** — "heterojenlik çürütüldü" DEĞİL |

**DÜZELTME, kayda geçiyor:** ilk hâl ÜÇ durum tanımlıyordu ve "katkı var" (`lo > 0`) ile
"katkı yok" (`hi < MDE`) **birbirini dışlamıyordu** — ikisi birden doğru olabilir ve o hâl,
mümkün olan **en bilgilendirici** sonuçtur: katkı gerçek ama MDE'nin altında. Üç durumla o hâl
tabloda karşılığı olmayan bir bölgeye düşüyordu; V12'nin sayısal sınavı bunu olasılıkların
NEGATİFE dönmesiyle yakaladı (n=10, gerçek katkı +1.0 puanda −%11.6). Dört durumun tükettiği
`scripts/rule_reachability.py` içinde assertion olarak kilitli.

Ölçülen dağılım (gerçek katkı = 0, n=5, **Holm'un kendi düzeyinde α=.025**): "MDE kadar değil"
%56.0, **"ölçemedik" %42.8**, "var" %1.2. n=10'da "ölçemedik" %2.0'a düşüyor.

**DÜZELTME:** bu satırın ilk hâli %24.6 yazıyordu, ama o sayı α=.05 aralıklarıyla üretilmişti —
oysa kuralın kendisi (ve MDE'nin türetimi) Holm'un ilk adımındaki α=.025'i söylüyor. Kuralı kendi
düzeyinde taramamak, başka bir kuralı sınamaktır: gerçek "ölçemedik" payı **1.7 kat** daha yüksek.
Yani n=5'te en olası tek sonuç "MDE kadar değil", ama **"ölçemedik" %42.8** ile ondan sonra
geliyor ve bu baştan kabul edilmiştir. `src/stats.decide()` birincil-2'ye
**uygulanmaz** (eşikleri tasarruf için kalibre edilmiştir); uygulanan `src/stats.contribution()`'dır
(`decide_primary()` içinden). Kontrastı `s_i(X↔M) − s_i(H↔M)` olarak **`contribution()` kendisi,
seed KİMLİĞİYLE kurar**: iki kolun tablosunu alır, hazır bir fark listesi kabul etmez (V4'teki
taşıma kuralı). Anahtar kümeleri aynı değilse hata verir.

**Birincil-2'de başarısız seed:** X ya da H'nin herhangi bir seed'i **1× bütçede hedefe
ulaşamadıysa** — sansürlü (2×'te de ulaşamadı) ya da yavaş (yalnızca zorunlu 2× koşusunda ulaştı;
V3) — fark o seed'de anlamlı bir gözlem değildir: farkı onlarca puan oynatır ve t-aralığını
patlatır. O zaman birincil-2 **"ölçemedik"** olarak raporlanır, `p₂ = 1` alınır (Holm'da birincil-1
α/2 ile sınanır: muhafazakâr yön), hangi seed(ler) olduğu kimliğiyle yazılır ve **seed
DÜŞÜRÜLMEZ**. Ölçülü risk: H'nin bir seed'i sansürlüyken tam fark listesi "ölçemedik", o seed
düşürülünce "katkı var, büyük" veriyordu.

### Birincil-2'nin saptanabilir en küçük etkisi (MDE) — ŞİMDİ yazılıyor

X↔H karşılaştırması **birincil-1'den yapısal olarak daha güçsüzdür** ve bu tag'den önce ölçülmüştür:

* kol başına ölçülen `sd(s_i)` = **0.88 puan**
* kollar arası seed korelasyonu **ρ = +0.07** (v01 kalibrasyonu; eşleştirme varyansı düşürmüyor,
  paired/unpaired kazancı 0.90× ve 0.82×)
* → X−H farkının sd'si = `0.88 × √(2 × 0.93)` = **1.20 puan**
* → **Holm'un ilk adımı α = .025**, çünkü aile büyüklüğü 2 (aşağıya bak). t(4) bu düzeyde 3.4954,
  t(9) 2.6850 — α = .05'in 2.7764 / 2.2622'sinden belirgin biçimde büyük.
* → n=5'te **%97.5** GA yarı-genişliği **1.88 puan** (α=.05, yani %95 GA varsayımıyla 1.49 olurdu)

| gerçek X−H | n=5 | n=10 | n=20 |
|---|---|---|---|
| +0.50 puan | %6.0 | %13.6 | %30.9 |
| +1.00 puan | %18.2 | %51.0 | %89.1 |
| +1.50 puan | %38.8 | %87.2 | %99.9 |
| +1.88 puan | %56.7 | %97.6 | %100.0 |
| +2.00 puan | %62.2 | %98.7 | %100.0 |
| +3.00 puan | %92.5 | %100.0 | %100.0 |

**Bağlayıcı yorum kuralı.** n=5'te birincil-2'nin MDE çıtası **+1.88 puan** (= %97.5 GA yarı-genişliği;
gerçek katkı tam bu çıtadayken "katkı var" oranı %56.7 — tablonun ilk hâli o satırı +1.8 puanda
hesaplayıp +1.88 diye etiketlemişti).
Dolayısıyla birincil-2'den çıkan bir null sonuç:

> **"heterojenlik çürütüldü" DİYE OKUNAMAZ. "yeterince güçlü ölçemedik" diye okunur.**

Rapor, birincil-2 null çıktığında bu cümleyi ve yukarıdaki güç tablosunu **birebir** içerecektir.

**DÜZELTME, kayda geçiyor:** bu tablonun ilk hâli MDE'yi α = .05'te türetip **+1.5 puan** yazıyordu,
ama aynı bölüm iki endpoint'e Holm uyguluyor ve Holm'un ilk adımı α/2 = .025'tir. Düzeltilmiş
değerler yukarıdadır; +1.5 puandaki güç %56.2 değil **%38.8**. Ölçülen sd de 0.75'ten 0.88'e
güncellendi (ikinci baseline kümesi).

**Eşik neden yükseltilmedi ve seed neden artırılmadı:** birincil-1'in eşiği (+%5) ön-kayıtta
donmuştur ve birincil-2 ondan türemez; birincil-2'nin işi büyüklük ölçmek değil, birincil-1'in
kazancının heterojenlikten mi geldiğini söylemektir. n=20'ye çıkmak 4 kol × 20 seed = 80 koşu
demektir (~50 GPU saati) ve bu deneyin bütçesinde yoktur. Güçsüzlük **giderilmiyor, ilan ediliyor.**

### Holm iki endpoint'e NASIL uygulanır — ve satır adını nasıl etkiler

Uygulama **`src/stats.decide_primary()`** — iki endpoint'i BİRLİKTE karara bağlayan tek giriş
noktası; içeride `multiplicity(..., family_size=2)`, `decide()` ve `contribution()` çağrılır.
Karar raporu `decide()`'ı tek başına çağırarak yazılamaz: o yol Holm'u atlar. (Bu fonksiyon ikinci
denetimden sonra eklendi; ondan önce kural yalnızca bu metinde vardı ve hiçbir koda bağlı değildi.)
Bağlayıcı sıra:

1. Her iki endpoint için iki yanlı tek-örneklem t-testi `p` değeri hesaplanır (seed başına `s_i`
   ve seed başına `s_i(X) − s_i(H)` üzerinden).
2. Holm **adım adım** (step-down) uygulanır: önce KÜÇÜK `p` α/2 = .025 ile karşılaştırılır;
   **geçemezse durulur ve büyük `p` hiç sınanmaz** (ikisi de reddedilmez). Geçerse büyük `p`
   α = .05 ile karşılaştırılır. İki eşiği bağımsız uygulamak Holm değildir.
3. **Birincil-1'in satır adı, `p₁` Holm'dan GEÇMEZSE "başarılı" ya da "güçlü başarı" OLAMAZ.**
   O durumda satır "belirsiz"e düşürülür ve not olarak "Holm düzeltmesinden geçmedi" yazılır.
4. Birincil-2 yukarıdaki **dört** durumla puanlanır. Aralığı **daima Holm'un İLK adımının
   düzeyindedir (α = .025, yani %97.5 GA)** — Holm'da o endpoint'e düşen adım hangisi olursa olsun.
   Bu bilinçli bir sadeleştirmedir ve muhafazakârdır: "katkı var" durumları `lo > 0` istediği için
   `p₂ < .025` demektir, o da Holm'un HER adımından geçer; yani dört durum ile Holm matematiksel
   olarak çelişemez. Kayan nokta bıçak sırtında çelişirse `decide_primary()` istisna atmaz:
   durumu **muhafazakâr yöne düşürür** ("katkı var" ilan edilmez) ve bunu not olarak yazar.
5. **MDE eşiği n'den bağımsız SABİTTİR: 1.88 puan.** Kesinlik ölçütü yüzünden n = 10'a çıkılırsa
   MDE'nin istatistiksel karşılığı 1.02 puana düşer, ama dört durumdaki "küçük/büyük" sınırı
   1.88'de kalır. Gerekçe: o sınır bir **önemlilik çıtasıdır** (bu deneyin n=5 tasarımının
   ayırt edebildiği en küçük katkı) ve veriyi gördükten sonra n ile birlikte kayarsa, "katkı
   var, küçük" ile "katkı var, büyük" arasındaki ayrım sonradan seçilebilir hâle gelir.

**Bu adım ilk hâlde EKSİKTİ:** Holm aile büyüklüğü 2 olarak pinlenmişti ama hiçbir şeye
bağlanmamıştı, yani `p = 0.0371` gibi bir değerde satır "başarılı" çıkarken Holm hiçbir endpoint'i
reddetmiyordu ve ikisi arasındaki çelişki raporda görünmeyebilirdi. Artık Holm satır adının
ÖNÜNDEDİR.

### İki endpoint, TEK yükseltme hakkı — kime ait?

BELİRSİZ SONUÇ tek bir ölçek yükseltme hakkı, EK1-B3 tek bir seed artırma hakkı veriyor. İki
endpoint olduğu için bu hakların kime ait olduğu belirsizdi; kural:

* **Seed artırma hakkını (n=5 → 10) YALNIZCA birincil-1'in kesinlik ölçütü tetikler** (satır
  "yüksek varyans / belirsiz"). Birincil-2'nin "ölçemedik" durumu seed artırmayı TETİKLEMEZ ve
  tetikleyemez: o durum gerçek katkı sıfırken bile n=5 kampanyalarının %43'ünde görülür, yani onu
  tetikleyici saymak veriyi gördükten sonra isteğe bağlı bir ikinci bakış hakkı demek olurdu.
  Tetiklendiğinde seed artırma **ZORUNLUDUR** (isteğe bağlı değil) ve **kampanya başına BİR kez**
  kullanılabilir (EK1-B3): S0'da harcandıysa S1'de kesinlik ölçütü tetiklenirse satır doğrudan
  "yetersiz kesinlik"tir (`decide_primary(..., seed_right_available=False)`).
* **Seed artırıldığında iki endpoint de yeni n ile yeniden hesaplanır** (`decide_primary()` iki
  endpoint'i aynı seed kümesinde hesaplamayı zorlar; X ve H'nin seed KİMLİKLERİ aynı değilse hata verir).
  Gerekçe: aksi hâlde "birincil-1'i n=5'te 'başarılı' bırak,
  birincil-2'yi n=10'da ölç" gibi bir seçim mümkün olurdu ve o seçim sonuca göre yapılabilirdi.
  Ölçülü risk: n=5'te GA [5.41, 5.59] ile "başarılı" olan bir sonuç, n=10'da GA [3.30, 5.20] ile
  terminal bir sonuca dönüşebilir — yani hangi analiz kümesinin bağladığı SONUCU değiştirir.
* **Ölçek yükseltme hakkı (S0 → S1) da ortaktır:** kullanılırsa dört kol da S1'de koşulur ve
  iki endpoint de S1'de yeniden hesaplanır.
* Raporda **hangi n ve hangi ölçekte** karar verildiği her iki endpoint için birlikte yazılır.

**Güç tablolarının kapsamı:** yukarıdaki ve V9'daki bütün güç sayıları **TEK BAKIŞLIDIR**
(single-look), yani n=5'teki tek bir analizi varsayar. Seed artırma hakkı devredeyken gerçek
prosedür iki bakışlıdır ve iki bakışlı bir prosedürün gerçek yanlış-pozitif oranı tek bakışlının
üstündedir. Bu ön-kayıtta alfa harcaması (alpha spending) **uygulanmıyor**; bunun yerine
kapsam sınırı yazılıyor: kesinlik ölçütü tetiklenip n=10'a çıkıldığı koşularda raporlanan
`p` değerleri ve GA'lar **nominaldir**, iki bakışlı prosedür için düzeltilmemiştir, ve bu
raporda anılır. Gerekçe: alfa harcaması tabloları bu ön-kaydın karmaşıklık bütçesini aşıyor ve
kesinlik ölçütü ölçülen σ = 0.88'de +%5 eşiği dışında hiç tetiklenmiyor (V9 tablosu).

## İKİNCİL ÖLÇÜT

Aynı hedefe ulaşmak için gereken **duvar saati** süresi.

**Kapsam sınırı, baştan ve BU korpusta ölçülerek yazılıyor.** Bu kartta (RTX 4070 SUPER, 12 GB)
MoE baseline'ı dense'e duvar saatinde kaybediyor, ve kaybı v2'de TinyStories'te ölçülenden
(%13.9, gruplu dispatch açıkken) **çok daha büyük**. `L_aux` hedefinde, 5 eşleştirilmiş seed:

| | FLOP/token | hedefe token `N*` | hedefe **FLOP** | hedefe **duvar saati** (TAM ölçüm) |
|---|---|---|---|---|
| M (MoE) | 267.755.520 | 98.75M | %13.88 **daha az** | **25.93 dk** |
| D (dense) | 267.411.456 | 114.82M | — | **21.87 dk** |

Eşleştirilmiş fark: **+4.06 dk = %+18.6**, %95 GA [+2.63, +5.49] dk, **p = .0014**.
**Yani MoE aynı hedefe %13.9 daha az FLOP'la ama %19 daha uzun sürede varıyor.**

**Süre nasıl ölçülüyor — zamana oranlama KULLANILMAZ.** Bu satırların ilk hâli
`süre = toplam_süre × N*/toplam_token` ile hesaplanmıştı, yani koşu içinde tok/s'in SABİT olduğunu
varsayıyordu. Oysa aynı kayıt tok/s'in koşu içinde 64.7k → 50.5k salındığını gösteriyor: varsayım
tam olarak ölçmek istediğimiz gürültüyü siliyordu. `metrics.jsonl` her 25 adımda kümülatif
`train_time` taşıdığı için hedefe ulaşma anı doğrudan okunabiliyor. Ölçülen fark: seed başına
ilk oturumda seed başına **%−4.4 ile %+7.3** arası hata ve **p = .0058 → .0115** (iki kat);
ikinci oturumda girişim az olduğu için hata **%−1.5 ile +%0.3** ve p = .0009 → .0014. Nokta tahmini
iki oturumda da sağlam, etkilenen şey belirsizliğin kendisi. **Kampanyada hedefe ulaşma süresi daima kümülatif
saatten okunur** (`scripts/baseline_reference.py`).

Kökeni belli: iki kolun **FLOP/token'ı neredeyse aynı** (fark %0.129), dolayısıyla süre farkı
algoritmik değil **verimlilik** farkıdır. İki ayrı sayı, adları karıştırılmadan:

* **tok/s düşüşü: %27.89** — M 63.502 tok/s, D 88.064 tok/s, yani `1 − M/D`.
* **süre artışı: %38.68** — M aynı bütçeyi 37.63 dk'da, D 27.13 dk'da işliyor, yani `M/D − 1`.

Bu ikisi **aynı sayı değildir.** Bu satırların ilk hâli süre artışını "tok/s %51 daha düşük" diye
yazıyordu: ters payda hatası, yani v01 EK1-A3'te düzeltilen (%23.2 → %18.8) ve B7'de **yasaklanan**
hatanın aynısı, onu devraldığını yazan belgede. Tutarlılık kontrolü şimdi kapanıyor:
`(N*_M/N*_D) × (tput_D/tput_M) = 0.8601 × 1.3868 = 1.1927` → süre farkı %+19.3, ölçülen tam değer
%+18.6. Üç sayı artık birbirini doğruluyor.

12 GB'da 16 expert'in AdamW state'i belleği uçuruma düşürüyor (ölçülen peak VRAM: **M 8.031 GB**,
D 6.480 GB — mikro-batch 16'da, koşuların kendi kaydından, beş seed'de sıfır yayılım); dispatch/gather maliyeti bunun üstüne
biniyor. Bu **donanımın sonucu, mimarinin değil.**

**Eval/checkpoint yükü ölçüte GİRMEZ ama raporlanır:** uçtan uca M 40.38 dk / D 28.44 dk, yani
yük M'de 2.75 dk, D'de 1.30 dk — yükün kendisi de MoE'de iki kat. Ölçüt `train_time` üzerinden
tanımlı (eval süresi kollar arasında eval maliyeti farkından etkilenir ve model kalitesini ölçmez),
ama bu fark gizlenmiyor.

Dört bağlayıcı sonuç:

1. Bu kampanyada **hesap verimliliği iddiası FLOP cinsinden gösterilebilir, duvar saati cinsinden
   gösterilemez.** Bu sınır **her** raporda anılır.
2. Duvar saati yine ölçülüp raporlanır ve **çelişki kuralları ona bakar**. "Mimari başarı / sistem
   başarısızlığı" bir **NOT'tur, karar tablosunda bir SATIR DEĞİLDİR** — `ROWS` onu içermez ve
   `src/stats.decide()` onu `notes` alanında döndürür. Bu ayrım önemli: not, satır adının yerine
   geçemez, ve null bir sonucun yanına yazılıp "mimari başarı" iddiasına dönüştürülemez. `decide()`
   notu ancak **FLOP kazancının GA'sı tamamen sıfırın üstündeyken** verir; `s̄ = +0.46` gibi
   GA'sı sıfırı kesen bir farkta "çelişki kuralı UYGULANMAZ" notu yazılır.
3. **`w̄ > 0` şartı bu deneyde FOTON'un LEHİNE kolaydır ve bu bir zayıflıktır — kayda geçiyor.**
   `w̄`'nin baseline'ı M'dir, ve M bu kartta **kendi başına** D'den token başına **%27.9** yavaştır
   (16 expert'in AdamW state'i + dispatch). FOTON kollarında ne 16 expert ne o dispatch var.
   Dolayısıyla `w̄ > 0`, FOTON'un kendi hız katkısı olmadan da sağlanabilir — bu, v01 EK1-B16'nın
   aynı kolda bulduğu sorundur ("baseline'ın v3'ün en yavaş kolu olduğunu anmıyordu"). Bu yüzden:
   * `w̄` **hem M'e hem D'ye karşı** raporlanır; "güçlü başarı" iddiası hangisine dayandığını yazar.
   * "güçlü başarı" satırı yalnızca `w̄(M) > 0` **ve** `w̄(D) > 0` iken ilan edilir. M'i yenip
     D'yi yenememek "başarılı"dır; çünkü M'i yenmek bu kartta MoE'nin bellek cezasını yenmektir,
     FOTON'un hızlı olduğunu göstermez.
   * Eşik **gevşetilmiyor ve sıkılaştırılan yön yukarıdadır**: `s̄ ≥ %10` olsa bile iki koşuldan
     biri sağlanmazsa satır "başarılı"dır.
4. İkincil ölçüt **FOTON'un aleyhine de sonuç verebilmek zorundadır.** Bu bölümün ilk hâli iki
   yönlü yazılmıştı: `w̄ ≤ 0` ise "beklenen sonuç, iddiayı çürütmez", `w̄ > 0` ise "güçlü başarı" —
   yani hiçbir dünya durumunda FOTON'un aleyhine sayılamıyordu, ki o zaman ölçüt değildir.
   Düzeltilmiş hâli: `w̄ ≤ 0` **çelişki notunu doğurur ve raporda anılır**; `w̄ < −%25` (yani
   baseline M'in D'ye karşı kendi cezası mertebesinde bir ek kayıp) ise **FOTON'un kendi
   sistem maliyeti** olarak ayrıca raporlanır ve O-X1'in gather ölçümüyle birlikte yorumlanır.

## KARAR TABLOSU

```
  >= %10 tasarruf + quality/sec pozitif ....... güçlü başarı
  >= %5 tasarruf, %95 GA tamamen > 0 .......... başarılı
  +%2 – +%5, veya GA sıfırı kesiyor ........... belirsiz
  -%2 – +%2 ................................... nötr
  < -%2, GA çoğunlukla negatif ................ başarısız
```

Bu beş satıra İKİsi daha eşlik eder: **"yüksek varyans / belirsiz"** (kesinlik ölçütü, çaresi
seed artırmak) ve **"yetersiz kesinlik"** (aynı ölçüt n = 10'da yine çözemezse **ya da** tek seed
artırma hakkı bu kampanyada zaten harcanmışken tetiklenirse; TERMİNAL, çaresi yok). Tam liste `src/stats.ROWS` — **yedi** satır — ve **karar tablosunda olmayan hiçbir ifade
sonuç beyanı değildir.**

**ÖNCELİK TABLOSU (v01 O5, buraya BİREBİR yeniden basılıyor).** Yukarıdaki beş satır tek başına
okunduğunda örtüşüyor ve v01'de bu bir amendment gerektirmişti; v02 "çelişki hâlinde bu dosya
geçerlidir" dediği için öncelik kuralının burada yazılı olması **zorunlu**:

| sıra | koşul | satır |
|---|---|---|
| 0 | ≥ 2 FOTON seed'i `L*`'e 2× bütçede bile ulaşmadı (**sansürlü**; yavaş seed bu sayıma GİRMEZ) | **başarısız** |
| 1a | aralık başarı eşiğini çözemiyor (kesinlik ölçütü), `n = 5` ve seed artırma hakkı DURUYOR | **yüksek varyans / belirsiz** → seed 10'a çıkılır (zorunlu) |
| 1b | aynı, ama `n = 10` **ya da** seed artırma hakkı bu kampanyada zaten harcanmış | **yetersiz kesinlik** (terminal) |
| 2 | `s̄ ≥ 10` ve GA alt > 0 ve `w̄(M) > 0` ve `w̄(D) > 0` | **güçlü başarı** |
| 3 | `s̄ ≥ 5` ve GA alt > 0 | **başarılı** |
| 4 | `s̄ > 2` ve (GA sıfırı kesiyor ya da `s̄ < 5`) | **belirsiz** |
| 5 | `−2 ≤ s̄ ≤ 2` | **nötr** |
| 6 | `s̄ < −2` | **başarısız** |

Kritik nokta (v01 O5'in kendisi): **satır 4 yalnızca POZİTİF ortalamalar için geçerlidir.**
`s̄ = −4` iken GA sıfırı kesse bile sonuç "başarısız"tır, "belirsiz" değildir — yoksa negatif bir
sonuç, GA'yı genişleterek yükseltme hakkı kazanırdı. Tablonun "< -%2, GA çoğunlukla negatif"
ifadesindeki "çoğunlukla negatif" **bağlayıcı bir koşul değildir**, betimlemedir: `decide()`
`s̄ < −2`'de GA'ya bakmaksızın "başarısız" döndürür ve GA sıfırı kesiyorsa bunu not olarak yazar.

**KESİNLİK ÖLÇÜTÜ (satır 1) — mutlak sd eşiğinin yerini aldı. TEK biçimi vardır.** Karar ancak
şu ikisinden biri sağlanırsa belirlidir: (a) `GA alt > 0` **ve** `s̄ ≥ 5` → başarı KURULDU
(`s̄` ve GA tam örnekten); (b) **`ub95 < 5`** → başarı DIŞLANDI. İkisi de sağlanmazsa aralık "eşiği
geçiyor" ile "geçmiyor" arasında ayrım yapamıyor demektir ve sonuç "yüksek varyans / belirsiz"dir;
seed 10'a çıkılır. `ub95`, `s̄`'nin başarısız-seed'e duyarlı %95 üst sınırıdır (tanımı VAZGEÇME
KOŞULU bölümünde); **başarısız seed yokken düz %95 GA üst sınırının aynısıdır**, yani o durumda
ölçüt "GA üst < 5" demektir ve aşağıdaki tablo ile V9'un güç sayıları o hâl için geçerlidir.

**Başarısız seed aralığı ŞİŞİRMEZ** (v01 EK1-B8(ii)'nin devamı). 1× bütçede hedefe ulaşamayan bir
seed — sansürlü (ikame ≈ −150) ya da **yavaş** (yalnızca zorunlu 2× koşusunda ulaştı, `s_i` ölçülü
ve tipik −25 … −150; V3) — "hedefe ulaşamadı"dır, "varyans yüksek" değil. Değeri `ub95`'e t-aralığının
içinden değil **sabit** olarak girer; belirsizlik yalnızca 1×'te ulaşan ("hızlı") seed'lerden gelir.
Satırın kendisi tam örnek `s̄`'sinden gelir (başarısız seed ortalamanın İÇİNDEDİR). Dördüncü denetimin
bulduğu: yavaş seed sıradan gözlem sayıldığında tek bir seed aralığı [−42, +20]'ye şişiriyor,
kampanya önce 10 seed'e sonra terminal "yetersiz kesinlik"e gidiyor, ve "başarısız" satırı ile
vazgeçme yavaş seed'li HİÇBİR kampanyada ulaşılamıyordu; üstelik 1.99×'te ulaşan seed (−148.75)
"yetersiz kesinlik", hiç ulaşamayan (−149.82) "başarısız" veriyordu. Şimdi ikisi aynı muameleyi görür.

**Anlam kararları, açıkça:**
* **Yavaş seed B8'in sansür sayımına GİRMEZ.** "≥ 2 sansür → başarısız" ve "≥ 1 sansür → başarı
  yasak" hükümleri ikamenin FOTON lehine YANLI olmasına dayanır (ikincisi fiilen yedekli bir
  savunmadır — v01 B8'in kendi deyişiyle "kemer + askı": tek sansürlü seed tam örnek GA alt sınırını
  n = 5'te hiçbir zaman, n = 10'da ancak hızlı seed'ler ≥ ~+%73 tasarruf ederken sıfırın üstüne
  çıkarır; yasak hiçbir zaman ihlal edilmez); yavaş seed'in değeri ikame değil
  ölçümdür. Yavaş seed başarıyı yasaklamaz, ama tam örnek GA'sına girdiği için **n = 5'te** (a)
  kolunu fiilen kapatır (`4 × +%60 + 1 yavaş −26` için bile GA alt < 0). **n = 10'da** hafif bir
  yavaş seed'le (a) kolu sağlanabilir (`9 × +%20 + 1 yavaş −30` → `s̄ = +15`, GA alt +3.7 → başarı
  satırı) ve bu bilinçlidir: on seed'den biri bütçeyi %4 aşmış, değeri ölçülmüş ve ortalamanın
  içindedir. O seed 2×'te de ulaşamasaydı (sansürlü; değeri ≈ −150 olurdu) B8 başarıyı yasaklardı.
* **Tek başarısız seed'de tek durum süreksizliği 1× sınırındadır ve ön-kayıtlıdır.** 1× bütçenin hemen içinde ulaşan
  seed (örn. `s_i = −24`) sıradan gözlemdir ve aralığı genişletir; hemen dışında ulaşan (`−26`)
  sabit sayılır. `4 × +%10 + bir seed` örneğinde birincisi "yüksek varyans / belirsiz" (daha çok
  seed ister), ikincisi belirli bir "belirsiz"dir (`ub95 = +3.38`) ve **iki ardışık ölçekte
  tekrarlanırsa vazgeçmeyi tetikleyebilir**. Bu fark bilinçli kabul edilmiştir: kategorik her
  kuralın bir sınırı vardır; sınırın yeri veriye göre değil bütçe tanımına göre (1× referans bütçe)
  önceden sabittir ve seed'in durumu analist tarafından değil `src/stats.resolve_n_star()` tarafından
  atanır. **Tek başarısız seed'de** 2× sınırında süreksizlik YOKTUR (1.99× ile sansür aynı satırı
  verir, `ub95` farkı 0.21 puan). **İki ya da daha çok başarısız seed'de** bir fark daha vardır ve o
  da yukarıdaki anlam kararının doğrudan sonucudur: satır 0 yalnızca SANSÜRLÜ seed'leri sayar, yani
  iki sansürlü seed doğrudan "başarısız" verirken 1.99×'te ulaşan iki yavaş seed satırını `s̄`'den
  alır. Gerçekçi aralıkta ikisi de "başarısız"tır (iki başarısız seed `s̄`'yi ~−30 puan çeker); satırlar
  ancak n = 10'da hızlı seed'ler ortalama ≥ ~+%34.7 tasarruf ederken ayrışır (`8 × ~+%35 + 2 yavaş` →
  "nötr", `+ 2 sansür` → "başarısız"; ikisi de iki ölçekte terk edilebilir). Tur 3 denetimi bu
  cümlenin ilk hâlinin ("2× sınırında süreksizlik YOKTUR", kayıtsız şartsız) fazla iddialı olduğunu
  buldu; senaryo artık ulaşılabilirlik tablosunda assertion olarak durur.
  **Sahibin kararı (2026-09-19, tag'den önce): süreksizlik KALIR.** Üç gerekçe: sınır veriden değil
  bütçe tanımından gelir; istismar edilemez (durum mekanik atanır); ve **yönü muhafazakârdır** —
  yavaş seed'in değeri negatiftir ve sabit girmesi üst sınırı yalnızca aşağı çeker, yani aralığı
  daima başarısızlık/terk yönüne daraltır; başarı kolu (a) tam örnek GA'sına bakar ve durumdan
  etkilenmez. Aleyhimize çalışan, istismar edilemeyen bir süreksizlik kabul edilebilir.
* **Süreksizlik kaldırılmaz, GÖRÜNÜR yapılır — 1× SINIRI TEŞHİSİ (karar kuralı DEĞİLDİR).** Sonuç
  raporunda, X ya da H kolunun herhangi bir seed'inin `N*`'ı 1× referans bütçenin **±%10 bandına**
  (`0.90 ≤ N*/bütçe ≤ 1.10`; bant bu satırda, veri görülmeden pinlenmiştir) düşerse, kararın
  sınırın DİĞER tarafında ne olacağı da yazılır: banddaki seed'lerin durumu tersine çevrilir
  (hızlı ↔ yavaş; `s_i` aynı kalır, değişen yalnızca muameledir), hepsi birlikte, ve
  `decide_primary()` yeniden koşulur. Uygulama `src/stats.boundary_diagnostic()`. **Bağlayıcı karar
  her hâlükârda mekanik durum atamasından çıkandır**; karşı-olgu satırı bir sonuç beyanı değildir,
  yalnızca "bu sonuç sınıra ne kadar yakın" sorusunun cevabıdır. Duyurulmuş süreksizlik gizli
  olandan iyidir.

v01'in "eşleştirilmiş iyileşmenin sd'si > 5 puan" kuralı **kaldırıldı**, çünkü eşik mutlaktı ve n ile
ölçeklenmiyordu: tek çaresi (10 seed) kendisini DAHA olası yapıyordu. Ölçüm
(`experiments/foton-v02-prereg/reachability.log`):

| gerçek σ | eski kural (sd>5) n=5 → n=10 | kesinlik ölçütü n=5 → n=10 |
|---|---|---|
| 0.88 (ölçülen) | %0.0 → %0.0 | %0.0 → %0.0 |
| 3.0 | %2.6 → %0.3 | %20.7 → **%0.4** |
| 5.0 | %40.7 → **%43.6** ✗ | %59.4 → **%19.6** |
| 6.0 | %59.7 → **%71.5** ✗ | %69.0 → **%34.6** |

Yeni ölçütte çare kuralı **gevşetiyor**; eskisinde sıkıştırıyordu. Yanlış pozitif oranı (gerçek
etki 0 iken "başarılı") her σ'da %5'in altında: en kötü σ=6, n=5'te %1.2.

**TERMİNAL KURAL — n=10'da (ya da seed hakkı harcanmışken) yine tetiklenirse ne olur.** Eski kuralın çıkmazı "n=10'da ateşlendi"
satırının olmamasıydı; yeni kuralda o satır var ve şudur: **sonuç "yetersiz kesinlik" olarak
raporlanır ve bu bir sonuç beyanıdır**, "nötr" değil ve "belirsiz" de değil. B3 ikinci bir seed
artırma hakkı vermiyor, ölçek yükseltmesi de bu durumun çaresi değil (sorun ölçek değil kesinlik).
Rapor şunu yazar: *"etki, başarı eşiğinin bu deneyin kesinliğiyle ayırt edilemeyeceği kadar yakınında;
ne kurulmuş ne dışlanmıştır."* Bu, gerçek etki tam eşikteyken **beklenen** sonuçtur: ölçülen σ=0.88'de
μ=+%5 iken kesinlik ölçütü n=5'te %47.7, n=10'da %47.6 tetikleniyor — eşikte belirsizlik içseldir,
seed artırmak onu kaldırmaz. Eşiğin ÜSTÜNDE (μ=+%7) ise çare çalışıyor: σ=3.0'da %91.6 → %98.2,
σ=5.0'da %61.7 → %89.5 (V9 tablosu).
"Nötr"den ayrımı kritik: nötr, etkinin sıfıra yakın olduğunu ölçtüğümüz durumdur; yetersiz kesinlik,
ölçemediğimiz durumdur.

**Zorunlu:** sonuç raporunda bu satırlardan birinin adı **birebir** yazılır. "İyi görünüyor",
"umut verici", "kısmen başarılı" gibi yorumlar geçerli sonuç beyanı değildir. Bağlayıcı sonucu
**`src/stats.decide_primary()`** döndürür ve **başka uygulama kullanılamaz**: birincil-1 için karar
tablosunun satır adı (içeride `decide()`, Holm'dan SONRA), birincil-2 için dört durumdan biri
(içeride `contribution()`). `decide()`'ı tek başına çağırmak Holm'u atlar ve bağlayıcı değildir.

## SEED POLİTİKASI

Karar koşuları 5 eşleştirilmiş seed: **{11, 22, 33, 44, 55}** (S0'da da, yükseltme hakkı
kullanılırsa S1'de de). Kesinlik ölçütü tetiklenirse eklenecek ikinci beş: **{66, 77, 88, 99, 110}**.
Bağlayıcı karar fonksiyonları (`strict=True`) yalnızca bu iki anahtar kümesini kabul eder
(`src/stats.DECISION_SEEDS`, `EXTRA_SEEDS`): n doğru olsa bile başka bir seed kümesi — on seed'in
"en iyi beşi", ya da konumlu veriyi `dict(enumerate(…))` ile yeniden ifade etmek — hata verir. Tarama/debug koşuları: **{101, 102, 103}** (karar seed'leriyle ayrık).
Aynı seed = aynı veri sırası, batch sırası, alan seçimi, dropout RNG politikası, eval örnekleri —
ve bu iddia artık **gerçekten doğru**: veri akışı dizi başına tanımlı ve mikro-batch bölüşümünden
bağımsız (V1). Eskiden `get_batch` generator'ı `batch_size`'a bağlı tükettiği için aynı seed farklı
bölüşümde farklı veri sırası veriyordu ve eşleştirme nominal kalıyordu.

Mimariler farklı olduğu için weight init birebir aynı olamaz; eşleştirme protokol hijyeni içindir.
Bağlayıcı olan ham sd değil **güven aralığıdır** (bkz. kesinlik ölçütü). Bağlayıcı karar yalnızca
**n = 5 ya da n = 10** ile verilir: `decide()` başka bir n'de hata verir. Gerekçe: kilit cümle beş
seed istiyor, ama `decide()`'ın ilk hâli iki seed'le "başarılı" döndürüyordu ve `paired()` eşleşmeyen
seed'leri sessizce düşürdüğü için kısa seed kümesi işleyişte ulaşılabilir bir durumdu.

**Not (v01 kalibrasyonundan devralınan uyarı):** eşleştirme varyansı ölçülebilir biçimde
düşürmüyordu (paired/unpaired kazancı 0.90× ve 0.82×, koşullar arası seed korelasyonu ρ = +0.07).
Eşleştirme protokol hijyeni için korunuyor; güç hesabı bu kazancı varsaymıyor.

## ÇELİŞKİ KURALLARI

FLOP kazancı var, duvar saati kaybı var → **mimari başarı / sistem başarısızlığı**. Fikir korunur,
kernel üzerinde çalışılır. (Bu kartta beklenen sonuç budur; bkz. kapsam sınırı.)

FLOP kaybı var, duvar saati kazancı var → **algoritmik olarak daha kötü**. FOTON'un iddiasını
desteklemez.

## BELİRSİZ SONUÇ

Yalnızca BİR yükseltme hakkı: bir üst ölçek basamağı + 5 seed. **"Trend artıyor" sayısal olarak
tanımlıdır ve tek başına devam gerekçesi değildir:** `s̄(S1) − s̄(S0)`'ın %95 GA'sının alt sınırı
sıfırın üstünde olmalı. Aksi hâlde fark gürültüdür; σ = 0.88–1.20 puanda `+0.4` puanlık bir artış
"artıyor" görünür ama ayırt edilemez. Ayırt edilemiyorsa sonuç **"ölçek bizi kurtarmıyor"** olarak
yazılır, "trend artıyor, devam edilecek" olarak yazılamaz — ve zaten gidilecek bir basamak da yoktur
(V8: dört kollu tavan 640×10, ve V5: en çok 2 ölçek).

### Her satırdan sonra ne olur (v01 EK1-B10, yedinci satırla birlikte yeniden basılıyor)

| satır | sonraki adım |
|---|---|
| güçlü başarı / başarılı | devam; S1 isteğe bağlı (yükseltme hakkı harcanmaz) |
| belirsiz | **tek yükseltme hakkı kullanılır:** S1 + 5 seed |
| yüksek varyans / belirsiz | 10 seed'e çıkılır (zorunlu, EK1-B3); tablo yeniden uygulanır. Yükseltme hakkı harcanmaz |
| **yetersiz kesinlik** | **TERMİNAL.** Seed hakkı bitmiştir ve ölçek yükseltmesi bu satırın çaresi DEĞİLDİR (sorun ölçek değil kesinlik): yükseltme hakkı bu satırdan **doğmaz**. Sonuç **"sonuçsuz: ölçülemedi"** kaydedilir; hipotez ne doğrulanmış ne terk edilmiş sayılır ve bu bir devam gerekçesi değildir |
| nötr | yükseltme hakkı kullanılabilir (S1 + 5 seed). Kullanılmazsa S0 sonucu nihaidir ve "sonuçsuz" kaydedilir |
| başarısız | yükseltme hakkı kullanılabilir. Kullanılmazsa vazgeçme koşulunun tek ölçek yolu değerlendirilir (aşağıda) |

**Sahibin seçimi S1 KOŞULMADAN ÖNCE loglanır (yöntem şartı).** "nötr" ve "başarısız" satırlarında
ölçek hakkını kullanıp kullanmamak sahibin takdirindedir ve o seçim sonucu değiştirir ("başarısız"ta
hak kullanılmazsa vazgeçmenin tek ölçek yolu, kullanılırsa iki ölçek yolu açılır). Seçim S1'in
sonucu görüldükten sonra kayda geçirilebilseydi sonuca göre yapılabilirdi. Bu yüzden S0 kararından
sonra, **her satırda** (zorunlu olan "belirsiz" dahil), seçim **zaman damgalı, gerekçeli TEK satır**
olarak `experiments/foton-v02/owner_decisions.jsonl`'a yazılır (`scripts/owner_decision.py`): seçim
("S1'e çık" / "S1'e çıkma"), gerekçe, yerel + UTC zaman, dayandığı S0 `decision.json`'unun SHA-256'sı,
`git_head`. Defter tek karar kabul eder; karar değiştirilemez. Mekanik bağlar: (i) S1 kuyruğu
(`make_campaign_queue.py`; S1 preset adları `-s1` ile biter ve koşu adları `s1-` önekini alır) "S1'e
çık" satırı olmadan ÜRETİLMEZ; (ii) rapor aşamasında `owner_decision.py --verify` her S1 koşusunun
koşu defterindeki başlangıcının karar satırından SONRA olduğunu denetler; (iii) `abandon()`'ın
`escalation_used` girdisi bu defterden okunur, varsayılan değeri yoktur.

## VAZGEÇME KOŞULU — eşdeğerlik biçimi

İki ardışık model ölçeğinde **`s̄`'nin başarısız-seed'e duyarlı %95 üst sınırı (`ub95`) başarı
eşiğinin (+%5) altında kalırsa** VE oracle-routing kazancı (null'ın üstünde) **her iki ölçekte**
< %2 ise, prefix-tanınabilir dizi düzeyi top-1 FOTON hipotezi terk edilir. `ub95 < +5` aynı zamanda
kesinlik ölçütünün (b) koludur; dolayısıyla bu şartı sağlayan bir ölçek "yüksek varyans / belirsiz"
ya da "yetersiz kesinlik" satırında OLAMAZ — tablodaki "hiçbiri … değil" şartı ayrı bir engel
değil, aynı şartın öbür yüzüdür ve reddin gerekçesi doğru yazılsın ("ölçemedik") diye ayrıca
denetlenir. `src/stats.abandon()` uygular; **metin ile kod arasında fark olursa bu bölümün tablosu
geçerlidir ve fark bir hatadır** (tablo `rule_reachability.py`'de senaryo senaryo assertion olarak
durur). Ölçek sonuçları ve oracle kazançları `abandon()`'a **ölçek kimliğiyle anahtarlı**
(`{"S0": …, "S1": …}`) verilir; iki paralel dizi kabul edilmez (V4'teki taşıma kuralı).

### Vazgeçmenin İKİ yolu

| yol | ne zaman | şartlar (HEPSİ) |
|---|---|---|
| **iki ölçek** (asıl kural) | elde ≥ 2 ölçek sonucu varsa DAİMA bu | son iki ölçeğin ikisinde de `ub95 < +5`; hiçbiri "yüksek varyans / belirsiz" ya da "yetersiz kesinlik" değil; oracle kazancı **her iki ölçekte** < %2 |
| **tek ölçek** (v01 EK1-B10) | elde TEK ölçek sonucu varken | satır **"başarısız"**; ölçek yükseltme hakkı **kullanılmamış**; `ub95 < +5`; oracle kazancı < %2 |

"Nötr" satırından tek ölçekle vazgeçilmez (v01 B10): hak kullanılmazsa sonuç "sonuçsuz" kaydedilir,
hipotez ne doğrulanmış ne terk edilmiş sayılır, ve bu bir devam gerekçesi değildir.
`[S0 "başarılı", S1 "başarısız"]` terk ETTİRMEZ: iki ölçek kuralı her iki ölçeğe bakar ve S0'ın
aralığı +5'in üstündedir.

### `ub95`: `s̄`'nin BAŞARISIZ-SEED'E DUYARLI %95 üst güven sınırı

    ub95 = min(  (100 × n_h + Σ sabit) / n ,
                 (n_h × (hızlı seed'lerin %95 GA üst sınırı) + Σ sabit) / n  )

`n_h` = 1× bütçede hedefe ulaşan ("hızlı") seed sayısı; "sabit" = başarısız seed'lerin değerleri:
sansürlü seed'in ikamesi (≈ −150) ve yavaş seed'in 2× eğrisinden ölçülen `s_i`'si. İlk terim
**aritmetik tavandır** (tasarruf seed başına +100 puanı aşamaz) ve **HER durumda** uygulanır;
`n_h < 2` iken ikinci terim kurulamaz ve yalnızca tavan kalır. (Dördüncü denetim: bu metnin bir
önceki hâli tavanı yalnızca `n_h < 2` için yazıyordu, kod ise her durumda uyguluyordu;
`[−62.5, −125, C, C, C]` için tavansız biçim **+31.4**, tavanlı biçim **−49.9** verir — fark
vazgeçme kararını çevirir. Kod doğruydu, metin düzeltildi.)

Başarısız seed yokken bu, `s̄`'nin düz %95 GA üst sınırının **aynısıdır**. Varken düz t-aralığı
KULLANILAMAZ: sansürlü bir seed'in ikamesi bir gözlem değil **tek yanlı bir sınırdır** (gerçek
değer ondan da küçüktür), yavaş seed'in değeri ise ölçülmüş bir sabittir; ikisini gözlem gibi
aralığa sokmak sd'yi şişirir. `[0.2, −0.1, 0.3, 0.0, −149.82]` için düz aralık **[−113, +53]** çıkar
ve "üst sınır < +5" şartı başarısız seed'li HER senaryoda sağlanamaz olur — yani kural tam ihtiyaç
duyulduğu yerde ulaşılamazdı (ikinci denetim). Başarısız seed'ler yukarı yönde belirsizlik taşımaz;
belirsizlik yalnızca hızlı seed'lerden gelir.

**Bu KOŞULLU bir sınırdır, açıkça:** gözlenen başarısızlık örüntüsü (kaç seed'in, hangi değerle
başarısız olduğu) veri sayılır; başarısız seed ORANININ kendi örnekleme belirsizliği sınıra dahil
değildir. Gerekçe v01 B8'in doktrinidir: beş seed'den birinin 1× bütçede hedefe ulaşamaması
yöntemin gerçek bir başarısızlığıdır, giderilecek bir varyans değil.

Bu biçim sansürlü seed sayısında **monotondur** (k = 0..4 için +0.3, −29.7, −59.5, −89.1, −99.9).
Bir önceki uygulama yalnızca sansürsüz ALT örneğin aralığına bakıyordu ve monoton değildi: 3/5
sansürde 2 seed kalıyor, t(1) = 12.7 aralığı patlatıyor ve **en net başarısızlık** (5 seed'in 3'ü
hedefe ulaşamamış) tek ölçekte %53, iki ölçekte %78 olasılıkla terk edilemiyordu (üçüncü denetim).

**Anlam kararı, açıkça: `4 × +%30 + 1 sansür`.** Birincil istatistik `s̄` hedefe ulaşamayan seed'i
de İÇERİR (EK1-B8 ikamesi): burada `s̄ = −5.95`, satır "başarısız", `ub95 = −5.85`. v01 B10:
"başarısız"ta yükseltme hakkı **kullanılabilir**; kullanılmazsa **ve oracle şartı sağlanıyorsa**
(kazanç < %2) hipotez terk edilir. Yani istatistik tarafında üçüncü bir "başarısız ama ne terk ne
devam" hâli YOKTUR; gerçekçi aralıkta terk etmeyi engelleyebilen tek şey oracle şartıdır (satır 0
"başarısız" verdiği hâlde `ub95 ≥ +5` kalması için n = 10'da iki sansüre rağmen hızlı seed'lerin GA
üst sınırının ≥ ~+%43.7 olması gerekir; o uç durumda tablo geçerlidir ve `ub95` de engeller), ve oracle engellediğinde sonuç
"terk edilmedi: mükemmel yönlendirmede hâlâ ≥ %2 kazanç var" diye yazılır — sorun o zaman fikirde
değil router'dadır. (Bir önceki uygulama sansürsüz seed'lerin aralığına bakıp bu sonucu ne terk
ediyor ne ilerletiyordu; o şart hiçbir bağlayıcı metinde yazmıyordu ve bir park yörüngesiydi.) Beş
seed'den birinin 2× bütçede bile hedefe ulaşamaması yöntemin gerçek bir başarısızlığıdır, varyans
değil. Aynı okuma yavaş seed için de geçerlidir: `4 başabaş + 1 yavaş (1.5×, −56)` → `s̄ = −11.12`,
"başarısız", `ub95 = −10.89`.

### Oracle şartı — hangi ölçek, hangi hesap

Oracle kazancı **ölçek başına** hesaplanır ve iki ölçek yolunda **her iki ölçekte de** < %2 olmalıdır
(`abandon()` iki ölçekte tek bir sayı kabul etmez: S0 = 1.5 / S1 = 2.5 iken sonuç verilen sayıya
bağlı olurdu). Tanım v01 O8/B11/B12'nindir: `s̄^oracle − s̄^oracle_null`, ikisi de **`N*` tabanlı**
tasarruftur, yani bir **eğriden** okunur. Bu yüzden oracle tek bir checkpoint'ten HESAPLANAMAZ:
FOTON'un eğitim döngüsü her eval noktasında val pencerelerini üç modülün üçünden de geçirip
`val_loss_oracle` (dizi başına en iyi modül) ve `val_loss_oracle_null` (B11'in null'ı) eğrilerini
`metrics.jsonl`'a loglamak ZORUNDADIR; `s^oracle` o eğrilerin `n_star()`'ından gelir. V11'deki
`last.pt` kuralı yalnızca alan başına val loss ve D1–D11 teşhisleri içindir, oracle için değil.
Oracle ve null eğrileri `L*`'i kesmiyorsa **V3'ün prosedürünün aynısı** uygulanır (`resolve_n_star()`:
1× eğrisi → aynı seed'in 2× koşusunun eğrisi → `N* = 2 × bütçe` ikamesi); ikisi de ikameye düşerse
o seed'in kazancı 0'dır (mükemmel yönlendirme bile 2× bütçede hedefe ulaştırmıyor: headroom yok).
**Oracle/null için ayrı bir 2× koşu YAPILMAZ:** 2× koşu yalnızca FOTON'un kendi 1× eğrisi `L*`'i
kesmediğinde vardır (V3 adım 2) ve o koşu oracle/null eğrilerini de loglar. FOTON 1×'te ulaşmış ama
null eğrisi 1×'te kesmiyorsa 2× eğrisi yoktur ve null doğrudan ikameye düşer; bu, kazancı
(`s^oracle − s^null`) BÜYÜTÜR, yani vazgeçmeyi ZORLAŞTIRIR (muhafazakâr yön). Oracle eğrisi tanım
gereği hiçbir eval noktasında FOTON'un kendi eğrisinin üstünde olamaz (dizi başına en iyi modül;
router'ın seçtiği modül de adaylar arasındadır), dolayısıyla FOTON 1×'te ulaştıysa oracle da ulaşmıştır.

**Eski biçim neden atıldı.** v01'den devralınan hâl `s̄ ≤ 0` **ve** `GA üst < +%2` idi: bir NOKTA
TAHMİNİNİ bir ARALIK SINIRIYLA "ve" ile bağlıyor. GA yarı-genişliği ≈ 1.167σ olduğundan koşul
`s̄ < 2 − 1.167σ` demeye geliyordu:

| σ | gereken `s̄` | başabaş FOTON (s̄=0) terk edilebilir mi? | P(vazgeç), iki ölçek, eski → yeni |
|---|---|---|---|
| 0.88 (ölçülen) | < +0.97 | evet | %24.8 → **%100** |
| **1.72** | **< 0.00** | **sınır** | %14.5 → **%99.3** |
| 3.00 | < −1.50 | **HAYIR** | %3.8 → **%62.6** |
| 6.00 (v01 O9'un beklentisi) | < −5.00 | **HAYIR** | %0.7 → %8.8 |

Yani kural tam ihtiyaç duyulduğu rejimde ölüyordu ve n=10 kurtarmıyordu: değersiz bir FOTON süresiz
"nötr / sonuçsuz" olarak park edebiliyordu. Bu, v01 EK1-B10'da bir kez düzeltilen hatanın **ikinci
kez** ortaya çıkışıdır; örüntü artık V12'de kayıtlı.

**Yeni biçim bir eşdeğerlik testidir:** "etki eşik kadar büyük olabilir" olasılığı veriyle DIŞLANIR.
Nokta tahmini koşulu yok. σ büyürse bu da zorlaşır, ama iki fark var: (i) n artırmak `GA üst`'ü
düşürür, yani çare kuralla aynı yöne çalışır; (ii) tetiklenmediğinde sonuç **"hipotez ayakta" değil
"yetersiz kesinlik"** olarak yazılır — iki durum artık ayrı ve `abandon()` hangisi olduğunu döndürür.

**Anlam kararı, açıkça:** `s̄ = +3` gibi GERÇEK ama eşik altı bir kazanç da vazgeçmeyi tetikler
(GA [2.07, 3.93], tamamı +5'in altında). Gerekçe: hipotez "en az %5", "sıfırdan büyük" değil.
Terk etmek bulguyu silmez — +%3 sonuç olarak raporlanır; terk edilen şey **"prefix-tanınabilir dizi
düzeyi top-1 yönlendirme %5+ getirir"** iddiasıdır. Eski biçimdeki `s̄ ≤ 0` şartı tam bu durumu
koruyordu ve atıldığımızın farkındayız; tek koruma **oracle şartıdır**: mükemmel yönlendirmede
hâlâ ≥%2 kazanç varsa hipotez terk edilmez, çünkü o zaman sorun fikirde değil router'dadır.

## KİLİT CÜMLE

FOTON v0.2 başarılı sayılacaksa, beş eşleştirilmiş seed üzerinde önceden belirlenmiş validation-loss
hedefine ulaşmak için gereken aktif FLOP'ta **M'e karşı** en az %5 ortalama azalma göstermeli ve
eşleştirilmiş %95 güven aralığının alt sınırı sıfırın üzerinde olmalıdır (birincil-1). Heterojenliğin
katkı yaptığı ancak **H'ye karşı** aynı ölçütte gösterilebilir (birincil-2) ve bu karşılaştırmanın
n=5'teki saptanabilir en küçük etkisi +1.88 puandır; oradan çıkan bir null "heterojenlik çürütüldü"
değil "yeterince güçlü ölçemedik" demektir. İki uç noktaya Holm uygulanır ve Holm satır adının
önündedir. Önce kesinlik ölçütüne bakılır: güven aralığı başarı eşiğini çözemiyorsa (ne "alt > 0 ve
s̄ ≥ 5" ne "üst sınır < 5") sonuç "yüksek varyans / belirsiz"dir ve seed 10'a çıkılır; bu hak kampanya
başına bir kezdir ve 10 seed'de ya da hak harcanmışken yine çözülemiyorsa sonuç terminal olarak
"yetersiz kesinlik"tir. Buradaki üst sınır, 1× bütçede hedefe ulaşamayan (sansürlü ya da yavaş)
seed'leri aralığa değil sabit değer olarak sokan `ub95`'tir; öyle bir seed bir yöntem
başarısızlığıdır, varyans değil. Aralık eşiği çözüyor ve s̄ %2-5 arasındaysa sonuç belirsiz sayılır
ve yalnızca bir üst ölçek basamağında tekrarlanır. İki ardışık ölçekte `ub95` +%5'in altında kalır
(ki bu, o ölçeklerin kesinlik ölçütünü de çözdüğü demektir) ve oracle-routing kazancı her iki
ölçekte < %2 ise hipotez terk edilir; yükseltme hakkı kullanılmamışken satır "başarısız" ise aynı
şartlarla tek ölçek yeter.
Bütün bunlar, FOTON modüllerinin baseline'ların YARISI kadar attention bağlantısı gördüğü —
yani kalan eşitsizliğin FOTON'un aleyhine olduğu — koşulunda geçerlidir.

---

## OPERASYONEL TANIMLAR

v01'in O1–O10 ve EK 1 B1–B18 maddeleri **aynen geçerlidir**; aşağıda v02'ye özgü olanlar ve
değişenler yazılıdır. Çelişki hâlinde bu dosya geçerlidir.

### V1. Adım başına token pinli, mikro-batch serbest — eşdeğerliği ölçüldü

Pinlenen şey **adım başına token = 32.768**'dir, mikro-batch değil. Bölüşüm (16 × grad_accum 4)
VRAM için değiştirilebilir, **üç şartla**: (i) adım başına token 32.768 kalır, (ii)
`batch_size × eval_iters = 1280` kalır, (iii) `batch_size`, `MultiDomainDataset.REF_BATCH = 16`'nın
katı olur.

**Bu serbestlik artık YAPISAL olarak güvenli — eskiden değildi.** v01 EK1-B1 pencere SAYISINI
sabitliyordu (`batch_size × eval_iters = 1280`) ama sayıyı sabitlemek aynı pencereleri görmek
anlamına GELMİYORDU: `get_batch` generator'ı `torch.multinomial(weights, batch_size)` sonra alan
başına `randint` ile tükettiği için çekim sırası `batch_size`'a bağlıydı. 40×32 ile 80×16 **farklı
pencere kümesi ve farklı alan karışımı** veriyordu; alanlar arası 0.9 nat fark olduğu için bu
ölçülen val loss'u kaydırır, yani `L*`'i bölüşüme bağlı yapardı. Aynı hata eğitim akışında
"aynı seed = aynı veri sırası" iddiasını da yanlış yapıyordu.

Düzeltme: akış **dizi başına** tanımlı (`sequence_stream`), ve referans bölüşümün (`REF_BATCH = 16`)
tüketim sırasını **birebir tekrar oynatıyor**. Adım başına dizi sayısı `tokens_per_step` pinli olduğu
için sabit (64 = 16×4), dolayısıyla akış bölüşümden bağımsız olabiliyor. Val penceresi kümesi de
sabit bir liste (`frozen_val_windows`, seed 12345, 1280 pencere; alan dağılımı nl 415 / math 434 /
code 431 — çok terimli çekimden geldiği için tam eşit değil, ama dört kolda AYNI).

Ölçüm (`checkpoints/foton-v02/baseline-m-s11/best.pt`, kayıtlı val 2.850059):

| bölüşüm | ölçülen val | kayıtlıdan fark |
|---|---|---|
| 16 × 80 | 2.850074 | +1.5e-05 |
| 32 × 40 | 2.850066 | +8e-06 |
| 8 × 160 | 2.850089 | +3.1e-05 |

Bölüşümler arası yayılım **2.3e-05** (önceki hâlde 4.38e-03 → **190× düşüş**), kalan fark bf16
toplama sırası gürültüsü. Puan cinsinden: 0.0066 puan (önceden 1.26 puan). `tests/test_data_multi.py`
üç şeyi regresyona bağlıyor: akış bölüşümden bağımsız, akış referans bölüşümü birebir yeniden
üretiyor (yani `L*` korunuyor), ve bir pencere alan sınırını geçemiyor.

**Kayda geçen düzeltme:** bu bölümün ilk hâli eşdeğerliği "bu korpusta ölçüldü" diye yazıp
`experiments/foton-v02-prereg/microbatch_check/summary.json` yolunu gösteriyordu. O yol **yok**;
sayılar (1.97e-03 / 3.17e-03 / 0.62×) `experiments/foton-v01/microbatch_check/`'ten geliyordu ve o
koşuların config'i `data/sp8k`, `vocab_size 8192` — yani **v01'in TinyStories korpusu**. Korpus
değişikliğinin sayı taşımayı geçersiz kıldığını savunan belgede taşınmış bir sayıydı. Üstelik
transfer edilemezdi: `src/data.py`'nin tek-alanlı `TokenDataset`'i tek bir `randint` çektiği için
orada bölüşüm gerçekten etkisizdi, `MultiDomainDataset`'te ise etkiliydi. Yukarıdaki yapısal garanti
ve bu korpusta yapılan ölçüm o alıntının yerini alıyor.

### V2. Sabit kalan her şey (baseline `config.json`'undan)

| ayar | değer |
|---|---|
| veri | `data/foton`, 3 alan × 100M train / 2M val token, hash'ler yukarıda |
| tokenizer | SentencePiece BPE, **32768**, alan-dengeli örnekte eğitildi |
| dizi uzunluğu, tokens/adım | 512, **32.768** (16 × 4 × 512) |
| bütçe | `max_steps` 4375 |
| LR | token tabanlı cosine (`src/train.lr_at`), lr 1e-3 → min_lr 1e-4, warmup 200 adım eşdeğeri |
| optimizer | AdamW, β = (0.9, 0.95), weight_decay 0.1, grad_clip 1.0 |
| eval | `eval_interval` **250**, `eval_iters` **80**, sabit generator seed 12345 |
| dtype | bf16 autocast |

**DÖRT kolda AYNI** (v01 EK1-B5, liste genişletildi): lr, min_lr, warmup, β, weight_decay, grad_clip,
dtype, dizi uzunluğu, tokens/adım, `max_steps`, `eval_interval`, `batch_size × eval_iters`,
`loss_prefix_skip = 64`, **`compile = false`**, **`vram_limit_frac = 0.92`**, `moe_impl = "grouped"`,
`dropout = 0.0`, `save_checkpoints = true`.

Son iki listeye sonradan eklendi ve gerekçesi şudur: V10 "FOTON'un uygulama ayrıntıları (kernel,
bellek düzeni)" değişebilir diyor, ve `compile` bu tanıma sığar. Yalnızca FOTON kollarını derlemek
FLOP'u değiştirmeden duvar saatinde %10–30 kazandırır, yani **`quality/sec`'i bir derleyici
bayrağıyla pozitife çevirip "başarılı"yı "güçlü başarı"ya yükseltirdi.** Liste artık KAPALI: burada
olmayan bir ayarın kollar arasında farklı olması sapmadır ve "Sapmalar"a yazılır.

FOTON için farklı bir değer gerekiyorsa **baseline da aynı değerle yeniden koşulur** — ve bu
V10 ile ÇELİŞİR (`L*` ve `F_base` değişemez). Çelişkinin çözümü: baseline'ı yeniden koşmak
gerekiyorsa bu **yeni bir deneydir**, tıpkı korpus değişikliği gibi (TRIPWIRE ile aynı mantık).
Bu ön-kayıt içinde o yol **kapalıdır**.

### V3. `N*` ve aktif FLOP — interpolasyon kuralı PİNLİ

`N*` = val loss eğrisinin `L*`'in altına **KALICI olarak** indiği token sayısı, **token ekseninde
doğrusal interpolasyonla**. "Kalıcı", v01 EK1-B14'ün (*"kesişimden sonraki eval noktası da ≤ L*
olmalı; değilse SONRAKİ kesişime bakılır"*) **bilinçli bir sıkılaştırmasıdır**: `N*`, `L*`'in
ÜSTÜNDEKİ son eval noktasından hemen sonraki kesişimdir — yani kesişimden sonraki eval
noktalarının **TÜMÜ** ≤ `L*` olmalıdır, yalnızca bir sonraki değil. B14'ün harfi tek bir sonraki
noktaya bakar; iki nokta altta kalıp üçüncüde geri çıkan bir eğride B14 erken kesişimi kabul eder,
bu kural etmez. Sıkılaştırma `N*`'ı yalnızca büyütebilir (FOTON'un aleyhine, dört kola aynı
biçimde). Eğrinin son noktası `L*`'in üstündeyse o EĞRİ hedefe ulaşmamıştır (`n_star()` None
döner) ve aşağıdaki prosedür devreye girer. Eval ızgarası **dört kolda da 250 adım** (v01 EK1-B6).
Uygulama **`src/stats.n_star()`**; başka uygulama kullanılamaz.

**Sansür prosedürü (v01 O3 ile B9'un birlikte okunuşu — tek anlamlı hâli):**
1. FOTON seed'inin **1× bütçeli** eğrisi `L*`'i (kalıcı) kesiyorsa `N*` o eğriden okunur (B9).
2. Kesmiyorsa aynı seed **2× bütçeyle yeniden koşulur** (O3). Bu koşu ZORUNLUDUR, isteğe bağlı
   değil — yapılıp yapılmayacağı sonuca göre seçilemez.
3. 2× koşunun eğrisi `L*`'i kesiyorsa `N*` **o eğrinin** kesişimidir ve seed **YAVAŞ**tır
   (sansürlü DEĞİLDİR; `s_i` ölçülmüştür, tipik olarak −25 ile −150 arasındadır). Yavaş seed
   B8'in sansür sayımına girmez, ama sansürlü seed gibi `ub95`'e sabit değer olarak girer ve
   birincil-2'yi "ölçemedik" yapar (KARAR TABLOSU'ndaki kesinlik ölçütü paragrafı).
4. Kesmiyorsa seed **SANSÜRLÜDÜR**: `N* = 2 × referans bütçe` ikamesiyle `s_i ≈ −150`. Bu ikame
   FOTON'un lehine yanlıdır (gerçek `N*` daha da büyüktür) ve raporda böyle anılır (B8(i)).

Prosedürü **`src/stats.resolve_n_star()`** uygular ve seed'in durumunu (**hızlı / yavaş / sansürlü**)
o atar; durum `s_i` ile AYNI kayıtta (`Obs`) taşınır, ayrı bir maske listesinde değil. 1× eğrisi
kesiyorsa 2× eğrisine hiç bakılmaz (hangi eğrinin okunacağı seçilemez) — 2× eğrisi daha erken
kesse bile. Fonksiyon eğrilerin yuvalarıyla tutarlı olduğunu da denetler: 1× yuvasındaki eğri 1×
bütçede, 2× yuvasındaki (1×, 2×] aralığında bitmek zorundadır; 2× koşunun eğrisi 1× yuvasına
düşerse yavaş bir seed "hızlı" etiketlenirdi (tur 3 denetiminde ölçüldü) ve bu artık hata verir.
Durum "hangi eğri kesti"ye göre atanır: zorunlu 2× koşu ayrı bir koşu olduğu için, yeniden koşum
gürültüsü içinde 1× bütçeden az önce kesebilir (`N*/bütçe` ≈ 0.99, durum yine "yavaş"); bu yalnızca
1× sınırının dibinde mümkündür ve o seed 1× sınırı teşhisinin bandındadır, raporlanır.

Üç kayıt düzeltmesi:

* **`L*` interpolasyonsuzdur ama `N*` DEĞİLDİR.** Bu dosyanın `L*` bölümü "interpolasyon gerekmiyor,
  o yargı kararı yok" diyor ve bu `L*` için doğru (bütçe, %80 noktası tam bir eval noktasına düşecek
  şekilde seçildi). Ama birincil ölçütün PAYI `N*`'dır ve o interpolasyonludur: eval ızgarası
  8.192.000 token aralıklı, `N*` ortalaması ızgaranın 14.0166 katı, seed başına değerlerin hiçbiri
  ızgara noktası değil. ln(token) ekseninde interpolasyon `N*`'ı bir aralık içinde ~%0.06 oynatır —
  küçük, ama serbest bir parametre olmamalı. **Token ekseninde doğrusal** olarak pinlendi.
* **B14 iki kez yanlış uygulandı.** İlk uygulama sürdürme kontrolünü hiç yapmıyordu (aşağıda).
  İkincisi ters yöne düştü: sürdürülmemiş bir kesişim görünce "sonraki kesişime bakmak" yerine
  seed'i SANSÜRLÜYORDU — tek bir 0.0005 nat'lık sıçrama, hedefin 0.025 nat altında biten bir koşuyu
  −150 puana ve bütün sonucu "başarısız"a çeviriyordu. Şimdiki uygulama B14'ün yukarıda tanımlanan
  SIKILAŞTIRILMIŞ hâlidir (kesişimden sonraki noktaların TÜMÜ ≤ `L*`).
* **B14 başta hiçbir yerde uygulanmamıştı.** Ön-kayıt kuralı yazıyordu, var olan tek `n_star` uygulaması
  (`scripts/calibrate_v02.py`) sürdürme kontrolünü atlıyordu, ve V9'un bütün sayıları o uygulamayla
  üretilmişti. Somut etkisi: 2750'de `L*`'in altına sarkıp 3000'de geri çıkan bir eğride ilk kesişim
  `N*`'ı 114.7M yerine 90.1M gösterir → `s_i` ~21 puan şişer → "nötr" bir sonuç "güçlü başarı" olur.
  Tek bir `if` farkı. Şimdi `src/stats.n_star()`'da uygulandı, kopya uygulamalar silindi, ve
  `tests/test_stats_v02.py` sarkma senaryosunu regresyona bağladı. Son noktası `L*`'in üstünde
  olan eğri için `n_star()` None döner; bu "seed sansürlü" demek DEĞİLDİR — yukarıdaki dört adım
  işler (önce zorunlu 2× koşu). Sürdürülmemiş bir ara kesişim seed'i sansürlemez: sonraki kalıcı
  kesişime bakılır.

    F(koşu) = N* × train_flops_per_token(koşunun config'i)

Analitik hesap `src/flops.py`, türetim `docs/flops.md`. Profiler yalnızca çapraz kontrol.
FOTON'un modül seçimi aktif FLOP'u değiştirdiği için `src/flops.py` FOTON için genişletilecek ve
genişletme `docs/flops.md`'de elle türetilip **karar koşularından önce commit'lenecek** (v01 O2, B13).
Genişletme, yönlendirilen katmanların attention-bağlam terimini **gerçekleşen bağlantı sayısından**
(`src/attn_patterns.py`) hesaplamak zorundadır; mevcut tam-nedensel `2·d·(T+1)` terimini modül
katmanlarına uygulamak FOTON'un FLOP'unu **fazla** gösterir (128.4 yerine 256.5 sayar) ve tasarrufu
haksız yere küçültür.

**FOTON'a özgü uyarı:** üç modül eşit FLOP'lu olduğu için (128.50 ± 0.33 bağlantı) FOTON'un aktif
FLOP/token'ı seçimden bağımsızdır — bu, birincil ölçütün paydasının router'ın kontrolünde olmaması
için ŞARTTIR. Modüller eşit FLOP'lu OLMAZSA aktif FLOP gerçekleşen seçim karışımına bağlı hale gelir;
o durumda formül koşu başına ölçülen karışımla hesaplanır **ve bu bir sapma olarak yazılır**, çünkü
o hâlde router en ucuz modüle çökerek ölçülen tasarrufu şişirebilir.

### V4. Birincil istatistik

    s_i = 100 × (1 − F_foton,i / F_base,i)     [yüzde puan, seed BAŞINA payda]

Uygulama **`src/stats.savings()`** (oran, seed KİMLİĞİNE göre) + **`src/stats.decide_primary()`**
(Holm'lu bileşik karar; içeride `decide()` ve `contribution()`);
başka uygulama kullanılamaz. Ters payda (`F_base/F_foton − 1`) **kullanılamaz** (v01 EK1-B7).

**Payda seed BAŞINADIR, dondurulmuş bir skaler DEĞİLDİR.** Bu bölümün ilk hâli `F_base,i` yazıyor,
`L*` bölümü ise tek bir skaler (`F_base = 0.8 × 143.360.000 × 267.755.520`) pinliyordu. İkisi aynı
şey değil ve ikisi birlikte geçerli olamaz: skaler payda kullanmak karşılaştırmayı **eşleştirilmiş
olmaktan çıkarır** (baseline gürültüsü paydan silinir, GA yapay olarak daralır, kesinlik ölçütü
kolaylaşır), ve hangisinin kullanılacağı sonucu gördükten sonra seçilebilirdi. Karar: **seed başına
payda**. Skaler `F_base` yalnızca bir **referans sayı** olarak kalır (ölçeği okumak için) ve
`s_i`'de kullanılmaz. Fark ölçüldü: M'in ölçülen `N*` ortalaması bütçenin %80.095'i, tanımsal %80
değil, yani skaler payda `s_i`'yi seed başına paydaya göre %0.118 düşük gösteriyordu.

Seed başına `N*` ve `F_base,i` değerleri **`experiments/foton-v02-prereg/baseline_reference.json`**
içinde kayıtlıdır. Bu bölümün ilk hâlinde hiçbir yerde kayıtlı değillerdi: birincil istatistiğin
paydası dondurulmuş artifact'lardan yeniden kurulamıyordu.

**Paydanın KAYNAĞI pinlidir** (kampanya M'i yeniden koştuğu için iki aday vardı ve hangisinin
kullanılacağı sonuca göre seçilebilirdi):
* **S0**, seed **{11, 22, 33, 44, 55}**: `F_base,i` = `baseline_reference.json`'daki DONDURULMUŞ değerler.
  Kampanyanın aynı seed'lerdeki M koşuları yalnızca `w̄` (duvar saati, aynı blokta ölçülmek zorunda;
  V11a) ve yeniden koşum sadakat kontrolü içindir; paydaya GİRMEZ.
* **S0**, seed **{66, 77, 88, 99, 110}** (yalnızca seed artırılırsa): dondurulmuş değer yoktur;
  `F_base,i` aynı seed bloğunda koşulan M'in `N*`'ından gelir (`resolve_n_star()` ile, aynı prosedür).
* **S1** (yükseltme hakkı kullanılırsa), bütün seed'ler: dondurulmuş değer yoktur (S0'ın değerleri
  başka bir modele ve başka bir `L*`'e aittir, S1'de KULLANILAMAZ); `F_base,i` aynı seed bloğunda
  koşulan, V8b'nin kuralıyla belirlenen en iyi baseline'ın `N*`'ından gelir. S1'de seed başına tek
  bir baseline koşusu vardır, yani seçilecek ikinci bir aday yoktur.

**VERİ TAŞIMA KURALI (yöntem şartı): seed başına sonuçlar seed KİMLİĞİYLE anahtarlı taşınır.**
`src/stats`'ın karar fonksiyonlarının hepsi — `paired()`, `savings()`, `decide()`, `ub95()`,
`contribution()`, `decide_primary()`, `abandon()` — girdi olarak **eşleme** alır
(`dict[seed] → değer` ya da `dict[seed] → Obs(s, durum)`; `abandon()` için `dict[ölçek] → …`) ve
liste/tuple verilince `TypeError` verir. İki kol ancak anahtar kümeleri AYNIYSA birleştirilir;
değilse hata — iki YÖNDE de (H'nin fazla seed'i de hatadır: "birincil-1 n=5, birincil-2 n=10"
ifade edilemez). Tek istisna `paired()`'dır: o da yalnızca kimlikle eşler ama farklı anahtar
kümesinde hata vermez, ortak seed'leri eşleyip düşenleri `dropped` alanında raporlar; karar yolunda
(`savings → decide_primary → abandon`) KULLANILMAZ, yalnızca duvar saati ve kalibrasyon
betiklerinde kullanılır ve onlar `dropped`'ın boş olduğunu assertion ile denetler. **Bağlayıcı modda
iki şey daha zorunludur:** anahtarlar ön-kayıtlı seed'lerdir (SEED POLİTİKASI) ve her seed'in durumu
AÇIKÇA verilir — `savings()`'in `status` argümanının varsayılanı yoktur ve durumsuz (çıplak sayı)
bir seed sessizce "hızlı" sayılmaz, reddedilir (tur 3'te ölçüldü: aynı FLOP'lar durumla
"başarısız", durumsuz "yüksek varyans / belirsiz" veriyordu). `abandon()`'ın skaler girdileri
yalnız anahtar kelimeyle verilir ve `escalation_used`'ın varsayılanı yoktur. Seed'in durumu (hızlı/yavaş/sansürlü) değerle aynı kayıtta taşınır, paralel bir maske
listesinde değil. Sonuçlar eşlemenin ekleme sırasından bit düzeyinde bağımsızdır (kanonik sıra).
Karar yolundaki yükleyiciler eşlemeyi `src/stats.keyed()` ile kurar: aynı seed iki kez gelirse
(yeniden koşulan seed'in iki kaydı) hata verir, `dict()` gibi sessizce ezmez. (`keyed()` `11` ile
`"11"`'i ayrı seed sayar; bu zararsızdır, çünkü iki kolun anahtar kümeleri eşit olmak zorundadır ve
bağlayıcı mod yalnızca ön-kayıtlı TAM SAYI seed'leri kabul eder.)

Neden bir yöntem şartı: **konuma göre eşleme hatası ÜÇ ayrı fonksiyonda üç kez çıktı** —
`paired()` (v3), `savings()` (v02'nin ilk denetimi), `decide_primary()` (dördüncü denetim; X-s22
ile H-s33 eşlenince "katkı var, büyük" olan sonuç "ölçemedik" çıkıyordu) — ve her seferinde
fonksiyon yamalandı. Sorun fonksiyonlar değildi: paralel listeler yanlış eşlemeyi İFADE EDEBİLEN
bir veri biçimidir. Yeni biçimde konum diye bir şey yoktur; yanlış kullanım çalışmaz, patlar.
Mutasyon sınavı (V13) bu sınıf için ayrı bir blok taşır: eşlemeyi ekleme sırasına göre yapan,
bir kaydıran, listeyi sessizce kabul eden, anahtar kümesi denetimini kapatan mutantlar.

**Kararı geçersiz kılan üç hata regresyona bağlıdır** (`tests/test_stats.py`, `test_stats_v02.py`):
eşleştirmenin seed kimliği yerine konuma göre yapılması (yukarıdaki kuralın bütün fonksiyonlarda),
t kritik değerinin n'den bağımsız sabit alınması, ve `n_star`'ın B14 sürdürme kontrolünü atlaması.
Karar raporu yazılmadan önce bu testlerin geçtiği doğrulanır.

### V5. Test AİLELERİ ve önceden sayılmış test sayısı

v01 EK1-B4b'nin aynısı, FOTON'a uyarlanmış ve **yeniden sayılmış** (ilk sayım dört kollu bir
kampanya için yanlıştı).

**Aile 1 — BİRİNCİL (karar veren): ölçek başına 2 test.** birincil-1 (X↔M) ve birincil-2 (X↔H),
**Holm, aile büyüklüğü 2**. "Düzeltmesiz çünkü tek" ifadesi kaldırıldı: tasarım üç eşleştirilmiş
karşılaştırma tanımlıyordu (X↔M, H↔M, X↔H) ve birini sayıyordu, bu da en iyi satırı veren
karşılaştırmayı sonradan seçme kapısıydı.

**Aile 2 — İKİNCİL: 5 test.** `w̄(X↔M)`, `w̄(X↔D)`, `w̄(H↔M)`, oracle-routing tasarrufu,
**oracle NULL'ı** (v01 B11 null'ı zorunlu kılıyor ve vazgeçme ölçütü `s̄^oracle − s̄^oracle_null`
olduğu için null bir testtir; ilk sayımda hiçbir ailede yoktu).

**Aile 3 — TEŞHİS: tam 11 test**, kendi içinde Holm (FWER .05) + BH (FDR .05):

| # | ölçüt |
|---|---|
| D1 | routing regret: CE(öğrenilmiş) − CE(oracle) |
| D2 | oracle gap: `s̄^oracle − s̄` |
| D3 | I(alan ; seçilen modül), bit |
| D4 | **marjinal** routing entropisi H(modül)/log 3 |
| D5 | H(modül \| alan) — alan dışı bilgiye bağlılık |
| D6 | modül yük CV |
| D7 | ölü modül sayısı |
| D8 | JS uzaklığı: modül çıktı dağılımları arası |
| D9 | korelasyon: routing kararı ↔ **alan** |
| D10 | korelasyon: routing kararı ↔ **dizi uzunluğu** |
| D11 | korelasyon: routing kararı ↔ **doküman türü** |

D10 ilk hâlde "dizi uzunluğu / doküman türü" diye tek satırda iki ayrı korelasyon taşıyordu
(v01 bunları iki satır olarak tutuyordu); ayrıldı.

**Toplam önceden sayılmış: 2 + 5 + 11 = 18 test/ölçek, en çok 2 ölçek → 36.**
Raporda yazılacak dize: `"18 (2+5+11), keşifsel: k"`.

**Aile büyüklüğü PİNLİDİR ve p değeri üretebilen test sayısından alınamaz.** `multiplicity()`
çağrıları `family_size=` ile yukarıdaki sayıları alır. Gerekçe ölçümle: teşhislerin bir kısmı
(D4 entropi, D6 yük CV, D7 ölü modül sayısı) tek sayıdır, null'ı yoktur, p üretmez. Onlar havuzdan
düşünce `m` küçülür ve Holm GEVŞER — `p(D2) = 0.02` iken `m=10`'da Holm D2'yi reddeder, `m=2`'ye
düşerse KABUL eder. Yani "hangi teşhis hesaplanabildi" sorusunun cevabı anlamlılığı sonradan
değiştirebiliyordu. `src/stats.multiplicity()` artık `family_size`'ı gerçek p sayısından küçük
kabul etmiyor ve ikisini ayrı ayrı raporluyor (`n_tests` / `n_with_p`).

D9–D11 **alan korelasyonu teşhistir, başarı ölçütü değildir** — hiçbir teşhis ölçütü tek başına
başarı iddiası doğuramaz, yalnızca birincil sonucu YORUMLAMAK için kullanılır. Listede olmayan her
ölçüm "keşifsel" etiketiyle raporlanır, düzeltilmiş aileye girmez.

**AMA teşhis bir başarı iddiasını GEÇERSİZ KILABİLİR (V6-G).** İlk hâl yalnızca tek yönü
yazıyordu ("teşhis başarı doğurmaz") ve ters yön açıkta kalıyordu: her teşhis "bu bir yönlendirme
sonucu değil" diye bağırırken birincil satır "başarılı" kalabiliyordu. Bkz. V6-G.

### V6. Router loglama (eğitim boyunca, her log aralığında)

Zorunlu, `metrics.jsonl`'ın `train` satırlarına şu alan adlarıyla yazılır:
`router_entropy_marginal` (**H(modül) / log 3**, dizi başına seçimlerin marjinal dağılımından,
log aralığı boyunca biriktirilmiş), `router_entropy_per_seq` (**dizi başına softmax entropisinin
ortalaması** / log 3), `router_switch_rate` (ardışık log aralıklarında aynı alanın dizileri için
modül değiştirme oranı), `router_dead` (ölü modül sayısı), `router_load_cv`, `router_margin`
(top-1 ile top-2 logit farkının ortalaması).

**İKİ entropi ayrı ayrı loglanır ve bu şart, ilk hâlin en sinsi belirsizliğini kapatıyor.**
İlk hâl yalnızca "router entropisi" diyordu, ama iki aday ölçüt **ZIT anlam taşır:**

* **marjinal** H(modül) düşük = modüller kullanılmıyor, **çökme** (arıza işareti),
* **dizi başına** entropi düşük = router her dizide kararlı, **uzmanlaşma** (hipotezin öngörüsü).

İdeal sonuç — dizi başına kararlı, diziler arası dengeli — birinde ~0, diğerinde ~log 3 okur. Tek bir
"router entropisi" eşiği bu yüzden ya çalışan bir router'ı durdurur ya bozuk bir router'ı aklar.

**Erken uyarı kuralı (düzeltilmiş).** Yalnızca **tarama tier'ında** ve yalnızca **MARJİNAL** entropi
üzerinde: `router_entropy_marginal < 0.1` ilk 500 adımda kalıcı olarak görülürse (tek bir log
aralığı değil, ardışık üç aralık) tarama koşusu durdurulur ve sebep incelenir. Dört değişiklik:

1. **Pencere 200 → 500 adım.** 200 adım tam olarak `warmup_steps`'tir; orada ölçülen şey davranış
   değil initialization'dır.
2. **Tek yönlü değil.** Ters arıza da kurallı: `router_entropy_per_seq > 0.9` ilk 500 adımda
   kalıcıysa router hiç öğrenmiyor demektir (top-1 gate'in klasik gradyan sorunu) ve bu da
   durdurma sebebidir. İlk hâl yalnızca çökmeye bakıyordu, oysa daha olası arıza budur ve
   entropi YÜKSEK kalır — yani sağlık kontrolü geçer ve null sonuç hipoteze yazılırdı.
3. **H kolu için erken uyarı UYGULANMAZ.** H'de A=B=C olduğu için çökme **optimal** cevaptır
   (bkz. "Kol H'nin rolü"); orada marjinal entropinin düşmesi arıza değildir.
4. **Çare SINIRLI.** "Sebep incelenir" sınırsız bir ayar döngüsüne izin veriyordu ve V10'un
   "değişebilir: uygulama ayrıntıları" maddesi router init ölçeği, gate sıcaklığı ve exploration
   gürültüsünü o başlığa sokabiliyordu — yani karar koşularından önce, aynı korpusun eğitim
   dinamiğine bakarak, FOTON'a özel sınırsız bir arama. Sınır: **en çok 3 tarama çevrimi**, her
   çevrim koşu defterine yazılır, ve taranan varyant sayısı V11'de raporlanır. 3 çevrimde
   çözülmezse sonuç **"router bu tasarımda eğitilemedi"** olarak raporlanır ve karar koşuları
   koşulmaz — bu, FOTON aleyhine bir sonuçtur ve öyle yazılır.

### V6-G. Teşhis bir başarı iddiasını GEÇERSİZ KILABİLİR

Birincil satır "başarılı" ya da "güçlü başarı" çıksa bile, aşağıdaki iki durumda rapor o satırı
**yönlendirme iddiası olarak kuramaz**:

| durum | ölçüt | sonuç nasıl yazılır |
|---|---|---|
| **Router çöktü** | `router_dead ≥ 2` (üç modülden ikisi ölü) ya da son 500 adımda `router_entropy_marginal < 0.1` | "Bu bir yönlendirme sonucu DEĞİLDİR: X fiilen tek modüllü sabit bir mimariye indi. Bulgu, o modülün attention grafiğinin tam nedensel baseline'a karşı sonucudur." |
| **Alan bilgisi yok** | `D3 = I(alan; modül) < 0.05 bit` | "Router alanları ayırt etmiyor; kazanç alan yönlendirmesinden gelmiyor." |

Gerekçe: bu deneyde **tek modüllü pencereli bir kol yok** (bkz. "Kapsam dışı"). Router B'ye çökerse
X, "dilated-256 sabit mimarisi" olur ve o sonuç ön-kayıtta "başarılı" damgası alıp metinde
"yönlendirme kazandı" diye yazılabilirdi. Satır adı yine yazılır (karar tablosu değişmez), ama
**yanına bu cümle zorunlu olarak eklenir.** Böylece teşhis başarı ÜRETEMEZ ama başarı iddiasının
kapsamını daraltabilir.

### V7. FOTON'un kendi router regularization ablasyonu

Tarama tier'ında (seed {101,102,103}), %50 bütçe, **tam üç koşul** (daha fazlası yok):
**regularizasyon yok**, **entropi bonusu**, **load-balancing aux**. 9 koşu.

**KAZANAN NASIL SEÇİLİR — kural şimdi yazılıyor.** İlk hâl seçim ölçütünü söylemiyordu, ve doğal
okuma (tarama val loss'u) seçim yanlılığı üretiyordu: 3 bağımsız çekimde beklenen maksimum 0.846·σ,
yani ölçülen σ = 0.88'de `s̄`'ye **+0.4 ile +0.7 puan** hediye (3 seed ortalaması üzerinden
seçimde 0.846 × 0.88/√3 = 0.43; tek koşu üzerinden seçimde 0.846 × 0.88 = 0.74) — nokta tahmini eşiğinin onda biri
mertebesinde ve birincil-1'in GA yarı-genişliğiyle aynı büyüklükte. Kural:

* Kazanan **val loss'a göre SEÇİLMEZ.** Seçim ölçütü **router sağlığıdır** ve sıralaması şudur:
  (i) `router_dead = 0` olan koşullar, (ii) bunların içinde `router_entropy_per_seq` en DÜŞÜK olan
  (dizi başına en kararlı router), (iii) eşitlik hâlinde alfabetik sıra (regularizasyon yok <
  entropi bonusu < load-balancing aux).
* Gerekçe: seçim ölçütü birincil ölçütle **ilişkili olmamalı**. Router sağlığı bir ön-koşuldur,
  karar ölçütü değildir; val loss ise doğrudan `s_i`'nin kendisidir.
* Üç koşulun **hepsinin** tarama sonucu raporlanır (V11), yalnızca kazananın değil.
* Bu seçim **aile 3'e girmez ama sayılır**: V11'in raporladığı "taranan FOTON varyantı sayısı"
  bu 3'ü içerir.

**Uyarı, kayda geçiyor:** V7'nin iki regularizasyon koşulu da hipotezi **cezalandırıyor** olabilir.
`load_balancing_loss` mikro-batch başına hesaplanıyor; 16 dizi üç alandan çok terimli çekildiği için
(9 kod, 4 NL, 3 matematik gibi) **mükemmel alan yönlendirmesi maksimum dengesizlik gibi görünür** ve
aux terimi tam olarak hipotezin öngördüğü davranışı bastırır. Entropi bonusu da dizi başına
posterior'u düzleştirir, yani anti-uzmanlaşmadır. M bu sorunla karşılaşmaz (32.768 token × 16 expert
dengeye bol yer bırakır). Bu yüzden "regularizasyon yok" koşulunun kazanması **beklenen** sonuçtur
ve kazanması bir bulgu sayılmaz.

**v3'ün (token düzeyi 16 expert) sonucu buraya TAŞINMAZ.** v3'ten devralınan tek şey *yöntemdir*:
null'lu ölçütler (ham `consistency`/`mi_bits` karşılaştırılamaz, kendi bağımsızlık tabanlarına göre
okunur), permütasyon karşı-olgusalı, ve "ölçütün işareti rejime bağlıdır" uyarısı. Sayı taşınmaz:
token düzeyi 16 expert ile sequence düzeyi top-1 üç modül farklı rejimlerdir.

### O-X1. X kolunun gather bedeli ayrı ölçülür

Eğitim koşularından **bağımsız** bir modül düzeyi mikrobenchmark: A (bitişik 256), B (dilated 256),
C (128+128) ve H'nin bitişik penceresi için, aynı `d_model`/head sayısında ve **eşit bağlantı
sayısında** ileri+geri süre. Amaç "gather mı yavaş, yoksa FOTON'un kendisi mi" sorusunu ayırmak.
Bu ölçüm karar koşularından **önce** yapılır ve sonucu raporda modül başına tok/s olarak verilir.

### V8. Ölçek basamakları

Ölçüm **pinli ayarda yeniden yapıldı**: mikro-batch **16**, vocab **32768**, 3 modül × 2 katman
FOTON ek maliyeti. Artifact `experiments/foton-v02-prereg/vram_ladder_mb16_32k.json`.

| basamak | d_model | n_layer | n_head | ffn_hidden | expert_hidden | MoE **ölçülen** VRAM | MoE tok/s | +FOTON (düzeltilmiş) | not |
|---|---|---|---|---|---|---|---|---|---|
| **S0** (karar) | 512 | 8 | 8 | 1408 | 352 | **8.02 G** | 50.5k | ~8.23 G | bu ön-kaydın ölçeği |
| **S1** (tek yükseltme) | 640 | 10 | 10 | 1760 | 440 | ~10.65 G | 31.5k | **~10.98 G** | sığıyor, pay ~0.9 G |
| S2 | 768 | 12 | 12 | 2112 | 528 | ~14.03 G | 3.1k | ~14.49 G | **MoE sığmıyor** |
| S3 | 896 | 14 | 14 | 2464 | 616 | ~18.40 G | 0.4k | — | OOM |

**Prob ölçümü gerçek tepeyi SİSTEMATİK OLARAK az gösteriyor — kayda geçiyor.** `scripts/vram_ladder.py`
yalnızca eğitim adımını ölçüyor; gerçek koşular eval de yapıyor ve eval'de `logits.float()` tensörü
`16 × 512 × 32768 × 4 B = 1.07 GB` ayırıyor. Ölçülen fark: S0'da MoE **+0.90 G** (prob 7.12 →
gerçek 8.022), dense **+0.71 G** (5.768 → 6.478). Yukarıdaki tablo S0 için **gerçek** koşu kaydını,
S1–S3 için prob + MoE offset'ini (+0.90 G) veriyor ve hangisi olduğu ayrıca yazılıdır. `foton-d`/
`foton-m`'in gerçek tepeleri beş seed'de de tam olarak 8.022 / 6.478 GB (sıfır yayılım).

`vram_limit_frac = 0.92` ile kullanılabilir tavan **11.87 GB**. S1'de FOTON kolu ~10.98 G ile sığıyor
ama payı ~0.9 GB; yükseltme hakkı kullanılırsa bu pay raporlanır ve OOM hâlinde V11a(iii) gereği
**bloğun tamamı** yeniden koşulur.

Merdiven 512/640/768/896 olarak seçildi çünkü `expert_hidden` 8'in katı olmalı (352/440/528/616);
aksi hâlde `torch._grouped_mm` sessizce döngü yoluna düşüp MoE'yi haksız cezalandırıyor.
**Dört kollu kampanyanın VRAM tavanı 640×10**; S2'de MoE kolu koşulamaz, dolayısıyla S2'ye
"yükseltme" diye gidilemez ve BELİRSİZ SONUÇ bölümündeki tek yükseltme hakkı S1 ile tükenir.

**Denetime bir düzeltme:** denetim bu tablonun mikro-batch 32'de ölçüldüğünü ve "S2 sığmıyor"
sonucunun o ayarın artefaktı olduğunu söyledi. Yeniden ölçüm bunu **doğrulamadı**: mikro-batch 16'da
512×8 dense tepesi 5.77 G çıkıyor, `vram_ladder_32k.json`'daki değer 5.768 G — yani o dosya da
mikro-batch 16'da ölçülmüş. Denetimin mikro-batch etkisine dair kanıtı (1.64× bellek, 18× tok/s)
**vocab 8192** dosyalarından geliyordu ve 32k tablosuna uygulanmıyor. Geçerli olan bulgu MAJOR 7'ydi
(prob gerçek tepeyi az gösteriyor) ve o düzeltildi.

### V8b. `L*(S1)` — kural ŞİMDİ yazılıyor, sayı S0 sonucundan sonra

Yükseltme hakkı kullanılırsa `L*(S1)` şu kuralla hesaplanır ve **FOTON'un S1 sonuçlarına
bakılmadan** "Sapmalar"a yazılır:

1. **S1 referans bütçesi:** `max_steps(S1) = 4375` — S0 ile **AYNI**. Bütçe token cinsinden sabit
   tutulur, token/aktif-parametre oranı sabit tutulMAZ. Gerekçe: v01 O7 oranı sabitliyordu
   (S0 3.282) ve bu S1'in bütçesini 5562 adıma çıkarıyordu; ama o zaman %80 noktası (4449.6) bir
   eval noktasına düşmez ve `L*` yeniden interpolasyon yargısına bağlanır — v02'nin bütçe seçimiyle
   ortadan kaldırdığı tam o yargı. 4375 adımda %80 = adım 3500 = 14 × 250, yine tam eval noktası.
   Yan etki kabul ediliyor ve raporlanıyor: S1'de token/aktif-parametre 143.36M / 71.2M = **2.014**
   (S0'da 3.375), yani S1 daha az eğitilmiş bir operasyon noktasıdır. Bu bir **ölçekleme iddiası
   değildir**; iki basamak arasında karşılaştırılan şey `s̄`'nin TRENDİdir, mutlak val loss değil.
2. **En iyi baseline S1'de yeniden belirlenir:** S0'da M idi; S1'de D ile M aynı kuralla
   (referans bütçede daha düşük maskeli val loss) karşılaştırılır ve kazanan `L*(S1)`'i verir.
   S1'de D kazanırsa **payda D olur** ve bu açıkça yazılır — "D payda değildir" ifadesi S0'a
   özgüdür, evrensel bir kural değil.
3. **Aynı ölçüt:** `loss_prefix_skip = 64`, `batch_size × eval_iters = 1280`, `eval_interval = 250`.
4. Seed kümesi S0'ın aynısı: **{11, 22, 33, 44, 55}** (B3'ün {66,…,110} kümesi yalnızca kesinlik
   ölçütü tetiklenirse kullanılır ve ölçek yükseltmesiyle karıştırılamaz).

### V9. Kalibrasyon

Tag'den **önce**, yalnızca baseline koşularının varyansından yapıldı; hiçbir FOTON verisi
kullanılmadı. Betik `scripts/calibrate_v02.py`, tam çıktı
`experiments/foton-v02-prereg/calibration.log`. Sayılar **maskeli ölçütten ve ikinci baseline
kümesinden** (`checkpoints/foton-v02r/`) türetildi.

**(1) Aynı konfigürasyonun seed'ler arası gürültüsü.** `foton-m`, 5 koşu: val@3500 sd **0.002949**;
`N*` (L*'e ulaşan token) **114.771M ± %0.91** — bütçenin %80.058'i, tanım gereği %80 olması
gerekiyordu, yani bütçe seçimi çalıştı. Eğri eğimi `L*` civarında
d(val)/d(ln token) = **−0.3346 nat**, yani %5 tasarruf ≈ **0.01716 nat**.

**Eğim aralığa duyarlı — kayda geçiyor.** −0.3346 değeri [3250, 3750] penceresinden, yani `L*`
kesişiminin etrafından. Geniş pencere [2500, 4375] de −0.3383 veriyor, ama [3500, 4375] sekantı
**−0.2385**: eğri sonlara doğru yatıklaşıyor, tek bir kuvvet yasası değil. Doğru olan kesişim
etrafındaki yerel eğimdir, ama dönüşüm **±%40 aralığa duyarlı**: 5 puan ↔ 0.0172 nat (yerel) ya da
0.0122 nat (geç sekant). **Karar bu dönüşüme BAĞLI DEĞİLDİR:** `s_i` doğrudan `N*`'dan hesaplanıyor,
nat'tan çevrilmiyor. Eğim yalnızca yorum ve güç çerçevelemesi için var; duyarlılık raporlanıyor.

**(2) Eşleştirilmiş `s_i`'nin sd'si — ölçüldü.** `L*`'te ölçülemedi çünkü **dense kolu `L*`'e
ulaşmıyor**. Gürültü kalibrasyonu için **iki kolun da ulaştığı** yardımcı hedefte ölçüldü:
`L_aux = 2.8929` (dense'in kendi %80 noktası, maskeli ölçütte). Karar hedefi **değildir**; izinli
kullanımları V11'de sayılıdır.

| | N* ortalama | bütçe payı | sd |
|---|---|---|---|
| M (MoE) | 98.75M | %68.9 | %0.44 |
| D (dense) | 114.82M | %80.1 | %0.86 |

`s_i` (M'in D'ye göre tasarrufu): 12.46 / 14.03 / 14.86 / 14.22 / 13.83 → ortalama **+13.88 puan**,
**SD 0.88 puan**, %95 GA [+12.78, +14.98], p < 1e-5, karar kuralı `'başarılı'`.
→ **Ölçülmüş eşleştirilmiş sd(s_i) = 0.88 puan** (gerçek iki mimari arası; v01'de 1.15–1.37 idi).

**Bu sd'nin FOTON'a transferi bir VARSAYIMDIR, ölçüm değil — üç sınırı yazılıyor:**
(a) `L_aux`'ta ölçüldü, `L*`'te değil; M'in `N*` sd'si `L*`'te %0.91, `L_aux`'ta %0.44, yani hedef
seçimi gürültüyü iki kat değiştiriyor ve `L_aux` daha SESSİZ olanı — transfer bu yönde iyimser.
(b) Tek bir kol çifti (M↔D), df=4; gerçek sd'nin %95 GA'sı χ²₄'ten **[0.53, 2.53] puan**.
(c) FOTON'un kendi `N*` sd'si ölçülmedi ve v01 O9 "FOTON farklı bir mimari olduğu için sd'nin daha
büyük çıkması beklenir" diyor. Bu yüzden güç tablosu 0.88'in yanı sıra **3.0 ve 5.0**'da da
veriliyor, ve V12'nin σ taraması 6.0'a kadar gidiyor.

**(3) Karar kuralının gücü** (Monte Carlo 200.000, ön-kayıttaki bileşik kural, **kesinlik ölçütü
dahil**). Hücreler "başarılı + güçlü başarı" oranı:

| gerçek etki | sd=0.88, n=5 | sd=0.88, n=10 | sd=3.0, n=5 | sd=3.0, n=10 | sd=5.0, n=5 | sd=5.0, n=10 |
|---|---|---|---|---|---|---|
| 0% | %0.0 | %0.0 | %0.0 | %0.0 | %0.7 | %0.1 |
| +2% | %0.0 | %0.0 | %1.2 | %0.1 | %4.9 | %2.8 |
| **+5%** | **%49.8** | %49.9 | %47.7 | %50.1 | %32.6 | %49.5 |
| +7% | **%100.0** | %100.0 | %91.6 | **%98.2** | %61.7 | **%89.5** |
| +10% | %100.0 | %100.0 | %100.0 | %100.0 | %90.6 | %99.9 |

**DÜZELTME, İKİ KEZ, kayda geçiyor.** Bu tablonun ilk hâli **kaldırılmış** mutlak sd tripwire'ının
(`sd > 5`) sayılarını taşıyordu: `scripts/calibrate_v02.py` bir kez eski kuralla koşulmuş, tablo o çıktıdan
yazılmış, betik sonra kesinlik ölçütüne çevrilmiş ama tablo güncellenmemişti. Fark kötümser
sütunlarda büyük — sd=5.0'da +%7 için %46.4 yerine **%61.7**, +%10 için %58.4 yerine **%90.6**
(32.2 puan). Yön FOTON'un lehine, yani hata gücü **az** gösteriyordu; yine de dondurulmuş bir
belgede kaldırılmış bir kuralın sayılarının durması kabul edilemez. İlk düzeltmede yalnızca `+5/+7/+10` satırları güncellendi ve `0%` ile `+2%` satırlarında **beş
hücre** eski kuralın sayılarıyla kaldı (`%0.6` yerine `%0.7`, `%1.9` yerine `%2.8` gibi; en büyük
sapma 0.9 puan). İkinci denetim bunu buldu; artık **otuz hücrenin tamamı**
`experiments/foton-v02-prereg/calibration.log` ile aynıdır ve `scripts/check_prereg_numbers.py`
bunu satır/sütun ayrımı yapmadan **hepsi için** denetliyor (ilk hâli yalnızca güncellenen üç satıra
bakıyordu, yani hatayı yapısal olarak göremiyordu). `rule_reachability.py --crosscheck` ayrıca
vektörize kuralı `decide()` yoluna karşı doğruluyor (fark 0.00 puan).

GA yarı-genişliği: n=5'te **1.03 puan**, n=10'da 0.61 puan (Monte Carlo ortalaması). Ölçülen sd'nin
ima ettiği tam değer `t(4)·0.88/√5` = **1.09 puan**; tabloda 1.03 çıkıyor çünkü örnek sd'si yanlıdır
(`E[s] = c₄σ`, c₄(5) = 0.940). Bağlayıcı yorumda **1.09** kullanılır — ilk hâl Monte Carlo
ortalamasını (0.87) sd'nin ima ettiği değer gibi yazıyordu.

**Bu tablonun kapsamı:** hücreler birincil-1'in karar tablosunu **tek başına** (Holm'suz, tek
bakışlı) tarar. Holm'un birincil-1'e etkisi ölçülen gürültüde sıfırdır (s̄ ≥ 5 ve σ = 0.88 iken
p₁ ~ 1e-4, ilk adımın .025'inden çok küçük); etkisi yalnızca σ ≳ 3 rejiminde ve s̄ eşiğe yakınken
görülür ve yönü muhafazakârdır (başarı oranını DÜŞÜRÜR). Bileşik kural `decide_primary()`'dedir.

**Gürültü +%5 eşiği için bağlayıcı değil.** Bağlayıcı olan, v01'de olduğu gibi, `s̄ ≥ 5`
**nokta-tahmin** şartıdır: gerçek etki tam eşikte iken güç ~%50'dir ve **seed sayısıyla düzelmez**
(n=10'da %49.9). Güvenilir biçimde saptanabilen en küçük etki **≈ +%7**. Ne eşik yükseltiliyor ne
seed artırılıyor; gerekçe v01 O9'la aynı.

**Ama seed artırmanın YARARLI olduğu bir rejim var ve tablo onu gösteriyor:** sd = 3.0'da +%7'lik
gerçek bir etkiyi saptama gücü n=5'te %91.6, n=10'da **%98.2**; sd = 5.0'da %61.7 -> **%89.5**.
Yani FOTON beklendiği gibi gürültülü çıkarsa kesinlik ölçütünün çaresi gerçekten çalışır. İlk hâl yalnızca sd = 0.88
sütununun n=10'unu gösterip "seed artırmak düzeltmez" diyordu; bu, en iyimser sütunla savunulan
bir iddiaydı. Doğru ifade: **eşikte düzeltmez, eşiğin üstünde düzeltir.**

**Referans nokta:** bu korpusta gerçek bir mimari iyileşmenin büyüklüğü ölçüldü — MoE, dense'e göre
`L_aux` hedefine **%13.88 daha az FLOP** harcıyor (p < 1e-5). FOTON'un +%5 eşiği, bu ölçekteki bir
etkinin yaklaşık üçte biridir. Bu karşılaştırma elma-armuttur ve **yalnızca ölçek okumak için**
kullanılabilir: %13.88 M↔D ve `L_aux`'ta, FOTON'un %5'i X↔M ve `L*`'te.

**(4) Yeniden koşum sadakat kontrolü.** `L*` iki kez türetildi: maskesiz ölçütle (ilk küme,
`checkpoints/foton-v02/`) ve maskeli ölçütle (ikinci küme, `checkpoints/foton-v02r/`). İkinci küme
her eval noktasında **iki** metriği de logladığı için sadakat doğrudan ölçülebiliyor:

| karşılaştırma | değer |
|---|---|
| maskesiz `L*`: dondurulmuş → yeniden koşum | 2.908229 → 2.906826, kayma **−0.0014 nat** |
| v2'de ölçülen aynı-komut val tekrar zarfı (100 adım) | 6.94e-03 nat |
| adım 3500'de en büyük tek sapma | 7.3e-03 nat |
| bütün eval noktalarının maksimumu (19 × 10 koşu) | 1.48e-02 nat |

Yeniden koşum **bit-birebir DEĞİLDİR** ve bu beklenen: GPU indirgeme sırası deterministik değil,
4375 adımda yörüngeler ayrışıyor. Ölçüt şu: `L*`'in kendisi tekrar zarfının **beşte biri** kadar
kaydı (0.0014 vs 0.0069), yani kararın bağlı olduğu nicelikte sadakat sağlandı. Tek nokta
maksimumu zarfın iki katı, ama o zarf 100 adımda ölçülmüştü ve 4375 adımda büyümesi bekleniyor.

**Maskeleme etkisi:** −0.0710 nat (2.906826 → 2.835815). İlk 64 pozisyon en zor pozisyonlardır, bu
yüzden maskeli ölçüt sistematik olarak daha düşüktür. Dört kolda da aynı maske uygulandığı için bu
bir **seviye kaymasıdır**, karşılaştırmayı etkilemez.

**(5) Birincil-2'nin (X↔H) sd'si ve MDE'si** — BİRİNCİL ÖLÇÜT bölümünde tablosuyla yazılı.
Özet: kol başına 0.88, ρ = +0.07 → fark sd'si **1.20 puan**; Holm'un ilk adımı α=.025 olduğu için
n=5'te **%97.5** GA yarı-genişliği **1.88 puan** = MDE çıtası (α=.05 varsayımıyla 1.49 olurdu).

### V10. Tag'den sonra nesi değişebilir

**Değişemez** (liste denetimden sonra genişletildi; eklenenler **koyu**):
`L*`, **`L_aux`**, `F_base` (referans sayı) ve `F_base,i`'nin tanımı, korpus (hash'ler) ve Tier A
kısıtı, referans bütçe, birincil ölçüt ve istatistik, **iki birincil uç nokta ve Holm aile
büyüklüğü**, test aileleri ve sayısı, karar tablosu ve eşikleri, **kesinlik ölçütü (tek biçim,
`ub95` ile)**, **`ub95`'in tanımı ve yavaş/sansürlü seed muamelesi**, **birincil-2'nin dört durumu,
MDE çıtası ve başarısız-seed kuralı**, **vazgeçme koşulunun iki yolu**, **her satırdan sonraki adım
tablosu**, **sahibin seçiminin S1'den önce loglanması kuralı**, **1× sınırı teşhisinin bandı
(±%10) ve tanımı**, **paydanın kaynağı**, **veri taşıma kuralı (seed kimliğiyle anahtarlı)**, seed kümeleri,
ölçek basamakları, oracle tanımı, **`src/stats.py`'deki karar fonksiyonlarının DAVRANIŞI**
(`n_star`, `resolve_n_star`, `savings`, `ub95`, `indeterminate`, `decide`, `contribution`,
`decide_primary`, `abandon`, `multiplicity`, `boundary_diagnostic` ve yardımcıları `Obs`, `keyed`, `paired`,
`ci95`, `t_crit`, `t_crit_alpha`, `_wbar`: tag'den sonra yalnızca gözlemsel eşdeğer refactor
yapılabilir; davranış değişikliği sapmadır ve testler + mutasyon sınavı bunu yakalamak içindir),
ve mimari spesifikasyonu —
şunları AÇIKÇA içerecek şekilde: **modül desen parametreleri (A: yerel W=150; B: dilated r=2, W=256;
C: yarı-yerel W=256 + yarı-strided s=4, W=256)**, **router girdisi ve prefix uzunluğu k=64**,
**gate işlemi (ham `p`, straight-through)**, **katman bölüşümü (3 paylaşılan + 2 yönlendirilen +
3 paylaşılan)**, **paylaşılan blokların tam nedensel olması**, **`loss_prefix_skip = 64`**, top-1,
sıralama kuralı.

Bu genişletme gerekliydi: ilk hâl "modül grafikleri" diyordu ama grafikler bir **aile**, bir grafik
değil. B'nin dilation rate'i, C'nin stride'ı ve head bölüşümü, router'ın girdisi, katman sayısı ve
paylaşılan blokların attention tipi yazılı olmadığı sürece "mimari donmuştur" ifadesi boştu; üstelik
bunların hepsi birincil ölçütü oynatır (B'nin rate'i 2 → 3 olsa bağlantı sayısı 128.50 → 85.83'e
düşer ve modüller arası eşitlik bozulur).

**Değişebilir:** FOTON'un uygulama ayrıntıları — **yalnızca gözlemsel eşdeğer olanlar**: kernel
seçimi, bellek düzeni, maske önbellekleme, kod refactor'ları. VRAM için mikro-batch/grad_accum
bölüşümü (V1'in üç şartıyla). Eval sıklığının **dört kolda birlikte** artırılması.

**"Uygulama ayrıntısı" sayılMAYANLAR** (denetim bunları o başlığa sokabileceğimizi gösterdi):
router init ölçeği, gate sıcaklığı, exploration gürültüsü, gate normalizasyonu, `compile`,
`vram_limit_frac`, `moe_impl`. Bunlar V2'nin kapalı listesinde ya da yukarıdaki donmuş mimari
spesifikasyonunda yer alır; değiştirilmeleri sapmadır.

**Karar koşularından önce yapılması zorunlu** (her biri commit edilmiş ve hash'i `pins.json`'da):
FOTON'un config'i `src/config.py`'de preset olarak sabitlenir ve **SHA-256'sı bu bölüme yazılır**;
`src/flops.py` genişletmesi ve `docs/flops.md` türetimi; O-X1 mikrobenchmark'ı; `src/attn_patterns.py`
ile modül bağlantı sayılarının doğrulanması; ve **bütün test dosyalarının geçmesi**
(`tests/test_stats.py`, `test_stats_v02.py`, `test_data_multi.py`, `test_attn_patterns.py`,
`test_run_ledger.py`, `test_verify_pins.py`, `test_campaign_queue.py`); `scripts/check_prereg_numbers.py`
ve `scripts/verify_pins.py --verify` sıfır hata ile; ve **`scripts/mutation_guards.py`'nin bütün
mutantları yakalaması** (V13).

    __DOLDURULACAK: FOTON config SHA-256__   (`foton-h` ve `foton-x` preset'leri henüz yazılmadı)

**Dürüstlük notu, iki kez:** (i) bu hash config'in **hash'lendikten sonra** değişmediğini kanıtlar,
H'nin sonucunun onu etkilemediğini kanıtlamaz — o güvence yukarıdaki sayısal spesifikasyonun
config'i TÜRETİLEBİLİR kılmasındadır; (ii) bu dosyanın kendisi `foton-v02-prereg` tag'i atılana
kadar değiştirilebilir ve tag öncesi hiçbir satırı "dondurulmuş" değildir. Tag atılmadan önce
yazılan "dondurulmuştur" cümlesi bir niyet beyanıdır, bir kanıt değil.

### V12. Her durdurma kuralı ULAŞILABİLİRLİK SINAVINDAN geçer (yöntem şartı)

**Kural.** Bu ön-kayıtta (ve sonraki ön-kayıtlarda) yazılan her durdurma, tripwire, vazgeçme ve
yükseltme kuralı, kabul edilmeden önce **makul bir σ aralığında sayısal olarak sınanır** ve sonucu
kayda geçer. Betik: `scripts/rule_reachability.py`, çıktı
`experiments/foton-v02-prereg/reachability.log`, genel yardımcı `src/stats.reachability()`.

**Neden bu bir yöntem şartı hâline geldi.** Aynı hata iki kez yapıldı:

* **v01 EK1-B10:** vazgeçme koşulu "iki ardışık ölçek" kısmı yüzünden aritmetik olarak ulaşılamazdı.
* **v02'nin ilk hâli:** aynı koşulun GA kısmı yüzünden yine ulaşılamazdı. σ ≥ 1.72'de tam başabaş
  bir FOTON asla terk edilemiyordu ve n=10 kurtarmıyordu.

**Örüntü belli: bir NOKTA TAHMİNİNİ bir ARALIK SINIRIYLA "ve" ile bağlayan durdurma kuralları,
varyans büyüdükçe ölür.** Aralık genişledikçe aralık-şartı sağlanamaz hâle gelir, ve bunu telafi
etmek için nokta tahmininin gitmesi gereken yer gerçekçi olmayacak kadar uzaklaşır. Bu gözle
görülmüyor; hesaplanınca görülüyor. İkinci kez aynı hatayı yaptıktan sonra kural yazmak yetmiyor.

**Sınav üç şeyi ölçer ve üçü de rapora girer:**

1. **Ulaşılabilirlik** — kural, tetiklenmesi GEREKEN durumda makul σ aralığında tetikleniyor mu?
2. **Çarenin yönü** — kuralın kendi çaresi (seed artırmak) kuralı GEVŞETİYOR mu, sıkıştırıyor mu?
   Mutlak sd tripwire'ı bu sınavı geçemedi: çare tripwire'ı daha olası yapıyordu (σ=6'da
   %59.7 → %71.5). Bir kuralın çaresi kuralı sıkıştırıyorsa o kural kendi çıkmazını üretir ve
   **kabul edilmez.**
3. **Yanlış pozitif** — gerçek etki sıfırken kural yanlış yönde tetiklenme oranı %5'in altında mı?

**σ aralığı nasıl seçilir:** ölçülen `sd(s_i)`'den başlar ve **v01 O9'un beklentisine** kadar
taranır (FOTON farklı bir mimari olduğu için sd'sinin büyük çıkması beklenir). Bu ön-kayıtta
0.88 → 6.0. Tek bir σ'da sınamak yetmez; tam olarak büyük σ rejiminde kurallar ölüyor.

**SAĞLAM BİÇİM, tercih edilen kalıp:** bir durdurma kuralı, bir **eşdeğerlik** ifadesi olarak
yazılır — "güven aralığının TAMAMI eşiğin bir tarafında" — nokta tahmini ile aralık sınırının
birleşimi olarak değil. Bu biçimde n artırmak daima yardım eder, ve kural tetiklenmediğinde sonuç
**"hipotez ayakta" değil "yetersiz kesinlik"** olarak ayrışır. Bu ön-kayıtta uygulandığı üç yer:
vazgeçme koşulu (`ub95 < +%5`), kesinlik ölçütü (satır 1; aynı `ub95`), ve birincil-2'nin null
yorumu (`GA üst < MDE`).

**Gauss-dışı kör nokta: başarısız seed'ler.** σ taraması yalnızca Gauss gürültüsünü tarar; bu
ön-kayıtta kuralların ÜÇ kez öldüğü yer ise Gauss-dışıydı — hedefe ulaşamayan seed'ler. (i) Sansürlü
ikame düz t-aralığına girince vazgeçme sansürlü her senaryoda ulaşılamazdı; (ii) yalnızca sansürsüz
alt örneğe bakınca 3/5 sansür terk edilemiyordu; (iii) yavaş seed sıradan gözlem sayılınca yavaş
seed'li her kampanya terminal "yetersiz kesinlik"e gidiyor, "başarısız" ve vazgeçme ulaşılamıyordu.
Bu yüzden sınav, σ tablolarının yanında **başarısız-seed senaryo tablosu** taşır (sansürlü, yavaş,
ikisi birlikte, n=10, 1× ve 2× sınırları, başarısız seed'in İLK anahtarda olduğu hâl, iki sınır
seed'i; 20 senaryo) ve her senaryonun beklenen sonucu (**satır**, tek ölçekte terk, iki ölçekte terk)
**assertion** olarak yazılıdır; tek başarısız seed'de 2× sınırındaki süreklilik, sonucun başarısız
seed'in HANGİ seed olduğundan bağımsızlığı ve aritmetik tavan da öyle. (Tur 3 denetimi buldu: satır
16 senaryonun yalnız 12'sinde assertion'dı, ve başarısız seed her fixture'da en büyük anahtardaydı —
ikisi de düzeltildi.) Yeni bir durdurma kuralı yazan, o tabloya senaryo eklemek zorundadır.

### V13. Her guard MUTASYON SINAVINDAN geçer (yöntem şartı)

**Kural.** Bu ön-kaydın dayandığı her doğrulayıcı — regresyon testleri, `check_prereg_numbers.py`,
`verify_pins.py`, `rule_reachability.py`'nin assertion'ları, `make_campaign_queue.py`'nin denge
kontrolü — için, koruduğu şey **kasten bozulur** ve doğrulayıcının bunu **yakaladığı gösterilir**.
Betik `scripts/mutation_guards.py`, kayıt `experiments/foton-v02-prereg/mutation_log.md`.
**Yakalamadığı kanıtlanmamış guard, guard değildir.**

**Neden bir yöntem şartı.** Bu belgenin denetim geçmişinde aynı sınıftan ALTI olay var:

* `%33.8` guard'ı iki ondalıklı deseni (`%33.79`) arıyordu ve belgedeki gerçek hatayı (`%33.8`)
  kaçırdı;
* `reachability.log` kendi defect'ini (`4 × +%7 + 1 sansür … terk EVET`) basıp geçti, çünkü
  beklenti düzyazıydı, assertion değildi;
* güç tablosu kontrolü tam olarak GÜNCELLENMİŞ satırlara bakıyordu, hatanın kaldığı satırlara değil;
* kuyruğun Latin kare assertion'ı yalnızca `blok % kol == 0` iken çalışıyordu, yani kampanyanın
  gerçek biçiminde (5 × 4) hiç çalışmıyordu — mutasyon sınavı tasarlanırken bulundu;
* `check_prereg_numbers.py`'a eklenen "her `L*` geçişini denetle" regex'i yanlış kaçışla yazılmıştı
  (literal soru işareti arıyordu) ve SIFIR eşleşmeyle sessizce geçiyordu;
* aritmetik tavan örneğindeki "tavansız biçim" sayısı koda, teste ve bu metne ELLE **+31.3** diye
  yazılmıştı; tarama onu hesaplamaya başlayınca **+31.4** çıktı. Tarama betiği sayıyı basıyor ama
  hesaplamıyordu — elle yazılmış bir sabit, onu denetleyen guard'ı kandırır.

Mutasyon sınavının ilk koşumu ayrıca iki gerçek boşluk buldu ve ikisi de kapatıldı: **+%5 başarı
eşiğinin kendisi** hiçbir testle kilitli değildi (`SUCCESS = 5.0 → 4.0` mutantı hayatta kalıyordu,
çünkü hiçbir test [4, 5) aralığında bir ortalama sınamıyordu), ve `evaluate()`'in prefix maskesi ile
dondurulmuş preset değerlerinin hiç regresyon testi yoktu.

Son durum (`mutation_log.md`): **136 mutant — 135 yakalandı, 1 kanıtlı eşdeğer, 0 hayatta kaldı.**
Eşdeğer mutant (S27) kodu değiştiren ama davranışı değiştirmeyen mutanttır; kanıtıyla kayıtlıdır ve
her koşumda yeniden denenir.

**Konuma göre eşleme sınıfı ayrı bir bloktur (E01–E21).** O hata üç ayrı fonksiyonda üç kez çıktı
(V4'teki taşıma kuralı); yapısal düzeltmeden sonra mutantlar YAPIYI geri bozar: `paired()` /
`savings()` / `contribution()` / `abandon()`'da eşlemeyi ekleme sırasına göre yapan (E01–E04, E06,
E07), eşlemeyi bir kaydıran (E05), listeyi
sessizce `dict(enumerate(…))` sayan ya da kapıyı hiç koymayan (E08, E09), anahtar kümesi denetimini
kapatan (E10, E15, E16), kanonik sırayı kaldıran (E11), tekrar eden seed'i sessizce ezen (E12),
tanımsız durumu ya da +100'ü aşan tasarrufu kabul eden (E13, E14). **Tur 3 denetimi bu sınıftan İKİ
mutantın hayatta kaldığını buldu ve ikisi de artık katalogdadır:** `ub95()` içinde hızlı/sabit
ayrımını "ilk n_h değer hızlıdır" diye KONUMLA yapan mutant (E17; bütün fixture'lar başarısız seed'i
en büyük seed anahtarına koyduğu için 93 testin hepsini ve taramayı geçiyordu — o mutantta aynı
örnek yavaş seed s11'deyken "yüksek varyans / belirsiz", s55'teyken "başarısız" veriyordu) ve
anahtar kümesi denetimini tek yönlü yapan mutant (E18; H'nin fazla seed'i). E19–E21 "yanlış kullanım
patlasın" kapılarıdır: ön-kayıtlı seed kümesi, açık durum, `savings()`'te zorunlu durum. Aynı
turun bulduğu üçüncü test boşluğu (`resolve_n_star` 1× kesse bile 2× eğrisini okuyan mutant, yalnız
2× eğrisi DAHA ERKEN kestiğinde görünür) S48'in testine eklendi. Onları yakalayan testler önce
konum eşlemesinin o girdide GERÇEKTEN farklı sonuç verdiğini gösterir (yoksa hiçbir şeyi
kilitlemiyor olurlardı), sonra X ile H'yi, FOTON ile baseline'ı, ölçek sonuçları ile oracle
kazançlarını BAĞIMSIZ karıştırıp çıktının birebir aynı kaldığını denetler. Bunların hepsi gözden geçirmeyle değil, **bozup denemeyle** ortaya çıkar. Mutasyon sınavı geçici bir kopya
üzerinde çalışır; depo ve dondurulmuş artifact'lar değişmez.

**Denetim kaydı.** Karar-kritik yüzeyin üç turluk adversaryal denetimi, her turun HIGH sayısı, son
turun (tur 3: **HIGH = 0**) bütün MEDIUM/LOW bulguları ve her birinin akıbeti
`experiments/foton-v02-prereg/audit_log.md` dosyasındadır. Tur 3'ten sonra, tag'den önce yapılan
sertleştirmeler dördüncü bir denetimden geçmemiştir (tur 4 yoktur); geçerli girdide hiçbir kararı
değiştirmedikleri, denetlenen commit ile son hâlin 40.000 rastgele kampanyada karşılaştırılmasıyla
gösterilmiştir (fark 0) ve o kayıtta yazılıdır.

### V11a. Koşu sırası: dört kol İÇ İÇE koşulur (yöntem şartı)

Karar koşuları **kol bloğu hâlinde değil, seed bloğu hâlinde** koşulur:

```
D-s11, M-s11, H-s11, X-s11,  D-s22, M-s22, H-s22, X-s22,  ...
```

**Gerekçe ölçümle:** `foton-m`'in 5 baseline koşusunda (aynı konfigürasyon, yalnızca seed farklı)
medyan tok/s **52.8k–66.6k** arasında değişti; koşular arası CV **%9.02**, en yavaş/en hızlı
**1.263×**. Eğitim hızı seed'e bağlı olamaz, dolayısıyla bu koşu-oturumu etkisidir. Koşu içi dilim
analizi gürültünün **zaman-korelasyonlu** olduğunu da gösterdi: s11'in son iki diliminde başlayan
yavaşlık (64.7k → 50.5k) s22'nin ilk dört dilimine taşındı ve ortasında bitti — yani ~40 dakikalık
bir girişim penceresi koşu sınırını aşarak iki koşuya yayıldı.

**İKİ OTURUM ÖLÇÜLDÜ VE SORU CEVAPLANDI: bu bir ZAMAN etkisidir, kol etkisi değil.**

| oturum | D medyanları (bin tok/s) | D CV | M medyanları | M CV |
|---|---|---|---|---|
| 1 (kol bloğu) | 91.9, 92.0, 92.0, 92.0, 92.0 | **%0.07** | 62.6, **52.8**, 66.6, 65.9, 64.4 | **%9.02** |
| 2 (Latin kare) | 86.8, 86.6, 88.2, 91.8, 91.8 | **%2.89** | 63.3, 59.6, 66.6, 65.8, 66.7 | **%4.65** |

İlk oturumda yalnız M'in CV'si yüksekti ve bu iki açıklamaya açıktı: (a) girişim penceresi yalnızca
M'in koştuğu saatlerdeydi — **zaman** etkisi; (b) MoE VRAM sınırında olduğu için çevresel değişime
**duyarlı**, dense değil — **kol** etkisi. Blok sıralı tek bir oturumda ikisi karışıktı.

İkinci oturum ayrıştırdı: **D'nin CV'si %0.07'den %2.89'a ÇIKTI**, M'inki %9.02'den %4.65'e DÜŞTÜ.
Yani dense kol olarak kararlı değil; ilk oturumdaki %0.07 o oturumun sessizliğiydi. İki kolun da
değişkenliği oturuma göre değişiyor, dolayısıyla **kola özgü duyarlılık hipotezi desteklenmiyor.**
Ortalama hızlar da kaydı: D 92.0k → 89.0k (%−3.2), M 62.5k → 64.4k (%+3.1).

**Bunun ikincil ölçüt için sonucu ağır ve ön-kayda yazılıyor.** Aynı nicelik — `L_aux` hedefinde
M'in D'ye karşı duvar saati kaybı — iki oturumda:

| oturum | fark | %95 GA | p |
|---|---|---|---|
| 1 | +6.25 dk | [+2.33, +10.17] dk | .0115 |
| 2 | **+4.06 dk** | [+2.63, +5.49] dk | **.0014** |

İşaret ve anlamlılık aynı, **büyüklük 11 puan farklı** (%30.0 vs %18.6). Dolayısıyla: duvar saati
farkının **işareti** ölçülebilir, **büyüklüğü** oturuma bağlıdır. Raporda duvar saati büyüklüğü
verilirken oturum kimliği ZORUNLUDUR ve tek bir sayı olarak alıntılanamaz; `w̄`'nin GA'sı bu
oturum-arası kaymayı İÇERMEZ, yani gerçek belirsizlik raporlanan GA'dan büyüktür.

Seed bloğu + Latin kare bu yüzden bağlayıcı: aynı pencereyi dört kol paylaşır ve blok içi pozisyon
kolla karışmaz. **Maliyeti sıfırdır, yalnızca kuyruk sırası değişir.**

Bağlayıcı kurallar: (i) bir seed bloğu içindeki dört koşu **kesintisiz** ve **kuyruk dosyasındaki
sırayla** koşulur — o sıra `scripts/make_campaign_queue.py`'nin ürettiği **Latin kare kaydırmasıdır**
(blok 1: D,M,H,X; blok 2: M,H,X,D; blok 3: H,X,D,M; blok 4: X,D,M,H; blok 5: D,M,H,X), sabit
D,M,H,X DEĞİL. (Bu maddenin ilk hâli "aynı sırada (D, M, H, X)" diyordu ve üreteç Latin kareye
çevrildikten sonra da öyle kalmıştı: bağlayıcı metin ile kuyruk birbiriyle çelişiyordu.) Sabit
sıra blok-içi pozisyonu kolla TAM karıştırırdı: D hep soğuk GPU'da ilk, X hep ısınmış GPU'da son.
**Artık dengesizlik, sayısıyla:** 5 blok × 4 kolda tam denge aritmetik olarak İMKÂNSIZDIR (ilk dört
blok tam Latin kare, beşinci blok 0..3'ü bir permütasyonla ekler), ortalama blok-içi pozisyonlar
D 1.20 / M 1.40 / H 1.60 / X 1.80; birincil karşıtlıklarda fark X−M = 0.40, X−H = 0.20 pozisyon.
`check_balance()` her kolun her pozisyonu en az ⌊5/4⌋ = 1 kez gördüğünü assertion olarak denetler
ve ihlalde kuyruk üretilmez; (ii) duvar saati karşılaştırması **yalnızca aynı seed bloğu içinde** geçerlidir; (iii) bir
blok yarıda kesilirse (çökme, OOM, makine yeniden başlatma) **o bloğun tamamı** yeniden koşulur ve
her ikisi de koşu defterine yazılır; (iv) blok sırası dışında koşulan hiçbir koşu duvar saati
ölçütüne girmez; (v) **kol-etkisi mi zaman-etkisi mi** sorusu raporda cevaplanır: blok-içi
kola-özgü tekrar eden yavaşlık (kol etkisi) ile bloklar arası kayan yavaşlık (zaman etkisi)
ayrı ayrı gösterilir. Ayırt edilemiyorsa **ayırt edilemediği yazılır**, bir açıklama seçilmez.

### V11. Zorunlu raporlama

Karar tablosu satır adı birebir; `s̄`, sd, %95 GA, `ub95`, n, **sansürlü ve yavaş seed sayıları
(seed kimlikleriyle)**, birincil-2'nin durumu ve GA'sı, Holm sonucu, `w̄` ve GA'sı; taranan
FOTON varyantı sayısı ve tablosu; koşu defterindeki toplam/iptal/çöken koşu sayısı; her kolun
`config.json` SHA-256'sı ve `git_head`/`src_sha256`'sı; `batch_size × eval_iters`; `eval_interval`;
her kolun tam val eğrisi **ve alan başına val loss**; profiler-analitik FLOP farkı (her kol);
oracle kazancı ve null'ı; yapılan test sayısı (**"18 (2+5+11), keşifsel: k"**); modül başına tok/s
(O-X1); ve **D'ye karşı bağlam sayıları** (kapsam sınırı gereği zorunlu).

**HER ZORUNLU MADDENİN TANIMLI BİR ARTIFACT'I VAR.** İlk hâlde yalnızca satır adının bağlayıcı bir
biçimi vardı (`decide_primary()` döndürüyor); geri kalan her şey "raporlanır" diyordu ama dosyası, yolu ve
şeması yoktu, yani sessizce düşürülebilirdi. Tablo:

| zorunlu madde | artifact |
|---|---|
| birincil-1 satır adı (Holm'dan SONRA), `s̄`, sd, GA, `ub95`, n, sansürlü/yavaş seed sayısı, notlar; birincil-2 durumu, GA'sı, `p`; Holm | **`src/stats.decide_primary()`** çıktısının TAMAMI → `experiments/foton-v02/decision.json` (`decide()`'ın tek başına çıktısı bağlayıcı değildir: Holm'u atlar) |
| vazgeçme kararı, yolu, gerekçesi, `escalation_used` | `src/stats.abandon()` çıktısı → aynı dosya |
| **sahibin ölçek hakkı seçimi** (S1'den ÖNCE, zaman damgalı, gerekçeli) ve sıra denetimi | `experiments/foton-v02/owner_decisions.jsonl` (`scripts/owner_decision.py`; `--verify` çıktısı rapora girer) |
| **1× sınırı teşhisi**: ±%10 bandındaki seed'ler ve sınırın diğer tarafındaki karşı-olgu satırı/durumu (TEŞHİS, karar değil) | `src/stats.boundary_diagnostic()` çıktısı → `experiments/foton-v02/boundary_diagnostic.json` (bantta seed yoksa da yazılır: "bantta seed yok") |
| `s_i`, seed başına `N*`, **durum (hızlı/yavaş/sansürlü)**, `F_foton,i`, `F_base,i` ve paydanın kaynağı — seed kimliğiyle anahtarlı | `experiments/foton-v02/savings.json` (`src/stats.resolve_n_star()` + `savings()`) |
| `w̄(M)`, `w̄(D)` ve GA'ları, **kümülatif saatten** | `experiments/foton-v02/walltime.json` (`scripts/baseline_reference.py` yöntemi) |
| **koşu defteri** toplam/iptal/çöken/sürdürülmüş | `experiments/foton-v02-prereg/run_ledger.jsonl` (`scripts/run_ledger.py --report`) |
| taranan FOTON varyantı sayısı ve tablosu (V7) | `experiments/foton-v02/v7_ablation.json` |
| kol başına `config.json` SHA-256, `git_head`, `src_sha256` | `experiments/foton-v02-prereg/pins.json` (`scripts/verify_pins.py`) |
| kol başına tam val eğrisi + alan başına val loss | `checkpoints/foton-v02/*/metrics.jsonl` + `scripts/eval_domains.py` çıktıları |
| profiler − analitik FLOP farkı (kol başına) | `experiments/foton-v02/flops_crosscheck.json` (v01 B13'ün 5 puan kuralı) |
| oracle kazancı **ve null'ı** | `experiments/foton-v02/oracle.json` |
| teşhis D1–D11, Holm + BH, aile büyüklüğü pinli | `experiments/foton-v02/diagnostics.json` |
| modül başına tok/s (O-X1) | `experiments/foton-v02/ox1_microbench.json` |
| router logları (V6'nın altı alanı) | `metrics.jsonl`'ın `train` satırları, V6'daki alan adlarıyla |
| kol-etkisi / zaman-etkisi ayrışması (V11a-v) | `experiments/foton-v02/session_variance.json` |
| eğri kesişim uyarısı (v01 B15) | `savings.json` içinde seed başına bayrak |
| ulaşılabilirlik sınavı (V12) | `experiments/foton-v02-prereg/reachability.log` |

**Değerlendirilen checkpoint PİNLİ.** Alan başına val loss ve D1–D11 teşhisleri
**`last.pt`** (son adım) üzerinde hesaplanır, `best.pt` üzerinde değil. (Oracle kazancı bir
checkpoint'ten değil **eğriden** gelir; bkz. VAZGEÇME KOŞULU → "Oracle şartı".) Gerekçe: `best.pt` val
loss'a göre seçilir, yani ~18 eval noktasından biri sonucu en iyi gösterecek şekilde seçilebilirdi —
ve vazgeçme koşulunun üçüncü şartı (oracle kazancı < %2) tam olarak buna açıktı. `last.pt` seçim
içermez. (`N*` zaten eğrinin tamamından okunuyor, checkpoint'ten değil.)

**D'ye karşı bağlam sayısı nasıl raporlanır.** D `L*`'e ulaşmadığı için "D'ye karşı hedefe ulaşma
süresi" `L*`'te tanımsızdır ve **uydurulmaz**. Bunun yerine üçü birlikte verilir:
(i) D'nin tam bütçe sonundaki val loss'u ve `L*`'e olan **eksiği** (nat cinsinden, sansür açıkça
anılarak); (ii) **eşit token** bütçesinde FOTON − D val farkı (hedefe-ulaşma değil, aynı-bütçe
karşılaştırması); (iii) `L_aux = 2.8929` hedefinde FLOP ve duvar saati farkı.

**Bu üçünün hiçbiri karar tablosuna girmez, ve "D'yi yendik" cümlesi yalnızca (ii)'ye dayanarak
kurulabilir.** İlk hâl bu cümleyi "(ii) veya (iii)" diye yazıyordu ve bu, `L_aux`'un 60 satır
yukarıdaki çitini ("`L_aux` sonuç beyanı doğurmaz") doğrudan çiğniyordu. Dahası (iii) üzerinden
kurulan cümle **boştur**: `L_aux`'ta M zaten D'yi +%13.88 (GA [+12.78, +14.98]) yeniyor, yani
M'e sadece denk gelen bir FOTON kolu bile "D'yi %14 yendik" manşeti üretir — ölçülen şey FOTON
değil, MoE–dense farkıdır. (iii) yalnızca **`w̄`'nin D'ye karşı bileşeni** için kullanılır
("güçlü başarı"nın ikinci şartı) ve orada da FOTON'un kendi sayısı olarak değil, kartın
bağlam sayısı olarak okunur.

`L_aux`'un izinli kullanımları, tam liste: (a) gürültü kalibrasyonu (V9), (b) D'ye karşı bağlam
sayıları (bu madde), (c) ikincil ölçütün kapsam sınırı argümanı (İKİNCİL ÖLÇÜT bölümü — üçüncü
kullanım, ilk hâlde "yalnızca iki yerde" yazıyordu ve o sayım eksikti), (d) `w̄(D)`. Bunların
hiçbiri karar satırını belirlemez.

**Koşu-oturumu varyansı da raporlanır** (V11a'nın şartı): aynı konfigürasyonun seed'ler arası
tok/s yayılımı — medyan, sd, CV, en yavaş/en hızlı oranı — ve koşu içi dilim analizi
(gürültü zaman-korelasyonlu mu). Bu sayı olmadan duvar saati farkının anlamlı olup olmadığı
söylenemez. Baseline ölçümü İKİ oturumda yapıldı ve ikisi de raporlanır (V11a'daki tablo):
1. oturum `foton-m` CV %9.02 / `foton-d` %0.07; 2. oturum M %4.65 / D %2.89. İki oturumun BİRLİKTE
raporlanması zorunludur, çünkü tek oturum yanıltıcıdır: ilk oturum "MoE duyarlı, dense kararlı"
izlenimi veriyordu, ikinci oturum bunu çürüttü (D'nin CV'si 41 kat arttı). Kampanya raporunda
ayrıca **blok içi pozisyon etkisi** verilir: her kolun gördüğü blok-içi pozisyonların ortalaması
ve pozisyona göre tok/s eğilimi (Latin kare bunu dengelemek için var, dengelendiği gösterilmeli).

---

## TOPLAM KOŞU SAYISI VE GPU SAATİ

Ölçülen uçtan uca sürelerden (D 0.474 sa, M 0.673 sa; H ve X **ölçülmedi**, M mertebesinde varsayıldı
— FOTON'un yönlendirilen katmanları pencereli olduğu için daha hızlı olması da mümkündür):

| aşama | koşu | tahmin |
|---|---|---|
| S0 karar (5 seed × 4 kol, iç içe) | 20 | **12.5 GPU saati** |
| V7 router regularization ablasyonu (3 koşul × 3 seed, %50 bütçe) | 9 | 3.0 sa |
| tarama tier (3 seed × 4 kol, %50 bütçe) | 12 | 3.7 sa |
| O-X1 modül mikrobenchmark'ı | — | < 0.2 sa |
| **tag öncesi toplam (bu ön-kaydı üreten baseline'lar)** | **20** | **11.5 sa** (harcandı: iki küme) |
| S1 yükseltme hakkı (640×10), kullanılırsa | 20 | +19.9 sa |

**Karar yoluna girmesi planlanan toplam: 41 koşu ≈ 19.2 GPU saati**; yükseltme hakkı kullanılırsa
61 koşu ≈ 39.1 saat. İlk hâlde bu tahmin belgede **hiç yoktu** (denetim bulgusu), oysa kullanıcı
onu kampanya planının şartı olarak istemişti.

---

## Sapmalar

Ön-kayıttan her sapma, gerekçesi ve fark edildiği tarihle buraya eklenir; yukarıdaki bölümler
düzenlenmez.

**S1 — 2026-09-17: dense kolu 1× bütçede sansürlü işlendi, 2× koşu yapılmadı.**
v01 O3 sansürü **yalnızca 2× bütçe koşusundan sonra** tanımlıyor: *"Bir koşu referans bütçenin 2
katına kadar koşturulabilir. `L*`'e hâlâ ulaşmıyorsa `N*` = 2 × referans bütçe olarak sansürlü
kaydedilir."* `foton-d` 1× bütçede (4375 adım) `L*`'e ulaşmadı ve **2× koşu yapılmadı**; yani v02,
"sansürlü"yü "1× bütçede ulaşmadı" anlamında kullandı. Gerekçe: D karar karşılaştırmasının paydası
değil (payda M'dir), sansürü yalnızca bağlam sayılarını etkiliyor, ve 2× koşu 5 × 52 dk ≈ 4.3 GPU
saati ek maliyet demek. **Sonuç üzerindeki etkisi:** yok — D'nin `N*`'ı hiçbir karar satırına
girmiyor. **Kalan risk:** "dense bu bütçede `L*`'e ulaşmıyor" ifadesi doğru, "dense `L*`'e
ulaşamıyor" ifadesi ise bu veriyle kurulamaz; raporda birinci biçim kullanılacak.
**FOTON kolları için O3 aynen geçerlidir:** bir FOTON seed'i 1×'te ulaşmazsa 2× koşulur ve sansür
ancak ondan sonra kaydedilir.

**S2 — 2026-09-17: baseline'lar iki kez koşuldu; `L*` İKİNCİ kümeden türetildi.**
İlk küme (`checkpoints/foton-v02/`) maskesiz val loss ile koştu ve `L* = 2.9082` verdi. Router'ın
nedensellik ihlali bulunduktan sonra raporlanan ölçüt pozisyon 64..511 ile sınırlandı
(`loss_prefix_skip = 64`) ve `L*` o ölçütte yeniden türetilmek zorunda kaldı. Checkpoint'lerden
türetilemedi çünkü yalnızca adım 4375'in checkpoint'i saklanıyor, `L*` ise adım 3500'ün değeri.
Bu yüzden 10 baseline koşusu **yeniden koşuldu** (`checkpoints/foton-v02r/`). İkinci küme her eval
noktasında **iki** metriği de logluyor; maskesiz metriğin ilk kümenin eğrisini yeniden üretmesi
yeniden koşumun sadakat kontrolüdür ve sonucu V9'da raporlanıyor. İlk kümenin eğrileri silinmedi ve
`pins.json`'da hash'li. **Kalan risk:** ilk kümenin `git_head`'i kirli bir ağacı gösteriyor
(`1a592a2` + 14 değişiklik); ikinci küme `0a43847`'den sonra koştu, ancak ilk koşu (`r-d-s11`)
commit'ten önce başladığı için onun provenance'ı da kirli — o koşu kuyruğun sonunda yeniden
koşulacak ki 10 koşunun tamamı aynı commit'i göstersin.

---

*Aşağıdaki girdiler `foton-v02-prereg` tag'inden SONRA, 06 (uygulama) adımında, hiçbir FOTON koşusu
yapılmadan ÖNCE yazıldı. Tag'li gövde değişmedi (`scripts/check_prereg_frozen.py` bunu mekanik olarak
denetler). Girdi numaraları ölçek basamaklarıyla (S0/S1) karışmasın diye kodda ve raporda
"Sapma-S3" biçiminde anılır.*

**S3 — 2026-09-19: O-X1 bölümü ESKİ modül tasarımını anlatıyor; uygulanan tasarım DONMUŞ tablodur.**
O-X1 "A (bitişik 256), B (dilated 256), C (128+128)" diyor. Bu, denetimin reddettiği ilk tasarımdır:
nedensel maske altında **192.25 / 128.50 / 88.31** bağlantı verir, yani tam olarak ortadan kaldırılan
compute confound'un kendisi. Bağlayıcı olan MİMARİ tablosu ve V10'un donmuş parametre listesidir:
**A yerel W=150 (128.17), B dilated r=2 W=256 (128.50), C yarım yerel W=256 + yarım strided s=4 W=256
(128.38)**; H'nin deseni A'dır (O-X1'in "H'nin bitişik penceresi" ayrı bir şey değildir). Stale metnin
koda sızmaması için iki kilit: (i) `tests/test_foton_model.py` her modülün bağlantısını modelin SDPA'ya
verdiği maskeden SAYAR (128.4 ± 0.4) ve grafiği attention çıktısının girdiye fonksiyonel bağımlılığından
ölçer; (ii) `GPT.verify_foton_masks()` her model kurulumunda — yani her koşunun başında — donmuş
sabitleri ve sayılmış bağlantıyı denetler; eski tasarımla model KURULAMAZ. Desen parametreleri
config'ten ya da komut satırından ezilemez. O-X1 mikrobenchmark'ı (`scripts/ox1_microbench.py`) donmuş
desenleri ölçer ve sayım tutmazsa koşmayı reddeder.

**S4 — 2026-09-19: ölçümü ÜRETEN kodda metnin açık bıraktığı seçimler, ilk FOTON koşusundan ÖNCE sabitlendi.**
Tag'li metin aşağıdakileri yazmıyordu; her biri koşulardan sonra onarılamayacağı için burada kayıtlıdır.
Kural: metnin en düz okuması; iki okuma varsa FOTON'un ALEYHİNE olan; ve mümkün olan her yerde HAM veri
de loglanır ki başka bir okuma sonradan hesaplanabilsin.
* **Bloklar.** FOTON'un bütün blokları (6 paylaşılan + modüller) D kolunun bloğudur: dense SwiGLU(1408),
  MoE yok. "D ve M'in bloklarıyla birebir aynı" cümlesi yalnızca attention için doğrudur (M'in FFN'i MoE'dir);
  eşitlenen eksenler listesindeki `ffn_hidden` ve "aktif FLOP tek modüllü bir modelle aynı" bunu söylüyor.
* **Derinlik ve init.** `n_layer = 8` bir dizinin GEÇTİĞİ derinliktir (kurulan 12 blok değil); residual
  projeksiyonlarının init ölçeği `(2·8)^-0.5`, D/M ile aynı. H'nin üç modülü bağımsız init çekimleridir.
  Maskeler RNG tüketmediği için aynı seed'de H ile X bit düzeyinde AYNI başlangıç ağırlıklarından başlar.
* **Router.** Girdi: yönlendirilen ilk katmanın girdisi olan HAM residual akış (normalizasyon yok, ek
  parametre yok), pozisyon 0..63'ün eşit ağırlıklı ortalaması, gradyan gövdeye akar (detach yok). Tek
  doğrusal katman d→3, bias yok, init N(0, 0.02) (diğer bütün doğrusal katmanlar gibi), sıcaklık 1, gürültü
  yok, float32'de ve autocast dışında (`src/moe.py` ile aynı gerekçe). Eşitlikte en küçük indeks.
  V10'un "donmuş spesifikasyonda yer alır" dediği init ölçeği / sıcaklık / gürültü değerleri metinde yoktu;
  değerleri bunlardır.
* **Gate'in yeri.** `x_çıkış = x + g · (Modül(x) − x)`, `g = p/p.detach()`: gate modülün residual akışa
  EKLEDİĞİ farkı çarpar, kimlik yolu gate'lenmez ("residual dalını zayıflatmaması için").
* **Dizi almayan modül.** Bir optimizer adımında hiç dizi almayan modülün forward'ı çalışmaz, gradyanı
  `None` kalır ve AdamW onu ATLAR (weight decay ve moment güncellemesi de uygulanmaz).
* **Mikro-batch.** H ve X için 16 × 4 bölüşümü sabittir (V1 bölüşümü serbest bırakıyor, ama router
  istatistikleri ve V7'nin aux terimi mikro-batch başınadır).
* **Aktif FLOP.** Yönlendirilen katmanların bağlam terimi `4·d·c`; `c` `src/attn_patterns`'ten
  programatik alınır: H için A'nın sayılmış ortalaması (128.17), X için üç modülün EN BÜYÜĞÜ (128.50,
  metnin "ortak hedef"i) — kol başına TEK sabit, FOTON'un aleyhine, router'ın seçiminden bağımsız
  (paydayı router oynatamasın). Fark 0.0015 puan mertebesindedir; gerçekleşen seçim karışımıyla hesap
  raporda duyarlılık satırı olarak verilir. Router: dizi başına `2·d·3` FLOP, token başına `2·d·3/T = 6`.
* **Oracle.** Dizi başına seçim ÖLÇÜT pozisyonları (64..511) üzerindeki ortalama CE ile yapılır (B12 prefix
  maskesinden önce yazılmıştı; "oracle eğrisi FOTON'un eğrisinin üstünde olamaz" ancak böyle doğrudur).
  `val_loss` dört kolda da aynı koddan gelir; oracle geçişi ek bir ölçümdür.
* **Oracle NULL'ı (B11: "modül çıktıları birbirine permüte edilerek").** Okuma: her dizide, her
  POZİSYONDA üç modülün token kayıpları kendi aralarında rastgele permüte edilir (sabit tohum 54321; her
  eval noktasında, her koşuda aynı permütasyon), aynı oracle prosedürü bu değiş-tokuş edilebilir
  "sözde-modüllere" uygulanır. Metnin en düz okuması (etiketleri dizi içinde permüte etmek) minimumu
  değiştirmez (null = oracle, kazanç hep 0); kayıpları DİZİLER arasında karıştırmak alanlar arası ~0.9
  nat'lık farkı seçime sokar (null ≪ oracle). İkisi de vazgeçme koşulunun oracle bacağını anlamsız kılar;
  seçilen okuma B11'in gerekçesini ("modüller aynı olsa bile seçim eğriyi aşağı çeker") taşıyan tek
  okumadır. **Bu seçim tartışılabilir olduğu için her eval noktasında HAM veri diske yazılır**
  (`<koşu>/eval_raw/step_*.npz`: üç modülün pozisyon başına CE'si, dizi ortalamaları, router seçimi /
  olasılıkları / logit'leri, alan kimliği): başka her okuma koşulardan sonra yeniden hesaplanabilir.
* **Router logları (V6).** `router_dead`: log aralığındaki yük payı ideal payın (1/3) %10'undan az olan
  modül sayısı (`src/moe.py`'nin eşiği). `router_load_cv`: popülasyon std (tam çökme = √2).
  `router_switch_rate`: eğitim pencereleri tekrar etmediği için "aynı dizinin modül değiştirmesi"
  tanımsızdır; ölçülen, ardışık iki log aralığında `p(modül | alan)` dağılımları arasındaki toplam değişim
  uzaklığının alanlar üzerinden ortalamasıdır. Ham sayımlar (modül; alan × modül) her satırda loglanır.
* **V7.** Kol X; üç koşul üç preset'tir (`foton-x` / `foton-x-ent` / `foton-x-lb`); iki katsayı da **0.01**
  (Switch'in aux katsayısı; bu depodaki `aux_loss_coef` varsayılanı; metinde yoktu); entropi bonusu
  `loss −= 0.01 · ort(H(p_dizi))` (nat), aux `loss += 0.01 · 3 · Σ_m f_m · P_m` mikro-batch başına.
  "Regularizasyon yok" koşulunda router'a HİÇBİR ek terim gelmez (z-loss dahil). **"%50 bütçe" =
  `--stop_step 2188`, LR schedule tam bütçe üzerinden**: tarama koşusu, karar koşusunun yörüngesinin ilk
  yarısıdır. Seçim penceresi son 500 adım; (i) `router_dead = 0` üç seed'in üçünde ve penceredeki her
  satırda; (ii) dizi-başı entropinin pencere ortalamasının seed ortalaması, en düşük; (iii) fark < 1e-3 ise
  metnin sırası. Hiçbir koşul (i)'i geçemezse (metin yazmıyor) en az ölü modüllü koşullar arasında (ii)–(iii)
  ve bu açıkça raporlanır. **Kazanan koşul HEM H HEM X preset'ine uygulanır** (yoksa birincil-2 heterojenliği
  regularizasyonla karıştırırdı). Uygulama `scripts/router_health.py`; val loss'u OKUMAZ.
* **V6 erken uyarı.** "İlk 500 adımda kalıcı" = 500. adımda biten ardışık üç log aralığının (450, 475, 500)
  üçünde de ihlal. "Herhangi üç ardışık aralık" okuması her koşuda ateşlenirdi: init'te router logit'leri
  ~0 olduğu için dizi-başı entropi ilk aralıklarda zorunlu olarak ~1.0'dır (duman testinde ölçüldü).
  Çare çevrimleri yalnızca hata düzeltebilir ya da V7 koşulunu değiştirebilir; V10'un dondurduğu
  düğmelere (init ölçeği, sıcaklık, gürültü, gate normalizasyonu) dokunmak sapmadır.
* **Zorunlu 2× koşu (V3 adım 2).** Sıfırdan, ayrı bir koşudur; `TrainConfig.budget_mult = 2`: LR schedule
  referans bütçe üzerinden tanımlı kalır (cosine 1×'te min_lr'a iner, 2×'e kadar min_lr'da SABİT kalır —
  v01 EK1-B9), `max_steps` 4375'te kalır (V2). Seed bloğunun dışındadır; duvar saatine girmez.
* **S1 katman bölüşümü.** 10 katmanda **4 + 2 + 4**: modül başına katman sayısı (2) donmuş eksendir,
  paylaşılan yığınlar simetrik büyür, router yine ilk paylaşılan yığının çıkışını okur. S0 sonuçları
  görülmeden, S0 preset'leriyle birlikte hash'lendi.
* **Config hash'i (V10).** Hash'lenen: preset başına `{model, train}` sözlüğünün kanonik JSON'u
  (`vocab_size` 32768 çözülmüş); toplam = ad→hash eşlemesinin hash'i (`scripts/preset_hash.py`,
  `experiments/foton-v02/preset_hashes.json`). V7 kazananı preset'e işlendikten SONRA, karar koşularından
  ÖNCE yazılır.
* **Blok yeniden koşumu ve provenance (V11a-iii).** Metin "o bloğun TAMAMI yeniden koşulur" diyordu ama
  kuyruk koşucusu bunu uygulamıyordu (biteni atlıyor, kalanı koşuyordu). Artık uygulanıyor: kısmen koşulmuş
  bloğun BÜTÜN klasörleri — bitmiş olanlar dahil — silinmeden `<ad>.blok-iptal-<n>` adıyla kenara alınır,
  deftere yazılır, blok ilk koşusundan yeniden başlar; iptal edilen denemeler duvar saati analizine girmez.
  Karar ve S1 kuyrukları SIKI'dır: ölçümü üreten kod (`src/`, `scripts/`, `tests/`, `docs/`, ön-kayıt)
  commit'siz ise kuyruk hiç başlamaz, kuyruk sürerken `src/` değişirse sıradaki koşu başlatılmaz. Her
  koşunun `provenance.json`'u oturum kimliğini (`session`) ve `code_dirty` alanını taşır (ham `dirty`,
  kuyruk kendi defterini yazdığı için kampanya boyunca hep True çıkıyor ve hiçbir şey söylemiyordu).

**S5 — 2026-09-19: "H, X'ten ÖNCE koşulur" ile V11a'nın Latin karesi çelişiyor; V11a geçerli.**
"Sıra" bölümü H'nin X'ten önce koşulmasını istiyor (router hatası varsa GPU yakmamak için); V11a ise
bağlayıcı olarak seed bloğu + Latin kare rotasyonu istiyor ve üretilen kuyrukta `x-s11`, `h-s22`'den önce,
blok s44'te X, H'den önce koşuyor. H'yi kol bloğu hâlinde öne almak V11a'nın kaldırdığı confound'u geri
getirir ve H'nin duvar saatini V11a(iv) gereği geçersiz kılardı. Okuma: "H önce" bir DEVREYE ALMA
sırasıdır ve amacı (router hatasını ucuza yakalamak) karar koşularından önceki duman testleri ve V7
taramasıyla karşılanır; karar ve S1 kuyruklarının sırası V11a'nınkidir.

**S6 — 2026-09-19: dondurulmuş korpusta doküman sınırı YOK; D10 tanımsız, D11 val'de D9 ile eşdoğrusal.**
KORPUS bölümü "doküman sınırları EOS ile işaretli" diyor. Altı `.bin` dosyasının tamamı (306M token)
tarandı: id ≤ 2 olan (unk / bos / eos) TEK bir token yok. Korpus yeniden kurulmaz (TRIPWIRE: korpus
değişikliği yeni bir deneydir) ve dört kol aynı veriyi gördüğü için karşılaştırma yanlı değildir. Sonuçları:
D10 (dizi uzunluğu) ölçülemez — her pencere 512 token'lık sabit bir kesittir ve doküman uzunluğu veriden
geri çıkarılamaz; D11 (doküman türü) val pencerelerinde alanla özdeştir (her val dosyası tek kaynaktan
gelir). Aile büyüklüğü 11'de KALIR (V5: aile büyüklüğü hesaplanabilen test sayısından alınamaz); D10
"ölçülemedi", D11 "D9 ile eşdoğrusal" olarak raporlanır.

**S7 — 2026-09-19: V8'in "prob gerçek tepeyi az gösteriyor" açıklaması yanlış sebebe bağlanmış.**
V8 farkı (MoE +0.90 GB, dense +0.71 GB) eval'deki `logits.float()` tensörüne bağlıyor. Ölçüldü
(`scripts/vram_probe_foton.py`): eval tepesi eğitim tepesinin ALTINDA kalıyor. Asıl sebep eğitim döngüsünün
`_, loss, info = model(x, y)` ataması: `_` önceki mikro-batch'in bf16 logit tensörünü (0.54 GB) bir sonraki
forward bitene kadar canlı tutuyor. O atama taklit edilince prob gerçek koşuları üç ondalığa kadar yeniden
üretiyor (dense 6.48 / 6.480 GB, MoE 8.03 / 8.031 GB). Dört kolda da aynı kod çalıştığı için kollar arası
bir fark değildir ve eğitim döngüsü DEĞİŞTİRİLMEDİ (baseline'larla aynı kalsın); S1'deki dar pay için
bilinmesi gerekir.

**S8 — 2026-09-19: tag'li metinde kalan, karara girmeyen sayı/ifade artıkları.**
V4 hâlâ ilk (maskesiz) kümenin sayılarını anıyor: "`N*` ortalaması bütçenin %80.095'i … %0.118";
`baseline_reference.json`'dan hesaplanan değerler **%80.058 / +%0.072**'dir (L* bölümü ve V9 doğrusunu
yazıyor); paydalar dosyadan okunduğu için etkilenmez. MİMARİ tablosunun "en uzak erişim" kolonu
(150 / 511 / 256 / 512) gevşek yazılmış; `src/attn_patterns` 149 / 510 / 255 / 508 verir (desenler
kendisini de görür: ofset 0 dahil). "en çok 0.33 bağlantı (%0.26)" → hesaplanan 0.3262 = %0.25;
"128.50 ± 0.33" tek yanlıdır (128.17 … 128.50, ortalama 128.35). Raporlar hesaplanan sayıları basar.
Sapma-S2'nin son cümlesi ("o koşu kuyruğun sonunda yeniden koşulacak") tarihseldir: `r-d-s11` yeniden
koşuldu ve on koşunun tamamı aynı commit'i gösteriyor.

**S9 — 2026-09-19: v01 EK1-B13'ün profiler kuralı M kolunda bilinen bir sayaç artefaktına takılıyor (07'de çözülecek).**
Kurulu torch'un FLOP sayacı `_grouped_mm`'i 0, SDPA'yı maskeden bağımsız T×T sayıyor. M'in (gruplu yol)
profiler−analitik farkı bu yüzden ~−%19, FOTON'unki ~+%7 çıkar; B13'ün "fark 5 puandan büyükse 'başarılı'
ilan edilemez" kuralı M'e karşı harfiyen uygulanırsa FOTON'un değil sayacın kusuru yüzünden tetiklenir.
Koşuları etkilemez (analiz); rapor aşamasında karşılaştırma bileşen bileşen ve M'in FLOP-özdeş `loop`
yoluyla yapılır, hangi yolun kullanıldığı raporda yazılır. Her koşu ANALİTİK aktif FLOP'unu kendi
`metrics.jsonl`'ının `meta` satırına yazar.
