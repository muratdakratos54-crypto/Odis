# Router ölçütleri: denge, uzmanlaşma ve "sahte denge"

Bu belge `scripts/router_analysis.py`'nin ölçtüğü şeylerin tanımı, neyi ölçtüğü ve **nasıl yanlış okunabileceğidir**.
Ablasyonun sonuçları: README, "Deney kaydı: `v3-router-ablasyonu`".

Temel ayrım: **denge** (expert'ler eşit mi kullanılıyor) ile **uzmanlaşma** (expert'ler farklı ve tutarlı işler mi
yapıyor) aynı şey değildir. Tokenları expert'lere rastgele dağıtan bir router mükemmel dengelidir ve hiç uzmanlaşmamıştır.
Dengeleyici baskı bu ikisini karıştırabilir: kota doldurmak için tokenı semantik uyumu düşük bir expert'e yollamak
utilization tablosunu düzeltir, modele bir şey kazandırmaz. Buna burada **sahte denge** diyoruz.

## 1. Denge

**top-k atama sayılarından** hesaplanır (gating türünden bağımsız): yük CV (0 = mükemmel denge), ölü expert
sayısı (ideal payın %10'undan az alan), maks/min pay, yük entropisi H(yük)/log E.

**Toplulaştırma tuzağı (yan yana duran sayılar aynı düzeyde değil):**

| ölçüt | nasıl toplanıyor |
|---|---|
| yük CV | katman başına CV'nin **katman ortalaması** |
| maks/min pay | **tüm katmanlardaki en kötü** hücre (112 expert içinde), katman ortalaması değil |
| yük entropisi | top-k sayımlarından, katman ortalaması |
| `utilization_entropy` | **yalnızca top-1** sayımlarından (§2'nin sayım matrisi C'den) |

Yani "yük CV 0.10" ile "maks pay 1.47x" tutarsız görünür ama değildir: biri ortalama, biri en kötü hücre.
`utilization_entropy` ise bir **uzmanlaşma** ölçütüdür, denge ölçütü değil: top-1 dağılımı, top-k yükü
mükemmel dengeli olan bir kolda bile belirgin dengesiz olabilir (bu deneyde top-1 CV'si top-k CV'sinin
3-15 katı). Denge iddiası için yük CV / yük entropisi, bilgi-teorik ölçütlerin tavanı için `utilization_entropy`.

## 2. Uzmanlaşma

Bu grubun **tamamı top-1 seçimden** hesaplanır (C: token türü × top-1 expert sayımları), yani top-k'nın
diğer k−1 seçimini görmez.

| ölçüt | tanım | yüksek değer ne demek | tuzak |
|---|---|---|---|
| `mi_bits` | I(token kimliği ; seçilen top-1 expert), bit | routing kararı token kimliğinden belirleniyor | üst sınırı H(expert); çökmüş routing'de küçük kalır. **Plug-in tahminci yukarı yanlı:** `mi_bits_null` (aynı marjinallerle bağımsızlık tabanı) ile okunmalı ve yanlılık token bütçesiyle değiştiği için farklı `--batches` ile üretilmiş analizler karşılaştırılamaz |
| `nmi` | I / H(expert) | aynı, kullanım entropisine göre normalize | **çökmüş routing'de şişer**: H(expert) küçüldükçe pay küçülür, oran büyür. Mutlak `mi_bits` ile birlikte okunmalı |
| `h_cond_bits` | H(expert \| token) = H(expert) − I(token;expert), bit (katman ortalaması) | aynı token bağlama göre farklı expert'e gidiyor: routing'in **bağlama bağlı** kısmı | **rastgele router bu ölçütü maksimize eder** (tabanı `h_cond_null`). Ve **dengeli kollar arasında `mi_bits`'in aynası**: H(expert) orada neredeyse sabit olduğu için `mi_bits + h_cond ≈ sabit`, yani ikisi ayrı kanıt değil tek ölçümün iki yüzüdür (bkz. §7). Gürültüden ayıran ölçüt `ΔCE permütasyon`dur, `mi_bits` değil |
| `consistency` | bir token türünün geçişlerinin kaçı o türün baskın expert'ine gidiyor (≥ min_count geçişli türler, geçişle ağırlıklı) | aynı token aynı yere gidiyor (sözlüksel uzmanlaşma) | bağlama duyarlı uzmanlaşmayı cezalandırır: düşük olması tek başına kötü değil. **Null'ı 1/E değil:** sonlu sayımda maksimum yukarı yanlı ve yük dengesizliğiyle yükselir, o yüzden çökmüş bir kolda ham değer "yüksek" okur. `consistency_null` ile birlikte okunmalı (bu deneyde rastgele+dengesiz bir kurulumda ham 0.71, null 0.70 — bkz. `tests/test_router_stats.py`) |
| `expert_diversity` | 1 − (aynı girdilerde expert çıktılarının ortalama ikili kosinüsü) | expert'ler birbirinin kopyası değil | **null'ı TAVAN:** hiç eğitilmemiş, rastgele başlatılmış expert'ler ≈ 1.000 verir; eğitim bu ölçütü DÜŞÜRÜR. Bu yüzden yüksek değeri hiçbir koşulda uzmanlaşma kanıtı değildir ve kollar arası küçük farkları yorumlanamaz. Yalnızca "hepsi birbirinin kopyası olmuş mu" (≈ 0) sorusunu cevaplar |
| `embedding_coherence` | expert'in baskın token kümesinin embedding'lerinde içsel benzerlik − kümeler arası benzerlik | expert semantik olarak tutarlı bir küme almış | modelin kendi embedding uzayına göredir (dolaylı kanıt) |
| lift tablosu | P(expert\|token)/P(expert) en yüksek tokenlar | nitel: expert neye uzmanlaşmış | nadir tokenlarda gürültülü (min_count ile sınırlanır) |

## 3. Router güveni

`gate_top1` (seçilenin gate değeri), `margin` (en iyi ile ikinci seçim skorunun farkı), `entropy`
(router dağılımının normalize entropisi), `raw_score_gap` (bias'sız skorlarda en iyi ile ikincinin farkı).

**Ölçek tuzağı:** softmax olasılıkları (16 expert'e bölünmüş, top ≈ 0.1-0.3) ile sigmoid skorları (her expert
bağımsız, ≈ 0.5-0.9) aynı ölçekte değildir. Marj karşılaştırması yalnızca softmax koşulları arasında anlamlıdır.

## 4. Eşleşme anlamlı mı: karşı-olgusal CE

Aynı sabit val pencerelerinde CE ölçülür; fark ne kadar büyükse routing o kadar "gerçek iş yapıyor":

| kip | ne değişir | ne değişmez | neyi ölçer |
|---|---|---|---|
| `permuted` | expert kimlikleri tutarlı bir permütasyonla değiştirilir (3 permütasyonun ortalaması) | yük dağılımı, gate değerleri, ağırlıkların kendisi | **token ↔ expert eşleşmesinin** değeri. Sahte dengede fark küçüktür |
| `random` | seçim token başına rastgele k expert | gate değerleri | routing'in tamamen rastgeleye karşı değeri |
| `next` | top-k yerine router'ın k+1..2k sıraları | — | router'ın kendi ikinci tercihine göre birinci tercihinin değeri |
| `nobias` | (yalnızca aux-loss-free koşul) seçim ham skorlarla, bias yok sayılır | ağırlıklar | dengeleme baskısını çıkarınca görev iyileşiyor mu? İyileşiyorsa baskı göreve zarar veriyordu |

Permütasyon testi bu setin en temiz üyesidir: yükü ve gate dağılımını sabit tutar, yalnızca hangi tokenın hangi
uzmana gittiğini bozar. Bu yüzden "denge aynı kalırken eşleşme ne kadar değerliydi" sorusunu doğrudan yanıtlar.

## 5. Dengeleme kuvvetinin kendisi

- **aux loss'lu koşullar — gradyan çekişmesi.** Eğitim sonunda bir batch'te router logit'lerine göre iki gradyan
  hesaplanır: görev (CE) ve dengeleme (aux). Raporlanan: `|coef · ∂aux/∂logits| / |∂CE/∂logits|` (dengelemenin
  göreli kuvveti) ve iki gradyanın kosinüs hizası. Negatif hiza, dengelemenin görevin istediği yönün tersine
  çektiğini gösterir; oran büyüdükçe routing kararını dengeleme belirler.
  Sınır: tek batch ve eğitimin sonundaki durum; eğitim boyunca ortalama değil.
- **aux-loss-free koşul — bias ezme oranı.** Seçim bias'lı ve bias'sız yapıldığında top-k kümesi kaç tokende
  farklı. Bu, dengeleme kuvvetinin routing'i kaç kez doğrudan çevirdiğinin **tam** ölçümüdür.
  **Yorum tuzağı:** router bias ile birlikte eğitildiği için yüksek ezme oranı tek başına "kötü" demek değildir;
  ham skorlar bias'ı telafi edecek şekilde şekillenmiş olabilir. Bu yüzden `nobias` karşı-olgusal CE'siyle
  birlikte okunur: ezme oranı yüksek ama bias'ı kaldırmak CE'yi bozuyorsa, bias çözümün parçası olmuş demektir;
  ezme oranı yüksek ve bias'ı kaldırmak CE'yi düzeltiyorsa, baskı göreve zarar veriyor.

## 6. Sahte dengeyi nasıl okuyoruz

Bir koşulda **sahte denge** iddiası şu örüntüyle desteklenir:
1. yük CV düşük (denge iyi), **ve**
2. `mi_bits` ve `consistency` diğer koşullara göre düşük (routing token kimliğinden kopmuş), **ve**
3. `ΔCE permütasyon` küçük (hangi expert'e gittiği önemsiz), **ve**
4. dengeleme kuvveti büyük: aux gradyan oranı yüksek / kosinüs negatif, ya da bias ezme oranı yüksek ve
   `ΔCE bias'sız` negatif (baskıyı kaldırmak iyileştiriyor).

Tersine, **gerçek denge**: yük CV düşük ama `mi_bits`, `consistency` ve `ΔCE permütasyon` yüksek kalıyor.
Tek bir ölçüte bakmak yanıltır; bu yüzden karar dört kalemin birlikte hareketine göre verilir.

**Eşikler.** Maddeler nitelikseldir ("yüksek/düşük"). Sayısal eşikler karar anında
`experiments/v3-router-ablation/report_numbers.py` içinde sabitlenir (v3'te: yük CV ≤ 0.15 ve ölü expert = 0;
2.–3. maddeler için eşleştirilmiş farkın %95 GA'sının sıfırı dışlaması; 4. madde için aux gradyan oranının
referansın 10 katını aşması ya da bias ezme oranı > 0.05 ve `ΔCE bias'sız` < 0). Kararı betik verir, metin değil.

**Permütasyon testinin geçerlilik koşulu.** Permütasyon yalnızca yük **histogramını** sabit tutar; modül başına
yük permüte olur. Ölü expert varsa test karışır: tokenların büyük bir kısmı hiç yük görmemiş (fiilen
başlatma ağırlıklarında kalmış) modüllere gider ve CE artışı "eşleşmenin değeri" değil "eğitilmemiş modüle
yönlendirmenin maliyeti" olur. v3'te (c) kolunda rastgele bir permütasyon tokenların ~%40'ını böyle modüllere
yönlendiriyor. **Bu yüzden permütasyon karşı-olgusalı yalnızca ölü expert'i olmayan kollarda okunur.**

## 7. İkinci bozulma kipi: sözlükleşme (ön-kayıtta yoktu, ablasyonda bulundu)

Yukarıdaki sahte denge örüntüsü, dengeleme baskısının routing'i **gürültüye** çevirdiği varsayımına dayanıyordu.
v3 ablasyonunda güçlü aux (katsayı 1.0) bunu yapmadı: denge en düşük yük CV'ye ulaştı (0.019), `mi_bits` ve
null üstü `consistency` **arttı**, ama val loss serinin optimumuna göre kötüleşti (+0.0132 nat, p = .011).
Yani baskı routing'i bozmuyor, **niteliğini değiştiriyor** — bağlam fonksiyonu olmaktan çıkıp token kimliğine
bakan öğrenilmiş bir sözlüğe (hash) yaklaşıyor. Örüntüsü sahte dengenin tersi:

1. yük CV düşük, **ve**
2. `mi_bits` ve null üstü `consistency` **yüksek** (sahte dengede düşüktü), **ve**
3. `h_cond_bits` düşük — bağlama bağlı kısım eriyor, **ve**
4. `ΔCE permütasyon` yüksek kalıyor (eşleşme hâlâ değerli; bozulma eşleşmenin *varlığında* değil *türünde*).

**2. ve 3. madde tek ölçümdür, iki kanıt değil.** `h_cond = H(expert) − mi_bits` ve dengeli kollarda
H(expert) neredeyse sabittir (v3: 3.923 ± 0.023 bit), dolayısıyla r(mi_bits, h_cond) = −0.97. Biri yükselirse
diğeri tanımı gereği düşer. Örüntüyü destekleyen bağımsız kalemler bu yüzden **1, (2≡3) ve 4**'tür; ayrıca
gürültüden ayıran ölçüt 4. maddedir (`ΔCE permütasyon`), `mi_bits` değil.

Mekanizma ölçütte değil kayıp fonksiyonunda: Switch aux'u `coef·E·Σ f_e·P_e`, atama sayılarının değil router'ın
ortalama **olasılıklarının** fonksiyonudur, bu yüzden tepe yapmış router dağılımını doğrudan cezalandırır
(`margin` aux gücüyle monoton düşer; v3'te optimumdan aux 1.0'a −0.041, p = .002 — bu, val loss kötüleşmesiyle
birlikte örüntünün çoklu karşılaştırma düzeltmesinden geçen tek iki kalemi). Seçim bias'ı (aux-loss-free) bunu
yapmaz: gate değerine dokunmaz. Ama v3'te bias kolu aynı anda gate tipini de değiştirdiği için bu
karşılaştırma **nedensel değildir** (bkz. kayıt README'si, eksik 2×2 hücreleri).

### Ölçütlerin işareti rejime bağlı

Aynı ölçüt iki rejimde ters yönde okunur; bu yüzden hiçbir ölçüt tek başına hedef alınamaz:

| ölçüt | çöküş rejiminde (tüm kollar) | denge sağlandıktan sonra (dengeli kollar arası) |
|---|---|---|
| yük CV | yüksek = kötü (r = +0.75) | bilgi taşımıyor (r = −0.18) |
| `mi_bits` | yüksek = iyi (r = −0.65) | yüksek = **kötü**, sözlükleşme (r = +0.87) |
| `h_cond_bits` | yüksek = iyi (r = −0.92) | yüksek = iyi (r = −0.88) |

(r değerleri v3 ablasyonunun 18 koşusundan, koşu düzeyinde; "dengeli kollar" = yük CV ≤ 0.15 ve ölü expert 0
olan dört koşul, n = 12.)

**Bu tablo bir uyarıdır, kanıt değil.** Üç sebeple: (i) gözlemsel — koşul ile ölçüt birlikte değişiyor,
nedensellik iddiası yok; (ii) **pseudo-replikasyon** — 18 koşu 6 koşulun 3'er tekrarı, bağımsız gözlem birimi
koşuldur, dolayısıyla gerçek serbestlik derecesi 4 (tüm kollar) ve 2 (dengeli kollar), koşu sayısının verdiği
16 ve 10 değil; (iii) işaret dönmesinin en çarpıcı hâli yalnızca üç koşul ortalamasına dayanıyor (df = 1) ve
üç noktada herhangi bir monoton ilişki |r| ≈ 1 verir. Koşul düzeyinde değerler:
yük CV +0.76 / −0.19, `mi_bits` −0.66 / +0.90, `h_cond_bits` −0.97 / −0.93.
Tablodan çıkarılacak tek şey şudur: **hiçbir ölçüt tek başına hedef alınamaz, çünkü işareti rejime bağlıdır.**
