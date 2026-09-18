# Denetim kaydı — PREREGISTRATION_V02 karar-kritik yüzeyi

Sahibin durma kuralı (2026-09-18): karar-kritik yüzeyde (`decide`, `abandon`, `contribution`, `_wbar`
ve bağlayıcı kural metni) HIGH yoksa tag; varsa düzelt + bir tur; üçüncü ardışık turda hâlâ HIGH
çıkıyorsa mevcut hâliyle tag ve açık defect'ler belgeye. Düzeltme (2026-09-19): son turda çıkan bir
HIGH **(a)** ölçümü ÜRETEN yerdeyse (eğitim döngüsü, eval protokolü, veri akışı, FLOP muhasebesi,
pin'ler) kampanyadan ÖNCE onarılır; **(b)** analiz/karar kodunda ya da belgedeyse tag atılır, açık
defect yazılır, kampanya koşulur (ham compute-to-L\* ölçümleri veridir; istatistik yeniden hesaplanabilir).
Tur 4 yoktur.

Her tur: üç mercek (vazgeçme/ub95/sansür; birincil uç noktalar + haklar / veri yapısı; metin ↔ metin),
her merceğin raporu ayrı bir ADVERSARYAL doğrulayıcıdan geçti; HIGH ancak çalıştırılmış bir
reprodüksiyonla ya da satır numaralı iki bağlayıcı pasajla sayıldı.

| tur | denetlenen commit | HIGH | kapatan commit |
|---|---|---|---|
| 1 | 3bf0717 | 8 (4 konu) | b81af02 |
| 2 | b81af02 | 4 (3 konu: ub95 tavanı metni; `decide_primary` konum eşlemesi; yavaş seed) | 552754d |
| **3 (son)** | **552754d** | **0** — (a) sınıfı da (b) sınıfı da yok | — |

Tur 2'nin "konum eşlemesi" bulgusu o hatanın ÜÇÜNCÜ ortaya çıkışıydı (`paired` v3, `savings` 1.
denetim, `decide_primary` 4. denetim). Sahibin kararıyla yama yerine YAPISAL düzeltme yapıldı:
tohum sonuçları seed kimliğiyle anahtarlı eşleme olarak taşınır, liste alan çağrı `TypeError` verir
(ön-kayıt V4, "VERİ TAŞIMA KURALI"; mutasyon bloğu E01–E21).

## Tur 3: bulgular ve akıbetleri

Tur 3'te HIGH yok. Aşağıdaki MEDIUM/LOW bulguların hepsi reprodüksiyonla doğrulandı. Sahibin kuralı:
bağlayıcı metne dokunanlar tag'den ÖNCE düzeltilir. "Yanlış kullanım çalışmasın, patlasın" standardına
dokunan kod bulguları da tag'den önce kapatıldı. Tag'den önce yapılan bu değişikliklerin GEÇERLİ
girdide hiçbir kararı değiştirmediği, denetlenen commit (552754d) ile son hâlin 40.000 rastgele
kampanyada (n ∈ {5, 10}; hızlı/yavaş/sansürlü karışımları; dört `w̄` biçimi; seed hakkı açık/kapalı;
tek ve iki ölçekli vazgeçme) karşılaştırılmasıyla gösterildi: `decide_primary` farkı 0, `abandon`
farkı 0; yedi satırın ve dört birincil-2 durumunun hepsine ulaşıldı. Bu değişiklikler dördüncü bir
denetim turundan GEÇMEDİ (tur 4 yok); güvenceleri testler, mutasyon sınavı ve o diferansiyel koşudur.

| # | sınıf | bulgu | akıbet |
|---|---|---|---|
| 1 | MEDIUM, metin | "2× sınırında süreksizlik YOKTUR" yalnız TEK başarısız seed için doğru; iki sınır seed'inde satır 0 yalnız sansürlüyü sayar (n=10, hızlı seed'ler ≥ ~+%34.7 iken `nötr` vs `başarısız`) | metin düzeltildi; iki senaryo ulaşılabilirlik tablosuna assertion olarak eklendi |
| 2 | MEDIUM, guard | **hayatta kalan konum-sınıfı mutant**: `ub95()`'te hızlı/sabit ayrımını KONUMLA yapan mutant 93 testi ve taramayı geçiyordu (bütün fixture'lar başarısız seed'i en büyük anahtara koyuyordu) | test (`test_failed_seed_position_does_not_matter_anywhere`), iki tarama senaryosu (s11'de başarısız seed), mutant **E17** + tarama mutantı **R10** |
| 3 | MEDIUM, guard | hayatta kalan mutant: anahtar kümesi denetimi tek yönlü (H'nin FAZLA seed'i: birincil-1 n=5, birincil-2 n=10) | test (iki yön, `decide_primary` ve `contribution`), mutant **E18** |
| 4 | MEDIUM, kod | seed durumu sessizce atlanabiliyordu: `savings(status=None)` ve çıplak sayı → "hızlı" (aynı FLOP'lar durumla `başarısız`, durumsuz `yüksek varyans / belirsiz`) | `savings()`'te `status` ZORUNLU; `strict=True` çıplak sayıyı reddeder; mutantlar **E20, E21** |
| 5 | MEDIUM, kod | `strict=True` yalnız n'ye bakıyordu, HANGİ seed'ler olduğuna değil (10 seed'in en iyi 5'i `başarılı`; `dict(enumerate())` arka kapısı) | bağlayıcı mod yalnız ön-kayıtlı seed kümelerini kabul eder (`DECISION_SEEDS`, `EXTRA_SEEDS`); mutant **E19** |
| 6 | MEDIUM, kod | `resolve_n_star()` 1× yuvasındaki eğrinin 1× koşusu olduğunu denetlemiyordu (2× eğrisi → `hızlı`, N\*/bütçe = 1.143) | eğriler yuvalarıyla tutarlı olmak zorunda (1× bütçede / (1×, 2×] aralığında biter); mutantlar **S51, S52** |
| 7 | MEDIUM, kod | `abandon(res, gains, True)` True'yu `oracle_threshold`'a bağlıyor, terk kararını sessizce çeviriyordu | skalerler yalnız anahtar kelimeyle; `escalation_used` ZORUNLU (varsayılanı yok); tip denetimi; mutantlar **S53–S55** |
| 8 | MEDIUM, metin | V4 "anahtar kümeleri farklıysa hata" cümlesi `paired()` için doğru değil (o kesişimi eşler, `dropped` raporlar) | metin ve modül docstring'i: `paired()` açık istisna, karar yolunda kullanılmaz |
| 9 | MEDIUM, metin | V3'te artık cümle: "Şimdiki uygulama B14'ün kendisidir" (tanım "bilinçli sıkılaştırma" diyor) | metin + `n_star` docstring'i düzeltildi; "İki kayıt düzeltmesi" → "Üç" |
| 10 | MEDIUM, metin | V4 payda pin'i ölçeğe göre kapsamlanmamıştı (S1'de dondurulmuş değer kullanılamaz) | S0 / S0-ek seed / S1 ayrı ayrı yazıldı |
| 11 | MEDIUM, metin | `L*` bölümünün başlığı pin'den önce yazılmıştı ("eşleştirilmiş karşılaştırma AYRI koşular kullanır") | "DUVAR SAATİ karşılaştırması (`w̄`) ayrı koşular kullanır; payda V4'te pinli" |
| 12 | LOW, metin | "terk etmeyi engelleyebilen tek şey oracle şartıdır" evrensel değil (n=10, 2 sansür, hızlı GA üst ≥ ~+%43.7 iken `ub95 ≥ 5` de engeller) | cümle "gerçekçi aralıkta" diye sınırlandı, uç durum yazıldı |
| 13 | LOW, metin | "yavaş" durumu N\*'a göre değil "hangi eğri kesti"ye göre atanıyor: 2× yeniden koşumu 1× bütçeden az önce kesebilir | V3'e yazıldı; o seed 1× sınırı teşhisinin bandındadır |
| 14 | LOW, guard | V12 "her senaryonun satırı assertion" diyordu; satır 16 senaryonun 12'sinde assertion'dı | tarama artık 20 senaryonun HEPSİNDE beklenen satırı assertion olarak taşıyor |
| 15 | LOW, guard | hayatta kalan mutant: `resolve_n_star` 1× kesse bile 2× eğrisini okuyor (yalnız 2× DAHA ERKEN kestiğinde görünür) | S48'in testine o durum eklendi |
| 16 | LOW, metin | V10 donmuş fonksiyon listesinde yardımcılar yoktu | `Obs`, `keyed`, `paired`, `ci95`, `t_crit`, `t_crit_alpha`, `_wbar` eklendi |
| 17 | LOW, metin/kod | docstring artıkları (`decide_primary` "ölçemedik ise p₂ = 1"; `ROWS` yorumu; test docstring'i "üçüncü hâl YOKTUR"); `abandon()` terminal satırda yanıltıcı gerekçe; `w_bar` anahtar sırası; `abandon()` str/bool kazanç kabulü | hepsi düzeltildi (S56, S57 mutantları) |
| 18 | LOW, metin | "yetersiz kesinlik" iki yerde yalnız "n=10" diye özetlenmişti; "Tripwire ateşlenirse"; gerçekleşemeyen varsayım ("aynı örnekte seed sansürlü olsaydı"); V13'te E03 iki sınıfta | düzeltildi |
| 19 | LOW, metin | oracle/null eğrisi 1×'te kesmezse hangi 2× eğrisi? (2× koşu yalnız FOTON 1×'te ulaşamayınca var) | yazıldı: ayrı 2× koşu yapılmaz; null ikameye düşer, bu kazancı büyütür (vazgeçmeyi zorlaştırır: muhafazakâr) |
| 20 | LOW, metin | B8'in "≥ 1 sansür → başarı yasak" dalı tek biçimli kesinlik ölçütünün gölgesinde: tam örnek GA alt sınırı tek sansürle n=5'te hiç, n=10'da ancak hızlı seed'ler ≥ ~+%73 iken sıfırı geçer | metne "yedekli savunma (kemer + askı)" olarak yazıldı; yasak hiçbir zaman ihlal edilmiyor |
| 21 | LOW, kod | yükleyiciler `keyed()` yerine dict comprehension kullanıyordu; `calibrate_v02.py`'nin (şu an ölü) `else` dalı liste indeksiyle kuruyordu | `baseline_reference.py` ve `calibrate_v02.py` `keyed()` kullanıyor; `calibration.log` birebir aynı çıktı |
| 22 | LOW, kapsam dışı | `scripts/compare_runs.py` (v2 dönemi, FOTON hattında değil) kendi `paired(a: list, b: list)`'ini `zip()` ile taşıyor | **AÇIK — tag sonrası.** Karar yolunda değil; v02 betikleri onu çağırmıyor |

## Bilinen ve kabul edilmiş sonuçlar (defect DEĞİL; sahibin kararları, 2026-09-19)

* **1× sınırındaki durum süreksizliği kalır.** Sınır veriden değil bütçe tanımından gelir, durum mekanik
  atanır (istismar edilemez) ve yönü muhafazakârdır (yavaş seed'in sabit girmesi üst sınırı yalnızca
  aşağı çeker: başarısızlık/terk yönü). Kaldırılmadı, GÖRÜNÜR yapıldı: ±%10 teşhis bandı
  (`boundary_diagnostic`), ön-kayıtta veri görülmeden pinli.
* **Hedefe ulaşamayan seed birincil istatistiğe girer**; `4 × +%30 + 1 sansür` hak kullanılmazsa ve
  oracle < %2 ise terk edilir. Tur 3 denetçisinin ölçtüğü sonuç: gerçek etki +%20, sd 0.88, S0'da
  bir yavaş seed ve hak kullanılmamışken tek ölçekli terk simüle kampanyaların %48'inde tetikleniyor;
  tek koruma oracle şartıdır. Bu, yukarıdaki kararın doğrudan sonucudur. Sahibin S1'e çıkma / çıkmama
  seçimi S1 koşulmadan ÖNCE zaman damgalı ve gerekçeli tek satır olarak loglanır
  (`scripts/owner_decision.py`).
