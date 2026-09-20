# v03 adım 1 — DENETİM KAYDI (2026-09-20; koşulardan ÖNCE)

v02 ön-kaydının denetim kaydı `experiments/foton-v02-prereg/audit_log.md`'dedir ve o klasör tag'le donmuştur (sahibin
kalıcı kuralı: koşu defteri dışında dokunulmaz). v03 adım 1'in denetim kaydı bu yüzden BURADA tutulur.

## "Başarısız olamayan ölçü" sınıfı — DÖRDÜNCÜ yakalanış, bu kez ajanın KENDİ kodunda

| # | nerede | ölçü | neden başarısız olamıyordu | kim yakaladı |
|---|---|---|---|---|
| 1–2 | önceki iki yakalanış | — | sahibin 2026-09-20 kaydı üçüncüyü "sınıfın üçüncü yakalanışı" diye adlandırır; ilk ikisinin ayrıntısı bu dosyada yeniden kurulmadı (uydurulmadı) | — |
| 3 | v03 adım 1, sahibin ilk karar ölçüsü | "alan atamalı val loss, en iyi sabit tek modüle karşı ≥ 3 puan" | X-dom'da her modül yalnızca kendi alanını görür; "her şeyi modül A'dan geçir" tabanı A'yı hiç görmediği iki alanda sınar → EVET garanti | ajan (itiraz); sahibi onayladı, ölçü δ = val(X-iid) − val(X-dom) oldu |
| **4** | **v03 adım 1, ajanın yazdığı eval kodu** (`src/train.py: evaluate`, X-iid dalı) | X-iid'de "`val_atanan` iki yoldan 1e-4 içinde aynı" sağlaması | iki "yol" AYNI tensörün (üç-modül geçişinin dizi ortalamaları) iki okunuşuydu: fark tanım gereği ~0; X-iid'in EĞİTİMDE kullandığı dispatch yolu hiçbir eval'de çalışmıyordu | birinci bağımsız inceleme turu (ajanın başlattığı); ajan düzeltti |

Dördüncünün düzeltmesi: X-iid'de birinci yol artık eğitimin forward yoludur (`model(x, y, assign=m)`, m = 0, 1, 2; ortalaması),
ikinci yol üç-modül geçişi + ham döküm. Test, eval sırasında forward yolunun 18 kez çağrıldığını casusla doğrular; mutant V14
("birinci yol hep modül 0") yakalanır. GPU'da gerçek ağırlıklarla iki yolun farkı 8e-8 (X-dom'da 1.3e-5).
Ders değişmedi: **her yeni ölçü / sağlama için — bu, hipotez (ya da kod) yanlışken de geçer miydi?** Aynı soru artık ajanın
kendi sağlamalarına da sorulur.

## Bağımsız inceleme turları (salt-okunur ajanlar; her bulgu ayrı bir ajanın çürütme denemesinden geçti)

| tur | kapsam | sonuç |
|---|---|---|
| 1 (3 ajan) | zorlanmış-atama yolu, eşleme iddiaları, kuyruk | v02 kollarında regresyon yok (eski / yeni kod bit-özdeş); eşleme iddiaları doğru (D init'te eşli DEĞİL). Bulgular: log niyet edilen atamayı yazıyordu; X-iid iki-yol sağlaması totolojikti (yukarıda #4); analiz koşu kimliğini klasör adından okuyordu (HIGH); NaN iki-yol sağlamasından geçiyordu; v02 tarafı denetlenmiyordu; ön koşul sistematik kaymaya kördü; (a) HAYIR + (b) GEÇTİ hücresi v02 ile çelişen bir durumu "4/8"e bağlıyordu |
| 2 (19 ajan: 3 inceleme + 16 çürütme) | tur 1 düzeltmelerinden sonraki hâl | HIGH yok; 16 bulgu (4 MEDIUM, 12 LOW), hepsi doğrulandı. Geçerli bir kampanyanın analizce REDDEDİLMEYECEĞİ uçtan uca CPU provasıyla gösterildi. MEDIUM'lar kanıt açığıydı: sentetik kampanya seed'ler arasında simetrikti (δ_D, ön koşul ve maske teşhisinin seed-eşlemesi bozulunca hiçbir sayı değişmiyordu) — reviewer'ların 13 sonda mutantı HAYATTA KALDI; iki-yol sağlamasının bacakları ayrı ayrı sınanmıyordu; "kol X, denetim Y'nin dışında" sınıfı 7 hücreden 2'sinde kanıtlıydı; Ö2 ve Ö3'ün ulaşılabilirliği ölçülmemişti |

| 3 (15 ajan; 5 çürütme ajanı kota sınırına takıldı, bulguları yine de tek tek incelendi) | sahibin onayından sonra yazılan seed-101 kapısı, kampanya sürücüsü, bölünmüş kuyruklar, LR denetimi, rapor sırası | HIGH yok; 12 bulgu (2 MEDIUM, 10 LOW). LR takvimi iddiası bağımsız doğrulandı (ortak loglanan 87 adımın tamamında bit-özdeş LR; 3.1198 ± 0.0027 yeniden hesaplandı). Bulgular: kuyruk ikiye bölününce "kampanya boyunca TEK src özeti" güvencesi iki kuyruğun ARASINDA kalmıyordu; kapı çıktısı X-iid-101'in `val_atanan`'ını toplamsal kaydırmayla sızdırıyordu ve "hiçbir kolun val kaybı yok (testle kilitli)" iddiası YANLIŞTI (test yalnız literal dizgi arıyordu); kapı dar bir istisna kümesi yakalıyordu; sürücü bayat kapı çıktısını silmiyor ve "running" durumu yazmıyordu; rapor okunmaz durumda da GEÇTİ / GEÇMEDİ hükmü basıyordu; çelişki bayraklarının rapordaki yeri belgede iki türlü yazılıydı; sentetik veri mean_fixed ≡ best_fixed ≡ val olduğu için ön koşulun hangi niceliği okuduğu kanıtsızdı |

**Tur 3'ün en pahalı bulgusu kodda değil, BENDEYDİ ve başlatmadan önce denenerek bulundu:** kampanya sürücüsü bağımsız
süreç olarak koşarken stdout dosyaya yönlenir ve Windows'ta cp1252 olur; ilk Türkçe günlük satırı `UnicodeEncodeError` ile
sürücüyü öldürecekti — kuyruk hiç başlamadan, durum dosyası da yazılmadan. Düzeltildi (`sys.stdout.reconfigure`, günlük önce
DOSYAYA, durum dosyası finally'de günlükten önce), alt süreçle (cp1252 stdout) sınandı ve mutantı eklendi (G18).

Tur 3'ten sonra: kapı, koşulardan ÖNCE kaydedilen src özetiyle karşılaştırma yapıyor; kapı GEÇTİĞİNDE hiçbir sayı yazmıyor
(yalnız iki bayrak) ve AŞILDIĞINDA yalnız iki kapının nicelikleri; testler JSON'un yanında stdout'u ve hata metinlerini de
yasak sayıların birkaç ondalık biçimi için tarıyor; sentetik X-iid / v02 sütunları ayrışık (mean_fixed ≠ best_fixed ≠ val)
ve ön koşulun kaynağı mutantla kanıtlı (P01–P03); rapor okunmaz durumda hüküm basmıyor; çelişki bayraklarının yeri belgede
ve testte sabit.

Tur 2'den sonra: temel sentetik kampanya (kol, seed)-asimetrik yapıldı ve her sayı seed başına kesin değerle assert ediliyor;
eşleme bozulmaları {veri sırası, başlangıç, val sırası, val özeti} × {ilgili her kol} çapraz çarpımı; D'nin val kümesi
eşlemesi artık VERİDEN doğrulanıyor (her kol val pencerelerinin token içeriği özetini logluyor; 12 koşuda tek değer VE
koşulardan önce kaydedilen değer); koşu-içi iki-yol durdurması NaN'ı uyuşmazlık sayıyor; Ö2'ye işaretli-ortalama bacağı
eklendi; Ö2 ve Ö3'ün ulaşılabilirlik tabloları assertion'larıyla `reachability.log`'da.

## Mutasyon sınavı geçmişi (`mutation_log.md` yalnızca SON tam koşumu taşır; geçmiş burada)

| sınav | mutant | sonuç |
|---|---|---|
| 1 (tur 1'den önce) | 38 | 38 yakalandı |
| 2 (tur 1 düzeltmeleri) | 84 | 83 yakalandı; **U19 hayatta kaldı** (kimlik denetimi eklenince val satırındaki atama-kuralı etiketini tek başına bozan test kalmamıştı) → test eklendi |
| 3 (tur 2 düzeltmeleri) | 119 | 119 yakalandı (tur 2'nin 13 sonda mutantı kataloğa girdi: N40–N67, R09–R12, D09–D11) |
| 4 (seed-101 kapısı + sürücü + LR denetimi + tur 3'ün bulguları) | 146 | 146 yakalandı (yeni: G01–G23 kapı / sürücü, N68–N69, P01–P03) |

06 sınavı (FOTON uygulamasının guard'ları) `src/train.py` / `src/model.py` her değiştiğinde yeniden koşuldu: 71 / 71.

## Sahibin kararları (2026-09-20)

* Karar kuralı + dört ek (n = 3, eşli D, iki soru / iki sayı, önceden yazılı harita) — ONAYLANDI.
* Dört öneri (Ö1 işaretli-ortalama bacağı 0.0075; Ö2 D akla-yatkınlık denetimi; Ö3 maske teşhisinde aynı işaret; Ö4
  "BEKLENMEYEN" hücresi) — DÖRDÜ DE ONAYLANDI. Ö2 için sahibin notu (LR takvimi): teyit edildi — adım 2000'de loglanan LR,
  tam bütçeli referans D koşularında ve tarama bütçeli koşularda bit düzeyinde aynı (0.0006469248450683149); analiz bunu her
  koşuda veriden denetler.
* "δ'nın işareti neredeyse hep pozitif" notu raporda açıkça yazılacak: büyüklük eşiği testi taşır, işaret yalnızca tutarlılık.
* 12 koşu ONAYLANDI. Yürütme talimatı: `decision_rule.md`, "YÜRÜTME" bölümü (seed 101 kapısı; 12 koşu bitmeden δ yok; kuyruk
  bitince ön koşul → (a) → (b) → harita hücresi; yorum yok).
