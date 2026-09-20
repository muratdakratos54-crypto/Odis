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

| 4 (28 ajan; biri API hatasıyla düştü) | dense TEŞHİS koşusunun ölçütü (koşudan ÖNCE): başarısız olabilir mi, ulaşılabilir mi, sahibin iki şartına uyuyor mu | HIGH yok; bulguların en ağırı: **gürültü çapası iki FARKLI kod sürümünden alınmıştı** ve diskte AYNI kodla koşmuş gerçek bir tekrar çifti vardı (`r-m-s11`; zarfı daha geniş: pencere ort +0.001133, maks 0.007758 — seçilen çapanın 6.4 / 1.34 katı). Eşikler o yüzden fazla sıkıydı: kod özdeşken bile P(SAĞLAM) ≈ %80–88. Ayrıca: "suçlu" eşiği tam tespit etmek istediği etkiye konmuştu (%50 güç), ölçüt TERS yöne kördü ve o hâlde olgusal olarak YANLIŞ bir gerekçe basıyordu, ikincil CE çapası belgede kodun hesapladığından başka bir küme üzerinden yazılmıştı |

Tur 4'ten sonra: çapa AYNI KODLA koşmuş çiftten alındı (üçüncü kanıt: `v2-speed` üçlüsü, aynı src ve seed, eğitim CE'sinde
bit-özdeş DEĞİL); eşikler 0.004 / 0.012'ye genişletildi; karar ÜÇ hâlden BEŞ hâle çıkarıldı (kısmi sapma ve ters yönde sapma
ayrı hâller, ikisi de "dense farkı ayrı iş olarak açılır"a bağlanıyor); OKUNMAZ gerekçesi artık ölçülen sayılardan üretiliyor;
takvim tahmini eğitimin kendi `token_budget()` yardımcısını kullanıyor; ulaşılabilirlik benzetimle ölçülüp assertion'a
bağlandı (`dcheck_reachability.log`: kod özdeşken P(SAĞLAM) %96–98, anomalinin tamamında P(SUÇLU) %100).

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
| 5 (dense teşhis ölçütü; tur 4'ün bulguları) | 170 | 170 yakalandı (yeni: E01–E24) |

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
* **Seed 101 kapısı AŞILDI (19:32).** Ön koşulun seed-başı bacağı TEMİZ (mean_fixed farkı −0.00255 nat, tavan farkı +0.108
  puan); D akla-yatkınlık bacağı AŞILDI (adım 2000 sapması +0.012132, sınır 0.012). Kuyruk durdu, kalan 8 koşu başlatılmadı,
  δ hesaplanmadı. Sahibin kararı: **"+0.012132 > 0.012. Kıl payı olması bir şey değiştirmiyor; z = +9.6 zaten kıl payı
  olmadığını söylüyor. (b) bu turda OKUNMAZ, haritanın '(b) okunamadı' hücresi geçerli. Bu kararı veriyi gördükten sonra
  tartışmaya açmıyoruz."** Tolerans gevşetilmedi ve gevşetilmeyecek.
* **Teşhis koşusu ONAYLANDI** (D, seed 11, 2188 adım) iki şartla: başarı ölçütü koşudan ÖNCE yazılacak (tek nokta değil,
  eğri boyunca) ve takvim eşleşmesi açıkça teyit edilecek. İkisi de yapıldı: `dcheck_rule.md`, `scripts/v03_dcheck.py`,
  `dcheck_schedule.log` (87 ortak adımda birebir aynı, koşudan önce koddan), `dcheck_reachability.log`.
* **Kalan 8 koşu: TAMAMI koşulacak, kuyruk kısaltılmayacak.** Sahibin gerekçesi: D-102 ve D-103'ün karar değeri kalmadı ama
  teşhis değeri var (yeni D koşuları genel olarak mı kaymış, yoksa yalnız 101 mi); ayrıca veriyi gördükten sonra kuyruk
  kompozisyonunu değiştirmek başlı başına bir sapmadır.
* **Yakın-kaçış dersi** (sahibin yazdırdığı): kapı atanmış kontrol noktasında sınırı yalnızca %1.1 aştı, oysa eğri adım
  1500'de z = +9.6 ile tamamen dışarıdaydı — tek noktalı akla-yatkınlık denetimi bu anomaliyi neredeyse kaçırıyordu ve
  kirli veriyle (b) okunabilirdi. Gelecekteki akla-yatkınlık kapıları EĞRİ BOYUNCA tanımlanacak. Kayıt:
  `experiments/foton-v02/v03_design_notes.md`, bölüm 3.
