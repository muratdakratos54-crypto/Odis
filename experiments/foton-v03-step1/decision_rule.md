# v03 ADIM 1 — KARAR KURALI (veri görülmeden yazıldı; 2026-09-20; sahibi onayladı: kural + dört ek + dört öneri; 12 KOŞU ONAYLANDI)

**Tek soru:** prefix router'ın GÖREBİLDİĞİ en iyi bölümleme (alan) ile yönlendirme ZORLANIRSA modüller, işe yarar biçimde
ayrışıyor mu? Hayırsa hiçbir prefix router kazanamaz (eğitim dinamiğinden bağımsız). Evetse "öğrenilmiş router devralabilir
mi" SONRAKİ deneydir. "Kararlılık mı, alan hizası mı" bu adımda cevaplanmaz. Mimari olguları: `docs/ARCHITECTURE_AS_BUILT.md`.
Kuralın TEK kopyası koddadır: `scripts/v03_step1_decision.py` (ulaşılabilirlik sınavı ve analiz onu çağırır).

> **ONAY DURUMU (2026-09-20).** İki eşik (3 puan / 5 puan), δ ve δ_D'nin tanımı, n = 3 kuralı ve harita sahibince ONAYLANDI. Onaydan
> sonra zorlanmış yolun iki bağımsız incelemesi kuralda dört açık buldu; aşağıda **[ÖNERİ 1–4]** etiketiyle yazılı dört ek
> **sahibince ONAYLANDI, dördü de** (veri görülmeden). Hiçbiri iki eşiğe ve δ / δ_D tanımına dokunmaz. Ö1, Ö2 ve Ö4 "karar
> okunmaz" durumlarını ÇOĞALTIR (Ö4 bunu haritanın tek bir hücresini değiştirerek yapar — haritayı değiştiren tek ek budur);
> Ö3 yalnızca bir TEŞHİS etiketini sıkılaştırır, karara ve haritaya dokunmaz. **Dört önerinin DÖRDÜ DE sahibince ONAYLANDI
> (2026-09-20).** 12 koşu onaylandı ve başlatıldı; yürütme talimatı belgenin sonundadır. İnceleme turlarının kaydı:
> `audit_log.md` (bu klasörde).
>
> **KOŞU DURUMU (2026-09-20 19:32): seed 101 kapısı AŞILDI, kuyruk durdu, kalan 8 koşu başlatılmadı, δ hesaplanmadı.**
> Ön koşulun seed-başı bacağı TEMİZ (mean_fixed farkı −0.00255 nat, tavan farkı +0.108 puan). D akla-yatkınlık bacağı
> AŞILDI: adım 2000 sapması +0.012132 nat, sınır 0.012. Sahibin kararı: tolerans gevşetilmez, **(b) kapısı bu turda
> OKUNMAZ** ve haritanın "(b) okunamadı" hücresi geçerlidir; bu karar veriyi gördükten sonra tartışmaya açılmaz. Sahibi
> ayrıca bir TEŞHİS koşusu onayladı (D, seed 11, 2188 adım; ölçütü koşudan önce yazıldı: `dcheck_rule.md`) ve kalan 8
> koşunun TAMAMININ koşulmasını, kuyruğun kısaltılmamasını istedi.
>
> **TEŞHİS SONUCU (2026-09-20 22:03): KOD SAĞLAM.** Bugünkü kod dondurulmuş dense referansını aynı seed'de yeniden üretti;
> D-101'in sapması seed çekimidir (`dcheck_rule.md` bölüm 5b). Sahibi sonucu kabul etti.
>
> **KAPANAN KAPI KAPALI KALIYOR** (sahibin kararı, 2026-09-21; ileride kimse yanlış hatırlamasın diye buraya yazıldı):
> "Referans ailesi beş seed'di, kuyruğu dar gösteriyordu" doğru bir gözlem ve iyi bir v03 dersi; ama bu turda (b)'yi
> yeniden açmanın gerekçesi DEĞİL. **Toleransın dar kalibre olmuş olması, kuralı veriyi gördükten sonra esnetme hakkı
> vermez — tersine, kuralın neden önceden yazıldığının kanıtıdır. (b) bu turda okunmaz.** (b) sorusu (X-dom eşli D'yi
> geçiyor mu) cevapsız kaldı; doğru kalibre edilmiş D toleransı ve genişletilmiş D referansıyla AYRI bir ön-kayıtta
> sorulacak. Kayıp değil, ertelendi.
>
> **KALAN 8 KOŞU ONAYLANDI (2026-09-21).** Ön-kayıtlı kuyruğun tamamı, kısaltma yok. Seed başına ön koşul bacağı yürürlükte:
> 102 ya da 103'te aşılırsa kuyruk durur ve sabahı bekler. Kuyruk bitince sırayla: ön koşul → (a) → (b) hücresi
> ("okunamadı") → harita hücresi; kuralın kendi cümleleriyle, yorum eklenmeden.

## Kollar — 12 koşu: 4 kol × seed {101, 102, 103}; T = 512; tarama bütçesi (adım 2188, LR takvimi tam bütçe üzerinden)

| kol | model | eğitimde VE eval'de atama kuralı |
|---|---|---|
| **X-dom** | X (A, B, C) | modül = alan kimliği (code→M0:A, math→M1:B, nl→M2:C), sabit |
| **H-dom** | H (A, A, A) | modül = alan kimliği — maske farkını ağırlık farkından ayırır |
| **X-iid** | X | çekim başına BAĞIMSIZ düzgün rastgele modül. Kararlılık iddiası YOK. Hem zorlanmış yolun null'ı hem karar ölçüsünün karşı kolu |
| **D** | dense | — (yönlendirme yok). v02'nin D referansı eşsiz ve tam bütçeliydi; burada AYNI seed'lerle ve tarama bütçesiyle eşlenir |

**Eşleme — neyin eşli olduğu, neyin OLMADIĞI.** Seed'ler v02 taramasıyla aynı. X-dom, H-dom, X-iid ve v02-X
(`v7-x-s101/102/103`) seed başına AYNI başlangıç ağırlıklarından, AYNI veri sırasıyla ve AYNI val kümesinde koşar. **D yalnızca
veri sırasında ve val kümesinde eşlidir, başlangıç ağırlıklarında DEĞİL** (başka mimari, başka init çekimi): δ_D'nin seed
başına gürültüsü δ'nınkinden büyüktür (v02'de seed'ler arası val sd'si D 0.0027, X 0.0085 → eşlenmemiş üst sınır ≈ 0.009
nat; (b)'nin ulaşılabilirlik taraması 0.003–0.010'u kapsar). Bu iddialar analizde VERİDEN doğrulanır (kod okumasına
dayanmaz): **veri sırası** — log aralığı başına alan sayımları (`data_domain_counts`; D dahil bütün kollar + v02-X); **val
kümesi** — her kolun (D dahil; dense kol ham döküm yazmaz) her eval noktasında logladığı val pencerelerinin TOKEN İÇERİĞİ özeti
(`val_windows_sha256`): 12 koşuda TEK değer olmalı VE o değer koşulardan ÖNCE kaydedilen özet olmalı (`9773961275…845e`;
GPU sağlaması aynı pencerelerle v02-X'in logladığı oracle'ı 1e-4 içinde yeniden üretir, yani bu v02'nin ölçüldüğü val kümesidir;
val dosyası arada yeniden üretilirse kimlik, veri sırası ve kod özeti denetimlerinin hepsi geçerdi); **v02-X'e bağ ve başlangıç ağırlıkları** — FOTON kollarının val `domain`
dizisi ve adım 0'daki ham val kayıpları (aynı seed'li v02 koşularında ölçülen fark 0.0; başka seed'de ortalama 0.06; v02
koşularında özet alanı yoktur, D'nin v02'ye bağı FOTON kolları üzerinden kurulur). **Koşudan ÖNCE prova**
(`scripts/v03_step1_replay_streams.py` → `stream_replay.log`; CPU, eğitim yok): iki akış da yalnızca seed'e bağlı olduğundan
yeniden oynatıldı — veri sırası üç seed'de de v02-X'in loguyla 88/88 aralıkta birebir aynı; X-iid'in uygulayacağı alan x modül
tablosu şimdiden biliniyor ve 1/3'ten en büyük pay sapması 0.0041 / 0.0029 / 0.0022 (analizin toleransı 0.01). Analizin meta
beklentileri (parametre, FLOP, maske harfleri) de testte GERÇEK modelden üretilir: geçerli bir kampanyayı reddetmeyecek.

M bu adımda koşulmaz (D'yi geçemeyen M'yi hiç geçemez). Router modelde KALIR (çıkarılırsa bağlı embedding'in init çekimi
kayar), çalışmaz ve gradyan almaz (AdamW tümüyle atlar); gate yok (ileri değeri zaten 1'di). X-iid'in ataması `train_gen`'den
ve global RNG'den AYRI bir generator'dan çekilir (tohum = seed + 7,777,777). Kuyruk: `queues/step1.json` — seed bloğu içinde
kollar dönüşümlü (Latin kare rotasyonu), SIKI (`code_dirty = False` her koşudan önce).

## Kolun kaybı: `val_atanan`

Sabit 1,280 val penceresinde, pozisyon 64..511, HER dizinin EĞİTİMDEKİ atama kuralıyla gittiği modülün kaybı. X-dom / H-dom:
`atama(b) = alan(b)`. X-iid: düzgün rastgele atamanın BEKLENEN değeri = üç modülün ortalaması (tek bir rastgele çekimin
gürültüsü ölçüme girmez). D: `val_loss`. **İki AYRI kod yolundan** hesaplanır ve her eval noktasında 1e-4 içinde eşleşmek
ZORUNDADIR (eşleşmezse ya da sağlama sonlu değilse analiz durur; 1e-3'ü aşarsa koşunun kendisi durur):

1. **eğitimin forward yolu:** `model(x, y, assign=...)` — X-dom / H-dom'da alan kimliği; X-iid'de bütün batch sırayla her
   modüle atanır (m = 0, 1, 2) ve üç sonucun ortalaması alınır;
2. **üç-modül geçişi** (`foton_eval_all` → `val_loss_assigned_check`) ve ondan bağımsız olarak `eval_raw` ham dökümünden sonradan.

(İlk hâlde X-iid'in iki "yolu" aynı tensörün iki okunuşuydu: sağlama başarısız OLAMIYORDU ve X-iid'in eğitimde kullandığı
dispatch yolu hiçbir eval'de sınanmıyordu — inceleme bulgusu, düzeltildi; test eval sırasında forward yolunun 18 kez
çağrıldığını casusla doğrular. Koşu-içi durdurma da sonlu olmayan bir sağlamayı UYUŞMAZLIK sayar: çıplak `abs(a − b) > tol`
NaN'ı geçirirdi.) Eğitimsiz GPU sağlaması (`scripts/v03_step1_gpu_two_route.py` → `gpu_two_route.log`; v02
ağırlıkları, bf16 / CUDA): iki yolun farkı X-dom 1.3e-5, X-iid 8e-8; oracle ve router'ın sayısı v02 koşusunun kendi logunu
1e-4 içinde yeniden üretiyor.

Router'ın (eğitilmemiş) seçimiyle hesaplanan sayı `val_loss_router_UNUSED` adını taşır ve hiçbir tabloda kol kaybı olarak
GÖRÜNMEZ. `router_*` metrikleri zorlanmış kollarda `null` ("uygulanamaz"); yerlerine UYGULANAN atamanın sayımları (modelin
gerçekten dispatch ettiği tensörden), modülsüz mikro-batch ve gradyansız-modüllü adım sayıları loglanır. FLOP bu adımda karar
ölçütü DEĞİLDİR, yalnızca loglanır (H, X'ten 4,008 FLOP/token az; zorlanmış yolda router çalışmadığı hâlde analitik sayım 18
FLOP/token'lık router terimini saymaya devam eder).

## KAYIT — "başarısız olamayan ölçü" sınıfının ÜÇÜNCÜ yakalanışı

Sahibin ilk önerdiği ölçü ("alan-etiketli atamanın gerçekleşen val loss'u, en iyi sabit tek modüle karşı ≥ 3 puan") X-dom'da
TOTOLOJİKTİ: X-dom'da her modül yalnızca kendi alanını görerek eğitilir; "bütün dizileri modül A'dan geçir" tabanı A'yı hiç
görmediği iki alanda sınar, kötü çıkması uzmanlaşmanın YARARINI değil modülün alan dışında eğitilmemiş olmasını ölçer ve
EVET'i garantilerdi. Aynı sebeple X-dom / H-dom'da modüller arası fark, oracle tavanı, B11 null'ını aşan kısım (`over_null`)
ve I(alan; en iyi modül) tanım gereği büyük çıkar: RAPORLANIR, "TOTOLOJİK" etiketiyle, kanıt SAYILMAZ. Ders (her yeni ölçü
için sorulacak soru): **bu ölçü, hipotez yanlışken de geçer miydi?** Geçerli karşılaştırma aynı kod yolu + seed + veri altında,
TEK farkı atamanın alanla hizası olan koldur: X-iid. (Aynı sınıfın dördüncü örneği bu adımın KENDİ kodunda çıktı: X-iid'in
iki-yol sağlaması — yukarıda.)

## İKİ SORU, İKİ SAYI (ikisi de önceden yazılı; ikisi de NAT cinsinden bağlayıcı)

    δ_s   = val_atanan(X-iid, s) − val_atanan(X-dom, s)       pozitif = alan ataması, rastgele atamadan iyi
    δ_D,s = val(D, s)            − val_atanan(X-dom, s)       pozitif = X-dom, eşli dense'ten iyi

* **(a) MEKANİZMA — uzmanlaşma oluyor mu:** `ort_s δ ≥ 0.010192 nat (3 puan)` VE üç seed'in ÜÇÜNDE de `δ_s > 0`.
* **(b) GEÇERLİLİK KAPISI — X-dom ≥ eşli D + 5 puan:** `ort_s δ_D ≥ 0.017163 nat (5 puan)` VE üç seed'in üçünde de `δ_D,s > 0`.
  Gerekçe: v02'de X, eşit token'da D'nin ~0.050 nat (~14 puan) gerisindeydi; muhtemel sebep her modülün token'ların 1/3'ünü
  görmesi. Uzmanlaşma bu maliyeti ödeyip üstüne 5 koymalı. Kapıdır, birincil ölçüt değil; ama önceden yazılı kapıdır.
* **Ön koşul (X-iid null'ı temiz mi), HER seed'de, adım 2188'de, aynı seed'li v02-X'e karşı:** `|mean_fixed(X-iid) −
  mean_fixed(v02-X)| ≤ 0.015 nat` VE `|oracle tavanı farkı| ≤ 0.4 puan`. (Kalibrasyon: v02'de aynı seed'li üç koşul arasında
  val farkı en çok 0.0073 nat, tavan farkı en çok 0.20 puan; sınırlar bunların 2 katı. v02-X'te modül yükleri dengeliydi —
  pay 0.33 / 0.34 / 0.33 —, yani v02-X'in modülleri de ~iid veri gördü: iki taraf kıyaslanabilir.) Aşılırsa zorlanmış yol bir
  şey değiştirmiştir: karar OKUNMAZ (iki sayı yine raporlanır), önce o anlaşılır.
  **[ÖNERİ 1] işaretli-ortalama bacağı:** ek olarak `|ort_s (mean_fixed(X-iid) − mean_fixed(v02-X))| ≤ 0.0075 nat`. Sebep:
  seed-başı bacak SİSTEMATİK kaymaya kördür — üç seed'de de aynı yöne 0.010 nat'lık ((a) eşiği kadar!) bir kaymayı %86 / %65 /
  %41 "temiz" sayıyordu; bu bacakla %7 / %16 / %23. Kayma yokken yanlış alarm: %0 / %0.5 / %17 (σ = 0.003 / 0.0045 / 0.0075).
* **[ÖNERİ 2] D akla-yatkınlık denetimi:** (b) bütünüyle YENİ D koşularına dayanır. HER seed'de `|val(D, adım 2000) − 3.1198|
  ≤ 0.012 nat` (dondurulmuş D referansı, seed 11..55, n = 5, adım 2000: 3.1198 ± 0.0027, `screening_context.json`; 0.012 =
  4·0.0027·√(1 + 1/5)) VE üç seed'in işaretli ortalama sapması `≤ 0.006 nat` (3·0.0027·√(1/3 + 1/5)). Başka seed ve veri sırası
  → eşli değil, yalnızca akla-yatkınlık. Aşılırsa **yalnızca (b) OKUNMAZ**; (a) okunur. Ulaşılabilirlik (aşağıda): D'nin 0.010
  nat'lık sistematik kaymasını — kapıyı 0.010 nat şişirir ya da söndürür — seed-başı bacak tek başına %45 / %33 / %27 "akla
  yatkın" sayardı; ortalama bacağıyla %0.5 / %4 / %8. Bedeli: kayma yokken yanlış alarm %0 / %1.6 / %12 (sd = 0.0027 / 0.004 /
  0.0055) — sonucu "(b) okunmaz, D incelenir"dir, yanlış karar değil. **LR takvimi (sahibin notu):** tolerans takvim farkı
  İÇERMEZ ve içermesi gerekmez — referans D koşuları tam bütçeli, yeni D'ler tarama bütçelidir ama `stop_step` koşuyu yalnızca
  KESER; takvim iki durumda da max_steps = 4375 üzerinden hesaplanır. Adım 2000'de loglanan LR referans D koşularında
  (`r-d-s11..55`) da tarama koşularında (`v7-x-s101`) da bit düzeyinde aynıdır: 0.0006469248450683149. Karşılaştırma AYNI takvim
  kesrindedir; analiz ve seed-101 kapısı bunu 12 koşunun her birinde VERİDEN denetler (tutmazsa karar üretilmez).
* **Maske mi ağırlık mı (teşhis, karar değil):** `val_atanan(H-dom) − val_atanan(X-dom)`. Pozitif = (A, B, C) maske kümesi, BU
  eşlemeyle (code→A, math→B, nl→C), (A, A, A)'dan iyi. **"Heterojenlik yararlı" diye OKUNAMAZ:** fark = [val(H-iid) −
  val(X-iid)] + [δ_X − δ_H] ve ilk terim bilinmiyor (H hiç koşmadı; v02'nin genelci modüllerinde maske A ≈0.001 nat daha
  iyiydi). Maskenin UZMANLAŞMAYA katkısı ancak H-iid ile ölçülür. |ortalama| < 0.005 nat ise "grafik farkının katkısı
  ölçülemedi". **[ÖNERİ 3]** "ölçülebilir" için ayrıca üç seed'de AYNI İŞARET: gerçek fark 0 iken sahte "ölçülebilir" oranı
  yalnız ortalama şartıyla %0.4 / %8 / %27, işaret şartıyla %0.4 / %7 / %16 (σ = 0.003 / 0.005 / 0.008). Bedeli güç kaybıdır:
  gerçek fark 0.0075 nat iken %92 / %81 / %70 yerine %92 / %73 / %52. Teşhistir; karara ve haritaya girmez.

### δ neyi ölçer, neyi ÖLÇMEZ (inceleme bulgusu; kural değişmedi, okuma daraldı)

* **δ'nın İŞARETİ neredeyse her zaman pozitiftir; testi BÜYÜKLÜK eşiği taşır.** X-dom'da modül m, kendi alanının verisinin
  %100'ünü görür; X-iid'de her modül her alanın 1/3'ünü. Modül başına toplam token iki kolda da ~1/3'tür ama (modül, alan)
  başına veri payı 3 kat farklıdır: val'de değerlendirilen (modül, alan) çifti X-dom'da 3 kat fazla alan-içi veri görmüştür.
  Yani "δ > 0" tek başına uzmanlaşmanın YARARINI göstermez (bu da "başarısız olamayan ölçü" sınıfına yakındır); gösteren,
  δ'nın 3 puanı aşıp aşmadığıdır. "Üç seed'de δ > 0" şartı bu yüzden bir tutarlılık koşuludur, kanıt değil.
* **δ, tanım gereği ≥ 0 olan küçük bir bileşen içerir:** X-iid'in kaybı üç modülün ORTALAMASIDIR; `mean_fixed − best_fixed`
  v02'de 0.0007–0.0011 nat (≈ 0.2–0.3 puan). Analiz bu yüzden bağlayıcı OLMAYAN `δ_best = best_fixed(X-iid) − val(X-dom)`'u
  da raporlar; (a)'nın cevabıyla çelişirse çelişki HARİTA HÜCRESİNİN HEMEN ARDINDAN, okuma notlarından önce yazılır
  (karar değişmez).
* **`over_null` X-dom / H-dom'da totolojiktir;** v02'deki değeri ortalama 0.28 puan, aralık 0.22–0.36 (9 koşu).

**PUAN ÇEVİRİMİ — tek ve tutarlı:** `puan = 100·(1 − exp(−Δ/0.3346))`; ön-kaydın V9'da çivilediği L* yerel eğimi. v02'nin
"1.34 puan"ı ve "X, D'nin ~14 puan gerisinde" (0.050 nat = 13.9 puan) okuması da bu çevirimdir; bütün raporlarda yalnızca bu
kullanılır. **Dürüstlük notu:** ölçüm adım 2188'de (bütçenin yarısı) yapılır ve orada eğrinin yerel eğimi daha diktir (v02 X
koşularında −0.44 … −0.49): aynı nat farkı o noktada daha AZ compute eder — 0.010192 nat ≈ 2.1–2.3, 0.017163 nat ≈ 3.4–3.8
"yerel puan". Yani nat eşikleri adlarından GEVŞEKTİR (FOTON lehine). Bu yüzden analiz, eğim GEREKTİRMEYEN doğrudan okumayı da
raporlar: X-dom, karşı kolun adım 2188'deki kaybına kaç adımda ulaştı → `100·(1 − N/2188)` (bağlayıcı DEĞİL). **Çelişkinin
ölçütü önceden yazılıdır: İŞARET.** Nat farkının üç-seed ortalaması pozitifken doğrudan okumanın ortalaması ≤ 0 ise (ya da
tersi), ya da bir seed'de okuma yapılamıyorsa, analiz bayrak kaldırır ve bu, HARİTA HÜCRESİNİN HEMEN ARDINDAN, okuma
notlarından önce yazılır; karar değişmez. (Bayrakların yeri koşulardan önce sabitlendi ve testle kilitlendi: veri görüldükten
sonra "nereye yazsak" sorusu kalmasın.) (Büyüklük
için ölçüt YOKTUR: "3 puan" ile "%N token" arasındaki çevirim zaten yukarıdaki dürüstlük notunun konusudur.)

## SONRAKİ ADIM HARİTASI (veriyi görmeden)

Hücreler `scripts/v03_step1_decision.py` içindeki `NEXT_STEP` metinlerinin KELİMESİ KELİMESİNE kopyasıdır (testle kilitli).

| (a) | (b) | sonraki adım |
|---|---|---|
| HAYIR | GEÇMEDİ | (a) HAYIR → yönlendirilen katman payı 2/8 → 4/8, AYNI ölçüm, AYNI kural. O da HAYIR ise dizi-düzeyi prefix yönlendirme bu ölçekte tükenmiş sayılır; uzun bağlam YENİ hipotez olarak ayrı ön-kayıtla açılır. |
| HAYIR | GEÇTİ | BEKLENMEYEN — KARAR OKUNMAZ: (b) geçildi ama (a) geçilmedi. δ − δ_D = val(X-iid) − val(D) olduğundan bu ancak X-iid eşli D'yi yakalamış ya da geçmişse olur ((a) ortalama şartından düştüyse X-iid, D'den en az 0.007 nat İYİ demektir) ve v02'nin 'X, D'nin 0.050 nat gerisinde' bulgusuyla çelişir. Önce D koşuları ve eşleme denetlenir, sahibe sorulur; 4/8'e de router'a da GEÇİLMEZ. |
| EVET | GEÇTİ | (a) EVET ve X-dom ≥ D + 5 (eşli) → tavuk-yumurta doğrulandı VE seyreklik maliyeti kapanıyor: v03 asıl deneyi (alan etiketiyle ısıtılıp öğrenilen router'a devredilen yönlendirme) yazılır. |
| EVET | GEÇMEDİ | (a) EVET ama X-dom < D + 5 → uzmanlaşma var, maliyeti ödemiyor: ÖNCE 4/8, router SONRA. Zorlanmış tavan D + 5'i geçmiyorsa öğrenilmiş router hiç geçemez. |
| HAYIR | okunamadı [Ö2] | (a) HAYIR → yönlendirilen katman payı 2/8 → 4/8, AYNI ölçüm, AYNI kural. O da HAYIR ise dizi-düzeyi prefix yönlendirme bu ölçekte tükenmiş sayılır; uzun bağlam YENİ hipotez olarak ayrı ön-kayıtla açılır. ((b) okunamadı: D koşuları akla yatkınlık denetimini geçmedi; (a) HAYIR iken sonraki adımı değiştirmez, ama D koşuları incelenir.) |
| EVET | okunamadı [Ö2] | (a) EVET, ama (b) OKUNAMADI: D koşuları akla yatkınlık denetimini geçmedi. Önce D koşuları incelenir / yeniden koşulur; router deneyi kapı okunmadan AÇILMAZ. |
| ön koşul kirli |  | KARAR OKUNMAZ: X-iid aynı seed'li v02-X'ten sapıyor — zorlanmış yol bir şey değiştirmiş; önce o anlaşılır. |

**[ÖNERİ 4]** (HAYIR, GEÇTİ) hücresi onaylı haritada "4/8" idi. Uzun bağlam açılırsa pencereler yeniden eşitlenir (T = 8192 için 148.6 / 248.0 / 246.0).

4/8 adımı için not: o adımda ön koşulun v02'de KARŞILIĞI YOKTUR (v02'de 4/8'lik bir X koşusu yok); X-iid'in null sağlaması
o adımın kendi kuralında yeniden tanımlanmak zorundadır.

## ULAŞILABİLİRLİK (`scripts/v03_step1_reachability.py` → `reachability.log`; n = 3; beklentiler ASSERTION)

σ = eşli farkın seed başına sd'si. (a) için v02'den 0.003–0.008 nat; (b)'de D ile X-dom farklı mimariler ve init'te eşli
değil, σ bilinmiyor: 0.003–0.010 tarandı.

| (a) gerçek etki | P(EVET): σ = 0.003 / 0.005 / 0.008 | | (b) X-dom'un D'ye gerçek üstünlüğü | P(GEÇTİ): σ = 0.003 / 0.006 / 0.010 |
|---|---|---|---|---|
| 0 puan | 0.0000 / 0.0003 / 0.011 | | −14 puan (v02'deki konum) | 0 / 0 / 0 |
| 1 puan | 0.0001 / 0.009 / 0.063 | | 0 | 0 / 0 / 0.001 |
| 2 puan | 0.025 / 0.117 / 0.206 | | +2.5 | 0 / 0.006 / 0.065 |
| 3 puan (eşik) | 0.50 / 0.50 / 0.47 | | +5 (eşik) | 0.50 / 0.50 / 0.50 |
| 3.5 puan | 0.84 / 0.72 / 0.61 | | +6 | 0.98 / 0.85 / 0.72 |
| 4 puan | 0.98 / 0.88 / 0.74 | | +7.5 | 1.00 / 0.995 / 0.93 |
| 5 puan | 1.00 / 0.99 / 0.91 | | +10 | 1.00 / 1.00 / 0.999 |

| ön koşul: X-iid − v02-X gerçek kayması | P(temiz): σ = 0.003 / 0.0045 / 0.0075 | [ÖNERİ 1] OLMADAN |
|---|---|---|
| 0 | 1.00 / 0.995 / 0.83 | 1.00 / 0.997 / 0.87 |
| 0.005 nat | 0.93 / 0.82 / 0.63 | 1.00 / 0.96 / 0.75 |
| 0.0075 nat | 0.50 / 0.49 / 0.43 | 0.98 / 0.86 / 0.60 |
| 0.010 nat ((a) eşiği kadar) | 0.07 / 0.16 / 0.23 | 0.86 / 0.65 / 0.41 |
| 0.015 nat | 0.00 / 0.002 / 0.03 | 0.12 / 0.13 / 0.13 |

| [Ö2] yeni D'lerin referanstan gerçek kayması | P(akla yatkın): sd = 0.0027 / 0.004 / 0.0055 | ortalama bacağı OLMADAN |
|---|---|---|
| 0 | 1.00 / 0.98 / 0.88 | 1.00 / 0.99 / 0.91 |
| 0.005 nat | 0.74 / 0.65 / 0.56 | 0.99 / 0.89 / 0.73 |
| 0.010 nat | 0.005 / 0.04 / 0.08 | 0.45 / 0.33 / 0.27 |
| 0.015 nat | 0 / 0 / 0.002 | 0.003 / 0.01 / 0.03 |

| [Ö3] val(H-dom) − val(X-dom) gerçek değeri | P("ölçülebilir"): σ = 0.003 / 0.005 / 0.008 | aynı-işaret şartı OLMADAN |
|---|---|---|
| 0 | 0.004 / 0.07 / 0.16 | 0.004 / 0.08 / 0.27 |
| 0.005 nat (taban) | 0.49 / 0.43 / 0.35 | 0.50 / 0.50 / 0.52 |
| 0.0075 nat | 0.92 / 0.73 / 0.52 | 0.92 / 0.81 / 0.70 |
| 0.010 nat | 1.00 / 0.92 / 0.69 | 1.00 / 0.96 / 0.86 |

Okuma, dürüstçe: kural sahte EVET üretmiyor ve 5+ puanlık gerçek bir etkiyi güvenle yakalıyor. **Ama eşik kuralı eşiğin
KENDİSİNDE tanım gereği yazı-turadır; n bunu değiştirmez, yalnızca geçiş bölgesini daraltır.** n = 2 → n = 3 (σ = 0.005):
3.5 puanda 0.69 → 0.72, 4 puanda 0.83 → 0.88, 2 puanda sahte EVET 0.16 → 0.12. Üçüncü seed'in asıl kazancı ikili karar
değil BÜYÜKLÜK tahminidir: `ort δ`'nın standart hatası σ/√3 (σ = 0.005'te ±0.0029 nat ≈ ±0.86 puan). Bu yüzden rapor ikili
sonucun yanında `ort δ`'yı, seed başına değerleri ve σ tahminini verir; sonraki adımı haritanın dalı belirler, büyüklük de
o dalın içinde 4/8 adımının beklentisini. Ön koşul için: en gürültülü uçta (σ = 0.0075) kayma yokken %17 yanlış alarm verir
— bedeli "karar okunmaz, önce anlaşılır"dır, yanlış karar değil.

v02 verisinden null sağlaması: genelci modüllerde alan→modül atamasının en iyi sabit modüle farkı −0.49 … −0.04 puan (9
koşu; 6 permütasyonun en iyisi bile ≤ +0.08 puan): alan ataması kendi başına kazanç üretmiyor.

## Analizin karar ÜRETMEDİĞİ durumlar (`scripts/v03_step1_analyze.py`; her biri mutantla sınanmış)

Koşunun kim olduğu KLASÖR ADINDAN okunmaz: `config.json` (seed, kol, atama kuralı, `n_experts`, `resume`), meta satırı (kol
harfi, maske harfleri, parametre + FLOP sayıları, bitiş adımı) ve `provenance.json`'daki komut satırı klasör adıyla UYUŞMAK
zorundadır; eğitim ayarları aynı seed'li v02-X'inkiyle (`run_name` / `out_dir` dışında) BİREBİR, model ayarları yalnızca
`foton` / `foton_assign`'da farklı olabilir. Ayrıca: 12 koşudan biri eksik / bitmemiş / başka adımda bitmiş / eval noktası
eksik / `val_loss`'u sonlu değil / bir eval noktası maskesiz ölçütle · iki yolun HERHANGİ bir bacağı (ikinci yol, ham döküm)
1e-4'ü aşıyor ya da sağlama NaN · ham döküm dondurulmuş val kümesi (431 / 434 / 415) değil, sonlu değil, başka adımın ya da
`metrics.jsonl` ile tutarsız · val pencerelerinin içerik özeti eksik, koşu içinde değişiyor ya da 12 koşuda (D dahil) tek
değer değil · veri sırası parmak izi (D dahil) ya da başlangıç parmak izi aynı seed'li v02-X'inkiyle aynı değil · üç "seed"in
veri sırası YA DA başlangıcı birbirinin kopyası · X-dom / H-dom atama tablosunun köşegen dışı sıfır değil ·
X-iid'de bir alanın modül payı 1/3'ten 0.01'den fazla sapıyor · zorlanmış kolda router metriği basılmış · `code_dirty = False`
değil ya da birden fazla `src` özeti · v02-X referansı aynı sıkılıkla denetlenir.

## YÜRÜTME (sahibin talimatı, 2026-09-20; koşulardan ÖNCE yazıldı)

* **Sıra:** seed blokları iç içe — seed 101 → X-dom, H-dom, X-iid, D; sonra 102; sonra 103 (blok içinde kollar dönüşümlü:
  102 H-dom ile, 103 X-iid ile başlar — sahibin bu adımdaki mesajı kol sırasını yalnız seed 101 için yazdı; dönüşüm, önceki
  kalıcı talimatın ve ön-kayıt V11a Latin karesinin gereğidir ve geri alınmadı. Kol sırası bu adımda hiçbir karar sayısına
  girmez). Temiz, commit'li ağaçtan; bağımsız süreç; koşarken depoya dokunulmaz. Sürücü:
  `scripts/v03_step1_campaign.py` (kuyruk `queues/step1_s101.json` → kapı → kuyruk `queues/step1_s102_s103.json`; ikisinin
  birleşimi onaylı 12 koşuluk `queues/step1.json`'dur, testle kilitli).
* **Seed 101 geçerlilik kapısı** (`scripts/v03_step1_gate.py`): seed 101'in dört kolu bitince YALNIZCA iki kapıya bakılır —
  ön koşulun seed-başı bacağı (X-iid-101, v02-X-101'e karşı: |mean_fixed farkı| ≤ 0.015 nat, |tavan farkı| ≤ 0.4 puan) ve D
  akla-yatkınlığın seed-başı bacağı (|val(D-101, adım 2000) − 3.1198| ≤ 0.012). Bacaklar kuyruk-sonu kuralıyla AYNI
  fonksiyonlardır (`precondition_seed_leg`, `d_sanity_seed_leg`; tek kopya). **δ ve δ_D HESAPLANMAZ**: X-dom ve H-dom'un
  kaybı, D'nin adım-2188 kaybı ve iki δ kapı çıktısının hiçbir yerinde yoktur (testle kilitli: JSON dosyası VE stdout, hata
  metinleri dahil, bu sayıların birkaç ondalık biçimi için taranır). Kapı aşılırsa kuyruğun kalanı BAŞLATILMAZ, sahibe rapor
  edilir; aşılmazsa devam edilir, ara sonuç yoktur.
  **Kapı AŞILMADIĞINDA çıktı yalnızca iki bayraktır (`clean`), hiçbir sayı yazılmaz.** Sebebi: iki kapının kendi nicelikleri
  masum değildir — X-iid kolunda `val_atanan` TANIM GEREĞİ `mean_fixed`'dir, dolayısıyla "mean_fixed farkı" diskteki sabit
  v02-X-101 değeriyle toplandığında X-iid-101'in karar kaybını verirdi (D'nin adım-2000 kaybı da 3.1198 + sapma). Kuyruk
  sürerken bunlar da görünmesin. Kapı AŞILDIĞINDA nicelikler yazılır: kampanya zaten durmuştur ve sahibin kararı için gerekir.
  (İlk yazımda bu satırda "hiçbir kolun val kaybı yer almaz" yazıyordu ve YANLIŞTI; ikinci inceleme turu yakaladı — sızıntı
  toplamsal bir kaydırmanın ardındaydı, literal dizgi arayan test onu göremiyordu. Aynı "başarısız olamayan ölçü" sınıfı.)
  İşaretli-ortalama bacakları (Ö1, Ö2) üç seed ister: kuyruk bitince bakılır. **Ek (benim, sahibin talimatının ötesinde):**
  dört koşunun BÜTÜNLÜĞÜ de analizin kendi denetimleriyle aynı anda sınanır (kimlik, bütçe, LR takvimi, iki yol, atama
  tablosu, eşleme) ve seed-101 koşularının src özeti ŞU ANKİ ağacınkiyle karşılaştırılır (kuyruk ikiye bölündüğü için
  `run_queue`'nun kuyruk-içi src denetimi iki kuyruğun ARASINI görmez; görmezse kalan 8 koşu başka kodla koşar ve analiz 12
  koşudan SONRA karar üretmezdi). Bozuk bir koşu ya da src sapması da kuyruğu durdurur — bu bir ara sonuç değildir, hiçbir
  kayıp sayısı üretmez, yalnızca DURMA yönünde çalışır. Bu üçüncü durdurma yolu sahibin "yalnızca iki kapı" talimatının
  ötesindedir ve bu yüzden burada ve başlatma raporunda açıkça yazılıdır.
* **DEVAM (sahibin onayı, 2026-09-21).** Sürücü `--continue-after-s101` ile: **0. adım** seed 101 bloğunun yeniden
  denetimi (`--seed 101 --legs precondition`, çıktı `gate_s101_recheck.json`; bütünlük + src sürekliliği) → kuyruk
  `queues/step1_s102.json` → seed 102 ön koşul kapısı → kuyruk `queues/step1_s103.json` → seed 103 ön koşul kapısı.
  0. adım başlatma öncesi incelemenin bulgusudur: seed 102 / 103 kapıları kendi koşularının src özetini ŞU ANKİ ağaçla
  karşılaştırır, o koşular o ağaçtan koştuğu için bu seed 101'e göre totolojiktir; 101 | 102 sınırını yalnızca bu adım görür
  (seed 101'den sonra src/ değişmişse 8 koşu başlamadan durulur). Bu adım durdurmaktan başka bir şey yapmaz, sayı üretmez. İki dosya onaylı ikinci kuyruğun
  (`step1_s102_s103.json`) seed sınırında BÖLÜNMÜŞ hâlidir: aynı 8 koşu, aynı sıra, aynı argümanlar (testle kilitli; kuyruk
  kısaltılmadı ve yeniden üretilmedi). Seed 102 / 103 kapıları `--legs precondition` ile çağrılır: YALNIZCA ön koşulun
  seed-başı bacağına bakar, D'ye bakmaz ((b) zaten okunmaz; D-102 / D-103'ün değeri teşhistir). Dört koşunun bütünlüğü (D
  dahil) ve src sürekliliği yine denetlenir. Kapı geçerse hiçbir sayı yazılmaz; aşılırsa yalnız ön koşulun nicelikleri
  yazılır ve kuyruk durur. Seed 101'in aşılmış kapı kaydı (`gate_s101.json`) silinmez; bir kopyası bu klasördedir.
* **12 koşu bitmeden hiçbir δ hesaplanmaz.** Analiz yalnızca sürücü `completed` yazdıktan sonra koşulur.
* **Kuyruk bitince, bu sırayla ve belgedeki kuralla:** ön koşul (üç bacak: seed-başı mean_fixed, işaretli ortalama, seed-başı
  tavan) → (a) → (b) → harita hücresi. Sonra sahibe rapor. Yorum eklenmez; kural konuşur. Raporda AÇIKÇA yazılır: δ'nın
  işareti neredeyse her zaman pozitiftir — testi büyüklük eşiği taşır, işaret yalnızca tutarlılık koşuludur (analiz bu okuma
  notlarını sonuçla birlikte basar: `reading_notes`).

## Raporlanacaklar (her kol, aynı seed'li v02-X ile yan yana; karar YALNIZCA yukarıdaki kuraldan)

modüller arası fark (nat/dizi; v02 0.010) · oracle tavanı (v02 1.34 puan) · B11 null'ını aşan kısım (v02 ort. 0.28) ·
`val_atanan`, δ, δ_D, δ_best (seed başına + ortalama) · doğrudan token okuması · X-dom / H-dom'un en iyi sabit tek modüle karşı
puanı ve I(alan; en iyi modül) ("TOTOLOJİK" etiketiyle) · H-dom − X-dom · alan x modül atama tablosu, modülsüz mikro-batch /
gradyansız-modüllü adım sayıları · eşleme doğrulamasının sonucu · provenance · v02'nin eşsiz D / M referansı yalnızca "bağlam
için, karar değil" etiketiyle.

Not: `ModelConfig`'e `foton_assign` alanı eklendiği için kampanya preset özetlerinin toplamı değişti (fb9cd95a… → 3cbbd98b…);
üç yeni preset o özetin KAPSAMINDA DEĞİLDİR (`preset_hash.py` yalnızca v02 kampanya kollarını dolaşır) — onları koruyan, testteki
"v02 kolundan yalnızca atama kuralında ayrılır" denetimi ve analizin aynı seed'li v02-X `config.json`'uyla birebir karşılaştırmasıdır.
Dondurulmuş hiçbir değer (tag'li belge, çivi manifesti, v02 kolları) etkilenmedi: v02 kollarının D / M bit-özdeşliği guard'ı
ve 06 mutasyon sınavı (71 / 71) yeni kodla yeniden geçti. v02 kollarının loguna iki alan EKLENDİ (`data_domain_counts`,
`val_windows_sha256`); eğitim matematiğine ve RNG tüketimine dokunulmadı.
