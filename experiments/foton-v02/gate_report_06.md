# 06 — KAPI RAPORU (karar kampanyasından ÖNCE; sahibin 06-EK B şartı)

Tarih: 2026-09-19. Durum: **karar kampanyası BAŞLATILMADI; onay bekleniyor.** V10 yer tutucusu doldurulmadı.
Duman testi ve V7 taraması (kapının önünde, onaysız) tamamlandı. Val loss bu rapora BİLEREK konmadı: tarama
koşularının val loss'u birincil ölçütün kendisidir ve kapı kararı ona kör kalmalıdır (ham veri `metrics.jsonl`'da).

## 0. Önce bilinmesi gereken: V6 erken uyarısı 9 koşunun 9'unda ateşlendi

Ön-kaydın V6 kuralı: tarama tier'ında `router_entropy_per_seq > 0.9` ilk 500 adımda kalıcıysa "router hiç öğrenmiyor
demektir ... ve bu da durdurma sebebidir"; çare en çok 3 tarama çevrimiyle sınırlı; çözülmezse sonuç **"router bu
tasarımda eğitilemedi"** olarak raporlanır ve **karar koşuları koşulmaz**. Bu kural dokuz tarama koşusunun dokuzunda
ateşlendi (bölüm 5). Uygulama hatası aradım, bulamadım (bölüm 6). Ayrıntı ve sapma kaydı: ön-kayıt "Sapmalar / S10".
Sonraki adım sahibin kararıdır (bölüm 7).

## 1. Üç modülün SAYILMIŞ bağlantı değerleri (ZORUNLU DOĞRULAMA 1)

Nedensel maske altında, T = 512'de, maske KURULUP True girdileri SAYILARAK (formülle değil); token/head başına ortalama:

| modül | desen (donmuş) | sayılmış bağlantı | hedef 128.4 ± 0.4 |
|---|---|---|---|
| A | yerel W=150 | **128.173828125** | içinde |
| B | dilated r=2, W=256 | **128.5** | içinde |
| C | yarı yerel W=256 + yarı strided s=4, W=256 | **128.375** | içinde |
| (D / M / paylaşılan bloklar) | tam nedensel | 256.5 | — |

Aynı sayım dört ayrı yerde yapılıyor ve dördü de aynı değeri veriyor: modelin SDPA'ya verdiği `attn_mask` buffer'ından
(`tests/test_foton_model.py`), her model kurulumunda (`GPT.verify_foton_masks()` — hedef dışındaysa model KURULMAZ),
FLOP muhasebesinde (`src/flops.py`), O-X1 mikrobenchmark'ında. Her koşu sayılmış değerleri kendi `metrics.jsonl`
`meta` satırına yazıyor (dokuz tarama koşusunda da 128.173828125 / 128.5 / 128.375). EK A pini: donmuş sabitler
`(150, 2, 256, 256, 4, 256)` değilse ya da eski O-X1 tasarımı (192.25 / 128.50 / 88.31) koda girerse model kurulmaz;
bunu bozan mutantlar (F03, F04, W04, X01, X02) yakalandı.

## 2. Router nedensellik testi (ZORUNLU DOĞRULAMA 2) — GEÇTİ

`test_router_is_causal_no_token_after_k_can_change_the_module_choice` (H ve X, train ve eval modunda): prefix (0..63)
sabit, kuyruk (64..511) TAMAMEN değiştirildi → seçim, olasılıklar ve logit'ler **bit düzeyinde aynı**.
`test_predictions_after_the_prefix_are_strictly_causal`: pozisyon t ≥ 64'ün logit'i yalnızca 0..t token'larına bağlı
(bağımlılık Jacobian'ından ölçüldü). Guard'ın guard olduğu kanıtlı: router'ı bütün diziyi havuzlayan (F01) ve k+1
pozisyon okuyan (F02) mutantlar yakalandı. ZORUNLU DOĞRULAMA 3 (modül başına grafik birim testi): her modülün
attention çıktısının girdiye fonksiyonel bağımlılığı tasarlanan desenle birebir; maskeli SDPA elle yazılmış maskeli
softmax'a eşit; maskeyi ters çeviren mutant (F05) yakalandı.

## 3. Ölçülen VRAM tepesi (eğitimden ÖNCE ölçüldü; `experiments/foton-v02/vram_probe.json`)

| kol | tepe (GB) | ayrılmış (GB) | tavana pay (11.85 GB) | tok/s |
|---|---|---|---|---|
| D | 6.48 | 7.13 | 5.37 | 90.6k |
| M | 8.03 | 10.19 | 3.81 | 68.3k |
| H | 6.77 | 8.39 | 5.08 | 88.5k |
| X | 6.81 | 8.41 | 5.04 | 88.1k |

Prob artık gerçek koşuyu yeniden üretiyor: X'in dokuz tarama koşusunda loglanan gerçek tepe **6.80 GB**, D/M'in gerçek
baseline koşuları 6.480 / 8.031 GB. "Prob 0.90 GB az gösteriyor" farkının sebebi eval değil, eğitim döngüsünün önceki
mikro-batch'in logit tensörünü canlı tutmasıymış (Sapma-S7); dört kolda aynı kod, döngü değiştirilmedi.

## 4. Analitik aktif FLOP (token başına, eğitim = 3 × ileri; profiler'a güvenilmedi)

| kol | FLOP/token | M'e göre |
|---|---|---|
| D | 267,411,456 | −%0.129 |
| M | 267,755,520 | — |
| H | 265,834,602 | −%0.717 |
| X | 265,838,610 | −%0.716 |

Her koşu bu sayıyı ve bileşenlerini kendi `meta` satırına yazıyor. İki dürüstlük notu: (i) X'in bağlam sabiti üç
modülün EN BÜYÜĞÜ (128.50), router'ın seçiminden bağımsız — FOTON'un aleyhine; (ii) FOTON'un paydası M'inkinden
%0.72 KÜÇÜK çıkıyor çünkü analitik sayım maskelenen (i, j) çiftlerini saymaz; oysa çekirdek (yoğun SDPA + boolean
maske) T×T skorun tamamını üretir — çekirdek düzeyinde sayılsaydı FOTON, M'den **+%1.04** büyük olurdu. Birincil
ölçüt ön-kayıt gereği analitik sayımı kullanır; bu fark raporda tablonun başında yazılmalı. O-X1 mikrobenchmark'ı
(`ox1_microbench.json`): A / B / C = 603.6k / 602.2k / 604.2k tok/s (modüller arası fark %0.33), tam nedensel hızlı
yol 641.0k — maskeli yolun bedeli ~%6, üç modülde eşit. v01 B13'ün profiler kuralı M kolunda bilinen bir sayaç
artefaktına takılıyor (Sapma-S9; 07'de bileşen bileşen karşılaştırılacak).

## 5. V7 taramasının çıktısı (`v7_ablation.json`, `v7_router_diag.json`, `router_diag/`)

Kol X, 3 koşul × seed {101, 102, 103}, `--stop_step 2188` (LR schedule tam bütçe üzerinden), 9 koşu × ~16.1 dk = 2.4
GPU saat, tek oturum, dokuz koşuda aynı `src/` özeti. **Mekanik seçim:** üç koşul da `router_dead = 0` süzgecini geçti;
dizi-başı entropi 0.9952 (yok) / 0.9972 (entropi) / 0.9980 (lb) → **kazanan: regularizasyon yok**; preset'ler değişmedi.
Seçim val loss'a bakmadı (betik o alanı okumuyor; mutant H05 yakalandı).

| koşu | erken uyarı | pencere Hseq | pencere Hmarj | maks ölü | router = en iyi modül (şans 0.33) | aynı dizinin modül değiştirmesi (rastgele 0.67) | dizi-içi yayılım (nat) | oracle − routed (nat) | oracle'ın null'a üstünlüğü (nat) | dL/dg en iyi modülü gösteriyor | prob: en iyi modül (null) | prob: alan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x-s101 | ATEŞLENDİ | 0.9955 | 0.9920 | 0 | 0.362 | 0.692 | 0.0110 | 0.0050 | 0.0010 | 0.84 | 0.354 (0.341) | 0.967 |
| x-ent-s101 | ATEŞLENDİ | 0.9973 | 0.9934 | 0 | 0.320 | 0.682 | 0.0103 | 0.0051 | 0.0008 | 0.84 | 0.355 (0.337) | 0.977 |
| x-lb-s101 | ATEŞLENDİ | 0.9976 | 0.9993 | 0 | 0.359 | 0.699 | 0.0107 | 0.0051 | 0.0009 | 0.92 | 0.334 (0.352) | 0.976 |
| x-s102 | ATEŞLENDİ | 0.9944 | 0.9862 | 0 | 0.338 | 0.741 | 0.0114 | 0.0055 | 0.0012 | 0.77 | 0.373 (0.354) | 0.974 |
| x-ent-s102 | ATEŞLENDİ | 0.9956 | 0.9873 | 0 | 0.344 | 0.582 | 0.0115 | 0.0055 | 0.0009 | 0.79 | 0.357 (0.335) | 0.966 |
| x-lb-s102 | ATEŞLENDİ | 0.9974 | 0.9986 | 0 | 0.361 | 0.710 | 0.0118 | 0.0056 | 0.0011 | 0.81 | 0.359 (0.369) | 0.970 |
| x-s103 | ATEŞLENDİ | 0.9956 | 0.9915 | 0 | 0.372 | 0.626 | 0.0096 | 0.0043 | 0.0010 | 0.95 | 0.417 (0.387) | 0.973 |
| x-ent-s103 | ATEŞLENDİ | 0.9986 | 0.9948 | 0 | 0.348 | 0.577 | 0.0090 | 0.0041 | 0.0007 | 0.92 | 0.377 (0.346) | 0.969 |
| x-lb-s103 | ATEŞLENDİ | 0.9990 | 0.9996 | 0 | 0.323 | 0.675 | 0.0096 | 0.0049 | 0.0008 | 0.91 | 0.370 (0.352) | 0.970 |

Erken uyarı = "ÖĞRENMİYOR": dizi başına entropi 450 / 475 / 500 adımlarının üçünde de > 0.9 (Sapma-S4'teki pinli okuma).
Okumaya duyarlı değil: entropi hiçbir koşuda, hiçbir log aralığında 0.98'in altına inmedi (en düşük 0.9890). Çökme yok (ölü modül 0,
marjinal entropi ≥ 0.91): arıza ön-kaydın "daha olası" dediği ters arızadır — router kararsız, seçim fiilen rastgele.

## 6. Sebep incelemesi (V6: "sebep incelenir") — uygulama hatası BULUNMADI

1. **Gate sinyali doğru ve sağlam.** Router'ın görev loss'undan aldığı tek sinyal dizi başına `dL/dg`'dir. Her dizi her
   modüle zorlanarak fp32'de ölçüldü: en negatif `dL/dg`, dizilerin %77–95'inde GERÇEKTEN en düşük kayıplı modülü
   gösteriyor (şans %33; dizi-içi Spearman ~0.95). Betiğin kendisi sonlu farkla sınandı.
2. **bf16 sebep değil.** Eğitim gate çarpımını bf16'da yapıyor (262k terimli indirgeme); CUDA'da bf16 gradyanı fp32
   ile 0.9999 korele, işaret uyumu %99.0, argmin uyumu %99.2.
3. **Router'ın girdisi alanı taşıyor ama en iyi modülü taşımıyor.** Havuzlanmış prefix vektörü üzerinde çapraz
   doğrulamalı doğrusal prob: alan %97; en iyi modül %33–42, permütasyon null'ı %34–39. Bu girdiyle hiçbir doğrusal
   router en iyi modülü öğrenemezdi — router'ın init'i, sıcaklığı ya da gürültüsü bunu değiştirmez.
4. **Öğrenilecek hedef durağan değil.** Aynı sabit val dizisinin en iyi modülünün ardışık eval noktalarında aynı kalma
   oranı adım 1000'e kadar şans düzeyinde (0.34), koşu sonunda 0.43–0.47. Modüller arası fark dizi başına ~0.010 nat;
   modül ortalamaları 3–4 ondalığa kadar aynı; en iyi modül alandan öngörülemiyor (I ≈ 0.01 bit).
5. **Oracle bacağı.** Mükemmel seçimin router'a üstünlüğü ~0.005 nat, bunun null'ı (seçim gürültüsünü) aşan kısmı
   ~0.001 nat. (Yalnızca %50 bütçedeki tarama koşuları; karar verisi DEĞİL.)

Okuma: router "öğrenemiyor" değil; bu bütçede öğreneceği kararlı bir şey yok. Üç farklı attention grafiği, prefix'ten
öngörülebilir bir uzmanlaşma üretmedi. Tavuk-yumurta payı olabilir (seçim rastgele döndüğü için modüller her türden
veriyi görüyor ve genelci kalıyor), ama grafiklerin KENDİ farkı da alan düzeyinde bir tercih üretmiyor.

## 7. Sahibin kararı — seçenekler

* **(A) Ön-kaydın yazdığı yol.** Sapma-S4'ün sınırı içinde (hata düzeltmesi ya da V7 koşulu) kalan bir çare yok: hata
  bulunmadı, üç V7 koşulu tükendi. 2. ve 3. çevrimin meşru bir içeriği olmadığı için sonuç şimdi yazılabilir: **"router
  bu tasarımda eğitilemedi"** — FOTON aleyhine bir sonuç; karar kampanyası (~19 GPU saat) koşulmaz; 07 bu sonucu,
  tarama verisini ve sebep incelemesini raporlar. Benim önerim budur.
* **(B) Kampanyayı yine de koşmak.** V6'dan sapmadır ve öyle yazılır. Ölçeceği şey yönlendirme değil, "fiilen rastgele
  seçimle üç seyrek grafik" olur; V6-G gereği olası bir "başarılı" satırı yönlendirme iddiası olarak kurulamaz.
  Tam bütçede uzmanlaşmanın belirip belirmediğini gösterir (durağanlık ölçüsü koşu sonuna doğru yükseliyordu).
* **(C) Donmuş bir düğmeye dokunan çare çevrimi** (router girdisinin normalizasyonu, bias, sıcaklık, alan-denetimli
  ısınma, ...). Sapmadır; fiilen yeni bir deneydir (v03 ön-kaydı). Teşhis 3'e göre yalnızca router'a dokunan bir çare
  yetmez: eksik olan router'ın kapasitesi değil, modüllerin öngörülebilir biçimde ayrışmasıdır.

## 8. Kapıdan geçilirse hazır olanlar

* SIKI karar kuyruğu: `experiments/foton-v02/queues/decision_strict.json` — koşu listesi commit'li `decision.json` ile
  birebir aynı (20 koşu, 5 seed bloğu, Latin kare; V11a); commit'siz kodla başlamaz, kuyruk sürerken `src/` değişirse durur;
  yarıda kesilen blok V11a(iii) gereği TAMAMEN yeniden koşulur.
* Preset hash'leri hesaplandı ama YAZILMADI (toplam `fb9cd95a8276293e…`); onayla birlikte `preset_hash.py --write` ve
  V10 yer tutucusu doldurulur.
* Testler: 710 geçiyor; `check_prereg_frozen` (gövde tag'dekiyle aynı), `check_prereg_numbers` (90 kontrol),
  `verify_pins` (39 çivi) temiz; 06 mutasyon sınavı 67/67 (`mutation_log_06.md`; ilk koşuda hayatta kalan 3 mutantın
  üçü de test eksiğiydi ve kapatıldı).
* Sapmalar S3–S10 belgenin sonunda; tag'li gövde değişmedi. Bilinmesi gerekenler: korpusta EOS yok (S6: D10 ölçülemez,
  D11 alanla eşdoğrusal), "H önce" ile V11a çelişkisinde V11a geçerli (S5), oracle null'ının okuması tartışılabilir
  olduğu için ham dökümler diskte (S4).

---

**07'den not (2026-09-19, bu rapor sahibin kararına dayanak olduktan SONRA eklendi; yukarıdaki metin değiştirilmedi).**
Sahibin kararı: (A). Sonuç raporu `REPORT_07.md`'dir ve bu rapordaki üç ifadeyi daraltır: (i) bölüm 4'teki "+%1.04"
bir ÜST SINIRDIR (yönlendirilen 2 katman yoğun T×T sayılırsa; `docs/flops.md` §8.5); (ii) bölüm 5–6'daki "seçim fiilen
rastgele" yerine doğrusu "seçim kararsız ve en iyi modülle ilişkisiz"dir — seçim herhangi bir anda alanla ilişkilidir
(I(alan; seçim) adım 2188'de 0.025–0.215 bit), ama bu ilişki zamanda tutunmaz; (iii) bölüm 6 madde 3'teki "hiçbir
doğrusal router en iyi modülü öğrenemezdi" cümlesi ölçülenden geniştir: ölçülen, BU modüllerle ve bu girdiyle PCA-64 + L2
doğrusal probun en iyi modülü çoğunluk tabanının üstünde öngörememesidir. Bölüm 8'deki "SIKI kuyruk" tarifi de eksikti:
o tarihte `code_dirty` yalnızca kuyruk başında denetleniyordu; 07 ile birlikte her koşudan önce denetlenir.
