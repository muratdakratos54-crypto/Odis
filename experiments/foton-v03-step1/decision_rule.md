# v03 ADIM 1 — KARAR KURALI (veri görülmeden yazıldı; 2026-09-20; KOŞU YOK, sahibin onayını bekliyor)

**Tek soru:** prefix router'ın GÖREBİLDİĞİ en iyi bölümleme (alan) ile yönlendirme ZORLANIRSA modüller, işe yarar biçimde
ayrışıyor mu? Hayırsa hiçbir prefix router kazanamaz (eğitim dinamiğinden bağımsız). Evetse "öğrenilmiş router devralabilir
mi" SONRAKİ deneydir. "Kararlılık mı, alan hizası mı" bu adımda cevaplanmaz. Mimari olguları: `docs/ARCHITECTURE_AS_BUILT.md`.

## Kollar (6 koşu: 3 kol × seed {101, 102}; T = 512; tarama bütçesi = adım 2188, LR takvimi tam bütçe üzerinden)

| kol | model | eğitimde VE eval'de atama kuralı |
|---|---|---|
| **X-dom** | X (A, B, C) | modül = alan kimliği (code→M0:A, math→M1:B, nl→M2:C), sabit |
| **H-dom** | H (A, A, A) | modül = alan kimliği — maske farkını ağırlık farkından ayırır |
| **X-iid** | X | çekim başına BAĞIMSIZ düzgün rastgele modül. Kararlılık iddiası YOK; zorlanmış yolun NULL'ı |

Seed'ler v02 taramasıyla aynı → aynı init, aynı veri sırası → v02-X (`v7-x-s101`, `v7-x-s102`) ile EŞLİ. Router modelde
KALIR (çıkarılırsa embedding başlangıcı değişir), donuktur (zorlanmış yolda gradyan almaz, AdamW atlar). X-iid'in rastgele
ataması `train_gen`'den DEĞİL ayrı, seed'den türetilen bir generator'dan çekilir (yoksa veri sırası kayar). Atama modele
dışarıdan, dizi başına geçirilir; gate yok (ileri değer zaten 1'di).

## Kolun kaybı nasıl hesaplanır (olgu #2)

`val_atanan` = sabit 1,280 val penceresinde, pozisyon 64..511, HER dizinin EĞİTİMDEKİ atama kuralıyla gittiği modülün
kaybı. Yöntem: `eval_raw/step_*.npz`'den sonradan — `seq_masked[b, atama(b)]` ortalaması; X-dom / H-dom'da `atama(b) =
domain[b]`, X-iid'de düzgün rastgele atamanın BEKLENEN değeri = üç modülün ortalaması (`mean_fixed`; tek bir rastgele çekimin
gürültüsü eklenmez). Eval yolu da aynı atamayı tanıyacak ve `val_loss`'a onu yazacak; iki yolun 1e-4 içinde aynı sayıyı
vermesi sınanır. Router'ın rastgele seçimiyle hesaplanan sayı hiçbir tabloda kol kaybı olarak GÖRÜNMEZ: log alanı
`val_loss_router_UNUSED` adını alır. `router_*` metrikleri zorlanmış kollarda `null` ("uygulanamaz") yazılır (olgu #3).
Mikro-batch'te dizi almayan modül sayısı ve bir optimizer adımında gradyansız kalan modül sayısı loglanır (olgu #5).
FLOP bu adımda karar ölçütü DEĞİLDİR, yalnızca loglanır; H, X'ten 4,008 FLOP/token azdır (olgu #4).

## ÖNEMLİ — sahibin önerdiği ölçü X-dom'da neden KULLANILAMAZ (asistanın itirazı; onaya sunulur)

Öneri: "alan-etiketli atamanın gerçekleşen val loss'u, EN İYİ SABİT TEK MODÜLE karşı ≥ 3 puan". X-dom'da her modül
YALNIZCA kendi alanını görerek eğitilir; "bütün dizileri modül A'dan geçir" tabanı A'yı hiç görmediği iki alanda sınar ve
kötü çıkması ayrışmanın YARARINI değil, modülün alan dışında eğitilmemiş olmasını ölçer. Aynı sebeple X-dom'da modüller arası
fark, oracle tavanı ve I(alan; en iyi modül) de tanım gereği büyük / 1'e yakın çıkar: bunlar RAPORLANIR ama kanıt SAYILMAZ
(totoloji). Geçerli karşılaştırma, aynı kod yolu + aynı seed + aynı veri + modül başına aynı token payı (1/3) altında
GENELCİ modüllerdir: **X-iid**. Karar ölçüsü bu yüzden eşli farktır:

    delta_s = val_atanan(X-iid, s) − val_atanan(X-dom, s)        [nat; pozitif = alan ataması daha iyi]

## KARAR KURALI

Eşik **3 puan = 0.01019 nat** (ön-kaydın L* yerel eğimi −0.3346 ile; v02'nin 1.34 puanıyla aynı çevirme). Kural NAT
cinsinden bağlayıcıdır; puan yalnızca okunuş.

* **EVET (tavuk-yumurta doğrulanır):** `ort_s(delta) ≥ 0.01019 nat` VE iki seed'de de `delta_s > 0`. → v03 asıl deneyi:
  alan etiketiyle ısıtılıp öğrenilen router'a devredilen yönlendirme.
* **HAYIR:** aksi hâlde. → sıradaki: yönlendirilen katman payı 2/8 → 4/8, AYNI ölçüm, AYNI kural. O da HAYIR ise dizi-düzeyi
  prefix yönlendirme bu ölçekte tükenmiş sayılır; uzun bağlam YENİ hipotez olarak ayrı ön-kayıtla açılır.
* **Ön koşul (X-iid null'ı temiz mi):** aynı seed'li v02-X'e karşı, adım 2188'de: `|mean_fixed(X-iid) − mean_fixed(v02-X)| ≤
  0.015 nat` VE oracle tavanı farkı `≤ 0.4 puan`. (Kalibrasyon: v02'de aynı seed'li üç koşul arasında val farkı en çok 0.0073
  nat, tavan farkı en çok 0.20 puan; sınırlar bunların 2 katı.) Aşılırsa zorlanmış yol bir şey değiştirmiştir: karar
  OKUNMAZ, önce o anlaşılır.
* **Maske mi ağırlık mı (teşhis, karar değil):** `val_atanan(H-dom) − val_atanan(X-dom)`; |fark| < 0.005 nat ise grafik
  farkının katkısı ölçülemedi denir. (H-iid koşulmadığı için H-dom'un kendi null'ı yok: H-dom yalnızca X-dom'la kıyaslanır.)

**Ulaşılabilirlik** (`scripts/v03_step1_reachability.py`, `reachability.log`; σ = eşli farkın seed başına sd'si, v02'den
0.003–0.008 nat): etki 0 iken P(EVET) = 0.0000 / 0.0021 / 0.034; gerçek etki 3 puanken ~0.50; 5 puanken 0.9995 / 0.976 /
0.885; 7.5 puanken ≥ 0.997. Yani kural sahte EVET üretmiyor ve 5+ puanlık gerçek bir etkiyi 2 seed'le yakalıyor; 3–4 puanlık
bir etki ise yarı yarıya kaçar — n = 2'nin bedeli, bilerek kabul ediliyor (HAYIR'ın sonucu "bırak" değil "payı büyüt, yeniden
ölç"). v02 verisinden null sağlaması: genelci modüllerde alan→modül atamasının en iyi sabit modüle farkı −0.49…−0.04 puan
(9 koşu; 6 permütasyonun en iyisi bile ≤ +0.08 puan): alan ataması kendi başına kazanç üretmiyor.

## Raporlanacaklar (her kol, aynı seed'li v02-X ile yan yana; karar yalnızca yukarıdaki kuraldan)

modüller arası fark (nat/dizi; v02 0.010) · oracle tavanı (v02 1.34 puan) · B11 null'ını aşan kısım (v02 ≤ 0.28) ·
`val_atanan` ve delta · X-dom / H-dom için en iyi sabit tek modüle karşı puan ("TOTOLOJİK" etiketiyle) · I(alan; en iyi
modül) · eşit token'da (adım 2000) D ve M'ye karşı konum — eşli değil, bağlam için, karar değil.
