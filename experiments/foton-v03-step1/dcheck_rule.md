# DENSE TEŞHİS KOŞUSU — BAŞARI ÖLÇÜTÜ (koşu görülmeden yazıldı; 2026-09-20; sahibi teşhis koşusunu onayladı)

**Soru.** Bugünkü kod, dondurulmuş dense referansını AYNI SEED'de yeniden üretiyor mu? Yani D-101'in referans ailesinden
sapması (pencere ortalaması +0.0253 nat) kodun mu, seed çekiminin mi?

**Koşu.** `--preset foton-d --seed 11 --stop_step 2188`, SIKI tek koşuluk kuyruk `queues/dcheck_s11.json`, çıktı
`checkpoints/foton-v03-dcheck/dcheck-d-s11`. Karar kampanyasının klasörüne DEĞİL, ayrı bir klasöre yazar: analiz ve
seed-101 kapısı onu hiçbir koşulda görmez. Bu bir KARAR koşusu değildir, teşhis koşusudur.

**Referans.** `checkpoints/foton-v02r/r-d-s11` — dondurulmuş D referansının (n = 5, `screening_context.json`) seed 11
üyesi. Eşli: aynı seed → aynı veri sırası, aynı başlangıç ağırlıkları. Referans tam bütçeli, yeni koşu tarama bütçeli.

## 1. TAKVİM EŞLEŞMESİ — koşudan ÖNCE teyit edildi

`stop_step` koşuyu yalnızca KESER; LR takvimi her durumda `max_steps = 4375` üzerinden hesaplanır (`src/train.py: lr_at`;
bütçe eğitimin kendi `token_budget()` yardımcısından gelir ve `stop_step` o fonksiyona hiç girmez). Yeni koşunun
LOGLAYACAĞI öğrenme oranı koddan hesaplandı ve referansın logladığıyla karşılaştırıldı: **87 ortak loglanan adımın
tamamında birebir aynı, ayrışan adım yok** (`dcheck_schedule.log`; üretici `scripts/v03_dcheck.py --predict`). Adım 2000'de
iki tarafta da `0.00064692484506831491`.

Aynı teyit koşudan SONRA, yeni koşunun KENDİ logundan yeniden yapılır ve teşhisin ön koşuludur (P1). Tutmazsa teşhis
OKUNMAZ: fark koddan değil takvimden gelmiş olur.

## 2. ÖLÇÜ — tek nokta değil, eğri boyunca

    d(s) = val(yeni D-s11, s) − val(r-d-s11, s)          ortak eval adımlarında, ADIM KİMLİĞİYLE eşli
    PENCERE = (750, 1000, 1250, 1500, 1750)              D-101'i yakalayan aralık
    TÜM     = (250, 500, 750, 1000, 1250, 1500, 1750, 2000)

Neden pencere: D-101 adım 2000'de sınırı yalnızca %1.1 aştı (+0.012132 / 0.012), ama 750–1750 aralığında referans
ailesinin dışındaydı (z = +3.8 … +9.6). Tek noktalı bir ölçü bu anomaliyi neredeyse kaçırıyordu.

## 3. ÇAPALAR — hepsi koşudan ÖNCE, diskteki veriden ölçüldü

| çapa | koşular | src | pencere ort. | maks_TÜM \|d\| | maks_PENCERE | adım 0 | eğitim CE (pencere) |
|---|---|---|---|---|---|---|---|
| **AYNI KOD tekrarı (asıl)** | `foton-v02r/r-m-s11` ↔ `foton-v02r.kirli-src-iptal/r-m-s11` | ikisi de **b645ec94** | **+0.001133** | **0.007758** (adım 250) | 0.005055 | −8.3e-08 | +0.000670 |
| farklı kod sürümü tekrarı | `foton-v02r/r-d-s11` ↔ `…kirli-src-iptal/r-d-s11` | b645ec94 / d1b93ff0 | −0.000177 | 0.005801 (adım 500) | 0.002575 | 0.0 | −0.001576 |
| D-101 ANOMALİSİ | `v03s1-d-s101` ↔ 5 seed'lik referans ortalaması | — | **+0.025295** | 0.032920 | 0.032920 | — | — |

Eşikler **AYNI KODLA** koşmuş çiftten türetildi. Bu ayrım önemli: ilk yazımda çapa, iki FARKLI kod sürümüyle koşmuş
çiftten alınmıştı ve zarfı olduğundan dar gösteriyordu (koşudan önceki düşmanca inceleme yakaladı). Üçüncü bir kanıt:
`checkpoints/v2-speed/speed-dense-grouped-r1/r2/r3` (aynı src, aynı seed, üç koşu) eğitim CE'sinde bit-özdeş değil, çiftler
arası maks fark 0.0045–0.0062.

Ders: aynı kodla bile eğitim ortasında ±0.008 nat'a kadar ayrışma OLAĞANdır (kaotik büyüme) ve sonra kapanır; ama
İŞARETLİ pencere ortalaması gürültüde ≤0.0012'de kalır. Ölçüt bu yüzden işaretli pencere ortalamasına dayanır, nokta
farkına değil.

## 4. KARAR — BEŞ hâl (veri görülmeden yazıldı, sonradan değiştirilmeyecek)

| hâl | koşul | anlamı |
|---|---|---|
| **KOD SAĞLAM** | `\|ort_PENCERE d\| ≤ 0.004` VE `maks_TÜM \|d\| ≤ 0.012` | kod referansı yeniden üretti; D-101 seed çekimidir |
| **KOD SUÇLU** | `ort_PENCERE d ≥ +0.012648` VE pencerede HER adımda `d > 0` | sapma aynı seed'de de var ve anomalinin en az yarısını açıklıyor |
| **KOD SAPIYOR (kısmi)** | `+0.004 < ort_PENCERE d < +0.012648` VE pencerede HER adımda `d > 0` | gerçek ama kısmi sistematik fark; D-101'i tek başına açıklamaz |
| **TERS YÖNDE SAPMA** | `ort_PENCERE d < −0.004` VE pencerede HER adımda `d < 0` | kod referanstan DAHA İYİ; D-101 kodla açıklanamaz (yön ters) |
| **OKUNMAZ** | hiçbiri (karışık işaret, ya da ortalama temiz ama tek noktada büyük sapma) | yorumlanmaz; basılan gerekçe ÖLÇÜLEN sayılardan üretilir, sabit değildir |

Eşiklerin gerekçesi: `0.004` = aynı-kod çiftinde ölçülen `|ort|`'un (0.001133) 3.5 katı. `0.012` = aynı-kod çiftinde
ölçülen `maks_TÜM`'ün (0.007758) 1.55 katı. `+0.012648` = D-101 anomalisinin (0.025295) yarısı.

## 5. ULAŞILABİLİRLİK (`scripts/v03_dcheck.py --reach` → `dcheck_reachability.log`; beklentiler ASSERTION)

Benzetim: `d(s) = gerçek etki + gürültü`, adım-başı sd **0.0035235** (iki tekrar çiftinin 16 adım farkının kök-ortalama-karesi),
40.000 çekim/hücre, komşu adım ilişkisi ρ = 0 ve ρ = 0.3.

| gerçek etki | SAĞLAM | KISMİ | SUÇLU | TERS | OKUNMAZ |
|---|---|---|---|---|---|
| 0 | **0.984 / 0.959** | 0.003 / 0.014 | 0.000 / 0.000 | 0.004 / 0.013 | 0.009 / 0.015 |
| +0.004 (sınır) | 0.480 / 0.484 | 0.369 / 0.404 | 0 | 0 | 0.151 / 0.112 |
| +0.008 | 0.003 / 0.012 | **0.941 / 0.931** | 0.002 / 0.008 | 0 | 0.054 / 0.049 |
| +0.012648 (eşik) | 0 | 0.499 / 0.497 | **0.500 / 0.502** | 0 | 0.001 |
| +0.025295 (anomalinin tamamı) | 0 | 0 | **1.000** | 0 | 0 |
| −0.025295 | 0 | 0 | 0 | **1.000** | 0 |

Okuma, dürüstçe: kod özdeşken %96–98 SAĞLAM, yanlış "suçlu" pratikte sıfır. Suçlu eşiğinde karar **yazı-turadır** (%50) —
ama orada yazı da tura da AYNI eyleme çıkar (SUÇLU ve KISMİ hâllerinin ikisi de dense farkını ayrı bir iş olarak açar),
yani eşikteki belirsizlik eylemi değiştirmez. Asıl ayrım "yeniden üretti / üretmedi"dir ve orada güç yüksektir: gerçek etki
+0.008'de bile %93–94 KISMİ. `+0.004` sınırında SAĞLAM ile KISMİ arası yazı-turadır; bu, sınırın tanımı gereği böyledir.

**Ön koşullar (tutmazsa hiçbir hâl açıklanmaz):** P1 takvim birebir · P2 `|d(0)| ≤ 1e-6` (aynı seed → bit-özdeş başlangıç;
iki tekrar çiftinde ölçülen fark 0.0 ve −8.3e-08) · P3 ayarlar yalnızca `run_name` / `out_dir` / `stop_step` / `budget_mult`
ve varsayılan `foton_*` alanlarında ayrılır, seed iki tarafta da 11 ve kol gerçekten dense (`foton` boş, `n_experts` 0) ·
P4 iki koşu da bitmiş, ortak eval adımları tam ve sonlu.

**İkincil, BAĞLAYICI DEĞİL:** eğitim CE'sinin PENCERE farkı (kodun hesapladığı beş eval adımında; aynı-kod çiftinde +0.000670).

## 5b. SONUÇ (koşu 2026-09-20 22:03'te bitti; ölçüt yukarıda, koşu görülmeden yazılmıştı)

**KOD SAĞLAM.** Ön koşulların dördü de temiz: takvim 87 ortak adımda birebir aynı, adım 0 farkı TAM 0, ayarlar ve seed
(11) doğru, kol gerçekten dense. Ölçülen: **pencere ortalaması +0.002636** (sınır 0.004), **maks_TÜM |d| 0.007549**
(adım 250; sınır 0.012), pencerede işaretler KARIŞIK (+, +, −, +, +) — sistematik kayma deseni yok.

| adım | teşhis d(s) | D-101 anomalisi | aile sd (n=5) |
|---|---|---|---|
| 250 | −0.007549 | −0.001072 | 0.008674 |
| 500 | +0.004777 | +0.025814 | 0.032150 |
| 750 | +0.004853 | +0.032920 | 0.007588 |
| 1000 | +0.003044 | +0.028352 | 0.007500 |
| 1250 | −0.000340 | +0.021949 | 0.005470 |
| 1500 | +0.005336 | +0.021571 | 0.002240 |
| 1750 | +0.000286 | +0.021684 | 0.002161 |
| 2000 | −0.000489 | +0.012152 | 0.002733 |
| **pencere ort.** | **+0.002636** | **+0.025295** | — |

Teşhis koşusu, seed 101'in düştüğü denetimi rahatça geçiyor: adım 2000'de dondurulmuş referanstan sapması **−0.004443**
(D akla-yatkınlık sınırı ±0.012). Yani bugünkü kod seed 11'de referansı yeniden üretiyor; **D-101'in +0.0253'lük
sapması SEED ÇEKİMİDİR**, kod farkı değildir.

Not: ölçülen maks_TÜM (0.007549) aynı-kod çapasının (0.007758) hemen altında. İlk yazdığım eşik 0.008 olsaydı bu koşu
sınırın %94'üne dayanırdı; koşudan önceki inceleme turunun eşiği 0.012'ye çekmesi sonucu değiştirmedi ama payı
%63'e indirdi. Ders kayda değer: gürültü zarfı AYNI KODLA koşmuş çiftten ölçülmeli.

## 6. SONRAKİ ADIM (sahibin talimatı, koşu görülmeden)

Beş hâlde de **kalan 8 koşu koşulur; kuyruk kısaltılmaz.** D-102 ve D-103'ün karar değeri kalmadı ama teşhis değeri var:
yeni D koşuları genel olarak mı kaymış, yoksa yalnızca 101 mi? Veriyi gördükten sonra kuyruk kompozisyonunu değiştirmek
başlı başına bir sapma olur; ön-kayıtlı kuyruk neyse o koşar.

* **KOD SAĞLAM** → kayda geçer, D-101 yavaş çekimdir.
* **KOD SUÇLU** / **KOD SAPIYOR (kısmi)** / **TERS YÖNDE SAPMA** → dense yolundaki fark AYRI bir iş olarak açılır. (a)'yı
  etkilemez: X ailesi aynı seed'de v02-X'e karşı doğrulandı. Ama bulunması gerekir, çünkü D gelecekteki HER
  karşılaştırmanın tabanıdır.
* **OKUNMAZ** → sahibe sorulur.

**Her hâlde (b) kapısı bu turda OKUNMAZ** (sahibin kararı, 2026-09-20: "+0.012132 > 0.012. Kıl payı olması bir şey
değiştirmiyor; z = +9.6 zaten kıl payı olmadığını söylüyor"). Tolerans gevşetilmez; bu karar veriyi gördükten sonra
tartışmaya açılmaz. Haritanın "(b) okunamadı" hücresi geçerlidir.
