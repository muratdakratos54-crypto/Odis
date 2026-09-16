# (a) ve (c) koşullarının yeniden kullanılabilirliğinin kanıtı

v3 adımında `src/moe.py`'ye sigmoid gating ve seçim bias'ı eklendi. (a) ve (c) koşulları v2'de koşulan
`moe-tok` / `moe-noaux-tok` koşularıdır, yani **v3 kodundan farklı bir kodla** eğitildiler:

| koşullar | eğiten kodun `src_sha256`'sı (provenance.json'lardan) |
|---|---|
| (a) `moe-tok-s*`, (c) `moe-noaux-tok-s*` | `970c99d6b89b4d5f20375c00b8515dc3dfab60c030e9b2e6e00cb7c7fe9fd49c` |
| (b) (d) (e) (f) — v3 koşuları | `8552cc82ec52877ea4745dc1646dad60b7826fe58f5bbf2267a32e9ab134680c` |

Bu yüzden softmax + bias'sız yolun değişmediği ayrıca gösterilmeliydi. İki kanıt var; ikincisi bağımsız
olarak yeniden üretilebilir, birincisi değil.

## 1. Eğitim yörüngesinin yeniden üretimi (bağımsız doğrulanabilir)

`repro_a_100adim.json` + iki `repro_a_100adim*.log`. Bugünkü v3 koduyla (a) koşusunun konfigürasyonu
(`--preset moe --seed 1337 --stop_step 100`) yeniden koşuldu ve kaydedilmiş v2 loss'larıyla karşılaştırıldı.
GPU'da bf16 + deterministik olmayan kernel'ler yüzünden bit bit aynılık beklenmiyor, bu yüzden referans
**aynı kodun kendi tekrar zarfı**: aynı komut iki kez koşuldu.

| karşılaştırma | maks \|fark\| (loss, 4 log noktası) |
|---|---|
| v3 kodu, iki koşu (aynı kod tekrar zarfı) | 1.33e-03 |
| v2 kodu (kayıt) vs v3 kodu | **7.21e-04** |

v2/v3 farkı, aynı kodun kendi tekrar zarfının **0.54 katı** — yani kod değişikliğinin etkisi GPU tekrar
gürültüsünden ayırt edilemiyor. Bu kontrol, kaybolmuş v2 kaynağına ihtiyaç duymaz: tek gereken kaydedilmiş
`metrics.jsonl` ve bugünkü kod.

## 2. CPU parmak izi (yeniden üretilemez, olduğu gibi bırakıldı)

`fingerprint.py`: aynı seed, aynı girdi, 3 AdamW adımı; loss'lar, aux değerleri ve tüm parametrelerin
SHA-256'sı. Tek iş parçacıklı CPU'da v2 kodu (`fingerprint_v2_code.json`) ile v3 kodu
(`fingerprint_v3_code.json`) **bit bit aynı** çıktı.

**Bu kanıtın sınırı, denetimde bulundu ve burada olduğu gibi yazıyorum:**

- v2 kaynak kodu repoda **hiçbir yerde arşivli değil**. `git HEAD` v1 kodudur; v2 ve v3 değişiklikleri
  commit'lenmemiş çalışma-ağacı durumlarıydı ve çalışma ağacı artık v3'tür. Yani eşitliğin v2 tarafı
  **bağımsız olarak yeniden üretilemez**; elde yalnızca o çalışma ağacı varken üretilmiş bir JSON var.
- `fingerprint_*.json` dosyaları **provenance taşımıyor** (kod hash'i, tarih, torch sürümü yok), bu yüzden
  hangi dosyanın hangi kodla üretildiği dosyanın kendisinden anlaşılamıyor; yalnızca adından.
- Kaydın önceki hâlinde burada `d1ededc913e70bd0` diye bir "v2 kodu src hash"i yazıyordu. Bu değer
  **hiçbir artefaktla eşleşmiyor**; (a)/(c) koşularını üreten kod yukarıdaki tabloda yazan `970c99d6...`'dır.
  Yanlış hash silindi.

Bu yüzden 1. kontrol asıl kanıttır, 2. kontrol destekleyicidir. Sonraki adımlarda ders: **karşılaştırmaya
girecek bir kodu koşudan önce commit'le**, yoksa eşdeğerlik iddiasının bir tarafı kurtarılamaz hale geliyor.
