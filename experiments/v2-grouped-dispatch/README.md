# v2-grouped-dispatch (2026-09-16)

MoE expert dispatch'inin döngüden gruplu matris çarpımına (`torch._grouped_mm`) geçirilmesi, doğrulanması ve
dense ile MoE'nin aynı token / aynı analitik FLOP / aynı duvar saati / aynı aktif parametrede karşılaştırılması.
Anlatı, tablolar ve yorum: ana README, "Deney kaydı: `v2-grouped-dispatch`". FLOP türetimi: `docs/flops.md`.

Donmuş kayıt: buradaki sayılar değiştirilmemeli; sonraki deneyler yeni bir klasör açmalı.

## Klasör içeriği

| dosya / klasör | ne |
|---|---|
| `environment.txt` | python, GPU, sürücü, torch, MKL/AVX, tokenizer meta, tam `pip freeze` |
| `summary.json` | hız benchmark'ı, koşu grupları, eşit süre bütçesi ve dört karşılaştırma (compare_runs.py çıktısı) |
| `runs/<koşu>/` | ana koşuların `config.json`, `metrics.jsonl`, `train.log`, `provenance.json` (kod SHA-256'sı, git durumu, sürümler) |
| `runs/queue.log` | ana koşuların kuyruk günlüğü (başlangıç/bitiş saatleri, kod hash'i) |
| `speed/<koşu>/`, `speed/queue.log` | 300 adımlık hız benchmark'ı koşuları (dense; moe ve moe-noaux için `loop` ve `grouped`), 3 dönüşümlü tekrar + kuyruk günlüğü |
| `verify_dispatch_deterministic.json` | deterministik çekirdeklerle `grouped` vs `loop`: 6 durumda bit bit eşitlik |
| `verify_dispatch_default_run*.{json,log}` | eğitimdeki varsayılan çekirdeklerle eşdeğerlik; tekrar zarfı, ölçek, negatif kontrol (run2-run6'da ayrıca simetrik sıfır aday; run5-run6 teacher-forced tabanı da zarflı) |
| `verify_100step.json` | 100 adımlık eşdeğerlik koşularının analizi (önceden kaydedilmiş protokol, 6 seed) |
| `verify_100step_losses.csv` | o koşuların adım başına train loss'u, LR'ı ve val loss'u (analizi baştan üretmek için) |
| `verify_100step_queue.log`, `verify_100step_warmup20_kaotik.json` | kuyruk günlüğü ve ilk (kusurlu) protokolün sonucu |
| `flops_report.json` | analitik FLOP tablosu + profiler çapraz kontrolü (ham sayılar) |
| `queues/*.json` | tüm koşu kuyrukları: hangi koşu hangi argümanlarla çalıştı |
| `queues/v1tag_stop_step.patch` | v1 tag kopyasında 100. adımda durmak için uygulanan 5 satırlık yama (v1'de `--stop_step` yok) |
| `diagnostics/report_numbers.py`, `.log` | raporun türetilmiş sayılarını kayıtlardan yeniden hesaplar (elle yazılan sayı kalmasın diye) |
| `diagnostics/stage_times_moe_run{1,2,3}.log` | MoE katmanının ve modelin diğer parçalarının aşama aşama fwd+bwd süresi, 3 koşu (süreçler arası oynamayı görmek için) |
| `diagnostics/layer_variants_*.log` | denenen dispatch varyantlarının katman süreleri (döngü, gruplu, pad'li bmm, istiflenmiş parametre, füzyon) |
| `diagnostics/bench_dispatch_*.log` | profiler kernel adları ve sayıları: gruplu çağrı bu GPU'da grup başına ayrı gemm çalıştırıyor |
| `diagnostics/*.py` (diğer) | dispatch varyantı ve deterministik mod tanı betikleri |
| `checksums.txt` | ana koşuların `best.pt` dosyalarının SHA-256'ları (checkpoint'ler git'te değil) |

## Koşu adları

`<model>-<bütçe>-s<seed>`; model: `dense` / `moe` / `moe-noaux`, seed: 1337/1338/1339.

| bütçe | ne |
|---|---|
| `tok` | 3000 adım × 32 768 token = 98.30M token (aynı token; aktif parametreler %0.2 içinde eşit olduğu için aynı aktif parametre de) |
| `flop` | dense'in, `moe`'nin 3000 adımıyla aynı analitik eğitim FLOP'unu görmesi: 3005 adım |
| `time` | dense ve `moe-noaux`'un, `moe`'nin 3000 adımdaki eğitim duvar saatini (19.14 dk) görmesi: 4139 ve 3043 adım |

100 adımlık doğrulama koşuları: `<preset>-{loop,grouped,loopb,loopc}-s<seed>` (gerçek schedule'ın ilk 100 adımı,
`loopb`/`loopc` = `loop`'un aynı seed'le tekrarları) ve `v1tag-<preset>-s1337` (v1 tag'inin kendi kodu).

## Yeniden üretmek

Sıra önemli: eşit süre bütçesi, eşit token koşularının ölçülen hızından hesaplanıyor.

```powershell
# hız benchmark'ı (checkpoints\v2-speed) ve ana koşular (checkpoints\v2)
.\.venv\Scripts\python scripts\run_queue.py experiments\v2-grouped-dispatch\queues\speed.json
.\.venv\Scripts\python scripts\run_queue.py experiments\v2-grouped-dispatch\queues\main_tok_flop.json
.\.venv\Scripts\python scripts\compare_runs.py --budget          # eşit süre bütçesini hesaplar
.\.venv\Scripts\python scripts\run_queue.py experiments\v2-grouped-dispatch\queues\main_time.json
.\.venv\Scripts\python scripts\compare_runs.py                   # tablolar + summary.json + docs\v2_comparisons.png

# eşdeğerlik: 100 adımlık koşular (checkpoints\v2-verify-sched) ve tek adımlık GPU karşılaştırmaları
.\.venv\Scripts\python scripts\run_queue.py experiments\v2-grouped-dispatch\queues\verify_100step_sched.json
.\.venv\Scripts\python scripts\run_queue.py experiments\v2-grouped-dispatch\queues\verify_100step_sched_v1tag.json  # v1 kopyası + yama gerekir
.\.venv\Scripts\python scripts\compare_curves.py
.\.venv\Scripts\python scripts\verify_dispatch.py --out experiments\v2-grouped-dispatch\verify_dispatch_default_run1.json
.\.venv\Scripts\python scripts\verify_dispatch.py --deterministic --out experiments\v2-grouped-dispatch\verify_dispatch_deterministic.json

# raporun türetilmiş sayıları + tanı ölçümleri
.\.venv\Scripts\python experiments\v2-grouped-dispatch\diagnostics\report_numbers.py
.\.venv\Scripts\python experiments\v2-grouped-dispatch\diagnostics\stage_times.py moe
.\.venv\Scripts\python experiments\v2-grouped-dispatch\diagnostics\layer_variants.py moe
.\.venv\Scripts\python experiments\v2-grouped-dispatch\diagnostics\bench_dispatch.py balanced
```

`verify_100step_sched_v1tag.json` kuyruğu, v1 tag'inin `src/`'sinin `cwd` alanındaki klasöre çıkarılmış olmasını
bekler (`git archive v1-dense-moe-baseline src`) ve üzerine `queues/v1tag_stop_step.patch` uygulanmasını; o klasör
geçici bir dizindeydi, kuyruktaki yol bu yüzden yeniden üretimde güncellenmeli.

GPU çekirdekleri deterministik olmadığından (varsayılan modda) sayılar birebir değil, aynı eğilimler beklenmeli.
Kuyruklar bitmiş koşuları atlar; `provenance.json` hangi kodun koştuğunu SHA-256 ile sabitler. Bu koşular
commit'lenmemiş bir çalışma ağacıyla yapıldı: `provenance.json`'daki `git_head` v1 sonrası son commit'i
(`5338ce3`), `git_status` ise değişen dosyaları listeler; kodun kendisi `src_sha256` ile sabitlenmiştir
(tüm koşularda `970c99d6b89b...`).
