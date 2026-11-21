# ENVANTER — `foton gama v1`

**Ölçüm zamanı:** 2026-10-05 23:52 – 2026-10-06 (yerel saat). **Depo durumu:** dal `foton-v01`, HEAD `f8dcf35`.
**Kapsam:** salt-okunur envanter. Proje dosyalarından hiçbiri değiştirilmedi, silinmedi, taşınmadı; commit atılmadı; eğitim
başlatılmadı. Yazılan tek dosya bu belgedir (`.git/objects` altında oturum aracının kendiliğinden bıraktığı erişilemez
anlık-görüntü nesneleri için bkz. bölüm 9, D18).

**Okuma kılavuzu**

- `dosya:satır` gösterimleri proje köküne göre görelidir ve HEAD `f8dcf35` çalışma ağacındaki satır numaralarıdır.
- **Sayı yazımı:** binlik ayırıcı boşluk (`29 893 120`), ondalık ayırıcı nokta (`1.354`) — deponun kendi belgeleriyle aynı.
- **"sonda"** = bu envanter için yazılmış, proje dışındaki geçici klasörde duran, CPU'da (`CUDA_VISIBLE_DEVICES=-1`) koşan,
  yalnızca okuyan küçük Python betiği (model kurup parametre saymak, tokenizer ve `.bin` dosyalarını taramak gibi). GPU kullanılmadı.
- **"doğrulanamadı"** = koddan ya da diskteki dosyadan teyit edilemedi; tahmin yürütülmedi.
- README / `docs/` / ön-kayıt metinleri kanıt sayılmadı; kodla çeliştikleri yerler bölüm 9'da listelendi.

## 0. Öne çıkan bulgular

1. **RoPE (θ = 10000), RMSNorm pre-norm, SwiGLU, bias'sız Linear, bağlı embedding ve SDPA kodda ZATEN var** (`src/model.py:47-151`,
   `213-216`). Olmayanlar: **GQA / MQA** (düz MHA, `src/model.py:104`), **QK-norm**, KV cache, 512'den uzun bağlam. FlashAttention
   çekirdeği bu torch derlemesinde mevcut değil; kayıtlı çekirdek mem-efficient (bölüm 3).
2. **Rakam rakam bölme iki tokenizer'da da ZATEN açık** (`split_digits=True`, `src/tokenizer.py:68`); byte fallback açık; özel token
   yalnız unk 0 / bos 1 / eos 2, pad yok, araç token'ı yok; Türkçe / Almanca harfler sözlükte yok, her biri 2 bayt token'ı (bölüm 4).
3. **`data/foton` korpusunda EOS ve BOS hiç yok** (306 000 000 token'da id 0 / 1 / 2 = 0). Kök neden tek satır:
   `scripts/prepare_foton_corpus.py:90` `line.strip() == DOC_SEP` — `"\x1e"` Python'da boşluk sayıldığı için koşul hiç doğru olmaz;
   sınırlar `[13, 33, 13]` bayt token'ları olarak veride (127 141 adet). `data/sp8k`'da BOS / EOS doğru (bölüm 5.3).
4. Korpus içeriği adını karşılamıyor: "code" alanı %100 Markdown (stack-edu'nun Markdown shard'ı indirilmiş); NL val'in tamamı İskoç
   Parlamentosu, caselaw val'de yok (bölüm 5.1).
5. **v03 adım 1 kampanyası yarıda:** 12 koşunun 8'i koştu, `gate_s102_exceeded` (2026-09-21 03:23); seed 103 koşmadı; sonuç yalnız
   `.gitignore`'lu klasörde ve 4 commit'siz defter satırında; sürücü yeniden başlatılırsa kaydı siler (bölüm 7.4).
6. `src/` altındaki ilk bayt değişikliği kampanya kapısını, gerçek seed-101 testini ve tek-özet şartını kırar; mimari değişikliği
   D / M bit-özdeşlik testini (`tests/test_foton_model.py:409-435`), değere çivili sayıları ve 161 metin-çivili mutantı düşürür
   (bölüm 7.6, 10.0).
7. Testler: 36 dosya, 558 fonksiyon, **988 öğe — hepsi geçti (148 sn, CPU)**; hiçbir test GPU kullanmıyor, hiçbiri proje köküne
   yazmıyor. Tokenizer, veri hazırlama, `load_model`, `sample.py` için test yok (bölüm 8).
8. 20 preset: dense 29 893 120; foton-d 42 476 032 (V = 32 768); foton-h / x 55 326 720 (aktif 42 477 568); foton-m 91 734 528
   (aktif 42 533 376); bağlam hepsinde 512 (tiny'ler 128) (bölüm 3.11).
9. Eğitim: AdamW (1e-3 → 1e-4 kosinüs, token tabanlı), bf16 autocast, clip 1.0, 32 768 token / adım; ölçülen tok/s dense ≈ 118k
   (V 8 192), foton-d ≈ 92k, foton-x ≈ 84–86k, foton-m ≈ 61k; tepe VRAM 6.30 / 6.48 / 6.78 / 8.02 GB — hepsi T = 512, başka
   bağlamda ölçüm yok (bölüm 6.8).
10. 51 betiğin 5'i olduğu gibi, 14'ü uyarlamayla yeniden kullanılabilir, 32'si FOTON / deneye özel; `src/stats.py`'nin 9 fonksiyonu
    genel, 10'u FOTON karar kuralı; 10 betik argümansız çalıştırılınca dondurulmuş kayıtların üzerine yazar (bölüm 7, 9).
11. README bütünüyle v1–v3 dönemini anlatıyor (5 / 20 preset, 14 / 51 betik, "196 test"); depoda TODO / FIXME işareti yok, açık
    işler düzyazı kayıtlarda (bölüm 9).
12. Yedek: uzak depo yok; `D:\foton gama v1 yedek\` git geçmişini HEAD'e kadar içeriyor ama `data/foton` ve FOTON
    checkpoint'leri orada yok (bölüm 9, O9).

**Yöntem:** 10 paralel salt-okunur okuyucu → taslak → 8 bağımsız doğrulayıcı (907 iddia denetlendi, 32 düzeltme işlendi) →
2 haritacı + 1 eleştirmen (etki haritası). Hiçbir adımda GPU kullanılmadı, eğitim / betik koşturulmadı.

**İçindekiler:** 1 Genel durum · 2 Ortam · 3 Model mimarisi · 4 Tokenizer · 5 Veri hattı · 6 Eğitim döngüsü ·
7 FOTON kalıntıları · 8 Testler · 9 Sorunlar · 10 Etki haritası

---

## 1. Genel durum

### 1.1 Klasör ağacı (2–3 seviye)

```
foton gama v1/                          git deposu; 507 izlenen dosya
├── .git/                               ~7.3 MB
├── .gitattributes                      604 B   "* -text": satır sonu normalizasyonu KAPALI (hash çivileri için)
├── .gitignore                          236 B   .venv/, __pycache__/, data/*, checkpoints/* yok sayılır
├── .pytest_cache/                      (yok sayılır)
├── .venv/                              ~3.2 GB (yok sayılır) Python 3.13.14 sanal ortamı
├── PREREGISTRATION.md                  43 189 B    FOTON v0.1 ön-kaydı
├── PREREGISTRATION_V02.md              164 268 B   FOTON v0.2 ön-kaydı
├── README.md                           58 205 B    başlık "moe-llm"; v1 / v2 / v3 deney kayıtları
├── requirements.txt                    664 B
├── ENVANTER.md                         (bu belge; izlenmiyor)
├── checkpoints/                        78 553 699 895 B (73.16 GiB); 1 075 dosya, 158 .pt; git'te yalnız .gitkeep
│   ├── dense/                          478 444 479 B       tek koşu (v1 dense baseline)
│   ├── moe/                            1 267 476 101 B     tek koşu
│   ├── moe-noaux/                      1 267 423 690 B     tek koşu
│   ├── smoke/                          23 246 880 B        tek koşu (tiny)
│   ├── smoke-moe/                      26 010 030 B        tek koşu (tiny-moe)
│   ├── v2/                             15 712 970 385 B    18 koşu klasörü, 36 .pt
│   ├── v2-speed/                       501 568 B           15 koşu klasörü, .pt yok
│   ├── v2-verify/                      2 539 960 400 B     20 koşu klasörü, 4 .pt
│   ├── v2-verify-sched/                2 549 205 616 B     50 koşu klasörü, 4 .pt
│   ├── v3/                             15 209 760 276 B    12 koşu klasörü, 24 .pt
│   ├── foton-v02/                      10 743 879 363 B    10 koşu (baseline-d / baseline-m × seed 11..55), 20 .pt
│   ├── foton-v02r/                     10 743 892 123 B    10 koşu (r-d / r-m × seed 11..55), 20 .pt
│   ├── foton-v02r.kirli-src-iptal/     2 148 475 774 B     2 koşu (iptal edilmiş), 4 .pt
│   ├── foton-v02r-queue.out            2 479 B             (dosya)
│   ├── foton-v02-tarama/               8 282 383 585 B     9 koşu (v7-x, v7-x-ent, v7-x-lb × seed 101..103), 18 .pt
│   ├── foton-v03-dcheck/               679 763 648 B       1 koşu (dcheck-d-s11), 2 .pt
│   └── foton-v03-step1/                6 880 303 498 B     8 koşu (4 kol × seed 101, 102), 16 .pt + kampanya / kapı dosyaları
├── data/                               6 898 225 149 B (6.42 GiB); 27 dosya; git'te yalnız .gitkeep
│   ├── TinyStoriesV2-GPT4-train.txt    2 227 753 162 B
│   ├── TinyStoriesV2-GPT4-valid.txt    22 502 601 B
│   ├── sp8k/                           1 102 908 443 B     tokenizer.model/.vocab, meta.json, train.bin, val.bin
│   └── foton/                          3 545 060 943 B     tokenizer.model/.vocab, meta.json, build.log, {code,math,nl}_{train,val}.bin
│       └── _raw/                       2 931 993 479 B     4 kaynak (json.gz + txt) + tok_sample.txt
├── docs/                               6 dosya: ARCHITECTURE_AS_BUILT.md, flops.md, router_metrics.md, 3 × png
├── experiments/                        391 izlenen dosya (11 418 666 B): dondurulmuş deney kayıtları
│   ├── v1-dense-moe-baseline/          11 dosya  (dense/, moe/, moe-noaux/)
│   ├── v2-grouped-dispatch/            180 dosya (diagnostics/, queues/, runs/, speed/)
│   ├── v3-router-ablation/             106 dosya (analysis/, analysis_ilk_kosu/, checks/, queues/, runs/)
│   ├── foton-v01/                      10 dosya  (microbatch_check/)
│   ├── foton-v01-prereg/               1 dosya
│   ├── foton-v02-prereg/               33 dosya  (domains/, queues/) — run_ledger.jsonl burada
│   ├── foton-v02/                      33 dosya  (queues/, router_diag/)
│   └── foton-v03-step1/                17 dosya  (queues/)
├── scripts/                            51 .py betik (11 373 satır) + .gitkeep (+ yok sayılan __pycache__/)
├── src/                                13 .py dosya (4 082 satır) + .gitkeep (+ yok sayılan __pycache__/)
└── tests/                              36 test dosyası (10 271 satır); conftest.py / pytest.ini / pyproject.toml YOK (+ __pycache__/)
```

`src/` satır sayıları: `stats.py` 981, `train.py` 735, `model.py` 546, `attn_patterns.py` 377, `flops.py` 370, `moe.py` 341,
`config.py` 315, `data_multi.py` 160, `tokenizer.py` 108, `sample.py` 56, `data.py` 51, `checkpoint.py` 42, `__init__.py` 0.

### 1.2 Disk boyutları

| klasör | bayt | GiB | not |
|---|---:|---:|---|
| `data/` | 6 898 225 149 | 6.42 | 27 dosyanın bayt toplamıyla birebir aynı (elle toplandı) |
| `checkpoints/` | 78 553 699 895 | 73.16 | alt klasör toplamlarıyla birebir aynı; 158 `.pt` dosyası 77 994 393 508 B (%99.3) |

`.pt` dosyaları: en küçük 5 809 651 B, en büyük 1 101 330 571 B. FOTON koşu klasörleri (40 klasör) üç boyda: D kolu ≈ 680 MB
(`best.pt` 169 927 861 B + `last.pt` 509 791 597 B; 14 klasör; örnek `checkpoints/foton-v03-step1/v03s1-d-s102/`), X / H kolları
≈ 920 MB (221 342 686 B + 664 023 755–664 036 805 B; 15 klasör), M kolu ≈ 1.47 GB (367 101 903 B + 1 101 330 507–571 B; 11 klasör).

`data/` dosyaları (bayt):

| dosya | bayt |
|---|---:|
| `data/TinyStoriesV2-GPT4-train.txt` | 2 227 753 162 |
| `data/TinyStoriesV2-GPT4-valid.txt` | 22 502 601 |
| `data/sp8k/train.bin` | 1 091 626 494 |
| `data/sp8k/val.bin` | 11 024 438 |
| `data/sp8k/tokenizer.model` / `.vocab` / `meta.json` | 134 173 / 123 155 / 183 |
| `data/foton/{code,math,nl}_train.bin` | 200 000 000 (her biri) |
| `data/foton/{code,math,nl}_val.bin` | 4 000 000 (her biri) |
| `data/foton/tokenizer.model` / `.vocab` / `meta.json` / `build.log` | 539 007 / 525 961 / 1 500 / 996 |
| `data/foton/_raw/arxiv.json.gz` / `arxiv.txt` | 240 000 001 / 571 000 300 |
| `data/foton/_raw/caselaw.json.gz` / `caselaw.txt` | 170 000 001 / 281 971 318 |
| `data/foton/_raw/hansard.json.gz` / `hansard.txt` | 170 000 001 / 281 593 379 |
| `data/foton/_raw/stackv2_edu.json.gz` / `stackv2_edu.txt` | 260 000 001 / 575 418 265 |
| `data/foton/_raw/tok_sample.txt` | 382 010 213 |

Proje `C:` sürücüsünde; `C:` boş alanı ölçüm anında (2026-10-05 23:5x) 42 542 034 944 B (39.6 GiB) / 994 212 573 184 B. Diğer
sürücüler: `D:` 148.7 GB, `E:` 472.5 GB, `H:` 439.1 GB boş (ondalık GB).

### 1.3 Git

| | |
|---|---|
| Dal | `foton-v01` (HEAD `f8dcf35432ff529747091bf1e9022c0645be9c5d`, 2026-09-21 02:21:13 +0200, yazar `murat`) |
| Diğer dal | `main` → `1a592a2` (HEAD'in atası; HEAD 29 commit ileride) |
| Commit sayısı | 38 (HEAD'den erişilebilen) |
| Remote | **yok** (`git remote -v` boş) |
| Stash | yok |
| İzlenmeyen (yok sayılmayan) dosya | yok — bu belge (`ENVANTER.md`) hariç |

**Tag'ler (5, hepsi annotated):**

| tag | işaret ettiği commit | tarih | ileti (kısaltılmış) |
|---|---|---|---|
| `v1-dense-moe-baseline` | `3fe6bf8` | 2026-09-15 | Referans: dense (val 1.354), moe (val 1.317), moe-noaux (val 1.337); 3000 adım × 32 768 token |
| `v3-router-ablation` | `0d054d7` | 2026-09-16 | v3 — Router dengeleme ablasyonu (ve v2 gruplu dispatch adımının kaydı) |
| `foton-v01-prereg` | `461c9de` | 2026-09-16 | FOTON v0.1 deney ön-kaydı |
| `foton-v01-prereg-r2` | `1a592a2` (= `main`) | 2026-09-16 | FOTON v0.1 ön-kaydı r2 — denetim boşlukları kapatıldı |
| `foton-v02-prereg` | `0970b8b` | 2026-09-19 | FOTON v0.2 ön-kaydı — FOTON koşularından ÖNCE donduruldu |

**Son 15 commit:**

```
f8dcf35 2026-09-21 v03 adim 1: kalan 8 kosu ONAYLANDI - devam yolu (seed 102 / 103 on kosul kapilari + 0. adim src surekliligi); kosular bu commit'ten sonra baslatilir
2d12d50 2026-09-20 v03 adim 1: TESHIS KOSUSU SONUCU = KOD SAGLAM (D-101'in sapmasi seed cekimi); kalan 8 kosu sahibin onayini bekliyor
561de36 2026-09-20 v03 adim 1: dense TESHIS kosusunun olcutu (kosu GORULMEDEN yazildi) + takvim teyidi; kosu bu commit'ten sonra baslatilir
d95d8ab 2026-09-20 v03 adim 1: seed 101 blogu kosuldu (4 kosu), KAPI ASILDI - kuyruk durdu, kalan 8 kosu baslatilmadi
3683718 2026-09-20 v03 adim 1: 12 kosu ONAYLANDI - seed-101 gecerlilik kapisi + kampanya surucusu + dort onerinin kodu (kosular bu commit'ten sonra baslatilir)
0de580e 2026-09-20 v03 adim 1: zorlanmis-atama yolu + test + mutant + n=3 ulasilabilirlik (KOSU YOK; 12 kosu sahibin onayini bekliyor)
7e909ac 2026-09-20 v03 adim 1: karar kurali + ulasilabilirlik (veri gorulmeden; KOSU YOK) - X-dom, H-dom, X-iid
3cd8503 2026-09-19 docs/ARCHITECTURE_AS_BUILT.md: kurulduğu hâliyle mimari (D, M, H, X) + make_campaign_queue'da SIKI varsayılan
6c6f873 2026-09-19 07: FOTON v0.2 sonuç raporu — V6 yolu "router bu tasarımda eğitilemedi"; kampanya koşulmadı (sahibin kararı A)
0be7044 2026-09-19 07/1: oracle tavanı (mevcut tarama verisinden) + donmuş desenlerin grafik örtüşmesi
8d52689 2026-09-19 06/9: V7 taraması tamam — V6 erken uyarısı 9/9 koşuda ateşlendi; Sapma-S10 + KAPI RAPORU
f3e6a10 2026-09-19 06/8b: router_grad_diag sağlaması saf CE ile karşılaştırır (V7'nin regularizasyonlu koşullarında loss ek terim taşır)
680abd8 2026-09-19 06/8: doğrusal prob teşhisi — en iyi modül router'ın girdisinden öngörülebilir mi
74875e5 2026-09-19 06/7: router öğrenme-sinyali teşhisi (dL/dg) + hedefin durağanlığı ölçüleri
2cfaa26 2026-09-19 06/6: router sebep incelemesi betiği (teşhis) — V6 erken uyarısı ilk tarama koşusunda ateşlendi
```

**Commit'lenmemiş değişiklik (bu belge dışında tek dosya):** `experiments/foton-v02-prereg/run_ledger.jsonl`, +4 satır. Dört satır, v03 adım 1'in
seed 102 bloğunun kuyruk kayıtlarıdır: `v03s1-h-dom-s102` (2026-09-21T02:37:12), `v03s1-x-iid-s102` (02:54:01),
`v03s1-d-s102` (03:07:50), `v03s1-x-dom-s102` (03:23:30); hepsi `status: "bitti"`, 2188 adım, `git_head` = `f8dcf35…`.
İlk satırda `"dirty": false`, sonraki üçünde `"dirty": true` (nedeni bölüm 9'da).

**Güncel iş durumu (diskten):** v03 adım 1 kampanyası 2026-09-21 03:23:31'de `gate_s102_exceeded` (çıkış kodu 3) ile
durmuş (`checkpoints/foton-v03-step1/campaign_status.json`, `campaign.log`); seed 103 bloğu hiç koşmamış. Ayrıntı bölüm 7.4.
Envanter sırasında GPU'da eğitim süreci yoktu (`nvidia-smi`, 2026-10-05 23:52: 1 521 MiB / 12 282 MiB, yalnız masaüstü
uygulamaları; python süreci yok).

---

## 2. Ortam

| bileşen | değer | kaynak |
|---|---|---|
| İşletim sistemi | Windows 11 Home 10.0.26200 | `Win32_OperatingSystem` |
| CPU | Intel Core i7-14700F (20 çekirdek / 28 mantıksal) | `Win32_Processor` |
| RAM | 34 193 756 160 B (31.85 GiB) | `Win32_ComputerSystem` |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12 282 MiB (torch `total_memory` 12 878 086 144 B), compute capability 8.9 | `nvidia-smi`, torch |
| NVIDIA sürücüsü | 616.92 (`nvidia-smi` başlığı: CUDA UMD 13.4). **Kayıtlı bütün koşular 616.64 ile koştu** (15 `queue.log` başlığı, `experiments/v1-dense-moe-baseline/environment.txt:5`); sürücü son koşudan (2026-09-21) sonra güncellenmiş | `nvidia-smi`; `checkpoints/*/queue.log` |
| CUDA Toolkit | ayrı kurulum yok (`nvcc` yok, `CUDA_PATH` boş); CUDA çalışma zamanı torch wheel'inden (cu130) | PowerShell |
| Python | 3.13.14 (`.venv`; taban `C:\Users\murat\AppData\Local\Programs\Python\Python313`) | `.venv/pyvenv.cfg`, `sys.version` |
| PyTorch | 2.14.0+cu130 | `torch.__version__` |
| CUDA (torch build) | 13.0 | `torch.version.cuda` |
| cuDNN | 92400 | `torch.backends.cudnn.version()` |
| `torch.cuda.is_available()` | True | ölçüldü |
| SDPA | `F.scaled_dot_product_attention` mevcut; süreç varsayılanları `flash_sdp_enabled = mem_efficient_sdp_enabled = math_sdp_enabled = cudnn_sdp_enabled = True` (bayraklar). **FlashAttention çekirdeği bu torch derlemesinde YOK:** `torch.backends.cuda.is_flash_attention_available()` = False ("Torch was not compiled with flash attention"); seçilebilecek arka uçlar mem-efficient / math / cudnn | ölçüldü (sonda) |
| `torch.compile` | API mevcut; `.venv` içinde triton paketi yok ve inductor triton yokken CUDA cihazında `TritonMissing` fırlatır (`.venv/Lib/site-packages/torch/_inductor/scheduler.py:9305-9312`) → **CUDA yolu bu ortamda çalışamaz**; CPU yolu (MSVC ister) denenmedi. Kayıtlı 213 koşu `config.json`'unun hepsinde `"compile": false` | sonda + dosya taraması |
| Sistemdeki diğer Python'lar | `py -0`: 3.14.6 (varsayılan), 3.13t, 3.13 | `py -0` |

**`requirements.txt`** (kök, 664 B): `--extra-index-url https://download.pytorch.org/whl/cu130`, `torch==2.14.0+cu130` (tek
sabitlenmiş sürüm), `numpy`, `sentencepiece`, `tiktoken`, `tqdm`, `matplotlib`, `pytest` (sürümsüz). Dosyadaki not: CUDA 13.0
build'i için sürücü ≥ 580.

**Kurulu paketler (`pip freeze`, 35 paket):** certifi 2026.7.22 · charset-normalizer 3.5.1 · colorama 0.4.6 · contourpy 1.4.0 ·
cycler 0.12.1 · filelock 3.32.6 · fonttools 4.65.0 · fsspec 2026.7.0 · idna 3.19 · iniconfig 2.3.0 · Jinja2 3.1.6 ·
kiwisolver 1.5.1 · MarkupSafe 3.0.3 · matplotlib 3.11.2 · mpmath 1.3.0 · networkx 3.6.1 · **numpy 2.5.3** · packaging 26.3 ·
pillow 12.3.0 · pluggy 1.6.0 · Pygments 2.21.0 · pyparsing 3.3.2 · **pytest 9.1.1** · python-dateutil 2.9.0.post0 ·
regex 2026.9.10 · requests 2.34.2 · **sentencepiece 0.2.2** · setuptools 84.0.0 · six 1.17.0 · sympy 1.14.0 ·
**tiktoken 0.14.0** · **torch 2.14.0+cu130** · **tqdm 4.70.1** · typing_extensions 4.16.0 · urllib3 2.7.0.
Bu liste `experiments/v1-dense-moe-baseline/environment.txt` (2026-09-15) ile satır satır aynı: Python paket ortamı v1'den beri
değişmemiş; FOTON dönemi için ayrı `environment.txt` yok (koşu başına `provenance.json`'da python / torch / GPU satırı var).

---

## 3. Model mimarisi

Kaynak: `src/model.py` (546 satır), `src/moe.py` (341), `src/attn_patterns.py` (377), `src/config.py` (315). Parametre sayıları
modeller CPU'da kurularak sayıldı (sonda) ve `train.log` / `metrics.jsonl` / `best.pt` kayıtlarıyla birebir tutuyor.

### 3.1 Özet

| soru | yanıt | kanıt |
|---|---|---|
| Normalizasyon | Elle yazılmış **RMSNorm** (eps 1e-6, yalnız kazanç, bias yok); **pre-norm**; LM head'den önce son RMSNorm | `src/model.py:47-60`, `184-189`, `202`, `460` |
| QK-norm | **YOK** | `src/model.py:111-130` |
| Konum kodlaması | **RoPE ZATEN VAR** (rotate-half), **theta = 10000.0**, yalnız q ve k'ye; öğrenilmiş / sinüzoidal konum yok | `src/model.py:63-85`, `117`, `213-216`; `src/config.py:23` |
| Attention | Düz **MHA**: `n_head = 8`, ayrı KV kafa sayısı alanı YOK (GQA / MQA değil); `head_dim = d_model // n_head = 64` | `src/config.py:19`; `src/model.py:91-93`, `112-114` |
| SDPA / FlashAttention | Tek hesap yolu `F.scaled_dot_product_attention`; elle matmul + softmax yolu yok; arka uç seçen ayar yok; flash çekirdeği bu torch derlemesinde mevcut değil, kayıtlı çekirdek mem-efficient | `src/model.py:120-130`; bölüm 3.5 |
| FFN | **SwiGLU**, gizli boyut `ffn_hidden = 1408` (= 2.75 × d_model), **bias yok** | `src/model.py:135-151`; `src/config.py:22` |
| MoE | 16 expert, top-3, 1 shared expert, doğrusal router, 7 / 8 katman; dengeleme: aux loss + z-loss + (isteğe bağlı) bias; dispatch `"grouped"` (varsayılan) / `"loop"` | bölüm 3.9 |
| Embedding / LM head | **Bağlı** (`lm_head.weight = tok_emb.weight`) | `src/model.py:204-211`; `src/config.py:25` |
| Init | N(0, 0.02); residual projeksiyonlarında × (2·n_layer)^-0.5 | `src/model.py:401-413` |
| Bağlam uzunluğu | `max_seq_len = 512` (tiny preset'lerde 128); ağırlıklarda T'ye bağlı parametre yok | `src/config.py:16` |
| Üretim | `generate()` var; **KV cache yok**; temperature + top-k + EOS; top-p yok | `src/model.py:514-546` |

Hedef listesindeki kalemlerden **RoPE (ve RMSNorm, pre-norm, SwiGLU, bias'sız Linear, weight tying, SDPA) kodda zaten var**;
**olmayanlar: GQA, QK-norm, KV cache, 512'den uzun bağlam.**

### 3.2 Normalizasyon

- Sınıf `RMSNorm(nn.Module)` — `src/model.py:47-60`; `nn.LayerNorm` / `nn.RMSNorm` kullanılmıyor.
- Formül (`src/model.py:59-60`): `x * rsqrt(mean(x², dim=-1) + eps)`, sonra `x * weight`. Ortalama çıkarılmaz, bias yok.
- `eps = 1e-6` (`src/model.py:50`; hiçbir çağrı farklı eps vermiyor: `src/model.py:162`, `164`, `202`).
- Öğrenilebilir kazanç `weight = ones(dim)` (`src/model.py:53`); kapatma anahtarı yok.
- Kareler ortalaması float32'de alınır, sonuç girdi dtype'ına çevrilir (`src/model.py:57-60`).
- **Pre-norm** (`src/model.py:184-189`): `x = x + drop(attn(attn_norm(x)))`, sonra `x = x + drop(ffn(ffn_norm(x)))`.
- **Son norm:** `self.norm = RMSNorm(cfg.d_model)` (`src/model.py:202`), `x = self.norm(x)` (`src/model.py:460`), ardından `lm_head` (`src/model.py:464-465`).
- Adet: dense'te 17 RMSNorm (8 blok × 2 + 1; sonda). Embedding çıkışında ayrı norm ya da √d ölçeklemesi yok (`src/model.py:440`).

### 3.3 QK-norm

**Yok.** q ve k iç çarpımdan önce normalize edilmiyor; öğrenilebilir ölçek / sıcaklık parametresi yok. SDPA'ya `scale`
verilmiyor → PyTorch varsayılanı 1/√head_dim. Attention'ın tam yolu (`src/model.py:109-132`): `qkv(x).split` (111) →
`view(B, T, n_head, head_dim).transpose(1, 2)` (112-114) → `apply_rope(q)`, `apply_rope(k)` (117) → SDPA (120-130) →
`transpose + view` (131) → `proj` (132). `src/` içinde `qk_norm`, `F.normalize`, `logit_scale` araması eşleşme vermiyor.

### 3.4 Konum kodlaması

- **RoPE.** Tablo üretimi `precompute_rope(head_dim, max_seq_len, theta)` (`src/model.py:63-72`):
  `inv_freq = 1 / theta^(arange(0, head_dim, 2) / head_dim)`, `freqs = outer(t, inv_freq)`, dönen `cos`, `sin`.
- **theta:** `rope_theta: float = 10000.0` (`src/config.py:23`); çağrı `src/model.py:213`. 20 preset'in ve diskteki 160
  koşu `config.json`'unun hiçbirinde farklı değer yok.
- Uygulama `apply_rope(x, cos, sin)` (`src/model.py:75-85`): "rotate-half" düzeni (vektörün ilk yarısı ikinci yarısıyla
  eşleşir); **yalnız q ve k'ye** (`src/model.py:117`), v'ye değil.
- Buffer'lar `rope_cos`, `rope_sin`: şekil `(max_seq_len, head_dim/2)` = (512, 32) float32, `persistent=False` → checkpoint'e
  girmez (`src/model.py:213-216`). Tablo float32 hesaplanır, uygulama anında `x.dtype`'a çevrilir (`src/model.py:84`); yani
  bf16 autocast altında döndürme bf16'da yapılır (CPU autocast sondasıyla gözlendi; kaliteye etkisi ölçülmedi).
- Öğrenilmiş mutlak konum tablosu, sinüzoidal toplama, ALiBi yok; modeldeki tek `nn.Embedding` token embedding'idir (`src/model.py:197`).
- **Bağlam uzunluğunun zorlandığı yerler:** `assert T <= self.cfg.max_seq_len` (`src/model.py:439`, çıplak assert);
  tablo dilimi `self.rope_cos[:T]` (`src/model.py:441`); üretimde kırpma `idx[:, -self.cfg.max_seq_len:]` (`src/model.py:527`).
- `head_dim`'in çift olması denetlenmiyor: tek `head_dim` ile model kurulur; `head_dim ≥ 5` tek değerlerde ilk forward
  `RuntimeError` verir, 1 ve 3'te hata vermez ve RoPE sessizce bozuk çalışır (sonda).

### 3.5 Attention

- Sınıf `CausalSelfAttention` (`src/model.py:88-132`).
- **Kafa sayısı:** `n_head: int = 8` (`src/config.py:19`). `ModelConfig`'in 28 alanı arasında KV kafa sayısı yok; k ve v de
  `n_head` kafaya bölünür (`src/model.py:113-114`) → **düz MHA**.
- **head_dim:** `cfg.d_model // cfg.n_head` (`src/model.py:93`); ön koşul `assert cfg.d_model % cfg.n_head == 0` (`src/model.py:91`).
  Değerler: 512 / 8 = **64** (13 preset), 640 / 10 = **64** (4 S1 preset'i), 128 / 4 = **32** (3 tiny preset).
- **Projeksiyonlar:** q, k, v tek matris `self.qkv = nn.Linear(d, 3·d, bias=False)` (`src/model.py:104`); çıkış
  `self.proj = nn.Linear(d, d, bias=False)` (`src/model.py:105`). d = 512'de blok başına 786 432 + 262 144 = 1 048 576 parametre.
- **SDPA** iki çağrı biçimiyle (`src/model.py:120-130`):
  1. `pattern is None` (dense / MoE'nin bütün blokları, FOTON'un ortak blokları): `is_causal=True`, `dropout_p = cfg.dropout` (yalnız eğitimde).
  2. `pattern` verilmiş (yalnız FOTON modül blokları): `attn_mask = self.attn_mask[:, :T, :T]` (bool, `[n_head, T, T]`), `is_causal` verilmez.
- **Elle yazılmış yol** (matmul + softmax) `src/` içinde yok.
- **FlashAttention:** eğitim yolunda (`src/`) SDPA arka ucunu seçen bir ayar yok; tek genel ayar
  `torch.set_float32_matmul_precision("high")` (`src/train.py:389`). Depo genelinde `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`
  yalnız tanı / test kodunda (`tests/test_flops.py:96`, `105`; `scripts/verify_dispatch.py:212`, `216`); TF32 anahtarları ve
  deterministik mod yalnız teşhis betiklerinde. **Flash çekirdeği bu torch derlemesinde mevcut değil:**
  `torch.backends.cuda.is_flash_attention_available()` = False ("Torch was not compiled with flash attention"; sonda). Kayıtlı iki
  profil — `experiments/foton-v02/ox1_microbench.json` (2026-09-19; cuda, bf16, T 512; dört yol) ve
  `experiments/v2-grouped-dispatch/flops_report.json:91-143` (2026-09-16; dense / moe-loop / moe-grouped) — hepsinde
  `aten::_scaled_dot_product_efficient_attention` (mem-efficient). Bugün GPU'da hangi çekirdeğin seçildiği bu envanterde
  gözlenmedi (GPU kullanılmadı); seçilebilecekler mem-efficient / math / cudnn.
- Dikkat dropout'u `cfg.dropout` (varsayılan 0.0, `src/config.py:24`; hiçbir preset ve hiçbir kayıtlı koşu değiştirmemiş).

### 3.6 FFN

- Sınıf `SwiGLU` (`src/model.py:135-151`): `w_down(silu(w_gate(x)) * w_up(x))` (`src/model.py:151`).
- Üç matris de `bias=False` (`src/model.py:145-147`).
- **Gizli boyut:** çarpan alanı yok, doğrudan tamsayı `ffn_hidden: int = 1408` (`src/config.py:22`) = 2.75 × 512
  (8/3·d ≈ 1365'in 64'ün katına yuvarlanmışı; `src/config.py:20-21`). S1 preset'lerinde 1760 (= 2.75 × 640), tiny'lerde 352.
- d = 512'de FFN 3 × 720 896 = 2 162 688 parametre. SwiGLU içinde dropout yok; blok düzeyinde `self.drop` (`src/model.py:181`, `189`).
- Dense blok toplamı (d = 512): **3 212 288** = attention 1 048 576 + FFN 2 162 688 + iki norm 1 024.

### 3.7 Embedding / LM head paylaşımı ve init

- **Weight tying açık:** `tie_embeddings: bool = True` (`src/config.py:25`); `self.lm_head.weight = self.tok_emb.weight`
  (`src/model.py:211`; koşul `204`). `lm_head = nn.Linear(d_model, vocab_size, bias=False)` (`src/model.py:203`). Hiçbir preset kapatmıyor.
- **Init** — `GPT._init_weights` (`src/model.py:401-413`), `self.apply(...)` ile (`src/model.py:218`):
  - bütün `nn.Linear` ve `nn.Embedding`: N(0, 0.02);
  - residual'a yazan projeksiyonlar (`attn.proj` — `src/model.py:106`; `SwiGLU.w_down` — `src/model.py:148`): 0.02 × (2·n_layer)^-0.5
    → n_layer = 8'de 0.005 (sonda: ölçülen std 0.00500);
  - bias varsa sıfır (modelde bias'lı Linear yok); RMSNorm kazancı 1.0.
  - MoE router'ı (`src/moe.py:215`), expert'ler ve FOTON router'ı (`src/model.py:259`) için özel init yok (genel kuralı alırlar).
- Bağlı embedding init sırasında en son `lm_head` üzerinden yeniden çekilir (sonda); sonuç: modele parametre eklemek / çıkarmak
  aynı seed'deki embedding başlangıcını değiştirir.

### 3.8 Blok yapısı, forward, loss, üretim

- `Block` (`src/model.py:154-190`): `attn_norm → attn → +residual`, `ffn_norm → ffn (SwiGLU ya da MoE) → +residual`.
  Dropout üç yerde (dikkat olasılıkları, attention çıkışı, FFN çıkışı), hepsi `cfg.dropout = 0.0`.
- `GPT.forward(idx, targets=None, force_module=None, assign=None)` → `(logits, loss, info)` (`src/model.py:431-432`).
  `targets is None` iken yalnız son konumun logit'i döner (`src/model.py:462-464`). `force_module` / `assign` yalnız FOTON'da geçerli.
- **Loss** (`src/model.py:468`): `F.cross_entropy(logits.float().view(-1, V), targets.reshape(-1), ignore_index=-1)`,
  varsayılan `mean`. `src/` içinde hedefe −1 yazan kod yok; yani `ignore_index` eğitim / eval yolunda fiilen tetiklenmiyor.
  MoE'de `loss = ce + aux_loss_coef·aux + z_loss_coef·z` (`src/model.py:477-479`).
- `generate(idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None)` (`src/model.py:514-546`): her adımda bağlam baştan
  işlenir (**KV cache yok**, `src/model.py:523-528`); `temperature <= 0` → argmax; top-k; `eos_id` verilirse batch'in tamamı
  bitince durur (`src/model.py:540-545`). top-p / tekrar cezası yok.
- Gradient checkpointing yok. `torch.compile` yalnız `src/train.py:485`'te, `compile: bool = False` (`src/config.py:128`).
- Parametre sayma yardımcısı: `GPT.num_params(exclude_embedding=False, active_only=False)` (`src/model.py:415-429`).

### 3.9 MoE

Kaynak `src/moe.py`; blok içinde FFN yuvasına takılır: `self.is_moe = cfg.n_experts > 0 and layer_idx >= cfg.first_dense_layers`
(`src/model.py:165`). FOTON ile birlikte kurulamaz (`src/model.py:237-238`).

| konu | bulgu | kanıt |
|---|---|---|
| Expert sayısı | `n_experts` varsayılan 0 (dense); MoE preset'lerinde **16** (tiny-moe'de 8) | `src/config.py:27`, `155-158` |
| top-k | varsayılan 2; MoE preset'lerinde **3** (tiny-moe'de 2) | `src/config.py:28` |
| Shared expert | varsayılan `n_shared_experts = 0`; altı MoE preset'inde **1**. Ayrı expert değil, gizli boyutu `expert_hidden × n_shared` olan TEK SwiGLU; her token'a gate'siz eklenir | `src/config.py:30`, `156`, `162`, `169`, `185`, `257`, `285`; `src/model.py:167`, `173`; `src/moe.py:270-271` |
| Hangi katmanlar | `first_dense_layers = 1` → katman 0 dense, katman 1..7 MoE (**7 / 8**; S1'de 9 / 10) | `src/config.py:33`; `src/model.py:165` |
| Expert FFN | dense ile aynı sınıf: `SwiGLU(d_model, expert_hidden)`, `expert_hidden = 352` (= 1408 / 4; S1'de 440) | `src/model.py:170`; `src/config.py:29` |
| Aktif genişlik eşitliği | `top_k·352 + 352 = 1408 = ffn_hidden` — kodda assert ile değil preset aritmetiğiyle (`tiny-moe`'da sağlanmaz: 2·88 + 88 = 264 ≠ 352) | `src/config.py:151-154`, `283-288` |
| Router türü | tek doğrusal katman `nn.Linear(d_model, n_experts, bias=False)`; girdi `ffn_norm(x)`; float32'de, autocast dışında; gürültü ve sıcaklık yok | `src/moe.py:215`, `221`, `228-229`; `src/model.py:186` |
| Kapı | `router_gate = "softmax"` (varsayılan): önce bütün E üzerinde softmax, SONRA top-k; `"sigmoid"`: expert başına bağımsız sigmoid, sonra top-k | `src/config.py:37`; `src/moe.py:232-238` |
| Yeniden normalizasyon | `norm_topk_prob = True` ve k > 1 iken seçilen k ağırlığın toplamı 1 yapılır | `src/config.py:34`; `src/moe.py:243-246` |
| Yük dengeleme (1) | Switch tarzı aux loss: `L = E · Σ f·P` (f = türevsiz yük payı, P = ortalama olasılık); katsayı `aux_loss_coef = 0.01` | `src/moe.py:55-77`; `src/config.py:45` |
| Yük dengeleme (2) | Router z-loss: `mean(logsumexp(logits)²)`; katsayı `z_loss_coef = 0.001` | `src/moe.py:80-88`; `src/config.py:46` |
| Yük dengeleme (3) | Aux-loss-free bias: `expert_bias` (persistent buffer, gradyansız) yalnız SEÇİME eklenir (eval / çıkarımda da; eğitime bağlı olan yalnız `bias_counts` birikimi, `273-274`); güncelleme `expert_bias -= rate · sign(load − 1/E)`; `bias_update_rate` varsayılan 0.0, `moe-auxfree` / `foton-m` / `foton-m-s1`'de 0.001 | `src/moe.py:211-212`, `233-236`, `273-274`, `287-298`; `src/config.py:41` |
| Bias güncelleme çağrısı | her optimizer adımından sonra bir kez | `src/train.py:675`; `src/moe.py:143-146` |
| Aux'un loss'a eklenmesi | modelin `forward`'ında; aux ve z MoE katmanlarının ORTALAMASI | `src/model.py:477-479` |
| Dispatch yolu | `moe_impl = "grouped"` (varsayılan): atamalar expert'e göre bir kez sıralanır, üç `torch._grouped_mm` çağrısı (özel / alt çizgili API); `"loop"`: expert başına dilim + E modül çağrısı; grouped desteklenmezse uyarıyla loop'a düşer | `src/config.py:49`; `src/moe.py:253-264`, `312-341` |
| Kapasite / token düşürme | **yok** (dropless); birleştirme `index_add_` | `src/moe.py:267-269` |
| Router ölçüleri | `load_cv`, `load_max_ratio`, `load_min_ratio`, `load_entropy`, `dead_experts`, `bias_range` | `src/moe.py:91-123`; `src/train.py:693-703` |

MoE'li preset'ler: `moe` (aux 0.01), `moe-noaux` (aux 0.0), `moe-auxfree` (sigmoid kapı + bias 0.001, aux 0.0), `foton-m`
(ModelConfig'i `moe-auxfree` ile alan alan aynı), `foton-m-s1` (expert 440, 9 / 10 katman), `tiny-moe` (8 expert, top-2, expert 88).
z-loss (0.001) altı MoE preset'inin hepsinde açık (aux-free kollar dahil); `moe-noaux`'ta aux loss hesaplanıp loglanır, yalnız katsayısı 0.
v3 ablasyonunun aux katsayısı kolları ayrı preset değil, `--preset moe --aux_loss_coef X` ezmesidir
(`experiments/v3-router-ablation/queues/router_ablation.json`).

Sayılar (sonda, `moe` preset'i, V = 8192): expert başına 540 672 parametre; katman başına router 8 192; MoE bloğu 10 249 216
(dense blok 3 212 288); toplam 79 151 616, aktif 29 950 464; aktif fark = 7 router = 57 344.

### 3.10 FOTON kolları (özet; ayrıntı bölüm 7)

- Alan `ModelConfig.foton: str = ""` (`src/config.py:57`); kollar `FOTON_ARMS = {"h": ("A","A","A"), "x": ("A","B","C")}` (`src/model.py:41`).
  D = `foton-d` (dense), M = `foton-m` (MoE), H = üç modül de aynı desen, X = üç farklı desen.
- Yapı: `foton_pre = 3` ortak blok → router → seçilen modülün `foton_routed = 2` bloğu → `foton_post = 3` ortak blok
  (`src/config.py:59-61`; `src/model.py:251-256`, `337-372`). Kurulan 12 blok, bir dizinin geçtiği 8. Modül ağırlıkları AYRI.
- Router: `nn.Linear(d_model, 3, bias=False)` (`src/model.py:259`); girdi ilk `foton_prefix_k = 64` pozisyonun ortalaması
  (`src/model.py:315-316`); **dizi düzeyi**, softmax → argmax (`src/model.py:319-320`), straight-through gate (`src/model.py:367-369`).
- `foton_assign` (`src/config.py:72`): `""` öğrenilmiş router, `"domain"` alan kimliğiyle, `"iid"` çekim başına rastgele zorlanmış atama.
- Dikkat desenleri (`src/attn_patterns.py:43-46`, `251-259`): A = yerel pencere W = 150; B = dilated r = 2, W = 256; C = kafaların
  yarısı yerel W = 256, yarısı strided s = 4. Bool maske `[n_head, T, T]`, persistent olmayan buffer (`src/model.py:99-102`).
  T = 512'de sayılmış ortalama bağlantı: A 128.173828125, B 128.5, C 128.375 (tam nedensel 256.5).

### 3.11 Config alanları ve preset'ler

- `ModelConfig` (`src/config.py:13-72`): **28 alan** — 9 taban mimari (`vocab_size`, `max_seq_len`, `d_model`, `n_layer`,
  `n_head`, `ffn_hidden`, `rope_theta`, `dropout`, `tie_embeddings`), 11 MoE (`n_experts`, `top_k`, `expert_hidden`,
  `n_shared_experts`, `first_dense_layers`, `norm_topk_prob`, `router_gate`, `bias_update_rate`, `aux_loss_coef`, `z_loss_coef`,
  `moe_impl`), 8 FOTON (`foton`, `foton_prefix_k`, `foton_pre`, `foton_routed`, `foton_post`, `foton_reg`, `foton_reg_coef`, `foton_assign`).
- `TrainConfig` (`src/config.py:75-138`): **28 alan** (bölüm 6).
- Config'te OLMAYAN, kodda sabit kararlar: norm türü, pre-norm, bias'sızlık, init şeması, KV kafa sayısı, QK-norm, FOTON modül sayısı (3).
- `PRESETS` (`src/config.py:142-289`): **20 preset**. `build_configs` (`src/config.py:300-315`) `--preset` + her dataclass alanı
  için otomatik `--<alan>` bayrağı üretir; alan türü yalnız `int / float / str / bool` olabilir (`src/config.py:304-308`).
- `vocab_size` preset'te verilmez (varsayılan 8192, `src/config.py:15`); eğitimde her zaman tokenizer'dan ezilir
  (`src/train.py:394`; `src/tokenizer.py:43`). Aşağıdaki sayılar **etkin** sözlükle (`data/sp8k` = 8 192, `data/foton` = 32 768) hesaplandı.

**"Aktif parametre"nin tanımı kodda:** `GPT.num_params(active_only=True)` (`src/model.py:415-429`) — MoE'de **token başına**
(katman başına `top_k` expert + shared + router; kalan `E − k` expert düşülür), FOTON'da **dizi başına** (tek modül; diğer iki
modül düşülür; router sayılır). Bağlı embedding bir kez sayılır. İki birim aynı değildir (tek dizide MoE expert'lerinin çoğu çalışır).

| # | preset | tanım (`src/config.py`) | d_model | n_layer | n_head | head_dim | bağlam | V (etkin) | **toplam parametre** | **aktif parametre** | embedding hariç toplam | diskte koşu |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `tiny` | 143-147 | 128 | 2 | 4 | 32 | 128 | 8 192 | 1 450 624 | 1 450 624 | 402 048 | 1 |
| 2 | `dense` | 150 | 512 | 8 | 8 | 64 | 512 | 8 192 | **29 893 120** | **29 893 120** | 25 698 816 | 13 |
| 3 | `moe` | 155-158 | 512 | 8 | 8 | 64 | 512 | 8 192 | **79 151 616** | **29 950 464** | 74 957 312 | 54 |
| 4 | `moe-noaux` | 161-165 | 512 | 8 | 8 | 64 | 512 | 8 192 | 79 151 616 | 29 950 464 | 74 957 312 | 48 |
| 5 | `moe-auxfree` | 168-172 | 512 | 8 | 8 | 64 | 512 | 8 192 | 79 151 616 | 29 950 464 | 74 957 312 | 3 |
| 6 | `foton-d` | 178-183 | 512 | 8 | 8 | 64 | 512 | 32 768 | **42 476 032** | **42 476 032** | 25 698 816 | 14 |
| 7 | `foton-m` | 184-191 | 512 | 8 | 8 | 64 | 512 | 32 768 | **91 734 528** | **42 533 376** | 74 957 312 | 11 |
| 8 | `foton-h` | 195-200 | 512 | 8 (12 blok kurulur) | 8 | 64 | 512 | 32 768 | **55 326 720** | **42 477 568** | 38 549 504 | 0 |
| 9 | `foton-x` | 201-206 | 512 | 8 (12) | 8 | 64 | 512 | 32 768 | 55 326 720 | 42 477 568 | 38 549 504 | 3 |
| 10 | `foton-x-ent` | 211-216 | 512 | 8 (12) | 8 | 64 | 512 | 32 768 | 55 326 720 | 42 477 568 | 38 549 504 | 3 |
| 11 | `foton-x-lb` | 217-222 | 512 | 8 (12) | 8 | 64 | 512 | 32 768 | 55 326 720 | 42 477 568 | 38 549 504 | 3 |
| 12 | `foton-x-dom` | 225-230 | 512 | 8 (12) | 8 | 64 | 512 | 32 768 | 55 326 720 | 42 477 568 | 38 549 504 | 2 |
| 13 | `foton-h-dom` | 231-236 | 512 | 8 (12) | 8 | 64 | 512 | 32 768 | 55 326 720 | 42 477 568 | 38 549 504 | 2 |
| 14 | `foton-x-iid` | 237-242 | 512 | 8 (12) | 8 | 64 | 512 | 32 768 | 55 326 720 | 42 477 568 | 38 549 504 | 2 |
| 15 | `foton-d-s1` | 249-254 | 640 | 10 | 10 | 64 | 512 | 32 768 | **71 160 960** | **71 160 960** | 50 189 440 | 0 |
| 16 | `foton-m-s1` | 255-262 | 640 | 10 | 10 | 64 | 512 | 32 768 | **170 094 720** | **71 253 120** | 149 123 200 | 0 |
| 17 | `foton-h-s1` | 263-269 | 640 | 10 (14) | 10 | 64 | 512 | 32 768 | **91 238 400** | **71 162 880** | 70 266 880 | 0 |
| 18 | `foton-x-s1` | 270-276 | 640 | 10 (14) | 10 | 64 | 512 | 32 768 | 91 238 400 | 71 162 880 | 70 266 880 | 0 |
| 19 | `tiny-foton` | 277-282 | 128 | 4 (8) | 4 | 32 | 128 | 8 192 | 2 656 768 | 1 852 928 | 1 608 192 | 0 |
| 20 | `tiny-moe` | 283-288 | 128 | 2 | 4 | 32 | 128 | 8 192 | 1 620 608 | 1 417 856 | 572 032 | 1 |

Notlar:

- "Diskte koşu" = `checkpoints/` altındaki 160 `config.json`'un KÖKEN preset'ine eşlenmesi (`provenance.json` / kuyruk dosyasındaki
  `--preset`; CLI ezmeleri dahil — ör. `moe`'nin 54 koşusunun 9'u `--aux_loss_coef` taraması, 27'si `--moe_impl loop`; model
  alanları preset'le birebir aynı olan koşu sayısı yalnız 97). `checkpoints/` altında koşu klasörü olmayan preset'ler: `foton-h`
  (yalnız VRAM sondasında 9 adım koştu: `experiments/foton-v02/vram_probe.json`), dört S1 preset'i, `tiny-foton` (yalnız testlerde).
- Embedding (V·d), d = 512: V = 8 192'de 4 194 304; V = 32 768'de 16 777 216 (`foton-d` toplamının %39.5'i); S1'de (d = 640)
  20 971 520; üç tiny preset'te (d = 128) 1 048 576.
- Zorlanmış atama kollarında (`foton-x-dom`, `foton-h-dom`, `foton-x-iid`) router modelde durur ama çalışmaz; kodun "aktif"
  sayısı onu yine sayar. Router'ın 1 536 parametresi düşülürse aktif = 42 476 032 (= `foton-d`). Bu "sıkı aktif" tanımı koda
  değil bu envantere aittir (tek token'lık ileri + geri geçişte gradyan alan parametreler sayılarak sınandı).
- Preset nesnesi tokenizer çözülmeden (V = 8 192 ile) kurulursa `foton-d` 29 893 120 parametre verir; koşulan model 42 476 032'dir.
- Bağımsız sağlamalar (hepsi birebir): `checkpoints/dense/train.log:1` ("29.89M"), `checkpoints/moe/train.log:1-2`,
  v03 koşularının `metrics.jsonl` meta kaydı (`params_total` 42476032 / 55326720, `params_active` 42476032 / 42477568),
  5 koşunun `best.pt` state_dict'i, `docs/ARCHITECTURE_AS_BUILT.md:29`, `40`, `60`.
- Analitik eğitim FLOP / token (`src/flops.py`; hesap, ölçüm değil): dense 191 913 984 · moe 192 258 048 · foton-d 267 411 456 ·
  foton-m 267 755 520 · foton-h 265 834 602 · foton-x 265 838 610 · foton-d-s1 446 584 320.

Preset aileleri arası farklar (ModelConfig): `moe` = dense + `n_experts=16, top_k=3, n_shared_experts=1`; `moe-noaux` = moe +
`aux_loss_coef=0.0`; `moe-auxfree` = moe-noaux + `router_gate="sigmoid", bias_update_rate=0.001`; `foton-d` = dense modeli;
`foton-m` = moe-auxfree modeli; `foton-h` / `foton-x` = `foton="h"` / `"x"`; `-ent` / `-lb` = `foton_reg`; `-dom` / `-iid` =
`foton_assign`; `-s1` = 640 × 10, n_head 10, ffn 1760 (M'de ayrıca `expert_hidden` 440; FOTON'da bölüşüm 4 + 2 + 4); `tiny-foton` =
128 × 4, 1 + 2 + 1, prefix 16.
Kod düzeyinde kalıtım yok: her preset bağımsız literal'dir; 13 `foton-*` preset'inin TrainConfig'i `run_name` dışında aynı
literal'in 13 kopyasıdır.

---

## 4. Tokenizer

İki tokenizer var; ikisi de **aynı fonksiyonla** eğitilmiş: `SentencePieceTokenizer.train(input_file, model_prefix, vocab_size)`
(`src/tokenizer.py:57-77`). Aşağıdaki "dosyadaki kayıt" sütunu `.model` dosyalarının içindeki `trainer_spec` / `normalizer_spec`
alanlarından okundu (sonda; `.venv`'de `protobuf` paketi olmadığı için tel biçimi elle çözüldü ve `SentencePieceProcessor` ile
parça parça çapraz doğrulandı).

### 4.1 Özet

| | `data/sp8k` | `data/foton` | kanıt |
|---|---|---|---|
| Tür | SentencePiece **BPE** | SentencePiece **BPE** | `src/tokenizer.py:64`; `trainer_spec.model_type = BPE` |
| Vocab (istenen → dosyada) | 8 192 → **8 192** | 32 768 → **32 768** | `scripts/prepare_data.py:92`; `scripts/prepare_foton_corpus.py:151`; `GetPieceSize()` |
| Parça türleri | 1 UNKNOWN + 2 CONTROL + 256 BYTE + 7 933 NORMAL | 1 UNKNOWN + 2 CONTROL + 256 BYTE + 32 509 NORMAL | sonda |
| `tokenizer.model` | 134 173 B, SHA-256 `691d5e2a…e6a4` (çivi `experiments/v1-dense-moe-baseline/checksums.txt:8` ile eşleşiyor) | 539 007 B, SHA-256 `0f85409c…e8af` (çivi `experiments/foton-v02-prereg/pins.json` ile eşleşiyor) | sonda |
| Kullanan preset'ler | 7: `tiny`, `dense`, `moe`, `moe-noaux`, `moe-auxfree`, `tiny-foton`, `tiny-moe` | 13: bütün `foton-*` preset'leri | `src/config.py:78` ve preset'lerdeki `data_dir` |

Tokenizer dosyaları git'te değil (`.gitignore`: `data/*`). `tiktoken` kurulu ve kodda bir yolu var (`TiktokenTokenizer`,
`src/tokenizer.py:80-97`; `scripts/prepare_data.py --tokenizer gpt2`) ama diskte tiktoken'lı veri kümesi, koşu ya da test yok
(160 koşu `config.json`'unun hepsi `data/sp8k` ya da `data/foton`): ölü / isteğe bağlı kod yolu.

### 4.2 Özel token'lar ve id'leri (iki tokenizer'da AYNI)

| işlev | id | parça | tür |
|---|---:|---|---|
| unk | 0 | `<unk>` | UNKNOWN |
| bos | 1 | `<s>` | CONTROL |
| eos | 2 | `</s>` | CONTROL |
| pad | **−1 (kapalı)** | sözlükte `<pad>` parçası yok | — |
| bayt parçaları | 3 … 258 | `<0x00>` … `<0xFF>` (**id = 3 + bayt değeri**) | BYTE |

- Kod: `unk_id=0, bos_id=1, eos_id=2, pad_id=-1` (`src/tokenizer.py:72`). Sık geçen bayt id'leri: `<0x09>` sekme = 12,
  `<0x0A>` LF = **13**, `<0x0D>` CR = 16, `<0x1E>` = **33**.
- `control_symbols` / `user_defined_symbols` boş (koda hiç geçilmiyor). Sözlükte OLMAYAN: `<pad>`, `<|endoftext|>`, `<mask>`,
  `<sep>`, `<cls>`, araç kullanımı için herhangi bir özel token.
- NORMAL parçalar id 259'dan başlar. `"</s>"` / `"<s>"` ham metni kontrol id'sine değil düz parçalara kodlanır; `decode` kontrol
  id'lerini sessizce düşürür.
- Sarmalayıcı (`src/tokenizer.py:23-55`) yalnız `vocab_size`, `bos_id`, `eos_id` sunar; **`encode` BOS / EOS eklemez**; pad / unk /
  özel token arayüzü yok. `load_tokenizer(data_dir)` (`src/tokenizer.py:100-108`) `meta.json`'dan yalnız `tokenizer` ve
  `model_file` (sentencepiece dalı, `104-105`) ya da `name` (tiktoken dalı, `107`) alanlarını okur; `vocab_size` / `bos_id` /
  `eos_id` alanları okunmaz, değerler `.model` dosyasından gelir (`src/tokenizer.py:43-45`).

### 4.3 Eğitim ayarları (`src/tokenizer.py`; iki `.model` dosyasındaki kayıtla birebir)

| ayar | değer | satır |
|---|---|---|
| `model_type` | `"bpe"` | 64 |
| `byte_fallback` | **True** | 66 |
| `character_coverage` | 0.9995 | 67 |
| `split_digits` | **True** | 68 |
| `normalization_rule_name` | `"identity"` (NFKC yok) | 69 |
| `remove_extra_whitespaces` / `allow_whitespace_only_pieces` | False / True | 70-71 |
| `input_sentence_size` / `shuffle_input_sentence` | 2 000 000 / True | 73-74 |
| `num_threads` | `os.cpu_count()` (kayıtta 28) | 75 |
| `minloglevel` | 2 (eğitim uyarıları kaydedilmiyor) | 76 |
| `max_sentence_length` | koda geçilmiyor → kütüphane varsayılanı 4192 bayt | — |
| `add_dummy_prefix` / `escape_whitespaces` | koda geçilmiyor → varsayılan true / true (her `encode` çıktısı `▁` ile başlar; sözcük başındaki sayı önce ayrı bir `▁` alır) | — |
| tohum (seed) | koda geçilmiyor | — |

### 4.4 Sayılar nasıl bölünüyor

**Rakam rakam bölme iki tokenizer'da da ZATEN AÇIK** (`split_digits=True`). İki sözlükte de birden çok rakam içeren parça **0**.
Fark: `foton`'da 10 tek-rakam parçası var; `sp8k`'da rakamlar alfabede yok, bayt parçasına düşüyor.

```
metin            sp8k                                             foton
"7"              ▁ <0x37>                                  (2)    ▁ 7                    (2)
"1234"           ▁ <0x31> <0x32> <0x33> <0x34>             (5)    ▁ 1 2 3 4              (5)
"3.14159"        ▁ <0x33> . <0x31> <0x34> <0x31> <0x35> <0x39>  (8)   ▁ 3 . 1 4 1 5 9   (8)
"1,000,000"      (10 token)                                       ▁ 1 , 0 0 0 , 0 0 0   (10)
"0x1F"           ▁ <0x30> x <0x31> F                       (5)    ▁ 0 x 1 F              (5)
"1e-5"           ▁ <0x31> e <0x2D> <0x35>                  (5)    ▁ 1 e - 5              (5)
"2 + 2 = 4"      (10 token)                                       ▁ 2 ▁+ ▁ 2 ▁= ▁ 4      (8)
"x = 12345 * 67" (14 token)                                       ▁x ▁= ▁ 1 2 3 4 5 ▁* ▁ 6 7  (12)
```

Sözcük başındaki her sayı, rakamlardan önce ayrı bir `▁` token'ı harcar (`"42"` → 3 token). Rakam token'larının korpustaki payı:
`foton` nl_train %9.022, code_train %6.543, math_train %11.022; `sp8k` train %0.007.

### 4.5 Hangi veride eğitildi

- **`foton`:** tek girdi `data/foton/_raw/tok_sample.txt` (382 010 213 B, 4 521 669 satır). Üretimi
  `scripts/prepare_foton_corpus.py:187-205`: alan başına 120 MiB (125 829 120 B) kota, alandaki kaynaklara eşit bölünür, her ham
  `.txt` dosyasının **başından** satır satır alınır. Bayt payı: caselaw %16.667, hansard %16.667, stackv2_edu %33.333, arxiv %33.333
  (satır payı farklı: nl %9.8, code %49.5, math %40.7). Bu mantık bugünkü ham dosyalar üzerinde yeniden oynatıldı; SHA-256
  diskteki dosyayla birebir aynı (`3cd2457e…244a`). Geçerli satırların %44.23'ü örneklenir (2 000 000 / 4 521 599); **hangi
  satırların seçildiği doğrulanamadı** (tohum yok, eğitim günlüğü yok). Dört kaynak da İngilizce ağırlıklı; Almanca / Türkçe kaynak yok.
- **`sp8k`:** `data/TinyStoriesV2-GPT4-train.txt`'nin ilk `--tok_train_mb` MB'ındaki (varsayılan 100) hikâyelerin boş olmayan
  satırları (`scripts/prepare_data.py:95`, `105-117`). Örnek dosyası (`tok_sample.txt`) diskte yok; `.model` içindeki kayıt
  deponun eski konumunu gösteriyor (`C:\Users\murat\moe-llm\data\sp8k\tok_sample.txt`). **Gerçekte kullanılan `--tok_train_mb`
  değeri doğrulanamadı** (varsayılanla yeniden oynatma, modelin 54 karakterlik alfabesini birebir veriyor: tutarlı, ama kanıt değil).

### 4.6 Byte fallback, bilinmeyen karakterler, Türkçe / Almanca

- **Byte fallback var:** 256 bayt parçası (id 3..258). Sözlükte olmayan her karakter UTF-8 baytlarına bölünür; `<unk>` üretilmez
  (sınanan 28 dizgide unk 0; sekiz `.bin`'in toplam 857 325 466 token'ında unk 0).
- **Alfabe** (tek karakterlik parçalar): `sp8k` 54 karakter — 94 yazdırılabilir ASCII karakterinin 41'i yok: rakamlar,
  `# $ % & ( ) * + - / : ; < = > @ [ \ ] ^ _ \` { | } ~` ve büyük `Q U V X Z` (yani kod / matematik noktalaması her zaman bayt
  parçası); `foton` 106 karakter — 94 ASCII'nin tamamı + ASCII dışı yalnız NBSP, £, §, é, –, —, ‘, ’, “, ”, ─.
- **Türkçe ve Almanca'ya özgü harflerin hiçbiri** (`çğıöşüÇĞİÖŞÜ`, `äöüßÄÖÜ`) iki sözlükte de parça değil → her biri 2 bayt token'ı.
  Neden: `foton` eğitim örneğinde ä/ö/ü/ß/Ä/Ö/Ü toplam 10 210 karakter (%0.00275; küçük harfler 9 908) ve `character_coverage = 0.9995`.

| örnek | `sp8k` | `foton` |
|---|---|---|
| `"çğıöşü ÇĞİÖŞÜ"` | 26 token (24'ü bayt parçası) | 26 token (24'ü bayt) |
| `"äöüß ÄÖÜ Straße Größe"` | 27 token (20'si bayt) | 26 token (20'si bayt) |
| TR: "Bugün hava çok güzel olduğu için parkta uzun bir yürüyüş yapmaya karar verdik." (78 karakter) | 58 token → 1.345 karakter / token | 51 token → 1.529 |
| DE: "Weil das Wetter heute sehr schön ist, haben wir beschlossen, einen langen Spaziergang im Park zu machen." (104 karakter) | 51 token → 2.039 | 42 token → 2.476 |
| EN: "Because the weather is very nice today, we decided to take a long walk in the park." (83 karakter) | 19 token → 4.368 | 19 token → 4.368 |

- **Kayıpsızlık:** `decode(encode(x)) == x` satır sonu (LF, CRLF), sekme, çoklu boşluk, girinti, baş / son boşlukta iki
  tokenizer'da da doğru. Tek kayıplı durum ham metindeki U+2581 (`▁`) karakteri (boşluğa döner).
- Satır sonu her zaman tek `<0x0A>` (id 13), sekme her zaman `<0x09>`; boşluk koşuları: `sp8k`'da 1 ve 2 boşlukluk parça,
  `foton`'da 1'den 16'ya kesintisiz. Satır başındaki sözcük `▁`'siz varyantla kodlanır (`foton`: `return` id 7147, satır içinde
  `▁return` id 1291; `sp8k`'da `return` parçası yok → `ret` + `urn`, `▁return` id 2470).

### 4.7 Üretimde kullanım

`src/sample.py:42` BOS ile başlar, `src/sample.py:46` `eos_id = tok.eos_id` ile `generate`'i çağırır; batch'in tamamı EOS üretince
ya da `max_new_tokens` dolunca durur (`src/model.py:540-545`). `data/foton` ile eğitilen modeller eğitimde ne BOS ne EOS gördü
(bölüm 5.3); bu modellerde durma koşulunun pratikte tetiklenmeyeceği koddan çıkarımdır, **koşturularak doğrulanmadı**.

Tokenizer sarmalayıcısı ve iki veri hazırlama betiği için **birim testi yok** (`tests/` altında `src.tokenizer`,
`prepare_data`, `prepare_foton_corpus` adları geçmiyor).

---

## 5. Veri hattı

### 5.1 Korpus kaynakları ve boyutları

**(a) TinyStories → `data/sp8k`** (dense / MoE deneyleri; tek alan)

| | |
|---|---|
| Kaynak | TinyStories V2 (yalnız GPT-4 üretimi); `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/` (`scripts/download_tinystories.py:15-16`) |
| Dosyalar | `TinyStoriesV2-GPT4-train.txt` 2 227 753 162 B; `TinyStoriesV2-GPT4-valid.txt` 22 502 601 B |
| Hikâye sayısı | train 2 717 495 (metinde 2 717 699 `<|endoftext|>`; 205 boş parça atılır, ayraçsız kuyruk +1); valid 27 630 |
| Token | train **545 813 247**, val **5 512 219** (`data/sp8k/meta.json`; bayt / 2 ile eşit) |
| Lisans kaydı | depoda YOK (kodda, `meta.json`'da, README'de) → **doğrulanamadı** |
| Bütünlük | indirme yalnız `Content-Length` ile denetlenir; `train.bin` / `val.bin` için hiçbir yerde hash yok |

Hazırlama (`scripts/prepare_data.py`): `<|endoftext|>` üzerinden bayt düzeyinde bölme (`27`, `37-45`), her parça
`decode(...).strip()` (`47`), boşlar atılır; uzunluk / dil / tekrar süzgeci yok.

**(b) FOTON korpusu → `data/foton`** (üç alan; `common-pile` v0.1'den dört shard; `scripts/prepare_foton_corpus.py:41-48`)

| kısa ad | alan | HF deposu / shard | diskteki `json.gz` | tutulan doküman (lisans sınıfı) | `.bin`'e giren pay |
|---|---|---|---:|---|---|
| caselaw | nl | `common-pile/caselaw_access_project_filtered` / `CAP-Dolma-0000.json.gz` | 170 000 001 B | 17 007 (Public Domain %100) | ilk 12 812 tam doküman + 1 kırpık (tutulan metnin bayt payı ≤ %72.67; doküman payı %75.33) |
| hansard | nl | `common-pile/uk_hansard_filtered` / `uk_hansard-0000.json.gz` | 170 000 001 B | 4 940 (Open Parliament Licence %100; sıralı: önce 4 500 Lordlar yazılı cevabı, sonra 440 İskoç Parlamentosu) | ilk 4 760 tam + 1 kırpık (bayt ≤ %82.99; doküman %96.36) |
| stackv2_edu | code | `common-pile/stackv2_edu_filtered` / `stack-edu-0000.json.gz` | 260 000 001 B | 171 121 (permissive SPDX %100; 974 lisanssız + 137 boş atıldı) | ilk 103 102 tam + 1 kırpık (bayt ≤ %60.29; doküman %60.25) |
| arxiv | math | `common-pile/arxiv_papers_filtered` / `arxiv-papers-0000.json.gz` | 240 000 001 B | 10 922 (CC BY 8 520 + PD / CC0 2 402; 626 BY-SA atıldı) | ilk 6 467 tam + 1 kırpık (bayt ≤ %55.04; doküman %59.21) |

- `…000 001` baytlık boyutlar **kasıtlı kısmi indirmedir**: `curl -r 0-N` (`scripts/prepare_foton_corpus.py:112`) shard'ın yalnız ilk
  N + 1 baytını alır; kırpılmış gzip sonu bilinçli yutulur (`141-142`). Bütünlük / hash denetimi yok.
- Lisans süzgeci `tier_a()` (`scripts/prepare_foton_corpus.py:54-81`): lisanssız, BY-SA, NC, ND, GFDL / GPL → ret; permissive, PD / CC0,
  Open Parliament / Open Government, CC BY → kabul. "Tutulan doküman" sütunu bu kuralın yazmadan yeniden oynatılmasıyla sayıldı (sonda);
  sayılar ham `.txt` dosyalarındaki doküman sayılarıyla birebir aynı.
- `data/foton/meta.json`'ın `licensing` bloğu **boş** (dört kaynakta `kept_docs: 0`, `reused: true`, `license_classes_bytes: {}`):
  son koşu önceki çıkarımı yeniden kullandığı için (`scripts/prepare_foton_corpus.py:119-122`). Tutulan alt kümenin lisans dökümü depoda
  kayıtlı değil; `experiments/foton-v02-prereg/corpus_report.json` kırpık shard'ın tamamını sayar.
- Alan başına token: her alan **100 000 000 train + 2 000 000 val** (toplam 306 000 000). Kota kırpmasının `.bin`'e soktuğu kırpık
  parçalar: caselaw'un 12 813. dokümanı `nl_train[50 935 731 : 51 000 000]` (64 269 token; ardından ayraçsız hansard), hansard 4 761.
  (`nl_val`'in son 7 960 token'ı), stackv2_edu 103 103. (`code_val`'in son 2 536 token'ı), arxiv 6 468. (`math_val`'in son 18 295 token'ı).
- İçerik hakkında ölçülen üç olgu (belgelerde yazmıyor):
  1. **"code" alanı %100 Markdown:** tutulan 171 121 dokümanın tamamında `metadata.language == "Markdown"` (README vb.); `.bin`'e giren
     kısmın baytlarının yalnız %16.52'si çitli kod bloğu içinde. Nedeni: indirilen `stack-edu-0000.json.gz` kayıtlarının
     `metadata.provenance` alanı 171 121 / 171 121'de `stack-edu-markdown-0000.json.gz` — yani shard, stack-edu'nun Markdown shard'ı.
  2. **NL val, NL train'i temsil etmiyor:** `nl_val.bin`'in tamamı İskoç Parlamentosu tutanağı (34 doküman parçası); `nl_train` %51.00
     caselaw + %35.58 Lordlar yazılı cevapları + %13.42 İskoç Parlamentosu. caselaw val'de hiç yok.
  3. **"math" alanı arXiv tam metinleridir** (kaynak adı; metadata'da konu / kategori yok — konu dağılımı doğrulanamadı).

### 5.2 `.bin` biçimi

- **`uint16`, başlıksız ham dizi**, yerel bayt sırası (little-endian; açıkça belirtilmemiş). Yazma: `np.asarray(..., dtype=np.uint16).tofile`
  (`scripts/prepare_data.py:73-74`; `scripts/prepare_foton_corpus.py:227`, `232`). Okuma: `np.memmap(path, dtype=np.uint16, mode="r")`
  (`src/data.py:24`; `src/data_multi.py:33`). Sözlük < 65 536 olmak zorunda; taşma denetimi yalnız `scripts/prepare_data.py:59`'da.
- Sekiz dosyanın hepsinde token = bayt / 2, `meta.json` ile eşit, sözlük dışı id 0:

| dosya | bayt | token | min id | maks id | çivi |
|---|---:|---:|---:|---:|---|
| `data/sp8k/train.bin` | 1 091 626 494 | 545 813 247 | 1 | 8 191 | yok |
| `data/sp8k/val.bin` | 11 024 438 | 5 512 219 | 1 | 8 191 | yok |
| `data/foton/nl_train.bin` | 200 000 000 | 100 000 000 | 12 | 32 767 | `pins.json` ile eşit |
| `data/foton/nl_val.bin` | 4 000 000 | 2 000 000 | 13 | 32 767 | eşit |
| `data/foton/code_train.bin` | 200 000 000 | 100 000 000 | 4 | 32 767 | eşit |
| `data/foton/code_val.bin` | 4 000 000 | 2 000 000 | 12 | 32 767 | eşit |
| `data/foton/math_train.bin` | 200 000 000 | 100 000 000 | 4 | 32 767 | eşit |
| `data/foton/math_val.bin` | 4 000 000 | 2 000 000 | 13 | 32 767 | eşit |

`data/foton` altındaki 9 çivili dosyanın (6 `.bin` + `meta.json` + `tokenizer.model` + `tokenizer.vocab`) bayt ve SHA-256'sı
`experiments/foton-v02-prereg/pins.json` ile 9 / 9 eşleşiyor (sonda; `verify_pins.py` çalıştırılmadı).

- `data/sp8k/meta.json`: `tokenizer: sentencepiece`, `model_file: tokenizer.model`, `vocab_size: 8192`, `bos_id: 1`, `eos_id: 2`,
  `train_tokens: 545813247`, `val_tokens: 5512219`.
- `data/foton/meta.json`: aynı ilk beş alan (`vocab_size: 32768`), `tier: "A-only"`, `domains` (nl / code / math: `train_tokens`
  100000000, `val_tokens` 2000000, `bytes_per_token` 4.264 / 3.272 / 3.033, `sources`), `licensing` (boş sayımlar), `notes`.
  Eğitim kodu çok alanlı yola bu dosyadaki `"domains"` anahtarının varlığıyla geçer (`src/train.py:397-409`).

### 5.3 EOS / doküman sınırı — gerçek dosyalarda

Sekiz `.bin` dosyasının tamamı tarandı (`np.memmap(mode="r")`, parça parça `bincount`):

| dosya | token | unk (id 0) | BOS (id 1) | EOS (id 2) | `<0x1E>` (id 33) |
|---|---:|---:|---:|---:|---:|
| `data/sp8k/train.bin` | 545 813 247 | 0 | **2 717 495** | **2 717 495** | 0 |
| `data/sp8k/val.bin` | 5 512 219 | 0 | **27 630** | **27 630** | 0 |
| `data/foton/nl_train.bin` | 100 000 000 | 0 | **0** | **0** | 17 539 |
| `data/foton/nl_val.bin` | 2 000 000 | 0 | **0** | **0** | 33 |
| `data/foton/code_train.bin` | 100 000 000 | 0 | **0** | **0** | 101 140 |
| `data/foton/code_val.bin` | 2 000 000 | 0 | **0** | **0** | 1 962 |
| `data/foton/math_train.bin` | 100 000 000 | 0 | **0** | **0** | 6 354 |
| `data/foton/math_val.bin` | 2 000 000 | 0 | **0** | **0** | 113 |

**`data/sp8k` — sınır var ve doğru.** Her hikâye `[BOS] … [EOS]` (`scripts/prepare_data.py:68-72`). BOS = EOS = boş olmayan hikâye
sayısı (train 2 717 495, val 27 630); ilk token BOS, son token EOS; EOS→BOS bitişik çift sayısı hikâye − 1; EOS'tan sonra BOS
olmayan durum 0. `<|endoftext|>` metni ayraç olarak tüketilip atılır; token dizisi olarak `.bin`'de 0 kez geçer. Kusurlar:
`train.txt` / `valid.txt` dosya sınırında bir hikâye ikiye bölünmüş (train `…"Yo` ile bitiyor, val `u don't have to be scared…`
ile başlıyor; ikisi ayrı "hikâye" olarak girmiş); 3 token'lık 3 dejenere hikâye; train içinde 1 birebir kopya (o da 3 token'lık
`[BOS] ▁Once [EOS]` dejenere hikâyesinin iki kez geçmesi; gerçek hikâyeler arasında kopya yok). Köken notu: `train.bin` / `val.bin` /
`tokenizer.model` (2026-09-15 00:09–00:10) `scripts/prepare_data.py`'nin ilk commit'inden (`f0f8494`, 00:18) önce üretildi ve
betik `2b13e7d`'de (00:36) değişti (`.bin` yazma mantığı aynı).

**`data/foton` — EOS ve BOS HİÇ YOK** (306 000 000 token'da id 0 / 1 / 2 sayısı 0). `data/foton/meta.json` yine de `bos_id: 1`,
`eos_id: 2` yazıyor.

- **Kök neden (tek satır):** `scripts/prepare_foton_corpus.py:51` `DOC_SEP = "\x1e"`; `scripts/prepare_foton_corpus.py:90`
  `if line.strip() == DOC_SEP:`. Python'da `"\x1e".isspace()` True'dur; `"\x1e\n".strip()` boş dizgi verir ve koşul hiçbir zaman
  doğru olmaz. `batched_docs` (`84-102`) bu yüzden her ham dosyayı TEK doküman olarak döndürür; `e + [tok.eos_id]`
  (`227`) kaynak başına tek EOS ekler ve o da kota kırpmasında (`230-231`) kesilir.
- **Sınır kaybolmamış:** ayraç karakteri metnin parçası olarak kodlanmış; her doküman sınırı akışta
  `<0x0A> <0x1E> <0x0A>` = id **`[13, 33, 13]`** üçlüsü olarak duruyor. Altı dosyadaki 127 141 `<0x1E>`'nin tamamı bu desende;
  ham `.txt`'lerden rastgele seçilen dokümanlar akıştaki karşılıklarıyla token token birebir eşleşti (iki bağımsız örneklem:
  1 204 / 1 204 ve 240 / 240). Kök nedenin ikinci yüzü: tokenizer örneği ayraç satırlarını yalnızca aynı `strip()` davranışı
  sayesinde atlıyor (`scripts/prepare_foton_corpus.py:200`); `<0x1E>` bu yüzden tokenizer eğitiminde hiç görülmedi ve bayt parçası
  (id 33) olarak kodlanıyor.
- Teşhisin sınanması (sonda; betiğin mantığı yazmadan yeniden oynatıldı, dosyaya dokunulmadı): `stackv2_edu.txt`'nin ilk
  2 993 389 karakterinde özgün koşul 1 doküman / 1 EOS / 788 `<0x1E>` üretiyor; koşul `line.strip('\r\n') == DOC_SEP` olarak
  değerlendirildiğinde 788 doküman / 788 EOS / 0 `<0x1E>`.
- Doküman ortasından yapılan kesimler (EOS'tan bağımsız): kaynak kotası kırpması (`230-231`) — `nl_train[51 000 000]`'de bir
  caselaw cümlesi hansard başlığına ayraçsız yapışıyor; train / val bölmesi token düzeyinde (`243-244`); val dosyalarının sonu.
- Belgeler veriyle iki yönde çelişiyor: "doküman sınırları EOS ile işaretli" (`scripts/prepare_foton_corpus.py:9-10`,
  `src/data_multi.py:7-9`, `PREREGISTRATION_V02.md:76`) yanlış; "doküman sınırı yok / uzunluk geri çıkarılamaz"
  (`PREREGISTRATION_V02.md:2019-2024`, `docs/ARCHITECTURE_AS_BUILT.md:116`) — EOS yok kısmı doğru, sınırın geri çıkarılamadığı kısmı
  yanlış. Kök neden (`strip()` davranışı) depodaki hiçbir belgede yazmıyor.
- Diskteki `.bin`'leri üreten betik sürümünün bugünkü commit'li sürümle aynı olduğu **doğrulanamadı** (betik veriden ~22 saat
  sonra commit'lendi); `.bin` yapısı bugünkü kodun davranışıyla tam tutarlı.

Doküman uzunlukları (token; `foton` satırları train akışında ardışık iki ayraç arasındaki parçalar — her kaynağın ilk dokümanı ve
kırpık son parçası hariç; uzunluk = iki ayraç arasındaki token sayısı):

| akış | doküman | medyan | p90 | p99 | maks | 2 048'den uzun doküman | 4 096'dan uzun |
|---|---:|---:|---:|---:|---:|---:|---:|
| TinyStories train (BOS + EOS dahil) | 2 717 495 | 179 | 274 | 576 | 1 514 | %0.00 | %0.00 |
| caselaw (nl) | 12 811 | 2 847 | 8 142 | 20 577 | 127 955 | %63.28 | %33.85 |
| hansard (nl) | 4 726 | 7 088 | 17 886 | 80 102 | 147 668 | %84.41 | %70.19 |
| stackv2_edu (code) | 101 139 | 490 | 2 070 | 7 523 | 326 015 | %10.15 | %3.19 |
| arxiv (math) | 6 353 | 11 705 | 29 461 | 73 795 | 357 535 | %98.61 | %94.05 |

### 5.4 Train / val ayrımı

| | `data/sp8k` | `data/foton` |
|---|---|---|
| Nasıl | ayrı yukarı akış dosyaları (`…-train.txt` → `train.bin`, `…-valid.txt` → `val.bin`) | alan akışının **son 2 000 000 token'ı** val (`scripts/prepare_foton_corpus.py:241-244`) |
| Düzey | hikâye düzeyi (dosya sınırındaki tek hikâye hariç) | **token düzeyi**; sınırdaki doküman ikiye bölünür |
| Birebir kopya (ölçüldü) | val hikâyelerinden train'de geçen 0 / 27 630 | val'e tam giren dokümanlardan train'de kopyası olan: hansard 0 / 32, stackv2_edu 0 / 1 961, arxiv 0 / 112 |
| Tokenizer sızıntısı | yok (tokenizer yalnız train metninde) | yok (örnek her dosyanın başından; val bölgeleriyle çakışmıyor) |

Yakın-kopya (şablon hikâyeler, README kalıpları) benzerliği ölçülmedi → **doğrulanamadı**.

### 5.5 Packing ve pencere mantığı

"Packing" yok: her `.bin` dokümanların uç uca eklendiği düz bir token akışı. `DataLoader` / worker kullanılmıyor; batch ana iş
parçacığında kurulur.

- **Tek alanlı yol** — `TokenDataset` (`src/data.py:22-47`): yerine koyarak rastgele ofset
  `torch.randint(len(self.data) - T, (batch_size,), generator=generator)` (`src/data.py:37`); pencere `T + 1` token
  (`src/data.py:38`); `x = xy[:, :-1]`, `y = xy[:, 1:]` (`src/data.py:47`); GPU'ya `pin_memory().to(device, non_blocking=True)`
  (`src/data.py:44`). `T = mcfg.max_seq_len` (`src/train.py:408-409`). Epoch kavramı yok.
- **Çok alanlı yol** — `MultiDomainDataset` (`src/data_multi.py:23-147`): alan başına ayrı memmap; alan kimliği `sorted` sırası
  (code = 0, math = 1, nl = 2); dizi başına bağımsız `torch.multinomial` ile alan (`src/data_multi.py:98`), sonra alan içinde
  `randint` ofset (`104-105`); bir pencere alan sınırını geçemez. Karışım 1/3 : 1/3 : 1/3. Eğitimde adım başına
  `batch_size × grad_accum = 64` dizi (`src/train.py:631-632`); `sequence_stream` 16'lık (`REF_BATCH = 16`,
  `src/data_multi.py:93`) parçalarla tüketir ve dizi sayısı 16'nın katı değilse `ValueError` verir (`116-117`). Alan etiketi modele
  verilmez (yalnız teşhis ve zorlanmış atama kolları için).
- **Üreteç:** `train_gen = torch.Generator().manual_seed(tcfg.seed)` (`src/train.py:426`); durumu checkpoint'e yazılır ve resume'da geri yüklenir.
- **Val pencereleri sabit:** tek alanlıda her eval'de `manual_seed(12345)` ile `eval_iters × batch_size` çekim (`src/train.py:306-311`);
  çok alanlıda `frozen_val_windows(total, 12345)` (`src/train.py:286-288`; `src/data_multi.py:133-135`) ve pencere içeriğinin
  SHA-256'sı `val_windows_sha256` olarak loglanır (`src/train.py:293-297`). İki yolda da **1 280 pencere**
  (dense / moe 32 × 40; foton 16 × 80; çok alanlıda ölçülen dağılım code 431 / math 434 / nl 415).
- **Pencereler doküman sınırını aşar; hizalama yok.** Padding yok, doküman için attention mask yok; `ignore_index=-1` tanımlı
  (`src/model.py:468`) ama yükleyiciler hiçbir zaman −1 üretmez. Bir pencerede iki doküman varsa ikincisi birincisine attend eder
  (`src/data.py:11-12` bunu bilinçli tercih olarak yazıyor).

En az bir sınır token'ı içeren pencere oranı · pencere başına ortalama sınır (ölçüldü):

| akış | T = 512 | T = 2 048 | T = 4 096 |
|---|---|---|---|
| `sp8k` train (EOS) | %98.99 · 2.55 | %100 · 10.20 | %100 · 20.39 |
| `foton` nl_train (`<0x1E>`) | %8.71 · 0.09 | %30.56 · 0.36 | %50.49 · 0.72 |
| `foton` code_train | %38.53 · 0.52 | %73.38 · 2.07 | %85.19 · 4.14 |
| `foton` math_train | %3.25 · 0.03 | %12.93 · 0.13 | %25.52 · 0.26 |

Kapsama (türetme): dense / moe bütçesi 3 000 adım × 32 768 = 98 304 000 token = train'in %18.01'i; `foton-*` bütçesi
4 375 × 32 768 = 143 360 000 = 300 000 000'un %47.79'u.

---

## 6. Eğitim döngüsü

Kaynak: `src/train.py` (735 satır), `src/checkpoint.py` (42), `src/config.py` (`TrainConfig`, 28 alan, `75-138`). Ölçüm sayıları
kayıtlı koşulardan okundu (213 `config.json`, 213 `metrics.jsonl`, `queue.log` dosyaları); bu envanterde benchmark koşulmadı.

### 6.1 Optimizer

| ayar | değer | kanıt |
|---|---|---|
| Tür | `torch.optim.AdamW`, CUDA'da `fused=True` | `src/model.py:501-512`; çağrı `src/train.py:425` |
| lr (tepe) / min_lr | 1e-3 / 1e-4 | `src/config.py:100-101` |
| betas | (0.9, 0.95) | `src/config.py:105-106` |
| eps | 1e-8 — kodda verilmiyor, PyTorch varsayılanı | `src/model.py:512`; gerçek `last.pt` `param_groups` |
| weight_decay | 0.1, yalnız `dim >= 2` tensörlere | `src/config.py:104`; `src/model.py:505-509` |
| Parametre grupları | ayrım kuralı tensör BOYUTU: grup 0 (`dim >= 2`, decay 0.1), grup 1 (`dim < 2`, decay 0). Modelde bias olmadığı için grup 1 fiilen yalnız RMSNorm kazançları (dense: 17 tensör / 8 704 parametre). **Bağlı embedding decay grubunda.** | `src/model.py:505-510`; sonda |

20 preset'in hepsinde bu değerler aynı (üç tiny preset'te tek fark `warmup_steps = 10`); kayıtlı 213 koşunun 213'ünde
`lr=0.001, min_lr=0.0001, weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0`.

### 6.2 LR programı

- `lr_at` (`src/train.py:62-65`): **görülen token üzerinden** doğrusal warmup + kosinüs azalma, `min_lr`'da biter; bütçe aşılırsa
  `min_lr`'da sabit kalır. LR her optimizer adımından ÖNCE yazılır (`src/train.py:625-627`).
- `tokens_per_step = batch_size × grad_accum × max_seq_len` (`src/train.py:433`);
  `warmup_tokens = warmup_tokens or warmup_steps × tokens_per_step`, `max_tokens = max_tokens or max_steps × tokens_per_step`
  (`src/train.py:37-38`).
- `stop_step > 0` programı değiştirmez, yalnız erken durdurur (`src/train.py:443`); `budget_mult` koşuyu uzatır, LR ufku 1×'te
  kalır (`src/train.py:439-441`).

| preset ailesi | token / adım | warmup | bütçe |
|---|---:|---|---|
| dense, moe, moe-noaux, moe-auxfree | 32 768 | 200 adım = 6 553 600 token | 3 000 adım = 98 304 000 token |
| 13 `foton-*` preset'i | 32 768 | 200 adım = 6 553 600 token | 4 375 adım = 143 360 000 token |
| tiny, tiny-foton, tiny-moe | 2 048 | 10 adım = 20 480 token | 60 adım = 122 880 token |

LR noktaları (gerçek `lr_at` çağrılarak; adım indeksi s): s = 0 → 5e-6; s = 199 → 1e-3 (tepe); `foton-*`'da s = 2187 →
5.84e-4 (v03 adım 1 ve V7 koşuları `--stop_step 2188` ile kosinüsün ortasında durur; kayıtlı son `lr` değeriyle birebir).

### 6.3 Precision, derleme, determinizm, seed

| konu | durum | kanıt |
|---|---|---|
| autocast | **bfloat16**, yalnız CUDA'da; izinli değerler `"bfloat16"` / `"float32"` | `src/config.py:127`; `src/train.py:478-484` |
| GradScaler | yok | `src/train.py:473-477` |
| TF32 | `torch.set_float32_matmul_precision("high")` | `src/train.py:389` |
| float32 kalanlar | CE (`logits.float()`), RMSNorm'un kareler ortalaması, MoE router'ı, FOTON router'ı; ağırlıklar ve AdamW momentleri float32 (karma hassasiyetin ana kopyası) | `src/model.py:468`, `56-60`; `src/moe.py:228-229`; `src/model.py:316-318` |
| `torch.compile` | varsayılan kapalı; kayıtlı hiçbir koşuda açılmamış | `src/train.py:485`; `src/config.py:128` |
| Deterministik ayar | yok (`use_deterministic_algorithms`, `cudnn.deterministic` `src/` içinde geçmiyor) | Grep |
| VRAM sınırı | `set_per_process_memory_fraction(0.92)` | `src/train.py:390-391`; `src/config.py:138` |
| Seed | `torch.manual_seed(seed)` (init) + ayrı CPU üreteci `manual_seed(seed)` (veri sırası): aynı `--seed` ikisini birlikte belirler; varsayılan 1337; eval pencereleri sabit 12345 | `src/train.py:386`, `426`, `287`, `306` |

GPU'da aynı kod + aynı seed iki koşu bit bit aynı değil: `experiments/foton-v01/microbatch_check/summary.json` (100 adım)
tekrar zarfı train loss'ta 0.003168, val loss'ta 0.006937.

### 6.4 Mikro-batch / gradient accumulation

Ayrı bir `micro_batch` alanı yok: `batch_size` GPU'ya tek seferde giren mikro-batch'tir (`src/config.py:85`), `grad_accum`
birikim sayısı (`src/config.py:86`).

| preset ailesi | `batch_size` × `grad_accum` × `max_seq_len` | dizi / adım | token / adım |
|---|---|---:|---:|
| dense, moe, moe-noaux, moe-auxfree | 32 × 2 × 512 | 64 | 32 768 |
| 13 `foton-*` preset'i | 16 × 4 × 512 | 64 | 32 768 |
| tiny, tiny-foton, tiny-moe | 16 × 1 × 128 | 16 | 2 048 |

Bir optimizer adımı (`src/train.py:625-677`): LR yaz → her mikro-batch için autocast altında forward ve
`(loss / grad_accum).backward()` (`656`) → `clip_grad_norm_` (`669`) → `optimizer.step()` (`672`) →
`zero_grad(set_to_none=True)` (`673`) → `update_expert_biases(model)` (`675`). Aux / z loss'ları döngüde değil modelin
`forward`'ında toplam loss'a katılır (`src/model.py:477-479`). 32 × 2 ile 16 × 4'ün eşdeğerliği ölçülmüş:
`experiments/foton-v01/microbatch_check/summary.json` (fark, aynı kodun tekrar zarfının 0.62× ve 0.63×'i).

### 6.5 Gradient clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)` (`src/train.py:669`), `grad_clip = 1.0`
(`src/config.py:107`); bütün preset'lerde ve 213 / 213 koşuda 1.0. Loglanan `grad_norm` kırpma ÖNCESİ normdur ve yalnız log
aralığının son adımına aittir (`src/train.py:689`); kırpılan adım sayısı kayıtlardan çıkarılamaz.

### 6.6 Checkpoint kaydetme ve devam ettirme

- **Ne zaman:** yalnız değerlendirme anında — `step % eval_interval == 0 or step >= end_step` (`src/train.py:558`),
  `eval_interval = 250` (`src/config.py:109`). Ayrı bir checkpoint aralığı ve adım numaralı geçmiş yok; iki dosya tutulur.
- **`best.pt`** (val iyileşince; adım 0 dahil): `model`, `model_cfg`, `train_cfg`, `step`, `val_loss` (`src/train.py:609-614`).
- **`last.pt`** (`step > 0` olan her eval'de): + `optimizer`, `best_val`, `train_gen_state`, `train_time`; `foton_assign="iid"`
  kolunda + `assign_gen_state` (`src/train.py:615-618`). Boyutu `best.pt`'nin 3.000 katı (model + AdamW'nin iki momenti); örnek
  `v03s1-d-s102`: 169 927 861 B / 509 791 597 B.
- **Atomik yazım var:** `save_atomic` (`src/checkpoint.py:20-35`) — `.tmp`'ye `torch.save` → `fsync` → `os.replace`; 5 denemeden
  sonra başarısızsa yalnız uyarı basar ve eğitim sürer. `config.json` ve `metrics.jsonl` atomik yazılmaz.
- `save_checkpoints=false` iki dosyayı da kapatır (`src/config.py:131`; `src/train.py:613`, `615`): kayıtlı 213 koşunun 101'inde
  false (12 eski config'te alan yok) → 213 koşu klasörünün yalnız 79'unda `.pt` çifti var.
- `state_dict`'te RoPE tabloları, FOTON maskeleri ve `bias_counts` yok (non-persistent); MoE `expert_bias` var.
- **`config.json`** eğitim betiği yazar (`src/train.py:497-498`); **`provenance.json`** ve **`DONE`** dosyasını kuyruk yazar
  (`scripts/run_queue.py:235-248` içerik, `409` + `429-430` yazım; `DONE` `435-436`); eğitimin kendi bitiş işareti
  `metrics.jsonl`'deki `{"type": "end"}` kaydıdır (`src/train.py:727-728`). `provenance.json` alanları (bugünkü kod ve 18 güncel
  dosya): `src_sha256`, `git_head`, `git_status`, `dirty`, `code_dirty`, `session`, `python`, `gpu`, `host`, `args`, `env`; eski 162
  dosya 8 alanlı (`dirty` / `code_dirty` / `session` yok).
- **Devam ettirme:** `--resume true` (`src/config.py:129`; `src/train.py:415`). Geri yüklenen: model ağırlıkları, optimizer
  durumu, veri üreteci, (iid) atama üreteci, `step`, `best_val`, `train_time` (`src/train.py:455-466`). LR durumdan değil
  formülden gelir. **Tek açık uyum denetimi token / adım eşitliğidir** (`src/train.py:450-454`); bütçe, program, `weight_decay`,
  `betas` komut satırından yeniden alınır (yani resume'da sessizce değiştirilebilir). `--resume true` ama `last.pt` yoksa koşu
  sessizce sıfırdan başlar (`src/train.py:415-422`); resume'da `metrics.jsonl` checkpoint adımından sonraki kayıtlar atılarak
  kırpılır (`502-507`) ve o adımın eval'i atlanır (`554`). Global torch RNG kaydedilmez.
- **Bit bit aynı devam: doğrulanamadı.** Hiçbir test ya da kayıt bunu iddia etmiyor (resume'u sınayan iki CPU testi var ama bit
  eşitliği değil: `tests/test_train.py:190-199` — `train_time` sürer, token / adım değişince ret; `tests/test_foton_assign.py:262-277`
  — resume edilen koşunun atama sayımları düz koşuyla aynı, son `val_loss` 1e-6 içinde); kayıtlı 213 koşunun 213'ünde
  `resume=false`; kuyruk yarım koşuyu resume etmez, klasörü kenara alıp baştan koşar (`scripts/run_queue.py:158-163`, `431`).
- **Eski checkpoint'ler bugünkü kodla yükleniyor** (sonda, CPU): 9 / 17 / 18 / 20 / 27 / 28 alanlı `model_cfg` taşıyan
  checkpoint'lerden örnekler `load_model` (`src/checkpoint.py:38-42`) ile yüklendi (diskteki dağılım: 9 ×2, 17 ×16, 18 ×36, 20 ×68,
  27 ×18, 28 ×18). Config alanı EKLEMEK güvenli; alan SİLMEK / yeniden adlandırmak `ModelConfig(**ckpt["model_cfg"])` çağrısını
  kırar (dataclass davranışı; ayrıca denenmedi).

### 6.7 Değerlendirme ve loglama

- Eval adım 0'da, her 250 adımda ve son adımda; **1 280 sabit pencere** (bölüm 5.5); ölçülen nicelik saf CE.
  `val_loss` ilk `loss_prefix_skip` pozisyonu atlar (dense / moe 0; `foton-*` preset'lerinde **64**), `val_loss_full` hepsini
  içerir (`src/train.py:565-568`). **Eğitim loss'u maskelenmez**; yani `foton-*` koşularında train `loss` ile `val_loss` aynı
  nicelik değildir. En iyi model ölçütü `val_loss < best_val` (`src/train.py:611-614`); erken durdurma yok.
- `metrics.jsonl` kayıt türleri (`src/train.py:510-533`, `567-592`, `608`, `681-728`): `meta` (FLOP, `params_total`,
  `params_active`, `tokens_per_step`, …), `val`, `sample`, `train` (`step, tokens, loss, ce, lr, grad_norm, tok_per_sec,
  peak_mem_gb, train_time, time` + çok alanlı korpusta `data_domain_counts` + MoE / router / atama alanları), `end`. `tok_per_sec` eval süresini içermez;
  `peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9` (aşağıdaki "GB" = 1e9 bayt).
- Kuyrukla koşan koşuların konsol çıktısı `train.log`'a, kuyruk özeti `queue.log`'a yazılır.

### 6.8 Ölçülmüş tok/s ve VRAM (kayıtlardan)

Hepsi RTX 4070 SUPER (sürücü 616.64), torch 2.14.0+cu130, bfloat16, `compile=false`, **bağlam 512** (T = 128'lik iki tiny duman
koşusu dışında her kayıt T = 512). 512'den uzun bağlamda hiçbir ölçüm yok (2K–4K için tok/s ve VRAM: **doğrulanamadı**).

| koşu grubu (`checkpoints/…`) | preset; V; B × accum | koşu | adım | tok/s (koşu geneli ort. [min..maks]) | tepe VRAM (GB; koşular arası ort.) | kuyruk süresi (dk) |
|---|---|---:|---:|---|---:|---|
| `v2/dense-tok-*` | dense; 8 192; 32 × 2 | 3 | 3 000 | 117 943.8 [117 716.5..118 117.6] | 6.300 | 14.4–14.5 |
| `v2/moe-tok-*` | moe (grouped); 8 192; 32 × 2 | 3 | 3 000 | 85 169.3 [84 664.2..85 445.6] | 8.541 | 20.1–20.4 |
| `v2-speed/speed-moe-loop-*` | moe (loop); 8 192; 32 × 2 | 3 | 300 | 82 695.9 [82 463.2..82 887.0] | 8.920 | 2.1 |
| `foton-v02/baseline-d-*` | foton-d; 32 768; 16 × 4 | 5 | 4 375 | 91 964.5 [91 833.0..92 026.9] | 6.478 | 27.0–27.1 |
| `foton-v02/baseline-m-*` | foton-m; 32 768; 16 × 4 | 5 | 4 375 | 61 280.7 [53 631.9..66 025.5] | 8.022 | 38.0–47.0 |
| `foton-v02-tarama/v7-x-s*` | foton-x (öğrenilmiş router); 32 768; 16 × 4 | 3 | 2 188 | 84 119.6 [83 771.6..84 765.8] | 6.801 | 15.9–16.1 |
| `foton-v02-tarama/v7-x-ent-s*` | foton-x-ent | 3 | 2 188 | 83 886.3 [83 451.6..84 470.4] | 6.801 | 16.0–16.1 |
| `foton-v02-tarama/v7-x-lb-s*` | foton-x-lb | 3 | 2 188 | 82 981.3 [82 861.1..83 185.0] | 6.801 | 16.2–16.3 |
| `foton-v03-step1/v03s1-d-*` | foton-d | 2 | 2 188 | 90 774.4 [90 350.9..91 197.9] | 6.477 | 13.8–14.0 |
| `foton-v03-step1/v03s1-x-dom-*` | foton-x-dom | 2 | 2 188 | 85 631.8 [85 169.9..86 093.8] | 6.782 | 15.6–15.8 |
| `foton-v03-step1/v03s1-h-dom-*` | foton-h-dom | 2 | 2 188 | 85 337.5 [84 877.8..85 797.2] | 6.782 | 15.7–15.9 |
| `foton-v03-step1/v03s1-x-iid-*` | foton-x-iid | 2 | 2 188 | 85 237.8 [84 896.2..85 579.3] | 6.779 | 16.8–16.9 |

- v1 koşuları (`checkpoints/dense`, `moe`, `moe-noaux`; eski kod): medyan 114 553.4 / 78 371.9 / 81 184.4 tok/s; tepe 6.300 /
  8.924 / 8.911 GB.
- `foton-m` hızı koşudan koşuya çok oynuyor (`baseline-m` 53.6k–66.0k; `foton-v02r/r-m-*` dahil 10 koşu 53.6k–66.1k); nedeni
  kayıtlardan **doğrulanamadı**.
- **VRAM sondası** (`experiments/foton-v02/vram_probe.json`; gerçek eğitim kodu, 16 × 4, T 512, V 32 768): tepe `foton-d` 6.483,
  `foton-m` 8.034, `foton-h` 6.767, `foton-x` 6.807 GB; ayırıcının tuttuğu (`peak_reserved_gb`) `foton-m`'de 10.186 GB;
  kullanılabilir tavan 11.848 GB (12.878 × 0.92).
- **VRAM merdiveni** (`scripts/vram_ladder.py`; rastgele token, tek mikro-batch 16, V 32 768, T 512;
  `experiments/foton-v02-prereg/vram_ladder_mb16_32k.json`): 512 × 8 dense 5.768 GB / moe 7.122; 640 × 10 dense 7.271 / moe 9.754;
  768 × 12 dense 9.169 / moe 13.126; 896 × 14 dense 11.431 GB. Merdiven gerçek tepeyi az gösterir (aynı yapılandırmada gerçek koşu
  +0.71 GB dense, +0.90 GB MoE — `baseline-d` 6.478 ve `baseline-m` 8.022 tabanıyla; `vram_probe` tabanıyla MoE +0.91) ve VRAM sınırı koymaz (12.878 GB'ı aşan satırlar sistem belleğine taşan ölçümlerdir). Dosyalar
  mikro-batch / vocab / bağlam alanı taşımıyor; bu değerler `params` ve zamanlamadan türetildi.
- README "12 GB VRAM notları" tablosunun (`README.md:668-684`; mikro-batch 16 / 32 / 48 / 64 → 3.3 / 6.2 / 9.0 / 11.9 GB) ham
  kaydı yok (`scripts/benchmark.py` yalnız ekrana basar) → **doğrulanamadı**.
- Bellekte baskın kalem logit tensörüdür (`B × T × V`): eğitim döngüsündeki `_, loss, info = train_model(x, y)` ataması
  (`src/train.py:647`, `653`) önceki mikro-batch'in logit'ini canlı tutar; forward ayrıca `logits.float()` kopyası alır
  (`src/model.py:468`). Aritmetik (ölçüm değil): B = 16, V = 32 768'de bf16 logit T = 512 → 0.537 GB, T = 2 048 → 2.147 GB,
  T = 4 096 → 4.295 GB; float32 kopya bunun iki katı.

### 6.9 Komut satırı

`build_configs` (`src/config.py:300-315`) `--preset` + iki dataclass'ın her alanı için bir bayrak üretir: **57 argüman**
(1 + 28 + 28). `--vocab_size` kabul edilir ama `src.train`'de etkisizdir (`src/train.py:394`). Tipik başlatma: elle
`.\.venv\Scripts\python -m src.train --preset dense` (`README.md:85`); kuyrukla `scripts/run_queue.py <kuyruk.json>` — kuyruğun
kurduğu satır örneği (`checkpoints/foton-v03-step1/queue.log`): `--preset foton-x-dom --seed 102 --stop_step 2188 --out_dir … --run_name v03s1-x-dom-s102`.

### 6.10 Döngüdeki FOTON / MoE / çok-alan dalları

`src/train.py`'nin yaklaşık 330 satırı FOTON + çok alanlı korpusa, ~26 satırı MoE'ye özeldir; kalan ~380 satır mimariden bağımsız
çekirdek (elle sayım, yaklaşık). Satır aralıkları: oracle (`68-97`, `218-239`, `571-576`, `588-590`); zorlanmış atama
(`100-176`, `259-274`, `427-428`, `463-464`, `577-583`, `616`, `644-650`, `670-671`, `715-720`); öğrenilmiş router logu
(`321-377`, `534-535`, `584-587`, `663-667`, `704-711`); çok alanlı korpus (`285-304`, `395-405`, `553`, `569-570`, `628-640`,
`712-714`); MoE (`489-493`, `593-599`, `661-662`, `675`, `693-703`).

---

## 7. FOTON deneylerinden kalanlar

Sınıflar: **GEREKSİZ** (FOTON / belirli bir deneye özel), **YENİDEN KULLANILABİLİR** (olduğu gibi), **UYARLAMAYLA** (neyin
değişmesi gerektiği yazıldı). Sınıflandırma standart (FOTON'suz) bir dense model açısındandır.

### 7.1 Yönlendirme / modül kodu (`src/`)

| bileşen | yer | sınıf | gerekçe |
|---|---|---|---|
| Dikkat desenleri, maske üretimi, eşitleme aritmetiği | `src/attn_patterns.py` (tamamı, 377 satır) | GEREKSİZ | standart model tam nedensel; sabitler yalnız T = 512 için eşitlenmiş |
| `CausalSelfAttention`'ın maske dalı (`pattern`, `attn_mask`) | `src/model.py:94-102`, `124-130` | GEREKSİZ | yoğun bool maske hesap kazandırmıyor; bellek T² ile büyüyor |
| Modül kurulumu, prefix router, yönlendirme, dağıtım | `src/model.py:222-259`, `305-372` | GEREKSİZ | dizi düzeyi modül yönlendirmesi |
| `foton_eval_all`, `foton_connection_counts`, `verify_foton_masks` | `src/model.py:261-303`, `374-399` | GEREKSİZ | oracle ölçümü ve desen kilidi |
| `force_module` / `assign` argümanları, FOTON regularizer'ları | `src/model.py:431-432`, `443-455`, `482-498` | GEREKSİZ | yalnız FOTON kolları |
| 8 `foton*` config alanı ve 14 FOTON preset'i | `src/config.py:50-72`, `173-282` | GEREKSİZ | (`foton-d`'nin TrainConfig'i çok alanlı korpusta sınanmış dense reçetesi olarak UYARLAMAYLA: `loss_prefix_skip = 64` → 0) |
| Oracle, zorlanmış atama, router / atama logları | `src/train.py:68-176`, `321-377` ve `evaluate` / `main` içindeki dallar (bölüm 6.10) | GEREKSİZ | 3 modül varsayar |
| `MultiDomainDataset` | `src/data_multi.py:23-147` | UYARLAMAYLA | çok kaynaklı örnekleyici olarak kullanılabilir; `REF_BATCH = 16` kısıtı (`93`, `116-117`) |
| `val_windows_sha256`, `data_domain_counts` parmak izleri | `src/train.py:290-304`, `553`, `631-636`, `712-714` | YENİDEN KULLANILABİLİR | modele dokunmayan veri kökeni kaydı |
| Token tabanlı LR / bütçe, `stop_step`, `budget_mult` | `src/train.py:35-65`, `436-443` | YENİDEN KULLANILABİLİR | FOTON'a özgü değil |
| `MoE` sınıfı, `load_balancing_loss`, `router_z_loss`, aux-free bias, `"loop"` dispatch, denge ölçüleri | `src/moe.py:55-123`, `143-146`, `181-305` | YENİDEN KULLANILABİLİR (MoE istenirse) | attention değişikliklerinden ve T'den bağımsız |
| `"grouped"` dispatch | `src/moe.py:126-140`, `149-178`, `312-341` | UYARLAMAYLA | özel API `torch._grouped_mm`'e ve torch sürümüne özgü hizalama kuralına bağlı |
| Ortak çekirdek: `RMSNorm`, RoPE, `SwiGLU`, `Block` iskeleti, `_init_weights`, `configure_optimizer`, `generate`, `num_params` | `src/model.py:47-190`, `401-429`, `501-546` | YENİDEN KULLANILABİLİR | standart modelin de kullandığı kod |
| `src/data.py`, `src/checkpoint.py`, `src/tokenizer.py`, `src/sample.py` | tamamı | YENİDEN KULLANILABİLİR | `foton` / `attn_patterns` / `data_multi` adı hiç geçmiyor (Grep: 0) |

"dense" preset'i FOTON kodunu model yolunda atlar (kurulan modelde `foton_modules`, `foton_router`, `attn_mask` yok; `info`
yalnız `ce`), ama şu artıklar dense'te de çalışır / yüklenir: `from src import attn_patterns` her import'ta (`src/model.py:37`,
`src/flops.py:37`); `FOTON_*` sabitleri (`src/model.py:41-44`), `if cfg.foton` dalları (`198-201`, `219-220`), `num_params`
FOTON dalı (`426-428`); `forward` imzasındaki FOTON argümanları ve guard'ı (`src/model.py:454-455`); `evaluate()`'teki FOTON
hazırlıkları (`src/train.py:205-216`) ve `main`'deki `None` atamaları (`src/train.py:427-428`, `517`, `534-538`); FLOP
sözlüğündeki `foton_*` anahtarları (`src/flops.py:185-186`); `foton-d` / `foton-m` preset'lerinde `loss_prefix_skip = 64`; her adım
`update_expert_biases` çağrısı (`src/train.py:675`).

### 7.2 Betikler (`scripts/`, 51 dosya)

**YENİDEN KULLANILABİLİR — olduğu gibi (5)**

| betik (satır) | amaç | not |
|---|---|---|
| `run_queue.py` (461) | kuyruk JSON'undaki koşuları tek GPU'da sırayla çalıştırır; `provenance.json`, `train.log`, `DONE` yazar; "sıkı" kipte commit'siz kodla başlamaz | preset'e kör. Yarım koşuyu `--resume` ETMEZ, kenara alıp baştan koşar. Olduğu gibi kullanılırsa her koşuyu FOTON defterine yazar (`scripts/run_ledger.py:51`) |
| `backup_bundle.py` (297) | bütün git geçmişini doğrulanmış tek bundle'a yedekler | varsayılan hedef `D:\foton gama v1 yedek\moe-llm.bundle` (`43`); `data/` ve `checkpoints/` bundle'a girmez |
| `benchmark.py` (67) | bir preset için mikro-batch boyutlarında tepe VRAM + tok/s (rastgele token) | **GPU ister**; yalnız ekrana basar; gerçek eğitim tepesini az gösterir |
| `plot_metrics.py` (108) | `metrics.jsonl` grafikleri | varsayılan çıktı `checkpoints/compare.png` |
| `download_tinystories.py` (69) | TinyStories indirme (kaldığı yerden devam) | ağ ister |

**UYARLAMAYLA (14)**

| betik (satır) | amaç | ne değişmeli |
|---|---|---|
| `run_ledger.py` (631) | koşu denemeleri için ekleme-yalnızca JSONL defter | `LEDGER` sabiti (`51`: `experiments/foton-v02-prereg/run_ledger.jsonl`) yeni kampanya klasörüne alınmalı |
| `make_campaign_queue.py` (205) | kol × seed'den blok + Latin kare sıralı kuyruk JSON'u | `foton-` öneki, S1 dalı, varsayılan seed / `out_dir` (`35-51`, `150-152`, `168-176`) |
| `verify_pins.py` (462) | git'te olmayan dosyaların SHA-256 manifesti (`pins.json`) | yeni korpus / tokenizer için YENİ manifest ve sabit küme (`84-128`); mevcut manifest tag'li |
| `preset_hash.py` (98) | 8 kampanya preset'inin kanonik-JSON özeti | `CAMPAIGN`, `VOCAB = 32768`, `OUT` (`31-33`). Çıktı dosyası diskte yok: `--write` hiç koşulmamış |
| `mutation_guards.py` (347) | mutasyon sınavı yürütücüsü (kopyala → boz → guard'ı koş → kaydet) | guard listeleri, suite bağları, katalog (`50-67`, `218-245`) |
| `v03_dcheck.py` (299) | "bugünkü kod dondurulmuş dense referansını aynı seed'de yeniden üretiyor mu" teşhisi | yollar, seed, eşikler, çapalar (`67-84`); kavram regresyon teşhisi olarak taşınabilir |
| `architecture_as_built.py` (178) | dört kolu CPU'da kurup parametre / FLOP dökümü basar | `ARMS`, `VOCAB` (`25-26`); `qkv`'nin tek matris olduğu varsayımı (`43`); `docs/ARCHITECTURE_AS_BUILT.md` ve testi birlikte değişir |
| `eval_domains.py` (88) | checkpoint'i alan başına değerlendirir | `--prefix_skip` varsayılanı 64; yalnız çok alanlı `meta.json` ile çalışır |
| `vram_probe_foton.py` (125) | gerçek eğitim koduyla tepe VRAM (eğitim + eval); **GPU ister** | yalnız `MultiDomainDataset` (`50-51`), `PRESETS["foton-d"]` (`105`), varsayılan preset listesi ve çıktı yolu |
| `vram_ladder.py` (126) | model boyutu merdiveni (tepe VRAM, tok/s); **GPU ister** | dizi uzunluğu bayrağı yok; head_dim 64 ve 2.75·d sabit (`66-69`); FOTON eki (`108-112`); bellek sınırı yok |
| `flops_report.py` (127) | analitik FLOP tablosu + profiler çapraz kontrolü | `RUNS = ["dense", "moe", "moe-noaux"]` (`27`); önce `src/flops.py` güncellenmeli |
| `corpus_report.py` (181) | korpus lisans / alan raporu + doğrulama (token aralığı, round-trip) | `SOURCES` / `tier_a` bağı, çıktı yolu; EOS / doküman sınırı denetimi YOK |
| `prepare_data.py` (142) | tek korpus → `[BOS] hikâye [EOS]` `.bin` + tokenizer eğitimi | depoda BOS + EOS'u doğru yazan tek betik (`68-72`); tokenizer seçenekleri, kaynaklar, uint16 sınırı (`59`) |
| `prepare_foton_corpus.py` (260) | çok alanlı, lisans süzgeçli korpus: indir → süz → tokenizer → alan başına `.bin` | lisans süzgeci ve alan-homojen akış taşınabilir; doküman ayracı hiç eşleşmiyor (`90`), val = akışın kuyruğu, kota kırpması (bölüm 5.3) |

**GEREKSİZ — FOTON / deneye özel (32)**

| grup | betikler (satır) |
|---|---|
| v02 ön-kayıt denetimi ve kalibrasyonu | `check_prereg_frozen.py` (77), `check_prereg_numbers.py` (296), `owner_decision.py` (140), `rule_reachability.py` (335), `calibrate_v02.py` (203), `baseline_reference.py` (193), `power_analysis.py` (288; v0.1) |
| Mutant katalogları (içerik) | `mutation_catalog_06.py` (204), `mutation_catalog_07.py` (69), `mutation_catalog_stats.py` (248), `mutation_catalog_v03s1.py` (466) |
| v03 adım 1 | `v03_step1_campaign.py` (142), `v03_step1_gate.py` (135), `v03_step1_decision.py` (171), `v03_step1_analyze.py` (452), `v03_step1_reachability.py` (155), `v03_step1_replay_streams.py` (75), `v03_step1_gpu_two_route.py` (94; CUDA şart) |
| FOTON router / topoloji teşhisi | `router_diag.py` (141), `router_probe_diag.py` (144), `router_grad_diag.py` (164), `router_health.py` (165), `oracle_ceiling.py` (274), `screening_context.py` (121), `topology_overlap.py` (89), `ox1_microbench.py` (528; varsayılan cuda, `--device cpu` seçeneği var) |
| MoE / v2 / v3 analizi | `analyze_experts.py` (110), `router_analysis.py` (489), `verify_dispatch.py` (329; GPU), `compare_runs.py` (312; karantinada), `compare_curves.py` (168), `compare_router.py` (325) |

- MoE tutulursa `analyze_experts.py`, `router_analysis.py`, `verify_dispatch.py` olduğu gibi kullanılabilir.
- `check_prereg_frozen.py`, `PREREGISTRATION_V02.md` ve `tests/test_prereg_frozen.py` durdukça kalmalı.
- 51 betiğin `from src.X import …` adlarının hepsi `src/` içinde tanımlı: **kırık içe aktarım 0** (statik çözümleme; betikler çalıştırılmadı).
- **Genel bir SAYISAL koşu karşılaştırma betiği yok** (grafik karşılaştırması `plot_metrics.py --runs …` ile var): `compare_curves.py`
  ve `compare_runs.py` v2'nin koşu adı desenine, `baseline_reference.py` FOTON'a sabit.
- Standart model için asgari yararlı set: `benchmark.py`, `vram_probe_foton.py` (uyarlanmış), `vram_ladder.py` (uyarlanmış),
  `plot_metrics.py`, `flops_report.py` + `src/flops.py` (güncellenmiş), `run_queue.py`, `run_ledger.py`, `backup_bundle.py`,
  `verify_pins.py`, `corpus_report.py`, veri hazırlama betikleri, `src/stats.py`'nin genel çekirdeği.
- Mutasyon sınavı: 4 katalog + satır içi liste toplam **418 mutant**; hedef metinlerinin hepsi bugünkü ağaçta bulunuyor (bayat
  mutant 0). 161 mutant `src/*.py` satırlarına METİN olarak çivili (`src/stats.py` 89, `src/train.py` 37, `src/model.py` 19,
  `src/flops.py` 7, `src/config.py` 6, `src/data_multi.py` 2, `src/attn_patterns.py` 1): bu dosyalar yeniden yazılınca suite'ler
  `AssertionError` ile düşer; pytest bunu görmez (mutasyon sınavını koşan test yok).

### 7.3 İstatistik ve FLOP kodu

**`src/stats.py`** (981 satır; modül düzeyinde yalnız standart kütüphane — `reachability` numpy varsa onu kullanır (`750`), yoksa
`random`'a düşer (`754-756`); scipy yok ve kurulu değil; `foton-v02-prereg` tag'indeki sürümle bayt bayt aynı): 19 public çağrılabilir.

| sınıf | adlar | not |
|---|---|---|
| GENEL (9) — olduğu gibi kullanılabilir | `t_crit` (72-77), `t_sf` (80-83), `keyed` (168-180), `paired` (225-246; seed kimliğine göre eşli fark, t, p, %95 GA), `n_star` (249-287; compute-to-L*), `ci95` (360-367), `t_crit_alpha` (526-541), `reachability` (738-759), `multiplicity` (762-790; Holm + BH) | yaklaşık 240 satır |
| FOTON'A ÖZEL (10) | `Obs` (122-148), `resolve_n_star` (290-323), `savings` (326-357), `indeterminate` (370-387), `ub95` (390-426), `abandon` (429-523), `contribution` (549-611), `decide_primary` (623-672), `boundary_diagnostic` (678-735), `decide` (837-981) | karar tablosu satır adları, eşikler (10 / 5 / 2 puan), seed kümeleri (11..55), kol adları (`M`, `D`, X / H), ölçekler (`S0`, `S1`) koda gömülü (`40-69`, `545-546`, `800-801`) |

- **Bootstrap GA bu modülde yok**; tek GA kaynağı t(n − 1) aralığı. Depodaki tek bootstrap FOTON'a özel `scripts/oracle_ceiling.py:99-104`'te.
- `resolve_n_star` ve `boundary_diagnostic` için `src/`, `scripts/`, `experiments/` altında çağrı yeri yok; `decide_primary` yalnız
  `boundary_diagnostic` içinden çağrılır (`src/stats.py:723`, `730`; o da çağrısız). Üçü de yalnız testler ve mutant kataloğunda:
  bağlayıcı karar yolu gerçek veride hiç koşmamış koddur.

**`src/flops.py`** (370 satır; yaklaşık 280'i FOTON'a özel): birincil tanım `forward_flops_per_token` (`154-187`), eğitim = 3 × ileri
(`245-247`). Gömülü mimari varsayımlar: tam MHA — `attn_qkv_proj = n_layer·2·4·d²` (`173`); tam nedensel bağlam
`2·d·(T + 1)` (`177`); SwiGLU üç matris (`179`); `lm_head = 2·d·V` (`184`); norm / RoPE / softmax birincil tanıma girmez.
Formül T ve V ile parametrik. `src/train.py:516-518` her koşunun `meta` kaydına bu modülden FLOP yazar: mimari değişip formül
güncellenmezse kayda hata vermeden yanlış FLOP yazılır.

### 7.4 v03 adım 1 kampanyasının şu anki durumu

**12 koşunun 8'i koştu (seed 101 ve 102 blokları); kampanya `gate_s102_exceeded` ile durdu (2026-09-21 03:23:31); seed 103'ün
4 koşusu başlatılmadı; analiz koşulmadı, δ hesaplanmadı.**

| an | olay | kanıt |
|---|---|---|
| 2026-09-20 18:29 → 19:32 | seed 101 bloğu (X-dom, H-dom, X-iid, D) koştu | `checkpoints/foton-v03-step1/queue.log` |
| 19:32:31 | seed 101 kapısı aşıldı (kod 3): ön koşul bacağı temiz; D akla-yatkınlık bacağında adım 2000 sapması +0.012132 nat, sınır 0.012 | `gate_s101.json` |
| 21:48 → 22:03 | dense teşhis koşusu `dcheck-d-s11`: sonuç `kod_saglam` | `experiments/foton-v03-step1/dcheck_result.json` |
| 2026-09-21 02:21 | commit `f8dcf35`; sürücü `--continue-after-s101` ile başladı; seed 101 yeniden denetimi geçti | `campaign.log`, `gate_s101_recheck.json` |
| 02:21 → 03:23 | seed 102 bloğu (H-dom, X-iid, D, X-dom) koştu | `queue.log` |
| 03:23:31 | **seed 102 ön koşul kapısı aşıldı (kod 3):** `mean_fixed_diff_nat` = 0.015534519, sınır 0.015; `ceiling_diff_points` = −0.163652 (sınır 0.4 içinde) | `gate_s102.json` |

- Diskte var: `v03s1-{x-dom, h-dom, x-iid, d}-s101` ve `-s102` (8 klasör, hepsinde `DONE`). Yok: `v03s1-*-s103`, `gate_s103.json`,
  `experiments/foton-v03-step1/result.json`.
- **Bu sonuç depoda kayıtlı değil:** `gate_s102.json`, `gate_s101_recheck.json`, `campaign.log`, `campaign_status.json` yalnız
  `.gitignore`'lu `checkpoints/` altında; commit'li belgeler (`decision_rule.md`, `audit_log.md`) "kalan 8 koşu onaylandı"da kalıyor.
- Koddan okunan iki sonuç (koşturulmadı): (1) sürücü `--continue-after-s101` ile yeniden başlatılırsa `campaign_status.json`,
  `gate_s101_recheck.json` ve `gate_s102.json`'ı başta siler (`scripts/v03_step1_campaign.py:108-111`), seed 102 kapısını aynı
  verilerle yeniden koşar ve seed 103'e geçemez; bayraksız başlatılırsa `campaign_status.json` ve `gate_s101.json`'ı siler, seed
  101 kapısını yeniden koşar ve `gate_exceeded` ile durur; (2) karar kuralı ön koşulun her seed'de geçmesini ister
  (`scripts/v03_step1_decision.py:112`), yani seed 103 koşsa da harita hücresi `"blocked"` olur.
- 8 koşunun sekizi aynı `src` özetiyle (`55703d68…`, bugünkü ağaçla aynı) ve `code_dirty = false` ile koştu.

### 7.5 `experiments/` klasörleri (hepsi git'te izleniyor)

| klasör | dosya | neyin kaydı | dondurma / çivi | bağlı tag |
|---|---:|---|---|---|
| `v1-dense-moe-baseline` | 11 | dense, moe, moe-noaux referans koşuları | `checksums.txt` (6 checkpoint + `data/sp8k/tokenizer.model`) | `v1-dense-moe-baseline` |
| `v2-grouped-dispatch` | 180 | gruplu dispatch doğrulaması + dense / MoE dört karşılaştırma | `checksums.txt` (18 `best.pt`) | kendi tag'i yok (`v3-router-ablation` ağacında) |
| `v3-router-ablation` | 106 | router dengeleme ablasyonu | `checksums.txt` (12 `best.pt`); `PREREGISTRATION.md` 6 SHA-256 çivisi | `v3-router-ablation` |
| `foton-v01-prereg` | 1 | v0.1 güç kalibrasyon günlüğü | — | `foton-v01-prereg`, `-r2` |
| `foton-v01` | 10 | VRAM merdivenleri + mikro-batch eşdeğerlik denetimi | — | ilk içeren tag `foton-v02-prereg` |
| `foton-v02-prereg` | 33 | v0.2 ön-kayıt paketi (`lstar.json`, `pins.json`, `corpus_report.json`, kuyruklar, **`run_ledger.jsonl`**) | `pins.json` (39 çivi; sonda ile 39 / 39 doğrulandı) | `foton-v02-prereg`; tag'den beri değişen tek dosya `run_ledger.jsonl` (tag'de 23 satır → HEAD 37 → disk 41) |
| `foton-v02` | 33 | tag sonrası v0.2 çalışması (`REPORT_07.md`, router teşhisleri, oracle tavanı, `vram_probe.json`, `v03_design_notes.md`) | — | tag yok |
| `foton-v03-step1` | 17 | v03 adım 1 kuralı, denetim / mutasyon günlükleri, kuyruklar, `gate_s101.json` | — | **tag yok** |

### 7.6 Standart modele geçişte çarpılacak dondurulmuş-deney kilitleri

`src/` altındaki İLK değişiklikte kırılanlar:

1. `scripts/v03_step1_gate.py:83-86` — koşuların `src_sha256`'sı bugünkü `src/` özetinden farklıysa bütünlük hatası.
2. `tests/test_v03_step1_gate.py:471-483` — `checkpoints/foton-v03-step1/v03s1-*-s101` diskte olduğu sürece GERÇEK kapıyı koşar;
   `src/` altında tek bayt değişirse kırılır (dört seed-101 klasörü yoksa atlanır; test ayrıca `checkpoints/foton-v02-tarama/v7-x-s101`'i
   okur ama `skipif` ona bakmaz — o klasör yoksa test atlanmaz, düşer).
3. `scripts/v03_step1_analyze.py:363-364` — 12 karar koşusunda tek `src` özeti şartı (seed 103 sonradan koşulamaz).
4. `tests/test_foton_model.py:409-435` — `foton == ""` modelinin `git show foton-v02-prereg:src/model.py` ile aynı seed'de bit
   düzeyinde aynı `state_dict` ve loss'u vermesi şartı (yeni alanların varsayılanı init sırasını / RNG tüketimini korumazsa kırılır).
5. Değere çivili sayılar: `scripts/v03_step1_analyze.py:55-64`; `tests/test_v03_step1.py:652-669`; `tests/test_flops.py:207-214`;
   `scripts/check_prereg_numbers.py:125-128`; `tests/test_architecture_doc.py` ↔ `docs/ARCHITECTURE_AS_BUILT.md`;
   `tests/test_foton_frozen.py:97-167`. `ModelConfig()` varsayılanları aynı zamanda `dense` ve `foton-d` preset'leridir
   (`src/config.py:150`, `178-179`): varsayılan değişirse bu gruplar kırılır.
6. 161 metin-çivili mutant (bölüm 7.2).

`scripts/preset_hash.py`'nin koruması fiilen yürürlükte değil: `experiments/foton-v02/preset_hashes.json` hiç yazılmamış,
`PREREGISTRATION_V02.md:1606` yer tutucusu boş; config'e alan eklemek özetleri değiştirir ama otomatik hiçbir şey kırılmaz.

---

## 8. Testler

### 8.1 Envanter

| nicelik | değer |
|---|---|
| Test dosyası | 36 (`tests/test_*.py`, 10 271 satır) |
| Test fonksiyonu (`def test_`) | **558** |
| Toplanan test öğesi (parametrize açılımıyla) | **988** (koşumla doğrulandı: 545 + 224 + 194 + 25) |
| `conftest.py`, `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tests/__init__.py` | hiçbiri yok |
| pytest eklentisi | yok (`pytest-timeout`, `xdist` kurulu değil) |
| `mark.slow` / `xfail` / `mark.skip` | 0 / 0 / 0; `skipif` 3 tanım, satır içi `pytest.skip` 6 (bu makinede hepsinin koşulu yanlış → skip yok) |
| CUDA gerektiren / kullanan test | **0** (her yerde `device="cpu"` ya da `torch.cuda.is_available` → False yaması) |
| `src.train.main()`'i uçtan uca koşturan dosya | 4 (`test_train`, `test_router`, `test_foton_train`, `test_foton_assign`): 29 çağrı, toplam 347 optimizer adımı, hepsi CPU + tiny preset + sentetik veri + sahte tokenizer |
| Proje kökü altına yazan test | **0** (hepsi `tmp_path` / `tempfile`; koşum sonrası ayrıca doğrulandı) |

| dosya (satır) | sınadığı | fonksiyon | öğe |
|---|---|---:|---:|
| `test_architecture_doc.py` (145) | `scripts/architecture_as_built.py` ↔ `docs/ARCHITECTURE_AS_BUILT.md` | 7 | 7 |
| `test_attn_patterns.py` (285) | `src/attn_patterns.py` | 21 | 118 |
| `test_backup_bundle.py` (422) | `scripts/backup_bundle.py` (geçici git depolarında) | 22 | 25 |
| `test_campaign_queue.py` (178) | `scripts/make_campaign_queue.py` | 12 | 12 |
| `test_compare_runs_quarantine.py` (73) | `scripts/compare_runs.py` karantinası | 5 | 9 |
| `test_data_multi.py` (108) | `src/data_multi.py` | 7 | 7 |
| `test_flops.py` (663) | `src/flops.py` (+ `docs/flops.md` ve ön-kayıt sayıları) | 30 | 76 |
| `test_foton_assign.py` (327) | zorlanmış atama (`src/model.py`, `src/train.py`) | 16 | 18 |
| `test_foton_frozen.py` (183) | `evaluate` prefix maskesi, preset çivileri, `REF_BATCH` | 9 | 14 |
| `test_foton_model.py` (435) | FOTON kolları, maskeler, router; tag'li kodla bit özdeşliği | 19 | 23 |
| `test_foton_train.py` (242) | `oracle_stats`, `RouterLog`, `main` | 10 | 10 |
| `test_model.py` (209) | `GPT`, RoPE, init, `generate`, `TokenDataset` | 12 | 16 |
| `test_moe.py` (380) | `src/moe.py` (dispatch, grouped ↔ loop, aux / z loss) | 18 | 52 |
| `test_oracle_ceiling.py` (153) | `scripts/oracle_ceiling.py` | 8 | 8 |
| `test_owner_decision.py` (135) | `scripts/owner_decision.py` | 8 | 8 |
| `test_ox1_microbench.py` (662) | `scripts/ox1_microbench.py` | 45 | 65 |
| `test_prereg_frozen.py` (57) | `PREREGISTRATION_V02.md` ↔ tag | 4 | 7 |
| `test_preset_hash.py` (57) | `scripts/preset_hash.py` | 4 | 10 |
| `test_router.py` (223) | sigmoid kapı, seçim bias'ı, `train.main` | 11 | 12 |
| `test_router_diag.py` (86) | `scripts/router_diag.py` | 5 | 5 |
| `test_router_grad_diag.py` (80) | `scripts/router_grad_diag.py` | 3 | 3 |
| `test_router_health.py` (124) | `scripts/router_health.py` | 8 | 8 |
| `test_router_metrics.py` (144) | `scripts/router_analysis.py` | 7 | 10 |
| `test_router_probe_diag.py` (35) | `scripts/router_probe_diag.py` | 2 | 2 |
| `test_router_stats.py` (186) | `scripts/router_analysis.py`, `compare_router.py` | 11 | 23 |
| `test_run_ledger.py` (474) | `scripts/run_ledger.py` | 22 | 22 |
| `test_run_queue.py` (696) | `scripts/run_queue.py` | 34 | 48 |
| `test_screening_context.py` (51) | `scripts/screening_context.py` | 3 | 3 |
| `test_stats.py` (253) | `src/stats.py` | 20 | 32 |
| `test_stats_v02.py` (1192) | `src/stats.py` (n_star, savings, decide, abandon, Holm) | 68 | 68 |
| `test_topology_overlap.py` (41) | `scripts/topology_overlap.py` | 2 | 4 |
| `test_train.py` (199) | `lr_at`, `token_budget`, `main` | 9 | 13 |
| `test_v03_dcheck.py` (306) | `scripts/v03_dcheck.py` | 20 | 41 |
| `test_v03_step1.py` (698) | `scripts/v03_step1_decision.py`, `v03_step1_analyze.py` | 33 | 137 |
| `test_v03_step1_gate.py` (483) | `scripts/v03_step1_gate.py`, `v03_step1_campaign.py` | 28 | 57 |
| `test_verify_pins.py` (286) | `scripts/verify_pins.py` | 15 | 15 |
| **toplam** | | **558** | **988** |

Çekirdek (FOTON'dan bağımsız) davranışı sınayanlar: `test_model`, `test_moe`, `test_router`, `test_router_metrics`,
`test_router_stats`, `test_train`, `test_flops` (ilk yarı), `test_data_multi`. Kalanların çoğu FOTON kampanyasını, ön-kaydı ve
süreç araçlarını (kuyruk, defter, kapı, çivi, yedek) sınar.

### 8.2 Çalıştırma sonucu (2026-10-06 00:56 – 00:59)

Paketin tamamı CPU'da, dört tur hâlinde koşturuldu; hiçbir test dışarıda bırakılmadı (GPU eğitimi başlatan ya da 5 dakikayı
aşan test yok). Her tur `timeout 300` ile sınırlandı.

```
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 \
  ./.venv/Scripts/python.exe -B -X utf8 -m pytest -p no:cacheprovider -q -rs <dosyalar>      (proje kökünden, Git Bash)
```

| tur | dosyalar | sonuç | süre |
|---|---|---|---:|
| 1 | 25 dosya (saf mantık / küçük tensör) | **545 passed**, 1 uyarı | 13.64 sn |
| 2 | `test_architecture_doc`, `test_flops`, `test_foton_assign`, `test_foton_model`, `test_foton_train`, `test_ox1_microbench`, `test_router`, `test_train` | **224 passed** | 65.83 sn |
| 3 | `test_v03_step1`, `test_v03_step1_gate` | **194 passed** | 26.64 sn |
| 4 | `test_backup_bundle` (yalnız git alt süreçleri; `GIT_OPTIONAL_LOCKS` verilmeden) | **25 passed** | 42.14 sn |
| **toplam** | 36 dosya | **988 passed, 0 failed, 0 skipped, 0 error** | **148.25 sn** |

- Tek uyarı: `tests/test_model.py::test_model_bf16_cast` — `src/moe.py:317` `UserWarning` ("grouped" yol bu dtype / boyutta
  desteklenmiyor, "loop" yoluna düşülüyor): beklenen düşüş yolu.
- En yavaş testler: `test_backup_bundle.py::test_many_distinct_ref_tips_do_not_break_later_runs` 6.54 sn;
  `test_foton_assign.py::test_every_arm_logs_a_content_digest_of_the_validation_windows` 6.08 sn.
- Koşum sonrası denetim: proje kökünde testlerin değiştirdiği / oluşturduğu dosya **yok** (`.pytest_cache` ve `__pycache__`
  dahil; `checkpoints/foton-v03-step1/` altındaki 6 kampanya kaydının SHA-256'sı öncekiyle aynı); `git status` yalnız baştaki
  defter değişikliğini ve bu belgeyi gösteriyor.
- Geçen iki çivi testi özellikle not edilmeli: `tests/test_v03_step1_gate.py:471-483` (gerçek seed-101 koşularının `src` özeti
  bugünkü `src/` ile aynı) ve `tests/test_foton_model.py:409-435` (bugünkü dense / MoE modeli tag'li `src/model.py` ile bit özdeş).

### 8.3 Kapsam boşlukları ve kırılganlıklar

- **Testi olmayan `src` parçaları:** `src/tokenizer.py` (her `main()` testinde sahte nesneyle değiştirilir: `vocab_size, bos_id,
  eos_id = 64, 1, 2`), `src/sample.py`, `src/checkpoint.py`'nin `load_model`'i, `train.main()`'in örnek üretim yolu
  (`src/train.py:600-608`; fixture'lar hep `--sample_tokens 0` verir).
- **Testi olmayan betikler (26):** `prepare_data`, `prepare_foton_corpus`, `download_tinystories`, `corpus_report`, `benchmark`,
  `vram_ladder`, `vram_probe_foton`, `verify_dispatch`, `flops_report`, `plot_metrics`, `compare_curves`, `analyze_experts`,
  `eval_domains`, `calibrate_v02`, `power_analysis`, `rule_reachability`, `baseline_reference`, `check_prereg_numbers`,
  `v03_step1_replay_streams`, `v03_step1_reachability`, `v03_step1_gpu_two_route`, `mutation_guards` ve dört mutant kataloğu.
  Yani tokenizer eğitimi, `.bin` üretimi (BOS / EOS yerleşimi), checkpoint yükleme ve örnek üretimi mevcut testlerce korunmuyor.
- **Çalıştırma ön koşulları:** paket yalnız proje kökünden `python -m pytest` ile çalışır (üç dosya `sys.path` kurmadan `src`
  import eder: `tests/test_attn_patterns.py:25`, `tests/test_data_multi.py:25`, `tests/test_ox1_microbench.py:37-38`). Geçici
  dizin yolu "foton" içerirse `tests/test_compare_runs_quarantine.py:56` yüzünden 4 test düşer (varsayılan `%TEMP%` uygundur).
- **Diske bağlı testler:** izlenmeyen `data/foton` ve `checkpoints/{foton-v02r*, foton-v03-step1}` yoksa ilgili testler sessizce
  atlanır → temiz klonda kapsam düşer (izlenen defter ve karar kuyruğu ile `foton-v02-prereg` tag'i klonla gelir; onlar da yoksa
  ilgili testler atlanır). Skip koşulları veri bağımlılığından DAR: `checkpoints/foton-v02-tarama` tek başına yoksa
  `tests/test_v03_step1_gate.py:476-483` atlanmaz, düşer; `r-m-s11` / `r-d-s22..55` / `v03s1-d-s101` eksikse
  `tests/test_v03_dcheck.py:64-91` düşer.
- `tests/test_v03_step1_gate.py`'deki 4 test gerçek kampanya kayıtlarını silebilen `camp.main()`'i çalıştırır; yalıtım yalnız
  modül sabitlerinin yamalanmasına dayanır (`186-189`, `243-246`, `271-276`, `401-404`). Yamalar satır satır okunarak doğrulandı;
  koşumda kayıtlar değişmedi.
- README'nin test iddiası (`README.md:80-81`: "CPU, birkaç saniye"): "CPU" doğru; süre bugün 148 saniye.


---

## 9. Sorunlar: bilinen hatalar, TODO'lar, tutarsızlıklar, ölü kod

**İşaret taraması:** `src/`, `scripts/`, `tests/`, `docs/`, kök `.md` dosyaları ve `experiments/**/*.md` içinde `TODO`, `FIXME`,
`XXX`, `HACK`, `BUG`, `workaround`, `deprecated` işareti **0**. Açık işler işaretle değil düzyazı kayıtlarla tutuluyor
(ön-kayıt "Sapmalar"ı, denetim günlükleri, raporlar). Aşağıdaki önem dereceleri "standart modele geçişi ya da sonuçların /
kayıtların doğruluğunu etkileme" ölçüsüne göredir.

### 9.1 Yüksek

| # | bulgu | kanıt |
|---|---|---|
| Y1 | **`data/foton` korpusunda EOS / BOS yok; kök neden doküman ayracı koşulunun hiç doğru olmaması** (`"\x1e"` Python'da boşluk sayılır). Sınırlar `[13, 33, 13]` bayt token'ları olarak veride. Kök neden hiçbir belgede yazmıyor; veri hazırlama betiklerinin testi yok; `corpus_report.py`'nin `verify()`'ı EOS'u sınamıyor | `scripts/prepare_foton_corpus.py:51`, `90`, `227`, `230-231`; `scripts/corpus_report.py:69-125`; bölüm 5.3 |
| Y2 | **v03 adım 1 kampanyası yarıda ve sonucu izlenen hiçbir dosyada yok:** `gate_s102_exceeded`; seed 103 koşmadı; 4 defter satırı 2026-09-21'den beri commit'siz; kapı kayıtları yalnız `.gitignore`'lu klasörde ve sürücü yeniden başlatılırsa silinir | `checkpoints/foton-v03-step1/campaign_status.json`, `gate_s102.json`; `scripts/v03_step1_campaign.py:108-111`; bölüm 7.4 |
| Y3 | **Geçiş engeli (tasarım gereği, hata değil):** doğrulayıcılar ve testler bugünkü `src/` baytlarını ve mimariyi çiviliyor; `src/` altındaki ilk değişiklik kampanya kapısını ve gerçek seed-101 testini kırar, mimari değişikliği bit-özdeşlik testini ve değere çivili sayıları düşürür | bölüm 7.6 |
| Y4 | **Argümansız çalıştırılınca dondurulmuş / çivili kaydın üzerine yazan betikler** (`verify_pins` dışında ret yok): `corpus_report.py`, `power_analysis.py`, `compare_router.py`, `compare_runs.py`, `compare_curves.py`, `prepare_foton_corpus.py` (çivili 7 dosya; yazım atomik değil), `baseline_reference.py` (koşulsuz), `calibrate_v02.py`, `rule_reachability.py`; `experiments/v3-router-ablation/report_numbers.py` içe aktarılınca da yazar. `verify_pins.py --generate` hash değişimini / çivi düşmesini `--force`suz reddeder (`317-328`) ama değişmemiş hash'lerle tag'li `pins.json`'ı yeni zaman damgası ve `git_head` ile yeniden yazar (`249-250`, `340-343`). Bugüne dek bozulma yok (dondurulmuş kayıtlar tag'deki hâliyle aynı) | `scripts/corpus_report.py:132`; `power_analysis.py:95`; `compare_router.py:103-104`; `compare_runs.py:156-157`; `compare_curves.py:71`; `prepare_foton_corpus.py:149`, `243-244`; `baseline_reference.py:186-188`; `calibrate_v02.py:62`, `197`; `rule_reachability.py:110`; `verify_pins.py:249-250`, `317-328`, `340-343` |
| Y5 | **Korpus içeriği adlarını karşılamıyor:** "code" alanı %100 Markdown (çitli kod %16.52); NL val'in tamamı tek alt kaynak (İskoç Parlamentosu), train'in %51'i olan caselaw val'de yok | bölüm 5.1 |

### 9.2 Orta

| # | bulgu | kanıt |
|---|---|---|
| O1 | **README bütünüyle eski** (son dokunan commit `461c9de`, 2026-09-16): başlık `moe-llm`; yol haritası "⏳ FOTON v0.1"; `src/` için 10 dosya (gerçek 13); 5 preset (gerçek 20); 14 betik (gerçek 51); "196 test"; v0.2 / v03, `PREREGISTRATION_V02.md`, `data/foton`, `docs/ARCHITECTURE_AS_BUILT.md` hiç anılmıyor. README'deki 25 komut satırının hepsi bugün de geçerli (betik, bayrak, preset var) | `README.md:1`, `19-22`, `27-57` |
| O2 | Belgeler EOS konusunda veriyle iki yönde çelişiyor ("EOS ile işaretli" ↔ 0 EOS; "sınır yok / geri çıkarılamaz" ↔ `<0x1E>` sınırları) | bölüm 5.3 |
| O3 | Eğitim döngüsü önceki mikro-batch'in logit tensörünü canlı tutuyor ve CE float32 kopya üzerinde: bellek `B × T × V` ile büyür (ön-kayıtta Sapma-S7 olarak belgeli, hâlâ kodda) | `src/train.py:647`, `653`; `src/model.py:468`; `PREREGISTRATION_V02.md:2028-2035` |
| O4 | `generate()` zorlanmış atama kollarında çalışmaz (`self(idx_cond)` atamasız çağırır → `ValueError`); `src/sample.py` bu checkpoint'lerden üretim yapamaz; preset'ler `sample_tokens=0` ile örtüyor. KV cache yok | `src/model.py:449-450`, `523-528`; `src/sample.py:46`; `src/config.py:228`, `234`, `240` |
| O5 | Resume, config'i checkpoint'ten değil komut satırından alır; tek denetim token / adım. Resume yolu kayıtlı hiçbir GPU koşusunda kullanılmamış; kuyruk yarım koşuyu sürdürmez, baştan koşar | `src/train.py:450-454`; `scripts/run_queue.py:158-163`, `431` |
| O6 | GPU koşuları tekrarlanabilir değil; determinizm ayarı yok (aynı seed, 100 adımda 0.0032 train / 0.0069 val fark) | bölüm 6.3 |
| O7 | Preset nesnesindeki `vocab_size` (8 192) gerçek koşu değeri değil; tokenizer yüklemeyen ölçüm betikleri açıkça verilmedikçe V = 8 192 ile ölçer; 32 768 literal olarak 10'dan fazla hesap satırında; aynı sayı "adım başına token" anlamında da geçiyor | `src/config.py:15`; `src/train.py:394`; `scripts/benchmark.py:34`, `40`; `scripts/flops_report.py:31-47`; `scripts/vram_ladder.py:96-99`; `src/flops.py:286`; `scripts/preset_hash.py:32` |
| O8 | Kuyruk izlenen bir dosyaya (defter) yazdığı için ham `dirty` alanı her oturumun ikinci koşusundan itibaren hep `true` (defterde 41 satırın 37'si); anlamlı alan `code_dirty`. v03 koşuları "tag'le donmuş" `experiments/foton-v02-prereg/` klasöründeki deftere ekleniyor; kuyruk olduğu gibi kullanılırsa yeni modelin koşuları da oraya yazılır | `scripts/run_queue.py:236-243`, `438-440`; `scripts/run_ledger.py:51`, `438` |
| O9 | Uzak depo yok; bilinen tek yedek aynı makinede `D:\foton gama v1 yedek\`: `moe-llm.bundle` (git geçmişi HEAD `f8dcf35`'e kadar; 2 dal + 5 tag) + `.prev`, `checksums.txt`, `checkpoints/{dense,moe,moe-noaux}` (3 013 344 270 B) ve `data/sp8k` tokenizer + meta (257 511 B; `.bin` yok). Orada OLMAYAN: `data/foton` (çivili korpus + 32k tokenizer), v2 / v3 / FOTON checkpoint'leri (~70 GiB), commit'siz 4 defter satırı, `checkpoints/foton-v03-step1/` kampanya kayıtları. Başka bir yedek konumu olup olmadığı **doğrulanamadı** | `git remote -v`; `scripts/backup_bundle.py:43`; `D:\foton gama v1 yedek\` listesi |
| O10 | İzlenmeyen artifact'lara (`data/foton`, `checkpoints/{foton-v02r*, foton-v03-step1}`) dayanan testler yoklarsa sessizce ATLANIR: `checkpoints/` temizliği ya da temiz klon kapsamı görünmeden düşürür; kısmi eksiklikte ise (ör. yalnız `foton-v02-tarama` ya da `r-d-s22..55` yok) atlanmayıp DÜŞERLER. `checkpoints/foton-v02r.kirli-src-iptal` adı "iptal" olsa da v03 dcheck testinin ve kuralının canlı bağımlılığıdır | `tests/test_architecture_doc.py:107`; `tests/test_v03_dcheck.py:32-34`, `72-73`, `87-88`; `tests/test_v03_step1_gate.py:474-481`; `scripts/v03_dcheck.py:77`, `80` |
| O11 | Tokenizer: Türkçe / Almanca harfler sözlükte yok; araç kullanımı için özel token altyapısı yok (`user_defined_symbols` boş, pad yok); eğitim yeniden üretilebilir değil (tohum yok, eğitim günlüğü yok, örnek dosyası çivisiz) | `src/tokenizer.py:57-77`; bölüm 4 |
| O12 | Preset hash koruması fiilen yürürlükte değil (`preset_hashes.json` hiç yazılmadı; hiçbir betik `--verify` çağırmıyor; hiçbir test gerçek özeti çivilemiyor) | `scripts/preset_hash.py:31`; `PREREGISTRATION_V02.md:1606` |
| O13 | `vram_ladder.py` ve `benchmark.py` gerçek eğitim tepesini az gösterir (gerçek koşu merdivenin +0.71 GB dense, +0.90 GB MoE üstünde; `baseline-d` / `baseline-m` tabanı); merdiven VRAM sınırı koymaz ve kayıtlı dosyaları koşul bilgisi (mikro-batch, vocab, bağlam) taşımaz | `scripts/vram_ladder.py:47`, `116-118`; `scripts/benchmark.py:53`; bölüm 6.8 |
| O14 | FOTON desenlerinin "eşit bağlantı" özelliği ve kilidi yalnız T = 512'de geçerli; T ≠ 512'de model itirazsız kurulur ve eşitlik sessizce düşer; maske buffer'ı blok başına `n_head·T²` bayt | `src/model.py:298`; `src/flops.py:124-128`; `src/attn_patterns.py:38-46` |
| O15 | Hiç koşmamış / hiç üretilmemiş parçalar: H kolu (öğrenilmiş router'la; yalnız 9 adımlık VRAM sondası), dört S1 preset'i, `budget_mult = 2`, `decide_primary` / `abandon` gerçek veride, `compile`, float32 eğitim, dropout > 0, bağsız embedding, 512'den uzun bağlam; ön-kaydın V11 çıktılarının çoğu ve profiler çapraz kontrolü (Sapma-S9) | 213 `config.json` taraması; `experiments/foton-v02/REPORT_07.md:42-44`, `62`, `225-232` |
| O16 | Lisans kaydı boşlukları: TinyStories için depoda lisans kaydı yok; `data/foton/meta.json`'ın `licensing` bloğu boş sayım taşıyor (tutulan alt kümenin dökümü depoda yok) | `data/sp8k/meta.json`; `data/foton/meta.json:35-72` |
| O17 | N* için üç farklı tanım (`src/stats.py:249-287` kalıcı kesişim; `scripts/power_analysis.py:72-77` ilk kesişim; `scripts/screening_context.py:62-70` ln(adım)'da doğrusal) ve t tablosunun `src/stats.T975` dışında dört ayrı kopyası (`scripts/compare_curves.py:32`, `power_analysis.py:40`, `compare_runs.py:49`, `experiments/v2-grouped-dispatch/diagnostics/report_numbers.py:20`) | — |
| O18 | Mutant katalogları kaynak satırlarına metin olarak çivili (161 mutant `src/*.py`): `src` yeniden yazılınca suite'ler düşer; pytest bunu görmez | `scripts/mutation_guards.py:195-209` |

### 9.3 Düşük

| # | bulgu | kanıt |
|---|---|---|
| D1 | Çıplak `assert` tutarsızlığı: kod başka yerde "çıplak assert değil: `python -O` atlamasın" kuralını ilan ediyor, ama taban doğrulamalar çıplak assert | `src/model.py:91`, `439`; `src/moe.py:202-204` ↔ `src/model.py:284` |
| D2 | `head_dim`'in çift olması denetlenmiyor (tek değerde model kurulur; `head_dim ≥ 5`'te ilk forward `RuntimeError`, 1 ve 3'te sessizce bozuk RoPE ile çalışır) | `src/model.py:82-85`, `91` |
| D3 | `--vocab_size` bayrağı `src.train`'de sessizce etkisiz; `argparse` kısaltmaları açık (`--stop` → `--stop_step`) | `src/train.py:394`; `src/config.py:302` |
| D4 | Yorum ↔ kod: `loss_prefix_skip` yorumu maskenin `ignore_index=-1` ile uygulandığını söylüyor, gerçekte eval'de dilimleme; modül docstring'i "AdamW adımı → LR güncelle" diyor, kodda LR adımdan önce; `src/model.py:118` "flash / mem-efficient" diyor, kayıtlarda yalnız mem-efficient çekirdek | `src/config.py:120-121` ↔ `src/train.py:269`, `283`; `src/train.py:8-10` ↔ `625-627`; `src/model.py:118` |
| D5 | `src/tokenizer.py:102` `meta.json`'ı kodlama vermeden açıyor (bu makinede cp1252); diğer okuyucular UTF-8. Bugünkü içerik ASCII olduğu için zararsız. Aynı şekilde `src/train.py:497` `config.json`'ı ve `scripts/run_queue.py:436` `DONE`'ı kodlamasız yazar (ASCII; zararsız) | `src/tokenizer.py:102` ↔ `src/train.py:398`, `src/data_multi.py:152` |
| D6 | Ölü kod / kullanılmayan adlar: `import math` (`src/model.py:31`; `scripts/baseline_reference.py:28`); `SD_TRIPWIRE = None` (`src/stats.py:69`); `MultiDomainDataset(weights=…)` dalı (`src/data_multi.py:42-44`); `CONTEXT_KEYS` (`src/flops.py:45`; `scripts/` altında tüketicisi yok, yalnız `src/flops.py:346` ve testler); `G_STATS2` (`scripts/mutation_guards.py:56`); `TiktokenTokenizer` yolu (`src/tokenizer.py:80-97`) | Grep |
| D7 | Yalnız testte / kendi `main`'inde yaşayan fonksiyonlar: `boundary_diagnostic`, `ROWS` (`src/stats.py:678`, `800`); `per_head_avg`, `equalization`, `baseline_gap`, `headline`, `self_check` (`src/attn_patterns.py:263-327`); `foton_flops_realized`, `foton_arm_connections`, `foton_module_connections` (`src/flops.py:132-145`, `250`); `MultiDomainDataset.get_batch` (`src/data_multi.py:59`) | Grep |
| D8 | Etkisiz bayraklar / erişilemeyen dal: `make_campaign_queue.py --strict` (`156-157`), `prepare_foton_corpus.py --doc_batch` (`156`; Y1 yüzünden), `calibrate_v02.py:134-141` | — |
| D9 | `requirements.txt`'te torch dışında sürüm sabiti yok; FOTON dönemi için `pip freeze` kaydı (`environment.txt`) yok | `requirements.txt:7-13` |
| D10 | Belgede anılıp diskte olmayan yollar (çoğu koşulmayan karar kampanyasının çıktıları): `experiments/foton-v02/decision.json`, `savings.json`, `owner_decisions.jsonl`, `preset_hashes.json`, `flops_crosscheck.json` … | `PREREGISTRATION_V02.md:1804-1819`, `2000`; `docs/flops.md:384` |
| D11 | Satır sonu geçmişi: `foton-v02-prereg` tag'inde `src/config.py`, `model.py`, `train.py` CRLF, HEAD'de LF; `src_sha256` bayt özeti olduğu için dönemler arası özet eşitliği kod eşitliğini göstermez. İzlenen dosyaların 314'ü CRLF (hepsi `experiments/` altında), 4'ü karışık | `git ls-files --eol`; `.gitattributes:8` |
| D12 | Depo hijyeni: dal adı `foton-v01` (içerik v03 adım 1); `main` 29 commit geride; v2 ve v03 adım 1 için tag yok; iki boş `.gitkeep` artığı (`src/`, `scripts/`) | bölüm 1.3 |
| D13 | Referanssız checkpoint artıkları: `checkpoints/smoke-moe` ve `foton-v02r-queue.out` hiçbir kayıt / betik / test tarafından anılmıyor (`checkpoints/smoke`, `README.md:84`'teki duman komutunun çıktısı); `v2-verify` (2 539 960 400 B) yalnız iki donmuş kuyruk dosyasının `out_dir`'i | Grep |
| D14 | Test paketindeki küçük sorunlar: totolojik test (`tests/test_preset_hash.py:54-57`); geçici dizin YOLU "foton" içerirse 4 test düşer (`tests/test_compare_runs_quarantine.py:56`); üç dosya `sys.path` kurmadan `src` import eder (yalnız kökten `python -m pytest` ile çalışır); zamanlamaya dayalı assert'ler (`tests/test_train.py:161-163`); `@needs_runs` test olmayan yardımcıya uygulanmış (`tests/test_v03_dcheck.py:58-61`, etkisiz); `.pytest_cache/v/cache/lastfailed` bayat | bölüm 8 |
| D15 | `scripts/prepare_foton_corpus.py`'de `uint16` taşma denetimi ve istenen / üretilen vocab karşılaştırması yok (sp8k betiğinde var) | `scripts/prepare_foton_corpus.py:208-209`, `227` ↔ `scripts/prepare_data.py:59`, `119-120` |
| D16 | Belge ↔ kod küçük uyumsuzlukları: `scripts/owner_decision.py:13`, `32` var olmayan `--scale S1` bayrağını anıyor; `scripts/mutation_guards.py:29-33` "iki sınav" diyor, kod 4 suite tanıyor; `scripts/plot_metrics.py:6` "token sayısına göre" diyor, kod yalnız adıma göre çiziyor | — |
| D17 | Disk: proje `C:` sürücüsünde, boş alan ≈ 40 GiB; `checkpoints/` 73.16 GiB; koşu klasörü D kolunda ≈ 0.68 GB, X / H'de ≈ 0.92 GB, M'de ≈ 1.47 GB (`best.pt` + `last.pt`) | bölüm 1.2 |
| D18 | `.git` içinde erişilemez (dangling) nesneler: 2026-09-15 tarihli eski bir stash kalıntısı ("On main: moe-wip") ve bu envanter oturumunda aracın (Claude Code) her tur başında kendiliğinden aldığı `git stash create` türü anlık görüntüler ("index on foton-v01" / "WIP on foton-v01" commit çiftleri; `.git/index` de bu sırada tazelendi). Ref'lere, HEAD'e ve çalışma ağacına dokunmazlar; `git status` / `fsck` çıkışını değiştirmezler; `git gc --prune` ile temizlenebilirler | `git fsck --unreachable` |

### 9.4 Belgelenmiş ve hâlâ geçerli bilinen sorunlar

| kaynak | kayıt | bugünkü durum |
|---|---|---|
| `PREREGISTRATION_V02.md` Sapma-S6 (`2019-2026`) | dondurulmuş korpusta doküman sınırı / EOS yok | geçerli; kök nedeni Y1 |
| Sapma-S7 (`2028-2035`) | döngü önceki mikro-batch'in logit'ini canlı tutuyor (0.54 GB) | hâlâ kodda (O3) |
| Sapma-S9 (`2047-2053`) | profiler sayaç artefaktı; "07'de çözülecek" | yapılmadı (`experiments/foton-v02/REPORT_07.md:42-44`) |
| Sapma-S11 (`2086-2111`) | karar kampanyası koşulmayacak; V10 yer tutucusu doldurulmayacak | nihai |
| `experiments/foton-v02-prereg/audit_log.md:60` | açık denetim bulgusu #22: `compare_runs.paired` konuma göre `zip()` | hâlâ kodda (`scripts/compare_runs.py:102-103`); betik karantinada |
| `README.md:659-666` "Sonraki adımlar" | KV-cache, MLA, daha uzun bağlam, daha büyük / çok dilli veri | KV-cache ve MLA yok; bağlam hâlâ 512 |
| `experiments/foton-v03-step1/audit_log.md:103-105` | (b) sorusu ("X-dom eşli D'yi geçiyor mu") ayrı ön-kayda ertelendi | cevapsız |

---

## 10. Etki haritası

Altı planlanan değişiklik için dokunulması gereken yerler — **yalnız liste; hiçbiri uygulanmadı, tasarım önerilmedi.** Harita iki
bağımsız taramayla (değişiklikten siteye / dosyadan değişikliğe) çıkarıldı, birleştirildi ve her satır dosya açılıp okunarak
doğrulandı. Tür etiketleri: **[ZORUNLU]** = değişmeden çalışmaz ya da yanlış sonuç üretir (kod yolu, config alanı, veri biçimi);
**[TÜREV]** = mimari değişince bayatlayan ya da kırılan sayı / belge / test çivisi / yorum; **[KİLİT]** = dondurulmuş kayıt ya da
doğrulayıcı — değişiklikle çakışır (dokunulmaması gereken ya da bilerek kırılacak şey). Mevcut durum özeti: RoPE, RMSNorm, SwiGLU
zaten var; GQA ve QK-norm yok; bağlam 512; `split_digits` zaten açık; `data/foton`'da EOS yok (bölüm 5.3).

### 10.0 Hepsi için ortak kilitler

`src/` altındaki İLK bayt değişikliğinde (yorum ya da boşluk dahil; `.gitattributes:8` `* -text`, dosyalar bit bit korunur) kırılan ya da çakışan doğrulayıcılar ve dondurulmuş kayıtlar. Her satır okunarak doğrulandı; sayılar salt-okur sonda `probes/ETKI_facts.py` çıktısından (S1–S6).

| kilit | dosya:satır | ne olur | tür |
|---|---|---|---|
| A. `src/` özeti (`src_sha256`) | `scripts/run_queue.py:184-195` (hash = `src/**/*.py` yol + ham bayt), `239` (provenance), `410-418` (sıkı kuyruk: kuyruk sürerken özet değişirse sıradaki koşu BAŞLAMAZ, çıkış 91), `59`, `63`, `369`, `421-427` (`code_dirty`: `src/ scripts/ tests/ docs/ PREREGISTRATION*.md` altında commit'siz değişiklik → sıkı kuyruk HİÇ başlamaz, çıkış 90) | şu anki özet `55703d68f224…` = 8 v03 adım-1 koşusu + `foton-v03-dcheck/dcheck-d-s11` provenance'ı ile AYNI (sonda S1); ilk bayt değişikliğinden sonra farklı | [KİLİT] |
| A1. v03 adım-1 kapısı ve sürücüsü | `scripts/v03_step1_gate.py:80-86` (seed koşularının özeti ŞU ANKİ ağaçla aynı olmalı, yoksa `ValueError`); `scripts/v03_step1_campaign.py:79-82` (0. adım yeniden denetim), `89-92` (her seed kapısı) | kampanya durumu `gate_s102_exceeded` (`checkpoints/foton-v03-step1/campaign_status.json`), seed 103 bloğu diskte yok → src değişirse kalan 4 koşu bu ağaçtan başlatılamaz | [KİLİT] |
| A2. v03 analizi | `scripts/v03_step1_analyze.py:359-364` | 12 koşuda TEK src özeti + `code_dirty False` zorunlu | [KİLİT] |
| A3. gerçek-veri src sürekliliği testi | `tests/test_v03_step1_gate.py:33` (`REAL_SRC = code_hash(ROOT)`), `474-480` (gerçek seed-101 dizinleri diskte → test ATLANMAZ, kapıyı çağırır) | src değişince test düşer | [KİLİT] |
| B. Metin-çivili mutantlar | `scripts/mutation_guards.py:195-209` (hedef metin TAM 1 kez ya da n. geçiş; yoksa `assert` durdurur), `79-97` (T01-T02 → `src/train.py:283`; C01-C03 → `src/config.py:182`, `190`, `181`; D01-D02 → `src/data_multi.py:93`, `100`; A01 → `src/attn_patterns.py:43`); `scripts/mutation_catalog_06.py:31-66` (F01-F14 → `src/model.py:316, 300-301, 288, 128, 41, 368, 334, 102, 200-202, 320, 399, 485`), `68-96` (G01-G12 → `src/train.py`, ör. 94, 96, 214, 223, 441, 518, 590), `131-136` (C04-C06 → `src/config.py:205`, `196`, `265`), `138-156` (W01-W07 → `src/flops.py:167, 166, 170, 126-127, 103, 276-277, 266`); `scripts/mutation_catalog_v03s1.py:22-32` (V01-V05 → `src/model.py:353, 449, 447, 351, 249`), `34-90` (V06-V16, R01-R12 → `src/train.py`) | altı dosyada 72 mutant hedefi (model.py 19, train.py 37, config.py 6, flops.py 7, data_multi.py 2, attn_patterns.py 1); pinli satır değişirse ya da aynı metin ikinci kez eklenirse ilgili sınav durur; kayıtlar `experiments/foton-v02-prereg/mutation_log.md`, `experiments/foton-v02/mutation_log_06.md`, `mutation_log_07.md`, `experiments/foton-v03-step1/mutation_log.md` donmuş | [KİLİT] |
| C. D/M bit-özdeşliği tag'e karşı | `tests/test_foton_model.py:409-435` (`git show foton-v02-prereg:src/model.py`, aynı seed → `state_dict` anahtar listesi + tensörler `torch.equal`, loss `torch.equal`) | `foton == ""` modelinin kurulum sırası, init çekim sırası ya da ileri yön sayısal yolu değişince kırılır (GQA, QK-norm, RoPE için); yalnız yorum değişikliğinde kırılmaz | [KİLİT] |
| D. Donmuş FLOP ve parametre sayıları | `tests/test_flops.py:207-214` (267 411 456 / 267 755 520 + `PREREGISTRATION_V02.md` metninde aranır), `244-258` (FOTON olmayan 10 preset: anahtar kümesi, sırası, değerleri bit bit; fazladan anahtar yalnız `foton_*` ve 0); `scripts/check_prereg_numbers.py:125`; `scripts/v03_step1_analyze.py:55-64` + `tests/test_v03_step1.py:652-669` (42 476 032 / 55 326 720; 267 411 456 / 265 838 610 / 265 834 602); `tests/test_architecture_doc.py:31-43`, `94-99` | varsayılan config'in FLOP'u ya da parametresi değişince hepsi | [KİLİT] |
| E. Preset değerleri | `tests/test_foton_frozen.py:97-110`, `125-137`, `140-154`, `157-167`; `scripts/preset_hash.py:32-42`, `72-77` (`asdict(ModelConfig)` + `asdict(TrainConfig)` hash'e girer → her yeni alan 8 kampanya hash'ini değiştirir) | gizil: `experiments/foton-v02/preset_hashes.json` diskte YOK (sonda S6), `PREREGISTRATION_V02.md:1606` yer tutucu boş | [KİLİT] |
| F. Belge ↔ kod çivileri | `docs/ARCHITECTURE_AS_BUILT.md` ↔ `tests/test_architecture_doc.py:31-43, 45-65, 68-76, 79-91, 94-99, 102-120`; `docs/flops.md` §8 ↔ `tests/test_flops.py:584, 602-614` (8 üretilen tablo), `617-632` (düz metin sayıları) | mimari / FLOP sayısı değişince belge yeniden üretilmeden testler düşer; `docs/` de `code_dirty` kapsamında (`scripts/run_queue.py:59`) | [KİLİT] |
| G. Ön-kayıt gövdesi | `scripts/check_prereg_frozen.py:26-53`; `tests/test_prereg_frozen.py:51-57`; `scripts/check_prereg_numbers.py:268-274` (metindeki 8 SHA çivisi diskle) | `PREREGISTRATION_V02.md` gövdesine yazılamaz; yalnız "Sapmalar" sonuna eklenir | [KİLİT] |
| H. Çivi manifesti | `experiments/foton-v02-prereg/pins.json:14-68` (39 dosya) ↔ `scripts/verify_pins.py:114-128`, `274-289` (`data/foton` altına eklenen çivisiz HER dosya `--verify`'ı düşürür), `294`, `319-325` (yeniden çivileme `--force` ister); `PREREGISTRATION_V02.md:417-424`, `435-441` | `data/foton` ve `checkpoints/foton-v02r/r-*` dosyalarına dokunulamaz, `data/foton`'a dosya eklenemez | [KİLİT] |
| I. Checkpoint hash'leri | `experiments/v1-dense-moe-baseline/checksums.txt:2-8` (dense / moe / moe-noaux `best.pt` + `last.pt`, `data/sp8k/tokenizer.model`) | eski `.pt`'ler yerinde "yükseltilemez" | [KİLİT] |
| J. Eski checkpoint'lerin yeniden kurulumu | `src/checkpoint.py:40-41` (`ModelConfig(**model_cfg)`: yeni alan → varsayılan, kaldırılan ya da yeniden adlandırılan alan → `TypeError`; `load_state_dict` KATI); `src/train.py:455`; `scripts/compare_runs.py:175-176`, `287-289`; `scripts/power_analysis.py:52`; `scripts/router_grad_diag.py:106-107`; `scripts/router_probe_diag.py:88`; `scripts/v03_step1_gpu_two_route.py:62-63`; `experiments/v2-grouped-dispatch/diagnostics/stage_times.py:26-27`, `layer_variants.py:22`, `bench_step.py:25`, `det_probe.py:23`, `det_localize.py:20`, `grad_diag.py:21` (hepsi `load_model`) | diskte 79 `best.pt` + 79 `last.pt`, 160 `config.json` (sonda S6: `max_seq_len` 512 ×158 / 128 ×2, `n_head` 8 ×158 / 4 ×2, `rope_theta` hepsi 10000.0, `vocab_size` 8192 ×120 / 32768 ×40, model alan sayısı {9, 17, 18, 20, 27, 28}); varsayılan eski davranışı korumazsa 158 checkpoint sessizce BAŞKA modelle yüklenir | [KİLİT] |
| K. Config kimlik denetimleri | `scripts/v03_step1_analyze.py:75`, `125-126`, `141-144` (`a.get(k, "") != b.get(k, "")`: eski v02-X `config.json`'da olmayan, varsayılanı `""` dışı her yeni `ModelConfig` alanı kimlik denetimini düşürür); `scripts/v03_dcheck.py:211-219` (`foton` ile başlamayan yeni model anahtarı → `model_unexpected`) | yeni alanla koşulan koşular v02 referansıyla eşlenemez | [KİLİT] |
| L. Köken ve parmak izi kayıtları | `experiments/foton-v02/ox1_microbench.json:430-432` (`src/attn_patterns.py`, `src/model.py`, `src/config.py` SHA'ları); `experiments/v3-router-ablation/checks/fingerprint.py:6-11`, `19-21` + `fingerprint_v2_code.json` / `fingerprint_v3_code.json` (aynı seed, aynı girdi, 3 AdamW adımı → loss'lar ve bütün parametrelerin SHA'sı) | dense / MoE yolundaki her sayısal değişiklik parmak izini değiştirir; kayıtlar donmuş | [KİLİT] |
| M. Koşu defteri | `scripts/run_ledger.py:51` (`LEDGER = experiments/foton-v02-prereg/run_ledger.jsonl`) | yeni koşular da donmuş klasördeki deftere yazılır | [KİLİT] |

### 10.1 GQA

Mevcut durum: düz MHA — `n_head = 8` (`src/config.py:19`), q/k/v tek `nn.Linear(d, 3d)` (`src/model.py:104`), ayrı KV-kafa alanı yok; 20 preset'te `n_head` {8: 13, 4: 3, 10: 4} (sonda S5).

| dosya | satır(lar) | neden dokunulur | tür |
|---|---|---|---|
| `src/config.py` | 14-25 (19) | `ModelConfig`'te KV-kafa sayısı alanı yok; eklenecek alan buraya | [ZORUNLU] |
| `src/config.py` | 304-308 | `build_configs` tip haritası yalnız int / float / str / bool: alan bu tiplerden olmalı (yoksa `KeyError`) | [ZORUNLU] |
| `src/config.py` | 142-289 | GQA'yı alacak YENİ preset burada tanımlanır; mevcut 20 preset değişmez (10.0-E) | [ZORUNLU] (yeni preset) |
| `src/config.py` | 19, 148-149, 193-194 | yorumlar "head_dim = 512/8 = 64", "8 head" | [TÜREV] |
| `src/model.py` | 91-93 | `d_model % n_head == 0`, `head_dim`: KV-kafa sayısının bölünebilirliği ve tanımı burada | [ZORUNLU] |
| `src/model.py` | 104 | `qkv = nn.Linear(d, 3d)`: GQA'da k ve v `n_kv·head_dim` boyutlu | [ZORUNLU] |
| `src/model.py` | 111 | `split(C, dim=2)` üç EŞİT parça varsayar | [ZORUNLU] |
| `src/model.py` | 112-114 | k ve v `view(B, T, n_head, head_dim)`: KV-kafa sayısı + tekrar / genişletme ya da SDPA'nın GQA yolu | [ZORUNLU] |
| `src/model.py` | 117 | `apply_rope(k, …)` k'nin yeni şekliyle; tekrarın RoPE'den önce mi sonra mı olduğu burada | [ZORUNLU] |
| `src/model.py` | 120-130 | iki SDPA çağrısı q/k/v kafa sayısı eşit varsayar; maskeli yol `attn_mask [n_head, T, T]` q kafası başına (FOTON) | [ZORUNLU] |
| `src/model.py` | 131-132 | çıkış `view(B, T, C)` + `proj`: q kafası × head_dim = d kaldıkça aynı | [TÜREV] (denetim noktası) |
| `src/model.py` | 213 | `precompute_rope(d_model // n_head, …)`: head_dim n_head'ten türetilir | [ZORUNLU] (head_dim tanımı değişirse) |
| `src/model.py` | 401-413, 106 | `_init_weights` yeni `nn.Linear`'ları otomatik N(0, 0.02) yapar; qkv ayrı matrislere bölünürse init çekim SIRASI değişir → 10.0-C kırılır | [TÜREV] |
| `src/model.py` | 103, 1-11, 19-20, 209-210 | yorum / docstring ("tek matris çarpımı", "6.4M parametre", "4.2M (~%14)") | [TÜREV] |
| `src/flops.py` | 172-173 | `attn_qkv_proj = n_layer · 2 · 4 · d²` (3d² + d² MHA varsayımı); `src/train.py:516-518` ile her koşunun meta satırına ve birincil ölçüye girer | [ZORUNLU] |
| `src/flops.py` | 201 | ikincil `rope_qk = n_layer · 2 · 3 · d` (k artık d boyutlu değil) | [TÜREV] |
| `src/flops.py` | 45, 154-187 | çıktı anahtar kümesi ve sırası çivili: `foton_` dışı yeni anahtar eklenemez (`tests/test_flops.py:253-256`) | [KİLİT] |
| `src/flops.py` | 286-364 | `foton_doc_tables` → `docs/flops.md` §8 yeniden üretilmeli (foton preset'leri değişirse) | [TÜREV] |
| `src/checkpoint.py` | 40-41 | eski 158 checkpoint `blocks.N.attn.qkv.weight` [3d, d]; katı yükleme → varsayılan MHA kalmalı ya da uyum katmanı (desen: `src/moe.py:300-305`, testi `tests/test_router.py:123-137`) | [KİLİT] |
| `src/train.py` | 455, 609 | resume katı `load_state_dict`; `asdict(mcfg)` checkpoint'e yazılır (yeni alan otomatik) | [KİLİT] (eski `last.pt`) / [TÜREV] |
| `scripts/architecture_as_built.py` | 36-37, 43 | `head_dim = d // n_head`; `blk.attn.qkv.weight.numel()` tek matris varsayımı | [ZORUNLU] |
| `scripts/ox1_microbench.py` | 72, 125-131 | `FROZEN_DIMS n_head 8`: `foton-x` preset'i saparsa RED | [KİLİT] |
| `scripts/vram_ladder.py` | 68, 110 | `"n_head": d_model // 64`; `per_layer = 4·dm·dm + …` (4d² attention parametresi varsayımı) | [TÜREV] |
| `scripts/compare_runs.py` | 175-176, 184, 287-289 | eski `config.json`'dan `ModelConfig`: yeni alan → varsayılan; varsayılan GQA'yı açarsa donmuş v2 koşularının `active_params` / FLOP'u sessizce değişir | [KİLİT] |
| `scripts/power_analysis.py` | 52 | donmuş v1 / v2 / v3 koşularının FLOP'u BUGÜNKÜ `dense` / `moe` preset'inden | [KİLİT] |
| `scripts/baseline_reference.py` 75; `scripts/calibrate_v02.py` 67; `scripts/check_prereg_numbers.py` 125 | — | `foton-d` / `foton-m` FLOP/token bugünkü formülden (267 411 456 / 267 755 520) | [KİLİT] |
| `scripts/v03_step1_analyze.py` | 55-64, 153-161 | `IDENTITY` `params_total` ve `flops_train_per_token` sabit; meta satırı birebir | [KİLİT] |
| `scripts/v03_step1_analyze.py` 125-126, 141-144; `scripts/v03_dcheck.py` 211-219 | — | yeni `ModelConfig` alanı eski kayıtta yok → kimlik denetimi düşer (10.0-K) | [KİLİT] |
| `scripts/preset_hash.py` | 42 | `asdict(ModelConfig)` hash'e girer (gizil, 10.0-E) | [KİLİT] |
| `scripts/mutation_catalog_06.py` | 40-42 (F05 → `model.py:128`), 53-55 (F10 → `:102`), 56-58 (F11 → `:200-202`) | SDPA çağrı satırı, maske buffer satırı, `GPT.__init__` gövdesi metin-çivili | [KİLİT] |
| `tests/test_model.py` | 130-134, 163-166 | parametre formülü `4·d·d` attention (sentetik CFG; varsayılan MHA kaldıkça geçer) | [TÜREV] |
| `tests/test_flops.py` | 30-51, 122-131 | aktif `nn.Linear`'ları modülden sayıp formülle eşitler: bayat `attn_qkv_proj`'u yakalayan kılavuz test | [TÜREV] |
| `tests/test_flops.py` | 62-79, 92-109 | profiler eşitlikleri; MATH-SDPA sayımı `n_layer · B · 4 · d · T · T` (k/v genişletme yoluna bağlı; koşulmadı) | [TÜREV] |
| `tests/test_flops.py` | 217-228, 244-258, 277, 469 | FOTON-öncesi formülün kopyası; `attn_qkv_proj == n_layer · 8 · d²` (foton preset'leri; S1 640 × 10) | [KİLİT] |
| `tests/test_foton_model.py` | 200-206, 264-267, 377 | `attn.qkv.in_features`, `qkv.weight[2*d:].view(H, hd, d)`, `qkv(x).split(64)`, `view(2, T, 8, 8)`, `m[0].attn.qkv.weight` (FOTON kolları GQA'ya geçerse) | [TÜREV] |
| `tests/test_foton_model.py` | 409-435 | 10.0-C | [KİLİT] |
| `tests/test_ox1_microbench.py` | 174, 306 | `blocks[0].attn.qkv` kancaları | [TÜREV] |
| `tests/test_architecture_doc.py` | 31-43, 53, 94-99 | `attn_qkv` 786 432 vb. belgede aranır; `router_weight_shape == [3, 512]`; X − H == 4 008 | [KİLİT] |
| `docs/ARCHITECTURE_AS_BUILT.md` | 11, 18, 21, 29-30, 40, 54-56, 60 | "n_head = 8 (head 64)", `qkv` 512→1536 = 786 432, blok 3 212 288, D 42 476 032 / 267 411 456, modül 6 424 576, H/X 55 326 720 (dört kolun belgesi, test kilitli) | [KİLİT] |
| `docs/flops.md` | 12, 32-34, 66-67, 79, 175, 257 | d = n_h·d_h; "q, k, v tek bir d → 3d matrisiyle … 8d² = 2 097 152"; toplam; §8 `attn_qkv_proj` 16 777 216 (koddan üretilir) | [TÜREV] (§2-7) / [KİLİT] (§8) |
| `README.md` | 147, 423 | "8 head"; "attention (projeksiyon + causal bağlam) 20 979 712" — "Donmuş kayıt" bölümü (140-141: sayılar değiştirilmez) | [KİLİT] |
| `PREREGISTRATION_V02.md` | 209-211 | "Eşitlenen eksenler: d_model, n_head, head_dim … Q/K/V boyutları" (donmuş gövde) | [KİLİT] |
| `experiments/v2-grouped-dispatch/diagnostics/stage_times.py` | 162, 26-27 | `blk.attn(…, model.rope_cos, model.rope_sin)`; v1 checkpoint'ini yükler | [KİLİT] |
| `experiments/v3-router-ablation/checks/fingerprint.py` | 6-11, 19-21 | dense + MoE modeli kurup parametre SHA'sı üretir; kayıt `fingerprint_v3_code.json` (10.0-L) | [KİLİT] |
| `experiments/**/config.json` (53), `checkpoints/**/config.json` (160) | ör. `experiments/v1-dense-moe-baseline/dense/config.json:7` `"n_head": 8` | yeni alan içermez; eski ağırlıklar [3d, d] | [KİLİT] |

Dokunmak gerekmeyen (sayımı modülden ya da config'ten alan): `src/model.py:415-429` (`num_params`), `tests/test_flops.py:182-190`, `src/attn_patterns.py` (maske q kafası başına; `N_HEAD_KAMPANYA = 8`, C deseninin yarı-yarıya bölünmesi q kafası sayısına bağlı).

### 10.2 QK-norm

Mevcut durum: yok — q ve k RoPE'den sonra doğrudan SDPA'ya girer (`src/model.py:117-130`), `scale=` verilmez (varsayılan 1/√head_dim), attention'da öğrenilir ölçek ya da norm parametresi yok.

| dosya | satır(lar) | neden dokunulur | tür |
|---|---|---|---|
| `src/config.py` | 14-25 | açma / kapama (ve varsa ölçek başlangıcı) alanı; `bool` `_parse_bool` (292-297) ile komut satırından | [ZORUNLU] |
| `src/config.py` | 142-289 | hangi preset'te açık olacağı (yeni preset) | [ZORUNLU] (yeni preset) |
| `src/model.py` | 88-107 | `CausalSelfAttention.__init__`: norm / ölçek parametresi burada kurulur (model bias'sız: 104-105, docstring 9) | [ZORUNLU] |
| `src/model.py` | 111-117 | q, k burada elde edilir; normun RoPE'den (117) önce mi sonra mı uygulanacağı; q/k autocast altında bf16 (`apply_rope` tabloyu `x.dtype`'a çevirir: 84) | [ZORUNLU] |
| `src/model.py` | 120-130 | iki SDPA çağrısında `scale=` yok → 1/√head_dim; normalize q·k'nin ölçeği burada | [ZORUNLU] |
| `src/model.py` | 47-60 | `RMSNorm` float32'de hesaplar, `x.dtype`'a döner; q/k normu için yeniden kullanılırsa dtype davranışı | [TÜREV] (denetim noktası) |
| `src/model.py` | 401-413 | `_init_weights` yalnız `nn.Linear` / `nn.Embedding`: ham `nn.Parameter` ölçeğine dokunmaz (başlangıç kurucuda verilir); `RMSNorm.weight = ones` (53) RNG tüketmez | [TÜREV] (denetim noktası) |
| `src/model.py` | 504-506 | weight decay `p.dim() >= 2`: 1-D ölçek decay'siz gruba, (H, 1, 1) şeklinde ≥ 2-D ölçek decay'li gruba düşer | [TÜREV] (denetim noktası) |
| `src/model.py` | 211, 218 | bağlı embedding `lm_head` üzerinden; attention'a eklenen her parametre `apply` sırasını kaydırır → 10.0-C | [TÜREV] |
| `src/model.py` | 1-11, 26-27, 115-119 | docstring / yorumlar ("init RNG tüketimi değişmez", RoPE yalnız q/k, çekirdek seçimi) | [TÜREV] |
| `src/flops.py` | 190-215 (199-202) | ikincil terimler: `rmsnorm (2L+1)·4d`, `attn_scale_softmax` "1/sqrt(hd) + exp…"; QK-norm terimi yok; `foton_` dışı yeni anahtar yasak (`tests/test_flops.py:253-256`) | [TÜREV] + [KİLİT] (anahtar kümesi) |
| `src/checkpoint.py` | 40-41 | eski 158 checkpoint yeni norm / ölçek anahtarını taşımaz → katı yükleme düşer; varsayılan KAPALI ya da şim (`src/moe.py:300-305` deseni) | [KİLİT] |
| `src/train.py` | 455 | resume yolu, aynı | [KİLİT] |
| `scripts/router_grad_diag.py` | 106-107 | `strict=False` + eksik anahtar beyaz listesi yalnız `attn_mask / rope_cos / rope_sin`: yeni parametre eski FOTON checkpoint'inde eksikse `sys.exit` | [KİLİT] |
| `scripts/router_probe_diag.py` | 88 | `strict=False`: yeni parametre sessizce init'te kalır | [TÜREV] |
| `scripts/v03_step1_gpu_two_route.py` | 62-63 | `assert not missing_keys and not unexpected_keys` (v02-X `last.pt`) | [KİLİT] |
| `scripts/architecture_as_built.py` | 43-46 | blok dökümü: `has_bias = any(p.dim() == 1 and "norm" not in name …)` — adında "norm" geçmeyen 1-D ölçek parametresi "bias" sayılır → belgenin "bias yok" iddiası ve `tests/test_architecture_doc.py:65` düşer | [TÜREV] |
| `scripts/mutation_catalog_06.py` | 40-42 (F05 → `model.py:128`) | SDPA çağrı satırı metin-çivili (`scale=` eklenirse satır değişir) | [KİLİT] |
| `tests/test_foton_model.py` | 256-270 (267) | elle attention `(q @ kᵀ) / 8 ** 0.5`: normsuz, 1/√hd ölçekli varsayım (FOTON kolu) | [TÜREV] |
| `tests/test_foton_model.py` | 409-435 | anahtar listesi + loss bit bit (10.0-C) | [KİLİT] |
| `tests/test_model.py` | 86-96, 105-117 | davranış eşikleri: init CE ≈ ln V ± 0.3; 150 adımda CE < 0.1 — QK-norm'lu config'te ölçülmedi | [TÜREV] |
| `tests/test_model.py` | 130-134, 163-166 | öğrenilir ölçek eklenirse parametre sayımı | [TÜREV] |
| `tests/test_flops.py` | 112-119 | `docs/flops.md` §5 oranları 0.32 / 0.36 / 1.00219 / 3006.6 (dense / moe preset'leri; ikincil terim ya da varsayılan değişirse) | [KİLİT] |
| `tests/test_flops.py` | 231-241, 244-258 | ikincil formülün FOTON-öncesi kopyası; FOTON olmayan preset'lerde bit bit | [KİLİT] |
| `tests/test_architecture_doc.py` | 41, 65 | `params_total == embedding + 8 · blok + norm`; dört kolda `not has_bias` | [KİLİT] (dört kol) |
| `docs/ARCHITECTURE_AS_BUILT.md` | 11-14, 20, 23-25 | ortak iskelet: RMSNorm (eps 1e-6), "hiçbir nn.Linear'da bias yok", init — QK-norm'suz | [KİLİT] (dört kol; yeni model için ayrı belge) |
| `docs/flops.md` | 44, 110, 303 | "1/√d_h ölçekleme, softmax ve RoPE eleman bazlıdır"; eleman bazlı işlem listesi; yönlendirilen katman softmax / ölçekleme satırı | [TÜREV] |
| `experiments/foton-v02/ox1_microbench.json` | 430-432 | O-X1 ölçümü QK-norm'suz çekirdekle ve o günkü `src/model.py` SHA'sıyla (donmuş ölçüm) | [KİLİT] |
| `experiments/v3-router-ablation/checks/fingerprint.py` + `fingerprint_*.json` | 6-11, 19-21 | attention yolundaki sayısal değişiklik parmak izini değiştirir | [KİLİT] |
| `checkpoints/**/best.pt`, `last.pt` (79 + 79) | — | yeni anahtarları taşımaz (katı yükleme) | [KİLİT] |

### 10.3 RoPE

Mevcut durum: RoPE ZATEN VAR — `precompute_rope` (`src/model.py:63-72`: `inv_freq = θ^(−arange(0, hd, 2)/hd)`, `t = arange(max_seq_len)`, float32, tablo (T, hd/2)), `apply_rope` (`75-85`: rotate-half, tablo `x.dtype`'a çevrilir → bf16), yalnız q ve k (`117`), buffer'lar `persistent=False` (`213-216`), `rope_theta = 10000.0` (`src/config.py:23`; 20 preset ve diskteki 160 `config.json`'da hep 10000.0 — sonda S5 / S6). Aşağıdaki liste (a) RoPE'nin ayarı / uygulaması / API'si değişirse, (b) uzun bağlamla ilişkisi.

**(a) RoPE'nin ayarı, uygulaması ya da API'si değişirse**

| dosya | satır(lar) | neden dokunulur | tür |
|---|---|---|---|
| `src/config.py` | 23 | `rope_theta: float = 10000.0` tek tanım, hiçbir preset ezmiyor; theta ya da ölçekleme alanı buraya | [ZORUNLU] (theta / ölçek değişecekse) |
| `src/model.py` | 63-72 | frekans tablosu (69), konum ekseni (70), `outer` (71); NTK / YaRN benzeri ölçekleme ya da theta burada | [ZORUNLU] |
| `src/model.py` | 75-85 | rotate-half düzeni (82-83); `cos / sin.to(x.dtype)` (84) → autocast altında döndürme bf16'da | [ZORUNLU] |
| `src/model.py` | 115-117 | yalnız q ve k'ye uygulanır | [ZORUNLU] |
| `src/model.py` | 213-216 | tablolar `d_model // n_head`, `max_seq_len`, `rope_theta`'dan; `persistent=False` → checkpoint'e girmez, her yüklemede config'ten yeniden hesaplanır | [ZORUNLU] |
| `src/model.py` | 384, 441 | `rope_cos[:T]` dilimleri (`foton_eval_all`, `forward`) | [ZORUNLU] (tablo biçimi değişirse) |
| `src/model.py` | 109, 183-184, 322-335, 337-372, 387-394 | `(x, cos, sin)` imzası `Block` / attention / `_foton_module` / `_foton_dispatch` / `_foton_trunk` / `foton_eval_all` boyunca elle taşınır; tabloların biçimi (ör. karmaşık ya da birleşik tensör) değişirse hepsi | [ZORUNLU] (API değişirse) |
| `src/model.py` | 4-5, 64-67, 78-80, 115-116 | docstring / yorumlar | [TÜREV] |
| `src/flops.py` | 201 | ikincil `rope_qk` | [TÜREV] |
| `src/checkpoint.py` | 40 | eski 158 checkpoint tablo saklamaz: theta `model_cfg`'de taşındığı için θ değişikliği eski kayda uygulanmaz (güvenli); uygulama config'siz değişirse eski ağırlıklar BAŞKA konum kodlamasıyla sessizce yüklenir | [KİLİT] |
| `scripts/router_grad_diag.py` | 47, 53-57, 107 | `model.rope_cos[:T]`, `blk(x, cos, sin)`, `_foton_module(m, x, cos, sin)`; beyaz listedeki `rope_cos` / `rope_sin` adları | [ZORUNLU] (API / buffer adı değişirse) |
| `scripts/router_probe_diag.py` | 104-106 | `model.rope_cos[: x.shape[1]]`, `blk(h, cos, sin)` | [ZORUNLU] (API değişirse) |
| `scripts/ox1_microbench.py` | 372-375, 257-258, 282, 405-415 | `precompute_rope(d // n_head, T, rope_theta)`; `ref.attn(z, cos, sin)`; `blk(h, cos, sin)`; `train_step / time_segment(…, cos, sin, …)` | [ZORUNLU] (API değişirse) |
| `scripts/mutation_catalog_06.py` 50-52 (F09 → `model.py:334`); `scripts/mutation_catalog_v03s1.py` 22-24 (V01 → `model.py:353`) | — | `cos, sin` argümanlı satırlar metin-çivili: imza değişirse mutant hedefi kaybolur | [KİLİT] |
| `tests/test_model.py` | 26, 51-69, 72-82 | `precompute_rope(hd, 64, 10000.0)`, `apply_rope(q, cos[m:m+1], …)`: göreli konum (64-65), norm koruma (69), konum kullanımı (72-82) | [TÜREV] (uygulama değişince yeniden geçmeli) |
| `tests/test_foton_model.py` | 208-211, 260-266 | `precompute_rope(hd, T, 10000.0)`, `attn(x, cos, sin)`, `apply_rope(q, cos, sin)` | [TÜREV] |
| `tests/test_foton_model.py` | 409-435 | loss `torch.equal`: RoPE'deki her sayısal fark (dtype dahil) kırar | [KİLİT] |
| `tests/test_ox1_microbench.py` | 171, 302, 506, 587 | `precompute_rope(…)` çağrıları; 587 `monkeypatch.setattr(foton_model, "precompute_rope", …)` (ad çivili) | [TÜREV] |
| `tests/test_router_grad_diag.py` | 26, 49, 55-58 | `GPT(cfg).double()` (tablolar float64'e çevrilir), `model.rope_cos[:48]`, `blk(x, cos, sin)` | [TÜREV] |
| `tests/test_flops.py` | 235 | `rope_qk` kopyası | [TÜREV] |
| `docs/ARCHITECTURE_AS_BUILT.md` 11-12; `docs/flops.md` 44, 110; `README.md` 9, 30, 53, 697 | — | "RoPE (θ = 10000; yalnızca q ve k'ye)" vb. metinler | [TÜREV] (`ARCHITECTURE_AS_BUILT` dört kol için [KİLİT]) |
| `PREREGISTRATION.md` | 607 | "eleman bazlı işlemleri (RMSNorm, RoPE, …)" (donmuş v01 ön-kaydı) | [KİLİT] |
| `experiments/v2-grouped-dispatch/diagnostics/stage_times.py` | 162 | `model.rope_cos`, `model.rope_sin` buffer adları (donmuş teşhis betiği) | [KİLİT] |
| `experiments/v3-router-ablation/checks/fingerprint.py` + `fingerprint_*.json` | 6-11, 19-21 | RoPE'deki sayısal değişiklik parmak izini değiştirir | [KİLİT] |
| `experiments/**/config.json`, `checkpoints/**/config.json` | ör. `experiments/v1-dense-moe-baseline/dense/config.json:9` `"rope_theta": 10000.0` | 213 kayıt θ = 10000 ile; yeni theta yalnız yeni koşularda | [KİLİT] |

**(b) Uzun bağlamla ilişkisi (kod değişmez, olgular)**

| dosya | satır(lar) | neden dokunulur | tür |
|---|---|---|---|
| `src/model.py` | 70, 213 | `t = arange(max_seq_len)`: tablo otomatik büyür; cos + sin float32, hd = 64: T = 512 → 131 072 B, 2 048 → 524 288 B, 4 096 → 1 048 576 B (sonda S5) | [TÜREV] (otomatik; sayı) |
| `src/model.py` | 69 | θ = 10000, hd = 64: en yavaş dönen çift (i = 31) için periyot 2π · 10000^(62/64) ≈ 47 100 token (formülden aritmetik; 2K-4K bu periyodun içinde kalır) | [TÜREV] (olgu) |
| `src/model.py` | 439, 527 | `assert T <= max_seq_len` (çıplak assert, `-O`'da atlanır); `generate` `idx[:, -max_seq_len:]` kırpması → kırpılan pencerede konumlar 0'dan başlar | [TÜREV] (denetim noktası) |
| `src/checkpoint.py` | 40 | eski checkpoint daha büyük `max_seq_len` ile kurulursa tablo config'ten büyür: eğitimde görülmemiş konumlar | [KİLİT] (eski ağırlıkların kapsamı) |

### 10.4 2K-4K bağlam

Mevcut durum: `max_seq_len = 512` (`src/config.py:16`); 20 preset'in 17'si devralır, 3 tiny 128 (sonda S5); kampanya 16 × 4 × 512 = 32 768 token/adım; bütün ölçümler, maske / FLOP pinleri ve dondurulmuş val kümesi T = 512'de.

| dosya | satır(lar) | neden dokunulur | tür |
|---|---|---|---|
| `src/config.py` | 16 | varsayılan; yeni T buraya ya da yeni preset'e | [ZORUNLU] |
| `src/config.py` | 144, 278, 284 | tiny preset'leri 128 (bağımsız) | [TÜREV] |
| `src/config.py` | 80-86, 180-182 (ve foton preset'leri 188-275) | `batch_size × grad_accum × max_seq_len` = token/adım (`src/train.py:433`): 16 × 4 × 2 048 = 131 072, × 4 096 = 262 144; `max_steps / warmup_steps / eval_interval` adım cinsinden → bütçe T ile ölçeklenir (`src/train.py:35-41`) | [ZORUNLU] (yeni preset'in bölüşümü / bütçesi) |
| `src/config.py` | 81-84, 86, 109-110, 114-115, 122, 174-177 | VRAM ölçümleri "seq 512"; "2 x 32 x 512 = 32 768"; `eval_iters × batch` pencere (eval token'ı T ile büyür); "token 6..511", "log(3)/512"; "32.768 SABIT (16 x 4 x 512)"; `loss_prefix_skip` 64 T = 512'ye göre | [TÜREV] |
| `src/train.py` | 403-404, 408-409, 433, 436 | `block_size = mcfg.max_seq_len`; `tokens_per_step`; `max_steps = ceil(max_tokens / tps)` — config'ten izler, sayılar değişir | [TÜREV] (otomatik; denetim noktası) |
| `src/train.py` | 449-454 | resume: checkpoint'in `tokens_per_step`'i eşit olmalı → eski T'li `last.pt`'den yeni T'ye devam edilemez | [KİLİT] (doğrulayıcı) |
| `src/train.py` | 285-288, 293-297, 304 | val kümesi `frozen_val_windows(batch × eval_iters, 12345)` → `randint(len − T)`: başka T = BAŞKA pencere kümesi ve başka `val_windows_sha256` | [KİLİT] (dondurulmuş val kümesi) |
| `src/train.py` | 204, 93, 266-269, 280-283 | `loss_prefix_skip = k` mutlak pozisyon: T büyüyünce atlanan pay küçülür (FOTON ölçütü 64..T−1) | [TÜREV] (FOTON) |
| `src/train.py` | 233, 245, 588-590 | `eval_raw` dökümü `ce_tok [B, 3, T]` float16 → T ile orantılı (FOTON) | [TÜREV] (FOTON) |
| `src/train.py` | 57-58 | docstring "(tüm presetler: 32 768, 2 048)" | [TÜREV] |
| `src/train.py` | 600-604 | örnekleme `generate` KV-cache'siz (`src/model.py:523-528`): uzun T'de her eval'de bağlam baştan işlenir | [TÜREV] (maliyet olgusu) |
| `src/data.py` | 24, 27-28, 37-38 | `block_size`; `len < T + 1` hatası; `randint(len − T)` | [TÜREV] (otomatik) |
| `src/data_multi.py` | 34, 37-38, 56-57, 63, 69-70, 104-105, 125-126, 141-142 | `block_size`; `rows (B, T+1) uint16`; `hi = len − T` → aynı seed'de pencere KİMLİKLERİ değişir (eğitim akışı ve val kümesi) | [KİLİT] (val kümesi) / [TÜREV] (otomatik) |
| `src/data_multi.py` | 93, 113-114, 133-135 | `REF_BATCH = 16`; "adım başına 64 dizi, eval'de 1280"; `frozen_val_windows(1280, 12345)`: T = 4 096'da 1 280 × 4 096 = 5 242 880 val token çekimi, alan başına val yalnız 2 000 000 token (`data/foton/meta.json:11, 19, 28`) | [TÜREV] (olgu) |
| `src/model.py` | 439, 441, 384, 527 | T sınırı, tablo dilimleri, üretim kırpması (otomatik) | [TÜREV] (otomatik) |
| `src/model.py` | 63-72, 213-216 | RoPE tabloları `max_seq_len` ile büyür (10.3-b) | [TÜREV] (otomatik) |
| `src/model.py` | 523-528 | KV-cache yok; docstring "512 token … yeterince hızlı" | [TÜREV] |
| `src/model.py` | 101-102, 128 | FOTON maskesi `[n_head, T, T]` bool buffer: 2 097 152 B (512) → 33 554 432 B (2 048) → 134 217 728 B (4 096) modül katmanı başına, 6 modül katmanı (sonda S5) | [TÜREV] (FOTON; otomatik büyür, bellek olgusu) |
| `src/model.py` | 42-43, 283-284, 298-302, 311 | 128.4 ± 0.4 kilidi YALNIZ `max_seq_len == T_KAMPANYA`: başka T'de model itirazsız kurulur, A/B/C eşitliği bozulur (T = 4 096'da 147.3 / 240.1 / 236.1: `experiments/foton-v02/v03_design_notes.md:55`) | [KİLİT] (FOTON; bilerek kırılır ya da yeniden eşitlenir) |
| `src/model.py` 315-316; `src/config.py` 58 | — | prefix k = 64 sabit (T'den bağımsız) | [TÜREV] (FOTON) |
| `src/attn_patterns.py` | 38-39, 43-50 | `T_KAMPANYA = 512`, `N_HEAD_KAMPANYA = 8`; W / r / s sabitleri T = 512 için eşitlenmiş; `EQ_TOL = 0.35` | [KİLİT] (FOTON) |
| `src/attn_patterns.py` | 8-21, 205-309 | docstring sayıları T = 512; `T: int = T_KAMPANYA` varsayılan argümanları | [TÜREV] (FOTON) |
| `src/flops.py` | 110-129 (124-128) | FLOP tarafındaki pin yalnız `T == T_KAMPANYA` | [KİLİT] (FOTON) |
| `src/flops.py` | 161, 170, 177, 193, 196, 221-241, 264, 298, 326-331, 342-347 | T-parametrik (`seq_len or cfg.max_seq_len`): kod değişmez, sayılar değişir — `foton-d` (V = 32 768): `attn_context` 4 202 496 → 16 785 408 → 33 562 624; eğitim/token 267 411 456 → 305 160 192 → 355 491 840 (sonda S5) | [TÜREV] (sayı) |
| `scripts/verify_dispatch.py` | 243-244 | `TokenDataset(…, 512, "cuda")`, `get_batch(32, …)` (v2 kaydının betiği) | [KİLİT] |
| `scripts/architecture_as_built.py` | 100, 111-112, 119, 132, 146-147 | `MultiDomainDataset(…, 512, "cpu")` iki kez; `gap < 512`; `× 512 / corpus`; `tokens_per_step` hesabı | [TÜREV] (dört kolun belgesi) |
| `scripts/v03_step1_gpu_two_route.py` 54; `scripts/v03_step1_replay_streams.py` 41 | — | `MultiDomainDataset(…, 512, …)` sabit | [KİLİT] (v03 adım-1 betikleri) |
| `scripts/v03_dcheck.py` | 183 | `tps = bs · ga · max_seq_len` → takvim tahmini referansla birebir | [KİLİT] |
| `scripts/ox1_microbench.py` | 72, 80, 125-131 | `FROZEN_DIMS max_seq_len 512`, `PINNED T`; `(T_KAMPANYA, N_HEAD_KAMPANYA) != (max_seq_len, n_head)` → RED | [KİLİT] |
| `scripts/compare_runs.py` 196; `scripts/baseline_reference.py` 41; `scripts/calibrate_v02.py` 34; `scripts/power_analysis.py` 38; `scripts/v03_step1_analyze.py` 65 | — | `tps / STEP / TOK_PER_STEP / TOKENS_PER_STEP = 32768` (donmuş kampanya analizleri; yeni T'li koşulara uygulanamaz) | [KİLİT] |
| `scripts/v03_step1_analyze.py` | 65-67, 71 | `SEQS_PER_STEP 64`, `VAL_DOMAIN_COUNTS [431, 434, 415]`, `VAL_WINDOWS_SHA256` (1 280 pencerenin içerik özeti) | [KİLİT] |
| `scripts/vram_ladder.py` 6, 88; `scripts/router_analysis.py` 345; `scripts/verify_dispatch.py` 31; `scripts/vram_probe_foton.py` 68 | — | "dizi 512" metinleri / yorumlar ("16 x 512 x 32768 x 2 B") | [TÜREV] |
| `scripts/eval_domains.py` | 9, 60 | "alan başına 1280 pencere" (eval token'ı T ile büyür) | [TÜREV] |
| `scripts/topology_overlap.py` | 71 | `--T` varsayılanı "512,2048,4096,8192" (uzun T zaten sayılıyor) | [TÜREV] |
| `tests/test_foton_frozen.py` | 102-103, 160 | `bs · ga · max_seq_len == 32_768` PİNLİ; `bs · eval_iters == 1280`; S1 `max_seq_len 512` | [KİLİT] |
| `tests/test_foton_model.py` | 34-36, 78, 83, 121-122 | `T, K = 512, 64`; preset `max_seq_len == 512`; maske (8, T, T); 128.4 ± 0.4; `ModelConfig`'te window / dilat / stride / local_w alanı OLAMAZ | [KİLİT] (FOTON) |
| `tests/test_attn_patterns.py` | 29, 32-39, 48-52, 95-115, 119-134 | `T_KAMPANYA`'daki başlık sayıları 256.50 / 128.17 / 128.50 / 128.38; yayılım 167/512; dilated tavanı | [KİLİT] (FOTON) |
| `tests/test_architecture_doc.py` | 68-76, 85-91, 102-120 | `masks(512, 8)` sayımları; `batch_size · grad_accum · 512`, "32,768", "143,360,000", "64..511"; veri akışı olguları (140 032 / 25 / 431-434-415) | [KİLİT] |
| `tests/test_v03_step1.py` | 389, 652-669 (669) | `tps = bs · ga · 512`; `bs · ga · max_seq_len == TOKENS_PER_STEP` | [KİLİT] |
| `tests/test_flops.py` | 82-89, 279, 418-423, 461-471, 479-485, 584, 632 | `attn_context` T'li kapalı form; `foton_router == 6` (T = 512); S1 `8 · 2 · 640 · 513`; docs başlığı "(V=32768)" ve "T = 512" metni | [KİLİT] (FOTON preset'leri) |
| `tests/test_train.py` | 37-38, 77, 145, 182 | token/adım 2'nin kuvveti olmalı (2 048 / 4 096 için de doğru); `32_768`, `98_467_840`; tiny `4 · 1 · 128` | [TÜREV] |
| `tests/test_ox1_microbench.py` | 47, 159, 629 | `T, N_HEAD = 512, 8`; `PINNED["T"] == 512`; `--T / --max_seq_len / --n_head` bayrağı YOK | [KİLİT] |
| `tests/test_topology_overlap.py` | 19, 38-40 | T 64 / 512 / 700; `analyze(512)` → 128.1738 / 128.5 / 128.375 | [KİLİT] (FOTON) |
| `docs/ARCHITECTURE_AS_BUILT.md` | 3-4, 64, 71, 89, 113, 116, 126-131 | "T = 512"; [8, 512, 512]; "512 pozisyon"; "32,768 token (16 × 4 × 512)"; "512 token'lık pencere"; ölçüt 64..511; `ce_tok [1280, 3, 512]` | [KİLİT] (dört kol, test kilitli) |
| `docs/flops.md` | 13, 42, 99-103, 150, 202, 236 | T 512; 2d(T+1) = 525 312; "3000 adım × 32,768"; §8 "T = 512"; "(T=512, n_head=8)"; "d = T = 512'de" | [TÜREV] (§2-7) / [KİLİT] (§8) |
| `README.md` | 147, 167, 230, 282, 670-677 | "3000 adım x 32 768 token"; "seq 512"; "32 × 512 token"; VRAM tablosu (dense 30M, seq 512) | [KİLİT] (donmuş kayıt) / [TÜREV] (VRAM notu) |
| `PREREGISTRATION.md` 31, 153; `PREREGISTRATION_V02.md` 184-187, 2023 | — | "32.768 token/adım", "512, 32.768 (batch 32 × grad_accum 2 × 512)"; "T = 512'de … en çok 128.50"; "her pencere 512 token'lık sabit bir kesittir" | [KİLİT] |
| `experiments/foton-v02/v03_design_notes.md` | 17-20, 25, 55, 77-81 | uzun bağlam yönü notu: "T ≥ 4096 … maliyeti ÖLÇÜLMEDİ"; T = 4 096'da A/B/C 147.3 / 240.1 / 236.1; "korpusun yeniden kurulması gerekir (TRIPWIRE: yeni deney)" | [KİLİT] (not; dokunulmaz) |
| `experiments/v2-grouped-dispatch/diagnostics/stage_times.py` | 28 | `TokenDataset(…, 512, dev)` (donmuş teşhis betiği) | [KİLİT] |
| `experiments/foton-v01/vram_ladder*.json`; `experiments/foton-v02-prereg/vram_ladder_mb16_32k.json`, `lstar.json`, `baseline_reference.json`; `experiments/foton-v02/vram_probe.json`, `ox1_microbench.json` | — | T = 512 ölçümleri ve eğrileri; yeni T ile karşılaştırılamaz (varlıkları listelendi; içerikleri bu turda okunmadı) | [KİLİT] |
| `experiments/**/config.json` (53), `checkpoints/**/config.json` (160) | ör. `experiments/v1-dense-moe-baseline/dense/config.json:4` `"max_seq_len": 512` | diskte 158 kayıt 512, 2 kayıt 128 (sonda S6) | [KİLİT] |

Veri notu: `data/foton` alan başına val 2 000 000 token (`meta.json:11, 20, 28`); dokümanlar `.bin`'de uç uca ve EOS'suz (10.6) → pencere doküman sınırını geçer; 2K-4K için doküman uzunluğu dağılımı ölçülmedi. Dokunmak gerekmeyen (T'yi `cfg.max_seq_len`'den okuyan): `scripts/benchmark.py:42-58`, `scripts/analyze_experts.py:65, 78`, `scripts/eval_domains.py:68`, `scripts/flops_report.py:39, 50`, `scripts/router_analysis.py:357`, `scripts/router_grad_diag.py:113`, `scripts/router_probe_diag.py:97`, `scripts/vram_probe_foton.py:50-51, 78`, `scripts/vram_ladder.py:41-42`.

### 10.5 Yeni tokenizer

Mevcut durum: SentencePiece BPE V = 32 768; `split_digits=True` ZATEN (`src/tokenizer.py:68`; `'2024'` → `['▁', '2', '0', '2', '4']`, sonda S4); `byte_fallback=True` (66); özel token yalnız unk 0 / bos 1 / eos 2, pad −1 (72); `user_defined_symbols` / `control_symbols` verilmiyor; Almanca / Türkçe harf içeren parça 0 (`'Größe für Straße'` → ö / ß / ü UTF-8 byte'larına düşer, 14 token); `'<tool_call>'` 5 parça; `.bin` dtype uint16; 20 preset'te `data_dir` {sp8k: 7, foton: 13}.

| dosya | satır(lar) | neden dokunulur | tür |
|---|---|---|---|
| `src/tokenizer.py` | 23-35 | `Tokenizer` arayüzü yalnız `vocab_size / bos_id / eos_id / encode / encode_batch / decode`: pad ya da özel-token (araç çağrısı) API'si yok | [ZORUNLU] |
| `src/tokenizer.py` | 38-55 | `SentencePieceTokenizer`: id'ler modelden (43-45); `encode` BOS / EOS eklemez (ekleme çağıranda) | [ZORUNLU] |
| `src/tokenizer.py` | 57-77 | `train()`: bpe (64), `byte_fallback` (66), `character_coverage=0.9995` (67), `split_digits=True` (68), identity (69), unk / bos / eos / pad (72), `input_sentence_size=2_000_000` (73); `user_defined_symbols` / `control_symbols` argümanı YOK | [ZORUNLU] |
| `src/tokenizer.py` | 100-108 | `load_tokenizer`: `meta.json` → tokenizer türü, `model_file` / `name`; başka tür `ValueError` | [ZORUNLU] (yeni tür ya da meta alanı gelirse) |
| `src/tokenizer.py` | 80-97 | tiktoken alternatifi (`bos == eos == eot`) | [TÜREV] |
| `src/tokenizer.py` | 1-16, 25-26 | docstring ("ticari … 32k-160k çok dilli sözlük gerekir"); bos / eos yorumları | [TÜREV] |
| `src/data.py` 6, 24; `src/data_multi.py` 33, 63; `scripts/corpus_report.py` 80, 106, 115 | — | `np.uint16` memmap / tampon: yalnız V ≥ 65 536 olursa dtype değişir | [ZORUNLU] (koşullu: V ≥ 65 536) |
| `src/data_multi.py` | 32, 150-160 | alan adları `sorted(paths)` → kimlikler (code 0 / math 1 / nl 2); Almanca ayrı alan olarak eklenirse kimlikler kayar; `<alan>_<split>.bin` + meta `domains` | [ZORUNLU] (yeni alan eklenirse) |
| `src/train.py` | 393-394 | `vocab_size` tokenizer'dan okunur (model boyutu otomatik) | [TÜREV] (otomatik) |
| `src/train.py` | 397-401 | `meta.json`'da `"domains"` → çok alanlı yol | [TÜREV] (otomatik) |
| `src/train.py` | 110-112, 427-428 | `make_assigner("domain")`: alan sayısı == modül sayısı (3) zorunlu → 4. alan `ValueError` | [KİLİT] (FOTON v03 kolları; yeni alan eklenirse) |
| `src/train.py` | 534-537, 553 | `RouterLog` / `AssignLog` ve `data_domain_counts` alan sayısına bağlı | [TÜREV] |
| `src/train.py` | 600-608 | örnek üretimi `tok.bos_id` ile başlar, `eos_id`'de durur, `tok.decode` | [ZORUNLU] (BOS / EOS / özel token sözleşmesi) |
| `src/sample.py` | 37, 42, 46, 49-51 | tokenizer `ckpt["train_cfg"]["data_dir"]`'den; `[bos_id] + encode(prompt)`; EOS'ta kesme — dizin yerinde değişirse eski checkpoint yeni sözlükle çözülür | [ZORUNLU] |
| `src/model.py` | 197, 203, 211 | `tok_emb` / `lm_head` V × d, bağlı (otomatik) | [TÜREV] |
| `src/model.py` 468, 396-397; `src/train.py` 267, 282 | — | `ignore_index=-1`: pad token (id ≥ 0) eklenirse loss maskesi buna bağlı değil | [ZORUNLU] (pad eklenirse) |
| `src/model.py` | 521, 540-545 | `generate` tek `eos_id`'de durur (araç-çağrısı sonu gibi ikinci durdurma token'ı yok) | [ZORUNLU] (özel token'larla üretim) |
| `src/config.py` | 15, 78, 174 | `vocab_size=8192` + "tokenizer'dan okunur" yorumu; `data_dir="data/sp8k"`; "(32768)" yorumu | [TÜREV] |
| `src/flops.py` | 161, 184, 209, 286, 351-352 | V; `lm_head 2dV`; `cross_entropy 3V` (otomatik); `foton_doc_tables(vocab_size=32768)`; "V=8192'de ölçüldü" | [TÜREV] |
| `scripts/prepare_foton_corpus.py` | 43-48, 163 | `SOURCES` 4 İngilizce kaynak (caselaw, hansard, stackv2_edu, arxiv): Almanca kaynak yok; bayt/token 5.5 varsayımı | [ZORUNLU] (veri) |
| `scripts/prepare_foton_corpus.py` | 54-81 | `tier_a` lisans süzgeci (yeni kaynak da doküman başına lisans taşımalı — `PREREGISTRATION_V02.md:146-147`) | [ZORUNLU] |
| `scripts/prepare_foton_corpus.py` | 149-156 | `--out data/foton` (YERİNDE ezer), `--vocab_size 32768`, `--tok_sample_mb_per_domain 120` | [ZORUNLU] |
| `scripts/prepare_foton_corpus.py` | 189-209 | mevcut `tokenizer.model` varsa YENİDEN EĞİTİLMEZ (190); alan-dengeli örnek (192-205); `train(sample, prefix, vocab)` özel token geçiremez (207) | [ZORUNLU] |
| `scripts/prepare_foton_corpus.py` | 213-216, 227, 241 | `meta.json` yalnız bos / eos; `np.uint16` | [ZORUNLU] |
| `scripts/prepare_data.py` | 59, 65-76, 92-93, 105, 117-120, 127-128, 134-137 | TinyStories yolu: `assert vocab < 2**16`; `[BOS] + ids + [EOS]`; `--vocab_size 8192`; mevcut model V'si farklıysa çıkar; round-trip; meta | [ZORUNLU] (bu yol kullanılacaksa) |
| `scripts/corpus_report.py` | 72-75, 93-101, 110, 132 | `tokenizer.model` sabit adı; `_raw` + `DOC_SEP` round-trip; `tok.sp.id_to_piece`; varsayılan `--out experiments/foton-v02-prereg/corpus_report.json` = DONMUŞ artifact'i ezer | [ZORUNLU] / [KİLİT] (varsayılan `--out`) |
| `scripts/preset_hash.py` | 32, 42, 72-77 | `VOCAB = 32768` hash'e girer; `data/foton/meta.json vocab_size != 32768` → `sys.exit` | [KİLİT] |
| `scripts/architecture_as_built.py` | 25, 34, 84 | `VOCAB = 32768` → belge sayıları | [TÜREV] |
| `scripts/verify_pins.py` | 86, 114-128, 274-289 | `DATA_DIR = "data/foton"`; 9 korpus dosyası çivili; dizine eklenen çivisiz her dosya `--verify`'ı düşürür | [KİLİT] |
| `scripts/baseline_reference.py` 75; `scripts/calibrate_v02.py` 67; `scripts/check_prereg_numbers.py` 125 | — | `vocab_size=32768` ile donmuş FLOP | [KİLİT] |
| `scripts/vram_ladder.py` | 81, 98-99 | `--vocab_size` "kampanya icin 32768" | [TÜREV] |
| `scripts/analyze_experts.py` 45, 99; `scripts/router_analysis.py` 344, 366, 412; `scripts/vram_probe_foton.py` 43, 68 | — | `load_tokenizer(ckpt data_dir)`; `tok.sp.id_to_piece` (SentencePiece'e özel); `--data_dir data/sp8k`; `tok.decode([i])`; logit tensörü yorumu "16 x 512 x 32768 x 2 B" (V ile büyür) | [TÜREV] |
| `tests/test_train.py` 95-96; `tests/test_foton_train.py` 140-141; `tests/test_foton_assign.py` 150-151; `tests/test_router.py` 203 | — | `_StubTokenizer` (vocab 64, bos 1, eos 2 + decode): `train.main()` yeni bir tokenizer niteliği kullanırsa dördü kırılır | [TÜREV] |
| `tests/test_train.py` 110; `tests/test_data_multi.py` 35; `tests/test_foton_frozen.py` 39; `tests/test_model.py` 199; `tests/test_foton_assign.py` 163, 326; `tests/test_router.py` 200; `tests/test_foton_train.py` 153 | — | fixture'lar `np.uint16` yazar (dtype değişirse) | [TÜREV] (koşullu) |
| `tests/` | — | `src/tokenizer.py`, `scripts/prepare_data.py`, `scripts/prepare_foton_corpus.py`, `scripts/corpus_report.py` için test YOK (Grep: hiçbir test bu modülleri anmıyor) | [TÜREV] (olgu) |
| `tests/test_flops.py` | 140, 207-214, 584, 632 | `V_FOTON = 32768`; donmuş D/M FLOP (V = 32 768) + PREREG metni; "## 8. FOTON v0.2 (V=32768)"; "V = 32768" | [KİLİT] |
| `tests/test_v03_step1.py` | 198, 660 | `vocab_size = 32768` | [KİLİT] |
| `tests/test_architecture_doc.py` | 31-43, 109, 119 | 16 777 216 embedding vb. sayılar; `domain_names_in_id_order == ["code", "math", "nl"]`; "code = 0 / math = 1 / nl = 2" | [KİLİT] |
| `docs/ARCHITECTURE_AS_BUILT.md` | 3-4, 13, 29-30, 40, 54, 60, 115 | `vocab_size = 32768`; 16 777 216; toplamlar; "code = 0 / math = 1 / nl = 2" | [KİLİT] (dört kol) |
| `docs/flops.md` | 14, 61, 145, 150, 263, 364-365 | V 8192; 2dV = 8 388 608; "(V=32768)"; `lm_head` 33 554 432; "V=8192'de ölçüldü" | [TÜREV] (§2-7) / [KİLİT] (§8) |
| `README.md` | 10, 29, 76, 147, 228, 666 | "kendi BPE tokenizer'ımız"; "8k'lık BPE tokenizer eğit"; "V=8192" (donmuş kayıt); "çok dilli veri ve tokenizer" (sonraki adım) | [TÜREV] / [KİLİT] (147, 228) |
| `PREREGISTRATION.md` 152; `PREREGISTRATION_V02.md` 19, 138-147, 417-424, 435-446 | — | "`data/sp8k`, SentencePiece 8k"; "`data/sp8k` tokenizer'ına çivilemişti"; TRIPWIRE (kaynak / oran değişirse yeni deney; Tier A doküman başına); korpus SHA'ları; `_raw` çivisiz | [KİLİT] |
| `requirements.txt` | 9-10 | `sentencepiece`, `tiktoken` sürümsüz | [TÜREV] |
| `data/foton/` `meta.json`, `tokenizer.model`, `tokenizer.vocab`, 6 `.bin` | `meta.json:2-6` (tokenizer / model_file / vocab 32768 / bos 1 / eos 2) | `pins.json:14-68` ile çivili (39 dosyanın 9'u) | [KİLİT] |
| `data/foton/_raw/{caselaw, hansard, stackv2_edu, arxiv}.txt` (+ `.json.gz`) | — | yeniden tokenizasyonun girdisi; çivisiz (`PREREGISTRATION_V02.md:444-446`); `extract()` boyut ≥ %95 ise yeniden kullanır (`scripts/prepare_foton_corpus.py:119-122`) | [ZORUNLU] (girdi) |
| `data/sp8k/tokenizer.model`, `train.bin`, `val.bin` | `experiments/v1-dense-moe-baseline/checksums.txt:8` | v1 kaydının hash'li parçası; 120 kayıt `data_dir = data/sp8k` (sonda S6) | [KİLİT] |
| `experiments/foton-v02-prereg/corpus_report.json` | 180-210 | min id 4 / 12 / 13 (EOS'suz, bu tokenizer'la kayıt); `corpus_report.py` varsayılan çıktısı burası | [KİLİT] |
| `experiments/foton-v02-prereg/lstar.json`, `baseline_reference.json` | — | val loss birimi (nat/token) tokenizer'a bağlı; L* taşınmaz (`PREREGISTRATION_V02.md:140-143`) | [KİLİT] |
| `scripts/v03_step1_analyze.py` | 67, 71 | `VAL_DOMAIN_COUNTS` / `VAL_WINDOWS_SHA256`: `data/foton` yeniden tokenize edilirse değişir | [KİLİT] |
| `checkpoints/**/best.pt`, `last.pt` (79 + 79) | — | embedding satırları eski token id'leriyle; `config.json`'daki `data_dir` (120 sp8k / 40 foton) + `vocab_size` → eski ağırlıklar eski sözlüğe bağlı | [KİLİT] |

### 10.6 EOS düzeltmesi

Mevcut durum: `data/foton` altı `.bin`'de id ≤ 2 (unk / bos / eos) hiç yok (sonda S3: 306 000 000 token, 0); kök neden `scripts/prepare_foton_corpus.py:90` (`line.strip() == DOC_SEP`: `"\x1e".isspace()` True, `strip()` ayırıcıyı siler, karşılaştırma hiç doğru olmaz); ayırıcı satır metne katılıp `[13, 33, 13]` (`<0x0A> <0x1E> <0x0A>`) olarak 127 141 kez veride (code_train 101 140 / code_val 1 962 / math_train 6 354 / math_val 113 / nl_train 17 539 / nl_val 33).

| dosya | satır(lar) | neden dokunulur | tür |
|---|---|---|---|
| `scripts/prepare_foton_corpus.py` | 90 | kök neden: karşılaştırma hiç doğru olmaz → kaynak dosyası TEK doküman, EOS yalnız dosya sonuna gelir ve quota kırpmasıyla (230-231) düşer | [ZORUNLU] |
| `scripts/prepare_foton_corpus.py` | 51, 84-102 | `DOC_SEP = "\x1e"`; `batched_docs` ayırıcı satırı metne KATAR (98) → `.bin`'lerdeki `[13, 33, 13]` | [ZORUNLU] |
| `scripts/prepare_foton_corpus.py` | 136 | yazıcı `t + "\n" + DOC_SEP + "\n"` (yalnız `\x00` temizlenir; metin içindeki `\x1e` korunur) | [ZORUNLU] (ayırıcı sözleşmesi) |
| `scripts/prepare_foton_corpus.py` | 200 | tokenizer örneğinde `line.strip() and line.strip() != DOC_SEP`: ikinci koşul ölü, ayırıcı satır ilk koşul (`strip()` boş) sayesinde doğru atlanır — aynı hata sınıfı, davranış doğru | [TÜREV] |
| `scripts/prepare_foton_corpus.py` | 225-236 (227, 230-231) | EOS yalnız doküman sonuna (227), BOS eklenmez (`scripts/prepare_data.py:69-72` BOS + … + EOS ekler); `a[: quota − src_got]` doküman ortasında keser | [ZORUNLU] (BOS / EOS sözleşmesi) |
| `scripts/prepare_foton_corpus.py` | 241-245, 218-219 | val = akışın SON `nval` token'ı; düzeltme akışı kaydırır (doküman başına 3 ayırıcı token → 1 EOS) → train / val sınırı başka metne düşer | [ZORUNLU] (denetim noktası) |
| `scripts/prepare_foton_corpus.py` | 119-122, 149-150, 190 | `_raw/*.txt` varsa yeniden kullanılır (diskte var); `--out data/foton` yerinde ezer; mevcut `tokenizer.model` yeniden eğitilmez (id'ler sabit kalır) | [ZORUNLU] / [KİLİT] |
| `scripts/prepare_foton_corpus.py` | 9-10 | docstring "Doküman sınırları EOS ile işaretli" (şu an yanlış) | [TÜREV] |
| `scripts/corpus_report.py` | 76-84 | doğrulama yalnız `max < V` sınar; EOS varlığını sınamaz (hatayı kaçıran denetim) | [TÜREV] (olgu) |
| `scripts/corpus_report.py` | 96, 132 | round-trip metninden `DOC_SEP + "\n"` siliniyor (ayırıcı sözleşmesine bağlı); varsayılan çıktı DONMUŞ `corpus_report.json` | [TÜREV] / [KİLİT] |
| `scripts/prepare_data.py` | 27, 44-54, 65-76 | TinyStories yolu: `DELIM = b"<…endoftext…>"` ayırıcısıyla byte düzeyinde bölme, BOS + EOS ekleme — düzgün çalışan karşı örnek | [TÜREV] (referans) |
| `src/data_multi.py` | 7-9 | docstring "Doküman sınırları tokenize sırasında EOS ile işaretlendi" (şu an YANLIŞ; düzeltmeden sonra doğru) | [TÜREV] |
| `src/data.py` | 4-5, 11-12 | `[BOS] hikâye [EOS]` biçimi (sp8k için doğru, foton için değil) | [TÜREV] |
| `src/tokenizer.py` | 25-26 | bos / eos yorumları | [TÜREV] |
| `src/sample.py` | 40-42, 46, 49-51 | "Eğitimde her hikâye BOS ile başladığı için" → `[bos_id] + encode`: `data/foton` modellerinde BOS hiç görülmedi; düzeltme yalnız EOS eklerse BOS öneki yine görülmemiş token | [ZORUNLU] (BOS sözleşmesi) |
| `src/train.py` | 600-604 | örnekleme `tok.bos_id` ile başlar, `eos_id`'de durur (13 foton preset'inde `sample_tokens=0` ile kapalı: `src/config.py:181` vd.) | [ZORUNLU] (aynı) |
| `src/model.py` | 540-545 | EOS'ta durma: veri EOS'suzken etkisiz (kod değişmez) | [TÜREV] (olgu) |
| `src/train.py` | 290-297, 304 | `val_windows_sha256` pencerelerin TOKEN İÇERİĞİ özeti → yeniden üretilen `.bin`'de değişir (indisler aynı kalsa bile) | [KİLİT] |
| `scripts/v03_step1_analyze.py` | 67-71, 119-121 | `VAL_WINDOWS_SHA256 = 9773961…`, `VAL_DOMAIN_COUNTS [431, 434, 415]` çivili; yeni koşular eski özeti vermez | [KİLİT] |
| `scripts/v03_step1_gpu_two_route.py` | 54, 62-63, 78, 82 | v02-X `last.pt` ile val pencerelerinde oracle'ı yeniden üretme + `val_windows_sha256` | [KİLİT] |
| `scripts/architecture_as_built.py` | 88-135 | akış olguları: uzunluklar aynı kalırsa (quota 100M / 2M tam dolu) pencere KİMLİKLERİ aynı, içerik farklı (koşulmadan doğrulanamaz) | [TÜREV] |
| `scripts/verify_pins.py` | 114-128, 274-289, 294, 319-325 | 6 `.bin` + `meta.json` SHA; değişince `--verify` FAIL; yeniden çivileme `--force` ister | [KİLİT] |
| `tests/test_architecture_doc.py` | 102-120 | veri akışı sayıları (140 032 / 25 / 431-434-415) `data/foton` üzerinden; `data/foton` yokken atlanır (106-107) | [TÜREV] |
| `tests/` | — | `batched_docs` / `DOC_SEP` için test YOK (Grep) | [TÜREV] (olgu) |
| `docs/ARCHITECTURE_AS_BUILT.md` | 116 | "veride doküman sınırı / EOS token'ı YOKTUR" — düzeltmeden sonra bayatlar (test bu cümleyi pinlemiyor) | [TÜREV] |
| `PREREGISTRATION_V02.md` | 76 | "Doküman sınırları EOS ile işaretli" (gövde: yanlış ama donmuş) | [KİLİT] |
| `PREREGISTRATION_V02.md` | 2019-2026 (S6) | "id ≤ 2 … TEK bir token yok. Korpus yeniden kurulmaz (TRIPWIRE)" | [KİLİT] |
| `PREREGISTRATION_V02.md` | 138-147 | TRIPWIRE: dağılım değişince L* taşınmaz → EOS'lu korpus yeni deney | [KİLİT] |
| `experiments/foton-v02/gate_report_06.md` 140; `experiments/foton-v02/v03_design_notes.md` 80-81 | — | "korpusta EOS yok (S6 …)"; "korpusta doküman sınırı YOK (Sapma-S6) … korpusun yeniden kurulması gerekir (TRIPWIRE)" | [KİLİT] |
| `experiments/foton-v02-prereg/corpus_report.json` | 180-210 | min id 4 / 12 / 13 (EOS'suz korpusun kaydı) | [KİLİT] |
| `experiments/foton-v02-prereg/pins.json` 15-20, 33-68; `PREREGISTRATION_V02.md` 417-424 | — | `meta.json` + 6 `.bin` hash'leri: yerinde yeniden üretim = 7 FAIL | [KİLİT] |
| `data/foton/{code, math, nl}_{train, val}.bin` | — | yeniden üretilmesi gereken 6 artefakt (306 000 000 token) | [ZORUNLU] (veri) + [KİLİT] (çivili) |
| `data/foton/meta.json` | 5-6, 35-72 | `bos_id 1`, `eos_id 2` doğru ama veride yok; `licensing` `kept_docs 0`, `reused true` (yeniden üretilince değişir) | [TÜREV] + [KİLİT] (çivili) |
| `data/foton/_raw/*.txt` | — | ayırıcı satırlı ham metin: düzeltmenin girdisi (çivisiz) | [ZORUNLU] (girdi) |
| `checkpoints/foton-v02*`, `foton-v03-step1`, `foton-v03-dcheck` | — | 40 kayıt (sonda S6: `vocab_size` 32768 × 40) EOS'suz veriyle eğitildi; L* bu veriye bağlı; düzeltilmiş veriyle karşılaştırılamaz | [KİLİT] |

### 10.7 Dokunulmaması gereken dondurulmuş kayıtlar

- git tag'leri: `v1-dense-moe-baseline`, `v3-router-ablation`, `foton-v01-prereg`, `foton-v01-prereg-r2`, `foton-v02-prereg` (10.0-C ve 10.0-G bunlara karşı denetler).
- `PREREGISTRATION.md` (v01) ve `PREREGISTRATION_V02.md` gövdesi (yalnız "Sapmalar" sonuna ek; `scripts/check_prereg_frozen.py`).
- `experiments/**`: 53 `config.json`; `v1-dense-moe-baseline/checksums.txt`; `foton-v02-prereg/{pins.json, lstar.json, baseline_reference.json, corpus_report.json, mutation_log.md, run_ledger.jsonl}`; `foton-v02/{ox1_microbench.json, gate_report_06.md, v03_design_notes.md, mutation_log_06.md, mutation_log_07.md}`; `foton-v03-step1/{decision_rule.md, mutation_log.md, gate_s101.json}`; `v2-grouped-dispatch/diagnostics/*.py`; `v3-router-ablation/checks/*` (`fingerprint.py` + iki JSON).
- `data/foton` (9 çivili dosya; dizine dosya eklenemez) ve `data/sp8k/tokenizer.model` (v1 `checksums.txt:8`).
- `checkpoints/**`: 79 `best.pt` + 79 `last.pt`, 160 `config.json`; `foton-v02r/r-*` üçlüleri `pins.json`'da; `foton-v03-step1` 8 koşu + `campaign_status.json` (`gate_s102_exceeded`); `foton-v03-dcheck`.
- `README.md` "Donmuş kayıt" bölümleri (140-141 ve devamı; sayılar değiştirilmez).
- `docs/ARCHITECTURE_AS_BUILT.md` ve `docs/flops.md` §8: dört kolun (D, M, H, X) test-kilitli belgesi — yeni model için ayrı belge / bölüm.

