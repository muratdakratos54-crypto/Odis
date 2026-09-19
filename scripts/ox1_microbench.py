"""O-X1: FOTON modüllerinin modül düzeyi mikrobenchmark'ı (ön-kayıt "O-X1. X kolunun gather bedeli ayrı ölçülür").

    .venv\\Scripts\\python scripts/ox1_microbench.py            # GPU BOŞKEN; çıktı experiments/foton-v02/ox1_microbench.json

Ne ölçer: eğitimde koşan 2 katmanlı FOTON modülünün TAMAMI — iki `src.model.Block` (norm + attention + dense
SwiGLU), `PRESETS['foton-x']` boyutlarında (d_model 512, 8 head, ffn 1408, T=512) — için ileri + geri süre.
Dört desen: 'A', 'B', 'C' ve 'full'. 'full', D/M katmanlarının bedelidir ve baseline'larla BİREBİR aynı yoldan,
`pattern=None` ile (SDPA `is_causal=True` hızlı yolu) kurulur. H kolunun üç modülü de A desenidir: A bir kez
ölçülür ve H için de geçerlidir (`src.model.FOTON_ARMS`'tan okunur, elle yazılmaz).

DÜRÜST NOT — bu ölçüm ön-kaydın sorduğu şeyin ADINI taşımıyor: yoğun SDPA + boolean maske ile ortada bir
GATHER YOK. Maske toplamsal bias gibi uygulanır; maskelenen (i, j) çiftleri hesaptan ATLANMAZ, çekirdek T x T
skorun tamamını üretir. Yani A/B/C algoritmik olarak token/head başına ~128.4 bağlantı görürken harcanan hesap
bağlantı sayısıyla ÖLÇEKLENMEZ. Bu betiğin ayırdığı soru şudur: "yavaşlama maskeli-attention çekirdek yolundan mı
(nedensel hızlı yola karşı), yoksa FOTON'un kendisinden mi (router + index_select/index_copy, ayrı ölçülür)".
A, B ve C arasındaki fark ise desenin DEĞİL yalnızca maske içeriğinin farkıdır; eşit çıkması beklenir.

BAYAT METİN UYARISI: ön-kaydın O-X1 paragrafı modülleri hâlâ ESKİ tasarımla anıyor ("A bitişik 256, B dilated
256, C 128+128"). O tasarım nedensel maske altında T=512'de 192.25 / 128.50 / 88.31 bağlantı verir — reddedilen
compute confound'un kendisi. Tag'li belge değiştirilemez; DONMUŞ tasarım şudur ve burada ÜÇ KİLİTLE pinlenir:
A yerel W=150 / B dilated r=2 W=256 / C yarım yerel W=256 + yarım strided s=4 W=256.
  kilit 1  `src.attn_patterns`'in donmuş sabitleri DONMUS_SPEC ile birebir aynı olmalı;
  kilit 2  kurulan Block'un GERÇEKLEŞEN maske buffer'ından SAYILAN bağlantı 128.4 ± 0.4 içinde olmalı
           (formülden değil: mask.sum() / (n_head * T)) — sahibin ZORUNLU DOĞRULAMA 1'i;
  kilit 3  aynı buffer, burada donmuş parametrelerden BAĞIMSIZ bir yolla (ofset serpme) yeniden kurulan maskeyle
           head head AYNI olmalı. Kilit 2 tek başına yetmez: "A" adına dilated(2, 256) konursa sayı 128.50 çıkar
           ve tolerans içinde kalır; desenin kendisi ancak maske karşılaştırmasıyla yakalanır.
Biri tutmazsa betik ÖLÇMEDEN çıkar (çıkış kodu != 0) ve JSON YAZMAZ. W / r / s / T / n_head'i değiştiren HİÇBİR
komut satırı bayrağı yoktur ve olmamalıdır: bayat O-X1 metni koda bir bayrakla sızamasın.

Protokol (JSON'a `protocol_pinned` olarak yazılır; gerçekleşen değerler ve sapmalar ayrıca): mikro-batch 16, T 512,
bf16 autocast, ileri + geri (loss = çıktının toplamı), AdamW DAHİL DEĞİL, zamanlamanın iki yanında
`torch.cuda.synchronize`, modül başına 10 ısınma iterasyonu, sonra 5 tekrar x 30 zamanlanmış iterasyon; raporlanan
sayı tekrarlar üzerinden tok/s MEDYANI (+ min/max), iterasyon başına token = 16 * 512. Modüller tekrarlar arasında
DÖNEN sırayla ölçülür (A,B,C,full / B,C,full,A / ...): ısıl sürüklenme tek modüle yüklenmesin — ön-kayıt V11a'nın
Latin karesiyle aynı gerekçe. Tekrar sonuçları listeyle değil tekrar KİMLİĞİYLE ("r0", "r1", ...) anahtarlı taşınır.

`--device cpu` YALNIZCA birim testleri içindir; o çıktı `reportable: false` ve gürültülü bir uyarı taşır.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src import attn_patterns  # noqa: E402
from src import model as foton_model  # noqa: E402
from src.config import PRESETS  # noqa: E402

PRESET = "foton-x"
MODULES = ("A", "B", "C", "full")       # temel ölçüm sırası; tekrar r'de r kadar döndürülür
FOTON_MODULES = ("A", "B", "C")         # bağlantı toleransı bunlara uygulanır ('full' referanstır: 256.50)

# --- DONMUŞ spesifikasyon (ön-kayıt "MİMARİ"). Bunlar PİN'dir, ölçüm değil: elle yazılmaları kasıtlı — kodun geri
# kalanından BAĞIMSIZ ikinci bir kopya olmasalar `src.attn_patterns` ile birlikte sessizce kayarlardı.
FROZEN_SPEC = {"A_LOCAL_W": 150, "B_DILATION_R": 2, "B_BUDGET_W": 256,
               "C_LOCAL_W": 256, "C_STRIDE_S": 4, "C_STRIDE_W": 256}
CONN_TARGET = 128.4     # sahibin ZORUNLU DOĞRULAMA 1'i: üç modül de 128.4 ± 0.4 (T=512, nedensel, SAYILARAK)
CONN_TOL = 0.4
# Ölçülen birimin boyutları: `PRESETS['foton-x']` bunlardan saparsa ölçülen şey eğitilen modül değildir.
FROZEN_DIMS = {"d_model": 512, "n_head": 8, "ffn_hidden": 1408, "max_seq_len": 512,
               "n_experts": 0, "dropout": 0.0, "foton_pre": 3, "foton_routed": 2}

# --- pinli protokol. Yalnızca `repeats` ve `iters` bayrakla değişebilir; batch, ısınma ve autocast yalnızca Python
# düzeyinde (birim testleri: `run(batch=..., warmup=..., bf16=False)`) değişir. HER sapma JSON'a yazılır ve çıktıyı
# raporlanamaz yapar. Sapma bu sözlüğe karşı hesaplanır; varsayılanlar da BURADAN okunur ki "pinli" ile
# "varsayılan" ayrışamasın. (bf16'nın testte kapatılabilmesinin nedeni: bu makinenin CPU'sunda bf16 matmul
# emülasyonu fp32'den ~100x yavaş — ölçüldü: adım başına 4.8 s'ye karşı 0.05 s.)
PINNED = {"batch": 16, "T": FROZEN_DIMS["max_seq_len"], "warmup_iters": 10, "repeats": 5, "iters": 30,
          "device": "cuda", "autocast": "bfloat16"}
DEFAULT_OUT = "experiments/foton-v02/ox1_microbench.json"
FROZEN_DIR = os.path.join(ROOT, "experiments", "foton-v02-prereg")     # tag'li artifact'lar: buraya YAZILMAZ

# 'full' referansının is_causal yolu ile ikiz maskeli yolun çıktıları arasındaki kabul edilen göreli fark (fp32).
# Eşiğin iki yanı `tests/test_ox1_microbench.py`'de YENİDEN ÖLÇÜLÜR (buradaki sayılar yalnızca gerekçe, CPU fp32):
# doğru maskede ~1e-7; TEK girdisi eksik maskede (yerel W=511: yalnızca son satırın ilk anahtarı) ~2e-3;
# yerel W=256'da ~4e-2. Eşik, en zor yanlışın bir mertebeden fazla altında ve doğrunun üç mertebe üstünde.
TWIN_REL_TOL = 1e-4

HONEST_NOTE = (
    "GATHER YOK: yoğun SDPA + boolean maske ile maskelenen (i, j) çiftleri hesaptan atlanmaz, çekirdek T x T skorun "
    "tamamını üretir. Bu ölçüm 'gather bedeli' DEĞİL, eşit algoritmik bağlantıda MASKELİ-ATTENTION çekirdek yolunun "
    "nedensel hızlı yola (is_causal) karşı bedelidir; 'yavaşlama maske yolundan mı, FOTON'un kendisinden mi' "
    "sorusunu ayırır. Router ve index_select/index_copy bedeli bu ölçümün DIŞINDADIR.")
STALE_NOTE = (
    "Ön-kaydın O-X1 paragrafındaki modül listesi (A bitişik 256 / B dilated 256 / C 128+128) BAYATTIR: nedensel "
    "maske altında 192.25 / 128.50 / 88.31 bağlantı verir (reddedilen compute confound). Burada ölçülen DONMUŞ "
    "tasarımdır ve üç kilitle pinlenmiştir (sabitler, sayılmış bağlantı, maske özdeşliği).")


def refuse(why: str) -> None:
    """Ölçmeden ve JSON YAZMADAN çık. SystemExit(str) -> çıkış kodu 1, mesaj stderr'e."""
    raise SystemExit(f"O-X1 RED: {why}\nÖlçüm YAPILMADI ve JSON YAZILMADI (diskte bir çıktı varsa ESKİ koşudandır).")


def rotated_order(repeat: int) -> tuple[str, ...]:
    """Tekrar `repeat`'in ölçüm sırası: temel sıra `repeat` kadar sola döndürülür (V11a'nın gerekçesi)."""
    k = repeat % len(MODULES)
    return MODULES[k:] + MODULES[:k]


# ------------------------------------------------------------------------------------------------ kilitler
def check_frozen_constants() -> dict:
    """KİLİT 1: `src.attn_patterns`'in donmuş sabitleri. Modül özniteliği ÇAĞRI ANINDA okunur."""
    live = {k: getattr(attn_patterns, k, None) for k in FROZEN_SPEC}
    if live != FROZEN_SPEC:
        diff = {k: {"beklenen": FROZEN_SPEC[k], "bulunan": live[k]} for k in FROZEN_SPEC if live[k] != FROZEN_SPEC[k]}
        refuse(f"src.attn_patterns donmuş spesifikasyondan sapmış: {diff} — bayat O-X1 tasarımı sızmış olabilir")
    return live


def check_frozen_dims(cfg) -> dict:
    """Ölçülen birim eğitilen modül mü: preset boyutları + attn_patterns'in kampanya T / n_head'i."""
    live = {k: getattr(cfg, k) for k in FROZEN_DIMS}
    if live != FROZEN_DIMS:
        diff = {k: {"beklenen": FROZEN_DIMS[k], "bulunan": live[k]} for k in FROZEN_DIMS if live[k] != FROZEN_DIMS[k]}
        refuse(f"PRESETS[{PRESET!r}] ölçülecek birimin donmuş boyutlarından sapmış: {diff}")
    if (attn_patterns.T_KAMPANYA, attn_patterns.N_HEAD_KAMPANYA) != (cfg.max_seq_len, cfg.n_head):
        refuse(f"attn_patterns kampanya (T, n_head) = ({attn_patterns.T_KAMPANYA}, {attn_patterns.N_HEAD_KAMPANYA}) "
               f"preset ile ({cfg.max_seq_len}, {cfg.n_head}) uyuşmuyor")
    return live


def check_arms() -> dict:
    """H'nin deseni ve X'in modül listesi `src.model.FOTON_ARMS`'tan OKUNUR; beklenenden saparsa RED."""
    arms = foton_model.FOTON_ARMS
    if tuple(arms["x"]) != FOTON_MODULES:
        refuse(f"FOTON_ARMS['x'] = {arms['x']!r}; ölçülen modül listesi {FOTON_MODULES!r} ile aynı değil")
    h_patterns = sorted(set(arms["h"]))
    if h_patterns != ["A"]:
        refuse(f"FOTON_ARMS['h'] = {arms['h']!r}; H'nin bütün modülleri A deseni olmalıydı (A bir kez ölçülüyor)")
    return {"pattern": h_patterns[0], "n_modules": len(arms["h"]),
            "note": "H kolunun üç modülü de A desenidir (ayrı ağırlık, aynı maske): A satırı H için de geçerlidir, "
                    "ayrıca ölçülmedi."}


def _expected_head(T: int, step: int, budget: int) -> torch.Tensor:
    """Donmuş tek-head deseni, `attn_patterns`'ten BAĞIMSIZ yolla: k. ofset (k*step) için köşegen serpme.

    `attn_patterns.Head.mask` yüklem ((off % step == 0) & (off // step < budget)) ile kurar; burada aynı küme
    ofset ofset yazılır. İki yol ayrı olduğundan birindeki hata diğerine kendiliğinden taşınmaz.
    """
    m = torch.zeros(T, T, dtype=torch.bool)
    for k in range(budget):
        off = k * step
        if off >= T:
            break
        i = torch.arange(off, T)
        m[i, i - off] = True
    return m


def _expected_heads(name: str, T: int) -> list[torch.Tensor]:
    """Modül adının DONMUŞ head desenleri (FROZEN_SPEC'ten). Head'ler bu desenlere EŞİT sayıda bölünür."""
    s = FROZEN_SPEC
    spec = {"A": [(1, s["A_LOCAL_W"])],
            "B": [(s["B_DILATION_R"], s["B_BUDGET_W"])],
            "C": [(1, s["C_LOCAL_W"]), (s["C_STRIDE_S"], s["C_STRIDE_W"])],
            "full": [(1, T)]}[name]
    return [_expected_head(T, step, budget) for step, budget in spec]


def check_mask_is_frozen_pattern(name: str, mask: torch.Tensor, T: int, n_head: int) -> None:
    """KİLİT 3: gerçekleşen [n_head, T, T] maske, donmuş desenle head head aynı mı (head SIRASINDAN bağımsız).

    Ön-kayıt C için "yarısı yerel, yarısı strided" der, hangi head'in hangisi olduğunu söylemez; o yüzden her head
    beklenen desenlerden BİRİNE eşit olmalı ve desen başına head sayısı n_head / desen sayısı olmalıdır.
    """
    if tuple(mask.shape) != (n_head, T, T) or mask.dtype != torch.bool:
        refuse(f"{name}: maske buffer'ı [{n_head}, {T}, {T}] bool değil: {tuple(mask.shape)} {mask.dtype}")
    mask = mask.cpu()
    expected = _expected_heads(name, T)
    if n_head % len(expected):
        refuse(f"{name}: n_head={n_head}, {len(expected)} desene eşit bölünemiyor")
    share = n_head // len(expected)
    hits = {g: 0 for g in range(len(expected))}
    for h in range(n_head):
        match = [g for g, exp in enumerate(expected) if torch.equal(mask[h], exp)]
        if not match:
            refuse(f"{name}: head {h}'in maskesi DONMUŞ desene uymuyor (satır başına bağlantı ilk/son: "
                   f"{int(mask[h, 0].sum())}/{int(mask[h, -1].sum())}) — bayat O-X1 tasarımı sızmış olabilir")
        hits[match[0]] += 1
    if any(n != share for n in hits.values()):
        refuse(f"{name}: head'ler desenlere eşit bölünmemiş: {hits}, desen başına {share} bekleniyordu")


def counted_connections(mask: torch.Tensor) -> float:
    """Token/head başına ORTALAMA bağlantı — maskeden SAYILARAK: True girdi sayısı / (n_head * T)."""
    return mask.sum().item() / (mask.shape[0] * mask.shape[1])


def check_connection_target(counts: dict[str, float]) -> None:
    """KİLİT 2: A, B, C'nin SAYILMIŞ bağlantısı 128.4 ± 0.4 içinde mi. NaN da RED'dir (`not <=`)."""
    bad = {k: v for k, v in counts.items() if k in FOTON_MODULES and not abs(v - CONN_TARGET) <= CONN_TOL}
    missing = [k for k in FOTON_MODULES if k not in counts]
    if bad or missing:
        refuse(f"SAYILMIŞ bağlantı {CONN_TARGET} ± {CONN_TOL} dışında: {bad} eksik: {missing} (hepsi: {counts}) — "
               f"bayat O-X1 tasarımı 192.25 / 128.50 / 88.31 verir")


# ------------------------------------------------------------------------------------------------ kurulum
def build_module(cfg, name: str, seed: int = 0) -> nn.ModuleList:
    """Eğitimdeki 2 katmanlı modül: `GPT._build_foton` ile aynı `Block(cfg, pre + j, pattern=...)` çağrısı.

    'full' -> pattern=None: D/M katmanlarıyla BİREBİR aynı yol (is_causal hızlı yolu, maske buffer'ı YOK).
    Dört modül aynı tohumla kurulur ve maskeler torch RNG'si tüketmez. Init `GPT._init_weights`'in aynısı
    (N(0, 0.02), residual projeksiyonları (2*n_layer)^-0.5 ile): süreyi etkilemesi beklenmez, ama ölçülen şey
    eğitimin başındaki modüle benzesin.
    """
    torch.manual_seed(seed)
    pattern = None if name == "full" else name
    blocks = nn.ModuleList([foton_model.Block(cfg, cfg.foton_pre + j, pattern=pattern)
                            for j in range(cfg.foton_routed)])
    for m in blocks.modules():
        if isinstance(m, nn.Linear):
            scale = (2 * cfg.n_layer) ** -0.5 if getattr(m, "is_residual_proj", False) else 1.0
            nn.init.normal_(m.weight, mean=0.0, std=0.02 * scale)
    return blocks


def module_masks(cfg, name: str, blocks: nn.ModuleList) -> dict[str, torch.Tensor]:
    """Modülün her katmanının GERÇEKLEŞEN maske buffer'ı, "layer<idx>" anahtarıyla (SDPA'ya verilen nesne)."""
    out = {}
    for j, blk in enumerate(blocks):
        if blk.attn.pattern != name or not hasattr(blk.attn, "attn_mask"):
            refuse(f"{name}: katman {cfg.foton_pre + j} beklenen desenle kurulmamış (pattern={blk.attn.pattern!r})")
        out[f"layer{cfg.foton_pre + j}"] = blk.attn.attn_mask
    return out


def full_reference_twin(cfg, blocks: nn.ModuleList, cos: torch.Tensor, sin: torch.Tensor, device: str):
    """'full' referansı maske TUTMAZ (is_causal yolu); bağlantısı, eşdeğerliği KANITLANMIŞ ikizinden sayılır.

    İkiz = aynı ağırlıklarla `Block(pattern='full')`. İki attention'ın çıktısı fp32'de karşılaştırılır: is_causal
    yolu gerçekten 'full' maskesini gerçekleştiriyorsa fark yuvarlama düzeyindedir. Böylece "256.50" referansın
    KENDİ davranışına bağlanmış olur, yanına yazılmış bir formüle değil. Döner: (ikiz maske, göreli fark).
    """
    ref = blocks[0]
    if ref.attn.pattern is not None or hasattr(ref.attn, "attn_mask"):
        refuse("'full' referansı is_causal hızlı yolunda DEĞİL (maske taşıyor): D/M katmanlarının bedelini ölçmez")
    twin = foton_model.Block(cfg, cfg.foton_pre, pattern="full").to(device)
    twin.load_state_dict(ref.state_dict())
    g = torch.Generator().manual_seed(1)
    z = torch.randn(2, cfg.max_seq_len, cfg.d_model, generator=g).to(device)
    with torch.no_grad():
        y_ref = ref.attn(z, cos, sin).float()
        y_twin = twin.attn(z, cos, sin).float()
    rel = ((y_ref - y_twin).abs().max() / y_ref.abs().max()).item()
    if not rel <= TWIN_REL_TOL:
        refuse(f"'full' referansı ile ikiz 'full' maskesi AYNI attention'ı vermiyor (göreli fark {rel:.3e} > "
               f"{TWIN_REL_TOL}): PATTERNS['full'] tam nedensel değil ya da is_causal yolu değişmiş")
    return twin.attn.attn_mask, rel


# ------------------------------------------------------------------------------------------------ ölçüm
def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def train_step(blocks: nn.ModuleList, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, device: str,
               bf16: bool = True) -> None:
    """Bir ileri + geri: bf16 autocast, loss = çıktının toplamı. Optimizer YOK (AdamW ölçüme dahil değil).

    `x.requires_grad`: eğitimde modülün girdisi paylaşılan ön blokların çıktısıdır ve gradyan ister; girdiye
    giden geri yayılım da modülün bedelidir. `bf16=False` yalnızca CPU birim testleri içindir (sapma kaydedilir).
    """
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
        h = x
        for blk in blocks:
            h, _ = blk(h, cos, sin)
        loss = h.float().sum()
    loss.backward()
    blocks.zero_grad(set_to_none=True)
    x.grad = None


def sdpa_ops(blocks: nn.ModuleList, x, cos, sin, device: str, bf16: bool = True) -> list[str]:
    """Hangi SDPA arka ucu koştu: dağıtıcının çağırdığı `aten::_scaled_dot_product_*` op ADI.

    Profiler burada YALNIZCA op adını okumak için ve zamanlanan bölgenin DIŞINDA kullanılır; süre ve FLOP
    profiler'dan alınmaz. Belirlenemezse nedenini söyleyen tek elemanlı liste döner (ölçümü düşürmez).
    (Bu torch derlemesinde kineto stderr'e "profiler_start/stop" satırları basar; zararsızdır, susturulamıyor.)
    """
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            train_step(blocks, x, cos, sin, device, bf16)
        names = sorted({e.key for e in prof.key_averages()
                        if "scaled_dot_product" in e.key and e.key != "aten::scaled_dot_product_attention"})
        return names or ["belirlenemedi: profiler kaydında SDPA alt-op'u yok"]
    except Exception as e:  # noqa: BLE001 — arka uç adı bilgi amaçlı; ölçümü bunun için düşürme
        return [f"belirlenemedi: {type(e).__name__}: {e}"]


def time_segment(blocks, x, cos, sin, device: str, iters: int, tokens_per_iter: int, bf16: bool = True) -> dict:
    """`iters` ileri+geri'nin duvar saati; iki yanında senkron. Tepe bellek bu dilim için sıfırlanıp okunur."""
    base = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated() / 1e9
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        train_step(blocks, x, cos, sin, device, bf16)
    _sync(device)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    return {"seconds": dt, "tok_per_sec": tokens_per_iter * iters / dt, "peak_gb": peak, "base_gb": base}


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment(device: str) -> dict:
    env = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "python": sys.version.split()[0],
           "host": platform.platform(), "device": device,
           "device_name": torch.cuda.get_device_name(0) if device == "cuda" else (platform.processor() or "cpu"),
           "git_head": _git("rev-parse", "HEAD"), "git_status": _git("status", "--porcelain").splitlines(),
           "sha256": {rel: _sha256(os.path.join(ROOT, *rel.split("/")))
                      for rel in ("src/attn_patterns.py", "src/model.py", "src/config.py",
                                  "scripts/ox1_microbench.py")}}
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        env["gpu_memory_at_start"] = {
            "free_gb": free / 1e9, "total_gb": total / 1e9, "used_gb": (total - free) / 1e9,
            "reserved_by_this_process_gb": torch.cuda.memory_reserved() / 1e9,
            "note": "used_gb = aygıtın TAMAMINDA dolu bellek (bu sürecin CUDA bağlamı ve ayırdıkları dahil, modüller "
                    "kurulduktan sonra okundu). Başka bir süreç (ör. eğitim kuyruğu) GPU'yu kullanıyorsa tok/s "
                    "ölçümü GEÇERSİZDİR: used_gb bu sürecin payını ve masaüstünü belirgin aşıyorsa GPU boşken yinele."}
    return env


def run(out: str, device: str = PINNED["device"], repeats: int = PINNED["repeats"], iters: int = PINNED["iters"],
        batch: int = PINNED["batch"], warmup: int = PINNED["warmup_iters"], bf16: bool = True) -> dict:
    """Kilitleri sına, ölç, JSON'u EN SON ve tek hamlede yaz. Herhangi bir kilit tutmazsa hiçbir şey yazılmaz."""
    if repeats < 1 or iters < 1 or batch < 1 or warmup < 0:
        refuse(f"repeats/iters/batch >= 1 ve warmup >= 0 olmalı: {repeats}/{iters}/{batch}/{warmup}")
    out_abs = os.path.abspath(out)
    if _inside(out_abs, FROZEN_DIR):
        refuse(f"--out dondurulmuş ön-kayıt dizinini gösteriyor ({FROZEN_DIR}); tag'li artifact'ların üstüne yazılmaz")
    if device not in ("cuda", "cpu"):
        refuse(f"device 'cuda' | 'cpu' olmalı: {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        refuse("CUDA yok. O-X1 GPU ölçümüdür; CPU yalnızca birim testleri içindir (--device cpu)")

    # --- kilitler: ÖLÇÜMDEN ÖNCE, sırayla ---
    frozen_live = check_frozen_constants()                                   # kilit 1
    cfg = PRESETS[PRESET][0]
    dims_live = check_frozen_dims(cfg)
    h_arm = check_arms()
    T, n_head = cfg.max_seq_len, cfg.n_head

    cos, sin = foton_model.precompute_rope(cfg.d_model // n_head, T, cfg.rope_theta)
    cos, sin = cos.to(device), sin.to(device)
    built = {name: build_module(cfg, name).to(device) for name in MODULES}

    counts: dict[str, float] = {}
    per_layer: dict[str, dict[str, float]] = {}
    source: dict[str, str] = {}
    for name in FOTON_MODULES:
        masks = module_masks(cfg, name, built[name])
        for mask in masks.values():
            check_mask_is_frozen_pattern(name, mask, T, n_head)              # kilit 3
        per_layer[name] = {layer: counted_connections(mask) for layer, mask in masks.items()}
        if len(set(per_layer[name].values())) != 1:
            refuse(f"{name}: katmanların SAYILMIŞ bağlantıları farklı: {per_layer[name]}")
        counts[name] = next(iter(per_layer[name].values()))
        source[name] = "ölçülen Block'un attn_mask buffer'ı: mask.sum() / (n_head * T)"
    twin_mask, twin_rel = full_reference_twin(cfg, built["full"], cos, sin, device)
    check_mask_is_frozen_pattern("full", twin_mask, T, n_head)
    counts["full"] = counted_connections(twin_mask)
    per_layer["full"] = {"twin": counts["full"]}
    source["full"] = ("ölçülen blok is_causal hızlı yolundadır ve maske TUTMAZ; sayı, aynı ağırlıklı ikiz "
                      f"Block(pattern='full')'un attn_mask buffer'ından. İki attention çıktısının göreli farkı "
                      f"{twin_rel:.3e} (eşik {TWIN_REL_TOL})")
    del twin_mask
    check_connection_target(counts)                                          # kilit 2

    # --- ölçüm ---
    env = environment(device)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(batch, T, cfg.d_model, generator=g).to(device).requires_grad_(True)
    tokens_per_iter = batch * T
    backends = {name: sdpa_ops(built[name], x, cos, sin, device, bf16) for name in MODULES}
    for name in MODULES:
        for _ in range(warmup):
            train_step(built[name], x, cos, sin, device, bf16)
    _sync(device)

    by_repeat: dict[str, dict] = {}
    for r in range(repeats):
        order, seg = [], {}
        for name in rotated_order(r):
            seg[name] = time_segment(built[name], x, cos, sin, device, iters, tokens_per_iter, bf16)
            order.append(name)                  # GERÇEKLEŞEN sıra: plandan değil, ölçüm anından
        by_repeat[f"r{r}"] = {"order": order, "segments": seg}

    modules = {}
    for name in MODULES:
        tps = {rid: rep["segments"][name]["tok_per_sec"] for rid, rep in by_repeat.items()}
        peaks = {rid: rep["segments"][name]["peak_gb"] for rid, rep in by_repeat.items()}
        modules[name] = {
            "pattern": "tam nedensel, is_causal hızlı yolu (D/M katmanlarının yolu)" if name == "full"
                       else " + ".join(h.label for h in attn_patterns.PATTERNS[name].heads),
            "connections_counted": counts[name], "connections_per_layer": per_layer[name],
            "connections_source": source[name],
            "tok_per_sec_median": statistics.median(tps.values()),
            "tok_per_sec_min": min(tps.values()), "tok_per_sec_max": max(tps.values()),
            "tok_per_sec_by_repeat": tps,
            "peak_gb": None if device != "cuda" else max(peaks.values()), "peak_gb_by_repeat": peaks,
            "sdpa_ops": backends[name],
        }
    ref = modules["full"]["tok_per_sec_median"]
    for name in MODULES:
        modules[name]["tok_per_sec_vs_full"] = modules[name]["tok_per_sec_median"] / ref
    positions = {name: {str(p): sum(rep["order"][p] == name for rep in by_repeat.values())
                        for p in range(len(MODULES))} for name in MODULES}

    realized = {"batch": batch, "T": T, "warmup_iters": warmup, "repeats": repeats, "iters": iters, "device": device,
                "autocast": "bfloat16" if bf16 else "KAPALI (float32)"}
    deviations = {k: {"pinned": PINNED[k], "realized": realized[k]} for k in PINNED if PINNED[k] != realized[k]}
    warnings = []
    if device != PINNED["device"]:
        warnings.append("!!! UYARI: BU ÇIKTI CPU'DA ÜRETİLDİ — YALNIZCA BİRİM TESTİ İÇİNDİR. Buradaki tok/s değerleri "
                        "O-X1 ölçümü DEĞİLDİR ve rapora GİREMEZ. !!!")
    if deviations:
        warnings.append(f"!!! UYARI: pinli protokolden sapıldı: {deviations} — bu çıktı RAPORLANAMAZ. !!!")
    if repeats % len(MODULES):
        warnings.append(f"sıra dengesi tam değil: {repeats} tekrar {len(MODULES)} modülle tam Latin karesi kurmaz "
                        f"(bkz. order_position_counts); medyan bu yüzden kullanılıyor")

    result = {
        "what": "O-X1 modül düzeyi mikrobenchmark (PREREGISTRATION_V02.md 'O-X1'); birim = eğitilen 2 katmanlı modül",
        "warnings": warnings, "reportable": not deviations,
        "honest_note": HONEST_NOTE, "stale_ox1_text": STALE_NOTE,
        "protocol_pinned": {**PINNED, "pass": "ileri + geri, loss = çıktının toplamı",
                            "optimizer": "YOK (AdamW dahil değil)",
                            "sync": "torch.cuda.synchronize, zamanlanan dilimin iki yanında",
                            "statistic": "tekrarlar üzerinden tok/s medyanı + min/max",
                            "tokens_per_iter": PINNED["batch"] * PINNED["T"],
                            "order": f"tekrar r'de temel sıra ({','.join(MODULES)}) r kadar sola döner",
                            "peak_memory": "dilim başına sıfırlanan torch.cuda.max_memory_allocated; dört modülün "
                                           "ağırlıkları da bellekte durur (dilim başı taban: repeats.*.segments.*.base_gb)"},
        "protocol_realized": {**realized, "tokens_per_iter": tokens_per_iter},
        "protocol_deviations": deviations,
        "frozen_spec": {"expected": FROZEN_SPEC, "found_in_attn_patterns": frozen_live},
        "frozen_dims": {"expected": FROZEN_DIMS, "found_in_preset": dims_live, "preset": PRESET},
        "connection_target": {"target": CONN_TARGET, "tol": CONN_TOL, "applies_to": list(FOTON_MODULES)},
        "modules": modules, "h_arm": h_arm,
        "repeats": by_repeat, "order_position_counts": positions,
        "environment": env, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    tmp = out_abs + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    os.replace(tmp, out_abs)                    # yarım yazılmış JSON kalmasın
    return result


def _inside(path: str, directory: str) -> bool:
    """`path`, `directory`'nin içinde mi. Windows: büyük/küçük harf duyarsız; farklı sürücü -> içinde değil
    (`os.path.commonpath` o durumda ValueError atar)."""
    p, d = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(directory))
    try:
        return os.path.commonpath([p, d]) == d
    except ValueError:
        return False


def build_parser() -> argparse.ArgumentParser:
    """YALNIZCA dört bayrak. Desen / boyut parametresi (W, r, s, T, n_head, batch, preset) bayrağı YOKTUR ve
    eklenmemelidir; `allow_abbrev=False` kısaltmayla yanlış bayrağa düşmeyi de kapatır."""
    p = argparse.ArgumentParser(description="O-X1 modül mikrobenchmark'ı (donmuş tasarım; desen bayrağı yok)",
                                allow_abbrev=False)
    p.add_argument("--out", default=DEFAULT_OUT)
    # Yardım metnindeki sayı PINNED'den OKUNUR: elle yazılsaydı pin değiştiğinde yardım sessizce yalan söylerdi.
    p.add_argument("--repeats", type=int, default=PINNED["repeats"],
                   help=f"pinli: {PINNED['repeats']} (değişirse çıktı raporlanamaz)")
    p.add_argument("--iters", type=int, default=PINNED["iters"],
                   help=f"pinli: {PINNED['iters']} (değişirse çıktı raporlanamaz)")
    p.add_argument("--device", choices=("cuda", "cpu"), default=PINNED["device"],
                   help="cpu YALNIZCA birim testleri için; çıktı raporlanamaz")
    return p


def main(argv: list[str] | None = None) -> dict:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # Windows konsolu cp1252
    args = build_parser().parse_args(argv)
    res = run(args.out, device=args.device, repeats=args.repeats, iters=args.iters)
    for w in res["warnings"]:
        print(w)
    print(f"{'modül':6s} {'bağlantı (SAYILMIŞ)':>20s} {'tok/s medyan':>14s} {'min':>11s} {'max':>11s} "
          f"{'full oranı':>11s} {'tepe GB':>8s}  SDPA")
    for name, m in res["modules"].items():
        peak = "—" if m["peak_gb"] is None else f"{m['peak_gb']:.2f}"
        print(f"{name:6s} {m['connections_counted']:20.4f} {m['tok_per_sec_median']:14,.0f} "
              f"{m['tok_per_sec_min']:11,.0f} {m['tok_per_sec_max']:11,.0f} {m['tok_per_sec_vs_full']:11.3f} "
              f"{peak:>8s}  {', '.join(m['sdpa_ops'])}")
    print(f"H kolu: {res['h_arm']['note']}")
    print(f"raporlanabilir: {res['reportable']} -> {os.path.abspath(args.out)}")
    return res


if __name__ == "__main__":
    main()
