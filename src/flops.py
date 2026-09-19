"""Analitik aktif FLOP hesabı: karşılaştırmalarda BİRİNCİL hesap ölçüsü.

Türetim ve varsayımlar adım adım: docs/flops.md. Kısaca:
  * Birim: bir çarpma + bir toplama çifti = 2 FLOP (Kaplan ve ark. 2020, Hoffmann ve ark. 2022 ile aynı).
    d_in -> d_out lineer katman, token başına 2 * d_in * d_out FLOP.
  * "Aktif": token başına gerçekten çalışan hesap. MoE katmanında E expert'ten yalnızca top_k tanesi + shared
    expert + router sayılır; token almayan expert'lerin parametreleri hesaba girmez.
  * Birincil toplam = matris çarpımları (tüm lineer katmanlar, lm_head) + causal attention'ın bağlam terimi
    (q.k skorları ve değerlerin ağırlıklı toplamı). Eleman bazlı işlemler (norm, SiLU, softmax, residual,
    gate çarpımı ...) ayrı ve yaklaşık olarak `secondary_forward_flops_per_token` ile verilir; bütçelere girmez.
  * Eğitim = ileri x 3: her matris çarpımının geri yayılımı iki matris çarpımıdır (girdi ve ağırlık gradyanı).
  * Dispatch (argsort, bincount, gather/scatter kopyaları) 0 FLOP: tamsayı ve bellek işlemleri, aritmetik
    değil. "loop" ile "grouped" yolların FLOP'u bu yüzden tanım gereği aynıdır; farkları yalnızca duvar saatinde.

FOTON v0.2 kolları (`cfg.foton` = "h" | "x"; ön-kayıt V3 "Analitik hesap ..." paragrafı, docs/flops.md bölüm 8):
  * Bir dizi `cfg.n_layer` (8) katmandan geçer: 6 paylaşılan tam-nedensel blok + TEK modülün 2 katmanı. Aktif
    FLOP tam olarak bu 8 katmanı sayar; KURULAN 12 bloğu değil (dizi almayan iki modül o dizi için çalışmaz).
  * Yönlendirilen katmanların bağlam terimi 4*d*c'dir; c = token/head başına ortalama nedensel bağlantı,
    `src/attn_patterns`'in kurduğu GERÇEK maskeden SAYILARAK alınır (elle yazılmış 128.x yok). Tam-nedensel
    2*d*(T+1) terimini bu katmanlara uygulamak c yerine 256.5 sayar ve FOTON'un FLOP'unu fazla gösterir.
  * Kol başına TEK sabit c: kolun modülleri üzerinden MAKSİMUM (H: üçü de A -> 128.17; X: B -> 128.50, ön-kaydın
    "ortak hedef"i, FOTON'un ALEYHİNE muhafazakâr). Böylece FLOP/token router'ın seçimine BAĞLI DEĞİLDİR:
    birincil istatistiğin paydası router'ın kontrolünde olamaz (ön-kayıt V3 "FOTON'a özgü uyarı").
    Gerçekleşen seçim karışımıyla hesap yalnızca raporun duyarlılık satırı içindir: `foton_flops_realized`.
  * Router: DİZİ başına tek d -> 3 lineer katman = 2*d*3 FLOP / dizi; token başına T'ye bölünerek yazılır.

Profiler (torch.utils.flop_counter) yalnızca ÇAPRAZ KONTROL: formülü olmayan bir çekirdeği (ör. gruplu matris
çarpımı) sıfır sayabilir ve attention'ı causal maskeyi yok sayarak T x T üzerinden sayar (bkz. scripts/flops_report.py).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction

from src import attn_patterns
from src.config import ModelConfig

# `forward_flops_per_token` çıktısında attention BAĞLAM terimini taşıyan anahtarlar; geri kalan her anahtar bir matris
# çarpımıdır. FOTON kollarında bağlam İKİ anahtara bölünür (paylaşılan katmanlar + yönlendirilen katmanlar). Profiler
# çapraz kontrolü (B13) "matris çarpımları ayrı, attention çekirdekleri ayrı" karşılaştırır: yalnızca "attn_context"i
# ayıran bir tüketici FOTON'un yönlendirilen bağlam terimini sessizce matris çarpımı tarafına yazardı. Tüketiciler
# anahtar adını elle yazmak yerine bu sabiti kullanmalıdır.
CONTEXT_KEYS = ("attn_context", "foton_attn_routed")


def _layer_counts(cfg: ModelConfig) -> tuple[int, int]:
    """(dense katman, MoE katmanı) - model.Block.is_moe ile aynı kural."""
    n_moe = sum(1 for i in range(cfg.n_layer) if cfg.n_experts > 0 and i >= cfg.first_dense_layers)
    return cfg.n_layer - n_moe, n_moe


# ------------------------------------------------------------------------------------------ FOTON v0.2
def foton_module_keys(cfg: ModelConfig) -> tuple[str, ...]:
    """Kolun modül kimlikleri, `GPT.foton_connection_counts()` ile AYNI biçimde: "M<indeks>:<desen>".

    Neden desen adı tek başına anahtar değil: H kolunda üç modülün deseni de "A"dır (ayrı ağırlık, aynı maske);
    ada göre anahtarlanan bir eşleme üç modülü tek girdiye çökertirdi. İndeks + ad kimliği tek anlamlı kılar.
    Kol tanımı (`FOTON_ARMS`) `src/model.py`'den okunur — burada ikinci bir kopyası tutulmaz, yoksa modelin
    koştuğu modül listesi ile FLOP'un saydığı liste ayrışabilir. İçe aktarma tembeldir: FOTON olmayan
    config'lerde bu dosya torch'a dokunmaz.
    """
    from src.model import FOTON_ARMS
    if cfg.foton not in FOTON_ARMS:
        raise ValueError(f"foton {sorted(FOTON_ARMS)} içinden olmalı, verilen: {cfg.foton!r}")
    return tuple(f"M{m}:{name}" for m, name in enumerate(FOTON_ARMS[cfg.foton]))


def _foton_check(cfg: ModelConfig) -> None:
    """FOTON config'inin FLOP formülünün varsaydığı yapıda olduğunu ZORUNLU kılar: `GPT._build_foton`'un
    kurallarından BİRİNCİL formüle giren üçü (kol adı, n_experts = 0, katman bölüşümü). Çıplak assert değil:
    `python -O` atlamasın.

    Neden burada da: formül "n_layer = pre + routed + post katmandan geçilir, hepsi dense SwiGLU" varsayar.
    Bölüşüm n_layer ile uyuşmazsa paylaşılan katman sayısı sessizce yanlış çıkar; n_experts > 0 ise MoE
    terimleri FOTON'un hiç kurmadığı katmanlar için sayılır. İkisi de modeli kurmadan FLOP isteyen bir
    betikte (kuyruk, güç analizi) fark edilmeden karar paydasına girerdi. Modelin öteki iki kuralı
    (`foton_prefix_k >= 1`, `foton_reg`) birincil formüle girmez; yalnızca kullanıldıkları yerde, ikincil
    sayımda (`secondary_forward_flops_per_token`) denetlenir.
    """
    foton_module_keys(cfg)      # kol adı geçerli mi
    if cfg.n_experts > 0:
        raise ValueError("FOTON kolları dense SwiGLU kullanır (n_experts = 0); MoE'li FOTON'un FLOP formülü yok")
    if cfg.foton_pre + cfg.foton_routed + cfg.foton_post != cfg.n_layer or min(
            cfg.foton_pre, cfg.foton_routed, cfg.foton_post) < 1:
        raise ValueError(f"katman bölüşümü {cfg.foton_pre}+{cfg.foton_routed}+{cfg.foton_post} "
                         f"n_layer={cfg.n_layer} ile uyuşmuyor")


def _counted_connections(pattern: attn_patterns.Pattern, T: int, n_head: int) -> Fraction:
    """Bir desenin token/head başına ortalama nedensel bağlantısı — maske KURULUP SAYILARAK, tam kesir olarak.

    Sahibin ZORUNLU DOĞRULAMA 1'i: "teorik formülle değil, maskeyi kurup sayarak". `masks()` modelin
    `attn_mask` buffer'ını ürettiği AYNI çağrıdır (src/model.py CausalSelfAttention), yani burada sayılan
    grafik eğitimde koşan grafiktir. T ve n_head AÇIKÇA verilir: attn_patterns'in varsayılanları S0 kampanya
    değerleridir (512, 8) ve S1'de n_head=10'dur. Kapalı form `avg()` ikinci yol olarak karşılaştırılır:
    head ağırlıklandırması (C'nin yarı-yarıya bölünmesi) iki yolda ayrışırsa hata verir.
    """
    masks = pattern.masks(T, n_head)
    if masks.shape != (n_head, T, T):
        raise AssertionError(f"{pattern.name}: maske şekli {masks.shape} != ({n_head}, {T}, {T})")
    c = Fraction(int(masks.sum()), n_head * T)
    closed = pattern.avg(T, n_head)
    if not abs(float(c) - closed) <= 1e-9:
        raise AssertionError(f"{pattern.name}: maskeden sayılan bağlantı {float(c)} != kapalı form {closed}")
    return c


def _foton_module_c(cfg: ModelConfig, T: int) -> dict[str, Fraction]:
    """"M<indeks>:<desen>" -> sayılmış c. Kampanya T'sinde (512) donmuş hedefe PİNLİ.

    Pin (sahibin 06-EK A maddesi, "ikinci kilit"in FLOP tarafı): ön-kaydın O-X1 bölümünde ESKİ tasarım kaldı
    (A bitişik 256 / B dilated 256 / C 128+128 -> 192.25 / 128.50 / 88.31 bağlantı; reddedilen compute
    confound'un kendisi). O desenler koda sızarsa FLOP paydası da sessizce onlarla hesaplanırdı; burada
    kampanya T'sinde her modülün SAYILMIŞ c'si 128.4 ± 0.4 dışındaysa hesap durur. Hedef ve tolerans
    `src/model.py`'deki sabitlerdir (tek kopya). Başka T'de (tiny preset, testler) hedef tanımlı değildir.
    """
    from src.model import FOTON_ARMS, FOTON_CONN_TARGET, FOTON_CONN_TOL
    _foton_check(cfg)
    keys = foton_module_keys(cfg)
    out = {key: _counted_connections(attn_patterns.PATTERNS[name], T, cfg.n_head)
           for key, name in zip(keys, FOTON_ARMS[cfg.foton])}
    if T == attn_patterns.T_KAMPANYA:
        bad = {k: float(v) for k, v in out.items() if not abs(float(v) - FOTON_CONN_TARGET) <= FOTON_CONN_TOL}
        if bad:
            raise AssertionError(f"SAYILMIŞ bağlantı {FOTON_CONN_TARGET} ± {FOTON_CONN_TOL} dışında: {bad} — "
                                 f"eski O-X1 tasarımı (192.25 / 128.50 / 88.31) koda sızmış olabilir")
    return out


def foton_module_connections(cfg: ModelConfig, seq_len: int | None = None) -> dict[str, float]:
    """Rapor için: kolun her modülünün SAYILMIŞ ortalama nedensel bağlantısı, modül kimliğiyle anahtarlı."""
    return {k: float(v) for k, v in _foton_module_c(cfg, seq_len or cfg.max_seq_len).items()}


def foton_arm_connections(cfg: ModelConfig, seq_len: int | None = None) -> float:
    """Kolun birincil FLOP'ta kullanılan TEK c sabiti = kolun modülleri üzerinden maksimum.

    H'de üç modül de A'dır, maksimum A'nın kendisidir (T=512'de 128.17). X'te maksimum B'dir (128.50 = ön-kaydın
    "ortak hedef"i). Maksimum seçimi FOTON'un ALEYHİNE muhafazakârdır: router hangi karışımı seçerse seçsin
    gerçekleşen bağlam FLOP'u bu sabitle hesaplanandan büyük olamaz, ve sabit olduğu için router en ucuz
    modüle çökerek ölçülen tasarrufu şişiremez.
    """
    return float(max(_foton_module_c(cfg, seq_len or cfg.max_seq_len).values()))


def _foton_routed_context(cfg: ModelConfig, c: Fraction) -> int:
    """Yönlendirilen katmanların bağlam terimi: katman başına 4*d*c (skorlar 2*d*c + değer toplamı 2*d*c).
    Tam kesirle hesaplanıp EN SONDA yuvarlanır: çıktılar tamsayı kalır, yuvarlama hatası birikmez."""
    return round(cfg.foton_routed * 4 * cfg.d_model * c)


def forward_flops_per_token(cfg: ModelConfig, seq_len: int | None = None) -> dict[str, int]:
    """Token başına İLERİ yön aktif FLOP'u, bileşen bileşen (birincil tanım). seq_len: eğitim dizisi uzunluğu.

    Bilerek seçim karışımı PARAMETRESİ ALMAZ: FOTON kolunda değer yalnızca config'e bağlıdır (kol başına tek c).
    FOTON anahtarları (`foton_*`) her config'te vardır, FOTON olmayanlarda 0'dır (MoE anahtarlarının dense'te 0
    olmasıyla aynı sözleşme): kollar arası tablo anahtarla kurulur, eksik anahtar sessizce atlanmaz.
    """
    d, T, V = cfg.d_model, seq_len or cfg.max_seq_len, cfg.vocab_size
    n_dense, n_moe = _layer_counts(cfg)
    n_full_causal, foton_routed_ctx, foton_router = cfg.n_layer, 0, 0
    if cfg.foton:
        per_module = _foton_module_c(cfg, T)
        n_full_causal = cfg.foton_pre + cfg.foton_post      # paylaşılan bloklar: D'nin bloklarıyla birebir aynı
        foton_routed_ctx = _foton_routed_context(cfg, max(per_module.values()))
        # DİZİ başına tek d -> n_modül matris çarpımı (havuzlanmış prefix vektörü üzerinde): 2*d*n_modül / dizi.
        # Token başına saymak router'ı T kat şişirirdi (d=512'de 6 yerine 3072).
        foton_router = round(Fraction(2 * d * len(per_module), T))
    return {
        # q, k, v (tek d -> 3d matrisi) + çıkış projeksiyonu (d -> d): 2 * (3d^2 + d^2)
        "attn_qkv_proj": cfg.n_layer * 2 * 4 * d * d,
        # Causal: t. konum (0'dan) t+1 konuma bakar. Skorlar 2*d*(t+1), değerlerin ağırlıklı toplamı 2*d*(t+1).
        # T uzunluğundaki dizide konum ortalaması: 4*d*(T+1)/2 = 2*d*(T+1)
        # (FOTON'da yalnızca PAYLAŞILAN katmanlar; yönlendirilen katmanlar `foton_attn_routed`'da)
        "attn_context": n_full_causal * 2 * d * (T + 1),
        # SwiGLU: w_gate, w_up (d -> h), w_down (h -> d)
        "ffn_dense": n_dense * 2 * 3 * d * cfg.ffn_hidden,
        "moe_router": n_moe * 2 * d * cfg.n_experts,
        "moe_routed_experts": n_moe * cfg.top_k * 2 * 3 * d * cfg.expert_hidden,
        "moe_shared_experts": n_moe * cfg.n_shared_experts * 2 * 3 * d * cfg.expert_hidden,
        # Tied embedding'de de çıkış projeksiyonu tam bir d -> V matris çarpımıdır (giriş lookup'ı 0 FLOP)
        "lm_head": 2 * d * V,
        "foton_attn_routed": foton_routed_ctx,
        "foton_router": foton_router,
    }


def secondary_forward_flops_per_token(cfg: ModelConfig, seq_len: int | None = None) -> dict[str, float]:
    """Eleman bazlı işlemlerin YAKLAŞIK ileri yön FLOP'u (1 aritmetik işlem = 1 FLOP). Bütçelere girmez;
    birincil tanımın bunları dışarıda bırakmasının karşılaştırmayı ne kadar etkilediğini göstermek için."""
    d, T, V, nh = cfg.d_model, seq_len or cfg.max_seq_len, cfg.vocab_size, cfg.n_head
    n_dense, n_moe = _layer_counts(cfg)
    E, k, he = cfg.n_experts, cfg.top_k, cfg.expert_hidden
    mean_ctx = (T + 1) / 2
    silu_and_gate = 4  # SiLU: exp, 1+, bölme; çarpı "up" kolu
    out = {
        # RMSNorm (kare d, ortalama d, rsqrt 1, çarp d, kazanç d ~ 4d): katman başına 2 + son norm
        "rmsnorm": (2 * cfg.n_layer + 1) * 4 * d,
        "rope_qk": cfg.n_layer * 2 * 3 * d,             # q ve k: 2 çarpma + 1 toplama / eleman
        "attn_scale_softmax": cfg.n_layer * 4 * nh * mean_ctx,  # 1/sqrt(hd) + exp, toplam, bölme
        "residual_add": cfg.n_layer * 2 * d,
        "ffn_dense_act": n_dense * silu_and_gate * cfg.ffn_hidden,
        "moe_expert_act": n_moe * (k + cfg.n_shared_experts) * silu_and_gate * he,
        "moe_router_softmax_topk": n_moe * (3 * E + E + 2 * k),  # softmax, top-k taraması, gate normalizasyonu
        "moe_combine": n_moe * (2 * k * d + cfg.n_shared_experts * d),  # gate x çıktı + index_add, shared ekleme
        "moe_aux_z_loss": n_moe * 3 * E,
        "cross_entropy": 3 * V,
        # FOTON'a özgü eleman bazlı işlemler (FOTON olmayan config'te 0; türetim docs/flops.md bölüm 8.5)
        "foton_prefix_pool": 0.0,
        "foton_router_softmax_argmax": 0.0,
        "foton_gate_residual": 0.0,
        "foton_reg": 0.0,
    }
    if cfg.foton:
        per_module = _foton_module_c(cfg, T)
        n_mod, c = len(per_module), float(max(per_module.values()))
        if cfg.foton_prefix_k < 1:      # model de kurmaz (`GPT._build_foton`); 0 ya da negatif k havuzlama terimini bozar
            raise ValueError(f"foton_prefix_k >= 1 olmalı, verilen: {cfg.foton_prefix_k}")
        k_pre = min(cfg.foton_prefix_k, T)
        # Yönlendirilen katmanlarda softmax c bağlantı üzerinde (algoritmik sayım; birincil terimle aynı c)
        out["attn_scale_softmax"] = ((cfg.foton_pre + cfg.foton_post) * 4 * nh * mean_ctx
                                     + cfg.foton_routed * 4 * nh * c)
        # Prefix ortalaması: k*d toplama + d bölme, DİZİ başına bir kez -> / T
        out["foton_prefix_pool"] = (k_pre * d + d) / T
        # Router: softmax 3*n_mod (exp, toplam, bölme) + argmax taraması n_mod, DİZİ başına bir kez -> / T
        out["foton_router_softmax_argmax"] = 4 * n_mod / T
        # Gate + residual, modül çıkışında BİR kez (katman başına değil): (y - x) d çıkarma, gate çarpımı d,
        # x + (...) d toplama = 3*d / token; p / p.detach() dizi başına 1 bölme -> 1 / T
        out["foton_gate_residual"] = 3 * d + 1 / T
        # V7 regularizer'ı, DİZİ başına -> / T (batch boyutu ModelConfig'te olmadığı için mikro-batch başına
        # bir kez yapılan işlemler dizi başına ÜST SINIRLA yazılır):
        #   entropy: dizi başına clamp + log + çarpma + toplama = 4*n_mod (batch ortalaması ihmal: < 1 işlem)
        #   lb     : dizi başına P_m ortalamasına n_mod toplama; mikro-batch başına f_m ve P_m bölmeleri,
        #            f_m * P_m çarpımı ve toplamı ~ 4*n_mod + 1 işlem, mikro-batch >= 5 dizide dizi başına
        #            <= n_mod -> toplam 2*n_mod (bincount tamsayı işlemidir: 0 FLOP)
        per_seq = {"none": 0, "entropy": 4 * n_mod, "lb": 2 * n_mod}
        if cfg.foton_reg not in per_seq:
            raise ValueError(f"foton_reg {sorted(per_seq)} içinden olmalı, verilen: {cfg.foton_reg!r}")
        out["foton_reg"] = per_seq[cfg.foton_reg] / T
    return out


def train_flops_per_token(cfg: ModelConfig, seq_len: int | None = None) -> int:
    """Birincil tanımla eğitim (ileri + geri) FLOP'u / token = 3 x ileri."""
    return 3 * sum(forward_flops_per_token(cfg, seq_len).values())


def foton_flops_realized(cfg: ModelConfig, mix: Mapping[str, float], seq_len: int | None = None) -> int:
    """`train_flops_per_token` ile AYNI toplam, ama yönlendirilen bağlam terimi GERÇEKLEŞEN seçim karışımıyla
    ağırlıklı: c = sum_m mix[m] * c_m. YALNIZCA raporun duyarlılık satırı için; birincil istatistiğe GİRMEZ
    (ön-kayıt V3: karışımla hesap ancak modüller eşit FLOP'lu değilse ve sapma olarak yazılarak kullanılabilir).

    mix: `foton_module_keys(cfg)` ile BİREBİR aynı anahtarlı eşleme ("M0:A", ...), değerler dizilerin o modüle
    giden payı. Liste kabul edilmez (paralel liste yasağı, bkz. src/stats.py): sıra kayması payları yanlış
    modüle yazardı. Anahtar kümesi farklıysa, bir pay [0, 1] dışındaysa ya da toplam 1'den 1e-9'dan fazla
    sapıyorsa (NaN dahil) hata verir. Sonuç her zaman <= `train_flops_per_token(cfg)` (c_kol = maksimum).
    """
    if not cfg.foton:
        raise ValueError("foton_flops_realized yalnızca FOTON kollarında tanımlı (cfg.foton boş)")
    if not isinstance(mix, Mapping):
        raise TypeError(f"mix modül kimliğiyle anahtarlı bir eşleme olmalı, verilen: {type(mix).__name__}")
    T = seq_len or cfg.max_seq_len
    per_module = _foton_module_c(cfg, T)
    if set(mix) != set(per_module):
        raise ValueError(f"mix anahtarları kolun modülleriyle birebir aynı olmalı: beklenen {sorted(per_module)}, "
                         f"verilen {sorted(mix, key=str)}")
    for key, share in mix.items():
        if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0.0 <= share <= 1.0:
            raise ValueError(f"mix[{key!r}] = {share!r}: pay [0, 1] aralığında bir sayı olmalı")
    total = math.fsum(mix.values())
    if not abs(total - 1.0) <= 1e-9:
        raise ValueError(f"mix toplamı 1 olmalı (tolerans 1e-9), verilen toplam {total!r}")
    terms = forward_flops_per_token(cfg, seq_len)
    terms["foton_attn_routed"] = _foton_routed_context(
        cfg, sum((Fraction(mix[key]) * c for key, c in per_module.items()), Fraction(0)))
    return 3 * sum(terms.values())


# ------------------------------------------------------------------------------------------ docs tablosu
def _fmt(n: int) -> str:
    return f"{n:,}"


def foton_doc_tables(vocab_size: int = 32768) -> str:
    """docs/flops.md bölüm 8'in tabloları, HESAPLANARAK (elle yazılmış sayı yok).

    `tests/test_flops.py` bu çıktının her satırının docs/flops.md'de birebir durduğunu sınar: kod değişip belge
    güncellenmezse (ya da belgede bir rakam elle değiştirilirse) test düşer. Yeniden üretmek için:
        .venv\\Scripts\\python -m src.flops
    """
    from src.config import PRESETS
    arms = {"D": "foton-d", "M": "foton-m", "H": "foton-h", "X": "foton-x"}
    cfgs = {a: replace(PRESETS[p][0], vocab_size=vocab_size) for a, p in arms.items()}
    fwd = {a: forward_flops_per_token(c) for a, c in cfgs.items()}
    train = {a: train_flops_per_token(c) for a, c in cfgs.items()}
    T, d = cfgs["X"].max_seq_len, cfgs["X"].d_model
    lines = [f"| terim (ileri / token) | {' | '.join(arms)} |", "|---|" + "---:|" * len(arms)]
    for term in fwd["D"]:
        lines.append(f"| `{term}` | " + " | ".join(_fmt(fwd[a][term]) for a in arms) + " |")
    lines.append("| **ileri / token** | " + " | ".join(f"**{_fmt(sum(fwd[a].values()))}**" for a in arms) + " |")
    lines.append("| **eğitim / token (3×)** | " + " | ".join(f"**{_fmt(train[a])}**" for a in arms) + " |")
    lines.append("")
    lines += ["| karşılaştırma (eğitim / token) | fark (FLOP) | fark (%) |", "|---|---:|---:|"]
    for a, b in (("X", "H"), ("H", "D"), ("X", "D"), ("H", "M"), ("X", "M")):
        diff = train[a] - train[b]
        lines.append(f"| {a} − {b} | {diff:+,} | {100 * diff / train[b]:+.4f} |")
    lines.append("")
    lines += ["| kol | modül | sayılmış c (T=%d, n_head=%d) | kolda kullanılan c |" % (T, cfgs["X"].n_head),
              "|---|---|---:|---:|"]
    for a in ("H", "X"):
        arm_c = foton_arm_connections(cfgs[a])
        for key, c in foton_module_connections(cfgs[a]).items():
            lines.append(f"| {a} | `{key}` | {c:.6f} | {arm_c:.6f} |")
    lines.append("")
    keys = foton_module_keys(cfgs["X"])
    mixes = {f"hepsi `{k}`": {kk: float(kk == k) for kk in keys} for k in keys}
    mixes["eşit karışım (1/3, 1/3, 1/3)"] = {k: 1 / len(keys) for k in keys}
    lines += ["| X kolu, gerçekleşen karışım | eğitim / token | birincil X değerine göre (%) |", "|---|---:|---:|"]
    for label, mix in mixes.items():
        v = foton_flops_realized(cfgs["X"], mix)
        lines.append(f"| {label} | {_fmt(v)} | {100 * (v - train['X']) / train['X']:+.5f} |")
    lines.append("")
    # Çekirdek düzeyi: boolean maskeli SDPA yoğun T x T skor hesaplar; analitik sayım ALGORİTMİK bağlantı sayısıdır.
    dense_kernel = cfgs["X"].foton_routed * 4 * d * T
    lines += ["| yönlendirilen 2 katmanın bağlam terimi (ileri / token) | FLOP |", "|---|---:|"]
    lines.append(f"| analitik, H (c = {foton_arm_connections(cfgs['H']):.2f}) | {_fmt(fwd['H']['foton_attn_routed'])} |")
    lines.append(f"| analitik, X (c = {foton_arm_connections(cfgs['X']):.2f}) | {_fmt(fwd['X']['foton_attn_routed'])} |")
    lines.append(f"| D'nin aynı 2 katmanı, tam nedensel 2·2d(T+1) | {_fmt(2 * 2 * d * (T + 1))} |")
    lines.append(f"| maskeli çekirdeğin yoğun T×T hesabı, 2·4dT | {_fmt(dense_kernel)} |")
    lines.append("")
    # Yönlendirilen katmanlar ALGORİTMİK c yerine çekirdeğin yoğun T x T'siyle sayılsaydı (B13 üst sınırı)
    lines += ["| üst sınır: yönlendirilen katmanlar yoğun T×T sayılırsa | eğitim / token | D'ye göre (%) | M'e göre (%) |",
              "|---|---:|---:|---:|"]
    for a in ("H", "X"):
        v = train[a] + 3 * (dense_kernel - fwd[a]["foton_attn_routed"])
        lines.append(f"| {a} | {_fmt(v)} | {100 * (v - train['D']) / train['D']:+.4f} | "
                     f"{100 * (v - train['M']) / train['M']:+.4f} |")
    lines.append("")
    # Profiler ÖNGÖRÜSÜ (ölçüm değil): sayaç SDPA'yı maskeden bağımsız T x T sayar, ileri 4dT / katman ve geri
    # 2.5 katı (bölüm 7'de ölçülen 29,360,128 = 8 * 4dT * 3.5); matris çarpımlarını birebir sayar ('loop' yolu).
    lines += ["| profiler − analitik farkı, ÖNGÖRÜ (ölçüm değil) | profiler eğitim / token | analitik | fark (%) |",
              "|---|---:|---:|---:|"]
    for a in arms:
        matmul = sum(v for k, v in fwd[a].items() if k not in CONTEXT_KEYS)
        sdpa = cfgs[a].n_layer * 4 * d * T * 7 // 2
        rows = [(f"{a}{' (`loop`)' if cfgs[a].n_experts else ''}", 3 * matmul + sdpa)]
        if cfgs[a].n_experts:
            # Varsayılan `grouped` yolu: sayaç `_grouped_mm`'i 0 sayar -> routed expert'lerin FLOP'u hiç görünmez.
            # Bölüm 7'deki %26.7 V=8192'de ölçüldü; kampanya V'sindeki karşılığı buradan HESAPLANIR (lm_head büyüdüğü
            # için yüzde küçülür) — B13'ün 5 puan kuralı eski V'nin sayısıyla okunmasın.
            rows.append((f"{a} (`grouped`; `_grouped_mm` 0 sayılır)", 3 * (matmul - fwd[a]["moe_routed_experts"]) + sdpa))
        for label, prof in rows:
            lines.append(f"| {label} | {_fmt(prof)} | {_fmt(train[a])} | {100 * (prof - train[a]) / train[a]:+.3f} |")
    lines.append("")
    sec = {a: secondary_forward_flops_per_token(c) for a, c in cfgs.items()}
    lines += [f"| ikincil terim (ileri / token, yaklaşık) | {' | '.join(arms)} |", "|---|" + "---:|" * len(arms)]
    for term in sec["D"]:
        if term.startswith("foton_") or term == "attn_scale_softmax":
            lines.append(f"| `{term}` | " + " | ".join(f"{sec[a][term]:,.3f}" for a in arms) + " |")
    lines.append("| ikincil toplam / birincil ileri (%) | " + " | ".join(
        f"{100 * sum(sec[a].values()) / sum(fwd[a].values()):.4f}" for a in arms) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows konsolu cp1252
    print(foton_doc_tables())
