"""Decoder-only Transformer (nanoGPT iskeleti + modern Llama/DeepSeek bileşenleri).

Orijinal GPT-2'ye göre yaptığımız değişiklikler ve nedenleri:
  * Öğrenilen pozisyon embedding'i yerine RoPE  -> göreli konum bilgisi, parametre yok,
    uzun bağlama genişletilebilir (Llama, DeepSeek-V3, Kimi K2 hepsi RoPE kullanır).
  * LayerNorm yerine RMSNorm                    -> daha basit/hızlı, pratikte aynı kalite.
  * GELU FFN yerine SwiGLU                      -> aynı parametre bütçesinde daha düşük loss
    (Shazeer 2020). MoE adımında her "expert" tam olarak bu SwiGLU bloğunun bir kopyası olacak.
  * Bias yok                                    -> bias'lar büyük modellerde katkı sağlamıyor.
  * Pre-norm (norm, alt katmandan ÖNCE)         -> residual yolu "temiz" kalır, derin ağlarda
    gradyan akışı stabil olur; warmup'a daha az duyarlı.

FOTON v0.2 kolları (`ModelConfig.foton` = "h" | "x"; PREREGISTRATION_V02.md "MİMARİ", tag'de DONMUŞ):

    Embedding -> 3 paylaşılan blok -> NEDENSEL prefix router -> {M0, M1, M2} (2 katman) -> 3 paylaşılan blok -> LM head

  * Paylaşılan 6 blok D kolunun bloklarıyla BİREBİR aynıdır (tam nedensel attention, dense SwiGLU).
    FOTON'un baseline'dan farkı yalnızca yönlendirilen 2 katmanda yaşar.
  * Modüller aynı `Block`'tur; TEK fark attention maskesi (`src/attn_patterns.py`'den, config'ten DEĞİL).
    X: A yerel W=150 / B dilated r=2 W=256 / C yarım yerel W=256 + yarım strided s=4 W=256. H: üçü de A.
  * Router dizinin yalnızca İLK k=64 pozisyonunun gizli durumunu görür (nedensel prefix), dizi düzeyinde top-1.
  * `foton == ""` iken bu dosyadaki FOTON kodunun HİÇBİRİ çalışmaz ve modelin kurulma sırası (dolayısıyla
    init RNG tüketimi) değişmez: D ve M kolları dondurulmuş baseline koşularıyla aynı modeldir.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import attn_patterns
from src.config import ModelConfig
from src.moe import MoE

FOTON_ARMS = {"h": ("A", "A", "A"), "x": attn_patterns.FOTON_MODULES}     # modül indeksi 0,1,2 bu SIRADA
FOTON_CONN_TARGET = 128.4      # sahibin ZORUNLU DOĞRULAMA 1'i: üç modül de 128.4 ± 0.4 (T=512, nedensel, SAYILARAK)
FOTON_CONN_TOL = 0.4


class RMSNorm(nn.Module):
    """x / RMS(x) * g.  LayerNorm'dan farkı: ortalamayı çıkarmaz (re-centering yok), bias yok."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kareler ortalamasını float32'de al: bf16'da küçük/büyük değerlerde hassasiyet kaybı olur.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE için cos/sin tabloları, şekil (T, head_dim/2).

    Her boyut çifti (i) kendi frekansıyla döner: freq_i = theta^(-2i/d). Düşük i -> hızlı dönen
    (yakın konumları ayırt eden), yüksek i -> yavaş dönen (uzak bağlamı kodlayan) boyutlar.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """q veya k vektörünü konumuna göre döndürür. x: (B, H, T, hd).

    Neden işe yarar: q_m ile k_n döndürüldükten sonra iç çarpımları yalnızca (m - n) farkına
    bağlı olur -> attention skorları doğal olarak *göreli* konum bilgisi taşır.
    (GPT-NeoX/Llama "rotate_half" düzeni: vektörün ilk yarısı ile ikinci yarısı eşleşir.)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, pattern: str | None = None):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        # FOTON modül katmanı: attention grafiği `src/attn_patterns.PATTERNS[pattern]`. Maske, bağlantı
        # sayılarının hesaplandığı AYNI nesneden kurulur (tek doğruluk kaynağı): ön-kaydın ilk
        # tasarımında sayılan desen ile koşulan desen ayrı yerlerde duruyordu ve 192/128/88 farkı
        # öyle gözden kaçtı. [n_head, T, T] bool, True = izinli; persistent=False (checkpoint'e girmez,
        # her kurulumda desenden yeniden üretilir — eski bir maske checkpoint'ten geri yüklenemesin).
        self.pattern = pattern
        if pattern is not None:
            m = attn_patterns.PATTERNS[pattern].masks(cfg.max_seq_len, cfg.n_head)   # T ve n_head AÇIKÇA
            self.register_buffer("attn_mask", torch.from_numpy(m.copy()), persistent=False)
        # q, k, v tek matris çarpımında: 3 ayrı kernel çağrısı yerine 1 (daha az GPU overhead'i)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj.is_residual_proj = True  # init'te özel ölçek (bkz. GPT._init_weights)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # RoPE yalnızca q ve k'ye uygulanır: konum, "kime bakacağım" kararını etkilemeli,
        # taşınan içeriği (v) değil.
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # PyTorch SDPA: T x T attention matrisini bellekte tutmayan (flash/mem-efficient) çekirdek
        # seçer. is_causal=True -> her token yalnızca kendisi ve öncekilere bakar (otoregresif).
        if self.pattern is None:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Boolean maske: True = bu (sorgu i, anahtar j) çifti hesaba katılır. Maske nedenseldir
            # (attn_patterns kurarken j > i girdisi olmadığını doğrular), bu yüzden is_causal verilmez.
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=self.attn_mask[:, :T, :T],
                dropout_p=self.dropout if self.training else 0.0
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    """FFN(x) = W_down( SiLU(W_gate x) * W_up x ).

    Kapı (gate) kolu, yukarı projeksiyonun hangi boyutlarının geçeceğini veriye bağlı olarak
    seçer -> düz ReLU/GELU'dan daha ifade gücü yüksek. Transformer parametrelerinin ~2/3'ü
    FFN'lerdedir; MoE'nin çoğalttığı kısım da tam olarak burası.
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.w_down.is_residual_proj = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    """Pre-norm Transformer bloğu: x + Attn(Norm(x)), sonra x + FFN(Norm(x)).

    MoE'ye geçişte değişen TEK şey FFN kısmı: attention, normlar ve residual yapı aynı kalır.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int, pattern: str | None = None):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, pattern)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.is_moe = cfg.n_experts > 0 and layer_idx >= cfg.first_dense_layers
        if self.is_moe:
            shared_hidden = cfg.expert_hidden * cfg.n_shared_experts
            self.ffn = MoE(
                cfg.d_model, cfg.n_experts, cfg.top_k,
                make_expert=lambda: SwiGLU(cfg.d_model, cfg.expert_hidden),
                # n_shared adet shared expert, tek bir geniş SwiGLU'ya eşdeğer (hepsi her tokena
                # uygulandığı için ayrı tutmanın anlamı yok) - DeepSeek de böyle uygular.
                make_shared=(lambda: SwiGLU(cfg.d_model, shared_hidden)) if shared_hidden > 0 else None,
                norm_topk_prob=cfg.norm_topk_prob,
                impl=cfg.moe_impl,
                gate=cfg.router_gate,
                bias_rate=cfg.bias_update_rate,
            )
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin))
        if self.is_moe:
            h, stats = self.ffn(self.ffn_norm(x))
        else:
            h, stats = self.ffn(self.ffn_norm(x)), None
        x = x + self.drop(h)
        return x, stats


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.foton:
            self._build_foton(cfg)
        else:
            self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)  # son norm: logit ölçeğini stabil tutar
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # Weight tying: "token -> vektör" ve "vektör -> token" aynı V x d matrisini paylaşır.
            # Çıkış (softmax) tarafı her adımda TÜM satırlar için gradyan alır; giriş embedding'inde
            # ise yalnızca batch'te geçen tokenların satırı güncellenir. Paylaşınca nadir tokenların
            # GİRİŞ vektörleri de çıkış tarafından her adımda eğitilir (Press & Wolf, 2017).
            # Ayrıca V x d parametre tasarrufu: dense preset'te 4.2M (~%14); çok küçük modellerde
            # (tiny preset: ~%72) embedding modelin çoğunluğu olabilir.
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
        # persistent=False: checkpoint'e yazılmaz, config'ten her zaman yeniden hesaplanabilir
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        if cfg.foton:
            self.verify_foton_masks()

    # ------------------------------------------------------------------ FOTON v0.2
    def _build_foton(self, cfg: ModelConfig) -> None:
        """3 paylaşılan + (3 modül x 2 katman) + 3 paylaşılan blok ve prefix router'ı kurar.

        `self.blocks` yalnızca PAYLAŞILAN blokları tutar (önce `foton_pre`, sonra `foton_post`);
        modüller `self.foton_modules[m][j]`'dedir, m = FOTON_ARMS[cfg.foton] sırasındaki modül
        indeksi (X: 0=A, 1=B, 2=C; H: üçü de A deseni, AYRI ağırlık). `cfg.n_layer` bir dizinin
        GEÇTİĞİ derinliktir (8), kurulan blok sayısı (12) değil: residual projeksiyonlarının init
        ölçeği (2*n_layer)^-0.5 bu yüzden D/M ile birebir aynıdır.

        H ve X aynı parametre şekillerini aynı sırada kurar ve maskeler torch RNG'si TÜKETMEZ:
        aynı seed'de H ile X bit düzeyinde aynı başlangıç ağırlıklarından başlar.
        """
        if cfg.foton not in FOTON_ARMS:
            raise ValueError(f"foton {sorted(FOTON_ARMS)} içinden olmalı, verilen: {cfg.foton!r}")
        if cfg.n_experts > 0:
            raise ValueError("FOTON kolları dense SwiGLU kullanır (n_experts = 0): fark yalnızca yönlendirilen katmanlarda")
        if cfg.foton_pre + cfg.foton_routed + cfg.foton_post != cfg.n_layer or min(
                cfg.foton_pre, cfg.foton_routed, cfg.foton_post) < 1:
            raise ValueError(f"katman bölüşümü {cfg.foton_pre}+{cfg.foton_routed}+{cfg.foton_post} "
                             f"n_layer={cfg.n_layer} ile uyuşmuyor")
        if cfg.foton_prefix_k < 1:
            raise ValueError("foton_prefix_k >= 1 olmalı")
        if cfg.foton_reg not in ("none", "entropy", "lb"):
            raise ValueError(f"foton_reg 'none' | 'entropy' | 'lb' olmalı, verilen: {cfg.foton_reg!r}")
        pre, r = cfg.foton_pre, cfg.foton_routed
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(pre)]
                                    + [Block(cfg, pre + r + i) for i in range(cfg.foton_post)])
        self.foton_modules = nn.ModuleList(
            [nn.ModuleList([Block(cfg, pre + j, pattern=name) for j in range(r)])
             for name in FOTON_ARMS[cfg.foton]])
        # Router: TEK doğrusal katman d -> 3, bias yok (modelin tamamı bias'sız), sıcaklık 1, gürültü yok.
        # Init diğer bütün nn.Linear'larla aynı: N(0, 0.02) (`_init_weights`).
        self.foton_router = nn.Linear(cfg.d_model, len(self.foton_modules), bias=False)

    def foton_connection_counts(self) -> dict[str, float]:
        """Her modülün token/head başına ORTALAMA nedensel bağlantısı — GERÇEKLEŞEN maskeden SAYILARAK.

        Formülden değil, `F.scaled_dot_product_attention`'a verilen `attn_mask` buffer'ının True
        girdileri sayılarak (modülün her katmanı için; katmanlar aynı olmak zorunda). Anahtar:
        "M<indeks>:<desen>".
        """
        out = {}
        for m, (name, layers) in enumerate(zip(FOTON_ARMS[self.cfg.foton], self.foton_modules)):
            per_layer = [blk.attn.attn_mask.sum().item() / (blk.attn.attn_mask.shape[0] * blk.attn.attn_mask.shape[1])
                         for blk in layers]
            if max(per_layer) != min(per_layer):
                raise AssertionError(f"modül {m} ({name}): katmanların maskeleri farklı: {per_layer}")
            out[f"M{m}:{name}"] = per_layer[0]
        return out

    def verify_foton_masks(self) -> dict[str, float]:
        """İKİNCİ KİLİT: uygulanan desenler DONMUŞ spesifikasyona uyuyor mu — her model kurulumunda.

        Ön-kaydın O-X1 bölümünde ESKİ tasarım kaldı (A bitişik 256 / B dilated 256 / C 128+128 ->
        192.25 / 128.50 / 88.31 bağlantı: reddedilen compute confound'un kendisi). O metin koda
        sızarsa deney çöpe gider. Burada (i) donmuş desen parametreleri, (ii) her maskenin nedensel
        olduğu ve her sorgunun en az kendisini gördüğü, (iii) kampanya T'sinde (512) SAYILMIŞ
        bağlantının 128.4 ± 0.4 içinde olduğu denetlenir. Çıplak assert değil: `python -O` atlamasın.
        """
        frozen = (attn_patterns.A_LOCAL_W, attn_patterns.B_DILATION_R, attn_patterns.B_BUDGET_W,
                  attn_patterns.C_LOCAL_W, attn_patterns.C_STRIDE_S, attn_patterns.C_STRIDE_W)
        if frozen != (150, 2, 256, 256, 4, 256):
            raise AssertionError(f"attn_patterns donmuş spesifikasyondan sapmış: {frozen} != (150, 2, 256, 256, 4, 256)")
        for m, layers in enumerate(self.foton_modules):
            for blk in layers:
                mask = blk.attn.attn_mask
                if bool(torch.triu(mask, diagonal=1).any()):
                    raise AssertionError(f"modül {m}: maske nedensel değil (j > i girdisi var)")
                if not bool(torch.diagonal(mask, dim1=-2, dim2=-1).all()):
                    raise AssertionError(f"modül {m}: kendini görmeyen sorgu var (softmax NaN üretirdi)")
        counts = self.foton_connection_counts()
        if self.cfg.max_seq_len == attn_patterns.T_KAMPANYA:
            bad = {k: v for k, v in counts.items() if abs(v - FOTON_CONN_TARGET) > FOTON_CONN_TOL}
            if bad:
                raise AssertionError(f"SAYILMIŞ bağlantı {FOTON_CONN_TARGET} ± {FOTON_CONN_TOL} dışında: {bad} "
                                     f"(hepsi: {counts}) — eski O-X1 tasarımı koda sızmış olabilir")
        return counts

    def _foton_route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(seçim [B], olasılıklar [B,3], logit'ler [B,3]) — YALNIZCA ilk k pozisyondan.

        Girdi, yönlendirilen ilk katmanın girdisidir (paylaşılan ön bloklardan sonraki residual akış);
        o pozisyonların gizli durumu nedensel olarak yalnızca 0..k-1 token'larına bağlıdır, yani k'dan
        sonraki hiçbir token seçimi etkileyemez (`tests/test_foton_model.py` bunu sınar). Havuzlama
        eşit ağırlıklı ortalama; T < k ise eldeki pozisyonlar (kampanyada T hep 512'dir).
        Router float32'de, autocast DIŞINDA koşar (`src/moe.py` ile aynı gerekçe: top-1 kararı bf16
        yuvarlamasına duyarlı). Eşitlikte `argmax` en küçük indeksi seçer (deterministik).
        """
        k = min(self.cfg.foton_prefix_k, x.shape[1])
        pooled = x[:, :k, :].float().mean(dim=1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(pooled, self.foton_router.weight.float())
        probs = F.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs, logits

    def _foton_module(self, m: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        for blk in self.foton_modules[m]:
            x, _ = blk(x, cos, sin)
        return x

    def _foton_trunk(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, force_module: int | None):
        """Paylaşılan ön bloklar -> router -> seçilen modül -> paylaşılan arka bloklar."""
        pre = self.cfg.foton_pre
        for blk in self.blocks[:pre]:
            x, _ = blk(x, cos, sin)
        choice, probs, logits = self._foton_route(x)
        if force_module is not None:                       # oracle değerlendirmesi: bütün diziler TEK modülden
            x_out = self._foton_module(force_module, x, cos, sin)
        else:
            y = torch.zeros_like(x)
            for m in range(len(self.foton_modules)):
                sel = (choice == m).nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue        # bu mikro-batch'te dizi almayan modül ÇALIŞMAZ: gradyanı None kalır
                y = y.index_copy(0, sel, self._foton_module(m, x.index_select(0, sel), cos, sin).to(y.dtype))
            # Straight-through gate: DEĞER p/p.detach() = 1.0, GRADYAN d(log p). k=1'de ham p'yi normalize
            # etmek (p/p) sabit 1 verir ve router görev loss'undan hiç gradyan alamaz; ham p ile çarpmak ise
            # dalı init'te 1/3'e kısar. Gate, modülün residual akışa EKLEDİĞİ farkı çarpar (kimlik yolu
            # gate'lenmez): x_out = x + g * (Modül(x) - x).
            p_top = probs.gather(1, choice[:, None]).squeeze(1)
            gate = (p_top / p_top.detach()).to(x.dtype)
            x_out = x + gate[:, None, None] * (y - x)
        for blk in self.blocks[pre:]:
            x_out, _ = blk(x_out, cos, sin)
        return x_out, choice, probs, logits

    @torch.no_grad()
    def foton_eval_all(self, idx: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """ORACLE için: aynı batch'i üç modülün ÜÇÜNDEN de geçirir (v01 EK1-B12, dizi düzeyi, saf CE).

        Döner: `ce_tok` [3, B, T] float32 — modül m zorlanmışken pozisyon başına CE; `choice` [B];
        `probs` [B, 3]; `logits` [B, 3]. Paylaşılan ön bloklar ve router BİR kez koşar. Oracle'ın hangi
        pozisyonlar üzerinden seçtiği ve null'ın nasıl kurulduğu burada DEĞİL `src/train.py`'dedir;
        burası yalnızca ham ölçümü üretir (ham hâli diske de yazılır ki her okuma sonradan hesaplanabilsin).
        """
        B, T = idx.shape
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.tok_emb(idx)
        pre = self.cfg.foton_pre
        for blk in self.blocks[:pre]:
            x, _ = blk(x, cos, sin)
        choice, probs, logits = self._foton_route(x)
        ce_tok = []
        for m in range(len(self.foton_modules)):
            h = self._foton_module(m, x, cos, sin)
            for blk in self.blocks[pre:]:
                h, _ = blk(h, cos, sin)
            lg = self.lm_head(self.norm(h))
            ce_tok.append(F.cross_entropy(lg.float().view(-1, lg.size(-1)), targets.reshape(-1),
                                          ignore_index=-1, reduction="none").view(B, T))
            del lg, h
        return {"ce_tok": torch.stack(ce_tok), "choice": choice, "probs": probs, "logits": logits}

    def _init_weights(self, module: nn.Module) -> None:
        # GPT-2 başlatması: N(0, 0.02). Residual'a yazan projeksiyonlar ek olarak
        # 1/sqrt(2*n_layer) ile küçültülür: her blok residual'a 2 katkı ekler, bu ölçek sayesinde
        # derinlik arttıkça residual stream'in varyansı başlangıçta patlamaz.
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "is_residual_proj", False):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, exclude_embedding: bool = False, active_only: bool = False) -> int:
        """active_only: token başına gerçekten çalışan parametreler. MoE katmanında E expert'ten
        yalnızca top_k tanesi çalışır; kalan (E - k) expert'in parametreleri o token için "uyur"."""
        n = sum(p.numel() for p in self.parameters())  # parameters() paylaşılan ağırlığı 1 kez sayar
        if exclude_embedding:
            n -= self.tok_emb.weight.numel()
        if active_only:
            for block in self.blocks:
                if block.is_moe:
                    per_expert = sum(p.numel() for p in block.ffn.experts[0].parameters())
                    n -= (block.ffn.n_experts - block.ffn.top_k) * per_expert
            if self.cfg.foton:      # dizi başına TEK modül çalışır: diğer ikisinin ağırlıkları o dizi için uyur
                per_module = sum(p.numel() for p in self.foton_modules[0].parameters())
                n -= (len(self.foton_modules) - 1) * per_module
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, force_module: int | None = None):
        """Döndürür: (logits, loss, info)
        loss = CE + aux_coef * aux_loss + z_coef * z_loss   (dense modelde sadece CE)
        info = {"ce", ve MoE varsa "aux_loss", "z_loss", "counts" (MoE katmanı x expert)} - hepsi detached.
        Model kalitesini karşılaştırırken info["ce"]'ye bakın: loss, MoE'de ceza terimlerini de içerir.
        """
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"dizi uzunluğu {T} > max_seq_len {self.cfg.max_seq_len}"
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        moe_stats = []
        foton = None
        if self.cfg.foton:
            x, choice, probs, logits_r = self._foton_trunk(x, cos, sin, force_module)
            foton = (choice, probs, logits_r)
        else:
            if force_module is not None:
                raise ValueError("force_module yalnızca FOTON kollarında anlamlı")
            for block in self.blocks:
                x, stats = block(x, cos, sin)
                if stats is not None:
                    moe_stats.append(stats)
        x = self.norm(x)

        if targets is None:
            # Üretimde yalnızca son konumun logit'i gerekir: V x T'lik tensörü boşuna hesaplama
            return self.lm_head(x[:, [-1], :]), None, {}
        logits = self.lm_head(x)
        # Cross-entropy float32'de: bf16 log-softmax büyük sözlükte sayısal olarak hassas değil
        # targets.reshape: veri yükleyicideki y = xy[:, 1:] bitişik olmayan bir görünümdür, view çalışmaz
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        loss, info = ce, {"ce": ce.detach()}
        if moe_stats:
            # Katmanların ORTALAMASI (HF Mixtral/Qwen-MoE gibi; Switch Transformer ve DeepSeek ise
            # katman başına ekler = TOPLAM). Ortalamada loss'taki aux teriminin büyüklüğü ~aux_loss_coef
            # kalır ve log'daki aux değeri derinlikten bağımsız ~1 okunur. Bedeli: her router'ın
            # gördüğü etkin katsayı aux_loss_coef / n_moe (moe preset'inde 0.01/7 ≈ 0.0014), yani MoE
            # katmanı sayısı değişirse router başına dengeleme gücü de değişir. Bu zayıf katsayıyla
            # bile denge sağlandı (bkz. README sonuçları).
            aux = torch.stack([s["aux_loss"] for s in moe_stats]).mean()
            z = torch.stack([s["z_loss"] for s in moe_stats]).mean()
            loss = ce + self.cfg.aux_loss_coef * aux + self.cfg.z_loss_coef * z
            info.update(aux_loss=aux.detach(), z_loss=z.detach(),
                        counts=torch.stack([s["counts"] for s in moe_stats]))
        if foton is not None:
            choice, probs, logits_r = foton
            # V7 ablasyonu. "none" koşulunda router'a HİÇBİR ek terim gelmez (z-loss dahil).
            if self.cfg.foton_reg == "entropy":
                ent = -(probs * torch.log(probs.clamp_min(1e-20))).sum(dim=-1).mean()
                loss = loss - self.cfg.foton_reg_coef * ent          # BONUS: dizi başına entropiyi ödüllendirir
                info["foton_reg"] = ent.detach()
            elif self.cfg.foton_reg == "lb":
                M = probs.shape[1]
                f = torch.bincount(choice, minlength=M).to(probs.dtype) / choice.numel()
                aux = M * (f.detach() * probs.mean(dim=0)).sum()     # Switch: N * sum_m f_m * P_m, bu mikro-batch'te
                loss = loss + self.cfg.foton_reg_coef * aux
                info["foton_reg"] = aux.detach()
            info.update(foton_choice=choice.detach(), foton_probs=probs.detach(), foton_logits=logits_r.detach())
        return logits, loss, info

    def configure_optimizer(self, lr: float, weight_decay: float, betas: tuple[float, float], device_type: str):
        # Weight decay yalnızca 2+ boyutlu tensörlere (matris çarpımı ağırlıkları, embedding).
        # Norm kazançları (1-boyutlu) sıfıra çekilmemeli: onlar ölçek parametresi, "karmaşıklık" değil.
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # fused=True: tüm parametre güncellemesini tek CUDA çekirdeğinde yapar (daha hızlı)
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Otoregresif örnekleme. Basitlik için KV-cache yok: her adımda bağlamı baştan işler.
        (512 token ve ~30M parametrede bu yeterince hızlı; KV-cache ileride eklenebilir.)"""
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)  # greedy
            else:
                logits = logits / temperature  # <1: daha keskin/tutarlı, >1: daha çeşitli/rastgele
                if top_k is not None and top_k > 0:  # top_k <= 0 veya None: filtre kapalı
                    # Olasılığı en yüksek k token dışındakileri ele: "kuyruk"taki saçma tokenları kes
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None:
                # Batch'teki her örnek kendi EOS'una farklı adımda ulaşır; hepsi bitince dur.
                # (EOS sonrası üretilenleri çağıran taraf keser.)
                finished |= next_id.squeeze(1) == eos_id
                if bool(finished.all()):
                    break
        return idx
