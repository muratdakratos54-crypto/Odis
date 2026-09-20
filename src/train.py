"""Eğitim döngüsü.

Kullanım (proje kökünden):
    .venv\\Scripts\\python -m src.train --preset tiny            # birkaç saniyelik duman testi
    .venv\\Scripts\\python -m src.train --preset dense           # ~30M dense model
    .venv\\Scripts\\python -m src.train --preset dense --resume true

Bir optimizer adımının anatomisi:
    grad_accum kez: batch al -> bf16 autocast altında forward -> loss/grad_accum -> backward
    -> gradyan normunu 1.0'a kırp -> AdamW adımı -> LR'ı schedule'a göre güncelle
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, replace

import torch

from src.checkpoint import save_atomic
from src.config import TrainConfig, build_configs
from src.data import TokenDataset, dataset_paths
from src.data_multi import MultiDomainDataset, multi_domain_paths
from src.flops import forward_flops_per_token, train_flops_per_token
from src.model import GPT
from src.moe import load_balance_metrics, load_sparkline, update_expert_biases
from src.tokenizer import load_tokenizer


def token_budget(cfg: TrainConfig, tokens_per_step: int) -> tuple[int, int]:
    """(warmup_tokens, max_tokens). 0 bırakılan alan adım sayısından türetilir."""
    warmup = cfg.warmup_tokens or cfg.warmup_steps * tokens_per_step
    total = cfg.max_tokens or cfg.max_steps * tokens_per_step
    if not 0 < warmup < total:
        raise ValueError(f"warmup ({warmup} token) 0 ile toplam bütçe ({total} token) arasında olmalı")
    return warmup, total


def lr_at(tokens_seen: int, tokens_per_step: int, warmup_tokens: int, max_tokens: int, cfg: TrainConfig) -> float:
    """GÖRÜLEN TOKEN'a göre lineer warmup + cosine decay. tokens_seen: bu adımdan önce görülen token.

    Warmup: başlangıçta Adam'ın moment tahminleri henüz oturmamıştır ve ağırlıklar rastgeledir;
    büyük LR ile ilk adımlar modeli kararsız bir bölgeye fırlatabilir. Cosine decay: sona doğru
    LR'ı düşürmek, loss vadisinin dibine yerleşmeyi sağlar (sabit LR'a göre belirgin kazanç).

    Neden adım değil token: eşit süre karşılaştırmasında hızlı model aynı sürede daha çok adım atar.
    Schedule adıma bağlı olsaydı warmup'ın uzunluğu ve decay'in nerede bittiği modelin hızına göre
    değişirdi; token'a bağlıyken her model warmup'ı aynı token'da bitirir ve LR kendi bütçesinin sonunda
    min_lr'a iner. (Token/adım bütün koşularda aynıyken bu, her modele kendi max_steps'ini veren bir adım
    schedule'ına eşdeğerdir; bütçeyi token olarak tanımlamak onu hıza göre seçmeyi doğrudan yapar.)

    v1 ile uyum: warmup'ta bu adımın tokenları da sayılır (v1'deki step + 1). Token/adım 2'nin kuvvetiyken
    (tüm presetler: 32 768, 2 048) LR dizisi v1'in adım formülüyle bit bit aynıdır; başka değerlerde warmup LR'ları
    1 ulp farklı olabilir, cosine kısmı yine bit bit aynıdır (bkz. tests/test_train.py). Warmup bir adımın ortasında
    biterse LR tepe değerini aşmasın diye kırpılır.
    """
    if tokens_seen < warmup_tokens:
        return cfg.lr * min(tokens_seen + tokens_per_step, warmup_tokens) / warmup_tokens
    progress = min(1.0, (tokens_seen - warmup_tokens) / max(1, max_tokens - warmup_tokens))
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


# Oracle NULL'ının permütasyon tohumu. Sabit: null eğrisi her eval noktasında, her koşuda ve her kolda AYNI
# permütasyonla kurulur (tek bir çekim; eğri gürültüsü eval noktaları arasında ortak kalsın diye).
ORACLE_NULL_SEED = 54321
_PERMS3 = torch.tensor([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])


def oracle_stats(ce_tok: torch.Tensor, choice: torch.Tensor, k: int, perm_idx: torch.Tensor) -> dict[str, torch.Tensor]:
    """Bir batch'in oracle nicelikleri. `ce_tok` [3, B, T]: modül m ZORLANMIŞKEN pozisyon başına CE.

    * `oracle`  (v01 EK1-B12, O8): dizi başına, ÖLÇÜT pozisyonları (k..T-1) üzerindeki token-ortalamalı saf
      CE'si en küçük olan modül seçilir. Seçim ölçütle AYNI pozisyonlarda yapılır; ancak o zaman "oracle eğrisi
      FOTON'un kendi eğrisinin üstünde olamaz" (ön-kayıt, oracle bölümü) doğrudur: router'ın seçtiği modül de
      adaylar arasındadır.
    * `null`    (v01 EK1-B11: "aynı oracle prosedürü, modül çıktıları birbirine permüte edilerek — seçimin
      bilgi taşımadığı durumda"). Okuma: her dizide, her POZİSYONDA üç modülün token kayıpları kendi aralarında
      rastgele permüte edilir; böylece üç "sözde-modül" değiş-tokuş edilebilir olur (hangisinin bu dizi için
      gerçekten iyi olduğu bilgisi silinir) ama en küçüğünü seçmenin GÜRÜLTÜDEN kazandırdığı pay kalır. Aynı
      oracle prosedürü (dizi başına en küçük ortalama) sözde-modüllere uygulanır. B11'in gerekçesi tam bu:
      "modüller istatistiksel olarak aynı olsa bile seçim eğriyi aşağı çeker". Metnin en düz okuması (modül
      ETİKETLERİNİ dizi içinde permüte etmek) minimumu değiştirmez (null = oracle, kazanç hep 0); kayıpları
      DİZİLER arasında karıştırmak ise alanlar arası ~0.9 nat'lık farkı seçime sokar (null << oracle). İkisi de
      vazgeçme koşulunun oracle bacağını anlamsız kılar. Bu okuma tartışmalı olabileceği için HAM veri de
      diske yazılır (`eval_raw/`): başka her okuma koşulardan sonra yeniden hesaplanabilir.
    * `routed`  : router'ın seçtiği modülün dizi-ortalaması — `val_loss`'un İKİNCİ yoldan hesabı (tutarlılık).
    """
    seq = ce_tok[:, :, k:].mean(dim=-1)                                   # [3, B]
    pseudo = ce_tok.gather(0, _PERMS3.to(ce_tok.device)[perm_idx].permute(2, 0, 1))   # [3, B, T]
    return {"seq_masked": seq, "seq_full": ce_tok.mean(dim=-1),
            "oracle": seq.min(dim=0).values, "null": pseudo[:, :, k:].mean(dim=-1).min(dim=0).values,
            "routed": seq.gather(0, choice[None, :])[0]}


ASSIGN_SEED_OFFSET = 7_777_777      # X-iid'in atama generator'ı = seed + bu sabit: `train_gen`'den AYRI (veri sırası kaymasın)


def make_assigner(mode: str, n_modules: int, domain_names: list[str] | None, seed: int):
    """v03 adım 1 — dışarıdan atama kuralı: `(alan kimlikleri [B] | None, batch boyutu, cihaz) -> assign [B] int64`.

    "domain": modül = alan kimliği. Alan sayısı modül sayısına EŞİT olmak zorunda (kod → M0, math → M1, nl → M2: alan
    kimliği `MultiDomainDataset.names` = sorted sırasıdır). "iid": çekim başına BAĞIMSIZ düzgün rastgele modül; kendi CPU
    generator'ından — `train_gen`'den ya da global RNG'den tek çekim yapmaz, yoksa aynı seed'li kollarla veri sırası /
    init eşlemesi bozulurdu. Generator döndürülür ki checkpoint'e yazılabilsin."""
    if mode == "domain":
        if not domain_names or len(domain_names) != n_modules:
            raise ValueError(f"alan ataması {n_modules} alan ister, korpusta: {domain_names}")

        def by_domain(domain_ids, batch_size, device):
            if domain_ids is None or domain_ids.shape != (batch_size,):
                raise ValueError("alan ataması için batch'in alan kimlikleri gerekli")
            return domain_ids.to(device=device, dtype=torch.long)
        return by_domain, None
    if mode == "iid":
        gen = torch.Generator().manual_seed(ASSIGN_SEED_OFFSET + seed)

        def iid(domain_ids, batch_size, device):
            return torch.randint(n_modules, (batch_size,), generator=gen).to(device)
        return iid, gen
    raise ValueError(f"bilinmeyen atama kuralı: {mode!r}")


def two_routes_agree(a: float, b: float, tol: float) -> bool:
    """İki yoldan hesaplanan kayıp `tol` içinde aynı mı. Sonlu olmayan değer UYUŞMAZLIKTIR: `abs(nan - x) > tol` False
    verir, yani çıplak karşılaştırma NaN'ı "uyuşuyor" sayar ve koşu sonuna kadar GPU harcayıp ancak analizde reddedilirdi."""
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


class AssignLog:
    """Zorlanmış atama kollarının log aralığı istatistikleri — GPU'da, `.item()`'sız. Router metrikleri bu kollarda
    UYGULANAMAZ (router çalışmıyor); onların yerine atamanın kendisi ve gradyansız kalan modüller sayılır:

      assign_counts / assign_domain_counts : ham sayımlar (modül; alan x modül)
      assign_microbatches_missing_module   : en az bir modülün HİÇ dizi almadığı mikro-batch sayısı
      assign_steps_gradless_module         : en az bir modülün adımın bütün mikro-batch'lerinde dizi almadığı (gradyanı
                                             None kalan, AdamW'nin tümüyle atladığı) optimizer adımı sayısı
    """

    def __init__(self, n_modules: int, n_domains: int, device: str):
        self.M, self.D = n_modules, max(1, n_domains)
        self.counts = torch.zeros(self.D, self.M, device=device)
        self.step_counts = torch.zeros(self.M, device=device)
        self.missing_mb = torch.zeros((), device=device)
        self.gradless_steps = torch.zeros((), device=device)

    def add(self, assign: torch.Tensor, domain_ids: torch.Tensor | None) -> None:
        dom = domain_ids if domain_ids is not None and self.D > 1 else torch.zeros_like(assign)
        self.counts += torch.zeros_like(self.counts).index_put_((dom, assign), torch.ones_like(assign, dtype=self.counts.dtype),
                                                                accumulate=True)
        per_module = torch.bincount(assign, minlength=self.M).to(self.step_counts.dtype)
        self.step_counts += per_module
        self.missing_mb += (per_module == 0).any()

    def end_step(self) -> None:
        self.gradless_steps += (self.step_counts == 0).any()
        self.step_counts.zero_()

    def flush(self) -> dict:
        c = self.counts.double().cpu()
        out = {"assign_counts": [int(v) for v in c.sum(dim=0).tolist()],
               "assign_domain_counts": [[int(v) for v in row] for row in c.tolist()],
               "assign_microbatches_missing_module": int(self.missing_mb.item()),
               "assign_steps_gradless_module": int(self.gradless_steps.item())}
        self.counts.zero_()
        self.missing_mb.zero_()
        self.gradless_steps.zero_()
        return out


ROUTER_FIELDS_NA = ("router_entropy_marginal", "router_entropy_per_seq", "router_margin", "router_load_cv", "router_dead",
                    "router_switch_rate", "router_counts", "router_domain_counts")


@torch.no_grad()
def evaluate(model: GPT, ds: TokenDataset, cfg: TrainConfig, autocast_ctx) -> dict:
    """Val loss. `{"full": ..., "primary": ...}` döner; FOTON kollarında ayrıca oracle nicelikleri.

    FOTON (`model.cfg.foton`): `oracle`, `oracle_null`, `routed_check` (skaler) ve `raw` (dizi başına ham
    ölçüm: üç modülün pozisyon başına CE'si float16, dizi ortalamaları float32, router seçimi / olasılıkları /
    logit'leri, alan kimliği). `val_loss`'un KENDİSİ dört kolda da aynı koddan gelir (`model(x, y)`); oracle
    geçişi ek bir ölçümdür ve karar metriğine dokunmaz.

    `full` = bütün pozisyonlar. `primary` = `cfg.loss_prefix_skip` kadar ilk pozisyon ATLANMIŞ hâli;
    karar bu ikincisine göre verilir (bkz. `TrainConfig.loss_prefix_skip`: dizi düzeyi router'ın
    nedenselliği). `loss_prefix_skip == 0` iken ikisi aynı sayıdır.

    İkisi de loglanıyor çünkü `full`, maskesiz ön-kayıttan gelen dondurulmuş eğrilerle
    KARŞILAŞTIRILABİLİR: yeniden koşumun sadık olduğunu kanıtlayan provenance kontrolü odur.
    """
    model.eval()
    # Her değerlendirmede *aynı* pencereleri kullan (sabit seed): böylece val loss'taki değişim
    # örnekleme gürültüsünden değil, modelin gerçekten iyileşmesinden gelir.
    #
    # Çok alanlı korpusta pencere kümesi `batch_size`'dan BAĞIMSIZ olmak zorunda: `get_batch`
    # generator'ı batch_size'a bağlı tükettiği için mikro-batch'i değiştirmek pencerelerin
    # kimliğini de değiştiriyordu, ve alanlar arası 0.9 nat fark olduğu için bu ölçülen val
    # loss'u kaydırıp `L*`'i bölüşüme bağlı hale getiriyordu (ön-kayıt V1-ii). Dondurulmuş
    # akış referans bölüşümün sırasını birebir tekrar oynatır, yani `L*` değişmez.
    k = cfg.loss_prefix_skip
    foton = bool(getattr(model, "cfg", None) and model.cfg.foton)
    # v03 adım 1 — ZORLANMIŞ atama kolları: kolun val loss'u EĞİTİMDEKİ atama kuralıyla hesaplanır.
    #   "domain": `model(x, y, assign=alan kimliği)`.  "iid": düzgün rastgele atamanın BEKLENEN değeri = üç modülün
    #   ortalaması (tek bir rastgele çekimin gürültüsü ölçüme girmesin).
    # İKİ AYRI KOD YOLU, iki kolda da: 1. yol EĞİTİMİN forward yoludur (`model(x, y, assign=...)`; iid'de m = 0..M-1 sabit
    # atamalarının ortalaması), 2. yol üç-modül geçişidir (`foton_eval_all`). İlk hâlde iid'in iki "yolu" aynı tensörün iki
    # okunuşuydu: sağlama BAŞARISIZ OLAMIYORDU ve X-iid'in eğitimde kullandığı dispatch yolu hiçbir eval'de sınanmıyordu.
    # Router'ın (eğitilmemiş) seçimiyle hesaplanan sayı `router_unused` olarak döner ve hiçbir yerde kol kaybı sayılmaz.
    assign_mode = model.cfg.foton_assign if foton else ""
    null_gen = torch.Generator().manual_seed(ORACLE_NULL_SEED)
    fo = {"oracle": 0.0, "null": 0.0, "routed": 0.0, "assigned": 0.0, "assigned_full": 0.0}
    raw: dict[str, list] = {n: [] for n in ("ce_tok", "seq_masked", "seq_full", "choice", "probs", "logits", "domain")}

    def oracle_pass(x: torch.Tensor, y: torch.Tensor) -> None:
        """Aynı batch üç modülün üçünden: oracle / null / ham döküm. `both()`'tan SONRA çağrılır ki onun
        logit tensörü serbest kalmış olsun (eval tepe belleği baseline'larınkini aşmasın)."""
        with autocast_ctx:
            ev = model.foton_eval_all(x, y)
        perm_idx = torch.randint(len(_PERMS3), tuple(y.shape), generator=null_gen).to(y.device)
        st = oracle_stats(ev["ce_tok"], ev["choice"], k, perm_idx)
        for n in ("oracle", "null", "routed"):
            fo[n] += st[n].mean().item()
        if assign_mode:
            # İKİNCİ YOL (eval yolundan bağımsız): aynı atama kuralı, üç-modül geçişinin dizi ortalamalarından
            dom_ids = getattr(ds, "last_domain_ids", None)
            for name, seq in (("assigned", st["seq_masked"]), ("assigned_full", st["seq_full"])):     # seq: [3, B]
                fo[name] += (seq.gather(0, dom_ids[None, :].to(seq.device)).mean() if assign_mode == "domain"
                             else seq.mean()).item()
        raw["ce_tok"].append(ev["ce_tok"].permute(1, 0, 2).to(torch.float16).cpu())      # [B, 3, T]
        raw["seq_masked"].append(st["seq_masked"].T.float().cpu())
        raw["seq_full"].append(st["seq_full"].T.float().cpu())
        for n in ("choice", "probs", "logits"):
            raw[n].append(ev[n].cpu())
        dom = getattr(ds, "last_domain_ids", None)
        raw["domain"].append(dom.cpu() if dom is not None else torch.full((y.shape[0],), -1))

    def finish(tf: float, tp: float, nb: int) -> dict:
        out = {"full": tf / nb, "primary": tp / nb}
        if foton:
            out.update(oracle=fo["oracle"] / nb, oracle_null=fo["null"] / nb,
                       raw={n: torch.cat(v).numpy() for n, v in raw.items()})
            if assign_mode:
                out.update(assign_mode=assign_mode, assigned_check=fo["assigned"] / nb, router_unused=fo["routed"] / nb)
            else:
                out["routed_check"] = fo["routed"] / nb
        return out

    def both(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        """Aynı batch'te tam ve maskeli CE — TEK forward'dan.

        İkinci bir forward yapmak eval maliyetini ikiye katlardı; bunun yerine logit'lerden
        pozisyon başına CE'yi bir kez hesaplayıp iki ortalamayı alıyoruz. `model(x, y)`'nin
        döndürdüğü skaler `ce` ile `per_pos.mean()` aynı niceliktir (aşağıda test ediliyor).
        """
        if assign_mode == "iid":
            # Beklenen değer, EĞİTİMİN forward yolundan: bütün batch sırayla her modüle atanır, M sonucun ortalaması alınır.
            n_mod = len(model.foton_modules)
            tot_full = tot_masked = 0.0
            for m in range(n_mod):
                with autocast_ctx:
                    logits, _, info = model(x, y, assign=torch.full((x.shape[0],), m, dtype=torch.long, device=x.device))
                per_pos = torch.nn.functional.cross_entropy(
                    logits.float().view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1, reduction="none").view(y.shape)
                tot_full += info["ce"].item()
                tot_masked += per_pos[:, k:].mean().item()
                del logits
            return tot_full / n_mod, tot_masked / n_mod
        with autocast_ctx:
            if assign_mode == "domain":
                logits, _, info = model(x, y, assign=ds.last_domain_ids.to(dtype=torch.long))
            else:
                logits, _, info = model(x, y)
        full = info["ce"].item()
        if k <= 0:
            return full, full
        per_pos = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=-1, reduction="none").view(y.shape)
        return full, per_pos[:, k:].mean().item()

    if isinstance(ds, MultiDomainDataset):
        total_windows = cfg.batch_size * cfg.eval_iters
        pairs = ds.frozen_val_windows(total_windows, 12345)
        batches = [pairs[i : i + cfg.batch_size] for i in range(0, total_windows, cfg.batch_size)]
        tf = tp = 0.0
        # Val kümesinin PARMAK İZİ: değerlendirilen pencerelerin TOKEN İÇERİĞİNİN özeti (indislerin değil — dosya yeniden
        # üretilirse indisler aynı kalır, içerik değişir). Dense kol ham döküm yazmadığı için "D, FOTON kollarıyla aynı val
        # kümesinde ölçüldü" iddiası yalnızca bu alandan VERİDEN doğrulanabilir. Modele ve kayba dokunmaz.
        digest = hashlib.sha256()
        for chunk in batches:
            x, y = ds.windows(chunk)
            digest.update(x.cpu().numpy().tobytes())
            digest.update(y[:, -1].cpu().numpy().tobytes())
            f, p = both(x, y)
            tf += f
            tp += p
            if foton:
                oracle_pass(x, y)
        model.train()
        return {**finish(tf, tp, len(batches)), "val_windows_sha256": digest.hexdigest()}

    g = torch.Generator().manual_seed(12345)
    tf = tp = 0.0
    for _ in range(cfg.eval_iters):
        # Toplam loss değil, saf cross-entropy: MoE'nin aux/z ceza terimleri model kalitesini
        # ölçmez; dense ile MoE'yi adil karşılaştırmak için yalnızca CE'ye bakıyoruz.
        x, y = ds.get_batch(cfg.batch_size, g)
        f, p = both(x, y)
        tf += f
        tp += p
        if foton:
            oracle_pass(x, y)
    model.train()
    return finish(tf, tp, cfg.eval_iters)


class RouterLog:
    """FOTON router'ının log aralığı boyunca biriken istatistikleri (ön-kayıt V6) — GPU'da, `.item()`'sız.

    Alan adları ön-kayıtta pinli. Tanımlar (metnin açık bıraktıkları burada sabitlenir ve HAM sayımlar da
    loglanır ki başka her okuma sonradan hesaplanabilsin):
      router_entropy_marginal : H(modül) / log M, aralıktaki dizi seçimlerinin marjinal dağılımından
      router_entropy_per_seq  : dizi başına softmax entropisinin ortalaması / log M
      router_margin           : top-1 ile top-2 LOGIT farkının ortalaması
      router_load_cv          : üç modülün yük paylarının değişim katsayısı (POPÜLASYON std; tam çökme = sqrt(2))
      router_dead             : ideal payın (1/M) %10'undan azını alan modül sayısı (`src/moe.py` ile AYNI eşik)
      router_switch_rate      : ardışık iki log aralığında p(modül | alan) dağılımları arasındaki toplam
                                değişim uzaklığının (TV) alanlar üzerinden ortalaması; ilk aralıkta None.
                                (Eğitim pencereleri tekrar etmediği için "aynı dizinin modül değiştirmesi"
                                tanımsızdır; ölçülebilen, aynı ALANIN dizilerinin dağılımındaki kaymadır.)
      router_counts / router_domain_counts : ham sayımlar (modül; alan x modül)
    """

    def __init__(self, n_modules: int, n_domains: int, device: str):
        self.M, self.D = n_modules, max(1, n_domains)
        self.counts = torch.zeros(self.D, self.M, device=device)
        self.ent = torch.zeros((), device=device)
        self.margin = torch.zeros((), device=device)
        self.prev: torch.Tensor | None = None

    def add(self, info: dict, domain_ids: torch.Tensor | None) -> None:
        choice, probs, logits = info["foton_choice"], info["foton_probs"], info["foton_logits"]
        dom = domain_ids if domain_ids is not None and self.D > 1 else torch.zeros_like(choice)
        self.counts += torch.zeros_like(self.counts).index_put_((dom, choice), torch.ones_like(choice, dtype=self.counts.dtype),
                                                                accumulate=True)
        self.ent += -(probs * torch.log(probs.clamp_min(1e-20))).sum(dim=-1).sum()
        top2 = logits.topk(2, dim=-1).values
        self.margin += (top2[:, 0] - top2[:, 1]).sum()

    def flush(self) -> dict:
        c = self.counts.double().cpu()
        n = c.sum().item()
        share = (c.sum(dim=0) / n)
        nz = share[share > 0]
        dist = c / c.sum(dim=1, keepdim=True).clamp_min(1)
        switch = None
        if self.prev is not None:
            seen = (c.sum(dim=1) > 0) & (self.prev.sum(dim=1) > 0)
            if bool(seen.any()):
                switch = float((0.5 * (dist - self.prev).abs().sum(dim=1))[seen].mean())
        out = {"router_entropy_marginal": float(-(nz * nz.log()).sum() / math.log(self.M)),
               "router_entropy_per_seq": self.ent.item() / n / math.log(self.M),
               "router_margin": self.margin.item() / n,
               "router_load_cv": float(share.std(unbiased=False) / share.mean()),
               "router_dead": int((share < 0.1 / self.M).sum()),
               "router_switch_rate": switch,
               "router_counts": [int(v) for v in c.sum(dim=0).tolist()],
               "router_domain_counts": [[int(v) for v in row] for row in c.tolist()]}
        self.prev = dist
        self.counts.zero_()
        self.ent.zero_()
        self.margin.zero_()
        return out


def main() -> None:
    # Windows konsolu/yönlendirmesi UTF-8 olmayabilir; üretilen metin yüzünden çökmeyelim
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # sys.exit mesajları stderr'e gider
    mcfg, tcfg = build_configs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tcfg.seed)
    # TF32: Ampere+ GPU'larda float32 matris çarpımlarını hafifçe hassasiyetten ödün vererek
    # hızlandırır (autocast dışında kalan işlemler için).
    torch.set_float32_matmul_precision("high")
    if device == "cuda" and tcfg.vram_limit_frac > 0:
        torch.cuda.set_per_process_memory_fraction(tcfg.vram_limit_frac)

    tok = load_tokenizer(tcfg.data_dir)
    mcfg = replace(mcfg, vocab_size=tok.vocab_size)
    # Çok alanlı korpus (FOTON kampanyası): alan başına ayrı .bin -> pencere alan sınırını GEÇEMEZ.
    # meta.json'da "domains" varsa otomatik o yola gidilir; tek alanlı eski korpuslar aynen çalışır.
    if os.path.exists(os.path.join(tcfg.data_dir, "meta.json")):
        with open(os.path.join(tcfg.data_dir, "meta.json"), encoding="utf-8") as _f:
            _multi = "domains" in json.load(_f)
    else:
        _multi = False
    if _multi:
        train_ds = MultiDomainDataset(multi_domain_paths(tcfg.data_dir, "train"), mcfg.max_seq_len, device)
        val_ds = MultiDomainDataset(multi_domain_paths(tcfg.data_dir, "val"), mcfg.max_seq_len, device)
        print(f"çok alanlı korpus: {train_ds.names} | train {len(train_ds):,} token")
    else:
        train_path, val_path = dataset_paths(tcfg.data_dir)
        train_ds = TokenDataset(train_path, mcfg.max_seq_len, device)
        val_ds = TokenDataset(val_path, mcfg.max_seq_len, device)

    run_dir = os.path.join(tcfg.out_dir, tcfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    last_path, best_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "best.pt")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    resuming = tcfg.resume and os.path.exists(last_path)
    if os.path.exists(last_path) and not tcfg.resume:
        sys.exit(f"{last_path} zaten var. Devam için --resume true, yeni deney için farklı --run_name kullan.")
    if not resuming and os.path.exists(metrics_path):
        # save_checkpoints=false koşularında last.pt yok: bitmiş bir ölçüm sessizce ezilmesin
        with open(metrics_path, encoding="utf-8") as f:
            if any(json.loads(line).get("type") == "end" for line in f if line.strip()):
                sys.exit(f"{metrics_path} bitmiş bir koşu içeriyor; yeni deney için farklı --run_name kullan.")

    model = GPT(mcfg).to(device)
    optimizer = model.configure_optimizer(tcfg.lr, tcfg.weight_decay, (tcfg.beta1, tcfg.beta2), device)
    train_gen = torch.Generator().manual_seed(tcfg.seed)
    assigner, assign_gen = (make_assigner(mcfg.foton_assign, len(model.foton_modules), getattr(train_ds, "names", None),
                                          tcfg.seed) if mcfg.foton_assign else (None, None))
    step, best_val = 0, float("inf")
    # Eğitim duvar saati: yalnızca optimizer adımları (eval, örnek üretimi ve checkpoint kaydı hariç).
    # Eşit süre karşılaştırması bu sayıya dayanır; resume'da checkpoint'ten devam eder.
    train_time = 0.0
    tokens_per_step = tcfg.batch_size * tcfg.grad_accum * mcfg.max_seq_len
    warmup_tokens, max_tokens = token_budget(tcfg, tokens_per_step)
    # Bütçe tam adıma yukarı yuvarlanır: son adım bütçeyi en fazla (token/adım - 1) token aşar
    max_steps = math.ceil(max_tokens / tokens_per_step)
    # budget_mult: ZORUNLU 2x koşu. LR ufku `max_steps`'te (1x) kalır — `lr_at` ilerlemeyi 1.0'da kırptığı için
    # cosine 1x'te min_lr'a iner ve koşu 2x'e kadar min_lr'da sürer (v01 EK1-B9).
    if tcfg.budget_mult < 1:
        sys.exit("budget_mult >= 1 olmalı")
    run_steps = max_steps * tcfg.budget_mult
    # stop_step: schedule bütçeye göre kalır ama eğitim bu adımda durur (ör. gerçek schedule'ın ilk 100 adımı)
    end_step = min(run_steps, tcfg.stop_step) if tcfg.stop_step > 0 else run_steps

    if resuming:
        # CPU'ya yükle: load_state_dict ağırlıkları zaten GPU'daki parametrelere kopyalar,
        # optimizer durumu da parametrenin cihazına taşınır. GPU'ya yükleseydik ckpt sözlüğü
        # eğitim boyunca ağırlıkların ikinci bir kopyasını VRAM'de tutardı.
        ckpt = torch.load(last_path, map_location="cpu", weights_only=True)
        ckpt_tps = ckpt["train_cfg"]["batch_size"] * ckpt["train_cfg"]["grad_accum"] * ckpt["model_cfg"]["max_seq_len"]
        if ckpt_tps != tokens_per_step:
            # adım sayısı token'a çevrilirken (step * token/adım) yanlış olurdu: schedule ve bütçe kayardı
            sys.exit(f"checkpoint {ckpt_tps} token/adım ile yazılmış, şimdiki ayar {tokens_per_step}; "
                     f"token/adım değişince resume desteklenmiyor.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # load_state_dict param_group ayarlarını da checkpoint'ten geri yükler; komut satırında
        # değiştirilmiş weight_decay/betas sessizce ezilmesin diye mevcut config'i yeniden uygula.
        # (LR zaten her adımda lr_at() ile ayarlanıyor.)
        for group, wd in zip(optimizer.param_groups, (tcfg.weight_decay, 0.0)):
            group["weight_decay"], group["betas"] = wd, (tcfg.beta1, tcfg.beta2)
        train_gen.set_state(ckpt["train_gen_state"])
        if assign_gen is not None:
            assign_gen.set_state(ckpt["assign_gen_state"])
        step, best_val = ckpt["step"], ckpt["best_val"]
        train_time = ckpt.get("train_time", 0.0)  # v1 checkpoint'lerinde yok
        del ckpt
        if step >= end_step:
            sys.exit(f"checkpoint adımı {step} >= bütçenin adım sayısı {end_step}; daha büyük --max_steps "
                     f"ya da --max_tokens verin.")
        print(f"devam ediliyor: adım {step}, en iyi val {best_val:.4f}, eğitim süresi {train_time / 60:.1f} dk")

    # bf16 autocast: matris çarpımları bf16'da (2x daha az aktivasyon belleği, Ada'da çok daha hızlı),
    # ağırlıklar ve optimizer durumu float32'de kalır ("mixed precision").
    # bf16'nın fp16'ya üstünlüğü: float32 ile aynı üs aralığı (fp16 yalnızca ~6e-8..65504).
    # fp16'da küçük gradyanlar sıfıra yuvarlanır (underflow), bu yüzden fp16 loss scaling
    # (GradScaler) ister; bf16'da bu sorun yok -> GradScaler gerekmez.
    if tcfg.dtype not in ("bfloat16", "float32"):
        sys.exit("dtype: 'bfloat16' veya 'float32'. float16 GradScaler gerektirir ve burada desteklenmiyor.")
    ptdtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[tcfg.dtype]
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=ptdtype)
        if device == "cuda" and ptdtype != torch.float32 else nullcontext()
    )
    train_model = torch.compile(model) if tcfg.compile else model

    print(f"model: {model.num_params() / 1e6:.2f}M parametre "
          f"({model.num_params(exclude_embedding=True) / 1e6:.2f}M embedding hariç)")
    if mcfg.n_experts > 0:
        n_moe = sum(b.is_moe for b in model.blocks)
        print(f"MoE: {n_moe} katman x {mcfg.n_experts} expert, top-{mcfg.top_k} + {mcfg.n_shared_experts} shared | "
              f"token başına aktif: {model.num_params(active_only=True) / 1e6:.2f}M "
              f"({model.num_params(exclude_embedding=True, active_only=True) / 1e6:.2f}M embedding hariç)")
    print(f"veri: train {len(train_ds) / 1e6:.1f}M token, val {len(val_ds) / 1e6:.2f}M token")
    print(f"adım başına {tokens_per_step:,} token, bütçe {max_tokens / 1e6:.2f}M token = {max_steps} adım "
          f"(warmup {warmup_tokens / 1e6:.2f}M token)" + (f", {end_step}. adımda durulacak" if end_step < max_steps else ""))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"model": asdict(mcfg), "train": asdict(tcfg)}, f, indent=2)
    # Yeni koşu: dosyayı sıfırdan aç (checkpoint'e ulaşamadan çökmüş eski bir koşunun kayıtları
    # karışmasın). Resume: checkpoint adımından SONRA yazılmış kayıtları at (o adımlar yeniden
    # eğitilecek; aksi halde aynı adım iki kez görünür), sonra kaldığı yerden ekle.
    if step > 0 and os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            # eski "end" kaydı da atılır: devam eden koşu bitmiş sayılmasın (bitince yenisi yazılır)
            kept = [line for line in f if (r := json.loads(line))["step"] <= step and r["type"] != "end"]
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.writelines(kept)
    metrics = open(metrics_path, "a" if step > 0 else "w", encoding="utf-8")

    def log(record: dict) -> None:
        metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
        metrics.flush()

    # ANALİTİK aktif FLOP her koşunun kendi kaydına girer (profiler'a güvenilmez: SDPA'yı maskeden bağımsız
    # T x T sayar, gruplu matris çarpımını 0 sayar). FOTON kollarında SAYILMIŞ bağlantılar da yazılır.
    fwd = forward_flops_per_token(mcfg)
    meta = {"type": "meta", "step": 0, "arm": mcfg.foton or ("m" if mcfg.n_experts > 0 else "d"),
            "flops_train_per_token": train_flops_per_token(mcfg), "flops_forward_per_token": fwd,
            "params_total": model.num_params(), "params_active": model.num_params(active_only=True),
            "tokens_per_step": tokens_per_step, "max_steps": max_steps, "budget_mult": tcfg.budget_mult,
            "end_step": end_step, "loss_prefix_skip": tcfg.loss_prefix_skip,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    if mcfg.foton:
        meta["foton_connections_counted"] = model.foton_connection_counts()
        meta["foton_layout"] = [mcfg.foton_pre, mcfg.foton_routed, mcfg.foton_post]
        meta["foton_prefix_k"], meta["foton_reg"] = mcfg.foton_prefix_k, [mcfg.foton_reg, mcfg.foton_reg_coef]
        meta["foton_assign"] = mcfg.foton_assign
        if mcfg.foton_assign:
            meta["router"] = "UYGULANAMAZ: zorlanmış atama — router modelde durur ama çalışmaz ve eğitilmez"
    print(f"analitik aktif FLOP: eğitim {meta['flops_train_per_token']:,} / token"
          + (f" | sayılmış bağlantı {meta['foton_connections_counted']}" if mcfg.foton else ""))
    if step == 0:
        log(meta)
    router_log = (RouterLog(len(model.foton_modules), len(getattr(train_ds, "names", [])), device)
                  if mcfg.foton and not mcfg.foton_assign else None)
    assign_log = (AssignLog(len(model.foton_modules), len(getattr(train_ds, "names", [])), device)
                  if mcfg.foton_assign else None)
    eval_raw_dir = os.path.join(run_dir, "eval_raw")

    def sync() -> None:
        # CUDA çağrıları asenkron: saati okumadan önce kuyruğun bitmesini bekle, yoksa bir aralığın GPU işi
        # sonrakine (ya da eval süresine) yazılır
        if device == "cuda":
            torch.cuda.synchronize()

    # Aralıklar monoton saatle: time.time() sistem saati ayarlanınca (NTP) geri/ileri atlayabilir
    t_start = t_log = time.perf_counter()
    # Log aralığı boyunca GPU'da biriken değerler (her adımda .item() çağırmamak için tensör olarak)
    accum = {k: torch.zeros((), device=device) for k in ("loss", "ce", "aux_loss", "z_loss")}
    load_accum: torch.Tensor | None = None  # (MoE katmanı, E) expert atama sayıları
    last_load: list[list[float]] | None = None
    steps_since_log = 0
    data_domain_counts = [0] * len(train_ds.names) if isinstance(train_ds, MultiDomainDataset) else []
    skip_eval_once = tcfg.resume and step > 0  # bu adımın eval'i kaydedilmeden önce zaten yapılmıştı

    while True:
        # ---------------- değerlendirme + checkpoint ----------------
        if (step % tcfg.eval_interval == 0 or step >= end_step) and not skip_eval_once:
            sync()
            t_eval = time.perf_counter()
            # eval, log aralığının ortasına düşerse son logdan beri geçen eğitim süresi de sayılır
            train_time_now = train_time + (t_eval - t_log if steps_since_log else 0.0)
            ev = evaluate(train_model, val_ds, tcfg, autocast_ctx)
            # KARAR metriği `primary` (prefix maskeli); `full` provenance kontrolü için loglanır.
            val_loss, val_full = ev["primary"], ev["full"]
            tag = "" if tcfg.loss_prefix_skip <= 0 else f" [maskesiz {val_full:.4f}]"
            val_rec = {"type": "val", "step": step, "tokens": step * tokens_per_step, "val_loss": val_loss,
                       "val_loss_full": val_full, "loss_prefix_skip": tcfg.loss_prefix_skip}
            if "val_windows_sha256" in ev:
                val_rec["val_windows_sha256"] = ev["val_windows_sha256"]
            if "oracle" in ev:
                # Oracle eğrileri eğitim SIRASINDA loglanır: yalnızca last.pt / best.pt saklandığı için
                # eğri sonradan üretilemez (ön-kayıt, "Oracle şartı"). Ham döküm her okumayı yeniden
                # hesaplanabilir kılar.
                val_rec.update(val_loss_oracle=ev["oracle"], val_loss_oracle_null=ev["oracle_null"])
                tag += f" [oracle {ev['oracle']:.4f} | null {ev['oracle_null']:.4f}]"
                if "assign_mode" in ev:
                    # `val_loss` = EĞİTİMDEKİ atama kuralıyla kayıp. İkinci yol (`assigned_check`) üç-modül geçişinden;
                    # ikisi 1e-4 içinde aynı olmalı (analiz bunu assert eder), kaba bir ayrışma koşuyu DURDURUR.
                    val_rec.update(val_assign_mode=ev["assign_mode"], val_loss_assigned_check=ev["assigned_check"],
                                   val_loss_router_UNUSED=ev["router_unused"])
                    if not two_routes_agree(ev["assigned_check"], val_loss, 1e-3):
                        sys.exit(f"atanan val loss iki yoldan uyuşmuyor: {val_loss:.6f} vs {ev['assigned_check']:.6f}")
                else:
                    val_rec["val_loss_routed_check"] = ev["routed_check"]
                    if abs(ev["routed_check"] - val_loss) > 1e-3:
                        print(f"[UYARI] val_loss iki yoldan uyuşmuyor: {val_loss:.6f} vs {ev['routed_check']:.6f}")
                os.makedirs(eval_raw_dir, exist_ok=True)
                import numpy as np
                np.savez_compressed(os.path.join(eval_raw_dir, f"step_{step:05d}.npz"), step=step, **ev["raw"])
            print(f"[eval] adım {step}: val_loss {val_loss:.4f} (ppl {math.exp(val_loss):.2f}){tag}")
            log({**val_rec, "train_time": train_time_now, "time": time.perf_counter() - t_start})
            if last_load is not None:
                # Expert kullanım tablosu: her satır bir MoE katmanı, her karakter bir expert.
                # '▄' ideal pay; '█' ~1.9x+ aşırı yüklü; boşluk = ideal payın %10'undan az (ölü).
                for li, row in enumerate(last_load):
                    layer = li + mcfg.first_dense_layers
                    print(f"[expert] katman {layer}: |{load_sparkline(row)}| "
                          f"min {min(row) * len(row):.2f}x maks {max(row) * len(row):.2f}x")
            if tcfg.sample_tokens > 0:
                model.eval()
                start = torch.tensor([[tok.bos_id]], device=device)
                with autocast_ctx:
                    out = model.generate(start, tcfg.sample_tokens, temperature=0.8, top_k=50, eos_id=tok.eos_id)
                model.train()
                text = tok.decode(out[0, 1:].tolist())
                print(f"[örnek] {text!r}")
                log({"type": "sample", "step": step, "text": text})
            state = {"model": model.state_dict(), "model_cfg": asdict(mcfg), "train_cfg": asdict(tcfg),
                     "step": step, "val_loss": val_loss}
            if val_loss < best_val:
                best_val = val_loss
                if tcfg.save_checkpoints:
                    save_atomic(state, best_path)
            if step > 0 and tcfg.save_checkpoints:
                extra = {"assign_gen_state": assign_gen.get_state()} if assign_gen is not None else {}
                save_atomic({**state, "optimizer": optimizer.state_dict(), "best_val": best_val, **extra,
                             "train_gen_state": train_gen.get_state(), "train_time": train_time_now}, last_path)
            t_log += time.perf_counter() - t_eval  # eval/örnek/kayıt süresi tok/s ölçümüne karışmasın
        skip_eval_once = False
        if step >= end_step:
            break

        # ---------------- bir optimizer adımı ----------------
        lr = lr_at(step * tokens_per_step, tokens_per_step, warmup_tokens, max_tokens, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        # Çok alanlı korpusta eğitim akışı da bölüşümden bağımsız: adım başına dizi sayısı
        # (batch_size × grad_accum) tokens_per_step pinli olduğu için sabittir, dolayısıyla akış
        # dizi başına tanımlanabilir ve "aynı seed = aynı veri sırası" iddiası GERÇEKTEN doğru olur.
        if isinstance(train_ds, MultiDomainDataset):
            seqs = train_ds.sequence_stream(tcfg.batch_size * tcfg.grad_accum, train_gen)
            # Veri sırasının PARMAK İZİ (CPU, senkronsuz; modele dokunmaz): log aralığı başına alan sayımı. "Aynı seed = aynı veri
            # sırası" eşleme iddiası (D dahil bütün kollar) analizde bu alandan DOĞRULANIR; yalnızca kod okumasına dayanmaz.
            for d_name, _ in seqs:
                data_domain_counts[train_ds.domain_id(d_name)] += 1
            micro = [seqs[i : i + tcfg.batch_size]
                     for i in range(0, len(seqs), tcfg.batch_size)]
        else:
            micro = [None] * tcfg.grad_accum
        for chunk in micro:
            x, y = train_ds.windows(chunk) if chunk is not None \
                else train_ds.get_batch(tcfg.batch_size, train_gen)
            if assigner is not None:
                assign = assigner(getattr(train_ds, "last_domain_ids", None), x.shape[0], x.device)
                with autocast_ctx:
                    _, loss, info = train_model(x, y, assign=assign)
                # UYGULANAN atama loglanır (modelin gerçekten dispatch ettiği tensör), döngüdeki `assign` değişkeni değil:
                # forward'a başka bir atama gitseydi log yine "doğru" görünür ve hiçbir denetim fark etmezdi.
                assign_log.add(info["foton_choice"], getattr(train_ds, "last_domain_ids", None))
            else:
                with autocast_ctx:
                    _, loss, info = train_model(x, y)
            # Gradyan biriktirme: loss'u grad_accum'a böl ki toplanan gradyan, büyük tek bir
            # batch'in *ortalama* gradyanına eşit olsun (LR'ı batch'ten bağımsız tutar).
            (loss / tcfg.grad_accum).backward()
            accum["loss"] += loss.detach() / tcfg.grad_accum
            for k in ("ce", "aux_loss", "z_loss"):
                if k in info:
                    accum[k] += info[k] / tcfg.grad_accum
            if "counts" in info:
                load_accum = info["counts"].clone() if load_accum is None else load_accum + info["counts"]
            if router_log is not None:
                router_log.add(info, getattr(train_ds, "last_domain_ids", None))
                if "foton_reg" in info:
                    accum.setdefault("foton_reg", torch.zeros((), device=device))
                    accum["foton_reg"] += info["foton_reg"] / tcfg.grad_accum
        # Gradyan kırpma: nadir "kötü batch"lerde oluşan dev gradyanların modeli bozmasını engeller.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        if assign_log is not None:
            assign_log.end_step()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # None: sıfır tensörü tutmaktan daha az bellek
        # Aux-loss-free dengeleme (bias_update_rate > 0): gradyandan bağımsız, adım başına bir işaret güncellemesi
        bias_range = update_expert_biases(model)
        step += 1
        steps_since_log += 1

        # ---------------- log ----------------
        # .item() GPU'yu CPU ile senkronize eder (yavaş); bu yüzden her adımda değil, aralıklarla.
        if step % tcfg.log_interval == 0 or step >= end_step:
            sync()
            dt = time.perf_counter() - t_log
            train_time += dt
            avg = {k: v.item() / steps_since_log for k, v in accum.items()}
            tok_s = steps_since_log * tokens_per_step / dt
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            record = {"type": "train", "step": step, "tokens": step * tokens_per_step, "loss": avg["loss"],
                      "ce": avg["ce"], "lr": lr, "grad_norm": grad_norm.item(), "tok_per_sec": tok_s,
                      "peak_mem_gb": mem, "train_time": train_time, "time": time.perf_counter() - t_start}
            line = (f"adım {step:5d} | loss {avg['loss']:.4f} | lr {lr:.2e} | grad_norm {grad_norm.item():.2f} "
                    f"| {tok_s / 1e3:.1f}k tok/s | tepe bellek {mem:.2f} GB")
            if load_accum is not None:
                bal = load_balance_metrics(load_accum)
                last_load = bal["load"]
                record.update(aux_loss=avg["aux_loss"], z_loss=avg["z_loss"], **bal)
                line += (f"\n           ce {avg['ce']:.4f} | aux {avg['aux_loss']:.3f} | z {avg['z_loss']:.2f} "
                         f"| yük CV {bal['load_cv']:.3f} | maks {bal['load_max_ratio']:.2f}x "
                         f"| min {bal['load_min_ratio']:.2f}x | ölü expert {bal['dead_experts']}")
                if mcfg.bias_update_rate > 0:
                    record["bias_range"] = bias_range
                    line += f" | bias aralığı {bias_range:.3f}"
                load_accum = None
            if router_log is not None:
                rl = router_log.flush()
                record.update(rl)
                if "foton_reg" in avg:
                    record["foton_reg"] = avg["foton_reg"]
                line += (f"\n           router: marjinal H {rl['router_entropy_marginal']:.3f} | dizi-başı H "
                         f"{rl['router_entropy_per_seq']:.3f} | marj {rl['router_margin']:.3f} | yük {rl['router_counts']} "
                         f"| ölü {rl['router_dead']}")
            if data_domain_counts:
                record["data_domain_counts"] = list(data_domain_counts)
                data_domain_counts = [0] * len(data_domain_counts)
            if assign_log is not None:
                al = assign_log.flush()
                record.update(al)
                record.update({f: None for f in ROUTER_FIELDS_NA})      # UYGULANAMAZ: router çalışmıyor — sayı basılmaz
                line += (f"\n           atama ({mcfg.foton_assign}): yük {al['assign_counts']} | modülsüz mikro-batch "
                         f"{al['assign_microbatches_missing_module']} | gradyansız-modüllü adım {al['assign_steps_gradless_module']}")
            print(line)
            log(record)
            for v in accum.values():
                v.zero_()
            steps_since_log, t_log = 0, time.perf_counter()

    log({"type": "end", "step": step, "tokens": step * tokens_per_step, "train_time": train_time,
         "time": time.perf_counter() - t_start, "best_val": best_val})
    metrics.close()
    print(f"bitti: {(time.perf_counter() - t_start) / 60:.1f} dk (eğitim {train_time / 60:.1f} dk), "
          f"en iyi val_loss {best_val:.4f}")


if __name__ == "__main__":
    main()
