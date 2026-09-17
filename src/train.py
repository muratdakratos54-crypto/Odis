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


@torch.no_grad()
def evaluate(model: GPT, ds: TokenDataset, cfg: TrainConfig, autocast_ctx) -> dict[str, float]:
    """Val loss. `{"full": ..., "primary": ...}` döner.

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

    def both(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        """Aynı batch'te tam ve maskeli CE — TEK forward'dan.

        İkinci bir forward yapmak eval maliyetini ikiye katlardı; bunun yerine logit'lerden
        pozisyon başına CE'yi bir kez hesaplayıp iki ortalamayı alıyoruz. `model(x, y)`'nin
        döndürdüğü skaler `ce` ile `per_pos.mean()` aynı niceliktir (aşağıda test ediliyor).
        """
        with autocast_ctx:
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
        for chunk in batches:
            f, p = both(*ds.windows(chunk))
            tf += f
            tp += p
        model.train()
        return {"full": tf / len(batches), "primary": tp / len(batches)}

    g = torch.Generator().manual_seed(12345)
    tf = tp = 0.0
    for _ in range(cfg.eval_iters):
        # Toplam loss değil, saf cross-entropy: MoE'nin aux/z ceza terimleri model kalitesini
        # ölçmez; dense ile MoE'yi adil karşılaştırmak için yalnızca CE'ye bakıyoruz.
        f, p = both(*ds.get_batch(cfg.batch_size, g))
        tf += f
        tp += p
    model.train()
    return {"full": tf / cfg.eval_iters, "primary": tp / cfg.eval_iters}


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
    step, best_val = 0, float("inf")
    # Eğitim duvar saati: yalnızca optimizer adımları (eval, örnek üretimi ve checkpoint kaydı hariç).
    # Eşit süre karşılaştırması bu sayıya dayanır; resume'da checkpoint'ten devam eder.
    train_time = 0.0
    tokens_per_step = tcfg.batch_size * tcfg.grad_accum * mcfg.max_seq_len
    warmup_tokens, max_tokens = token_budget(tcfg, tokens_per_step)
    # Bütçe tam adıma yukarı yuvarlanır: son adım bütçeyi en fazla (token/adım - 1) token aşar
    max_steps = math.ceil(max_tokens / tokens_per_step)
    # stop_step: schedule bütçeye göre kalır ama eğitim bu adımda durur (ör. gerçek schedule'ın ilk 100 adımı)
    end_step = min(max_steps, tcfg.stop_step) if tcfg.stop_step > 0 else max_steps

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
            print(f"[eval] adım {step}: val_loss {val_loss:.4f} (ppl {math.exp(val_loss):.2f}){tag}")
            log({"type": "val", "step": step, "tokens": step * tokens_per_step, "val_loss": val_loss,
                 "val_loss_full": val_full, "loss_prefix_skip": tcfg.loss_prefix_skip,
                 "train_time": train_time_now, "time": time.perf_counter() - t_start})
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
                save_atomic({**state, "optimizer": optimizer.state_dict(), "best_val": best_val,
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
            micro = [seqs[i : i + tcfg.batch_size]
                     for i in range(0, len(seqs), tcfg.batch_size)]
        else:
            micro = [None] * tcfg.grad_accum
        for chunk in micro:
            x, y = train_ds.windows(chunk) if chunk is not None \
                else train_ds.get_batch(tcfg.batch_size, train_gen)
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
        # Gradyan kırpma: nadir "kötü batch"lerde oluşan dev gradyanların modeli bozmasını engeller.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
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
