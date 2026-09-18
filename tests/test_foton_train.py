"""FOTON kollarının EĞİTİM DÖNGÜSÜNDE ürettiği ölçümler: oracle / null eğrileri, router logları, 2x koşu, meta satırı.

Bunlar ölçümü ÜRETEN koddur: kampanya koştuktan sonra onarılamaz (yalnızca last.pt / best.pt saklanır, eğriler
eğitim sırasında loglanır). Ön-kaydın açık bıraktığı her tanım `src/train.py`'de sabitlenmiştir ve HAM veri de
diske yazılır; buradaki testler hem sabitlenen tanımı hem ham verinin gerçekten yazıldığını kilitler.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src import train  # noqa: E402
from src.train import ORACLE_NULL_SEED, RouterLog, oracle_stats  # noqa: E402

LOG3 = math.log(3)


# ------------------------------------------------------------------ oracle ve null

def _perm(B: int, T: int, seed: int = 0) -> torch.Tensor:
    return torch.randint(6, (B, T), generator=torch.Generator().manual_seed(seed))


def test_oracle_selects_per_sequence_on_the_metric_positions_only():
    """Seçim ÖLÇÜT pozisyonları (k..T-1) üzerinden yapılır: prefix'te çok iyi olan modül seçilmez."""
    B, T, k = 2, 8, 4
    ce = torch.full((3, B, T), 5.0)
    ce[0, 0, :k] = 0.0            # modül 0: dizi 0'ın PREFIX'inde mükemmel, ölçüt pozisyonlarında 5.0
    ce[1, 0, k:] = 4.0            # modül 1: dizi 0'ın ölçüt pozisyonlarında en iyi
    ce[2, 1, k:] = 3.0            # modül 2: dizi 1'de en iyi
    st = oracle_stats(ce, torch.tensor([0, 0]), k, _perm(B, T))
    assert st["oracle"].tolist() == [4.0, 3.0]
    assert st["seq_masked"][:, 0].tolist() == [5.0, 4.0, 5.0] and st["seq_full"][0, 0].item() == 2.5
    assert st["routed"].tolist() == [5.0, 5.0]                          # router ikisinde de modül 0'ı seçmişti
    assert bool((st["oracle"] <= st["routed"]).all())                   # oracle, router'ın seçtiğinden kötü OLAMAZ


def test_null_keeps_selection_noise_but_destroys_module_information():
    """B11'in gerekçesi: modüller istatistiksel olarak AYNIYKEN bile seçim eğriyi aşağı çeker; null o payı taşır.

    (i) Üç modül birbirinin aynısıysa null = oracle = her biri. (ii) Modüller yalnızca gürültüyle farklıysa
    (bilgi yok) null ile oracle aynı dağılımdan gelir: fark ~0. (iii) Bir modül bir dizide GERÇEKTEN iyiyse
    oracle onu bulur, null bulamaz: oracle < null. Null hiçbir zaman sözde-modüllerin ortalamasından büyük olamaz.
    """
    g = torch.Generator().manual_seed(1)
    B, T, k = 64, 512, 64
    base = torch.rand(1, B, T, generator=g) * 2 + 2
    same = base.expand(3, B, T).clone()
    st = oracle_stats(same, torch.zeros(B, dtype=torch.long), k, _perm(B, T, 2))
    assert torch.allclose(st["null"], st["oracle"]) and torch.allclose(st["oracle"], st["routed"])
    noise = base + 0.3 * torch.randn(3, B, T, generator=g)              # (ii) bilgi yok, yalnızca gürültü
    st = oracle_stats(noise, torch.zeros(B, dtype=torch.long), k, _perm(B, T, 3))
    gain_over_mean = (noise[:, :, k:].mean(-1).mean(0) - st["oracle"]).mean().item()
    assert gain_over_mean > 0.005                                       # seçim gürültüden "kazandırıyor"...
    assert abs((st["null"] - st["oracle"]).mean().item()) < 0.004       # ...ve null aynı payı taşıyor
    info = noise.clone()
    info[1, : B // 2] -= 0.5                                            # (iii) modül 1 dizilerin yarısında GERÇEKTEN iyi
    st = oracle_stats(info, torch.zeros(B, dtype=torch.long), k, _perm(B, T, 4))
    assert (st["null"] - st["oracle"])[: B // 2].mean().item() > 0.25   # bilgi null'da silindi
    assert bool((st["null"] <= info[:, :, k:].mean(-1).mean(0) + 1e-6).all())


def test_null_is_deterministic_and_invariant_to_module_relabelling_of_the_permutation_seed():
    g = torch.Generator().manual_seed(5)
    ce = torch.rand(3, 8, 64, generator=g)
    a = oracle_stats(ce, torch.zeros(8, dtype=torch.long), 8, _perm(8, 64, 7))
    b = oracle_stats(ce, torch.zeros(8, dtype=torch.long), 8, _perm(8, 64, 7))
    assert torch.equal(a["null"], b["null"])
    assert ORACLE_NULL_SEED == 54321                                   # her eval'de, her koşuda AYNI permütasyon
    # sözde-modüller gerçek kayıpların pozisyon başına yeniden DİZİLİMİDİR: pozisyon başına çoklu küme korunur
    perm = _perm(8, 64, 7)
    pseudo = ce.gather(0, train._PERMS3[perm].permute(2, 0, 1))
    assert torch.equal(pseudo.sort(dim=0).values, ce.sort(dim=0).values)


# ------------------------------------------------------------------ router logları (V6)

def _info(choice, probs=None, logits=None):
    choice = torch.tensor(choice)
    B = len(choice)
    probs = torch.full((B, 3), 1 / 3) if probs is None else torch.tensor(probs)
    logits = torch.zeros(B, 3) if logits is None else torch.tensor(logits)
    return {"foton_choice": choice, "foton_probs": probs, "foton_logits": logits}


def test_router_log_definitions():
    rl = RouterLog(3, 3, "cpu")
    # 12 dizi: alan 0 -> hep modül 0; alan 1 -> hep modül 1; alan 2 -> yarı yarıya 1 ve 2
    choice = [0] * 4 + [1] * 4 + [1, 1, 2, 2]
    dom = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    logits = [[2.0, 0.5, 0.0]] * 12
    rl.add(_info(choice, logits=logits), dom)
    out = rl.flush()
    share = np.array([4, 6, 2]) / 12
    assert out["router_counts"] == [4, 6, 2]
    assert out["router_domain_counts"] == [[4, 0, 0], [0, 4, 0], [0, 2, 2]]
    assert out["router_entropy_marginal"] == pytest.approx(-(share * np.log(share)).sum() / LOG3)
    assert out["router_entropy_per_seq"] == pytest.approx(1.0)          # olasılıklar düz: dizi başına entropi = log 3
    assert out["router_margin"] == pytest.approx(1.5)                   # top-1 − top-2 LOGIT
    assert out["router_load_cv"] == pytest.approx(share.std() / share.mean())     # POPÜLASYON std
    assert out["router_dead"] == 0 and out["router_switch_rate"] is None          # ilk aralık: önceki yok
    # ikinci aralık: alan 0 tamamen modül 2'ye geçti (TV = 1), diğerleri aynı -> ortalama 1/3
    rl.add(_info([2] * 4 + [1] * 4 + [1, 1, 2, 2]), dom)
    out2 = rl.flush()
    assert out2["router_switch_rate"] == pytest.approx(1 / 3)
    assert out2["router_counts"] == [0, 6, 6] and out2["router_dead"] == 1


def test_router_dead_uses_the_ten_percent_of_ideal_share_threshold_and_collapse_reads_sqrt2():
    rl = RouterLog(3, 1, "cpu")
    rl.add(_info([0] * 97 + [1] * 3), None)                             # modül 1: %3 < %3.33 -> ölü; modül 2: 0 -> ölü
    out = rl.flush()
    assert out["router_dead"] == 2
    rl.add(_info([0] * 96 + [1] * 4), None)                             # %4 > %3.33 -> ölü DEĞİL
    assert rl.flush()["router_dead"] == 1
    rl.add(_info([0] * 50), None)
    full = rl.flush()
    assert full["router_load_cv"] == pytest.approx(math.sqrt(2)) and full["router_entropy_marginal"] == 0.0


def test_per_sequence_entropy_is_low_for_a_decisive_router_while_marginal_stays_high():
    """İki entropi ZIT anlam taşır (V6): dizi başına kararlı + diziler arası dengeli = biri ~0, diğeri ~1."""
    rl = RouterLog(3, 1, "cpu")
    p = [[0.98, 0.01, 0.01]] * 10 + [[0.01, 0.98, 0.01]] * 10 + [[0.01, 0.01, 0.98]] * 10
    rl.add(_info([0] * 10 + [1] * 10 + [2] * 10, probs=p), None)
    out = rl.flush()
    assert out["router_entropy_marginal"] == pytest.approx(1.0) and out["router_entropy_per_seq"] < 0.12


# ------------------------------------------------------------------ uçtan uca: tiny-foton koşusu

class _StubTokenizer:
    vocab_size, bos_id, eos_id = 64, 1, 2

    def decode(self, ids):
        return " ".join(map(str, ids))


@pytest.fixture
def run_foton(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    (tmp_path / "meta.json").write_text(json.dumps({"domains": {"aaa": {}, "bbb": {}, "ccc": {}}}), encoding="utf-8")
    for dom in ("aaa", "bbb", "ccc"):
        for split, n in (("train", 30_000), ("val", 8_000)):
            rng.integers(0, 64, n, dtype=np.uint16).tofile(tmp_path / f"{dom}_{split}.bin")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train, "load_tokenizer", lambda data_dir: _StubTokenizer())

    def run(run_name: str, *args: str, preset: str = "tiny-foton") -> dict:
        monkeypatch.setattr(sys, "argv", ["train", "--preset", preset, "--data_dir", str(tmp_path),
                                          "--out_dir", str(tmp_path), "--run_name", run_name,
                                          "--batch_size", "16", "--eval_iters", "1", "--sample_tokens", "0",
                                          "--max_seq_len", "64", "--save_checkpoints", "false", *args])
        train.main()
        rec: dict = {}
        with open(tmp_path / run_name / "metrics.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rec.setdefault(r["type"], []).append(r)
        rec["dir"] = str(tmp_path / run_name)
        return rec

    return run


def test_foton_run_logs_oracle_curves_router_fields_meta_and_raw_dumps(run_foton):
    rec = run_foton("f", "--max_steps", "20", "--eval_interval", "10", "--log_interval", "5", "--warmup_steps", "5")
    meta = rec["meta"][0]
    assert meta["arm"] == "x" and meta["step"] == 0 and meta["budget_mult"] == 1 and meta["end_step"] == 20
    assert meta["flops_train_per_token"] == 3 * sum(meta["flops_forward_per_token"].values())
    assert set(meta["foton_connections_counted"]) == {"M0:A", "M1:B", "M2:C"}
    assert meta["foton_layout"] == [1, 2, 1] and meta["foton_prefix_k"] == 16
    assert [v["step"] for v in rec["val"]] == [0, 10, 20]
    for v in rec["val"]:
        assert v["val_loss_oracle"] <= v["val_loss"] + 1e-6             # oracle eğrisi FOTON'un eğrisinin üstünde OLAMAZ
        # (null'ın oracle'dan büyük olması bir DEĞİŞMEZ değildir: modüller ayırt edilemezken sözde-modüllerin
        #  en küçüğü gerçek modüllerin en küçüğünün altına da düşebilir; o zaman kazanç <= 0 okunur.)
        assert abs(v["val_loss_oracle_null"] - v["val_loss_oracle"]) < 0.05
        assert v["val_loss_routed_check"] == pytest.approx(v["val_loss"], abs=1e-5)   # val_loss iki yoldan aynı
        raw = np.load(os.path.join(rec["dir"], "eval_raw", f"step_{v['step']:05d}.npz"))
        assert raw["ce_tok"].shape == (16, 3, 64) and raw["ce_tok"].dtype == np.float16
        assert raw["seq_masked"].shape == (16, 3) and raw["probs"].shape == (16, 3)
        assert set(raw["domain"].tolist()) <= {0, 1, 2} and raw["choice"].shape == (16,)
        # ham dökümden oracle YENİDEN hesaplanabiliyor (sonradan başka bir okuma gerekirse)
        assert raw["seq_masked"].min(axis=1).mean() == pytest.approx(v["val_loss_oracle"], abs=1e-5)
        assert raw["seq_masked"][np.arange(16), raw["choice"]].mean() == pytest.approx(v["val_loss"], abs=1e-5)
    for t in rec["train"]:
        for key in ("router_entropy_marginal", "router_entropy_per_seq", "router_switch_rate", "router_dead",
                    "router_load_cv", "router_margin", "router_counts", "router_domain_counts"):
            assert key in t, key
        assert sum(t["router_counts"]) == 5 * 16 and np.array(t["router_domain_counts"]).shape == (3, 3)
    assert rec["train"][0]["router_switch_rate"] is None and rec["train"][1]["router_switch_rate"] is not None


def test_baseline_arms_get_a_meta_row_but_no_foton_fields(run_foton):
    rec = run_foton("d", "--max_steps", "10", "--eval_interval", "10", "--log_interval", "5", "--warmup_steps", "5",
                    preset="tiny")
    assert rec["meta"][0]["arm"] == "d" and "foton_connections_counted" not in rec["meta"][0]
    assert all("val_loss_oracle" not in v for v in rec["val"])
    assert all("router_counts" not in t for t in rec["train"])
    assert not os.path.exists(os.path.join(rec["dir"], "eval_raw"))


def test_mandatory_2x_run_keeps_the_1x_lr_horizon_then_stays_at_min_lr(run_foton):
    """ZORUNLU 2x koşu (ön-kayıt V3 adım 2, v01 B9): cosine 1x'te min_lr'a iner, 2x'e kadar min_lr'da KALIR.
    `--max_steps`'i ikiye katlamak cosine'i 2x'e yayardı: BAŞKA bir eğitim."""
    one = run_foton("x1", "--max_steps", "20", "--eval_interval", "10", "--log_interval", "1", "--warmup_steps", "4")
    two = run_foton("x2", "--max_steps", "20", "--eval_interval", "10", "--log_interval", "1", "--warmup_steps", "4",
                    "--budget_mult", "2")
    assert two["meta"][0]["budget_mult"] == 2 and two["meta"][0]["max_steps"] == 20 and two["meta"][0]["end_step"] == 40
    assert two["end"][0]["step"] == 40 and [v["step"] for v in two["val"]] == [0, 10, 20, 30, 40]
    lr1 = [t["lr"] for t in one["train"]]
    lr2 = [t["lr"] for t in two["train"]]
    assert lr2[:20] == lr1                                              # ilk 1x: BİREBİR aynı schedule
    assert all(lr == pytest.approx(1e-4) for lr in lr2[20:])            # sonrası: sabit min_lr
    stretched = run_foton("x3", "--max_steps", "40", "--eval_interval", "10", "--log_interval", "1", "--warmup_steps", "4")
    assert [t["lr"] for t in stretched["train"]][:20] != lr1            # yanlış yol gerçekten başka bir schedule
