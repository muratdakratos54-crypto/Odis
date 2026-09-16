"""Eğitim döngüsü: token tabanlı bütçe ve LR schedule, eğitim duvar saati, kayıtlar, resume (CPU)."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.train as train  # noqa: E402
from src.config import PRESETS, TrainConfig  # noqa: E402
from src.train import lr_at, token_budget  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def lr_v1(step: int, cfg: TrainConfig) -> float:
    """v1'in (tag v1-dense-moe-baseline) adım tabanlı formülü, birebir kopya."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


@pytest.mark.parametrize("preset", ["dense", "moe", "tiny"])
def test_token_schedule_reproduces_v1_step_schedule_bitexact(preset):
    """Token/adım sabit ve 2'nin kuvvetiyken (tüm presetler) token tabanlı schedule v1 koşularının LR dizisini bit bit
    vermeli: v1 ile karşılaştırma ve dispatch eşdeğerlik koşuları schedule değişikliğinden etkilenmesin."""
    mcfg, tcfg = PRESETS[preset]
    tps = tcfg.batch_size * tcfg.grad_accum * mcfg.max_seq_len
    assert tps & (tps - 1) == 0
    warmup, total = token_budget(tcfg, tps)
    assert (warmup, total) == (tcfg.warmup_steps * tps, tcfg.max_steps * tps)
    for step in range(tcfg.max_steps + 1):
        assert lr_at(step * tps, tps, warmup, total, tcfg) == lr_v1(step, tcfg), step


def test_non_power_of_two_tokens_per_step_matches_v1_within_one_ulp():
    """Token/adım 2'nin kuvveti değilse warmup LR'ları 1 ulp'e kadar farklı olabilir; cosine kısmı bit bit aynı."""
    tcfg = replace(TrainConfig(), max_steps=300, warmup_steps=30)
    tps = 3 * 7 * 100
    warmup, total = token_budget(tcfg, tps)
    for step in range(tcfg.max_steps + 1):
        new, old = lr_at(step * tps, tps, warmup, total, tcfg), lr_v1(step, tcfg)
        if step < tcfg.warmup_steps:
            assert abs(new - old) <= math.ulp(old), step
        else:
            assert new == old, step


def test_same_token_budget_same_lr_regardless_of_step_size():
    """Warmup ve decay token'a bağlı: iki kat büyük adımlarla aynı token noktalarında aynı LR (warmup'ta
    adımın tokenları da sayıldığı için 'adımdan sonra görülen token' noktasında karşılaştırılır)."""
    tcfg = replace(TrainConfig(), max_tokens=1_000_000, warmup_tokens=100_000)
    small, big = 10_000, 20_000
    for tokens in range(0, 1_000_000, big):
        if tokens >= 100_000:  # cosine bölgesi: adım başındaki token'a bakar
            assert math.isclose(lr_at(tokens, small, 100_000, 1_000_000, tcfg),
                                lr_at(tokens, big, 100_000, 1_000_000, tcfg), rel_tol=1e-12)
        elif tokens + big <= 100_000:  # warmup bölgesi: adım sonundaki token'a bakar
            assert math.isclose(lr_at(tokens + small, small, 100_000, 1_000_000, tcfg),
                                lr_at(tokens, big, 100_000, 1_000_000, tcfg), rel_tol=1e-12)
    assert lr_at(999_999, small, 100_000, 1_000_000, tcfg) == pytest.approx(tcfg.min_lr, rel=1e-6)
    assert lr_at(10**9, small, 100_000, 1_000_000, tcfg) == tcfg.min_lr  # bütçe aşılınca min_lr'da kalır


@pytest.mark.parametrize("warmup_tokens", [100_000, 1_000, 32_768 * 3])
def test_warmup_never_exceeds_peak_and_schedule_shape(warmup_tokens):
    """Warmup bir adımın ortasında bitse bile LR tepe değeri aşmamalı; warmup'ta azalmamalı, sonra artmamalı."""
    tcfg, tps, total = TrainConfig(), 32_768, 98_467_840
    lrs = [lr_at(s * tps, tps, warmup_tokens, total, tcfg) for s in range(math.ceil(total / tps))]
    assert max(lrs) <= tcfg.lr
    n_warm = math.ceil(warmup_tokens / tps)  # warmup adımları: s * tps < warmup_tokens -> s = 0 .. n_warm - 1
    assert lrs[n_warm - 1] == tcfg.lr  # warmup'ın son adımı tam tepe değer
    assert all(a <= b for a, b in zip(lrs[:n_warm], lrs[1:n_warm]))
    assert all(a >= b for a, b in zip(lrs[n_warm - 1:], lrs[n_warm:]))


def test_token_budget_overrides_and_validation():
    tcfg = TrainConfig()
    assert token_budget(replace(tcfg, max_tokens=123, warmup_tokens=7), 32_768) == (7, 123)
    with pytest.raises(ValueError):
        token_budget(replace(tcfg, max_tokens=1000, warmup_tokens=1000), 10)


# --- main() uçtan uca, CPU ----------------------------------------------------------------------------------

class _StubTokenizer:
    vocab_size, bos_id, eos_id = 64, 1, 2

    def decode(self, ids):
        return " ".join(map(str, ids))


EVAL_SLEEP = 0.3


@pytest.fixture
def run_main(tmp_path, monkeypatch):
    """tiny preset, rastgele token verisi, CPU. evaluate() yapay olarak EVAL_SLEEP saniye sürer."""
    rng = np.random.default_rng(0)
    for name, n in (("train.bin", 20_000), ("val.bin", 5_000)):
        rng.integers(0, 64, n, dtype=np.uint16).tofile(tmp_path / name)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train, "load_tokenizer", lambda data_dir: _StubTokenizer())
    real_eval = train.evaluate

    def slow_eval(*a, **k):
        time.sleep(EVAL_SLEEP)
        return real_eval(*a, **k)

    monkeypatch.setattr(train, "evaluate", slow_eval)

    def run(run_name: str, *args: str) -> dict:
        monkeypatch.setattr(sys, "argv", ["train", "--preset", "tiny", "--data_dir", str(tmp_path),
                                          "--out_dir", str(tmp_path), "--run_name", run_name, "--batch_size", "4",
                                          "--eval_iters", "1", "--sample_tokens", "0", *args])
        train.main()
        rec = {"train": [], "val": [], "end": []}
        with open(tmp_path / run_name / "metrics.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rec.setdefault(r["type"], []).append(r)
        rec["dir"] = str(tmp_path / run_name)
        return rec

    return run


def _run_queue_module():
    spec = importlib.util.spec_from_file_location("run_queue", os.path.join(ROOT, "scripts", "run_queue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_lr_timing_and_records(run_main):
    tps = 4 * 1 * 128
    rec = run_main("r", "--max_steps", "12", "--warmup_steps", "3", "--eval_interval", "5", "--log_interval", "1",
                   "--save_checkpoints", "false")
    tcfg = replace(TrainConfig(), max_steps=12, warmup_steps=3)
    # LR çağrı yeri: adım s'nin kaydındaki LR, o adımdan önce görülen tokenla hesaplanır
    assert [r["lr"] for r in rec["train"]] == [lr_at((s - 1) * tps, tps, 3 * tps, 12 * tps, tcfg) for s in range(1, 13)]
    assert [r["tokens"] for r in rec["train"]] == [s * tps for s in range(1, 13)]
    # train_time = aralık sürelerinin toplamı (her aralık tam olarak tokens / tok_per_sec saniye)
    total = 0.0
    for r in rec["train"]:
        total += tps / r["tok_per_sec"]
        assert r["train_time"] == pytest.approx(total, rel=1e-9)
    (end,) = rec["end"]
    assert end["step"] == 12 and end["tokens"] == 12 * tps and end["train_time"] == rec["train"][-1]["train_time"]
    # eval (adım 0, 5, 10, 12 -> 4 kez, her biri >= EVAL_SLEEP) eğitim süresine karışmamalı
    assert [v["step"] for v in rec["val"]] == [0, 5, 10, 12]
    assert end["time"] - end["train_time"] >= 4 * EVAL_SLEEP
    assert end["train_time"] < end["time"] - 4 * EVAL_SLEEP + 0.25
    assert [v["train_time"] for v in rec["val"]] == [0.0] + [rec["train"][s - 1]["train_time"] for s in (5, 10, 12)]
    rq = _run_queue_module()
    assert rq.finished(rec["dir"])
    # save_checkpoints=false'ta last.pt yok: bitmiş koşu aynı adla yeniden başlatılıp ezilmemeli
    with pytest.raises(SystemExit):
        run_main("r", "--max_steps", "12", "--save_checkpoints", "false")


def test_eval_between_logs_counts_training_time_so_far(run_main):
    rec = run_main("mid", "--max_steps", "8", "--warmup_steps", "2", "--eval_interval", "3", "--log_interval", "4",
                   "--save_checkpoints", "false")
    by_step = {r["step"]: r for r in rec["train"]}
    val = {v["step"]: v for v in rec["val"]}
    # adım 3'teki eval, adım 4'teki logdan önce: o ana kadarki süre > 0 ama adım 4'e kadarki süreden az
    assert 0.0 < val[3]["train_time"] < by_step[4]["train_time"]
    assert by_step[4]["train_time"] < val[6]["train_time"] < by_step[8]["train_time"]


def test_stop_step_keeps_budget_schedule(run_main):
    tps = 4 * 128
    rec = run_main("stop", "--max_steps", "40", "--warmup_steps", "10", "--stop_step", "6", "--log_interval", "1",
                   "--eval_interval", "100", "--save_checkpoints", "false")
    tcfg = replace(TrainConfig(), max_steps=40, warmup_steps=10)
    assert rec["end"][0]["step"] == 6 and [v["step"] for v in rec["val"]] == [0, 6]
    assert [r["lr"] for r in rec["train"]] == [lr_at((s - 1) * tps, tps, 10 * tps, 40 * tps, tcfg) for s in range(1, 7)]


def test_resume_continues_train_time_and_refuses_changed_tokens_per_step(run_main):
    first = run_main("res", "--max_steps", "6", "--warmup_steps", "2", "--eval_interval", "3", "--log_interval", "3")
    t6 = first["end"][0]["train_time"]
    second = run_main("res", "--max_steps", "9", "--warmup_steps", "2", "--eval_interval", "3", "--log_interval", "3",
                      "--resume", "true")
    assert len(second["end"]) == 1 and second["end"][0]["step"] == 9  # eski "end" kaydı atıldı
    assert [r["step"] for r in second["train"]] == [3, 6, 9]
    assert second["train"][1]["train_time"] == pytest.approx(t6) and second["end"][0]["train_time"] > t6
    with pytest.raises(SystemExit):  # token/adım değişirse step * token/adım yanlış olurdu
        run_main("res", "--max_steps", "20", "--warmup_steps", "2", "--resume", "true", "--batch_size", "2")
