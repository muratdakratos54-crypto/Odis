"""v03 adım 1 — ZORLANMIŞ yönlendirme yolu (X-dom, H-dom, X-iid): model, atama kuralı, eğitim/eval döngüsü.

Karar kuralı (experiments/foton-v03-step1/decision_rule.md) bu yolun ürettiği TEK sayıya dayanır: kolun, EĞİTİMDEKİ atama
kuralıyla hesaplanan val loss'u. Burada sınananlar, o sayının sessizce başka bir şey olmasına yol açacak hatalardır:
atamanın uygulanmaması, router'a sessizce düşülmesi, X-iid'in rastgele çekiminin veri sırasını kaydırması, eval'in
eğitimdekinden başka bir kuralla ölçmesi, router'ın modelden çıkıp init eşlemesini bozması.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src import train  # noqa: E402
from src.config import PRESETS  # noqa: E402
from src.model import GPT  # noqa: E402

TINY = replace(PRESETS["tiny-foton"][0], vocab_size=61, max_seq_len=48)


def _model(**kw) -> GPT:
    torch.manual_seed(3)
    return GPT(replace(TINY, **kw)).double().eval()


def _batch(b: int = 6, t: int = 48):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 61, (b, t), generator=g), torch.randint(0, 61, (b, t), generator=g)


# ------------------------------------------------------------------------------------------------ model
def test_assigned_forward_sends_each_sequence_through_exactly_its_assigned_module():
    model = _model(foton_assign="domain")
    idx, tgt = _batch()
    assign = torch.tensor([0, 1, 2, 2, 1, 0])
    logits, loss, info = model(idx, tgt, assign=assign)
    for m in range(3):
        forced, _, _ = model(idx, tgt, force_module=m)                 # bütün batch modül m'den
        rows = (assign == m).nonzero(as_tuple=True)[0]
        assert torch.allclose(logits[rows], forced[rows], atol=1e-10)   # atanan diziler o modülün çıktısını taşıyor...
        other = (assign != m).nonzero(as_tuple=True)[0]
        assert not torch.allclose(logits[other], forced[other], atol=1e-6)   # ...diğerleri taşımıyor
    assert torch.equal(info["foton_choice"], assign) and info["foton_assigned"] is True
    assert "foton_probs" not in info and "foton_logits" not in info     # router çalışmadı: sayı basılmaz


def test_router_stays_in_the_model_but_gets_no_gradient_and_unassigned_modules_stay_gradless():
    model = _model(foton_assign="iid").float().train()
    idx, tgt = _batch()
    _, loss, _ = model(idx, tgt, assign=torch.tensor([1, 1, 1, 1, 1, 1]))
    loss.backward()
    assert model.foton_router.weight.grad is None                       # donuk: AdamW tümüyle atlar
    assert all(p.grad is None for m in (0, 2) for p in model.foton_modules[m].parameters())
    assert all(p.grad is not None for p in model.foton_modules[1].parameters())
    assert all(p.grad is not None for p in model.blocks.parameters())


@pytest.mark.parametrize("arm, assign", [("x", "domain"), ("x", "iid"), ("h", "domain")])
def test_forced_arms_start_from_the_same_weights_as_the_learned_router_arm(arm, assign):
    """Router modelden ÇIKARILMADI: aynı seed'de zorlanmış kol, v02'nin aynı kolu ile bit düzeyinde aynı init'ten başlar
    (çıkarılsaydı bağlı embedding'in init çekimi kayardı ve eşli karşılaştırma bozulurdu)."""
    torch.manual_seed(11)
    learned = GPT(replace(TINY, foton=arm)).state_dict()
    torch.manual_seed(11)
    forced = GPT(replace(TINY, foton=arm, foton_assign=assign)).state_dict()
    assert learned.keys() == forced.keys() and "foton_router.weight" in forced
    assert all(torch.equal(learned[k], forced[k]) for k in learned)


def test_misuse_does_not_run_it_raises():
    idx, tgt = _batch()
    ok = torch.tensor([0, 1, 2, 0, 1, 2])
    forced, learned = _model(foton_assign="domain"), _model()
    with pytest.raises(ValueError, match="dışarıdan atama ister"):
        forced(idx, tgt)                                                # sessizce eğitilmemiş router'a DÜŞMEZ
    with pytest.raises(ValueError, match="yalnızca foton_assign kollarında"):
        learned(idx, tgt, assign=ok)                                    # öğrenilmiş-router kolu atamayla çağrılamaz
    with pytest.raises(ValueError, match="yalnızca foton_assign kollarında"):
        forced(idx, tgt, assign=ok, force_module=0)
    with pytest.raises(ValueError, match="yalnızca FOTON"):
        GPT(replace(TINY, foton="", foton_pre=1, n_layer=4)).double()(idx, tgt, assign=ok)
    for bad in (ok[:5], ok.float(), torch.tensor([0, 1, 3, 0, 1, 2]), torch.tensor([0, -1, 2, 0, 1, 2])):
        with pytest.raises(ValueError, match="assign"):
            forced(idx, tgt, assign=bad)
    with pytest.raises(ValueError, match="foton_assign"):
        GPT(replace(TINY, foton_assign="rand"))
    with pytest.raises(ValueError, match="regularizasyonu"):
        GPT(replace(TINY, foton_assign="domain", foton_reg="lb"))
    forced(idx, tgt, force_module=1)                                    # teşhis yolu (tek modüle zorlama) açık kalır


# ------------------------------------------------------------------------------------------------ atama kuralı
def test_domain_assigner_is_the_identity_on_domain_ids_and_needs_as_many_domains_as_modules():
    fn, gen = train.make_assigner("domain", 3, ["code", "math", "nl"], seed=101)
    dom = torch.tensor([2, 0, 1, 1])
    assert gen is None and torch.equal(fn(dom, 4, "cpu"), dom) and fn(dom, 4, "cpu").dtype == torch.long
    for names in (["code", "math"], None, ["a", "b", "c", "d"]):
        with pytest.raises(ValueError, match="alan"):
            train.make_assigner("domain", 3, names, seed=1)
    with pytest.raises(ValueError, match="alan kimlikleri"):
        fn(None, 4, "cpu")
    with pytest.raises(ValueError, match="alan kimlikleri"):
        fn(dom, 5, "cpu")
    with pytest.raises(ValueError, match="bilinmeyen"):
        train.make_assigner("rand", 3, ["a", "b", "c"], seed=1)


def test_iid_assigner_has_its_own_generator_and_touches_neither_the_data_stream_nor_the_global_rng():
    fn, gen = train.make_assigner("iid", 3, ["code", "math", "nl"], seed=101)
    data_gen = torch.Generator().manual_seed(101)
    before_data, before_global = data_gen.get_state().clone(), torch.get_rng_state().clone()
    draws = torch.cat([fn(torch.tensor([0] * 16), 16, "cpu") for _ in range(400)])
    assert torch.equal(data_gen.get_state(), before_data) and torch.equal(torch.get_rng_state(), before_global)
    assert draws.dtype == torch.long and set(draws.tolist()) == {0, 1, 2}
    share = torch.bincount(draws, minlength=3) / len(draws)
    assert float((share - 1 / 3).abs().max()) < 0.03                     # düzgün
    dom = torch.arange(3).repeat(len(draws) // 3 + 1)[: len(draws)]
    joint = torch.zeros(3, 3).index_put_((dom, draws), torch.ones(len(draws)), accumulate=True)
    assert float((joint / joint.sum(1, keepdim=True) - 1 / 3).abs().max()) < 0.05     # alandan BAĞIMSIZ
    again, _ = train.make_assigner("iid", 3, None, seed=101)
    other, _ = train.make_assigner("iid", 3, None, seed=102)
    first = lambda f: torch.cat([f(None, 16, "cpu") for _ in range(10)])   # noqa: E731
    assert torch.equal(first(again), draws[:160]) and not torch.equal(first(other), draws[:160])
    assert gen.initial_seed() == train.ASSIGN_SEED_OFFSET + 101 != 101


def test_assign_log_counts_missing_modules_per_microbatch_and_gradless_modules_per_step():
    log = train.AssignLog(3, 3, "cpu")
    dom = torch.tensor([0, 1, 2, 0])
    for a in ([0, 1, 2, 0], [0, 0, 1, 1]):                               # adım 1: ikinci mikro-batch'te modül 2 yok
        log.add(torch.tensor(a), dom)
    log.end_step()                                                        # ...ama adım içinde dizi aldı: gradyansız DEĞİL
    for a in ([0, 0, 1, 1], [1, 1, 0, 0]):                               # adım 2: modül 2 hiçbir mikro-batch'te yok
        log.add(torch.tensor(a), dom)
    log.end_step()
    out = log.flush()
    assert out["assign_counts"] == [8, 7, 1] and out["assign_microbatches_missing_module"] == 3
    assert out["assign_steps_gradless_module"] == 1 and np.array(out["assign_domain_counts"]).sum() == 16
    assert log.flush()["assign_counts"] == [0, 0, 0]                     # flush sıfırlar


# ------------------------------------------------------------------------------------------------ eğitim / eval döngüsü
class _StubTokenizer:
    vocab_size, bos_id, eos_id = 64, 1, 2

    def decode(self, ids):
        return " ".join(map(str, ids))


@pytest.fixture
def run(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    (tmp_path / "meta.json").write_text(json.dumps({"domains": {"aaa": {}, "bbb": {}, "ccc": {}}}), encoding="utf-8")
    for dom in ("aaa", "bbb", "ccc"):
        for split, n in (("train", 30_000), ("val", 8_000)):
            rng.integers(0, 64, n, dtype=np.uint16).tofile(tmp_path / f"{dom}_{split}.bin")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train, "load_tokenizer", lambda data_dir: _StubTokenizer())

    def go(run_name: str, *args: str) -> dict:
        monkeypatch.setattr(sys, "argv", ["train", "--preset", "tiny-foton", "--data_dir", str(tmp_path),
                                          "--out_dir", str(tmp_path), "--run_name", run_name, "--seed", "101",
                                          "--batch_size", "16", "--grad_accum", "4", "--eval_iters", "2",
                                          "--sample_tokens", "0", "--max_seq_len", "64", "--save_checkpoints", "false",
                                          "--max_steps", "12", "--eval_interval", "6", "--log_interval", "3",
                                          "--warmup_steps", "3", *args])
        train.main()
        rec: dict = {}
        with open(tmp_path / run_name / "metrics.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rec.setdefault(r["type"], []).append(r)
        rec["dir"] = str(tmp_path / run_name)
        return rec

    return go


def _raw(rec: dict, step: int):
    return np.load(os.path.join(rec["dir"], "eval_raw", f"step_{step:05d}.npz"))


def test_domain_arm_trains_and_evaluates_with_module_equal_domain_and_both_routes_agree(run):
    rec = run("xdom", "--foton_assign", "domain")
    assert rec["meta"][0]["foton_assign"] == "domain" and "UYGULANAMAZ" in rec["meta"][0]["router"]
    for t in rec["train"]:
        c = np.array(t["assign_domain_counts"])
        assert c.sum() == 3 * 4 * 16 and (c - np.diag(np.diag(c))).sum() == 0          # modül = alan, istisnasız
        assert all(t[f] is None for f in train.ROUTER_FIELDS_NA)                        # router metrikleri: uygulanamaz
        assert "assign_microbatches_missing_module" in t and "assign_steps_gradless_module" in t
    for v in rec["val"]:
        raw = _raw(rec, v["step"])
        posthoc = raw["seq_masked"][np.arange(32), raw["domain"]].mean()                # İKİNCİ yol: ham dökümden
        assert v["val_assign_mode"] == "domain" and "val_loss_routed_check" not in v
        assert abs(v["val_loss"] - posthoc) < 1e-4 and abs(v["val_loss"] - v["val_loss_assigned_check"]) < 1e-4
        assert "val_loss_router_UNUSED" in v
        routed = raw["seq_masked"][np.arange(32), raw["choice"]].mean()                 # router'ın seçimi: BAŞKA bir sayı
        assert v["val_loss_router_UNUSED"] == pytest.approx(routed, abs=1e-5)


def test_iid_arm_reports_the_expected_loss_under_uniform_assignment_and_sees_the_same_data_as_the_learned_arm(run, monkeypatch):
    # casus: eval SIRASINDA eğitimin forward yolu (assign=...) gerçekten çağrılıyor mu — X-iid'in "iki yolu" ilk hâlde aynı
    # tensörün iki okunuşuydu (sağlama başarısız OLAMIYORDU)
    eval_calls, real_forward = [], GPT.forward

    def spy(self, idx, targets=None, force_module=None, assign=None):
        if not self.training and assign is not None:
            eval_calls.append(sorted(set(assign.tolist())))
        return real_forward(self, idx, targets, force_module, assign)

    monkeypatch.setattr(GPT, "forward", spy)
    iid = run("xiid", "--foton_assign", "iid")
    assert len(eval_calls) == 3 * 2 * 3 and eval_calls[:3] == [[0], [1], [2]]           # 3 eval x 2 batch x 3 modül
    eval_calls.clear()
    learned = run("xlearned")
    # AYNI seed -> AYNI veri sırası: X-iid'in rastgele çekimi `train_gen`'den yapılsaydı alan sayımları kayardı
    dom_iid = [np.array(t["assign_domain_counts"]).sum(1).tolist() for t in iid["train"]]
    dom_learned = [np.array(t["router_domain_counts"]).sum(1).tolist() for t in learned["train"]]
    assert dom_iid == dom_learned
    assert [v["val_loss"] for v in iid["val"]][0] == pytest.approx(learned["val"][0]["val_loss_oracle"], abs=0.2)   # aynı init
    total = np.sum([t["assign_counts"] for t in iid["train"]], axis=0)
    assert total.sum() == 12 * 64 and (np.abs(total / total.sum() - 1 / 3) < 0.08).all()
    joint = np.sum([t["assign_domain_counts"] for t in iid["train"]], axis=0)
    assert (joint > 0).all()                                                            # her alan her modüle gidiyor
    for v in iid["val"]:
        raw = _raw(iid, v["step"])
        assert v["val_assign_mode"] == "iid"
        assert v["val_loss"] == pytest.approx(raw["seq_masked"].mean(), abs=1e-5)       # beklenen değer = üç modülün ortalaması
        assert v["val_loss_full"] == pytest.approx(raw["seq_full"].mean(), abs=1e-5)
        assert abs(v["val_loss"] - v["val_loss_assigned_check"]) < 1e-4                 # İKİ AYRI kod yolu, aynı sayı
    assert all("val_assign_mode" not in v and "val_loss_routed_check" in v for v in learned["val"])
    assert all(t["router_dead"] is not None and "assign_counts" not in t for t in learned["train"])


def _replay_domains(data_dir: str, seed: int, steps: int, per_step: int):
    from src.data_multi import MultiDomainDataset, multi_domain_paths
    ds = MultiDomainDataset(multi_domain_paths(data_dir, "train"), 64, "cpu")
    gen = torch.Generator().manual_seed(seed)
    return [[ds.domain_id(d) for d, _ in ds.sequence_stream(per_step, gen)] for _ in range(steps)]


def test_gradless_step_counter_sits_after_the_whole_step_not_inside_the_microbatch_loop(run, tmp_path):
    """4'lük mikro-batch'lerde bir alanın (= modülün) hiç olmaması SIK, 16'lık adımın tamamında olmaması NADİR. Sayaç
    mikro-batch döngüsünün içine kaysaydı iki sayı eşitlenirdi ve rapora 'AdamW şu kadar adımda bir modülü atladı' yanlış girerdi."""
    # seed 21: 12 adımın BİRİNDE bir alan hiç yok (seed 101'de sıfırdı: ölü bir adım-sonu sayacı da "0 == 0" diye geçerdi)
    rec = run("xdom-small", "--foton_assign", "domain", "--batch_size", "4", "--eval_iters", "4", "--seed", "21")
    doms = _replay_domains(str(tmp_path), 21, 12, 16)                                   # aynı seed -> aynı akış
    want_mb = sum(len(set(step[i:i + 4])) < 3 for step in doms for i in range(0, 16, 4))
    want_steps = sum(len(set(step)) < 3 for step in doms)
    got_mb = sum(t["assign_microbatches_missing_module"] for t in rec["train"])
    got_steps = sum(t["assign_steps_gradless_module"] for t in rec["train"])
    assert (got_mb, got_steps) == (want_mb, want_steps) and want_mb > want_steps >= 1 and want_mb >= 10


def test_iid_assignment_stream_in_a_real_run_is_the_seeded_generator_and_survives_a_resume(run):
    """Koşunun logladığı (UYGULANAN) atama sayımları, `make_assigner("iid", seed=koşunun seed'i)` akışının yeniden oynatılmasıyla
    birebir aynı; 6. adımda durup `--resume` ile sürdürülen koşu da düz 12 adımla aynı sayımları verir."""
    straight = run("iid-duz", "--foton_assign", "iid")
    fn, _ = train.make_assigner("iid", 3, None, seed=101)
    want = []
    for _ in range(4):                                                                  # 4 log aralığı x 3 adım x 4 mikro-batch
        draws = torch.cat([fn(None, 16, "cpu") for _ in range(3 * 4)])
        want.append(torch.bincount(draws, minlength=3).tolist())
    assert [t["assign_counts"] for t in straight["train"]] == want
    other, _ = train.make_assigner("iid", 3, None, seed=102)
    assert torch.bincount(torch.cat([other(None, 16, "cpu") for _ in range(12)]), minlength=3).tolist() != want[0]
    run("iid-parca", "--foton_assign", "iid", "--save_checkpoints", "true", "--stop_step", "6")
    resumed = run("iid-parca", "--foton_assign", "iid", "--save_checkpoints", "true", "--resume", "true")
    assert [t["assign_domain_counts"] for t in resumed["train"]] == [t["assign_domain_counts"] for t in straight["train"]]
    assert resumed["val"][-1]["val_loss"] == pytest.approx(straight["val"][-1]["val_loss"], abs=1e-6)


def test_the_three_step1_presets_differ_from_the_v02_arms_only_in_the_assignment_rule():
    for name, base, mode in (("foton-x-dom", "foton-x", "domain"), ("foton-h-dom", "foton-h", "domain"),
                             ("foton-x-iid", "foton-x", "iid")):
        m, t = PRESETS[name]
        bm, bt = PRESETS[base]
        assert m == replace(bm, foton_assign=mode) and replace(t, run_name="") == replace(bt, run_name="")
    assert replace(PRESETS["foton-d"][1], run_name="") == replace(PRESETS["foton-x-dom"][1], run_name="")


def test_every_arm_logs_the_data_order_fingerprint_and_same_seed_means_same_fingerprint(run, tmp_path):
    """`data_domain_counts` = log aralığı başına alan sayımı; akışın yeniden oynatılmasıyla birebir aynı, DENSE kolda da var
    (eşli D'nin 'aynı veri sırası' iddiası analizde bu alandan doğrulanır), başka seed'de başka."""
    dense = run("dense", "--foton", "")
    forced = run("xdom-fp", "--foton_assign", "domain")
    doms = _replay_domains(str(tmp_path), 101, 12, 64)
    want = [np.bincount(np.concatenate(doms[i:i + 3]), minlength=3).tolist() for i in range(0, 12, 3)]
    assert [t["data_domain_counts"] for t in dense["train"]] == want
    assert [t["data_domain_counts"] for t in forced["train"]] == want
    assert [np.sum(t["assign_domain_counts"], axis=1).tolist() for t in forced["train"]] == want
    other = _replay_domains(str(tmp_path), 102, 12, 64)
    assert [np.bincount(np.concatenate(other[i:i + 3]), minlength=3).tolist() for i in range(0, 12, 3)] != want
    assert "assign_counts" not in dense["train"][0] and "val_assign_mode" not in dense["val"][0]


def test_two_routes_agree_treats_a_non_finite_value_as_disagreement():
    assert train.two_routes_agree(3.1, 3.1 + 5e-4, 1e-3) and train.two_routes_agree(3.1, 3.1, 0.0)
    assert not train.two_routes_agree(3.1, 3.1 + 2e-3, 1e-3) and not train.two_routes_agree(3.1 + 2e-3, 3.1, 1e-3)
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert not train.two_routes_agree(bad, 3.1, 1e-3) and not train.two_routes_agree(3.1, bad, 1e-3)
        assert not train.two_routes_agree(bad, bad, 1e-3)                                # çıplak `abs(a - b) > tol` bunu GEÇİRİRDİ


def test_a_forced_run_stops_itself_at_the_first_eval_when_the_two_routes_disagree(run, monkeypatch):
    monkeypatch.setattr(train, "two_routes_agree", lambda a, b, tol: False)
    with pytest.raises(SystemExit, match="iki yoldan uyuşmuyor"):
        run("xdom-durur", "--foton_assign", "domain")
    assert len(run("ogrenilmis-durmaz")["val"]) == 3                                     # öğrenilmiş router kolu bu kapıya bağlı değil


def test_every_arm_logs_a_content_digest_of_the_validation_windows(run, tmp_path):
    """`val_windows_sha256`: dense kol ham döküm yazmaz; 'D, FOTON kollarıyla AYNI val kümesinde ölçüldü' yalnızca bu alandan
    doğrulanabilir. Seed'den bağımsız, pencere sayısına ve dosyanın İÇERİĞİNE bağlı (indisler aynı kalsa bile)."""
    dense, forced = run("dense-val", "--foton", ""), run("xiid-val", "--foton_assign", "iid", "--seed", "102")
    digests = {v["val_windows_sha256"] for rec in (dense, forced) for v in rec["val"]}
    assert len(digests) == 1 and len(next(iter(digests))) == 64 and len(dense["val"]) == 3
    assert {v["val_windows_sha256"] for v in run("dense-uzun", "--foton", "", "--eval_iters", "4")["val"]}.isdisjoint(digests)
    np.random.default_rng(5).integers(0, 64, 8_000, dtype=np.uint16).tofile(tmp_path / "bbb_val.bin")     # aynı boy, başka içerik
    assert {v["val_windows_sha256"] for v in run("dense-yeni-val", "--foton", "")["val"]}.isdisjoint(digests)
