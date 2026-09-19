"""`scripts/ox1_microbench.py` (ön-kayıt O-X1) testleri — CPU'da, saniyeler içinde.

Bu betik karar kampanyasından ÖNCE bir kez koşar ve çıktısı rapora girer; ölçümü üreten kod sonradan onarılamaz.
Ön-kaydın O-X1 paragrafı modülleri hâlâ ESKİ tasarımla anıyor (A bitişik 256 / B dilated 256 / C 128+128 ->
nedensel maske altında 192.25 / 128.50 / 88.31 bağlantı: reddedilen compute confound). Buradaki testlerin asıl işi
o bayat metnin koda HİÇBİR yoldan (sabit, desen kaydı, bayrak) sızamadığını kilitlemek:

  1. JSON dört girdiyi de SAYILMIŞ bağlantıyla taşır (A 128.17.., B 128.50, C 128.375, full 256.5);
  2. üç kilidin HER BİRİ tek başına RED verir ve RED'de dosya YAZILMAZ (her test kendi kilidinin mesajını arar:
     bir kilit sökülürse başka bir kilidin RED'i testi geçiremesin);
  3. argparse desen / boyut parametresi bayrağı TANIMAZ;
  4. ölçüm sırası tekrarlar arasında döner ve kaydedilen sıra GERÇEKLEŞEN sıradır;
  5. pinli protokolden her sapma (CPU dahil) çıktıyı gürültüyle "raporlanamaz" yapar;
  6. ÖLÇÜMÜN KENDİSİ (doğrulama turunda eklendi — kilitler sağlamken bile yanlış SAYI üretebilecek yerler): her
     dilim ADINI taşıdığı modülü zamanlar (nesne kimliğiyle sınanır), ısınma her modüle pinli sayıda uygulanır,
     saat iki senkronun İÇİNDE okunur, tepe bellek dilim başına sıfırlanıp döngüden SONRA okunur, adım gerçekten
     ileri + GERİ'dir ve iki katmanı da girdiyi de kapsar, kilit 3 modülün HER katmanına uygulanır.

Hız için ölçüm testleri `run(batch=1, warmup=.., bf16=False)` ile çağırır: bu makinenin CPU'sunda bf16 emülasyonu
adım başına ~5 s. T ve n_head'e DOKUNULMAZ (bağlantı sayımı yalnızca T=512'de anlamlı) ve bu sapmaların JSON'a
yazıldığı ayrıca sınanır.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import statistics
import types
from dataclasses import replace

import pytest
import torch

from src import attn_patterns
from src import model as foton_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("ox1_microbench", os.path.join(ROOT, "scripts", "ox1_microbench.py"))
ox1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ox1)

# Bir kilit SÖKÜLMÜŞ olsa bile (mutasyon sınavı) test saniyeler içinde bitsin diye her koşu bu ayarla çağrılır.
FAST = {"device": "cpu", "repeats": 1, "iters": 1, "batch": 1, "warmup": 0, "bf16": False}
T, N_HEAD = 512, 8


def _mask_tensor(pattern: attn_patterns.Pattern) -> torch.Tensor:
    return torch.from_numpy(pattern.masks(T, N_HEAD).copy())


def _assert_refused(exc: pytest.ExceptionInfo, out) -> None:
    """RED = sıfırdan farklı çıkış kodu + dosya YOK (yarım .tmp de yok)."""
    assert exc.value.code not in (0, None)
    assert not os.path.exists(out) and not os.path.exists(str(out) + ".tmp")


@pytest.fixture(scope="module")
def happy(tmp_path_factory):
    """Tek bir gerçek (CPU, minik) koşu; pinli tekrar sayısıyla (5) ki dönen sıranın başa sarması da görülsün."""
    out = tmp_path_factory.mktemp("ox1") / "ox1_microbench.json"
    returned = ox1.run(str(out), **{**FAST, "repeats": 5, "warmup": 1})
    with open(out, encoding="utf-8") as f:
        on_disk = json.load(f)
    return returned, on_disk


# ---------------------------------------------------------------------------------------- 1. JSON içeriği
def test_json_has_all_four_modules_with_counted_connections(happy):
    """Dört girdi ve SAYILMIŞ bağlantılar. Beklenen değerler burada donmuş tanımdan ayrıca (tam sayı
    aritmetiğiyle) hesaplanır: A = yerel 150, B = her 2. ofset, C = (yerel 256 + her 4. ofset) / 2."""
    _, doc = happy
    assert list(doc["modules"]) == ["A", "B", "C", "full"]
    got = {k: m["connections_counted"] for k, m in doc["modules"].items()}
    local256 = sum(min(i + 1, 256) for i in range(T)) / T
    expected = {"A": sum(min(i + 1, 150) for i in range(T)) / T,
                "B": sum(i // 2 + 1 for i in range(T)) / T,
                "C": (local256 + sum(i // 4 + 1 for i in range(T)) / T) / 2,
                "full": (T + 1) / 2}
    assert got == expected, got
    assert (round(got["A"], 2), got["B"], got["C"], got["full"]) == (128.17, 128.5, 128.375, 256.5)   # ön-kayıt


def test_connections_are_counted_per_layer_from_the_mask_buffer(happy):
    _, doc = happy
    for name in ("A", "B", "C"):
        m = doc["modules"][name]
        assert sorted(m["connections_per_layer"]) == ["layer3", "layer4"]      # 3+2+3 bölüşümünde modül katmanları
        assert set(m["connections_per_layer"].values()) == {m["connections_counted"]}
        assert "attn_mask" in m["connections_source"]
    full = doc["modules"]["full"]
    assert "ikiz" in full["connections_source"] and "is_causal" in full["connections_source"]


def test_returned_dict_is_what_was_written(happy):
    returned, doc = happy
    assert json.loads(json.dumps(returned, ensure_ascii=False)) == doc


def test_statistic_is_the_median_over_repeats_keyed_by_repeat_id(happy):
    """Tekrar sonuçları kimlikle ("r0".."r4") anahtarlı; raporlanan sayı MEDYAN (+ min/max), ortalama değil."""
    _, doc = happy
    assert sorted(doc["repeats"]) == ["r0", "r1", "r2", "r3", "r4"]
    for name, m in doc["modules"].items():
        per = m["tok_per_sec_by_repeat"]
        assert per == {rid: rep["segments"][name]["tok_per_sec"] for rid, rep in doc["repeats"].items()}
        assert m["tok_per_sec_median"] == statistics.median(per.values())
        assert (m["tok_per_sec_min"], m["tok_per_sec_max"]) == (min(per.values()), max(per.values()))
        assert m["tok_per_sec_vs_full"] == m["tok_per_sec_median"] / doc["modules"]["full"]["tok_per_sec_median"]
        assert m["peak_gb"] is None                                            # CPU'da tepe bellek ölçülmez


def test_tok_per_sec_is_tokens_times_iters_over_seconds(happy):
    _, doc = happy
    tokens = doc["protocol_realized"]["tokens_per_iter"]
    assert tokens == doc["protocol_realized"]["batch"] * T
    for rep in doc["repeats"].values():
        for seg in rep["segments"].values():
            assert seg["tok_per_sec"] == pytest.approx(tokens * doc["protocol_realized"]["iters"] / seg["seconds"])


def test_time_segment_runs_exactly_iters_steps(monkeypatch):
    calls = []
    monkeypatch.setattr(ox1, "train_step", lambda *a, **k: calls.append(1))
    seg = ox1.time_segment(None, None, None, None, "cpu", iters=7, tokens_per_iter=100)
    assert len(calls) == 7
    assert seg["tok_per_sec"] == pytest.approx(700 / seg["seconds"])


def test_sdpa_backend_is_reported_for_every_module(happy):
    _, doc = happy
    for name, m in doc["modules"].items():
        assert m["sdpa_ops"] and all("scaled_dot_product" in op for op in m["sdpa_ops"]), (name, m["sdpa_ops"])


def test_honest_note_stale_note_and_h_arm(happy):
    """'Gather yok' notu, bayat O-X1 uyarısı ve 'H = A, bir kez ölçüldü' JSON'da yazılı."""
    _, doc = happy
    assert "GATHER YOK" in doc["honest_note"] and "is_causal" in doc["honest_note"]
    assert "192.25 / 128.50 / 88.31" in doc["stale_ox1_text"]
    assert doc["h_arm"]["pattern"] == "A" and doc["h_arm"]["n_modules"] == 3 and "H" in doc["h_arm"]["note"]
    assert doc["environment"]["torch"] == torch.__version__ and doc["environment"]["device_name"]
    assert doc["frozen_spec"]["found_in_attn_patterns"] == doc["frozen_spec"]["expected"]


# ---------------------------------------------------------------------------------------- 5. protokol sapmaları
def test_cpu_output_is_loudly_marked_unreportable(happy):
    _, doc = happy
    assert doc["reportable"] is False
    assert any("CPU" in w and "UYARI" in w for w in doc["warnings"])
    # tekrar sayısı pinli değerde (5) koşuldu: sapma listesinde OLMAMALI; geri kalan her sapma adıyla yazılı
    assert set(doc["protocol_deviations"]) == {"batch", "warmup_iters", "iters", "device", "autocast"}
    assert doc["protocol_deviations"]["batch"] == {"pinned": 16, "realized": 1}


def test_pinned_protocol_is_the_preregistered_one():
    assert ox1.PINNED == {"batch": 16, "T": 512, "warmup_iters": 10, "repeats": 5, "iters": 30,
                          "device": "cuda", "autocast": "bfloat16"}
    sig = inspect.signature(ox1.run).parameters
    assert {k: sig[k].default for k in ("device", "repeats", "iters", "batch", "warmup", "bf16")} == {
        "device": "cuda", "repeats": 5, "iters": 30, "batch": 16, "warmup": 10, "bf16": True}


def test_bf16_autocast_is_really_on_in_the_timed_step():
    """Hız için testler bf16'yı kapatıyor; açıkken attention girdisinin GERÇEKTEN bf16 olduğu burada sınanır
    (minik bloklarla: bu CPU'da bf16 çok yavaş). Adım sonunda gradyanlar temizlenmiş olmalı."""
    cfg = ox1.PRESETS["tiny-foton"][0]
    blocks = torch.nn.ModuleList([foton_model.Block(cfg, 1, pattern="A")])
    cos, sin = foton_model.precompute_rope(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
    x = torch.randn(1, cfg.max_seq_len, cfg.d_model).requires_grad_(True)
    seen = []
    blocks[0].attn.qkv.register_forward_hook(lambda mod, inp, outp: seen.append(outp.dtype))
    ox1.train_step(blocks, x, cos, sin, "cpu", bf16=True)
    ox1.train_step(blocks, x, cos, sin, "cpu", bf16=False)
    assert seen == [torch.bfloat16, torch.float32]
    assert x.grad is None and all(p.grad is None for p in blocks.parameters())


def test_output_is_reportable_exactly_when_nothing_deviates(monkeypatch, tmp_path):
    """`reportable`'ın İKİ yönü: sapma yoksa True (ve UYARI yok), tek bir sapma varsa False. Gerçek pinli protokol
    CPU'da koşulamaz (bf16 adımı ~5 s); o yüzden PİN bu testte hızlı ayara çekilir — sınanan şey sayıların kendisi
    değil, 'sapma yok <=> raporlanabilir' bağıdır. (Pinin gerçek değerleri `test_pinned_protocol_...`'da kilitli.)"""
    for k, v in {"device": "cpu", "batch": 1, "warmup_iters": 0, "repeats": 1, "iters": 1,
                 "autocast": "KAPALI (float32)"}.items():
        monkeypatch.setitem(ox1.PINNED, k, v)
    clean = ox1.run(str(tmp_path / "temiz.json"), **FAST)
    assert clean["protocol_deviations"] == {} and clean["reportable"] is True
    assert not any("UYARI" in w for w in clean["warnings"])
    dirty = ox1.run(str(tmp_path / "sapmis.json"), **{**FAST, "iters": 2})
    assert dirty["protocol_deviations"] == {"iters": {"pinned": 1, "realized": 2}} and dirty["reportable"] is False
    assert any("UYARI" in w and "RAPORLANAMAZ" in w for w in dirty["warnings"])


def test_stale_design_numbers_in_the_notes_are_the_computed_ones():
    """Sayı elle yazılmaz: nottaki ve RED mesajındaki '192.25 / 128.50 / 88.31' burada bayat tasarımdan (bitişik 256 /
    dilated 256 / 128+128) HESAPLANIR ve metinle karşılaştırılır. Sahibin verdiği üçlüyle de aynı olmalı."""
    stale = {"A": attn_patterns.local(256).avg(T, N_HEAD), "B": attn_patterns.dilated(2, 256).avg(T, N_HEAD),
             "C": attn_patterns.head_split("eski C", attn_patterns.Head(1, 128, "yerel W=128"),
                                           attn_patterns.Head(4, 128, "strided s=4 W=128")).avg(T, N_HEAD)}
    text = " / ".join(f"{stale[k]:.2f}" for k in ("A", "B", "C"))
    assert text == "192.25 / 128.50 / 88.31"                                   # sahibin 06 ek A'daki üçlüsü
    assert text in ox1.STALE_NOTE and text in ox1.__doc__
    with pytest.raises(SystemExit) as exc:
        ox1.check_connection_target(stale)
    assert f"bayat O-X1 tasarımı {text} verir" in str(exc.value)


# ---------------------------------------------------------------------------------------- 6. ölçümün kendisi
def _name_built_modules(monkeypatch) -> dict[int, str]:
    """Kurulan her modülü NESNE KİMLİĞİYLE adına bağlar. 'Bu ad altında hangi modül koştu' sorusu ad listesinden
    değil ancak böyle sınanır (`Module.to(device)` aynı nesneyi döndürür; `run` modülleri koşu boyunca canlı tutar)."""
    names: dict[int, str] = {}
    real = ox1.build_module

    def build(cfg, name, seed=0):
        blocks = real(cfg, name, seed)
        names[id(blocks)] = name
        return blocks

    monkeypatch.setattr(ox1, "build_module", build)
    return names


def test_every_segment_times_the_module_it_is_named_after_and_every_module_is_warmed_up(monkeypatch, tmp_path):
    """Kimlik eşleşmesi: `segments[name]`'e yazılan süre GERÇEKTEN `name` modülünün adımlarından gelmeli. Kaydedilen
    'order' yalnızca ADLARI tutar; `time_segment(built[<yanlış>], ...)` ad listesini bozmadan A'nın süresini B'nin
    satırına yazardı. Aynı kayıt ısınmayı da sınar: JSON 'warmup_iters' YAZDIĞI için değil, adımlar SAYILDIĞI için.
    Adımın girdisi de pinlenir: [batch, T, d_model] ve gradyan İSTER — tok/s `batch * T` ile hesaplandığından girdi
    başka bir batch'le kurulursa sayı sessizce yanlış olur."""
    names = _name_built_modules(monkeypatch)
    steps = []

    def fake_step(blocks, x, cos, sin, device, bf16=True):
        steps.append({"module": names[id(blocks)], "x": tuple(x.shape), "grad": x.requires_grad, "bf16": bf16})

    monkeypatch.setattr(ox1, "train_step", fake_step)
    batch, warmup, repeats, iters = 2, 3, 5, 2
    doc = ox1.run(str(tmp_path / "o.json"), device="cpu", repeats=repeats, iters=iters, batch=batch, warmup=warmup,
                  bf16=False)
    base = list(ox1.MODULES)
    expected = list(base)                                         # SDPA arka uç adı için profil geçişi: modül başına 1
    expected += [name for name in base for _ in range(warmup)]    # ısınma: HER modül, istenen sayıda, ölçümden ÖNCE
    for r in range(repeats):
        assert doc["repeats"][f"r{r}"]["order"] == list(ox1.rotated_order(r))
        expected += [name for name in ox1.rotated_order(r) for _ in range(iters)]
    assert [s["module"] for s in steps] == expected
    d_model = ox1.PRESETS[ox1.PRESET][0].d_model
    assert {(s["x"], s["grad"], s["bf16"]) for s in steps} == {((batch, T, d_model), True, False)}
    assert doc["protocol_realized"]["tokens_per_iter"] == batch * T


def test_time_segment_reads_the_clock_inside_the_two_syncs(monkeypatch):
    """CUDA çekirdekleri asenkrondur: baştaki senkron olmazsa ısınmanın kuyruğu dilime sızar, sondaki olmazsa saat
    çekirdekler BİTMEDEN okunur ve tok/s şişer. CPU'da ikisi de görünmez; o yüzden olay SIRASI sınanır:
    senkron, saat, adımlar, senkron, saat. Sahte saatle süre ve tok/s de TAM değerle doğrulanır."""
    events = []
    ticks = iter((10.0, 12.5))
    monkeypatch.setattr(ox1, "_sync", lambda device: events.append(f"sync:{device}"))
    monkeypatch.setattr(ox1, "train_step", lambda *a, **k: events.append("step"))
    monkeypatch.setattr(ox1, "time", types.SimpleNamespace(
        perf_counter=lambda: (events.append("clock"), next(ticks))[1]))       # yalnızca ox1'in gördüğü `time`
    seg = ox1.time_segment(None, None, None, None, "cpu", iters=3, tokens_per_iter=100)
    assert events == ["sync:cpu", "clock", "step", "step", "step", "sync:cpu", "clock"]
    assert seg["seconds"] == 2.5 and seg["tok_per_sec"] == 100 * 3 / 2.5
    assert seg["peak_gb"] is None and seg["base_gb"] is None


def test_time_segment_cuda_branch_syncs_for_real_and_brackets_the_peak_memory(monkeypatch):
    """cuda dalı GPU'ya DOKUNMADAN: `torch.cuda`'nın dört işlevi sahtelenir (bir eğitim kuyruğu koşuyor olabilir).
    `_sync('cuda')` gerçekten `torch.cuda.synchronize` çağırmalı; tepe bellek sayacı adımlardan ÖNCE tam bir kez
    sıfırlanmalı ve adımlardan SONRA okunmalı — yoksa bir modülün tepesi olarak önceki modülünki raporlanır."""
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: events.append("reset"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: (events.append("base"), 3e9)[1])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: (events.append("peak"), 7e9)[1])
    monkeypatch.setattr(ox1, "train_step", lambda *a, **k: events.append("step"))
    seg = ox1.time_segment(None, None, None, None, "cuda", iters=2, tokens_per_iter=100)
    where = {kind: [i for i, e in enumerate(events) if e == kind] for kind in ("step", "sync", "reset", "base", "peak")}
    first_step, last_step = where["step"][0], where["step"][-1]
    assert len(where["step"]) == 2
    assert len(where["sync"]) == 2 and where["sync"][0] < first_step and where["sync"][1] > last_step
    assert len(where["reset"]) == 1 and where["reset"][0] < first_step
    assert len(where["base"]) == 1 and where["base"][0] < first_step
    assert len(where["peak"]) == 1 and where["peak"][0] > last_step
    assert (seg["peak_gb"], seg["base_gb"]) == (7.0, 3.0)

    events.clear()
    ox1._sync("cpu")                                                           # CPU'da CUDA'ya hiç dokunulmaz
    assert events == []


def test_train_step_is_forward_plus_backward_through_every_layer_and_the_input_without_an_optimizer():
    """Pinli protokol 'ileri + GERİ, AdamW dahil DEĞİL'. Geri yayılım sessizce düşerse tok/s ~3 kat şişer ve satır
    yine 'ileri + geri' diye raporlanır; adım yalnızca ilk katmanı koşarsa birim '2 katmanlı modül' olmaz. Gradyan
    kancalarıyla sınanır: iki katmanın ağırlığına VE girdiye gradyan ulaşmalı; ağırlıklar adımdan sonra AYNI kalmalı."""
    cfg = ox1.PRESETS["tiny-foton"][0]
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList([foton_model.Block(cfg, cfg.foton_pre + j, pattern="A") for j in range(2)])
    cos, sin = foton_model.precompute_rope(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
    x = torch.randn(2, cfg.max_seq_len, cfg.d_model).requires_grad_(True)
    before = {k: v.clone() for k, v in blocks.state_dict().items()}
    seen = {}
    blocks[0].attn.qkv.weight.register_hook(lambda g: seen.update(layer0=bool(g.abs().sum() > 0)))
    blocks[1].ffn.w_down.weight.register_hook(lambda g: seen.update(layer1=bool(g.abs().sum() > 0)))
    x.register_hook(lambda g: seen.update(x=(tuple(g.shape), bool(g.abs().sum() > 0))))
    ox1.train_step(blocks, x, cos, sin, "cpu", bf16=False)
    assert seen == {"layer0": True, "layer1": True, "x": (tuple(x.shape), True)}
    after = blocks.state_dict()
    assert list(after) == list(before) and all(torch.equal(after[k], before[k]) for k in before)


# ---------------------------------------------------------------------------------------- 4. dönen sıra
def test_rotated_order_is_a_latin_square():
    base = ("A", "B", "C", "full")
    rows = [ox1.rotated_order(r) for r in range(4)]
    assert rows[0] == base and rows[1] == ("B", "C", "full", "A")
    for pos in range(4):
        assert sorted(row[pos] for row in rows) == sorted(base)               # her konumda her modül tam bir kez
    assert [ox1.rotated_order(r) for r in range(4, 8)] == rows


def test_measurement_order_rotates_across_repeats(happy):
    """Kaydedilen sıra ölçüm anında tutulur: plan dönüp döngü sabit sırada koşarsa bu test düşer."""
    _, doc = happy
    orders = {rid: tuple(rep["order"]) for rid, rep in doc["repeats"].items()}
    assert orders == {f"r{r}": ox1.rotated_order(r) for r in range(5)}
    assert len({orders[f"r{r}"] for r in range(4)}) == 4 and orders["r4"] == orders["r0"]
    assert doc["order_position_counts"]["A"] == {"0": 2, "1": 1, "2": 1, "3": 1}
    assert any("Latin" in w for w in doc["warnings"])                          # 5 tekrar 4 modülle tam dengelenmez


# ---------------------------------------------------------------------------------------- 2. RED'ler
def test_refuses_when_frozen_constant_and_pattern_are_both_stale(monkeypatch, tmp_path):
    """Bayat O-X1'in A'sı: A_LOCAL_W=256 VE PATTERNS['A'] = yerel 256. main() RED verir, dosya yazılmaz."""
    out = tmp_path / "o.json"
    monkeypatch.setattr(attn_patterns, "A_LOCAL_W", 256)
    monkeypatch.setitem(attn_patterns.PATTERNS, "A", attn_patterns.local(256))
    real = ox1.run
    monkeypatch.setattr(ox1, "run", lambda o, **kw: real(o, **{**kw, **FAST}))
    with pytest.raises(SystemExit, match="O-X1 RED") as exc:
        ox1.main(["--out", str(out), "--device", "cpu"])
    _assert_refused(exc, out)


@pytest.mark.parametrize("const,stale", [("A_LOCAL_W", 256), ("B_DILATION_R", 4), ("B_BUDGET_W", 128),
                                         ("C_LOCAL_W", 128), ("C_STRIDE_S", 2), ("C_STRIDE_W", 128)])
def test_lock1_frozen_constants_alone(monkeypatch, tmp_path, const, stale):
    """KİLİT 1 tek başına: yalnızca sabit değişmiş (desen kaydı yerinde) — altı sabitin HER BİRİ için."""
    out = tmp_path / "o.json"
    monkeypatch.setattr(attn_patterns, const, stale)
    with pytest.raises(SystemExit, match="donmuş spesifikasyondan sapmış") as exc:
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)


def test_frozen_spec_pin_is_the_owners_design():
    assert ox1.FROZEN_SPEC == {"A_LOCAL_W": 150, "B_DILATION_R": 2, "B_BUDGET_W": 256,
                               "C_LOCAL_W": 256, "C_STRIDE_S": 4, "C_STRIDE_W": 256}
    assert (ox1.CONN_TARGET, ox1.CONN_TOL) == (128.4, 0.4)
    # ikinci kilit (`GPT.verify_foton_masks`) ile aynı hedefi söylemeli; ayrışırlarsa biri bayattır
    assert (foton_model.FOTON_CONN_TARGET, foton_model.FOTON_CONN_TOL) == (ox1.CONN_TARGET, ox1.CONN_TOL)


def test_lock3_stale_ox1_design_with_intact_constants(monkeypatch, tmp_path):
    """Sahibin senaryosu: sabitler yerinde ama desen kaydı ESKİ tasarım (256 / 256 / 128+128)."""
    out = tmp_path / "o.json"
    monkeypatch.setitem(attn_patterns.PATTERNS, "A", attn_patterns.local(256))
    monkeypatch.setitem(attn_patterns.PATTERNS, "C", attn_patterns.head_split(
        "C", attn_patterns.Head(1, 128, "yerel W=128"), attn_patterns.Head(4, 128, "strided s=4 W=128")))
    with pytest.raises(SystemExit, match="DONMUŞ desene uymuyor") as exc:
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)


def test_lock3_catches_a_wrong_pattern_whose_count_is_inside_the_tolerance(monkeypatch, tmp_path):
    """Kilit 2'nin GÖREMEDİĞİ sızıntı: 'A' adına dilated(2, 256) -> 128.50 bağlantı, tolerans içinde."""
    out = tmp_path / "o.json"
    impostor = attn_patterns.dilated(2, 256)
    assert abs(impostor.avg(T, N_HEAD) - ox1.CONN_TARGET) <= ox1.CONN_TOL          # sayı gerçekten masum
    monkeypatch.setitem(attn_patterns.PATTERNS, "A", impostor)
    with pytest.raises(SystemExit, match="A: head 0'in maskesi DONMUŞ desene uymuyor") as exc:
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)


def test_lock3_requires_heads_split_evenly(monkeypatch, tmp_path):
    """C 'YARISI yerel, YARISI strided': 2 yerel + 6 strided head donmuş tasarım değildir."""
    out = tmp_path / "o.json"
    loc = attn_patterns.Head(1, 256, "yerel W=256")
    strd = attn_patterns.Head(4, 256, "strided s=4 W=256")
    monkeypatch.setitem(attn_patterns.PATTERNS, "C", attn_patterns.head_split("C", loc, strd, strd, strd))
    with pytest.raises(SystemExit, match="eşit bölünmemiş") as exc:
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)


def test_lock3_accepts_every_frozen_mask_in_any_head_order():
    for name in ("A", "B", "C", "full"):
        ox1.check_mask_is_frozen_pattern(name, _mask_tensor(attn_patterns.PATTERNS[name]), T, N_HEAD)
    ox1.check_mask_is_frozen_pattern("C", _mask_tensor(attn_patterns.PATTERNS["C"]).flip(0), T, N_HEAD)
    with pytest.raises(SystemExit, match="bool değil"):
        ox1.check_mask_is_frozen_pattern("A", _mask_tensor(attn_patterns.PATTERNS["A"])[:1], T, N_HEAD)


def test_lock2_counted_connections_alone(monkeypatch, tmp_path):
    """KİLİT 2 tek başına: kilit 3 DEVRE DIŞI bırakılsa bile eski tasarımın SAYILMIŞ bağlantısı RED'dir."""
    out = tmp_path / "o.json"
    monkeypatch.setattr(ox1, "check_mask_is_frozen_pattern", lambda *a, **k: None)
    monkeypatch.setitem(attn_patterns.PATTERNS, "A", attn_patterns.local(256))
    with pytest.raises(SystemExit, match="SAYILMIŞ bağlantı 128.4 ± 0.4 dışında") as exc:
        ox1.run(str(out), **FAST)
    assert "192.25" in str(exc.value)
    _assert_refused(exc, out)


def _tamper_realized_mask_of_A(monkeypatch, layers: tuple[int, ...]) -> None:
    """Desen KAYDI ve formüller yerinde dururken, kurulan A modülünün GERÇEKLEŞEN buffer'ını tam nedensele çevirir.
    Sayım formülden yapılsaydı (128.17) bu fark görünmezdi — ilk tasarımda gözden kaçan şeyin ta kendisi."""
    real = ox1.build_module

    def build(cfg, name, seed=0):
        blocks = real(cfg, name, seed)
        if name == "A":
            for j in layers:
                blocks[j].attn.attn_mask.copy_(torch.ones(T, T).tril().bool())
        return blocks

    monkeypatch.setattr(ox1, "build_module", build)


def test_count_comes_from_the_realized_buffer_not_from_the_formula(monkeypatch, tmp_path):
    out = tmp_path / "o.json"
    _tamper_realized_mask_of_A(monkeypatch, layers=(0, 1))
    assert attn_patterns.PATTERNS["A"].avg(T, N_HEAD) == pytest.approx(128.173828125)     # formül hâlâ "masum"
    with pytest.raises(SystemExit, match="A: head 0'in maskesi DONMUŞ desene uymuyor") as exc:   # kilit 3 görür
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)
    monkeypatch.setattr(ox1, "check_mask_is_frozen_pattern", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="SAYILMIŞ bağlantı 128.4 ± 0.4 dışında") as exc:        # kilit 2 de görür
        ox1.run(str(out), **FAST)
    assert "'A': 256.5" in str(exc.value)
    _assert_refused(exc, out)


def test_both_layers_of_a_module_must_have_the_same_count(monkeypatch, tmp_path):
    out = tmp_path / "o.json"
    _tamper_realized_mask_of_A(monkeypatch, layers=(1,))
    monkeypatch.setattr(ox1, "check_mask_is_frozen_pattern", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="katmanların SAYILMIŞ bağlantıları farklı") as exc:
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)


def test_lock3_checks_every_layer_even_when_the_count_is_unchanged(monkeypatch, tmp_path):
    """Kilit 3 modülün HER katmanına uygulanmalı. Yalnızca İKİNCİ katmanın maskesinde tek bir bağlantı TAŞINIR
    (son satırda pencerenin en eski anahtarı kapanır, j=0 açılır): True sayısı değişmez, yani kilit 2 de
    'katmanlar aynı sayıda' kontrolü de KÖR kalır — bunu yalnızca o katmanın maske karşılaştırması görür."""
    out = tmp_path / "o.json"
    real = ox1.build_module
    counts = {}

    def build(cfg, name, seed=0):
        blocks = real(cfg, name, seed)
        if name == "A":
            mask = blocks[1].attn.attn_mask
            oldest = T - attn_patterns.A_LOCAL_W                   # son satırın (i = T-1) penceresindeki en eski anahtar
            assert bool(mask[:, T - 1, oldest].all()) and not bool(mask[:, T - 1, 0].any())
            mask[:, T - 1, oldest] = False
            mask[:, T - 1, 0] = True
            counts.update({j: ox1.counted_connections(blocks[j].attn.attn_mask) for j in (0, 1)})
        return blocks

    monkeypatch.setattr(ox1, "build_module", build)
    with pytest.raises(SystemExit, match="A: head 0'in maskesi DONMUŞ desene uymuyor") as exc:
        ox1.run(str(out), **FAST)
    assert counts[0] == counts[1]                                  # sayım gerçekten kör: fark yalnızca desende
    _assert_refused(exc, out)


def test_lock2_tolerance_band():
    ok = {"A": 128.17, "B": 128.5, "C": 128.375, "full": 256.5}
    ox1.check_connection_target(ok)                                            # 'full' toleransa TABİ DEĞİL
    ox1.check_connection_target({**ok, "A": 128.01, "B": 128.79})
    stale = {"A": attn_patterns.local(256).avg(T, N_HEAD), "B": attn_patterns.dilated(2, 256).avg(T, N_HEAD),
             "C": attn_patterns.head_split("x", attn_patterns.Head(1, 128, "y"),
                                           attn_patterns.Head(4, 128, "s")).avg(T, N_HEAD)}
    for bad in ({**ok, "A": 127.99}, {**ok, "C": 128.81}, {**ok, "B": math.nan}, {"A": 128.4, "B": 128.4}, stale):
        with pytest.raises(SystemExit, match="SAYILMIŞ bağlantı"):
            ox1.check_connection_target(bad)


def test_counted_connections_counts_true_entries():
    m = torch.zeros(2, 4, 4, dtype=torch.bool)
    m[0] = torch.ones(4, 4).tril().bool()          # 10 bağlantı
    m[1, :, 0] = True                              # 4 bağlantı
    assert ox1.counted_connections(m) == 14 / 8


def test_full_reference_is_the_is_causal_fast_path_and_its_twin_is_verified(monkeypatch):
    """'full' D/M'nin yolu olmalı (maske YOK) ve ikiz maskeyle sayısal olarak aynı attention'ı vermeli.
    Eşiğin iki yanı burada yeniden ölçülür: doğru maske çok altında, TEK girdisi eksik maske üstünde."""
    cfg = ox1.PRESETS[ox1.PRESET][0]
    cos, sin = foton_model.precompute_rope(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
    full = ox1.build_module(cfg, "full")
    assert all(blk.attn.pattern is None and not hasattr(blk.attn, "attn_mask") for blk in full)
    mask, rel = ox1.full_reference_twin(cfg, full, cos, sin, "cpu")
    assert rel < ox1.TWIN_REL_TOL / 10 and ox1.counted_connections(mask) == 256.5

    with pytest.raises(SystemExit, match="hızlı yolunda DEĞİL"):               # maskeli bir 'referans' kabul edilmez
        ox1.full_reference_twin(cfg, ox1.build_module(cfg, "A"), cos, sin, "cpu")

    monkeypatch.setitem(attn_patterns.PATTERNS, "full", attn_patterns.local(T - 1))   # tek (i=511, j=0) girdisi eksik
    with pytest.raises(SystemExit, match="AYNI attention'ı vermiyor"):
        ox1.full_reference_twin(cfg, full, cos, sin, "cpu")


def test_full_twin_mask_is_pinned_even_if_the_numeric_check_is_blind(monkeypatch, tmp_path):
    """İkiz maskenin tam nedensel olduğu AYRICA maske özdeşliğiyle kilitli (sayısal eşik kör edilse de)."""
    out = tmp_path / "o.json"
    monkeypatch.setattr(ox1, "TWIN_REL_TOL", 1e9)
    monkeypatch.setitem(attn_patterns.PATTERNS, "full", attn_patterns.local(T - 1))
    with pytest.raises(SystemExit, match="full: head 0'in maskesi DONMUŞ desene uymuyor") as exc:
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)


def test_modules_are_built_like_the_trained_ones():
    """Birim = 2 katman, katman indeksleri 3 ve 4, desen adı Block'a geçer; dört modülün ağırlıkları aynı tohumdan."""
    cfg = ox1.PRESETS[ox1.PRESET][0]
    a, full = ox1.build_module(cfg, "A"), ox1.build_module(cfg, "full")
    assert len(a) == 2 and [blk.attn.pattern for blk in a] == ["A", "A"]
    assert all(isinstance(blk, foton_model.Block) and not blk.is_moe for blk in a)
    assert sorted(ox1.module_masks(cfg, "A", a)) == ["layer3", "layer4"]
    sa, sf = a.state_dict(), full.state_dict()
    assert list(sa) == list(sf) and all(torch.equal(sa[k], sf[k]) for k in sa)
    with pytest.raises(SystemExit, match="beklenen desenle kurulmamış"):
        ox1.module_masks(cfg, "B", a)


def test_refuses_when_preset_dims_drift(monkeypatch, tmp_path):
    out = tmp_path / "o.json"
    mcfg, tcfg = ox1.PRESETS["foton-x"]
    monkeypatch.setitem(ox1.PRESETS, "foton-x", (replace(mcfg, ffn_hidden=1024), tcfg))
    with pytest.raises(SystemExit, match="donmuş boyutlarından sapmış") as exc:
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)


def test_refuses_when_h_arm_is_not_all_A(monkeypatch, tmp_path):
    """'A bir kez ölçülür, H için de geçerlidir' cümlesi FOTON_ARMS'tan okunur; H değişirse cümle yanlış olur."""
    out = tmp_path / "o.json"
    monkeypatch.setitem(foton_model.FOTON_ARMS, "h", ("A", "B", "A"))
    with pytest.raises(SystemExit, match="H'nin bütün modülleri A") as exc:
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)


def test_refuses_when_x_arm_is_not_the_measured_module_list(monkeypatch, tmp_path):
    out = tmp_path / "o.json"
    monkeypatch.setitem(foton_model.FOTON_ARMS, "x", ("A", "B", "B"))
    with pytest.raises(SystemExit, match=r"FOTON_ARMS\['x'\]") as exc:
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)


def test_refuses_when_attn_patterns_campaign_T_disagrees_with_the_preset(monkeypatch, tmp_path):
    """Başlık sayıları (128.17 / 128.50 / 128.38) T_KAMPANYA'da hesaplanır; ölçülen T ile aynı olmak zorunda."""
    out = tmp_path / "o.json"
    monkeypatch.setattr(attn_patterns, "T_KAMPANYA", 256)
    with pytest.raises(SystemExit, match="kampanya") as exc:
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)


def test_refuses_cuda_when_unavailable(monkeypatch, tmp_path):
    """CUDA yokken sessizce CPU'ya düşmek yok. Emniyet: kontrol sökülmüş olsa bile bu test GPU'ya DOKUNMASIN diye
    (bir eğitim kuyruğu koşuyor olabilir) cihaza ilk taşımadan önceki adım tuzaklanır."""
    out = tmp_path / "o.json"

    def tripwire(*a, **k):
        raise RuntimeError("CUDA kontrolü aşıldı")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(foton_model, "precompute_rope", tripwire)
    with pytest.raises(SystemExit, match="CUDA yok") as exc:
        ox1.main(["--out", str(out)])                                          # varsayılan device = cuda
    _assert_refused(exc, out)


def test_refuses_to_write_into_the_frozen_prereg_directory(monkeypatch, tmp_path):
    """Gerçek dondurulmuş dizine DOKUNMADAN: FROZEN_DIR geçici bir dizine yönlendirilir."""
    assert ox1.FROZEN_DIR == os.path.join(ROOT, "experiments", "foton-v02-prereg")
    frozen = tmp_path / "foton-v02-prereg"
    frozen.mkdir()
    monkeypatch.setattr(ox1, "FROZEN_DIR", str(frozen))
    out = frozen / "alt" / "ox1_microbench.json"
    with pytest.raises(SystemExit, match="dondurulmuş ön-kayıt dizini") as exc:
        ox1.run(str(out), **FAST)
    _assert_refused(exc, out)
    assert ox1._inside(str(frozen).upper(), str(frozen)) == (os.path.normcase("A") == "a")   # Windows: harf duyarsız
    assert not ox1._inside(str(tmp_path / "foton-v02-prereg-x" / "o.json"), str(frozen))     # önek benzerliği yetmez


@pytest.mark.parametrize("kw", [{"repeats": 0}, {"iters": 0}, {"batch": 0}, {"warmup": -1}, {"device": "mps"}])
def test_refuses_nonsense_counts(tmp_path, kw):
    out = tmp_path / "o.json"
    with pytest.raises(SystemExit, match="O-X1 RED") as exc:
        ox1.run(str(out), **{**FAST, **kw})
    _assert_refused(exc, out)


# ---------------------------------------------------------------------------------------- 3. komut satırı
def test_cli_exposes_no_pattern_or_size_flags():
    """W / r / s / T / n_head / batch / preset bayrağı YOK: bayat O-X1 metni bir bayrakla sızamaz."""
    parser = ox1.build_parser()
    flags = {s for a in parser._actions for s in a.option_strings}
    assert flags == {"-h", "--help", "--out", "--repeats", "--iters", "--device"}
    args = parser.parse_args([])
    assert vars(args) == {"out": "experiments/foton-v02/ox1_microbench.json", "repeats": 5, "iters": 30,
                          "device": "cuda"}
    assert set(inspect.signature(ox1.run).parameters) == {"out", "device", "repeats", "iters", "batch", "warmup",
                                                          "bf16"}


@pytest.mark.parametrize("argv", [["--W", "256"], ["--A_LOCAL_W", "256"], ["--r", "4"], ["--s", "2"],
                                  ["--T", "128"], ["--max_seq_len", "128"], ["--n_head", "4"], ["--batch", "1"],
                                  ["--preset", "tiny-foton"], ["--pattern", "A"], ["--rep", "2"],
                                  ["--device", "mps"]])
def test_cli_rejects_pattern_flags_and_abbreviations(argv, tmp_path, capsys, monkeypatch):
    out = tmp_path / "o.json"

    def tripwire(*a, **k):                      # argparse bayrağı geçirirse pinli (saatler süren CPU) koşu BAŞLAMASIN
        raise AssertionError(f"argparse {argv} bayrağını kabul etti")

    monkeypatch.setattr(ox1, "run", tripwire)
    with pytest.raises(SystemExit) as exc:
        ox1.main(["--out", str(out), "--device", "cpu", *argv])
    assert exc.value.code == 2                                                 # argparse hatası: hiçbir şey koşmadı
    assert not out.exists()
    capsys.readouterr()


def test_main_passes_flags_through_and_prints_the_warnings(monkeypatch, tmp_path, capsys):
    out = tmp_path / "alt" / "o.json"
    real, seen = ox1.run, {}

    def fast_run(o, **kw):
        seen.update(kw)
        return real(o, **{**kw, "batch": 1, "warmup": 0, "bf16": False})

    monkeypatch.setattr(ox1, "run", fast_run)
    ox1.main(["--out", str(out), "--device", "cpu", "--repeats", "2", "--iters", "1"])
    assert seen == {"device": "cpu", "repeats": 2, "iters": 1}                 # main batch/warmup/bf16'ya DOKUNMAZ
    with open(out, encoding="utf-8") as f:
        doc = json.load(f)
    assert sorted(doc["repeats"]) == ["r0", "r1"] and doc["protocol_deviations"]["repeats"] == {
        "pinned": 5, "realized": 2}
    printed = capsys.readouterr().out
    assert "UYARI" in printed and "raporlanabilir: False" in printed and "H kolu" in printed
