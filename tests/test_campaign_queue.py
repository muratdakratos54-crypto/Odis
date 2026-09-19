"""Kampanya kuyruğu üretecinin regresyon testleri (ön-kayıt V11a: seed bloğu + Latin kare).

Mutasyon sınavının yakaladığı boşluk: Latin kare denge assertion'ı yalnızca `blok % kol == 0`
iken çalışıyordu, yani kampanyanın GERÇEK biçiminde (5 seed × 4 kol) hiç çalışmıyordu. Rotasyon
satırı silinip sıra sabitlendiğinde hiçbir kontrol ateşlenmiyordu.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("mcq", os.path.join(ROOT, "scripts", "make_campaign_queue.py"))
mcq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcq)

ARMS = ["foton-d", "foton-m", "foton-h", "foton-x"]
SEEDS = [11, 22, 33, 44, 55]


def test_blocks_are_seed_blocks_and_every_arm_runs_once_per_block():
    runs = mcq.build_runs(ARMS, SEEDS)
    assert len(runs) == 20
    for bi, s in enumerate(SEEDS):
        block = runs[bi * 4:(bi + 1) * 4]
        assert {r["_block"] for r in block} == {f"s{s}"}, "blok bitişik değil: seed blokları iç içe geçmiş"
        assert sorted(r["_arm"] for r in block) == ["d", "h", "m", "x"]
        assert all(r["args"][-1] == str(s) for r in block)


def test_latin_square_rotation_balances_positions():
    runs = mcq.build_runs(ARMS, SEEDS)
    pos = mcq.positions(runs, ARMS, SEEDS)
    for a, p in pos.items():
        assert all(p.count(k) >= 1 for k in range(4)), f"{a} bir pozisyonu hiç görmedi: {p}"
    mcq.check_balance(runs, ARMS, SEEDS)                 # gerçek kampanya biçiminde de ÇALIŞIR


def test_check_balance_catches_a_fixed_order_queue():
    """Sabit sıralı (rotasyonsuz) bir kuyruk REDDEDİLMELİ — 5 × 4 biçiminde de."""
    fixed = []
    for s in SEEDS:
        for a in ARMS:
            fixed.append({"name": f"{a}-s{s}", "args": [], "_block": f"s{s}", "_arm": a.replace("foton-", "")})
    with pytest.raises(AssertionError, match="Latin kare dengesi"):
        mcq.check_balance(fixed, ARMS, SEEDS)


def test_check_balance_catches_a_missing_arm():
    runs = [r for r in mcq.build_runs(ARMS, SEEDS) if r["name"] != "x-s33"]
    with pytest.raises(AssertionError, match="kol kümesi bozuk"):
        mcq.check_balance(runs, ARMS, SEEDS)


def test_residual_position_imbalance_is_the_arithmetic_minimum():
    """5 blok × 4 kolda tam denge İMKÂNSIZ: ilk 4 blok tam Latin kare (kol başına toplam 6),
    beşinci blok 0..3'ü bir permütasyonla ekler -> ortalamalar daima {1.2, 1.4, 1.6, 1.8}.
    Ön-kayıt bu artık dengesizliği sayısıyla yazıyor; test sayının doğru olduğunu kilitler."""
    pos = mcq.positions(mcq.build_runs(ARMS, SEEDS), ARMS, SEEDS)
    means = sorted(round(sum(p) / len(p), 2) for p in pos.values())
    assert means == [1.2, 1.4, 1.6, 1.8]
    assert round(sum(pos["x"]) / 5 - sum(pos["m"]) / 5, 2) == 0.4
    assert round(sum(pos["x"]) / 5 - sum(pos["h"]) / 5, 2) == 0.2


# --- blok kimliği: katmanlar arasında tekil (V11a-ii: duvar saati yalnızca AYNI blok içinde) -------------

S1_ARMS = [a + mcq.S1_SUFFIX for a in ARMS]


def _tiers() -> dict[str, tuple[list[str], list[dict]]]:
    """Aynı seed'lerle dört katman: kimlikleri ayıran tek şey önek olsun."""
    return {"S0 karar": (ARMS, mcq.build_runs(ARMS, SEEDS)),
            "S0 tarama": (ARMS, mcq.build_runs(ARMS, SEEDS, tag="screen", stop_step=2000)),
            "S1": (S1_ARMS, mcq.build_runs(S1_ARMS, SEEDS)),
            "S1 tarama": (S1_ARMS, mcq.build_runs(S1_ARMS, SEEDS, tag="screen"))}


def test_block_ids_do_not_collide_between_tiers():
    """Blok kimliği koşu defterine yazılıyor ve V11a(ii) duvar saatini yalnızca AYNI blok içinde
    karşılaştırıyor. Kimlik her katmanda `s<seed>` olsaydı S0 kararının `s11`'i ile S1'in (ya da taramanın)
    `s11`'i defterde TEK blok görünürdü; `run_queue.py` de aynı out_dir'e yöneltilen iki katmanın koşularını
    tek blok sayıp V11a(iii)'ü yanlış koşulara uygulardı."""
    ids = {t: {r["_block"] for r in runs} for t, (_, runs) in _tiers().items()}
    assert ids["S0 karar"] == {f"s{s}" for s in SEEDS}              # commit'li karar kuyruğuyla aynı kalmalı
    assert ids["S0 tarama"] == {f"screen-s{s}" for s in SEEDS}
    assert ids["S1"] == {f"s1-s{s}" for s in SEEDS}
    assert ids["S1 tarama"] == {f"s1-screen-s{s}" for s in SEEDS}
    tiers = sorted(ids)
    for i, a in enumerate(tiers):
        for b in tiers[i + 1:]:
            assert not ids[a] & ids[b], f"blok kimlikleri çakışıyor: {a} ∩ {b} = {sorted(ids[a] & ids[b])}"


def test_block_id_carries_the_same_prefix_as_the_run_names_and_balance_checks_still_run():
    """Kimliğin öneki koşu adınınkiyle AYNI kuraldan gelir (ad = önek + kol + seed), ve `check_balance` /
    `positions` önekli kimliklerle de çalışır — ilk hâlleri blokları `s<seed>` diye yeniden türetiyordu ve
    etiketli ya da S1 kuyruğunda boş blok görüp kuyruğu reddederdi."""
    base = mcq.positions(mcq.build_runs(ARMS, SEEDS), ARMS, SEEDS)
    for tier, (arms, runs) in _tiers().items():
        for r in runs:
            prefix, seed = r["_block"].rsplit("s", 1)
            assert r["name"] == f"{prefix}{r['_arm']}-s{seed}", f"{tier}: ad ile blok kimliği ayrışmış: {r}"
        mcq.check_balance(runs, arms, SEEDS)
        assert mcq.positions(runs, arms, SEEDS) == base, f"{tier}: Latin kare sırası katmana göre değişmiş"
        assert [r["_arm"] for r in runs] == [r["_arm"] for r in mcq.build_runs(ARMS, SEEDS)]


def test_check_balance_catches_two_seeds_sharing_one_block_id():
    """İki seed aynı kimliği alırsa yürütücü 8 koşuyu TEK blok sayar: birindeki kesinti diğerinin bitmiş
    koşularını da iptal ettirir, defterde de iki blok tek blok görünür."""
    runs = mcq.build_runs(ARMS, SEEDS)
    for r in runs:
        if r["_block"] == "s22":
            r["_block"] = "s11"
    with pytest.raises(AssertionError, match="blok sayısı"):
        mcq.check_balance(runs, ARMS, SEEDS)


def test_an_s0_tag_cannot_impersonate_the_s1_prefix():
    """`--tag s1` S0 koşularına `s1-` öneki verirdi: adlar S1 koşusu gibi görünür ve blok kimlikleri
    S1'inkilerle birebir çakışırdı (`s1-s11`)."""
    assert {r["_block"] for r in mcq.build_runs(S1_ARMS, [11])} == {"s1-s11"}
    with pytest.raises(ValueError, match="S1 önekiyle"):
        mcq.build_runs(ARMS, [11], tag="s1")
    with pytest.raises(ValueError, match="tek ölçek"):      # karışık ölçekte bloğun TEK kimliği olamaz
        mcq.build_runs(["foton-d", "foton-x-s1"], [11])


def test_committed_decision_queue_run_list_is_reproduced_exactly():
    """Commit'li karar kuyruğu ön-kayıt paketinin parçası: üreteç, koşu listesini (adlar, argümanlar, sıra,
    `_block`, `_arm`) BİREBİR yeniden üretmek zorunda. Blok kimliği değişikliği etiketsiz S0'a dokunmamalı."""
    path = os.path.join(ROOT, "experiments", "foton-v02-prereg", "queues", "decision.json")
    if not os.path.exists(path):
        pytest.skip("commit'li karar kuyruğu bu ağaçta yok")
    with open(path, encoding="utf-8") as f:
        committed = json.load(f)
    assert mcq.build_runs(ARMS, SEEDS) == committed["runs"]


def test_strict_flag_writes_strict_true_and_nothing_else_changes(tmp_path, monkeypatch):
    """`--strict` kuyruğa `"_strict": true` yazar (run_queue.py kirli kodla başlamaz); bayrak yoksa anahtar
    da yoktur. Koşu listesi iki durumda aynıdır: sıkılık sırayı ya da adları etkileyemez."""
    out = {}
    for flag in ([], ["--strict"]):
        path = str(tmp_path / f"q{len(flag)}.json")
        monkeypatch.setattr(sys, "argv", ["make_campaign_queue.py", "--arms", ",".join(ARMS),
                                          "--seeds", "11,22,33,44,55", "--out", path, *flag])
        mcq.main()
        with open(path, encoding="utf-8") as f:
            out[bool(flag)] = json.load(f)
    assert "_strict" not in out[False]
    assert out[True]["_strict"] is True
    assert out[True]["runs"] == out[False]["runs"] == mcq.build_runs(ARMS, SEEDS)
    assert out[True]["_blocks"] == out[False]["_blocks"] and len(out[True]["_blocks"]) == 5
