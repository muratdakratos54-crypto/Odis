"""Sahibin ölçek hakkı kararının defteri: S1 KOŞULMADAN ÖNCE, zaman damgalı, gerekçeli TEK satır.

Kilitlenen şey: seçim S1'in sonucu görüldükten sonra kayda geçirilemesin. "başarısız"ta hak
kullanılmazsa vazgeçmenin tek ölçek yolu, kullanılırsa iki ölçek yolu açılır — yani seçim sonucu
değiştirir ve sırası denetlenebilir olmak zorunda.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


od = _load("owner_decision")
mcq = _load("make_campaign_queue")

REASON = "S0 satırı 'belirsiz': B10 gereği tek yükseltme hakkı kullanılıyor"
T0 = time.mktime((2026, 10, 1, 12, 0, 0, 0, 0, -1))


@pytest.fixture()
def paths(tmp_path):
    basis = tmp_path / "decision.json"
    basis.write_text(json.dumps({"primary1": {"row": "belirsiz"}}), encoding="utf-8")
    return str(tmp_path / "owner_decisions.jsonl"), str(basis)


def test_record_writes_one_timestamped_line_with_reason_and_basis_hash(paths):
    log, basis = paths
    rec = od.record("S1'e çık", REASON, basis, log=log, now=T0)
    lines = open(log, encoding="utf-8").read().splitlines()
    assert len(lines) == 1 and json.loads(lines[0]) == rec
    assert rec["ts_local"] == "2026-10-01T12:00:00" and rec["ts_utc"].endswith("Z")
    assert rec["choice"] == "S1'e çık" and rec["reason"] == REASON
    assert rec["basis_sha256"] == od._sha256(basis) and len(rec["basis_sha256"]) == 64


def test_record_requires_a_real_reason_a_known_choice_and_an_existing_basis(paths):
    log, basis = paths
    for bad in ("", "ok", "   çünkü   "):
        with pytest.raises(ValueError, match="gerekçe"):
            od.record("S1'e çık", bad, basis, log=log)
    with pytest.raises(ValueError, match="seçim"):
        od.record("belki", REASON, basis, log=log)
    with pytest.raises(ValueError, match="dayanak"):
        od.record("S1'e çık", REASON, basis + ".yok", log=log)
    assert not os.path.exists(log)                       # reddedilen hiçbir çağrı deftere yazmadı


def test_the_decision_cannot_be_changed_after_it_is_logged(paths):
    log, basis = paths
    od.record("S1'e çıkma", REASON, basis, log=log, now=T0)
    with pytest.raises(ValueError, match="ZATEN kayıtlı"):
        od.record("S1'e çık", REASON, basis, log=log, now=T0 + 60)
    assert len(od.read(log)) == 1


def test_s1_queue_needs_a_logged_go_decision(paths):
    log, basis = paths
    with pytest.raises(PermissionError, match="kayıtlı değil"):
        od.require_s1_authorized(log)
    od.record("S1'e çıkma", REASON, basis, log=log, now=T0)
    with pytest.raises(PermissionError, match="karar değiştirilemez"):
        od.require_s1_authorized(log)


def test_escalation_used_is_read_from_the_log_and_has_no_default(paths):
    log, basis = paths
    with pytest.raises(ValueError, match="kayıtlı değil"):
        od.escalation_used(log)                          # varsayılan değer loglanmamış kararı gizlerdi
    od.record("S1'e çık", REASON, basis, log=log, now=T0)
    assert od.escalation_used(log) is True and od.require_s1_authorized(log)["choice"] == "S1'e çık"


def test_verify_order_flags_s1_runs_started_before_the_decision(paths):
    log, basis = paths
    s0 = {"run": "x-s11", "started": "2026-09-25T10:00:00"}
    early = {"run": "s1-x-s11", "started": "2026-10-01T11:59:59"}
    late = {"run": "s1-m-s11", "started": "2026-10-01T12:00:01"}
    assert od.verify_order([s0], log) == []              # S1 koşusu yoksa karar satırı da gerekmez
    assert "kayıtlı DEĞİL" in od.verify_order([s0, late], log)[0]
    od.record("S1'e çık", REASON, basis, log=log, now=T0)
    assert od.verify_order([s0, late], log) == []
    bad = od.verify_order([s0, early, late], log)
    assert len(bad) == 1 and "s1-x-s11" in bad[0] and "ÖNCE" in bad[0]
    assert len(od.verify_order([{"run": "s1-h-s11", "started": None}], log)) == 1   # bilinmeyen başlangıç temiz SAYILMAZ


def test_verify_order_flags_s1_runs_when_the_logged_choice_was_no_go(paths):
    log, basis = paths
    od.record("S1'e çıkma", REASON, basis, log=log, now=T0)
    bad = od.verify_order([{"run": "s1-m-s11", "started": "2026-10-02T09:00:00"}], log)
    assert any("S1'e çıkma" in b for b in bad)


def test_queue_builder_refuses_an_s1_queue_without_the_decision(paths, tmp_path, monkeypatch):
    log, basis = paths
    out = str(tmp_path / "s1.json")
    s1_arms = "foton-d-s1,foton-m-s1,foton-h-s1,foton-x-s1"       # ölçek KOL ADINDAN çıkar; ayrı bayrak yok
    argv = ["make_campaign_queue.py", "--arms", s1_arms, "--out", out, "--owner_log", log]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as e:
        mcq.main()
    assert e.value.code not in (0, None) and not os.path.exists(out)
    od.record("S1'e çık", REASON, basis, log=log, now=T0)
    mcq.main()
    q = json.load(open(out, encoding="utf-8"))
    assert q["_scale"] == "S1" and all(r["name"].startswith(od.S1_PREFIX) for r in q["runs"])
    assert q["_owner_decision"]["ts_local"] == "2026-10-01T12:00:00"
    # S0 kuyruğu karar satırı İSTEMEZ ve adlarına önek koymaz
    assert [r["_arm"] for r in q["runs"][:4]] == ["d", "m", "h", "x"] and q["runs"][0]["name"] == "s1-d-s11"
    out0 = str(tmp_path / "s0.json")
    monkeypatch.setattr(sys, "argv", ["make_campaign_queue.py", "--arms", "foton-d,foton-m,foton-h,foton-x",
                                      "--out", out0, "--owner_log", str(tmp_path / "yok.jsonl")])
    mcq.main()
    q0 = json.load(open(out0, encoding="utf-8"))
    assert q0["_scale"] == "S0" and not any(r["name"].startswith(od.S1_PREFIX) for r in q0["runs"])
    # karışık ölçek tek kuyrukta OLMAZ (S1 kolunu S0 kuyruğuna saklayıp kapıyı atlamak mümkün olmasın)
    with pytest.raises(ValueError, match="tek ölçek"):
        mcq.scale_of(["foton-d", "foton-x-s1"])
    assert mcq.S1_PREFIX == od.S1_PREFIX
