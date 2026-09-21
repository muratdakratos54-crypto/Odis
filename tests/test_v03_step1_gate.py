"""v03 adım 1 — seed 101 geçerlilik kapısı (`v03_step1_gate.py`) ve kampanya sürücüsü (`v03_step1_campaign.py`).

Sahibin talimatı: seed 101'in dört kolu bitince YALNIZCA iki geçerlilik kapısına bakılır (ön koşulun seed-başı bacağı, D
akla-yatkınlığın seed-başı bacağı); δ ve δ_D HESAPLANMAZ; kapı aşılırsa kuyruk DURUR. Sınanan sınıflar: kapının bir
bacağının yok sayılması, aşılmış kapıya rağmen devam, bozuk / eksik koşunun "geçti" sayılması, kapı çıktısına ara sonuç
(bir kolun val kaybı ya da δ) sızması, sürücünün kapıyı atlaması ya da kapı / kuyruk hatasından sonra sürmesi, diskteki
iki kuyruğun onaylı 12 koşuluk kuyruktan sapması.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t = _load("tests/test_v03_step1.py", "step1_helpers")               # sentetik koşu yazıcıları (GERÇEK preset'lerden config)
gate = _load("scripts/v03_step1_gate.py", "v03_step1_gate")
camp = _load("scripts/v03_step1_campaign.py", "v03_step1_campaign")
dec = gate.dec
REAL_SRC = gate.run_queue.code_hash(ROOT)                           # kapı, koşuların src'sini ŞU ANKİ ağacınkiyle karşılaştırır
# Çıktıya SIZMAMASI gereken sayılar ayırt edilebilsin diye kolların düzeyleri kendine özgü seçildi
L = {"x-dom": 3.0437, "h-dom": 3.0519, "x-iid": 3.1303, "d": 3.1211, "v02": 3.1311}
# Kapı çıktısında (JSON + stdout) HİÇBİR biçimde görünmemesi gerekenler: X-dom / H-dom'un kaybı, D'nin adım-2188 kaybı ve
# her iki delta. Yalnız literal düzeyler değil, TÜRETİLMİŞ sayılar da aranır (afin kodlanmış sızıntı: "başarısız olamayan ölçü").
D_END = L["d"] + 0.00002 * (t.END - dec.D_REF_STEP)                 # D'nin adım-2188 kaybı (kapı yalnız adım 2000'i okur)
FORBIDDEN = {"val(X-dom)": L["x-dom"], "val(H-dom)": L["h-dom"], "val(D, 2188)": D_END,
             "delta": L["x-iid"] - L["x-dom"], "delta_D": D_END - L["x-dom"],
             "H-dom - X-dom": L["h-dom"] - L["x-dom"]}


def assert_no_arm_loss_or_delta(text: str, where: str) -> None:
    for name, value in FORBIDDEN.items():
        for digits in (3, 4, 5, 6):
            assert f"{value:.{digits}f}" not in text, f"{where}: {name} sızdı ({value:.{digits}f})"


def seed101(tmp_path, levels=None, overrides=None, skip=(), src=None):
    """YALNIZCA seed 101'in koşuları diskte: kapı, seed 102 / 103 hiç yokken çalışmak ZORUNDA."""
    lv, kw = {**L, **(levels or {})}, overrides or {}
    runs, v02 = tmp_path / "runs", tmp_path / "v02"
    for arm in t.KIND:
        if arm not in skip:
            t._write_arm(runs, v02, arm, 101, lv[arm], **{"src": src or REAL_SRC, **kw.get(arm, {})})
    return str(runs), str(v02)


def run_main(tmp_path, runs, v02, capsys=None):
    out = tmp_path / "gate.json"
    rc = gate.main(["--runs", runs, "--v02", v02, "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    if capsys is not None:
        text += "\n" + capsys.readouterr().out                      # JSON DOSYASI + stdout: ikisi de sınanır
    return rc, json.loads(out.read_text(encoding="utf-8")), text


def test_a_passing_gate_writes_only_the_two_flags_no_number_at_all(tmp_path, capsys):
    """Kapı AŞILMADIYSA kuyruk sürer: çıktıda HİÇBİR sayı olmamalı. İki kapının kendi nicelikleri bile, diskteki sabitlerle
    toplandığında X-iid-101'in val_atanan'ını (tanım gereği mean_fixed'i) ve D-101'in adım-2000 kaybını verir."""
    rc, res, text = run_main(tmp_path, *seed101(tmp_path), capsys=capsys)
    assert rc == gate.EXIT_OK == 0 and res["exceeded"] is False and res["seed"] == 101 == gate.GATE_SEED
    assert set(res) == {"what", "seed", "integrity", "precondition_seed_leg", "d_sanity_seed_leg", "exceeded", "NOT_computed"}
    assert res["precondition_seed_leg"] == {"clean": True} and res["d_sanity_seed_leg"] == {"clean": True}
    assert_no_arm_loss_or_delta(text, "geçen kapı")
    for leaked in (L["x-iid"], L["v02"], L["d"], L["x-iid"] - L["v02"], L["d"] + 0.00002 * 188 - dec.D_REF_VAL):
        for digits in (3, 4, 5, 6):
            assert f"{abs(leaked):.{digits}f}" not in text, leaked       # kapının KENDİ nicelikleri de geçmiyorsa yazılmaz
    assert "HESAPLANMAZ" in res["NOT_computed"] and "101" in text


def test_an_exceeded_gate_writes_the_two_gate_quantities_and_still_no_arm_loss_or_delta(tmp_path, capsys):
    """Kapı AŞILDIYSA kampanya durur ve sahibe rapor gider: iki kapının nicelikleri yazılır. X-dom / H-dom'un kaybı,
    D'nin adım-2188 kaybı ve iki delta yine hiçbir yerde yoktur."""
    rc, res, text = run_main(tmp_path, *seed101(tmp_path, {"x-iid": L["v02"] + 0.0151}), capsys=capsys)
    pre, d = res["precondition_seed_leg"], res["d_sanity_seed_leg"]
    assert rc == gate.EXIT_EXCEEDED and res["exceeded"] is True
    assert pre["mean_fixed_diff_nat"] == pytest.approx(0.0151, abs=1e-6) and pre["clean"] is False
    assert abs(pre["ceiling_diff_points"]) < 0.4 and pre["limits"] == [dec.PRE_MEAN_FIXED_NAT, dec.PRE_CEILING_POINTS]
    assert d["deviation_nat_at_ref_step"] == pytest.approx(L["d"] + 0.00002 * 188 - dec.D_REF_VAL, abs=1e-6) and d["clean"] is True
    assert set(pre) == {"mean_fixed_diff_nat", "ceiling_diff_points", "limits", "clean"}
    assert set(d) == {"deviation_nat_at_ref_step", "reference", "clean"}
    assert_no_arm_loss_or_delta(text, "aşılan kapı")


@pytest.mark.parametrize("levels, leg", [({"x-iid": L["v02"] + 0.0151}, "precondition_seed_leg"), ({"x-iid": L["v02"] - 0.0151}, "precondition_seed_leg"),
                                         ({"d": 3.1198 + 0.0121 - 0.00376}, "d_sanity_seed_leg"), ({"d": 3.1198 - 0.0121 - 0.00376}, "d_sanity_seed_leg")])
def test_each_gate_leg_stops_the_campaign_on_either_side_and_just_inside_the_limit_it_does_not(tmp_path, levels, leg):
    rc, res, _ = run_main(tmp_path, *seed101(tmp_path, levels))
    other = ({"precondition_seed_leg", "d_sanity_seed_leg"} - {leg}).pop()
    assert rc == gate.EXIT_EXCEEDED == 3 and res["exceeded"] is True and res[leg]["clean"] is False and res[other]["clean"] is True
    inside = {k: (L["v02"] + (v - L["v02"]) * 0.98 if k == "x-iid" else 3.1198 - 0.00376 + (v - 3.1198 + 0.00376) * 0.98) for k, v in levels.items()}
    rc, res, _ = run_main(tmp_path / "ic", *seed101(tmp_path / "ic", inside))
    assert rc == 0 and res["exceeded"] is False and res[leg] == {"clean": True}


def test_the_ceiling_leg_alone_stops_the_campaign(tmp_path):
    wide = {"x-iid": {"sm_of_step": t.generalist(L["x-iid"], "x-iid", 101, sd=0.010)}}              # v02-X-101'in gürültüsü 0.004
    rc, res, _ = run_main(tmp_path, *seed101(tmp_path, overrides=wide))
    pre = res["precondition_seed_leg"]
    assert rc == 3 and abs(pre["mean_fixed_diff_nat"]) < dec.PRE_MEAN_FIXED_NAT and pre["ceiling_diff_points"] > dec.PRE_CEILING_POINTS


@pytest.mark.parametrize("kw, skip, msg", [({"x-dom": {"check_shift": 3e-4}}, (), "iki yoldan uyuşmuyor"),
                                           ({"d": {"order_seed": 999}}, (), "veri sırası"),
                                           ({"h-dom": {"init_seed": 999}}, (), "aynı ağırlıklardan başlamamış"),
                                           ({"d": {"digest": "ef" * 32}}, (), "aynı val pencerelerinde"),
                                           ({"x-iid": {"dirty": True}}, (), "code_dirty"),
                                           ({"h-dom": {"src": "b" * 64}}, (), "TEK src"),
                                           ({"d": {"lr_2000": 0.00035}}, (), "LR takvimi"),
                                           ({"x-dom": {"cfg_patch": [("train", "seed", 102)]}}, (), "koşu kimliği"),
                                           ({"x-iid": {"finished": False}}, (), "bitmemiş"),
                                           ({}, ("d",), "metrics.jsonl yok"), ({}, ("x-dom",), "metrics.jsonl yok")])
def test_a_broken_or_missing_seed_101_run_stops_the_campaign_and_is_never_a_pass(tmp_path, kw, skip, msg, capsys):
    rc, res, text = run_main(tmp_path, *seed101(tmp_path, overrides=kw, skip=skip), capsys=capsys)
    assert rc == gate.EXIT_INTEGRITY == 4 and res["exceeded"] is None and msg in res["integrity_error"]
    assert "precondition_seed_leg" not in res and "d_sanity_seed_leg" not in res
    assert_no_arm_loss_or_delta(text, f"bütünlük hatası ({msg})")        # hata METNİ de bir kolun kaybını sızdırmamalı


def test_a_seed_101_block_run_with_another_src_than_the_current_tree_stops_the_campaign(tmp_path, capsys):
    """Kuyruk ikiye bölündü: `run_queue`'nun kuyruk-içi src denetimi iki kuyruğun ARASINI görmez. Dört koşu kendi içinde
    tutarlı ama başka bir src ile koşmuşsa, kalan 8 koşu başka kodla koşardı ve analiz 12 koşudan SONRA karar üretmezdi."""
    rc, res, text = run_main(tmp_path, *seed101(tmp_path, src="c" * 64), capsys=capsys)
    assert rc == gate.EXIT_INTEGRITY and "src/ değişti" in res["integrity_error"] and "cccccccccccc" in res["integrity_error"]
    assert_no_arm_loss_or_delta(text, "src sürekliliği")
    assert run_main(tmp_path / "tamam", *seed101(tmp_path / "tamam"))[0] == gate.EXIT_OK     # şu anki ağacın özetiyle geçer


def test_an_unexpected_exception_is_an_integrity_error_not_a_pass(tmp_path, monkeypatch, capsys):
    """Kapı `except Exception` yakalar: yarım yazılmış bir npz (zipfile.BadZipFile) ya da başka beklenmedik hata da kuyruğu
    DURDURUR ve sebebi DOSYAYA yazar; traceback + rc=1 ile sessizce ölmez."""
    monkeypatch.setattr(gate.an, "read_v02", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("beklenmedik")))
    rc, res, text = run_main(tmp_path, *seed101(tmp_path), capsys=capsys)
    assert rc == gate.EXIT_INTEGRITY and res["exceeded"] is None and "RuntimeError: beklenmedik" in res["integrity_error"]
    assert_no_arm_loss_or_delta(text, "beklenmedik istisna")


def test_the_gate_legs_are_the_same_functions_the_full_rule_uses():
    """Kapı ile kuyruk-sonu kuralı AYNI bacak fonksiyonlarını çağırır: biri gevşerse öbürü de gevşer, ikisi ayrışamaz."""
    same, ceil = {s: 3.13 for s in dec.SEEDS}, {s: 1.4 for s in dec.SEEDS}
    for dm in (0.0149, 0.0151, -0.0149, -0.0151):
        full = dec.precondition({**same, 101: 3.13 + dm}, same, ceil, ceil)["clean"]
        assert dec.precondition_seed_leg(dm, 0.0) is (abs(dm) <= dec.PRE_MEAN_FIXED_NAT) and full is dec.precondition_seed_leg(dm, 0.0)
    for dc in (0.39, 0.41, -0.41):
        assert dec.precondition_seed_leg(0.0, dc) is dec.precondition(same, same, {**ceil, 101: 1.4 + dc}, ceil)["clean"]
    for dev in (0.0119, 0.0121, -0.0121):
        full = dec.d_sanity({101: 3.1198 + dev, 102: 3.1198 - dev / 2, 103: 3.1198 - dev / 2})["clean"]
        assert dec.d_sanity_seed_leg(3.1198 + dev) is (abs(dev) <= dec.D_REF_TOL) and full is dec.d_sanity_seed_leg(3.1198 + dev)
    for bad in (float("nan"), None, True, "3.12"):
        with pytest.raises((TypeError, ValueError)):
            dec.d_sanity_seed_leg(bad)
        with pytest.raises((TypeError, ValueError)):
            dec.precondition_seed_leg(bad, 0.0)


# ------------------------------------------------------------------------------------------------ sürücü
def drive(codes: dict):
    """Sahte `run`: komutun türüne göre çıkış kodu döndürür, çağrı sırasını kaydeder."""
    calls = []

    def run(cmd):
        kind = "gate" if "v03_step1_gate.py" in cmd[1] else ("A" if cmd[2] == camp.QUEUE_A else "B" if cmd[2] == camp.QUEUE_B else "?")
        calls.append((kind, cmd))
        return codes.get(kind, 0)
    status, rc = camp.campaign(run, lambda msg: None)
    return status, rc, [k for k, _ in calls], calls


def test_driver_marks_itself_running_before_the_first_queue_and_clears_a_stale_gate_output(tmp_path, monkeypatch):
    """Durum dosyası yoksa 'bu sürücü hiç başlamadı', 'running' kalmışsa süreç öldü demektir; ikisi karışmamalı. Bayat bir
    gate_s101.json da silinir: yeniden başlatmada eski denemenin kapı sonucu bu kampanyanınki sanılmasın."""
    out = tmp_path / "ckpt"
    out.mkdir()
    for name, value in (("OUT", str(out)), ("STATUS", str(out / "campaign_status.json")), ("LOG", str(out / "campaign.log")),
                        ("GATE_OUT", str(out / "gate_s101.json"))):
        monkeypatch.setattr(camp, name, value)
    monkeypatch.setattr(camp.os, "chdir", lambda p: None)
    (out / "gate_s101.json").write_text(json.dumps({"exceeded": False, "eski": True}), encoding="utf-8")
    seen = []

    def fake_run(cmd, cwd=None):
        seen.append(json.loads((out / "campaign_status.json").read_text(encoding="utf-8")))
        assert not (out / "gate_s101.json").exists()                      # bayat kapı çıktısı silindi
        raise KeyboardInterrupt                                            # finally: durum dosyası son hâline yazılır

    monkeypatch.setattr(camp.subprocess, "run", fake_run)
    with pytest.raises(KeyboardInterrupt):
        camp.main()
    assert seen and seen[0]["status"] == "running" and seen[0]["pid"] == os.getpid()
    assert json.loads((out / "campaign_status.json").read_text(encoding="utf-8"))["status"] == "crashed"


def test_driver_runs_seed_101_then_the_gate_then_the_rest_in_that_order():
    status, rc, kinds, calls = drive({})
    assert (status, rc, kinds) == ("completed", 0, ["A", "gate", "B"])
    assert all(os.path.basename(c[1]) == "run_queue.py" for k, c in calls if k != "gate")
    gate_cmd = calls[1][1]
    assert gate_cmd[2:] == ["--out", camp.GATE_OUT] and os.path.normpath(camp.GATE_OUT).startswith("checkpoints")       # depoya YAZMAZ
    assert all(os.path.normpath(p).startswith("checkpoints") for p in (camp.STATUS, camp.LOG))


@pytest.mark.parametrize("gate_rc, status", [(3, "gate_exceeded"), (4, "gate_integrity_error"), (1, "gate_integrity_error"), (2, "gate_integrity_error")])
def test_driver_never_starts_the_remaining_queue_when_the_gate_does_not_return_zero(gate_rc, status):
    got, rc, kinds, _ = drive({"gate": gate_rc})
    assert (got, rc, kinds) == (status, gate_rc, ["A", "gate"])


def test_driver_stops_without_the_gate_when_the_first_queue_fails_and_reports_a_failed_second_queue():
    assert drive({"A": 90})[:3] == ("queue_a_failed", 90, ["A"])
    assert drive({"B": 1})[:3] == ("queue_b_failed", 1, ["A", "gate", "B"])


def test_the_two_queues_on_disk_are_the_approved_twelve_runs_split_at_the_seed_101_gate():
    def load(path):
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            return json.load(f)
    full, a, b = load("experiments/foton-v03-step1/queues/step1.json"), load(camp.QUEUE_A), load(camp.QUEUE_B)
    assert a["runs"] + b["runs"] == full["runs"] and len(full["runs"]) == 12
    assert [r["name"] for r in a["runs"]] == [f"v03s1-{arm}-s101" for arm in ("x-dom", "h-dom", "x-iid", "d")]          # sahibin sırası
    assert {r["name"].rsplit("-s", 1)[1] for r in b["runs"]} == {"102", "103"}
    assert [r["name"].rsplit("-s", 1)[1] for r in b["runs"]] == ["102"] * 4 + ["103"] * 4                                  # bloklar bitişik
    for q in (full, a, b):
        assert q["_strict"] is True and os.path.normpath(q["out_dir"]) == os.path.normpath(camp.OUT) and q["cwd"] == "."
    first = [blk[0].split("-s")[0] for blk in full["_blocks"]]
    assert len(set(first)) == 3                                                                                           # dönüşümlü: her blok başka kolla başlar


def test_driver_main_writes_the_status_file_outside_the_repo_and_replaces_a_stale_one(tmp_path, monkeypatch):
    out = tmp_path / "ckpt"
    out.mkdir()
    for name, value in (("OUT", str(out)), ("STATUS", str(out / "campaign_status.json")), ("LOG", str(out / "campaign.log")),
                        ("GATE_OUT", str(out / "gate_s101.json"))):
        monkeypatch.setattr(camp, name, value)
    monkeypatch.setattr(camp.os, "chdir", lambda p: None)
    (out / "campaign_status.json").write_text(json.dumps({"status": "completed", "rc": 0}), encoding="utf-8")     # bayat
    seen = []

    class Done:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(cmd, cwd=None):
        seen.append(cmd)
        return Done(0 if len(seen) == 1 else 3)                                          # kuyruk A tamam, kapı AŞILDI

    monkeypatch.setattr(camp.subprocess, "run", fake_run)
    assert camp.main() == 3 and len(seen) == 2
    status = json.loads((out / "campaign_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "gate_exceeded" and status["rc"] == 3
    assert "gate_exceeded" in (out / "campaign.log").read_text(encoding="utf-8")


HARNESS = """
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("camp", sys.argv[1])
camp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camp)
out = sys.argv[2]
camp.OUT, camp.STATUS, camp.LOG = out, os.path.join(out, "campaign_status.json"), os.path.join(out, "campaign.log")
camp.GATE_OUT = os.path.join(out, "gate_s101.json")
camp.os.chdir = lambda p: None
class Done:
    returncode = 90
camp.subprocess.run = lambda cmd, cwd=None: Done()
sys.exit(camp.main())
"""


def test_driver_survives_a_redirected_cp1252_stdout_as_a_detached_windows_process_has(tmp_path):
    """Bağımsız süreçte stdout dosyaya yönlenir; Windows'ta o zaman cp1252'dir ve Türkçe karakterli İLK günlük satırı
    UnicodeEncodeError atar. İlk hâlde sürücü kuyruğu hiç başlatmadan, durum dosyasını da yazmadan ölüyordu."""
    import subprocess
    harness, out = tmp_path / "harness.py", tmp_path / "ckpt"
    out.mkdir()
    harness.write_text(HARNESS, encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    with open(tmp_path / "stdout.txt", "wb") as so, open(tmp_path / "stderr.txt", "wb") as se:
        rc = subprocess.run([sys.executable, str(harness), os.path.join(ROOT, "scripts", "v03_step1_campaign.py"), str(out)],
                            stdout=so, stderr=se, env=env).returncode
    err = (tmp_path / "stderr.txt").read_text(encoding="utf-8", errors="replace")
    assert rc == 90, err
    assert json.loads((out / "campaign_status.json").read_text(encoding="utf-8"))["status"] == "queue_a_failed"
    assert "kuyruğu" in (tmp_path / "stdout.txt").read_text(encoding="utf-8") and "UnicodeEncodeError" not in err
    assert "queue_a_failed" in (out / "campaign.log").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------ DEVAM: seed 102 / 103 (sahibin onayı)
def block(tmp_path, seed, levels=None, overrides=None, src=None):
    lv, kw = {**L, **(levels or {})}, overrides or {}
    runs, v02 = tmp_path / "runs", tmp_path / "v02"
    for arm in t.KIND:
        t._write_arm(runs, v02, arm, seed, lv[arm], **{"src": src or REAL_SRC, **kw.get(arm, {})})
    return str(runs), str(v02)


def run_gate(tmp_path, runs, v02, seed, legs, capsys=None):
    out = tmp_path / f"gate_s{seed}.json"
    rc = gate.main(["--runs", runs, "--v02", v02, "--out", str(out), "--seed", str(seed), "--legs", legs])
    text = out.read_text(encoding="utf-8") + ("\n" + capsys.readouterr().out if capsys is not None else "")
    return rc, json.loads(out.read_text(encoding="utf-8")), text


@pytest.mark.parametrize("seed", [102, 103])
def test_the_precondition_only_gate_passes_with_a_wildly_off_dense_arm_and_writes_no_number(tmp_path, capsys, seed):
    """Seed 102 / 103'te YALNIZ ön koşulun seed-başı bacağı yürürlükte: D'ye bakılmaz ((b) zaten okunmaz), D-102 / D-103
    ne kadar saparsa sapsın kuyruk durmaz. Kapı geçtiğinde yine hiçbir sayı yazılmaz."""
    rc, res, text = run_gate(tmp_path, *block(tmp_path, seed, {"d": 3.30}), seed, "precondition", capsys)
    assert rc == gate.EXIT_OK and res["exceeded"] is False and res["seed"] == seed
    assert res["precondition_seed_leg"] == {"clean": True} and res["d_sanity_seed_leg"] == gate.D_NOT_EVALUATED
    assert "YALNIZCA ön koşulun seed-başı bacağı" in res["what"] and "deviation_nat" not in text
    assert_no_arm_loss_or_delta(text, f"seed {seed} ön koşul kapısı")
    # AYNI veri, seed 101 kipinde (iki bacak) kapıyı DURDURURDU: fark gerçekten kipten geliyor
    assert run_gate(tmp_path / "iki", *block(tmp_path / "iki", seed, {"d": 3.30}), seed, "both")[0] == gate.EXIT_EXCEEDED


@pytest.mark.parametrize("seed", [102, 103])
def test_the_precondition_only_gate_stops_the_campaign_when_the_precondition_leg_is_exceeded(tmp_path, capsys, seed):
    rc, res, text = run_gate(tmp_path, *block(tmp_path, seed, {"x-iid": L["v02"] + 0.0151}), seed, "precondition", capsys)
    pre = res["precondition_seed_leg"]
    assert rc == gate.EXIT_EXCEEDED and res["exceeded"] is True and pre["clean"] is False
    assert pre["mean_fixed_diff_nat"] == pytest.approx(0.0151, abs=1e-6) and res["d_sanity_seed_leg"] == gate.D_NOT_EVALUATED
    assert "deviation_nat" not in text                                    # D'nin sayısı aşılmış kapıda bile yazılmaz
    assert_no_arm_loss_or_delta(text, f"seed {seed} aşılan ön koşul kapısı")


@pytest.mark.parametrize("kw, msg", [({"d": {"lr_2000": 0.00035}}, "LR takvimi"), ({"d": {"order_seed": 999}}, "veri sırası"),
                                     ({"d": {"digest": "ef" * 32}}, "aynı val pencerelerinde"),
                                     ({"x-dom": {"check_shift": 3e-4}}, "iki yoldan uyuşmuyor")])
def test_the_precondition_only_gate_still_checks_the_integrity_of_all_four_runs_including_dense(tmp_path, kw, msg):
    """D'nin DEĞERİNE bakılmaz ama BÜTÜNLÜĞÜNE bakılır: analiz 12 koşunun hepsini ister; bozuk bir D-102 kuyrukta kalmasın."""
    rc, res, _ = run_gate(tmp_path, *block(tmp_path, 102, overrides=kw), 102, "precondition")
    assert rc == gate.EXIT_INTEGRITY and msg in res["integrity_error"] and res["seed"] == 102


def test_the_gate_rejects_an_unknown_mode_and_an_unknown_seed(tmp_path):
    with pytest.raises(ValueError, match="legs"):
        gate.gate(*block(tmp_path, 102), seed=102, legs="yalniz-d")
    with pytest.raises(SystemExit):
        gate.main(["--out", str(tmp_path / "g.json"), "--seed", "104"])


def drive_continuation(codes: dict):
    calls = []

    def run(cmd):
        if "v03_step1_gate.py" in cmd[1]:
            seed, out = int(cmd[cmd.index("--seed") + 1]), cmd[cmd.index("--out") + 1]
            kind = f"gate{seed}:" + cmd[cmd.index("--legs") + 1]
            assert os.path.normpath(out).startswith("checkpoints")                                   # depoya YAZMAZ
            assert out == (camp.RECHECK_OUT if seed == 101 else camp.GATE_OUT_S[seed])               # kendi dosyasına
            assert out != camp.GATE_OUT                                   # seed 101'in AŞILMIŞ kapı kaydının üzerine YAZMAZ
        else:
            kind = {camp.QUEUE_S102: "q102", camp.QUEUE_S103: "q103"}.get(cmd[2], "BEKLENMEYEN:" + cmd[2])
        calls.append(kind)
        return codes.get(kind, 0)
    status, rc = camp.continuation(run, lambda msg: None)
    return status, rc, calls


def test_the_continuation_runs_seed_102_its_gate_seed_103_its_gate_in_that_order():
    status, rc, calls = drive_continuation({})
    assert (status, rc) == ("completed", 0)
    # 0. adım seed 101'in yeniden denetimi (yalnız ön koşul kipinde); sonra 102, kapısı, 103, kapısı — seed 101 KUYRUĞU yok
    assert calls == ["gate101:precondition", "q102", "gate102:precondition", "q103", "gate103:precondition"]


R0 = ["gate101:precondition"]


@pytest.mark.parametrize("codes, want_status, want_calls", [
    ({"gate101:precondition": 4}, "s101_recheck_failed", R0),
    ({"gate101:precondition": 3}, "s101_recheck_failed", R0),
    ({"gate102:precondition": 3}, "gate_s102_exceeded", R0 + ["q102", "gate102:precondition"]),
    ({"gate102:precondition": 4}, "gate_s102_integrity_error", R0 + ["q102", "gate102:precondition"]),
    ({"gate102:precondition": 1}, "gate_s102_integrity_error", R0 + ["q102", "gate102:precondition"]),
    ({"q102": 90}, "queue_s102_failed", R0 + ["q102"]),
    ({"q103": 1}, "queue_s103_failed", R0 + ["q102", "gate102:precondition", "q103"]),
    ({"gate103:precondition": 3}, "gate_s103_exceeded", R0 + ["q102", "gate102:precondition", "q103", "gate103:precondition"]),
    ({"gate103:precondition": 4}, "gate_s103_integrity_error", R0 + ["q102", "gate102:precondition", "q103", "gate103:precondition"])])
def test_the_continuation_stops_at_every_non_zero_code_and_waits(codes, want_status, want_calls):
    status, rc, calls = drive_continuation(codes)
    assert status == want_status and rc == max(codes.values()) and calls == want_calls


def test_the_continuation_preserves_the_record_of_the_exceeded_seed_101_gate(tmp_path, monkeypatch):
    out = tmp_path / "ckpt"
    out.mkdir()
    s102, s103, s101, rechk = out / "gate_s102.json", out / "gate_s103.json", out / "gate_s101.json", out / "gate_s101_recheck.json"
    for name, value in (("OUT", str(out)), ("STATUS", str(out / "campaign_status.json")), ("LOG", str(out / "campaign.log")),
                        ("GATE_OUT", str(s101)), ("GATE_OUT_S", {102: str(s102), 103: str(s103)}), ("RECHECK_OUT", str(rechk))):
        monkeypatch.setattr(camp, name, value)
    monkeypatch.setattr(camp.os, "chdir", lambda p: None)
    for f in (s101, s102, s103, rechk):
        f.write_text(json.dumps({"exceeded": True, "eski": f.name}), encoding="utf-8")
    seen = []

    class Done:
        returncode = 0

    def fake_run(cmd, cwd=None):
        seen.append(json.loads((out / "campaign_status.json").read_text(encoding="utf-8"))["status"])
        assert s101.exists() and not s102.exists() and not s103.exists() and not rechk.exists()   # 101'in kaydı DURUR
        return Done()

    monkeypatch.setattr(camp.subprocess, "run", fake_run)
    assert camp.main(["--continue-after-s101"]) == 0 and seen == ["running"] * 5
    assert json.loads(s101.read_text(encoding="utf-8"))["eski"] == "gate_s101.json"
    assert json.loads((out / "campaign_status.json").read_text(encoding="utf-8"))["status"] == "completed"
    log = (out / "campaign.log").read_text(encoding="utf-8")
    assert "DEVAM (sahibin onayı" in log and "seed 102 ön koşul kapısı" in log and "KAMPANYA DURUMU: completed" in log


def test_the_split_queues_are_the_approved_second_queue_cut_at_the_seed_boundary_nothing_else():
    """Kuyruk kısaltılmadı ve YENİDEN ÜRETİLMEDİ (Latin kare dönüşümü değişirdi): aynı 8 koşu, aynı sıra, aynı argümanlar."""
    def load(path):
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            return json.load(f)
    b, s102, s103 = load(camp.QUEUE_B), load(camp.QUEUE_S102), load(camp.QUEUE_S103)
    assert s102["runs"] + s103["runs"] == b["runs"] and len(b["runs"]) == 8
    for q, seed in ((s102, 102), (s103, 103)):
        assert {r["name"].rsplit("-s", 1)[1] for r in q["runs"]} == {str(seed)} and len(q["runs"]) == 4
        assert q["_blocks"] == [[r["name"] for r in q["runs"]]] and q["_strict"] is True and "_sliced_from" in q
        assert {k: v for k, v in q.items() if k not in ("runs", "_blocks", "_sliced_from")} == \
               {k: v for k, v in b.items() if k not in ("runs", "_blocks")}
    assert s102["runs"][0]["name"].startswith("v03s1-h-dom") and s103["runs"][0]["name"].startswith("v03s1-x-iid")



def test_the_real_output_paths_of_the_continuation_never_collide_with_the_seed_101_record():
    """Gerçek (monkeypatch'siz) yollar: seed 102 / 103 kapıları ve 0. adımın yeniden denetimi seed 101'in aşılmış kapı
    kaydının (gate_s101.json) üzerine YAZAMAZ ve birbirinin üzerine yazmaz."""
    paths = [camp.GATE_OUT, camp.RECHECK_OUT, camp.GATE_OUT_S[102], camp.GATE_OUT_S[103]]
    assert len({os.path.normpath(p) for p in paths}) == 4 and set(camp.GATE_OUT_S) == {102, 103}
    assert all(os.path.normpath(p).startswith(os.path.normpath(camp.OUT)) for p in paths)
    assert [os.path.basename(p) for p in paths] == ["gate_s101.json", "gate_s101_recheck.json", "gate_s102.json", "gate_s103.json"]


def test_the_precondition_only_gate_still_checks_src_continuity_against_the_current_tree(tmp_path, capsys):
    rc, res, text = run_gate(tmp_path, *block(tmp_path, 102, src="c" * 64), 102, "precondition", capsys)
    assert rc == gate.EXIT_INTEGRITY and "src/ değişti" in res["integrity_error"] and res["legs"] == "precondition"
    assert_no_arm_loss_or_delta(text, "seed 102 src sürekliliği")


def test_the_precondition_only_gate_stops_on_the_ceiling_leg_alone(tmp_path):
    """Ön koşulun seed-başı bacağı İKİ parçadır (mean_fixed VE tavan); 102 / 103'te tavan bacağı da yürürlükte."""
    wide = {"x-iid": {"sm_of_step": t.generalist(L["x-iid"], "x-iid", 102, sd=0.014)}}
    rc, res, _ = run_gate(tmp_path, *block(tmp_path, 102, overrides=wide), 102, "precondition")
    pre = res["precondition_seed_leg"]
    assert rc == gate.EXIT_EXCEEDED and abs(pre["mean_fixed_diff_nat"]) < dec.PRE_MEAN_FIXED_NAT
    assert pre["ceiling_diff_points"] > dec.PRE_CEILING_POINTS


def test_the_record_of_a_precondition_only_gate_names_its_own_seed_and_no_dense_quantity(tmp_path):
    rc, res, _ = run_gate(tmp_path, *block(tmp_path, 103, {"x-iid": L["v02"] + 0.0151}), 103, "precondition")
    assert rc == gate.EXIT_EXCEEDED and "X-iid-103" in res["NOT_computed"] and "D-101" not in res["NOT_computed"]
    assert "YALNIZCA ön koşulun seed-başı" in res["NOT_computed"] and gate.not_computed(101, "both") == gate.NOT_COMPUTED


S101_DIR = os.path.join(ROOT, "checkpoints", "foton-v03-step1")


@pytest.mark.skipif(not all(os.path.isdir(os.path.join(S101_DIR, f"v03s1-{a}-s101")) for a in ("d", "x-dom", "h-dom", "x-iid")),
                    reason="gerçek seed-101 koşuları diskte yok (mutasyon kumu)")
def test_the_step_zero_recheck_passes_on_the_real_seed_101_block_and_the_original_gate_still_fails(tmp_path):
    """Bu gece koşacak 0. adım GERÇEK veride geçmeli (yoksa 8 koşu hiç başlamaz); aynı veride iki bacaklı kapı yine aşılmalı
    (D-101'in +0.012132'lik sapması değişmez: (b) bu turda okunmaz)."""
    res = gate.gate(S101_DIR, os.path.join(ROOT, "checkpoints", "foton-v02-tarama"), seed=101, legs="precondition")
    assert res["exceeded"] is False and res["precondition_seed_leg"] == {"clean": True}
    both = gate.gate(S101_DIR, os.path.join(ROOT, "checkpoints", "foton-v02-tarama"), seed=101, legs="both")
    assert both["exceeded"] is True and both["d_sanity_seed_leg"]["clean"] is False
    assert both["d_sanity_seed_leg"]["deviation_nat_at_ref_step"] == pytest.approx(0.012132, abs=1e-6)
