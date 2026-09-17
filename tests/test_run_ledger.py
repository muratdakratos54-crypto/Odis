"""scripts/run_ledger.py testleri: defterin SAYMAYI bıraktığı an ön-kaydın V11 raporu geçersiz olur.

Kilitlenen regresyonlar ve neden önemli oldukları:

* **Sürdürülen koşu düz "bitti" görünmemeli.** Denetimin bulduğu boşluk buydu: `--resume true` ile
  devam eden koşu bitince kaydı normal bir koşudan ayırt edilemiyordu, oysa duvar saati İKİ oturuma
  yayılmıştı ve V10(ii) gereği duvar saati ölçütüne girmez. Testler sürdürmenin iki diskteki
  imzasını (`time` geri atlaması ve `time < train_time`) ayrı ayrı doğruluyor — biri kaybolsa
  diğeri yakalamalı.
* **Kenara alınan (`.yarim-`) koşu deftere girmeli.** Kuyruk o klasörü bırakır, klasör elle silinir
  ve deneme tamamen kaybolur. Defterin var olma sebebi tam olarak bu.
* **Koşan bir koşu "iptal" diye kaydedilmemeli.** Defterin kendi uydurduğu bir iptal, saymadığı bir
  iptalden daha kötüdür.
* **Durum sözlüğü kapalı kalmalı.** Serbest metinli durum ilk acelede "ok"/"fail" olur ve V11'in
  istediği iptal/çökme ayrımı biter.
* **Yeniden kurulum idempotent olmalı.** Tarama iki kez koşarsa sayılar iki katına çıkardı.
* **Rapordaki sayılar kayıtlarla tutmalı.** Toplam = durumların toplamı, çöken = çöktü+oom+nan.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location("run_ledger", os.path.join(ROOT, "scripts", "run_ledger.py"))
rl = importlib.util.module_from_spec(_spec)
sys.modules["run_ledger"] = rl
_spec.loader.exec_module(rl)


# --- sentetik checkpoint ağacı ------------------------------------------------------------------------

def metrics(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def session(start_step: int, end_step: int, t0: float, train0: float, step: int = 250,
            loss: float = 3.0) -> list[dict]:
    """Tek oturumun `metrics.jsonl` satırları. `time` süreç başlangıcına göre (t0'dan), `train_time`
    checkpoint'ten devam ediyor (train0'dan) — gerçek `src/train.py` düzeni."""
    rows = []
    for s in range(start_step + step, end_step + 1, step):
        k = (s - start_step) / step
        rows.append({"type": "train", "step": s, "tokens": s * 32768, "loss": loss,
                     "train_time": train0 + 10 * k, "time": t0 + 12 * k})
        rows.append({"type": "val", "step": s, "tokens": s * 32768, "val_loss": loss + 0.2,
                     "train_time": train0 + 10 * k, "time": t0 + 12 * k + 1})
    return rows


def make_run(base, name: str, rows: list[dict], *, done: bool = False, end: bool = False,
             log: str = "", preset: str = "foton-m", seed: int = 11, dirty: bool = True) -> str:
    """Kuyruğun ve `src/train.py`'nin bıraktığı dosya düzenini birebir kurar."""
    d = base / name
    d.mkdir(parents=True)
    (d / "provenance.json").write_text(json.dumps({
        "src_sha256": "a" * 64, "git_head": "b" * 40,
        "git_status": [" M src/train.py"] if dirty else [],
        "args": ["--preset", preset, "--seed", str(seed)], "env": {}}), encoding="utf-8")
    (d / "config.json").write_text(json.dumps({"model": {}, "train": {"run_name": name, "seed": seed}}),
                                   encoding="utf-8")
    if end and rows:
        last = rows[-1]
        rows = rows + [{"type": "end", "step": last["step"], "tokens": last["step"] * 32768,
                        "train_time": last["train_time"], "time": last["time"] + 1, "best_val": 3.2}]
    (d / "metrics.jsonl").write_text(metrics(rows), encoding="utf-8")
    if log:
        (d / "train.log").write_text(log, encoding="utf-8")
    if done:
        (d / "DONE").write_text("", encoding="utf-8")
    old = time.time() - 4 * 3600           # koşuyor sanılmasın: her şey 4 saat önce yazılmış
    for f in d.iterdir():
        os.utime(f, (old, old))
    return str(d)


TRACEBACK = ("adım  250 | loss 3.00\nTraceback (most recent call last):\n"
             "  File \"src/train.py\", line 1, in <module>\nRuntimeError: bir şey patladı\n")
OOM_LOG = ("adım  250 | loss 3.00\nTraceback (most recent call last):\n"
           "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")


@pytest.fixture
def tree(tmp_path):
    """Dört koşu: düz bitmiş, sürdürülüp bitmiş, kenara alınmış yarım, bitmemiş çökme."""
    base = tmp_path / "checkpoints" / "kampanya"
    make_run(base, "d-s11", session(0, 1000, 0.0, 0.0), done=True, end=True, preset="foton-d", seed=11)
    # sürdürülen: ilk oturum 0->500 (time 0..), ikinci oturum 500->1000 ama `time` yeniden 0'dan
    make_run(base, "h-s11", session(0, 500, 0.0, 0.0) + session(500, 1000, 0.0, 250.0),
             done=True, end=True, preset="foton-h", seed=11)
    make_run(base, "m-s11.yarim-1", session(0, 500, 0.0, 0.0), preset="foton-m", seed=11)
    make_run(base, "x-s11", session(0, 250, 0.0, 0.0), log=TRACEBACK, preset="foton-x", seed=11)
    return base


# --- ekleme ve okuma ---------------------------------------------------------------------------------

def test_append_read_roundtrip_is_append_only_and_utf8(tmp_path):
    p = str(tmp_path / "alt" / "run_ledger.jsonl")        # üst klasör yoksa oluşturulmalı
    a = rl.append(rl.blank(run="d-s11", arm="d", seed=11, block="s11", status="bitti", steps=4375,
                           git_head="b" * 40, dirty=False, src_sha256="a" * 64,
                           config_sha256="c" * 64, started="2026-09-17T00:00:00",
                           ended="2026-09-17T00:30:00", reason="kuyruk bitirdi"), p)
    rl.append(rl.blank(run="m-s11", arm="m", seed=11, block="s11", status="çöktü",
                       reason="OOM değil, RuntimeError"), p)
    recs = rl.read(p)
    assert [r["run"] for r in recs] == ["d-s11", "m-s11"]
    assert recs[0] == a and recs[0]["steps"] == 4375 and recs[0]["dirty"] is False
    assert recs[1]["status"] == "çöktü" and recs[1]["dir"] == "m-s11"  # dir verilmezse run'dan
    assert all(r["schema"] == rl.SCHEMA and r["logged_at"] for r in recs)
    assert "çöktü" in open(p, encoding="utf-8").read()    # Türkçe karakterler kaçış dizisi olmadan
    assert set(recs[0]) == set(rl.FIELDS)                 # alanların TAMAMI her satırda tanımlı

    rl.append(rl.blank(run="h-s11", status="iptal", reason="elle durduruldu"), p)
    assert [r["run"] for r in rl.read(p)] == ["d-s11", "m-s11", "h-s11"]  # eski satırlar ezilmedi


def test_unknown_status_and_unknown_field_are_refused(tmp_path):
    """Durum sözlüğü ve şema kapalı: "ok"/"fail" gibi bir durum ya da yanlış yazılmış bir alan
    sessizce kabul edilirse V11'in iptal/çökme ayrımı ve alanların tamlığı kaybolur."""
    p = str(tmp_path / "run_ledger.jsonl")
    with pytest.raises(ValueError, match="sözlük kapalı"):
        rl.append({"run": "d-s11", "status": "ok"}, p)
    with pytest.raises(ValueError, match="alan yok"):
        rl.append({"run": "d-s11", "status": "bitti", "durum": "bitti"}, p)
    with pytest.raises(ValueError, match="'run' zorunlu"):
        rl.append({"status": "bitti"}, p)
    assert not os.path.exists(p)                          # reddedilen kayıt defteri yaratmamalı


def test_half_written_line_from_a_killed_process_does_not_corrupt_the_next_record(tmp_path):
    """Defter, kendini yazarken ölebilen bir süreçten geliyor: yarım satır yarım kalmalı, ama
    sonraki kayıt onunla BİRLEŞMEMELİ ve rapor bozuk satırı saklamamalı."""
    p = str(tmp_path / "run_ledger.jsonl")
    rl.append(rl.blank(run="d-s11", status="bitti"), p)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write('{"run": "m-s11", "status": "çök')       # süreç satırın ortasında öldü
    rl.append(rl.blank(run="h-s11", status="iptal", reason="makine yeniden başladı"), p)
    assert [r["run"] for r in rl.read(p)] == ["d-s11", "h-s11"]
    assert [i for i, _ in rl.damaged(p)] == [2]
    assert rl.counts(rl.read(p))["toplam"] == 2


# --- diskten yeniden kurulum -------------------------------------------------------------------------

def test_reconstruct_reads_status_arm_seed_and_provenance_from_disk(tree):
    recs, skipped = rl.reconstruct(str(tree))
    assert skipped == []
    by = {r["dir"]: r for r in recs}
    assert {d: r["status"] for d, r in by.items()} == {
        "d-s11": "bitti", "h-s11": "sürdürüldü", "m-s11.yarim-1": "iptal", "x-s11": "çöktü"}
    assert [r["arm"] for r in recs if r["dir"] == "x-s11"] == ["x"]
    assert all(r["seed"] == 11 and r["git_head"] == "b" * 40 and r["dirty"] is True for r in recs)
    assert all(r["src_sha256"] == "a" * 64 and len(r["config_sha256"]) == 64 for r in recs)
    assert all(r["source"] == "yeniden-kurulum" and r["started"] and r["ended"] for r in recs)
    # kenara alınan klasör: koşu adı sade, `dir` diskteki adı taşıyor (deneme diske geri bağlanabilsin)
    assert by["m-s11.yarim-1"]["run"] == "m-s11" and "kenara" in by["m-s11.yarim-1"]["reason"]
    assert by["m-s11.yarim-1"]["steps"] == 500 and by["d-s11"]["steps"] == 1000
    assert by["x-s11"]["evidence"] == "train.log-traceback"
    # blok diskte yazılı değil: uydurulmamalı, ama gerekçede söylenmeli
    assert all(r["block"] is None and "blok kimliği diskte" in r["reason"] for r in recs)
    recs2, _ = rl.reconstruct(str(tree), blocks={"d-s11": "s11", "h-s11": "s11", "m-s11": "s11"})
    assert {r["dir"]: r["block"] for r in recs2}["m-s11.yarim-1"] == "s11"


def test_resumed_finished_run_is_not_recorded_as_plain_finished(tree):
    """Denetimin bulduğu boşluk: sürdürülen koşu bitince "bitti" diye geçiyordu ve duvar saatinin
    iki oturuma yayıldığı kaydın hiçbir yerinde görünmüyordu."""
    by = {r["dir"]: r for r in rl.reconstruct(str(tree))[0]}
    plain, resumed = by["d-s11"], by["h-s11"]
    assert plain["status"] == "bitti" and plain["resumed"] is False and plain["sessions"] == 1
    assert resumed["status"] == "sürdürüldü" and resumed["resumed"] is True and resumed["sessions"] == 2
    assert resumed["status"] != plain["status"]            # defterden TEK BAŞINA ayırt edilebilir
    assert "duvar saati" in resumed["reason"] and "V10(ii)" in resumed["reason"]
    # ikisi de DONE + end taşıyor: ayrım yalnızca oturum imzasından gelebilir
    assert resumed["evidence"] == plain["evidence"] == "DONE"
    c = rl.counts([plain, resumed])
    assert c["bitti"] == 1 and c["sürdürüldü"] == 1 and c["sürdürülen"] == 1


def test_resume_detected_even_when_time_never_goes_backwards(tmp_path):
    """İkinci imza tek başına çalışmalı: `src/train.py` resume'da metrics'i checkpoint adımına kadar
    KIRPAR, yani ilk oturumdan hiç satır kalmadığında `time` geri atlaması görünmez olur. O durumda
    tek kanıt `time < train_time`'dır (tek oturumda `time` eval+kayıt süresini de içerdiği için
    daima `time >= train_time`)."""
    rows = [{"type": "train", "step": 4000, "tokens": 0, "loss": 3.0, "train_time": 900.0, "time": 8.0},
            {"type": "train", "step": 4025, "tokens": 0, "loss": 3.0, "train_time": 910.0, "time": 20.0}]
    assert all(b["time"] > a["time"] for a, b in zip(rows, rows[1:]))   # saat hiç geri atlamıyor
    n, sig = rl.sessions_of(rows)
    assert (n, sig) == (2, "time<train_time")
    d = make_run(tmp_path, "m-s22", rows, done=True, end=True, seed=22)
    assert rl.scan_run_dir(d)["status"] == "sürdürüldü"
    # sağlamlık: gerçek tek oturumlu koşularda bu imza YOK (10 baseline koşusunda 0 ihlal ölçüldü)
    assert rl.sessions_of(session(0, 1000, 0.0, 0.0)) == (1, "")


def test_oom_and_nan_are_separated_from_a_plain_crash(tmp_path):
    """oom, nan ve çöktü ayrı durumlar: sebebi (VRAM bütçesi / sayısal kararsızlık / hata) ve
    düzeltmesi farklı. NaN, "bitmiş" koşuda da kazanır — zehirli eğri karar hesabına girmemeli."""
    oom = make_run(tmp_path, "m-s33", session(0, 250, 0.0, 0.0), log=OOM_LOG, seed=33)
    crash = make_run(tmp_path, "m-s44", session(0, 250, 0.0, 0.0), log=TRACEBACK, seed=44)
    nan = make_run(tmp_path, "m-s55", session(0, 500, 0.0, 0.0, loss=float("nan")),
                   done=True, end=True, seed=55)
    assert rl.scan_run_dir(oom)["status"] == "oom"
    assert rl.scan_run_dir(oom)["evidence"] == "train.log-oom"
    assert rl.scan_run_dir(crash)["status"] == "çöktü"
    rec = rl.scan_run_dir(nan)
    assert rec["status"] == "nan" and rec["evidence"] == "metrics-nan"  # DONE'a rağmen
    assert rl.counts([rl.scan_run_dir(p) for p in (oom, crash, nan)])["çöken"] == 3


def test_a_still_running_run_is_not_invented_as_cancelled(tmp_path):
    """Koşmakta olan koşu (bitmemiş, çökme izi yok, metrics AZ ÖNCE yazılmış) deftere girmemeli:
    defterin uydurduğu bir iptal, saymadığı bir iptalden daha kötüdür."""
    base = tmp_path / "ckpt"
    live = make_run(base, "x-s11", session(0, 250, 0.0, 0.0), seed=11, preset="foton-x")
    now = time.time()
    os.utime(os.path.join(live, "metrics.jsonl"), (now, now))
    assert rl.scan_run_dir(live) is None
    recs, skipped = rl.reconstruct(str(base))
    assert recs == [] and skipped == ["x-s11"]
    # aynı klasör, süre geçmiş sayılırsa: iptal, ama kanıt yokluğu gerekçeye yazılmış olmalı
    rec = rl.scan_run_dir(live, running_grace_sec=0.0)
    assert rec["status"] == "iptal" and rec["evidence"] == "kanit-yok" and "ayrımının kanıtı" in rec["reason"]


def test_exit_code_tells_a_ctrl_c_cancel_from_a_crash(tmp_path):
    """Kuyruğun kancası çıkış kodunu biliyor: Windows'ta Ctrl-C ile öldürülen süreç ÇÖKMEDİ.
    Bu ayrım yalnızca burada kurulabilir, diskte izi yok."""
    d = make_run(tmp_path, "m-s11", session(0, 250, 0.0, 0.0), seed=11)
    assert rl.scan_run_dir(d, exit_code=1, running_grace_sec=0.0)["status"] == "çöktü"
    for code in rl.CTRL_C_CODES:
        rec = rl.scan_run_dir(d, exit_code=code, running_grace_sec=0.0)
        assert rec["status"] == "iptal" and rec["evidence"] == f"cikis-kodu-{code}"
    # çıkış kodu 0 ve DONE varsa kod değil KANIT konuşur
    done = make_run(tmp_path, "d-s11", session(0, 250, 0.0, 0.0), done=True, end=True, preset="foton-d")
    assert rl.scan_run_dir(done, exit_code=0)["status"] == "bitti"


def test_record_run_dir_writes_one_line_and_marks_the_queue_as_source(tree, tmp_path):
    """`scripts/run_queue.py`'nin kancasının çağırdığı uç: tek satır, kaynağı 'kuyruk', kuyruğun
    gerekçesi kaydın gerekçesinin BAŞINDA (diskten çıkarılanı silmeden)."""
    p = str(tmp_path / "run_ledger.jsonl")
    rec = rl.record_run_dir(str(tree / "m-s11.yarim-1"), block="s11",
                            reason="kuyruk yarım kalan koşuyu kenara aldı", path=p)
    assert rec["source"] == "kuyruk" and rec["block"] == "s11" and rec["status"] == "iptal"
    assert rec["reason"].startswith("kuyruk yarım kalan") and "kanıtı yoktur" in rec["reason"]
    assert len(rl.read(p)) == 1
    # klasör hiç oluşmadıysa (süreç açılışta öldü) deneme yine sayılmalı
    yok = rl.record_run_dir(str(tree / "hic-yok"), reason="süreç başlamadan öldü", path=p)
    assert yok["status"] == "iptal" and yok["evidence"] == "klasor-yok" and len(rl.read(p)) == 2


def test_set_aside_hook_does_not_count_an_attempt_the_queue_already_logged(tmp_path):
    """Kuyruk koşu bitince bir satır yazıyor; SONRAKİ kuyruk aynı klasörü `.yarim-n` diye kenara
    alırken aynı denemeyi ikinci kez görür. Deneme iki kez sayılırsa V11'in "toplam" sayısı şişer.
    Kenara alma kancasının işi yalnızca kaydı OLMAYAN denemeyi (kuyruk/makine ölmüş) yakalamaktır."""
    p = str(tmp_path / "run_ledger.jsonl")
    d = make_run(tmp_path / "ckpt", "m-s22", session(0, 250, 0.0, 0.0), seed=22)
    first = rl.record_run_dir(d, block="s22", exit_code=1, reason="kuyruk çalıştırdı", path=p)
    assert first["status"] == "çöktü" and len(rl.read(p)) == 1
    os.rename(d, d + ".yarim-1")                          # kuyruğun yaptığı şey; mtime'lar korunur
    assert rl.record_run_dir(d + ".yarim-1", block="s22", reason="kenara alındı",
                             only_if_new_attempt=True, path=p) is None
    assert len(rl.read(p)) == 1 and rl.counts(rl.read(p))["toplam"] == 1

    # kaydı olmayan deneme: kenara alma o denemenin TEK izi, mutlaka yazılmalı
    kayitsiz = make_run(tmp_path / "ckpt", "m-s33.yarim-1", session(0, 250, 0.0, 0.0), seed=33)
    rec = rl.record_run_dir(kayitsiz, block="s33", reason="defterde kaydı yoktu",
                            only_if_new_attempt=True, path=p)
    assert rec is not None and rec["status"] == "iptal" and len(rl.read(p)) == 2


# --- idempotanlık ve rapor ---------------------------------------------------------------------------

def test_reconstruction_is_idempotent_but_records_a_later_status_change(tree, tmp_path):
    """Tarama iki kez koşarsa sayılar iki katına çıkmamalı. Ama aynı klasörün DURUMU değişirse
    (yarım -> sürdürülüp bitti) V10(iii) her ikisinin de deftere yazılmasını istiyor."""
    p = str(tmp_path / "run_ledger.jsonl")
    recs, _ = rl.reconstruct(str(tree))
    assert len(rl.append_missing(recs, p)) == 4
    assert rl.append_missing(rl.reconstruct(str(tree))[0], p) == []      # ikinci tarama: sıfır satır
    assert rl.counts(rl.read(p))["toplam"] == 4

    changed = dict(recs[0], status="sürdürüldü", resumed=True, sessions=2)
    assert len(rl.append_missing([changed], p)) == 1
    assert rl.counts(rl.read(p))["toplam"] == 5 and rl.counts(rl.read(p))["koşu_adı_sayısı"] == 4


def test_report_counts_match_the_records(tree, tmp_path, capsys):
    p = str(tmp_path / "run_ledger.jsonl")
    recs, _ = rl.reconstruct(str(tree), blocks={"d-s11": "s11", "h-s11": "s11", "x-s11": "s11"})
    rl.append_missing(recs, p)
    rl.append(rl.blank(run="m-s22", status="iptal", block="s22", reason="elle durduruldu"), p)
    c = rl.print_report(p)
    assert c["toplam"] == 5 == sum(c[s] for s in rl.STATUSES)            # toplam = durumların toplamı
    assert (c["bitti"], c["sürdürüldü"], c["iptal"], c["çöktü"]) == (1, 1, 2, 1)
    assert c["çöken"] == c["çöktü"] + c["oom"] + c["nan"] == 1
    assert c["sürdürülen"] == 1 and c["koşu_adı_sayısı"] == 5
    # duvar saati ölçütü: 'bitti' + tek oturum + blok belli -> yalnızca d-s11
    assert c["duvar_saati_uygun"] == 1 and c["blok_bilinmeyen"] == 1
    out = capsys.readouterr().out
    assert "TOPLAM deneme 5 | İPTAL 2 | ÇÖKEN 1" in out                  # V11'in istediği üç sayı
    assert "bozuk satır 0" in out and "sürdürüldü 1" in out


def test_report_on_an_empty_or_missing_ledger_says_zero_instead_of_failing(tmp_path, capsys):
    """Rapor, defter yokken de sayı vermeli: "dosya yok" diye susmak, "0 iptal" demekle aynı değildir
    ve raporu yazan kişi ikisini ayırt edemezse V11 sayısı doğrulanamaz."""
    c = rl.print_report(str(tmp_path / "hic-yok.jsonl"))
    assert c["toplam"] == 0 and c["çöken"] == 0
    assert "TOPLAM deneme 0 | İPTAL 0 | ÇÖKEN 0" in capsys.readouterr().out


def test_real_ledger_holds_the_ten_baseline_runs():
    """Ön-kaydın istediği artefakt GERÇEKTEN var ve dolu: sabit yolda, 10 baseline koşusuyla."""
    if not os.path.exists(rl.LEDGER):
        pytest.skip("defter henüz kurulmadı (scripts/run_ledger.py --reconstruct)")
    recs = rl.read(rl.LEDGER)
    base = [r for r in recs if r["run"].startswith("baseline-")]
    assert len(base) == 10 and all(r["status"] == "bitti" and r["steps"] == 4375 for r in base)
    assert sorted(r["run"] for r in base) == \
        sorted(f"baseline-{a}-s{s}" for a in "dm" for s in (11, 22, 33, 44, 55))
    assert rl.damaged(rl.LEDGER) == []
