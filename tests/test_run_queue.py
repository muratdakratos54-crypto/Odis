"""scripts/run_queue.py testleri: kuyruk yürütücüsü ön-kayıt V11a'ya uyuyor mu, kod izi temiz mi.

Kilitlenen kurallar ve neden önemli oldukları:

* **Yarıda kesilen blok TAMAMEN yeniden koşulur (V11a-iii).** Eski yürütücü biten koşuyu atlayıp yalnızca yarım
  kalanı yeniden başlatıyordu: kesintiden önce biten D ile saatler sonra koşan X aynı "blok" sayılıyor, yani
  V11a'nın kırmak için yazıldığı koşu-oturumu confound'u sessizce geri geliyordu. Kampanyadan sonra onarılamaz.
* **Kenara alma SİLMEZ.** İptal edilen denemenin klasörü o denemenin tek ham kaydıdır; "her ikisi de koşu
  defterine yazılır" (V11a-iii) ancak klasör duruyorsa denetlenebilir.
* **Defter şişmez ve yalan söylemez.** İptal edilen bitmiş koşu ikinci bir `bitti` satırı OLMAZ (duvar saatine
  uygun sayısı iki katına çıkardı); kuyruğun zaten yazdığı çöküş ikinci kez sayılmaz.
* **Oturum kimliği süreç başına TEKTİR.** V11a duvar saati büyüklüğünü oturum kimliği olmadan alıntılamayı yasaklıyor.
* **`code_dirty` kampanyanın kendi çıktılarını saymaz**, sıkı kuyruk ise commit'lenmemiş kodla BAŞLAMAZ ve
  kuyruk sürerken src/ değişirse sıradaki koşuyu başlatmadan DURUR.

Hiçbir test GPU ya da eğitim alt süreci kullanmaz: `execute(launch=..., cmd=...)` sahte başlatıcı alır, karar
mantığı (`plan_blocks`) saf fonksiyondur. Defter HER ZAMAN geçici bir yola yazılır (`ledger_path` zorunlu).
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rl = _load("run_ledger")           # önce: run_queue'nun kancası `import run_ledger` ile BU modülü bulsun
rq = _load("run_queue")

ARMS = ("d", "m", "h", "x")


# --- sentetik kuyruk, sahte başlatıcı, sahte git --------------------------------------------------------

def make_runs(blocks=("s11", "s22"), arms=ARMS) -> list[dict]:
    return [{"name": f"{a}-{b}", "args": ["--preset", f"foton-{a}", "--seed", b.lstrip("s")],
             "_block": b, "_arm": a} for b in blocks for a in arms]


def loose(names) -> list[dict]:
    """`_block` taşımayan (eski tip) koşular."""
    return [{"name": n, "args": ["--preset", "dense", "--seed", "1"]} for n in names]


@pytest.fixture
def env(tmp_path):
    """Geçici "depo" (src/ özeti gerçek `code_hash` ile alınır), çıktı klasörü ve GEÇİCİ defter."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "model.py").write_text("x = 1\n", encoding="utf-8")

    class Env:
        out = tmp_path / "ckpt"
        ledger = str(tmp_path / "defter" / "run_ledger.jsonl")
        src_file = repo / "src" / "model.py"

        def queue(self, runs, **extra) -> dict:
            return {"out_dir": str(self.out), "cwd": str(repo), "python": sys.executable, "runs": runs, **extra}

        def execute(self, q, launch, status: str | None = "", **kw) -> int:
            return rq.execute(q, "test-kuyruğu.json", ledger_path=self.ledger, launch=launch,
                              cmd=fake_git(status), echo=lambda *a, **k: None, **kw)

        def rows(self) -> list[dict]:
            return rl.read(self.ledger)

        def dirs(self) -> list[str]:
            return sorted(p.name for p in self.out.iterdir() if p.is_dir()) if self.out.exists() else []

        def log(self) -> str:
            return (self.out / "queue.log").read_text(encoding="utf-8")

        def prov(self, name: str) -> dict:
            return json.loads((self.out / name / "provenance.json").read_text(encoding="utf-8"))

    return Env()


def fake_git(status: str | None, head: str = "c" * 40):
    """`run_queue._cmd` yerine: git cevapları sabit, python/nvidia-smi sorguları boş (torch import edilmez).
    Gerçek `_cmd` gibi tüm çıktıyı `strip()`'ler — İLK satırın baştaki boşluğu bu yüzden düşer."""
    def cmd(args, cwd):
        if args[:2] == ["git", "status"]:
            return None if status is None else status.strip()
        if args[:2] == ["git", "rev-parse"]:
            return None if status is None else head
        return ""
    return cmd


class Launcher:
    """Eğitim süreci yerine geçer: metrics.jsonl + train.log yazar, istenen koşuda hata koduyla döner.
    `train.log`'a kaçıncı deneme olduğunu yazar ki kenara alınan klasörün İÇERİĞİ de doğrulanabilsin."""

    def __init__(self):
        self.launched: list[str] = []
        self.attempts: dict[str, int] = {}      # {koşu adı: kaçıncı deneme}; `launched` temizlense de sürer
        self.fail: dict[str, int] = {}          # {koşu adı: çıkış kodu}; kullanılınca düşer (tek seferlik)
        self.on_launch = None                   # (koşu adı) -> None: koşu "sırasında" olan şeyler

    def __call__(self, cmd, cwd, env, log_path) -> int:
        name = cmd[cmd.index("--run_name") + 1]
        self.launched.append(name)
        self.attempts[name] = self.attempts.get(name, 0) + 1
        run_dir = os.path.dirname(log_path)
        assert os.path.exists(os.path.join(run_dir, "provenance.json")), "kod izi koşudan ÖNCE yazılmalı"
        if self.on_launch:
            self.on_launch(name)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"deneme {self.attempts[name]}\n")
        rows = [{"type": "train", "step": 250, "loss": 3.0, "train_time": 10.0, "time": 12.0}]
        rc = self.fail.pop(name, 0)
        if rc == 0:
            rows.append({"type": "end", "step": 250, "train_time": 10.0, "time": 13.0})
        with open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        return rc


def put_dir(env, run: dict, state: str, provenance: bool = True) -> None:
    """Önceki bir kuyruğun bıraktığı klasörü elle kurar: 'bitti' | 'sürdürüldü' | 'yarım'. Deftere HİÇBİR şey
    yazılmaz — makinenin yeniden başladığı / kuyruğun öldüğü durum budur."""
    assert state in ("bitti", "sürdürüldü", "yarım"), state
    d = env.out / run["name"]
    d.mkdir(parents=True)
    if provenance:
        (d / "provenance.json").write_text(json.dumps(
            {"src_sha256": "a" * 64, "git_head": "b" * 40, "git_status": [], "args": run["args"]}), encoding="utf-8")
    rows = [{"type": "train", "step": 250, "loss": 3.0, "train_time": 10.0, "time": 12.0}]
    if state == "sürdürüldü":
        # `--resume true` ile ikinci oturum: `time` SÜREÇ başına göre sayıldığı için geriye atlar, `train_time`
        # checkpoint'ten devam eder — koşu defterinin "sürdürüldü" imzası (run_ledger.sessions_of).
        rows.append({"type": "train", "step": 500, "loss": 2.9, "train_time": 20.0, "time": 5.0})
    if state in ("bitti", "sürdürüldü"):
        rows.append({"type": "end", "step": rows[-1]["step"], "train_time": rows[-1]["train_time"],
                     "time": rows[-1]["time"] + 1.0})
        (d / "DONE").write_text("", encoding="utf-8")
    if provenance:
        (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "train.log").write_text("önceki oturum\n", encoding="utf-8")


def ops(plan, op) -> list[str]:
    return [a["run"] for a in plan if a["op"] == op]


def state(runs, **given) -> tuple[dict, dict]:
    """(finished, exists) — KİMLİKLE anahtarlı. `given`: {ad_altçizgili: 'bitti' | 'yarım'}; gerisi 'yok'."""
    s = {r["name"]: "yok" for r in runs}
    for k, v in given.items():
        name = k.replace("_", "-")
        assert name in s, name
        s[name] = v
    return {n: v == "bitti" for n, v in s.items()}, {n: v != "yok" for n, v in s.items()}


# --- plan_blocks: saf karar mantığı ----------------------------------------------------------------------

def test_untouched_queue_runs_everything_in_queue_order():
    runs = make_runs()
    plan = rq.plan_blocks(runs, *state(runs))
    assert [a["op"] for a in plan] == ["run"] * 8
    assert ops(plan, "run") == [r["name"] for r in runs]


def test_fully_finished_block_is_skipped_and_the_next_block_runs():
    runs = make_runs()
    plan = rq.plan_blocks(runs, *state(runs, d_s11="bitti", m_s11="bitti", h_s11="bitti", x_s11="bitti"))
    assert ops(plan, "skip") == ["d-s11", "m-s11", "h-s11", "x-s11"]
    assert ops(plan, "run") == ["d-s22", "m-s22", "h-s22", "x-s22"]
    assert ops(plan, "set_aside") == []                   # bitmiş bloğa DOKUNULMAZ


def test_interrupted_block_is_set_aside_entirely_and_rerun_from_its_first_run():
    """V11a(iii): D ve M bitmiş, H yarım, X hiç başlamamış -> var olan ÜÇ klasör de kenara, DÖRT koşu da yeniden.
    Eski davranış (D ve M'yi atla, H'den devam et) kesintiden önceki ve sonraki koşuları tek blok sayıyordu."""
    runs = make_runs()
    plan = rq.plan_blocks(runs, *state(runs, d_s11="bitti", m_s11="bitti", h_s11="yarım"))
    aside = [a for a in plan if a["op"] == "set_aside"]
    assert [(a["run"], a["kind"], a["block"]) for a in aside] == [
        ("d-s11", "blok-iptal", "s11"), ("m-s11", "blok-iptal", "s11"), ("h-s11", "blok-iptal", "s11")]
    assert ops(plan, "skip") == []
    assert ops(plan, "run") == [r["name"] for r in runs]   # bitmiş D ve M DAHİL, bloğun ilk koşusundan
    assert plan[:3] == aside                               # kenara alma kuyruk BAŞLARKEN, ilk koşudan önce


def test_block_interrupted_between_two_runs_is_also_rerun():
    """Makine D ve M bittikten SONRA, H başlamadan yeniden başladı: diskte yarım klasör YOK ama blok yine
    kesintiye uğradı. "Yarım klasör var mı" diye bakan bir kural bunu kaçırır ve H, X'i başka oturumda koşardı."""
    runs = make_runs()
    plan = rq.plan_blocks(runs, *state(runs, d_s11="bitti", m_s11="bitti"))
    assert ops(plan, "set_aside") == ["d-s11", "m-s11"]
    assert ops(plan, "run")[:4] == ["d-s11", "m-s11", "h-s11", "x-s11"] and ops(plan, "skip") == []


def test_every_interrupted_block_is_handled_at_queue_start_and_finished_ones_are_left_alone():
    runs = make_runs(blocks=("s11", "s22", "s33"))
    fin, ex = state(runs, d_s11="bitti", m_s11="bitti", h_s11="bitti", x_s11="bitti",   # s11 tamam
                    m_s22="yarım",                                                        # s22: ilk koşusu yok!
                    d_s33="bitti", x_s33="yarım")                                         # s33 de kesilmiş
    plan = rq.plan_blocks(runs, fin, ex)
    kinds = [a["op"] for a in plan]
    first_run = kinds.index("run")
    assert ops(plan[:first_run], "set_aside") == ["m-s22", "d-s33", "x-s33"]
    assert "set_aside" not in kinds[first_run:]            # bloklu kuyrukta sonradan kenara alma olmaz
    assert ops(plan, "skip") == ["d-s11", "m-s11", "h-s11", "x-s11"]
    assert ops(plan, "run") == [r["name"] for r in runs if r["_block"] != "s11"]


def test_runs_without_a_block_keep_the_old_per_run_behaviour():
    runs = loose(["a", "b", "c"])
    plan = rq.plan_blocks(runs, *state(runs, a="bitti", b="yarım"))
    assert plan == [{"op": "skip", "run": "a"},
                    {"op": "set_aside", "run": "b", "kind": "yarim", "block": None},
                    {"op": "run", "run": "b"},
                    {"op": "run", "run": "c"}]


def test_plan_invariants_hold_for_every_disk_state():
    """Uygulamadan BAĞIMSIZ yazılmış bir hakem, 3^5 = 243 disk durumunun hepsinde: iki blok (2'şer koşu) ve
    bir bloksuz koşu; her koşu yok / yarım / bitti."""
    runs = [{"name": "d-s11", "_block": "s11"}, {"name": "m-s11", "_block": "s11"},
            {"name": "serbest"},
            {"name": "m-s22", "_block": "s22"}, {"name": "d-s22", "_block": "s22"}]
    names = [r["name"] for r in runs]
    for combo in itertools.product(("yok", "yarım", "bitti"), repeat=len(runs)):
        st = dict(zip(names, combo))                       # kimlikle anahtarlı; zip yalnızca sözlüğü kurar
        plan = rq.plan_blocks(runs, {n: s == "bitti" for n, s in st.items()}, {n: s != "yok" for n, s in st.items()})
        ran, skipped, aside = ops(plan, "run"), ops(plan, "skip"), ops(plan, "set_aside")
        ctx = f"durum {st} -> plan {plan}"
        assert sorted(ran + skipped) == sorted(names), ctx            # her koşu TAM bir kez: ya koşar ya atlanır
        assert ran == [n for n in names if n in ran], ctx             # kuyruk sırası korunur
        for b in ("s11", "s22"):
            members = [r["name"] for r in runs if r.get("_block") == b]
            all_done = all(st[n] == "bitti" for n in members)
            # blok ya TAMAMEN atlanır (hepsi bitmişse) ya TAMAMEN koşulur — karışık blok V11a(iii) ihlalidir
            assert [n in ran for n in members] == [not all_done] * len(members), ctx
            want_aside = [] if all_done else [n for n in members if st[n] != "yok"]
            assert [n for n in aside if n in members] == want_aside, ctx
        assert ("serbest" in skipped) == (st["serbest"] == "bitti"), ctx
        assert ("serbest" in aside) == (st["serbest"] == "yarım"), ctx
        for a in plan:
            if a["op"] == "set_aside":
                assert a["kind"] == ("yarim" if a["run"] == "serbest" else "blok-iptal"), ctx
        # koşulacak hiçbir koşunun klasörü yerinde kalmaz (yoksa üzerine yazılırdı), ve kenara alma koşudan ÖNCEDİR
        for n in ran:
            if st[n] != "yok":
                kinds = [(a["op"], a["run"]) for a in plan]
                assert kinds.index(("set_aside", n)) < kinds.index(("run", n)), ctx
        block_aside = [i for i, a in enumerate(plan) if a["op"] == "set_aside" and a["kind"] == "blok-iptal"]
        first_run = min((i for i, a in enumerate(plan) if a["op"] == "run"), default=len(plan))
        assert all(i < first_run for i in block_aside), ctx


def test_plan_refuses_duplicate_names_and_inconsistent_or_missing_state():
    runs = make_runs(blocks=("s11",))
    fin, ex = state(runs)
    with pytest.raises(ValueError, match="yinelenen koşu adı"):
        rq.plan_blocks(runs + [dict(runs[0])], fin, ex)
    with pytest.raises(ValueError, match="klasörü yok"):
        rq.plan_blocks(runs, {**fin, "d-s11": True}, ex)
    with pytest.raises(KeyError):                          # eksik ad sessizce "yok" sayılmaz
        rq.plan_blocks(runs, fin, {k: v for k, v in ex.items() if k != "x-s11"})


def test_plan_refuses_a_block_whose_runs_are_not_adjacent_in_the_queue_file():
    """V11a(i): "bir seed bloğu içindeki dört koşu KESİNTİSİZ ve kuyruk dosyasındaki sırayla koşulur". Elle
    düzenlenmiş bir kuyrukta s11'in arasına başka bir koşu girmişse kuyruk hiç kesilmeden koşsa bile blok
    kesintisiz koşmamış olur — ama defterde tek blok görünür ve duvar saati ölçütüne girerdi. Kampanyadan sonra
    onarılamaz; kuyruk BAŞLAMADAN reddedilir."""
    runs = make_runs()
    by = {r["name"]: r for r in runs}
    split = [by[n] for n in ("d-s11", "m-s11", "d-s22", "h-s11", "x-s11", "m-s22", "h-s22", "x-s22")]
    with pytest.raises(ValueError, match="'s11' kuyrukta bitişik değil.*V11a-i"):
        rq.plan_blocks(split, *state(split))
    pierced = [by["d-s11"], *loose(["serbest"]), by["m-s11"], by["h-s11"], by["x-s11"]]   # araya BLOKSUZ koşu
    with pytest.raises(ValueError, match="'s11' kuyrukta bitişik değil"):
        rq.plan_blocks(pierced, *state(pierced))
    between = [*runs[:4], *loose(["serbest"]), *runs[4:]]   # bloklar ARASINDA bloksuz koşu serbest
    assert ops(rq.plan_blocks(between, *state(between)), "run") == [r["name"] for r in between]


def test_committed_decision_queue_is_accepted_and_planned_in_file_order():
    """Reddetme kuralları (yinelenen ad, bitişik olmayan blok) commit'li karar kuyruğunu REDDETMEMELİ: temiz
    diskte plan, dosyadaki 20 koşunun dosyadaki sırayla (Latin kare kaydırması, V11a-i) koşulmasıdır."""
    path = os.path.join(ROOT, "experiments", "foton-v02-prereg", "queues", "decision.json")
    if not os.path.exists(path):
        pytest.skip("commit'li karar kuyruğu bu ağaçta yok")
    with open(path, encoding="utf-8") as f:
        runs = json.load(f)["runs"]
    plan = rq.plan_blocks(runs, *state(runs))
    assert [a["op"] for a in plan] == ["run"] * len(runs) and ops(plan, "run") == [r["name"] for r in runs]
    assert all(r.get("_block") for r in runs)              # karar kuyruğunun HER koşusu blok kuralına tabi


# --- kenara alma: taşı, ASLA silme ----------------------------------------------------------------------

def test_set_aside_renames_never_deletes_and_takes_the_first_free_number(tmp_path):
    d = tmp_path / "m-s11"
    for k in (1, 2):
        d.mkdir()
        (d / "metrics.jsonl").write_text(f"deneme {k}\n", encoding="utf-8")
        dst = rq.set_aside(str(d), rq.BLOCK_SUFFIX)
        assert dst == str(tmp_path / f"m-s11.blok-iptal-{k}") and not d.exists()
    # iki denemenin İÇERİĞİ de duruyor: klasör o denemenin tek ham kaydı
    assert (tmp_path / "m-s11.blok-iptal-1" / "metrics.jsonl").read_text(encoding="utf-8") == "deneme 1\n"
    assert (tmp_path / "m-s11.blok-iptal-2" / "metrics.jsonl").read_text(encoding="utf-8") == "deneme 2\n"
    d.mkdir()
    assert rq.set_aside(str(d), rq.HALF_SUFFIX) == str(tmp_path / "m-s11.yarim-1")   # ekler ayrı sayılır
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m-s11.blok-iptal-1", "m-s11.blok-iptal-2", "m-s11.yarim-1"]


# --- execute: blok kuralı uçtan uca (sahte başlatıcıyla) --------------------------------------------------

def test_crashed_block_is_rerun_whole_and_both_attempts_are_in_the_ledger(env):
    runs = make_runs()
    q = env.queue(runs)
    launch = Launcher()
    launch.fail = {"h-s11": 1}
    assert env.execute(q, launch) == 1                     # H çöktü: kuyruk durdu
    assert launch.launched == ["d-s11", "m-s11", "h-s11"]
    assert [(r["run"], r["status"]) for r in env.rows()] == [("d-s11", "bitti"), ("m-s11", "bitti"), ("h-s11", "çöktü")]
    first = {r["run"]: r for r in env.rows()}

    launch.launched.clear()
    assert env.execute(q, launch) == 0
    assert launch.launched == [r["name"] for r in runs]    # D ve M YENİDEN koştu; X ve ikinci blok da
    assert env.dirs() == sorted([r["name"] for r in runs] + [f"{n}-s11.blok-iptal-1" for n in "dmh"])
    for n in "dmh":                                        # eski deneme DURUYOR, yenisi ayrı klasörde
        assert (env.out / f"{n}-s11.blok-iptal-1" / "train.log").read_text(encoding="utf-8") == "deneme 1\n"
        assert (env.out / f"{n}-s11" / "train.log").read_text(encoding="utf-8") == "deneme 2\n"
    assert all(rq.finished(str(env.out / r["name"])) for r in runs)

    rows = env.rows()
    cancel = [r for r in rows if r["evidence"] == "blok-iptal"]
    assert [(r["run"], r["dir"], r["status"], r["block"], r["source"]) for r in cancel] == [
        ("d-s11", "d-s11.blok-iptal-1", "iptal", "s11", "kuyruk"),
        ("m-s11", "m-s11.blok-iptal-1", "iptal", "s11", "kuyruk")]
    for r in cancel:
        assert r["reason"].startswith("V11a(iii)") and "V11a-iv" in r["reason"] and r["arm"] == r["run"][0]
        assert rl.attempt(r) == rl.attempt(first[r["run"]])   # AYNI denemenin durum değişikliği, yeni deneme değil
    # kuyruğun zaten yazdığı çöküş ikinci kez sayılmadı; iptal edilen bitmiş koşu ikinci bir `bitti` olmadı
    c = rl.counts(rows)
    assert (c["çöktü"], c["iptal"], c["bitti"], c["toplam"]) == (1, 2, 2 + 8, 3 + 2 + 8)
    assert {r["run"] for r in rows} == {r["name"] for r in runs}   # `.blok-iptal-` eki koşu ADINA sızmadı

    launch.launched.clear()
    before = (env.dirs(), len(rows))
    assert env.execute(q, launch) == 0 and launch.launched == []   # biten kampanya yeniden başlatılınca hiçbir şey olmaz
    assert (env.dirs(), len(env.rows())) == before


def test_second_interruption_of_the_same_block_takes_the_next_free_number(env):
    runs = make_runs(blocks=("s11",))
    q = env.queue(runs)
    launch = Launcher()
    for failing in ("m-s11", "x-s11"):
        launch.fail = {failing: 1}
        assert env.execute(q, launch) == 1
    assert env.execute(q, launch) == 0
    assert env.dirs() == sorted(["d-s11", "m-s11", "h-s11", "x-s11",
                                 "d-s11.blok-iptal-1", "m-s11.blok-iptal-1",
                                 "d-s11.blok-iptal-2", "m-s11.blok-iptal-2", "h-s11.blok-iptal-1", "x-s11.blok-iptal-1"])
    assert launch.launched == ["d-s11", "m-s11"] + ["d-s11", "m-s11", "h-s11", "x-s11"] * 2


def test_attempts_the_dead_queue_never_ledgered_are_written_when_the_block_is_cancelled(env):
    """Makine yeniden başladı: önceki kuyruk HİÇBİR satır yazamadı. Kenara alınan her klasör deftere girmeli —
    bitmiş olan `iptal/blok-iptal`, yarım kalan diskteki durumuyla; koşu adı ve kol/seed doğru olmalı."""
    runs = make_runs(blocks=("s11",))
    put_dir(env, runs[0], "bitti")
    put_dir(env, runs[1], "yarım")
    put_dir(env, runs[3], "yarım", provenance=False)       # kod izi bile yazılamadan ölmüş: kol/seed kuyruktan
    launch = Launcher()
    assert env.execute(env.queue(runs), launch) == 0
    assert launch.launched == ["d-s11", "m-s11", "h-s11", "x-s11"]
    old = [r for r in env.rows() if ".blok-iptal-" in r["dir"]]
    assert [(r["run"], r["dir"], r["status"], r["evidence"], r["arm"], r["seed"]) for r in old] == [
        ("d-s11", "d-s11.blok-iptal-1", "iptal", "blok-iptal", "d", 11),
        ("m-s11", "m-s11.blok-iptal-1", "iptal", "kanit-yok", "m", 11),
        ("x-s11", "x-s11.blok-iptal-1", "iptal", "kanit-yok", "x", 11)]
    assert all(r["reason"].startswith("V11a(iii)") and r["block"] == "s11" and r["source"] == "kuyruk" for r in old)
    assert "'bitti' (DONE)" in old[0]["reason"]            # diskteki gerçek durum gerekçede KORUNUR
    assert (env.out / "d-s11.blok-iptal-1" / "train.log").read_text(encoding="utf-8") == "önceki oturum\n"


def test_blockless_half_run_is_still_set_aside_as_yarim_and_finished_one_skipped(env):
    runs = loose(["a", "b"])
    put_dir(env, runs[0], "bitti")
    put_dir(env, runs[1], "yarım")
    launch = Launcher()
    assert env.execute(env.queue(runs), launch) == 0
    assert launch.launched == ["b"] and env.dirs() == ["a", "b", "b.yarim-1"]
    assert [(r["run"], r["dir"], r["status"]) for r in env.rows()] == [("b", "b.yarim-1", "iptal"), ("b", "b", "bitti")]
    assert "kuyruk kenara aldı" in env.rows()[0]["reason"]


def test_blockless_crash_the_queue_already_ledgered_is_not_counted_again_when_set_aside(env):
    """Bloksuz yol yeniden yazıldı; eski kuralı korumalı: kuyruğun ÇÖKERKEN deftere yazdığı deneme, sonraki
    başlatmada `.yarim-<n>` diye kenara alınırken İKİNCİ kez sayılmaz (V11'in ÇÖKEN / TOPLAM sayıları şişerdi)."""
    runs = loose(["a", "b"])
    launch = Launcher()
    launch.fail = {"b": 1}
    assert env.execute(env.queue(runs), launch) == 1
    assert env.execute(env.queue(runs), launch) == 0
    assert env.dirs() == ["a", "b", "b.yarim-1"] and launch.launched == ["a", "b", "b"]
    assert [(r["run"], r["dir"], r["status"]) for r in env.rows()] == [
        ("a", "a", "bitti"), ("b", "b", "çöktü"), ("b", "b", "bitti")]
    c = rl.counts(env.rows())
    assert (c["çöken"], c["toplam"]) == (1, len(launch.launched))   # satır sayısı = deneme sayısı


def test_resumed_finished_run_in_a_cancelled_block_also_gets_the_status_change_row(env):
    """İptal edilen blokta `sürdürüldü` (iki oturuma yayılmış ama bitmiş) koşu da `bitti` gibi DURUM DEĞİŞİKLİĞİ
    satırı alır: `iptal` / `blok-iptal`. Diskteki durumuyla yazılsaydı defter, V11a(iii)'ün geçersiz saydığı bir
    denemeyi "sürdürülüp bitmiş koşu" diye gösterirdi; gerçek durum gerekçede korunur."""
    runs = make_runs(blocks=("s11",))
    put_dir(env, runs[0], "sürdürüldü")
    put_dir(env, runs[1], "bitti")
    assert rl.scan_run_dir(str(env.out / "d-s11"), running_grace_sec=0.0)["status"] == "sürdürüldü"   # kurulum doğru mu
    assert env.execute(env.queue(runs), Launcher()) == 0
    old = [r for r in env.rows() if ".blok-iptal-" in r["dir"]]
    assert [(r["run"], r["status"], r["evidence"], r["resumed"]) for r in old] == [
        ("d-s11", "iptal", "blok-iptal", True), ("m-s11", "iptal", "blok-iptal", False)]
    assert "'sürdürüldü' (DONE)" in old[0]["reason"] and "'bitti' (DONE)" in old[1]["reason"]
    assert rl.counts(env.rows())["sürdürüldü"] == 0


def test_every_ledger_row_carries_the_block_identity_from_the_queue_file(env):
    """V11a(ii): duvar saati YALNIZCA aynı blok içinde karşılaştırılır ve blok kimliği diskte yazılı DEĞİL —
    tek kaynağı kuyruk dosyasındaki `_block`. Kanca onu deftere taşımazsa biten her koşu "blok bilinmiyor" olur
    ve defterin duvar saatine uygun satır sayısı sıfıra düşer. Eşleme koşu ADIYLA yapılır, sırayla değil."""
    runs = make_runs() + loose(["serbest"])
    assert env.execute(env.queue(runs), Launcher()) == 0
    rows = env.rows()
    assert {r["run"]: r["block"] for r in rows} == {r["name"]: r.get("_block") for r in runs}
    blocked = [r["name"] for r in runs if r.get("_block")]
    c = rl.counts(rows)
    assert (c["blok_bilinmeyen"], c["duvar_saati_uygun"]) == (len(runs) - len(blocked), len(blocked))


def test_a_directory_that_cannot_be_set_aside_stops_the_queue_before_anything_runs(env, monkeypatch):
    """Windows'ta klasördeki bir dosya açıksa (log'a bakan editör, TensorBoard) yeniden adlandırma reddedilir.
    Kuyruk o bloğu "kısmen kenara alınmış" hâlde koşmaya DEVAM ETMEMELİ: eski denemenin klasörü yerinde dururken
    blok baştan koşulamaz. Klasör silinmez, üzerine yazılmaz; kuyruk kendi çıkış koduyla durur."""
    runs = make_runs(blocks=("s11",))
    put_dir(env, runs[0], "bitti")

    def locked(run_dir, kind):
        raise PermissionError(13, "dosya başka bir süreçte açık", run_dir)
    monkeypatch.setattr(rq, "set_aside", locked)
    launch = Launcher()
    assert env.execute(env.queue(runs), launch) == rq.EXIT_SET_ASIDE_FAILED
    assert launch.launched == [] and env.dirs() == ["d-s11"] and "kenara alınamadı" in env.log()
    assert (env.out / "d-s11" / "train.log").read_text(encoding="utf-8") == "önceki oturum\n"


def test_a_ledger_that_cannot_be_written_warns_but_never_stops_the_queue(env, tmp_path):
    """Defter yazılamazsa kuyruk DURMAZ (saatler süren bloğu bir kayıt satırı yüzünden kaybetmek daha pahalı),
    ama susmaz da: her kaçan satır queue.log'a uyarı olarak düşer ve `--reconstruct` ile geri kurulabilir."""
    blocker = tmp_path / "dosya"
    blocker.write_text("defterin üst klasörü olması gereken yerde bir DOSYA var", encoding="utf-8")
    env.ledger = str(blocker / "run_ledger.jsonl")
    runs = make_runs(blocks=("s11",))
    put_dir(env, runs[0], "bitti")                         # blok iptali kancası da aynı korumadan geçmeli
    launch = Launcher()
    assert env.execute(env.queue(runs), launch) == 0
    assert launch.launched == [r["name"] for r in runs]
    assert env.log().count("[uyarı] koşu defterine yazılamadı") == 1 + 4   # 1 kenara alma + 4 koşu


def test_a_queue_with_duplicate_names_or_a_non_boolean_strict_never_starts(env):
    runs = make_runs(blocks=("s11",))
    launch = Launcher()
    assert env.execute(env.queue(runs + [dict(runs[0])]), launch) == rq.EXIT_BAD_QUEUE
    assert env.execute(env.queue(runs, _strict="true"), launch) == rq.EXIT_BAD_QUEUE   # "true" metni sıkı DEĞİL sayılmasın
    split = [runs[0], *make_runs(blocks=("s22",)), *runs[1:]]                           # s11'in arasına s22 girmiş (V11a-i)
    assert env.execute(env.queue(split), launch) == rq.EXIT_BAD_QUEUE and "bitişik değil" in env.log()
    assert launch.launched == [] and env.dirs() == [] and env.rows() == []


def test_queue_exit_codes_are_nonzero_distinct_and_not_the_training_process_codes():
    """Kuyruğu arka planda koşturan sarmalayıcı yalnızca ÇIKIŞ KODUNU görür. Ret kodu 0 olursa reddedilen sıkı
    kuyruk "tamamlandı" diye okunur; 1/2 olursa eğitim sürecinin hatasıyla, birbirine eşit olursa öbür ret
    sebebiyle karışır. Değerler elle yazılmaz: modüldeki BÜTÜN `EXIT_*` sabitleri özellikleriyle denetlenir."""
    codes = {k: v for k, v in vars(rq).items() if k.startswith("EXIT_")}
    assert set(codes) >= {"EXIT_STRICT_DIRTY", "EXIT_STRICT_SRC_CHANGED", "EXIT_BAD_QUEUE", "EXIT_SET_ASIDE_FAILED"}
    assert len(set(codes.values())) == len(codes), codes
    for k, v in codes.items():
        # 3..255: 0 = başarı, 1 = eğitim hatası, 2 = argparse; 255 üstü POSIX kabuğunda (Git Bash) mod 256 sarar
        assert type(v) is int and 2 < v < 256 and v not in rl.CTRL_C_CODES, (k, v)


def test_main_hands_the_queue_exit_code_to_the_shell_and_uses_the_fixed_ledger(tmp_path, monkeypatch):
    """`execute` kodu DÖNDÜRÜR; onu sürecin çıkış koduna çeviren `main`'dir. Yutulursa reddedilen ya da yarıda
    duran kuyruk kabuğa 0 verir. `main` ayrıca defterin SABİT yolunu (`ledger_path=None`) ve gerçek
    başlatıcı/git'i kullanmalı — sahte olanlar yalnızca testlerin `execute`'a verdiği şeylerdir."""
    q = {"out_dir": str(tmp_path / "ckpt"), "runs": loose(["a"]), "_strict": True}
    path = tmp_path / "kuyruk.json"
    path.write_text(json.dumps(q), encoding="utf-8")
    calls = []

    def fake_execute(queue, label, **kw):
        calls.append((queue, label, kw))
        return results[len(calls) - 1]
    monkeypatch.setattr(rq, "execute", fake_execute)
    monkeypatch.setattr(rq, "keep_awake", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run_queue.py", str(path)])
    results = [rq.EXIT_STRICT_DIRTY, 1, 0]
    for want in results[:2]:
        with pytest.raises(SystemExit) as stop:
            rq.main()
        assert stop.value.code == want
    rq.main()                                              # 0: kuyruk tamamlandı, SystemExit yok
    assert calls == [(q, str(path), {"ledger_path": None})] * len(results)


# --- kod izi: oturum kimliği, code_dirty ------------------------------------------------------------------

def test_session_identity_is_one_per_queue_process_and_differs_between_processes(env, monkeypatch):
    """V11a: "duvar saati büyüklüğü verilirken oturum kimliği ZORUNLUDUR". Kimlik sürecin başlangıç anıdır ve
    sürecin başlattığı HER koşuda aynıdır; koşu başına yeniden üretilirse oturumu değil koşuyu adlandırır."""
    stamps = (f"2026-09-19T10:00:{i:02d}" for i in range(60))
    monkeypatch.setattr(rq, "session_id", lambda: next(stamps))   # her çağrıda FARKLI: kaç kez çağrıldığı görünsün
    runs = make_runs(blocks=("s11",))
    assert env.execute(env.queue(runs), Launcher()) == 0
    assert [env.prov(r["name"])["session"] for r in runs] == ["2026-09-19T10:00:00"] * 4
    assert "oturum 2026-09-19T10:00:00" in env.log().splitlines()[0]
    assert all("oturum 2026-09-19T10:00:00" in r["reason"] for r in env.rows())
    more = runs + make_runs(blocks=("s22",))
    assert env.execute(env.queue(more), Launcher()) == 0          # ikinci süreç: YENİ kimlik, yine tek
    assert {env.prov(r["name"])["session"] for r in more[4:]} == {"2026-09-19T10:00:01"}
    assert env.prov("d-s11")["session"] == "2026-09-19T10:00:00"  # eski koşunun kimliği değişmedi


def test_real_session_id_is_a_local_iso_timestamp():
    import re
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", rq.session_id())


CAMPAIGN_OUTPUT = (" M experiments/foton-v02-prereg/run_ledger.jsonl\n"
                   "?? checkpoints/foton-v02/\n"
                   "?? experiments/foton-v02/decision.json\n"
                   " M README.md")


@pytest.mark.parametrize("line", [
    " M src/model.py", "M  src/model.py", "MM scripts/run_queue.py", "?? tests/test_yeni.py", "?? src/",
    " D docs/flops.md", " M PREREGISTRATION_V02.md", "A  PREREGISTRATION.md", " M Tests/test_stats.py",
    'R  experiments/eski.py -> src/yeni.py', 'R  src/eski.py -> experiments/yeni.py', ' M "scripts/bo\\305\\237luk li.py"',
    "M src/config.py",                 # tüm çıktının strip()'i İLK satırın baştaki boşluğunu düşürür
])
def test_code_dirty_sees_every_code_path(line):
    assert rq.code_dirty_paths(CAMPAIGN_OUTPUT.splitlines() + [line]), line
    assert rq.code_dirty_paths([line]), line


def test_code_dirty_ignores_the_campaigns_own_outputs():
    """Kuyruk her koşudan sonra experiments/ (koşu defteri) ve checkpoints/ altını KENDİSİ değiştiriyor: ham
    `dirty` bu yüzden kampanya boyunca hep True ve hiçbir şey söylemiyordu."""
    assert rq.code_dirty_paths(CAMPAIGN_OUTPUT.splitlines()) == []
    assert rq.code_dirty_paths(CAMPAIGN_OUTPUT.strip().splitlines()) == []
    assert rq.code_dirty_paths([" M srcx/model.py", "?? experiments/src/x.py", " M data/PREREGISTRATION.md", ""]) == []
    assert rq.code_dirty_paths([" M src/model.py", "?? checkpoints/x/"]) == ["src/model.py"]


def test_provenance_keeps_the_raw_dirty_flag_next_to_code_dirty(env):
    prov = rq.provenance(str(env.src_file.parent.parent), sys.executable, "OTURUM", cmd=fake_git(CAMPAIGN_OUTPUT))
    assert prov["dirty"] is True and prov["code_dirty"] is False and prov["session"] == "OTURUM"
    assert prov["git_status"] == CAMPAIGN_OUTPUT.strip().splitlines() and prov["git_head"] == "c" * 40
    assert bool(rq.code_dirty_paths(prov["git_status"])) == prov["code_dirty"]   # izden YENİDEN hesaplanabilir
    both = rq.provenance(str(env.src_file.parent.parent), sys.executable, "OTURUM",
                         cmd=fake_git(CAMPAIGN_OUTPUT + "\n M scripts/run_ledger.py"))
    assert both["dirty"] is True and both["code_dirty"] is True
    clean = rq.provenance(str(env.src_file.parent.parent), sys.executable, "OTURUM", cmd=fake_git(""))
    assert clean["dirty"] is False and clean["code_dirty"] is False and clean["git_status"] == []
    # git ÇALIŞMADIYSA temiz sayılmaz: bilinmiyor
    unknown = rq.provenance(str(env.src_file.parent.parent), sys.executable, "OTURUM", cmd=fake_git(None))
    assert unknown["code_dirty"] is None and unknown["git_status"] == [] and unknown["git_head"] == ""
    assert len(prov["src_sha256"]) == 64 and set(prov) >= {"python", "gpu", "host"}


def test_cmd_tells_a_failed_command_from_an_empty_answer():
    ok = rq._cmd([sys.executable, "-c", "print('  x  ')"], ROOT)
    assert ok == "x" and rq._cmd([sys.executable, "-c", "pass"], ROOT) == ""
    assert rq._cmd([sys.executable, "-c", "import sys; print('x'); sys.exit(3)"], ROOT) is None
    assert rq._cmd(["boyle-bir-komut-yok-foton"], ROOT) is None


def test_every_run_of_the_queue_gets_both_flags_in_provenance_and_the_queue_log(env):
    runs = make_runs(blocks=("s11",))
    assert env.execute(env.queue(runs), Launcher(), status=CAMPAIGN_OUTPUT) == 0
    for r in runs:
        p = env.prov(r["name"])
        assert (p["dirty"], p["code_dirty"]) == (True, False) and p["args"] == r["args"]
    head = env.log().splitlines()[0]
    assert "+4 değişiklik" in head and "code_dirty=False" in head and "sıkı değil" in head
    assert all("code_dirty=False" in r["reason"] and r["dirty"] is True for r in env.rows())


# --- sıkı katman ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("status, why", [(" M src/model.py", "src/model.py"),
                                         (CAMPAIGN_OUTPUT + "\n?? tests/test_yeni.py", "tests/test_yeni.py"),
                                         (None, "git durumu okunamadı")])
def test_strict_queue_refuses_to_start_with_uncommitted_code_and_touches_nothing(env, status, why):
    """Karar kampanyası commit'lenmemiş kodla koşarsa `git_head` ölçümü üreten kodu TANIMLAMAZ ve bu, kampanyadan
    sonra onarılamaz. Ret, blokları kenara almadan ÖNCE gelmeli: reddedilen başlatma diski değiştirmemeli."""
    runs = make_runs(blocks=("s11",))
    put_dir(env, runs[0], "bitti")                         # kısmen koşulmuş blok: başlasaydı kenara alınırdı
    launch = Launcher()
    assert env.execute(env.queue(runs, _strict=True), launch, status=status) == rq.EXIT_STRICT_DIRTY
    assert launch.launched == [] and env.dirs() == ["d-s11"] and env.rows() == []
    assert "SIKI kuyruk başlatılmadı" in env.log() and why in env.log()
    # aynı durumda sıkı OLMAYAN kuyruk eskisi gibi koşar (yalnızca kaydeder)
    assert env.execute(env.queue(runs), launch, status=status) == 0
    assert launch.launched == ["d-s11", "m-s11", "h-s11", "x-s11"]


def test_strict_queue_starts_when_only_campaign_outputs_are_dirty(env):
    runs = make_runs(blocks=("s11",))
    launch = Launcher()
    assert env.execute(env.queue(runs, _strict=True), launch, status=CAMPAIGN_OUTPUT) == 0
    assert launch.launched == [r["name"] for r in runs] and "| SIKI |" in env.log().splitlines()[0]


def test_strict_queue_rechecks_code_dirty_before_every_run_not_only_at_start(env):
    """Kural: karar koşularında `code_dirty = False` zorunlu, İSTİSNASIZ. V7 taramasında kuyruk temiz başladı, ama
    kuyruk SÜRERKEN scripts/ altına commit'siz bir teşhis betiği yazıldı: src/ özeti değişmedi, yine de sonraki iki
    koşunun provenance'ı `code_dirty = True` oldu. Sıkı kuyruk bunu başlangıçta DEĞİL her koşudan önce denetlemeli."""
    runs = make_runs(blocks=("s11",))
    launch = Launcher()
    state = {"status": CAMPAIGN_OUTPUT}
    git = lambda args, cwd: fake_git(state["status"])(args, cwd)          # noqa: E731 — durum koşular arasında değişir

    def write_a_script(name):
        if name == "d-s11":
            state["status"] = CAMPAIGN_OUTPUT + "\n?? scripts/yeni_teshis.py"
    launch.on_launch = write_a_script
    run = lambda q: rq.execute(q, "test-kuyruğu.json", ledger_path=env.ledger, launch=launch, cmd=git,   # noqa: E731
                               echo=lambda *a, **k: None)
    assert run(env.queue(runs, _strict=True)) == rq.EXIT_STRICT_DIRTY
    assert launch.launched == ["d-s11"] and env.dirs() == ["d-s11"]       # m-s11'in klasörü bile açılmadı
    assert "SIKI kuyruk durduruldu" in env.log() and "scripts/yeni_teshis.py" in env.log() and "m-s11 başlatılmadı" in env.log()
    assert env.prov("d-s11")["code_dirty"] is False

    launch.launched.clear()                                # sıkı OLMAYAN kuyruk eskisi gibi koşar ve kirliliği KAYDEDER
    launch.on_launch = None
    loose = make_runs(blocks=("s22",))
    assert run(env.queue(loose)) == 0 and launch.launched == [r["name"] for r in loose]
    assert all(env.prov(r["name"])["code_dirty"] is True for r in loose)


def test_strict_queue_stops_before_the_next_run_when_src_changes_and_loose_queue_only_warns(env):
    """Kuyruk sürerken src/ değişirse sonraki kollar FARKLI kodla koşar. Sıkı kuyruk sıradaki koşuyu
    BAŞLATMADAN durur (klasörü bile açmaz); yarıda kalan blok sonraki başlatmada V11a(iii) ile baştan koşulur."""
    runs = make_runs(blocks=("s11",))
    launch = Launcher()

    def edit_src(name):
        if name == "d-s11":
            env.src_file.write_text("x = 2\n", encoding="utf-8")
    launch.on_launch = edit_src
    assert env.execute(env.queue(runs, _strict=True), launch) == rq.EXIT_STRICT_SRC_CHANGED
    assert launch.launched == ["d-s11"] and env.dirs() == ["d-s11"]
    assert "SIKI kuyruk durduruldu" in env.log() and "m-s11 başlatılmadı" in env.log()

    launch.on_launch = None                                # yeni süreç: kod artık sabit, blok BAŞTAN
    launch.launched.clear()
    assert env.execute(env.queue(runs, _strict=True), launch) == 0
    assert launch.launched == ["d-s11", "m-s11", "h-s11", "x-s11"] and "d-s11.blok-iptal-1" in env.dirs()
    assert len({env.prov(r["name"])["src_sha256"] for r in runs}) == 1   # bloğun dört kolu AYNI kodla koştu

    loose_env_runs = make_runs(blocks=("s22",))            # sıkı olmayan kuyruk: eski davranış, yalnızca uyarı
    launch.launched.clear()
    launch.on_launch = lambda name: env.src_file.write_text("x = 3\n", encoding="utf-8") if name == "d-s22" else None
    assert env.execute(env.queue(loose_env_runs), launch) == 0
    assert launch.launched == ["d-s22", "m-s22", "h-s22", "x-s22"]
    assert "[uyarı] kuyruk başladığından beri src/ değişti" in env.log()
    # Uyarı yetmez: her koşunun kod izi o koşunun FİİLEN koştuğu kodu yazmalı (kuyruk başındaki özeti değil),
    # yoksa farklı kodla koşmuş kollar provenance.json'da aynı kodla koşmuş görünür.
    now = rq.code_hash(str(env.src_file.parent.parent))
    assert [env.prov(r["name"])["src_sha256"] == now for r in loose_env_runs] == [False, True, True, True]
