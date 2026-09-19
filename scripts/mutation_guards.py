"""MUTASYON SINAVI: her guard için koruduğu şeyi KASTEN boz, yakaladığını göster (ön-kayıt V13).

    .venv\\Scripts\\python scripts/mutation_guards.py

**Yakalamadığı kanıtlanmamış guard, guard değildir.** Bu betiğin var olma sebebi, aynı sınıftan
tekrarlayan olaylar:

  * `%33.8` guard'ı iki ondalıklı deseni (`%33.79`) arıyordu; belgedeki gerçek hatayı kaçırdı.
  * `reachability.log` kendi defect'ini basıp geçti: beklenti düzyazıydı, assertion değildi.
  * Güç tablosu kontrolü tam olarak GÜNCELLENMİŞ satırlara bakıyordu, hatanın kaldığı satırlara değil.
  * Kuyruğun Latin kare assertion'ı yalnızca `blok % kol == 0` iken çalışıyordu — kampanyanın
    gerçek biçiminde (5 × 4) hiç.
  * `check_prereg_numbers`'a eklenen "her `L*` geçişini denetle" regex'i yanlış kaçışla yazılmıştı
    (`L\\?\\*` literal soru işareti arıyordu) ve SIFIR eşleşmeyle sessizce geçiyordu.

Hepsi gözden geçirmeyle değil, bozup denemeyle ortaya çıkar.

NASIL ÇALIŞIR. Depo GEÇİCİ bir dizine kopyalanır (`src/`, `tests/`, `scripts/`, ön-kayıt ve küçük
artifact'lar); her mutant o kopyaya uygulanır, ilgili guard koşulur, kopya eski hâline döner.
Gerçek depo ve dondurulmuş artifact'lar DEĞİŞMEZ. İki istisna, ikisi de salt-okunur: belge
guard'ı gerçek depodaki betikle ama `--prereg <bozulmuş kopya>` ile koşar (SHA çivileri gerçek
`data/` dosyalarına bakmak zorunda), ve çivi guard'ı bozulmuş bir MANİFEST KOPYASIYLA koşar.

Her guard önce BOZULMAMIŞ hâlde koşulur ve GEÇMEK zorundadır — yoksa "mutantı yakaladı" demek
anlamsızdır (zaten kırık olan bir guard her şeyi "yakalar").

Çıkış kodu: hayatta kalan mutant varsa 1. Kayıt: `experiments/foton-v02-prereg/mutation_log.md`.

İKİ SINAV (`--suite`). `prereg` (varsayılan): tag'li ön-kaydın guard'ları; kaydı `foton-v02-prereg` tag'iyle
DONMUŞTUR, yeniden koşmak o dosyayı ezer — tag'den sonra yalnızca `--only` ile (kayda yazmadan) koşulur.
`06`: tag'den SONRA, FOTON uygulamasıyla yazılan guard'lar (`scripts/mutation_catalog_06.py`); kaydı
`experiments/foton-v02/mutation_log_06.md`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
LOG = "experiments/foton-v02-prereg/mutation_log.md"
PREREG = "PREREGISTRATION_V02.md"

PYTEST = lambda f: [PY, "-m", "pytest", f, "-q", "-x", "-p", "no:cacheprovider"]  # noqa: E731
# Karar kuralı mutantları İKİ test dosyasının BİRLEŞİMİYLE koşulur. İlk hâl mutantı tek dosyaya
# elle eşliyordu ve S10 yanlış dosyaya bağlandığı için "hayatta kaldı" görünmüştü — guard'ın değil
# sınavın hatası. Birleşim, "bu mutantı HİÇBİR test yakalamıyor" sorusunu doğru soruyor.
G_STATS = [PY, "-m", "pytest", "tests/test_stats.py", "tests/test_stats_v02.py", "-q", "-x",
           "-p", "no:cacheprovider"]
G_STATS2 = G_STATS
G_FROZEN = PYTEST("tests/test_foton_frozen.py")
G_DATA = PYTEST("tests/test_data_multi.py")
G_ATTN = PYTEST("tests/test_attn_patterns.py")
G_LEDGER = PYTEST("tests/test_run_ledger.py")
G_QUEUE = PYTEST("tests/test_campaign_queue.py")
G_OWNER = PYTEST("tests/test_owner_decision.py")
G_REACH = [PY, "scripts/rule_reachability.py", "--sims", "4000", "--out", "_reach_mutant.log"]
G_QSELF = [PY, "scripts/make_campaign_queue.py", "--arms", "foton-d,foton-m,foton-h,foton-x",
           "--seeds", "11,22,33,44,55", "--out", "_queue_mutant.json"]
G_DOC = "DOC"          # gerçek depodaki check_prereg_numbers.py, --prereg <bozulmuş kopya>
G_PINS = "PINS"        # gerçek depodaki verify_pins.py, --manifest <bozulmuş kopya>

S = "src/stats.py"

# (kimlik, guard, dosya, eski, yeni, NE bozuldu) — `nth`: kaçıncı geçiş (1 tabanlı); "all": hepsi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutation_catalog_stats import REACH_MUTANTS, STATS_MUTANTS  # noqa: E402

MUTANTS: list[dict] = [
    # karar kuralı: katalog `scripts/mutation_catalog_stats.py`, guard = iki test dosyasının birleşimi
    *[{**m, "guard": G_STATS, "file": S} for m in STATS_MUTANTS],
    # ------------------------------------------------------------------ ölçüt ve veri akışı
    dict(id="T01", guard=G_FROZEN, file="src/train.py", what="evaluate(): maske prefix yerine SUFFIX'i tutuyor",
         old="        return full, per_pos[:, k:].mean().item()",
         new="        return full, per_pos[:, :k].mean().item()"),
    dict(id="T02", guard=G_FROZEN, file="src/train.py", what="evaluate(): maske hiç uygulanmıyor",
         old="        return full, per_pos[:, k:].mean().item()",
         new="        return full, per_pos.mean().item()"),
    dict(id="C01", guard=G_FROZEN, file="src/config.py", what="foton-d: loss_prefix_skip 64 -> 0", nth=1,
         old="loss_prefix_skip=64),", new="loss_prefix_skip=0),"),
    dict(id="C02", guard=G_FROZEN, file="src/config.py", what="foton-m: loss_prefix_skip 64 -> 0", nth=2,
         old="loss_prefix_skip=64),", new="loss_prefix_skip=0),"),
    dict(id="C03", guard=G_FROZEN, file="src/config.py", what="foton-d: bütçe 4375 -> 4000 adım", nth=1,
         old="max_steps=4375,", new="max_steps=4000,"),
    dict(id="D01", guard=G_FROZEN, file="src/data_multi.py", what="REF_BATCH 16 -> 32 (val pencereleri sessizce kayar)",
         old="    REF_BATCH = 16", new="    REF_BATCH = 32"),
    dict(id="D02", guard=G_DATA, file="src/data_multi.py", what="akışın alan döngüsü ters sırada (referansı yeniden üretmiyor)",
         old="        for di, name in enumerate(self.names):          # get_batch ile aynı döngü sırası",
         new="        for di, name in reversed(list(enumerate(self.names))):"),
    dict(id="A01", guard=G_ATTN, file="src/attn_patterns.py", what="modül A penceresi 150 -> 151 (eşitlik bozulur)",
         old="A_LOCAL_W = 150", new="A_LOCAL_W = 151"),
    # ------------------------------------------------------------------ koşu defteri ve kuyruk
    dict(id="L01", guard=G_LEDGER, file="scripts/run_ledger.py", what="deneme kimliğine klasör adı giriyor (çift sayım)",
         old="    return (rec.get(\"run\"), rec.get(\"started\"))",
         new="    return (rec.get(\"run\"), rec.get(\"dir\"), rec.get(\"started\"))"),
    dict(id="L02", guard=G_LEDGER, file="scripts/run_ledger.py", what="kuyruk satırı otoriter değil",
         old="        if attempt(rec) in from_queue or key(rec) in have:", new="        if key(rec) in have:"),
    dict(id="Q01", guard=G_QUEUE, file="scripts/make_campaign_queue.py", what="Latin kare rotasyonu yok (sabit sıra) — pytest",
         old="        order = arms[bi % len(arms):] + arms[:bi % len(arms)]", new="        order = list(arms)"),
    dict(id="Q02", guard=G_QSELF, file="scripts/make_campaign_queue.py", what="Latin kare rotasyonu yok — betiğin KENDİ assertion'ı",
         old="        order = arms[bi % len(arms):] + arms[:bi % len(arms)]", new="        order = list(arms)"),
    # ------------------------------------------------------------------ ulaşılabilirlik sınavının assertion'ları
    # aynı bozulmalar, bu kez ulaşılabilirlik taramasının KENDİ assertion'larına karşı
    *[{**m, "guard": G_REACH, "file": S} for m in REACH_MUTANTS],
    # ------------------------------------------------------------------ sahibin karar defteri (S1'den ÖNCE loglama)
    dict(id="O01", guard=G_OWNER, file="scripts/owner_decision.py", what="gerekçesiz karar kabul ediliyor",
         old='    if len((reason or "").strip()) < MIN_REASON:', new="    if False:"),
    dict(id="O02", guard=G_OWNER, file="scripts/owner_decision.py", what="karar SONRADAN değiştirilebiliyor (ikinci satır kabul)",
         old="    if read(log):", new="    if False:"),
    dict(id="O03", guard=G_OWNER, file="scripts/owner_decision.py", what="'S1'e çıkma' kararı S1 kuyruğuna izin veriyor",
         old="    if recs[0][\"choice\"] != \"S1'e çık\":\n        raise PermissionError(", new="    if False:\n        raise PermissionError("),
    dict(id="O04", guard=G_OWNER, file="scripts/owner_decision.py", what="S1 koşusunun karardan ÖNCE başladığı yakalanmıyor",
         old='        if not r.get("started") or r["started"] <= d["ts_local"]:', new="        if False:"),
    dict(id="O05", guard=G_OWNER, file="scripts/owner_decision.py", what="karar loglanmamışken escalation_used sessizce False",
         old='        raise ValueError(f"ölçek hakkı kararı kayıtlı değil ({log}): vazgeçme değerlendirmesinden önce loglanmalı")',
         new="        return False"),
    dict(id="O06", guard=G_OWNER, file="scripts/owner_decision.py", what="karar satırı dayanak dosyasının SHA-256'sını taşımıyor",
         old='"basis_sha256": _sha256(basis),', new='"basis_sha256": None,'),
    dict(id="O07", guard=G_OWNER, file="scripts/make_campaign_queue.py", what="S1 kuyruğu sahibin kararı OLMADAN üretiliyor (kapı kapalı)",
         old='    if scale == "S1":', new="    if False:"),
    dict(id="O08", guard=G_OWNER, file="scripts/make_campaign_queue.py", what="ölçek kol adından çıkarılmıyor (S1 preset'leri 'S0 kuyruğu' diye geçiyor)",
         old='    return "S1" if all(s1) and arms else "S0"', new='    return "S0"'),
    dict(id="O09", guard=G_OWNER, file="scripts/make_campaign_queue.py", what="karışık ölçekli kuyruk kabul ediliyor",
         old="    if any(s1) and not all(s1):", new="    if False:"),
    # ------------------------------------------------------------------ belge ↔ artifact tutarlılığı
    dict(id="N01", guard=G_DOC, file=PREREG, what="eski L_aux (2.9646) belgeye geri girdi", append="\nL_aux = 2.9646\n"),
    dict(id="N02", guard=G_DOC, file=PREREG, what="eski tok/s düşüşü BİR ondalıkla (%33.8) geri girdi",
         append="\nM, D'den token başına %33.8 yavaştır.\n"),
    dict(id="N03", guard=G_DOC, file=PREREG, what="L_aux'un kopyalarından yalnızca BİRİ bozuk", nth=2,
         old="L_aux = 2.8929", new="L_aux = 2.8930"),
    dict(id="N04", guard=G_DOC, file=PREREG, what="L* hedefi 2.8358 -> 2.8359", nth=1,
         old="L* = 2.8358", new="L* = 2.8359"),
    dict(id="N05", guard=G_DOC, file=PREREG, what="seed başına val@3500 değeri bozuk",
         old="2.834774", new="2.834775", all=True),
    dict(id="N06", guard=G_DOC, file=PREREG, what="M'in FLOP/token'ı bozuk", old="267.755.520", new="267.755.521", all=True),
    dict(id="N07", guard=G_DOC, file=PREREG, what="F_base,i ortalaması bozuk", old="3.073053e16", new="3.073054e16", all=True),
    dict(id="N08", guard=G_DOC, file=PREREG, what="duvar saati hücresi artifact'tan sapıyor (25.93 -> 25.94)",
         old="25.93 dk", new="25.94 dk", all=True),
    dict(id="N09", guard=G_DOC, file=PREREG, what="eşleştirilmiş fark bozuk (+4.06 -> +4.07)", old="+4.06 dk", new="+4.07 dk", all=True),
    dict(id="N10", guard=G_DOC, file=PREREG, what="tok/s düşüşü bozuk (%27.89 -> %27.90)", old="%27.89", new="%27.90", all=True),
    dict(id="N11", guard=G_DOC, file=PREREG, what="güç tablosu `0%` satırında eski kuralın hücresi",
         old="| 0% | %0.0 | %0.0 | %0.0 | %0.0 | %0.7 | %0.1 |", new="| 0% | %0.0 | %0.0 | %0.0 | %0.0 | %0.6 | %0.1 |"),
    dict(id="N12", guard=G_DOC, file=PREREG, what="güç tablosu `+2%` satırında eski kuralın hücresi",
         old="| %4.9 | %2.8 |", new="| %4.9 | %1.9 |"),
    dict(id="N13", guard=G_DOC, file=PREREG, what="güç tablosu `+5%` satırında eski kuralın hücresi",
         old="| %32.6 | %49.5 |", new="| %27.2 | %49.5 |"),
    dict(id="N14", guard=G_DOC, file=PREREG, what="güç tablosu `+7%` satırında eski kuralın hücresi",
         old="| %61.7 | **%89.5** |", new="| %46.4 | **%89.5** |"),
    dict(id="N15", guard=G_DOC, file=PREREG, what="güç tablosu `+10%` satırında eski kuralın hücresi",
         old="| %90.6 | %99.9 |", new="| %58.4 | %99.9 |"),
    dict(id="N17", guard=G_DOC, file=PREREG, what="birincil-2 güç tablosunun +1.88 satırı bozuk (3. denetimde guard'sızdı)",
         old="| +1.88 puan | %56.7 | %97.6 | %100.0 |", new="| +1.88 puan | %52.9 | %96.3 | %100.0 |"),
    dict(id="N18", guard=G_DOC, file=PREREG, what="birincil-2 'ölçemedik' payı α=.05'in eski sayısına dönmüş",
         old='**"ölçemedik" %42.8**, "var"', new='**"ölçemedik" %24.6**, "var"'),
    # --- başarısız-seed senaryolarının metindeki sayıları (4. denetim turunda eklenen kontrol 5d) ---
    dict(id="N19", guard=G_DOC, file=PREREG, what="1x sınırı örneğinin ub95'i bozuk (+3.38 -> +3.83)",
         old="`ub95 = +3.38`", new="`ub95 = +3.83`"),
    dict(id="N20", guard=G_DOC, file=PREREG, what="aritmetik tavan örneğinin tavanlı değeri bozuk",
         old="tavanlı biçim **−49.9**", new="tavanlı biçim **−94.9**"),
    dict(id="N21", guard=G_DOC, file=PREREG, what="yavaş seed örneğinin ortalaması bozuk (−11.12 -> −11.21)",
         old="`s̄ = −11.12`", new="`s̄ = −11.21`"),
    dict(id="N22", guard=G_DOC, file=PREREG, what="2x sınırındaki ub95 farkı bozuk (0.21 -> 0.12)",
         old="`ub95` farkı 0.21 puan", new="`ub95` farkı 0.12 puan"),
    dict(id="N23", guard=G_DOC, file=PREREG, what="ub95 monotonluk dizisinde bir değer bozuk",
         old="k = 0..4 için +0.3, −29.7, −59.5", new="k = 0..4 için +0.3, −29.7, −95.5"),
    dict(id="N24", guard=G_DOC, file=PREREG, what="`4 × +%30 + 1 sansür` örneğinin ub95'i bozuk",
         old="`ub95 = −5.85`", new="`ub95 = −5.58`"),
    dict(id="N16", guard=G_DOC, file=PREREG, what="belgedeki bir SHA-256 çivisi bozuk",
         old="b0f6570240ac87a5", new="c0f6570240ac87a5"),
    # ------------------------------------------------------------------ çivi manifesti
    dict(id="P01", guard=G_PINS, file="pins", what="manifestteki bir dosyanın SHA-256'sı bozuk", pins="flip"),
    dict(id="P02", guard=G_PINS, file="pins", what="manifestten bir kanıt dosyası silinmiş", pins="drop"),
]


EXTRA_ENV: dict[str, str] = {}
EXTRA_DIRS: list[str] = []
TITLE = "# Mutasyon sınavı kaydı (ön-kayıt V13)"


def run(cmd: list[str], cwd: str) -> tuple[int, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", **EXTRA_ENV)
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=600)
    tail = (p.stdout + p.stderr).strip().split("\n")
    return p.returncode, " | ".join(x.strip() for x in tail[-3:] if x.strip())[-220:]


def mutate(text: str, m: dict) -> str:
    if "append" in m:
        return text + m["append"]
    n = text.count(m["old"])
    if m.get("all"):
        assert n >= 1, f"{m['id']}: hedef metin bulunamadı"
        return text.replace(m["old"], m["new"])
    nth = m.get("nth", 1)
    if "nth" not in m:
        assert n == 1, f"{m['id']}: hedef metin {n} kez geçiyor (tam 1 olmalı): {m['old'][:60]!r}"
    assert n >= nth, f"{m['id']}: {nth}. geçiş yok (toplam {n})"
    i = -1
    for _ in range(nth):
        i = text.index(m["old"], i + 1)
    return text[:i] + m["new"] + text[i + len(m["old"]):]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="virgüllü kimlik önekleri (ör. E,B,O0): HIZLI yineleme içindir, "
                                                "kayıt dosyasına YAZMAZ — kayıt yalnızca tam koşumdan çıkar")
    ap.add_argument("--suite", choices=("prereg", "06"), default="prereg",
                    help="prereg: tag'li ön-kaydın sınavı (kaydı DONMUŞ); 06: FOTON uygulamasının guard'ları")
    args = ap.parse_args()
    only = tuple(x.strip() for x in args.only.split(",") if x.strip())
    global MUTANTS, LOG, TITLE
    if args.suite == "06":
        from mutation_catalog_06 import MUTANTS_06
        MUTANTS = [{**m, "guard": PYTEST(m["guard"])} for m in MUTANTS_06]
        LOG = "experiments/foton-v02/mutation_log_06.md"
        TITLE = "# Mutasyon sınavı kaydı — 06 (FOTON uygulaması) guard'ları (ön-kayıt V13)"
        # Geçici kopyada `.git` yok; iki guard tag'li sürümle KARŞILAŞTIRIR (D/M bit-özdeşliği, donmuş gövde) ve
        # tag'e erişemezse `skip` eder — atlanan test hiçbir mutantı yakalamaz. Salt-okunur git erişimi verilir.
        EXTRA_ENV["GIT_DIR"] = os.path.join(ROOT, ".git")
        EXTRA_DIRS.append("docs")                 # tests/test_flops.py docs/flops.md tablolarını koddan yeniden üretir
    elif not only:
        sys.exit("`--suite prereg` kaydı foton-v02-prereg tag'iyle DONMUŞTUR; tam koşum onu ezer. "
                 "Yeniden sınamak için `--only <önek>` kullan (kayda yazmaz).")
    if only:
        MUTANTS = [m for m in MUTANTS if m["id"].startswith(only)]
    tmp = tempfile.mkdtemp(prefix="foton_mut_")
    rows, survivors = [], []
    pins_tmp = os.path.join(ROOT, "experiments", "foton-v02-prereg", "_mutant_pins.json")
    try:
        for d in ("src", "tests", "scripts", *EXTRA_DIRS):
            shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d),
                            ignore=shutil.ignore_patterns("__pycache__"))
        for f in (PREREG, "PREREGISTRATION.md"):
            shutil.copy(os.path.join(ROOT, f), os.path.join(tmp, f))

        def guard_cmd(m: dict, mutated: bool) -> tuple[list[str], str]:
            g = m["guard"]
            if g == G_DOC:
                doc = os.path.join(tmp, PREREG) if mutated else os.path.join(ROOT, PREREG)
                return [PY, "scripts/check_prereg_numbers.py", "--prereg", doc], ROOT
            if g == G_PINS:
                man = ("experiments/foton-v02-prereg/_mutant_pins.json" if mutated
                       else "experiments/foton-v02-prereg/pins.json")
                return [PY, "scripts/verify_pins.py", "--verify", "--manifest", man], ROOT
            return g, tmp

        # 1) her guard BOZULMAMIŞ hâlde geçmeli
        print("## 1. Guard'lar bozulmamış hâlde geçiyor mu?")
        base: dict[str, int] = {}
        for m in MUTANTS:
            cmd, cwd = guard_cmd(m, mutated=False)
            k = " ".join(cmd[1:])
            if k not in base:
                base[k], tail = run(cmd, cwd)
                print(f"  {'GEÇTİ' if base[k] == 0 else 'KIRIK'}  {k[:100]}")
                assert base[k] == 0, f"guard bozulmamış hâlde KIRIK, mutasyon sınavı anlamsız: {k}\n{tail}"

        # 2) mutantlar
        print(f"\n## 2. {len(MUTANTS)} mutant")
        for m in MUTANTS:
            t0 = time.time()
            if m["guard"] == G_PINS:
                man = json.load(open(os.path.join(ROOT, "experiments/foton-v02-prereg/pins.json"), encoding="utf-8"))
                if m["pins"] == "flip":
                    e = man["files"][-1]
                    e["sha256"] = ("0" if e["sha256"][0] != "0" else "1") + e["sha256"][1:]
                else:
                    man["files"].pop(len(man["files"]) // 2)
                json.dump(man, open(pins_tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                cmd, cwd = guard_cmd(m, mutated=True)
                rc, tail = run(cmd, cwd)
            else:
                path = os.path.join(tmp, m["file"])
                orig = open(path, encoding="utf-8").read()
                open(path, "w", encoding="utf-8", newline="\n").write(mutate(orig, m))
                try:
                    cmd, cwd = guard_cmd(m, mutated=True)
                    rc, tail = run(cmd, cwd)
                finally:
                    open(path, "w", encoding="utf-8", newline="\n").write(orig)
            caught = rc != 0
            eq = "equivalent" in m
            tag = "YAKALADI" if caught else ("EŞDEĞER (kanıtlı)" if eq else "*** HAYATTA KALDI ***")
            print(f"  {m['id']}  {tag:22s} {m['what'][:78]}  ({time.time()-t0:.0f}s)")
            rows.append((m, caught, tail))
            if not caught and not eq:
                survivors.append(m)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.exists(pins_tmp):
            os.remove(pins_tmp)

    # 3) kayıt
    n_eq = sum(1 for m, c, _ in rows if not c and "equivalent" in m)
    out = [TITLE, "",
           "Her satır: bir guard'ın koruduğu şey KASTEN bozuldu ve guard koşuldu. \"Bozdum, yakaladı\".",
           "Bütün guard'lar önce bozulmamış hâlde koşuldu ve geçti. Gerçek depo değiştirilmedi.", "",
           f"**{len(rows)} mutant: {len(rows) - len(survivors) - n_eq} yakalandı, {n_eq} kanıtlı eşdeğer, "
           f"{len(survivors)} hayatta kaldı.**", "",
           "Eşdeğer mutant = kodu değiştiren ama DAVRANIŞI değiştirmeyen mutant; hiçbir test öldüremez.",
           "Sessizce atlanmaz: kanıtıyla birlikte aşağıda durur ve her koşumda yeniden denenir (bir gün",
           "yakalanırsa kanıt yanlış demektir).", "",
           "| # | bozulan şey | guard | sonuç | guard'ın son satırı |", "|---|---|---|---|---|"]
    for m, caught, tail in rows:
        g = m["guard"] if isinstance(m["guard"], str) else " ".join(m["guard"][1:4])
        g = {"DOC": "check_prereg_numbers.py", "PINS": "verify_pins.py --verify"}.get(g, g)
        res = "**yakaladı**" if caught else ("eşdeğer (kanıtlı)" if "equivalent" in m else "**HAYATTA KALDI**")
        out.append(f"| {m['id']} | {m['what']} | `{g}` | {res} | {tail.replace('|', '/')[:150]} |")
    for m, caught, _ in rows:
        if "equivalent" in m and not caught:
            out += ["", f"**{m['id']} neden eşdeğer:** {m['equivalent']}"]
    if not only:                              # kısmi koşum (`--only`) kaydı EZMEZ
        os.makedirs(os.path.dirname(os.path.join(ROOT, LOG)), exist_ok=True)
        with open(os.path.join(ROOT, LOG), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(out) + "\n")
    print(f"\n{len(rows)} mutant: {len(rows) - len(survivors) - n_eq} yakalandı, {n_eq} kanıtlı eşdeğer, "
          f"{len(survivors)} hayatta kaldı -> {LOG if not only else '(--only: kayda YAZILMADI)'}")
    if survivors:
        for m in survivors:
            print(f"  HAYATTA: {m['id']} — {m['what']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
