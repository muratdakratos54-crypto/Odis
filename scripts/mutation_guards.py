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
G_REACH = [PY, "scripts/rule_reachability.py", "--sims", "4000", "--out", "_reach_mutant.log"]
G_QSELF = [PY, "scripts/make_campaign_queue.py", "--arms", "foton-d,foton-m,foton-h,foton-x",
           "--seeds", "11,22,33,44,55", "--out", "_queue_mutant.json"]
G_DOC = "DOC"          # gerçek depodaki check_prereg_numbers.py, --prereg <bozulmuş kopya>
G_PINS = "PINS"        # gerçek depodaki verify_pins.py, --manifest <bozulmuş kopya>

S = "src/stats.py"

# (kimlik, guard, dosya, eski, yeni, NE bozuldu) — `nth`: kaçıncı geçiş (1 tabanlı); "all": hepsi
MUTANTS: list[dict] = [
    # ------------------------------------------------------------------ karar kuralı: src/stats.py
    dict(id="S01", guard=G_STATS, file=S, what="paired(): seed KİMLİĞİ yerine liste KONUMU",
         old="    if seeds_a is not None and seeds_b is not None:\n        pos = {s: i",
         new="    if False:\n        pos = {s: i"),
    dict(id="S02", guard=G_STATS, file=S, what="t kritik değeri n'den bağımsız sabit (v3 hatası)",
         old="    if df in T975:\n        return T975[df]", new="    if df in T975:\n        return T975[2]"),
    dict(id="S03", guard=G_STATS2, file=S, what="n_star: B14 sürdürülmüş-kesişim kontrolü kapalı",
         old="            if any(y > target for _, y in pts[j + 2:]):", new="            if False:"),
    dict(id="S04", guard=G_STATS2, file=S, what="savings(): baseline KONUMLA bulunuyor",
         old="        fb = f_base[pos[s]]", new="        fb = f_base[len(s_i)]"),
    dict(id="S05", guard=G_STATS2, file=S, what="savings(): seed kümesi uyuşmazlığı sessiz",
         old="    if sorted(seeds_f) != sorted(seeds_b):", new="    if False:"),
    dict(id="S06", guard=G_STATS2, file=S, what="decide(): n ∈ {5,10} şartı kapalı (2 seed'le başarı)",
         old="    if strict and n not in REQUIRED_N:", new="    if False:"),
    dict(id="S07", guard=G_STATS2, file=S, what="decide(): 'güçlü başarı' iki baseline'ı da İSTEMİYOR",
         old="and wmin > 0 and w_complete:", new="and wmin > 0:"),
    dict(id="S08", guard=G_STATS2, file=S, what="decide(): çelişki etiketi GA yerine yalnız mean > 0",
         old="            if lo > CI_EPS:\n                notes.append(\"FLOP kazancı var (GA tamamen > 0)",
         new="            if mean > 0:\n                notes.append(\"FLOP kazancı var (GA tamamen > 0)"),
    dict(id="S09", guard=G_STATS2, file=S, what="decide(): n=10'da terminal satır yok",
         old="        if n >= max(REQUIRED_N):", new="        if False:"),
    dict(id="S10", guard=G_STATS, file=S, what="decide(): ≥2 sansür -> 'başarısız' kuralı kapalı",
         old="    if n_cens >= 2:", new="    if n_cens >= 99:"),
    dict(id="S11", guard=G_STATS, file=S, what="kesinlik ölçütü (tripwire) tamamen kapalı",
         old="    return hi >= SUCCESS", new="    return False"),
    dict(id="S12", guard=G_STATS2, file=S, what="abandon(): sansürsüz GA yerine TAM örnek GA'sı",
         old="        if unc is not None:\n            return unc[1]",
         new="        if unc is not None:\n            return r[\"ci95\"][1]"),
    dict(id="S13", guard=G_STATS2, file=S, what="abandon(): B10 yolunda sansürsüz GA şartı kapalı (+%30 terk)",
         old="        if hi_u >= SUCCESS:", new="        if False:"),
    dict(id="S14", guard=G_STATS2, file=S, what="abandon(): 'yükseltme hakkı kullanılmadı' ön-koşulu kapalı",
         old="        if escalation_used:\n            why.append(", new="        if False:\n            why.append("),
    dict(id="S15", guard=G_STATS2, file=S, what="abandon(): oracle şartı bağlamıyor",
         old="    oracle_ok = oracle_gain_over_null < oracle_threshold", new="    oracle_ok = True"),
    dict(id="S16", guard=G_STATS2, file=S, what="contribution(): Holm düzeyi (.025) yerine .05 GA",
         old="    h = t_crit_alpha(n - 1, alpha) * sd / math.sqrt(n)",
         new="    h = t_crit(n - 1) * sd / math.sqrt(n)"),
    dict(id="S17", guard=G_STATS2, file=S, what="contribution(): 'küçük' durumu hiç oluşmuyor (dört durum çöküyor)",
         old="    real, small = lo > CI_EPS, hi < mde", new="    real, small = lo > CI_EPS, False"),
    dict(id="S18", guard=G_STATS2, file=S, what="decide_primary(): Holm satır adını DÜŞÜRMÜYOR",
         old="    if row1 in (\"güçlü başarı\", \"başarılı\") and \"birincil-1\" not in passed:",
         new="    if False:"),
    dict(id="S19", guard=G_STATS2, file=S, what="multiplicity(): pinli aile büyüklüğü yok sayılıyor",
         old="    m = family_size if family_size is not None else m_eff", new="    m = m_eff"),
    dict(id="S20", guard=G_STATS2, file=S, what="_wbar(): alakasız anahtar kabul ediliyor",
         old="        if extra:", new="        if False:"),
    dict(id="S21", guard=G_STATS, file=S, what="decide(): 1 sansürlü seed varken başarı BLOKLANMIYOR",
         old="        if row in (\"güçlü başarı\", \"başarılı\"):\n            notes.append(\"EK 1 B8 gereği",
         new="        if False:\n            notes.append(\"EK 1 B8 gereği"),
    dict(id="S22", guard=G_STATS, file=S, what="başarı eşiği 5 -> 4 puan",
         old="SUCCESS = 5.0           # puan", new="SUCCESS = 4.0           # puan"),
    dict(id="S23", guard=G_STATS, file=S, what="'güçlü başarı' eşiği 10 -> 9 puan",
         old="STRONG = 10.0           # puan", new="STRONG = 9.0            # puan"),
    dict(id="S24", guard=G_STATS, file=S, what="'nötr' bandı ±2 -> ±1 puan",
         old="NEUTRAL = 2.0           # puan", new="NEUTRAL = 1.0           # puan"),
    dict(id="S25", guard=G_STATS, file=S, what="birincil-2'nin MDE çıtası 1.88 -> 1.00 puan",
         old="MDE_PRIMARY2 = 1.88      # puan", new="MDE_PRIMARY2 = 1.00      # puan"),
    dict(id="S26", guard=G_STATS, file=S, what="Holm aile büyüklüğü 2 -> 1 (düzeltme fiilen kapalı)",
         old="PRIMARY_FAMILY = 2       # birincil-1", new="PRIMARY_FAMILY = 1       # birincil-1"),
    dict(id="S27", guard=G_STATS, file=S, what="decide(): 'başarılı' dalı GA alt > 0 şartını aramıyor",
         old="    elif mean >= SUCCESS and lo > CI_EPS:", new="    elif mean >= SUCCESS:",
         equivalent=("EŞDEĞER MUTANT — hiçbir test öldüremez, çünkü davranış değişmiyor. Kanıt: kesinlik "
                     "ölçütü bu daldan ÖNCE koşar ve karar ancak `(lo > 0 ve s̄ >= 5)` ya da `hi < 5` ise "
                     "belirlidir. `s̄ >= 5` iken `hi >= 5` olduğundan ikinci yol kapalıdır; yani bu dala "
                     "`s̄ >= 5` ile gelen her örnekte `lo > 0` ZATEN sağlanmıştır. Sansürlü durumda ölçüt "
                     "sansürsüz alt örneğe bakar ama o zaman tam örnek ortalaması ikame yüzünden ~−25'tir "
                     "ve bu dala hiç girilmez. Koşul yedekli savunmadır; silinmesi satırı değiştirmez.")),
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
    dict(id="R01", guard=G_REACH, file=S, what="(S13 mutantı) tarama +%7/+%30 senaryosunun terk edildiğini YAKALAMALI",
         old="        if hi_u >= SUCCESS:", new="        if False:"),
    dict(id="R02", guard=G_REACH, file=S, what="(S12 mutantı) tarama sansürde ulaşılamazlığı YAKALAMALI",
         old="        if unc is not None:\n            return unc[1]",
         new="        if unc is not None:\n            return r[\"ci95\"][1]"),
    dict(id="R03", guard=G_REACH, file=S, what="(S15 mutantı) tarama oracle'ın bağlamadığını YAKALAMALI",
         old="    oracle_ok = oracle_gain_over_null < oracle_threshold", new="    oracle_ok = True"),
    dict(id="R04", guard=G_REACH, file="scripts/rule_reachability.py", what="dört durum birbirini dışlamıyor (toplam ≠ 1)",
         old="            cells = [(real & ~small).mean(), (real & small).mean(),",
         new="            cells = [(real).mean(), (real & small).mean(),"),
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
    dict(id="N16", guard=G_DOC, file=PREREG, what="belgedeki bir SHA-256 çivisi bozuk",
         old="b0f6570240ac87a5", new="c0f6570240ac87a5"),
    # ------------------------------------------------------------------ çivi manifesti
    dict(id="P01", guard=G_PINS, file="pins", what="manifestteki bir dosyanın SHA-256'sı bozuk", pins="flip"),
    dict(id="P02", guard=G_PINS, file="pins", what="manifestten bir kanıt dosyası silinmiş", pins="drop"),
]


def run(cmd: list[str], cwd: str) -> tuple[int, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
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
    tmp = tempfile.mkdtemp(prefix="foton_mut_")
    rows, survivors = [], []
    pins_tmp = os.path.join(ROOT, "experiments", "foton-v02-prereg", "_mutant_pins.json")
    try:
        for d in ("src", "tests", "scripts"):
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
    out = ["# Mutasyon sınavı kaydı (ön-kayıt V13)", "",
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
    with open(os.path.join(ROOT, LOG), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n{len(rows)} mutant: {len(rows) - len(survivors) - n_eq} yakalandı, {n_eq} kanıtlı eşdeğer, "
          f"{len(survivors)} hayatta kaldı -> {LOG}")
    if survivors:
        for m in survivors:
            print(f"  HAYATTA: {m['id']} — {m['what']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
