"""Tag'li ön-kayıt belgesinin gövdesi DONMUŞTUR: yalnızca sona ("Sapmalar") eklenebilir, bir de V10'un tek
yer tutucu satırı doldurulabilir. `scripts/check_prereg_frozen.py`'nin bunu gerçekten yakaladığı burada sınanır."""
from __future__ import annotations

import importlib.util
import os
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("check_prereg_frozen", os.path.join(ROOT, "scripts", "check_prereg_frozen.py"))
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

TAGGED = ("# belge\n\nbaşarı eşiği +%5\n\n    __DOLDURULACAK: FOTON config SHA-256__   (preset'ler henüz yazılmadı)\n\n"
          "## Sapmalar\n\n**S1 — eski sapma.**\n")
HASH = "a" * 64
FILLED_LINE = f"    FOTON preset SHA-256 (`experiments/foton-v02/preset_hashes.json`): {HASH}"


def test_unchanged_and_appended_documents_pass():
    assert cf.check(TAGGED, TAGGED) == []
    assert cf.check(TAGGED, TAGGED + "\n**S3 — yeni sapma.** O-X1 eski tasarımı anlatıyor.\n") == []


def test_filling_the_single_v10_placeholder_is_the_only_allowed_body_edit():
    filled = TAGGED.replace("    __DOLDURULACAK: FOTON config SHA-256__   (preset'ler henüz yazılmadı)", FILLED_LINE)
    assert cf.check(TAGGED, filled) == [] and cf.check(TAGGED, filled + "\n**S3 — ek.**\n") == []
    # doldurulmuş satır TEKTİR: sona ("Sapmalar") ikinci bir hash satırı eklenemez — hangisinin bağlayıcı olduğu
    # belirsizleşirdi (gövde farkı bunu yakalamaz: ek, izinli bölgededir; yakalayan satır SAYIMIDIR)
    other = FILLED_LINE.replace(HASH, "b" * 64)
    assert cf.check(TAGGED, filled + "\n**S3 — hash güncellendi.**\n" + other + "\n")
    assert cf.check(TAGGED, TAGGED + "\n" + FILLED_LINE + "\n")
    # yer tutucu satırı SİLİNEMEZ ve serbest metinle doldurulamaz
    assert cf.check(TAGGED, TAGGED.replace("    __DOLDURULACAK: FOTON config SHA-256__   (preset'ler henüz yazılmadı)\n", ""))
    assert cf.check(TAGGED, TAGGED.replace("    __DOLDURULACAK: FOTON config SHA-256__", "    hash sonra yazılacak"))


@pytest.mark.parametrize("edit", [
    lambda s: s.replace("+%5", "+%4"),                       # bir eşiğin sessizce değişmesi
    lambda s: s.replace("**S1 — eski sapma.**", "**S1 — eski sapma (yumuşatıldı).**"),   # eski bir sapmanın yeniden yazılması
    lambda s: s.replace("# belge\n", "# belge\n\nyeni bir kural\n"),                      # gövdeye ekleme
    lambda s: s[: len(s) // 2],                              # kırpma
])
def test_any_other_change_to_the_tagged_body_is_caught(edit):
    bad = cf.check(TAGGED, edit(TAGGED))
    assert bad and ("DEĞİŞMİŞ" in bad[0] or "yer tutucu" in bad[0])


def test_the_real_document_is_the_tagged_body_plus_appended_deviations():
    if subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{cf.TAG}"], cwd=ROOT,
                      capture_output=True).returncode != 0:
        pytest.skip("tag yok")
    p = subprocess.run([os.sys.executable, os.path.join(ROOT, "scripts", "check_prereg_frozen.py")],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stdout + p.stderr
