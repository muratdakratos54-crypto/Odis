"""Çivi doğrulayıcının regresyon testleri: bir bayt değişirse `--verify` ÇÖKMEK ZORUNDA.

Neden bu testler var: çivilenen dosyalar `.gitignore` yüzünden git'te değil, bu yüzden ön-kaydın
commit'lenen tek kanıtı `experiments/foton-v02-prereg/pins.json` manifestinin kendisi. Manifest bir
dosyanın değiştiğini yakalamazsa çivi süs olur — "korpus donduruldu" ve "L* şu eğriden türedi"
cümlelerinin arkasında hiçbir mekanizma kalmaz. `--verify`'ın sessizce geçmesi, hiç doğrulama
yapmamaktan daha kötüdür: doğrulanmış görünür.

Testler beş şeyi kilitler:
  1. manifest BEKLENEN dosya kümesinin tamamını kapsıyor (9 korpus + 10 koşu × 3 kanıt = 39),
  2. tek bir bayt değişince doğrulama başarısız oluyor ve hangi dosya olduğunu SÖYLÜYOR,
  3. dosya silinince "eksik" diye başarısız oluyor (sessizce atlamıyor),
  4. iki koruma çalışıyor: yeniden çivileme `--force` olmadan reddediliyor, çivilenmemiş bir
     baseline koşusu ve ön-kayıt bloğundan sapma başarısızlık sayılıyor.
  5. KAPSAM: `--verify` yalnızca "hash'ler tuttu mu" değil "kaç çivi var" sorusunu da soruyor.
     Denetimde ölçülen boşluk buydu: 38 satırlık bir manifest 38 PASS ile exit 0 alıyordu, yani
     kanıt zinciri sessizce küçültülebiliyordu. İki yoldan da (üretim anında dosya yoktu /
     manifestten satır elle silindi) artık başarısız oluyor.

Sahte ağaç `tmp_path`'te kurulur: gerçek korpus 612 MB, testin ona dokunması gerekmiyor — sınanan
şey hash'in değeri değil, sapmanın YAKALANMASI.
"""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location("verify_pins", os.path.join(ROOT, "scripts", "verify_pins.py"))
vp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vp)

MAN = "pins.json"                     # sahte ağaçta manifest köke yazılır


def _write(path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# Koşu dizini modülün kendi sabitinden gelir: kanıt zinciri hangi kümeye işaret ediyorsa
# (`checkpoints/foton-v02r`, bkz. Sapmalar/S2) testler de orayı kursun. Sabit yazmak, varsayılan
# değiştiğinde 13 testin birden kırılmasına yol açtı.
RUNS = os.path.join(*vp.RUNS_DIR.split("/"))


@pytest.fixture
def tree(tmp_path):
    """Gerçek düzenin sahtesi: 9 korpus dosyası + 10 koşu × 3 kanıt dosyası.

    `metrics.jsonl` gerçek biçimde yazılıyor (adım 3500 val satırı dahil), çünkü kanıt notunun
    o eğrinin TAŞIDIĞI sayıyı da yazması sınanan davranışlardan biri.
    """
    for name in vp.CORPUS:
        _write(tmp_path / "data" / "foton" / name, f"icerik:{name}".encode())
    for arm in vp.ARMS:
        for seed in vp.SEEDS:
            d = tmp_path / RUNS / f"baseline-{arm}-s{seed}"
            lines = [json.dumps({"type": "val", "step": s, "tokens": s * 32768,
                                 "val_loss": 3.0 + seed / 1000 + s / 1_000_000})
                     for s in (0, vp.STEP80, 4375)]
            _write(d / "metrics.jsonl", ("\n".join(lines) + "\n").encode())
            _write(d / "config.json", json.dumps({"train": {"seed": seed}}).encode())
            _write(d / "provenance.json", json.dumps({"src_sha256": "0" * 64}).encode())
    return tmp_path


def _gen(tree, force: bool = False, prereg: str = "") -> int:
    return vp.do_generate(root=str(tree), manifest=MAN, prereg=prereg, force=force)


def _ver(tree, prereg: str = "") -> int:
    return vp.do_verify(root=str(tree), manifest=MAN, prereg=prereg)


def _manifest(tree) -> dict:
    with open(tree / MAN, encoding="utf-8") as f:
        return json.load(f)


def test_manifest_covers_every_expected_file(tree):
    """39 çivinin hepsi manifestte, her biri yol + bayt + sha256 + kanıt notu taşıyor."""
    assert _gen(tree) == 0
    man = _manifest(tree)
    assert man["counts"] == {"pinned": 39, "missing": 0, "expected": 39}
    assert man["missing"] == []
    paths = [e["path"] for e in man["files"]]
    assert len(paths) == len(set(paths)) == 39
    # Önek diskten çıkarılıyor (koşu kümesi `baseline-` ya da `r-` olabilir; bkz. vp.run_prefix),
    # bu yüzden beklenen küme de bu ağacın önekiyle kurulmalı. Sabit bir önek varsaymak, yanlış
    # kümeye bakıldığında testin sessizce geçmesine yol açardı.
    pref = vp.run_prefix(str(tree), vp.RUNS_DIR)
    assert pref == "baseline", f"bu ağacın öneki `baseline` olmalıydı, çıkarılan: {pref}"
    assert paths == [p for p, _ in vp.expected_pins(prefix=pref)],         "beklenen küme ile manifest sırası ayrıştı"
    for e in man["files"]:
        assert "\\" not in e["path"], "yol ayırıcısı platforma bağlı olmamalı"
        assert e["bytes"] > 0 and len(e["sha256"]) == 64
        assert e["note"].strip(), f"{e['path']}: kanıt notu boş — çivi neyin kanıtı olduğunu söylemeli"
    # tokenizer.vocab denetimde çivisizdi; kapsamda olduğu burada kilitleniyor
    assert "data/foton/tokenizer.vocab" in paths
    # koşu başına kanıt üçlüsü: eğri + config + köken birbirinden koparılamaz
    for arm in vp.ARMS:
        for seed in vp.SEEDS:
            for fname in vp.RUN_FILES:
                assert f"{vp.RUNS_DIR}/baseline-{arm}-s{seed}/{fname}" in paths


def test_metrics_note_carries_the_measured_value(tree):
    """`metrics.jsonl` çivisi adım 3500 val_loss'unu da yazar: L* yeniden türetilirken beklenen sayı."""
    assert _gen(tree) == 0
    note = next(e["note"] for e in _manifest(tree)["files"]
                if e["path"].endswith("baseline-m-s11/metrics.jsonl"))
    assert f"adım {vp.STEP80} val_loss = 3.014500" in note, note


def test_verify_passes_on_untouched_tree(tree, capsys):
    """Dokunulmamış ağaçta 39 PASS, exit 0."""
    assert _gen(tree) == 0
    assert _ver(tree) == 0
    out = capsys.readouterr().out
    assert out.count("PASS ") == 39
    assert "39 PASS, 0 FAIL, 0 EKSİK" in out
    # Başarı satırı fazla iddia etmemeli: bütünlük != koşudan önce üretilmişlik
    assert "GÖSTERMEZ" in out


def test_verify_fails_and_names_the_mutated_file(tree, capsys):
    """Tek bayt değişince FAIL, exit 1, ve satır o dosyanın adını taşır."""
    assert _gen(tree) == 0
    victim = tree / "data" / "foton" / "nl_val.bin"
    b = bytearray(victim.read_bytes())
    b[0] ^= 0x01                                  # boyut aynı: yakalayan şey hash, boyut değil
    victim.write_bytes(bytes(b))
    assert _ver(tree) == 1
    out = capsys.readouterr().out
    assert "FAIL  data/foton/nl_val.bin" in out
    assert "hash farklı" in out
    assert out.count("PASS ") == 38, "yalnızca bozulan dosya FAIL olmalı"
    assert "38 PASS, 1 FAIL" in out


def test_verify_fails_on_truncation_as_size_mismatch(tree, capsys):
    """Kırpılan dosya hem hash hem boyut olarak sapar; rapor boyutu ayrıca söyler."""
    assert _gen(tree) == 0
    (tree / RUNS / "baseline-d-s44" / "metrics.jsonl").write_bytes(b"{}")
    assert _ver(tree) == 1
    out = capsys.readouterr().out
    assert f"FAIL  {vp.RUNS_DIR}/baseline-d-s44/metrics.jsonl" in out
    assert "boyut farklı" in out


def test_verify_fails_when_a_file_is_missing(tree, capsys):
    """Silinen dosya EKSİK olarak raporlanır — sessizce atlanmaz (manifest 39 dosya iddia ediyor)."""
    assert _gen(tree) == 0
    os.remove(tree / RUNS / "baseline-m-s33" / "metrics.jsonl")
    assert _ver(tree) == 1
    out = capsys.readouterr().out
    assert f"EKSİK {vp.RUNS_DIR}/baseline-m-s33/metrics.jsonl" in out
    assert "dosya yok" in out
    assert "38 PASS, 0 FAIL, 1 EKSİK" in out


def test_verify_explains_the_clean_clone_case(tree, capsys):
    """Hiçbir dosya yoksa (temiz klon) başarısız olur AMA nedenini açıklar: iddianın gerçek sınırı."""
    assert _gen(tree) == 0
    for e in _manifest(tree)["files"]:
        os.remove(vp.abspath(str(tree), e["path"]))
    assert _ver(tree) == 1
    out = capsys.readouterr().out
    assert "temiz klonda beklenen sonuç budur" in out
    assert ".gitignore" in out


def test_generate_refuses_to_repin_without_force(tree, capsys):
    """Yeniden çivileme sessizce olamaz: değişen hash `--force` olmadan yazılmaz."""
    assert _gen(tree) == 0
    before = (tree / MAN).read_bytes()
    (tree / "data" / "foton" / "meta.json").write_bytes(b"baska icerik")
    assert _gen(tree) == 1
    assert (tree / MAN).read_bytes() == before, "reddedilen üretim manifeste dokunmamalı"
    assert "REDDEDİLDİ" in capsys.readouterr().out
    assert _gen(tree, force=True) == 0                # kasıtlı yeniden çivileme geçer
    assert _ver(tree) == 0


def test_generate_records_missing_files_instead_of_shrinking(tree, capsys):
    """Eksik dosya manifeste 'missing' olarak girer; kapsam sessizce 38'e düşmez."""
    os.remove(tree / "data" / "foton" / "math_val.bin")
    assert _gen(tree) == 0
    man = _manifest(tree)
    assert man["counts"] == {"pinned": 38, "missing": 1, "expected": 39}
    assert man["missing"] == ["data/foton/math_val.bin"]
    assert "EKSİK dosyalar" in capsys.readouterr().out


def test_verify_flags_an_unpinned_baseline_run(tree, capsys):
    """Diskte çivilenmemiş 11. koşu varsa doğrulama başarısız: kanıt zinciri sessizce büyümesin."""
    assert _gen(tree) == 0
    extra = tree / RUNS / "baseline-m-s66"
    for fname in vp.RUN_FILES:
        _write(extra / fname, b"{}")
    assert _ver(tree) == 1
    out = capsys.readouterr().out
    assert "ÇİVİSİZ" in out
    assert "baseline-m-s66/metrics.jsonl" in out
    assert out.count("PASS ") == 39, "çivilenmiş 39 dosya hâlâ tutuyor olmalı"


def test_verify_cross_checks_the_preregistration_block(tree, capsys):
    """Manifest, ön-kayıttaki çivi bloğundan sapamaz: aynıysa geçer, farklıysa FAIL."""
    assert _gen(tree) == 0
    real = next(e["sha256"] for e in _manifest(tree)["files"] if e["path"] == "data/foton/meta.json")
    prereg = tree / "PREREG.md"
    prereg.write_text(f"```\n{real}  data/foton/meta.json\n```\n", encoding="utf-8", newline="\n")
    assert _ver(tree, prereg="PREREG.md") == 0
    assert "1 çivi, 1 tanesi manifestle aynı" in capsys.readouterr().out

    prereg.write_text(f"```\n{'a' * 64}  data/foton/meta.json\n```\n", encoding="utf-8", newline="\n")
    assert _ver(tree, prereg="PREREG.md") == 1
    out = capsys.readouterr().out
    assert "ön-kayıt çivisi manifestle uyuşmuyor: data/foton/meta.json" in out
    assert "1 ön-kayıt sapması" in out


def test_verify_fails_when_the_manifest_was_generated_incomplete(tree, capsys):
    """Üretim anında dosya yoksa manifest 38 çiviyle çıkar; `--verify` bunu GEÇMEMELİ.

    Ölçülen boşluk: `--generate` eksik dosyayı `missing` diye kaydedip 0 dönüyor (doğru davranış,
    disk o an hazır olmayabilir) ama `--verify` kalan 38'i hash'leyip "bütün çiviler tuttu" diyordu.
    O hâlde ön-kaydın "kanıt bu 39 dosyadır" cümlesinin arkasında 38 dosya kalırdı ve kimse görmezdi.
    """
    os.remove(tree / "data" / "foton" / "math_val.bin")
    assert _gen(tree) == 0                         # üretim eksiği kaydeder, reddetmez
    assert _manifest(tree)["counts"]["pinned"] == 38
    assert _ver(tree) == 1, "38 çivilik manifest doğrulamadan geçmemeli"
    out = capsys.readouterr().out
    assert "KAPSAM DIŞI" in out
    assert "! data/foton/math_val.bin" in out
    assert "1 KAPSAM DIŞI" in out
    assert out.count("PASS ") == 38                # kalan 38 gerçekten tutuyor: hata hash'te değil kapsamda


def test_verify_fails_when_an_entry_is_edited_out_of_the_manifest(tree, capsys):
    """Manifestten satır silip dosyayı da silmek doğrulamayı atlatmamalı.

    Kapsam denetimi manifestin kendi `counts`/`missing` alanlarına GÜVENMEMELİ: manifesti düzenleyen
    biri onları da düzenler. Beklenen küme `expected_pins()` ile kodda ilan edildiği için, L*'in
    türetildiği seed 11 eğrisini kanıt zincirinden çıkarmak yakalanıyor.
    """
    assert _gen(tree) == 0
    victim = f"{vp.RUNS_DIR}/baseline-m-s11/metrics.jsonl"
    man = _manifest(tree)
    man["files"] = [e for e in man["files"] if e["path"] != victim]
    man["counts"] = {"pinned": 38, "missing": 0, "expected": 38}     # sayaçlar da "tutarlı" yapıldı
    with open(tree / MAN, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, ensure_ascii=False)
    os.remove(vp.abspath(str(tree), victim))
    assert _ver(tree) == 1
    out = capsys.readouterr().out
    assert f"! {victim}" in out


def test_prereg_pin_regex_does_not_span_lines(tree, capsys):
    """Tek başına duran 64 hex'lik satır uydurma çivi üretmemeli.

    `\\s+` satır sonunu da yerdi: yalnız bir hash satırı (provenance `src_sha256` de 64 hex) bir
    sonraki satırın ilk kelimesiyle eşleşir, o uydurma yol manifestte bulunmaz ve `--verify`
    sebepsiz FAIL verir. Ön-kayıt hâlâ düzenlendiği için bu tuzak her an patlayabilir.
    """
    assert not vp.PREREG_PIN_RE.findall("f" * 64 + "\nbaska_bir_kelime\n")
    assert _gen(tree) == 0
    real = next(e["sha256"] for e in _manifest(tree)["files"] if e["path"] == "data/foton/meta.json")
    prereg = tree / "PREREG.md"
    prereg.write_text(f"src_sha256:\n{'f' * 64}\ndata/foton/meta.json bir metin\n\n"
                      f"```\n{real}  data/foton/meta.json\n```\n", encoding="utf-8", newline="\n")
    assert _ver(tree, prereg="PREREG.md") == 0, "yalnız hash satırı uydurma çivi üretti"
    assert "1 çivi, 1 tanesi manifestle aynı" in capsys.readouterr().out


def test_verify_fails_when_the_manifest_itself_is_gone(tree, capsys):
    """Manifest yoksa doğrulama yapılamaz ve bu exit 0 ile geçemez (commit'lenen kanıt o dosya)."""
    assert _ver(tree) == 1
    assert "MANİFEST OKUNAMADI" in capsys.readouterr().out
