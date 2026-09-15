"""scripts/backup_bundle.py testleri (geçici git depoları; git PATH'te olmalı).

Bir yedekleme betiğinin asıl işi kötü durumu yakalamak: eksik ref'li ya da bozuk bir bundle
doğrulamadan geçmemeli, başarısız bir yedekleme mevcut sağlam yedeği ezmemeli, depoda kaybolan
geçmiş de yedeklerden (dest, .prev) sessizce silinmemeli.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location("backup_bundle", os.path.join(ROOT, "scripts", "backup_bundle.py"))
bb = importlib.util.module_from_spec(_spec)
sys.modules["backup_bundle"] = bb  # dataclass, modülü sys.modules'ta arar
_spec.loader.exec_module(bb)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git bulunamadı")


@pytest.fixture(autouse=True)
def _isolated_git_config(tmp_path, monkeypatch):
    """Kullanıcının global git ayarları (hook'lar, clone.defaultRemoteName, imzalama ...) testleri etkilemesin."""
    empty = tmp_path / "bos.gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))


def run(cwd, *args: str, stdin: bytes | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    r = subprocess.run(["git", *args], cwd=cwd, env=env, input=stdin, capture_output=True)
    assert r.returncode == 0, f"git {' '.join(args)} (kod {r.returncode}): {r.stderr.decode(errors='replace')}"
    return r.stdout.decode()


def commit(repo: str, name: str) -> None:
    with open(os.path.join(repo, name), "w") as f:
        f.write(name)
    run(repo, "add", name)
    run(repo, "commit", "-qm", name)


def flip_byte_near_end(path: str) -> None:
    """Başlık (ref listesi) sağlam kalır, pack verisinin sonuna yakın bir bayt bozulur."""
    with open(path, "r+b") as f:
        f.seek(-30, os.SEEK_END)
        b = f.read(1)
        f.seek(-30, os.SEEK_END)
        f.write(bytes([b[0] ^ 0xFF]))


def read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture
def repo(tmp_path) -> str:
    """main (annotated tag v1) + 'yan' branch'i + hiçbir branch'te olmayan bir commit'e tag + bir stash."""
    r = str(tmp_path / "repo")
    os.makedirs(r)
    run(r, "init", "-q", "-b", "main")
    commit(r, "a")
    run(r, "tag", "-a", "v1", "-m", "kilometre taşı")
    run(r, "checkout", "-q", "-b", "yan")
    commit(r, "b")
    run(r, "checkout", "-q", "--detach")
    commit(r, "c")
    run(r, "tag", "dal-disi")
    run(r, "checkout", "-q", "main")
    with open(os.path.join(r, "a"), "w") as f:
        f.write("yarim is")
    run(r, "stash", "-q")
    return r


# --- eski yedeklerin korunması ---------------------------------------------------------------------------

def test_backup_restores_everything_and_rotates_prev_only_on_new_refs(repo, tmp_path):
    dest = str(tmp_path / "yok" / "klasor" / "yedek.bundle")  # üst klasörler yoksa oluşturulmalı
    r1 = bb.backup(repo, dest)
    assert r1.status == "ilk" and not os.path.exists(dest + ".tmp")
    assert bb.bundle_refs(dest, repo) == r1.refs == bb.repo_refs(repo)
    assert "refs/stash" in r1.refs

    # Kullanıcının yapacağı gibi bağımsız bir klon: tüm tag'ler ve branch'ler gelmeli
    clone = str(tmp_path / "klon")
    run(str(tmp_path), "clone", "-q", dest, clone)
    assert run(clone, "rev-parse", "v1", "dal-disi", "origin/yan").split() == \
        run(repo, "rev-parse", "v1", "dal-disi", "yan").split()

    assert bb.backup(repo, dest).status == "ayni" and not os.path.exists(dest + ".prev")  # değişiklik yok

    commit(repo, "d")
    r2 = bb.backup(repo, dest)
    assert r2.status == "prev" and r2.refs != r1.refs and not r2.kept
    assert bb.bundle_refs(dest + ".prev", repo) == r1.refs and bb.bundle_refs(dest, repo) == r2.refs

    assert bb.backup(repo, dest).status == "ayni"  # tekrar çalıştırmak son kilometre taşının .prev'ini ezmemeli
    run(repo, "checkout", "-q", "yan")  # yalnızca HEAD değişti: kilometre taşı değil
    assert bb.backup(repo, dest).status == "ayni" and bb.bundle_refs(dest + ".prev", repo) == r1.refs

    run(repo, "stash", "drop", "-q")  # stash'i atmak günlük iş: geçmiş kaybı sayılmamalı
    res = bb.backup(repo, dest)
    assert res.status == "prev" and not res.kept
    assert sorted(os.listdir(os.path.dirname(dest))) == ["yedek.bundle", "yedek.bundle.prev"]


def test_lost_history_keeps_old_backup_permanently(repo, tmp_path):
    dest = str(tmp_path / "yedek.bundle")
    commit(repo, "d")
    bb.backup(repo, dest)
    commit(repo, "e")
    old_refs = bb.backup(repo, dest).refs  # .prev = 'd'li yedek
    prev = read(dest + ".prev")

    run(repo, "tag", "-d", "v1")                  # kilometre taşı silindi
    run(repo, "reset", "-q", "--hard", "HEAD~2")  # d ve e'ye artık hiçbir ref'ten ulaşılamıyor
    commit(repo, "f")
    res = bb.backup(repo, dest)
    assert res.status == "kalici" and len(res.kept) == 1
    assert any("v1" in n for n in res.notes) and any("2 commit" in n for n in res.notes)
    assert bb.bundle_refs(res.kept[0], repo) == old_refs
    assert read(dest + ".prev") == prev and bb.bundle_refs(dest, repo) == res.refs

    assert bb.backup(repo, dest).status == "ayni"  # sonraki çalıştırmalar kalıcı kopyaya dokunmaz
    assert bb.bundle_refs(res.kept[0], repo) == old_refs


def test_stash_does_not_hide_rewritten_history(repo, tmp_path):
    dest = str(tmp_path / "yedek.bundle")
    commit(repo, "x")
    x = run(repo, "rev-parse", "HEAD").strip()
    bb.backup(repo, dest)
    with open(os.path.join(repo, "x"), "w") as f:
        f.write("yarim is")
    run(repo, "stash", "-q")                                    # stash'in üst commit'i x
    run(repo, "commit", "-q", "--amend", "-m", "x duzeltildi")  # x'e artık yalnızca stash'ten ulaşılıyor
    res = bb.backup(repo, dest)
    assert res.status == "kalici" and bb.bundle_refs(res.kept[0], repo)["refs/heads/main"] == x


def test_detached_commit_in_removed_worktree_counts_as_lost(repo, tmp_path):
    wt = str(tmp_path / "wt")
    run(repo, "worktree", "add", "-q", "--detach", wt, "main")
    commit(wt, "deneme")
    w = run(wt, "rev-parse", "HEAD").strip()
    dest = str(tmp_path / "yedek.bundle")
    bb.backup(repo, dest)
    run(repo, "worktree", "remove", wt)  # git uyarmadan siler: w'ye artık hiçbir şeyden ulaşılamıyor
    res = bb.backup(repo, dest)
    assert res.status == "kalici" and bb.bundle_heads(res.kept[0], repo)["worktrees/wt/HEAD"] == w
    # Uyarı, hiçbir klona gelmeyen bu commit'i geri almanın komutunu vermeli ve o komut çalışmalı
    assert any(w[:7] in n and '"worktrees/wt/HEAD:refs/heads/kurtarilan"' in n for n in res.notes), res.notes
    clone = str(tmp_path / "klon")
    run(str(tmp_path), "clone", "-q", res.kept[0], clone)
    run(clone, "fetch", "-q", res.kept[0], "worktrees/wt/HEAD:refs/heads/kurtarilan")
    assert run(clone, "rev-parse", "kurtarilan").strip() == w


def test_branch_left_only_as_remote_tracking_ref_after_restore_is_kept(repo, tmp_path):
    """Bundle'dan düz `git clone` ile geri yüklenen depoda 'deney' yalnızca origin/deney olarak kalır: sonraki
    yedeklerde refs/heads'ten düşmesi, düz klonla bir daha geri yüklenemeyeceği için kayıp sayılmalı."""
    run(repo, "checkout", "-q", "-b", "deney")
    commit(repo, "x")
    x = run(repo, "rev-parse", "HEAD").strip()
    run(repo, "checkout", "-q", "main")
    dest = str(tmp_path / "yedek.bundle")
    bb.backup(repo, dest)
    restored = str(tmp_path / "geri")
    run(str(tmp_path), "clone", "-q", dest, restored)
    commit(restored, "d")
    res = bb.backup(restored, dest)
    assert res.status == "kalici" and bb.bundle_refs(res.kept[0], restored)["refs/heads/deney"] == x
    commit(restored, "e")
    assert bb.backup(restored, dest).status == "prev"  # uyarı bir kez: sonraki kilometre taşları normal


def test_prev_with_history_missing_from_new_backup_is_kept_aside(repo, tmp_path):
    """Yarım kalan bir çalıştırmadan sonra (dest yok, .prev var) silinen geçmiş, .prev ezilirken kaybolmamalı."""
    dest = str(tmp_path / "yedek.bundle")
    old = bb.backup(repo, dest).refs
    os.replace(dest, dest + ".prev")  # PlacementError sonrası durum
    run(repo, "tag", "-d", "v1")
    assert bb.backup(repo, dest).status == "ilk"
    commit(repo, "d")
    res = bb.backup(repo, dest)
    assert res.status == "prev" and len(res.kept) == 1 and bb.bundle_refs(res.kept[0], repo) == old


def test_kept_copies_with_same_timestamp_do_not_overwrite_each_other(repo, tmp_path):
    dest = str(tmp_path / "yedek.bundle")
    refs_a = bb.backup(repo, dest).refs
    run(repo, "tag", "-d", "v1")
    res_b = bb.backup(repo, dest)
    first = res_b.kept[0]
    same = os.path.getmtime(first)
    os.utime(dest, (same, same))  # B'nin bundle'ı A'nınkiyle aynı saniyede yazılmış gibi
    run(repo, "tag", "-d", "dal-disi")
    res_c = bb.backup(repo, dest)
    assert res_c.status == "kalici" and res_c.kept[0] != first
    assert bb.bundle_refs(first, repo) == refs_a and bb.bundle_refs(res_c.kept[0], repo) == res_b.refs


@pytest.mark.parametrize("spec", [["--branches", "--tags"], ["main", "yan", "v1", "dal-disi"]])
def test_old_backup_without_head_line_is_not_treated_as_corrupt(repo, tmp_path, spec):
    dest = str(tmp_path / "yedek.bundle")
    run(repo, "bundle", "create", "-q", dest, *spec)  # elle yapılmış eski bir yedek: başlıkta HEAD yok
    old = read(dest)
    run(repo, "tag", "-d", "v1")
    res = bb.backup(repo, dest)
    assert res.status == "kalici", res.notes
    assert read(res.kept[0]) == old


@pytest.mark.parametrize("damage", ["header", "pack"])
def test_damaged_existing_backup_is_kept_aside_and_prev_untouched(repo, tmp_path, damage):
    dest = str(tmp_path / "yedek.bundle")
    bb.backup(repo, dest)
    commit(repo, "d")
    bb.backup(repo, dest)  # .prev = ilk yedek
    prev = read(dest + ".prev")
    if damage == "header":  # okunamaz dosya
        with open(dest, "wb") as f:
            f.write(b"bozuk")
    else:  # başlığı okunur ama pack'i bozuk: en olası bit çürümesi
        flip_byte_near_end(dest)
    damaged = read(dest)

    commit(repo, "e")
    res = bb.backup(repo, dest)
    assert res.status == "bozuk" and bb.bundle_refs(dest, repo) == res.refs
    assert read(dest + ".prev") == prev  # sağlam son kilometre taşı ezilmedi
    assert len(res.kept) == 1 and ".bozuk" in res.kept[0] and read(res.kept[0]) == damaged  # bozuk dosya da silinmedi


# --- doğrulama: yanlış kabul -------------------------------------------------------------------------------

def test_verify_rejects_bundle_with_missing_refs(repo, tmp_path):
    partial = str(tmp_path / "eksik.bundle")
    run(repo, "bundle", "create", "-q", partial, "main", "v1")  # 'yan', 'dal-disi', stash ve HEAD yok
    with pytest.raises(bb.BackupError, match="uyuşmuyor"):
        bb.verify_bundle(partial, bb.repo_refs(repo), cwd=repo)


def test_verify_rejects_corrupt_pack(repo, tmp_path):
    bundle = str(tmp_path / "bozuk.bundle")
    run(repo, "bundle", "create", "-q", bundle, "--all")
    flip_byte_near_end(bundle)
    assert bb.bundle_refs(bundle, repo) == bb.repo_refs(repo)  # ref karşılaştırması tek başına yakalayamaz
    with pytest.raises(bb.BackupError):
        bb.verify_bundle(bundle, bb.repo_refs(repo), cwd=repo)


def test_verify_rejects_bundle_missing_objects_of_refs_clone_skips(repo, tmp_path):
    """Başlık stash'i listeliyor ama pack'te nesneleri yok: düz klon stash'i almadığı için bunu görmez."""
    full, partial = str(tmp_path / "tam.bundle"), str(tmp_path / "stashsiz.bundle")
    run(repo, "bundle", "create", "-q", full, "--all")
    run(repo, "bundle", "create", "-q", partial, "--exclude=refs/stash", "--all")
    header = read(full).split(b"\n\n", 1)[0]  # v2 bundle: ref satırları, boş satır, pack
    pack = read(partial).split(b"\n\n", 1)[1]
    forged = str(tmp_path / "sahte.bundle")
    with open(forged, "wb") as f:
        f.write(header + b"\n\n" + pack)
    assert bb.bundle_refs(forged, repo) == bb.repo_refs(repo)
    with pytest.raises(bb.BackupError):
        bb.verify_bundle(forged, bb.repo_refs(repo), cwd=repo)


def test_verify_rejects_malformed_object_that_clone_accepts(repo, tmp_path):
    # hash'i geçerli ama biçimi bozuk commit (author satırı yok): index-pack ve bağlantı kontrolü geçer, yalnızca fsck yakalar
    tree = run(repo, "rev-parse", "main^{tree}").strip()
    body = f"tree {tree}\ncommitter t <t@t> 1 +0000\n\nauthor satiri yok\n".encode()  # bayt: Windows \r\n eklemesin
    sha = run(repo, "hash-object", "--literally", "-w", "-t", "commit", "--stdin", stdin=body).strip()
    run(repo, "update-ref", "refs/heads/bozuk", sha)
    bundle = str(tmp_path / "bicimsiz.bundle")
    run(repo, "bundle", "create", "-q", bundle, "--all")
    run(str(tmp_path), "clone", "-q", bundle, str(tmp_path / "klon"))  # düz klon bunu kabul ediyor
    with pytest.raises(bb.BackupError, match="fsck"):
        bb.verify_bundle(bundle, bb.repo_refs(repo), cwd=repo)


def test_verify_rejects_clone_missing_a_tag(repo, tmp_path, monkeypatch):
    """Bundle'ın ref listesi tam olsa bile düz bir `git clone` tag'i getirmezse yedek geçmemeli."""
    bundle = str(tmp_path / "tam.bundle")
    run(repo, "bundle", "create", "-q", bundle, "--all")
    real_git = bb.git

    def git_clone_without_v1(*args, cwd, **kw):
        out = real_git(*args, cwd=cwd, **kw)
        if args[0] == "for-each-ref" and os.path.basename(cwd) == "klon":
            out = "".join(line + "\n" for line in out.splitlines() if not line.endswith(" refs/tags/v1"))
        return out

    monkeypatch.setattr(bb, "git", git_clone_without_v1)
    with pytest.raises(bb.BackupError, match="refs/tags/v1"):
        bb.verify_bundle(bundle, bb.repo_refs(repo), cwd=repo)


# --- doğrulama: yanlış red (sağlam yedeği engellememeli) ---------------------------------------------------

def test_linked_worktree_does_not_block_backup(repo, tmp_path):
    run(repo, "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "deney")  # bundle'a 'worktrees/wt/HEAD' yazılır
    res = bb.backup(repo, str(tmp_path / "yedek.bundle"))
    assert res.status == "ilk" and "refs/heads/deney" in res.refs


def test_refs_differing_only_in_case_do_not_block_backup(repo, tmp_path):
    run(repo, "pack-refs", "--all")  # gc / otomatik bakım bunu kendiliğinden yapar
    run(repo, "tag", "V1", "yan")    # NTFS'te v1 paketli olduğu için sessizce kabul edilir
    refs = bb.repo_refs(repo)
    assert refs["refs/tags/V1"] != refs["refs/tags/v1"]
    assert bb.backup(repo, str(tmp_path / "yedek.bundle")).status == "ilk"


def test_many_distinct_ref_tips_do_not_break_later_runs(repo, tmp_path):
    """Her annotated tag ayrı bir sha: ~400 tanesi komut satırını Windows'un 32767 karakter sınırının üstüne çıkarırdı."""
    tags = "".join(f"tag cok{i}\nfrom refs/heads/main\ntagger t <t@t> 1 +0000\ndata 1\nx\n" for i in range(1000))
    run(repo, "fast-import", "--quiet", stdin=tags.encode())
    dest = str(tmp_path / "yedek.bundle")
    assert bb.backup(repo, dest).status == "ilk"
    assert bb.backup(repo, dest).status == "ayni"


def test_inherited_git_env_does_not_redirect_verification(repo, tmp_path, monkeypatch):
    """Hook'tan çalıştırılınca miras kalan GIT_DIR/GIT_INDEX_FILE, klon komutlarını kaynak depoya yönlendirmemeli."""
    index = os.path.join(repo, ".git", "index")
    before = read(index)
    monkeypatch.setenv("GIT_DIR", os.path.join(repo, ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", index)
    res = bb.backup(repo, str(tmp_path / "yedek.bundle"))
    assert res.status == "ilk" and res.refs["HEAD"]
    assert read(index) == before  # klonun checkout'u kaynak deponun index'ine yazmadı


# --- başarısızlıklar ---------------------------------------------------------------------------------------

def test_failed_backup_leaves_existing_backup_untouched(repo, tmp_path, monkeypatch):
    dest = str(tmp_path / "yedek.bundle")
    bb.backup(repo, dest)
    before = read(dest)

    def fail(*a, **k):
        raise bb.BackupError("doğrulama başarısız (test)")

    monkeypatch.setattr(bb, "verify_bundle", fail)
    commit(repo, "d")
    with pytest.raises(bb.BackupError):
        bb.backup(repo, dest)
    assert read(dest) == before
    assert not os.path.exists(dest + ".tmp") and not os.path.exists(dest + ".prev")


def test_placement_failure_after_rotation_is_reported_truthfully(repo, tmp_path, monkeypatch):
    dest = str(tmp_path / "yedek.bundle")
    bb.backup(repo, dest)
    commit(repo, "d")
    real_replace = os.replace

    def replace_fails_for_tmp(src, dst):
        if src.endswith(".tmp"):
            raise PermissionError("kilitli (test)")
        real_replace(src, dst)

    monkeypatch.setattr(bb.os, "replace", replace_fails_for_tmp)
    with pytest.raises(bb.PlacementError, match="yerine konamadı"):
        bb.backup(repo, dest)
    assert os.path.exists(dest + ".prev") and os.path.exists(dest + ".tmp")  # hiçbir şey kaybolmadı
    monkeypatch.setattr(bb.os, "replace", real_replace)
    res = bb.backup(repo, dest)  # sonraki çalıştırma düzeltir: bırakılan .tmp yerine konup denetlenir
    assert res.status == "ayni" and bb.bundle_refs(dest, repo) == bb.repo_refs(repo)
    assert not os.path.exists(dest + ".tmp")


@pytest.mark.parametrize("failed_rerun", [False, True])
def test_bundle_left_by_placement_error_is_not_silently_lost(repo, tmp_path, monkeypatch, failed_rerun):
    """PlacementError'ın bıraktığı .tmp o kilometre taşının tek yedeği: sonraki çalıştırma (arada başarısız bir
    çalıştırma olsa bile) onu denetlemeden ezmemeli ya da silmemeli."""
    dest = str(tmp_path / "yedek.bundle")
    bb.backup(repo, dest)
    commit(repo, "d")
    real_replace, real_verify = os.replace, bb.verify_bundle

    def replace_fails_for_tmp(src, dst):
        if src.endswith(".tmp"):
            raise PermissionError("kilitli (test)")
        real_replace(src, dst)

    monkeypatch.setattr(bb.os, "replace", replace_fails_for_tmp)
    with pytest.raises(bb.PlacementError):
        bb.backup(repo, dest)
    monkeypatch.setattr(bb.os, "replace", real_replace)
    left = read(dest + ".tmp")  # 'd'nin tek yedeği
    if failed_rerun:
        def fail(*a, **k):
            raise bb.BackupError("geçici hata (test)")

        monkeypatch.setattr(bb, "verify_bundle", fail)
        with pytest.raises(bb.BackupError):
            bb.backup(repo, dest)
        monkeypatch.setattr(bb, "verify_bundle", real_verify)
    run(repo, "commit", "-q", "--amend", "-m", "d duzeltildi")  # d'ye artık hiçbir ref'ten ulaşılamıyor
    res = bb.backup(repo, dest)
    assert res.status == "kalici" and len(res.kept) == 1 and read(res.kept[0]) == left, res


def test_stale_bundle_lock_from_killed_run_does_not_block_later_runs(repo, tmp_path):
    """Bundle yazılırken zorla öldürülen çalıştırma git'in <dest>.tmp.lock kilidini bırakır: sonraki her
    çalıştırma 'File exists' ile reddedilmemeli."""
    dest = str(tmp_path / "yedek.bundle")
    bb.backup(repo, dest)
    open(dest + ".tmp.lock", "wb").close()
    commit(repo, "d")
    res = bb.backup(repo, dest)
    assert res.status == "prev" and bb.bundle_refs(dest, repo) == bb.repo_refs(repo)
    assert not os.path.exists(dest + ".tmp.lock")
