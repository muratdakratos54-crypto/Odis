"""Git deposunun tamamını (tüm geçmiş, branch'ler, tag'ler) tek bir bundle dosyasına yedekler.

    .venv\\Scripts\\python scripts/backup_bundle.py                                  # varsayılan hedef: D:
    .venv\\Scripts\\python scripts/backup_bundle.py --dest "E:\\yedek\\moe-llm.bundle"

Her önemli kilometre taşından (commit + tag) sonra çalıştırın. Yeni bundle önce <dest>.tmp'ye
yazılır ve doğrulanır; yalnızca tüm kontroller geçerse yerine konur:
  1. bundle'daki ref'ler (HEAD, branch'ler, tag'ler, stash ...) depodakilerle birebir aynı
  2. bundle'dan geçici klasöre temiz `git clone`: pack verisi sağlam, geçmiş eksiksiz (ön koşul commit'i
     yok), HEAD/branch'ler/tag'ler aynı nesneleri gösteriyor; sonra bundle'daki TÜM ref'ler klona
     çekilip `git fsck --full` ile her nesnenin biçimi denetleniyor
Eski yedekler için tek kural: bir yedek dosyası ancak içindeki geçmişin tamamı yeni bundle'da varsa
ezilir, yoksa yazıldığı zamanı taşıyan <ad>.<tarih>.bundle adıyla kenara alınır ve bir daha dokunulmaz:
  * ref'ler aynı                        -> doğrulanmış yenisiyle değiştirilir
  * yeni commit / branch / tag var      -> önceki yedek <dest>.prev olur (bir önceki kilometre taşı)
  * geçmiş kaybolmuş (tag silinmiş/taşınmış, reset/rebase/amend sonrası bazı commit'lere branch, tag ya da
    HEAD'den ulaşılamıyor; stash'ten ya da origin/* kopyasından ulaşılabilmesi sayılmaz) -> kalıcı saklanır
  * mevcut yedek bozuk                  -> silinmez, <ad>.<tarih>.bozuk.bundle olarak kenara alınır
Taşınacak eski yedeğin önce kendisi de doğrulanır.

Bundle yalnızca commit'lenmiş ve bir ref'ten ulaşılabilen geçmişi içerir: data/ ve checkpoints/
git'te değildir (ayrıca yedeklenmeli), commit'lenmemiş değişiklikler girmez (betik uyarır), stash'in
yalnızca son girdisi girer, Git LFS kullanılırsa LFS dosyalarının içeriği girmez.
Geri yükleme:  git clone "D:\\foton gama v1 yedek\\moe-llm.bundle" moe-llm
  Tag'ler gelir; yalnızca HEAD'in branch'i yerel olur, diğerleri origin/<ad> olarak gelir
  (`git branch <ad> origin/<ad>`). Tüm ref'leri (stash dahil) aynen almak için: git clone --mirror.
  Uyarıda 'worktrees/<ad>/HEAD' gibi sahte adlar hiçbir klona gelmez:
  git fetch <yedek> "worktrees/<ad>/HEAD:refs/heads/kurtarilan"
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DEST = r"D:\foton gama v1 yedek\moe-llm.bundle"


class BackupError(RuntimeError):
    pass


class PlacementError(BackupError):
    """Eski yedek taşındıktan sonra yenisi yerine konamadı: 'mevcut yedeğe dokunulmadı' denemez."""


@dataclasses.dataclass
class Result:
    refs: dict[str, str]
    status: str  # 'ilk' | 'ayni' | 'prev' | 'kalici' (geçmiş kaybı) | 'bozuk' (mevcut yedek bozuktu)
    kept: list[str]  # kenara alınan, bir daha dokunulmayacak eski yedekler
    notes: list[str]  # kullanıcıya gösterilecek uyarılar


@functools.cache
def _repo_local_env_vars() -> frozenset[str]:
    out = subprocess.run(["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True).stdout
    return frozenset(out.split())


def git(*args: str, cwd: str, input: str | None = None) -> str:
    # Betik bir git hook'undan çalıştırılırsa GIT_DIR, GIT_INDEX_FILE ... miras kalır: bunlar temizlenmezse
    # doğrulama klonundaki komutlar kaynak depoyu okur, klonun checkout'u da kaynak deponun index'ine yazar.
    env = {k: v for k, v in os.environ.items() if k not in _repo_local_env_vars()}
    # stdin bayt olarak: Windows'ta text modu '\n'i '\r\n'e çevirir ve git satır sonundaki '\r'yi ada katar
    r = subprocess.run(["git", *args], cwd=cwd, env=env, input=None if input is None else input.encode(),
                       capture_output=True)
    if r.returncode != 0:
        raise BackupError(f"git {' '.join(args)} başarısız (kod {r.returncode}):\n"
                          f"{r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout.decode("utf-8", "replace")


def parse_refs(text: str) -> dict[str, str]:
    """'<sha> <ref>' satırlarını {ref: sha} sözlüğüne çevirir."""
    return {ref: sha for sha, ref in (line.split(maxsplit=1) for line in text.splitlines() if line.strip())}


def repo_refs(repo: str) -> dict[str, str]:
    """Depodaki tüm ref'ler + HEAD; `bundle --all` ile aynı biçimde (symref'ler hedefin sha'sına çözülür,
    annotated tag'lerde tag nesnesinin kendi sha'sı)."""
    refs = parse_refs(git("for-each-ref", "--format=%(objectname) %(refname)", cwd=repo))
    refs["HEAD"] = git("rev-parse", "HEAD", cwd=repo).strip()
    return refs


def bundle_heads(bundle: str, cwd: str) -> dict[str, str]:
    """Bundle başlığının tamamı. `--all` başka worktree'lerin HEAD'lerini de 'worktrees/<ad>/HEAD' sahte
    adlarıyla yazar: geçmiş kaybı hesabı bunları da görmeli (orada yapılmış detached commit'ler)."""
    return parse_refs(git("bundle", "list-heads", bundle, cwd=cwd))


def only_refs(heads: dict[str, str]) -> dict[str, str]:
    # for-each-ref sahte worktree adlarını listelemez, klon da almaz: ref karşılaştırmasına girselerdi
    # worktree varken her yedek reddedilirdi.
    return {r: s for r, s in heads.items() if r == "HEAD" or r.startswith("refs/")}


def bundle_refs(bundle: str, cwd: str) -> dict[str, str]:
    return only_refs(bundle_heads(bundle, cwd))


def verify_bundle(bundle: str, expected: dict[str, str], cwd: str) -> None:
    """Bundle beklenen ref'lerin hepsini birebir içermeli ve temiz bir klon olarak açılabilmeli."""
    got = bundle_refs(bundle, cwd)
    if got != expected:
        diff = sorted(set(got.items()) ^ set(expected.items()))
        lines = [f"  {'bundle' if (r, s) in got.items() else 'depo  '}: {s[:10]} {r}" for r, s in diff]
        raise BackupError("bundle'daki ref'ler depodakilerle uyuşmuyor:\n" + "\n".join(lines))

    with tempfile.TemporaryDirectory(prefix="bundle-dogrula-") as tmp:
        clone = os.path.join(tmp, "klon")
        # Checkout'lu normal klon: kullanıcının geri yükleme komutunun aynısı. index-pack pack'teki her nesnenin
        # hash'ini, clone da ön koşul commit'lerini ve branch/tag'lerin bağlantısını denetler.
        git("clone", "-q", "--origin", "origin", bundle, clone, cwd=tmp)
        cloned = parse_refs(git("for-each-ref", "--format=%(objectname) %(refname)", cwd=clone))
        if "HEAD" in expected:  # elle `git bundle create x main` ile yapılmış eski yedeklerde HEAD satırı yok
            cloned["HEAD"] = git("rev-parse", "HEAD", cwd=clone).strip()
        # Klonda branch'ler refs/remotes/origin/<ad>, tag'ler refs/tags/<ad> olarak gelir
        for ref, sha in expected.items():
            if ref.startswith("refs/heads/"):
                want = "refs/remotes/origin/" + ref.removeprefix("refs/heads/")
            elif ref.startswith("refs/tags/") or ref == "HEAD":
                want = ref
            else:
                continue
            if cloned.get(want) != sha:
                raise BackupError(f"klonda {want} = {cloned.get(want)}, beklenen {sha} ({ref})")
        # Klonun almadığı ref'ler (stash, uzak dal kopyaları) de nesneleriyle birlikte gelmeli. Hedef adlar
        # numaralı: NTFS'te yalnızca harf büyüklüğüyle ayrılan ref'ler (v1/V1) ayrı dosyalar olarak yazılamaz.
        # Refspec'ler stdin'den: yüzlerce ref komut satırını Windows'un 32767 karakter sınırının üstüne çıkarır.
        specs = "".join(f"+{r}:refs/yedek/{i}\n" for i, r in enumerate(sorted(expected)) if r != "HEAD")
        git("fetch", "-q", "--no-tags", "--stdin", bundle, cwd=clone, input=specs)
        # fsck, index-pack'in geçirdiği ama biçimi bozuk nesneleri de (ör. author satırı olmayan commit) yakalar
        git("fsck", "--full", "--no-dangling", cwd=clone)


def lost_history(repo: str, old: dict[str, str], new: dict[str, str]) -> list[str]:
    """Eski yedekte olup yenisine girmeyecek geçmiş: silinen/taşınan tag'ler ve yeni başlıktaki branch, tag,
    HEAD'lerin hiçbirinden ulaşılamayan commit'ler. İki tarafta da sayılmayanlar:
      * stash: her `stash drop` uyarı üretmesin, ama amend/reset ile kaybolan bir commit'e yalnızca stash'ten
        ulaşılabiliyor diye de gizlenmesin
      * uzak dal kopyaları (refs/remotes/*): bundle'dan `git clone` ile geri yüklenen depoda branch'ler yalnızca
        origin/* olarak kalır; sonraki yedeklerde refs/heads'ten sessizce düşüp düz klonda kaybolmasınlar."""
    def tip_of(r: str) -> bool:
        return r != "refs/stash" and not r.startswith("refs/remotes/")

    lost = [f"tag {r} silinmiş/taşınmış" for r, s in old.items() if r.startswith("refs/tags/") and new.get(r) != s]
    tips = sorted({s for r, s in old.items() if tip_of(r)})
    keep = sorted({s for r, s in new.items() if tip_of(r)})
    try:  # sha'lar stdin'den (komut satırı sınırı, bkz. verify_bundle)
        gone = set(git("rev-list", "--stdin", cwd=repo,
                       input="".join([f"{s}\n" for s in tips] + [f"^{s}\n" for s in keep])).split())
    except BackupError:  # eski uçlardan biri depoda artık yok (gc)
        lost.append("eski yedekteki bazı nesneler depoda artık yok")
        return lost
    if gone:
        lost.append(f"{len(gone)} commit'e artık hiçbir branch/tag/HEAD'den ulaşılamıyor; kaybolan uçlar:")
        for r, s in sorted(old.items()):
            if s in gone and tip_of(r) and not r.startswith("refs/tags/"):  # tag'ler yukarıda adıyla yazıldı
                # worktrees/<ad>/HEAD gibi sahte adlar hiçbir klona gelmez; ancak açıkça fetch edilebilir
                hint = "" if r == "HEAD" or r.startswith("refs/") else \
                    f'  (geri almak için: git fetch <kenara alınan yedek> "{r}:refs/heads/kurtarilan")'
                lost.append(f"  {s[:7]} {r}{hint}")
    return lost


def keep_aside(path: str, dest: str, tag: str = "") -> str:
    """Eski yedeği, yazıldığı zamanı taşıyan ve var olan hiçbir dosyayı ezmeyen bir adla kenara alır."""
    root, ext = os.path.splitext(dest)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(path)))
    name, n = f"{root}.{stamp}{tag}{ext}", 1
    while os.path.exists(name):  # aynı saniyede yazılmış iki yedek
        n += 1
        name = f"{root}.{stamp}{tag}_{n}{ext}"
    os.replace(path, name)
    return name


def backup(repo: str, dest: str) -> Result:
    """Bundle'ı oluşturur, doğrular ve dest'e koyar; eski yedekleri yukarıdaki kurala göre korur."""
    dest = os.path.abspath(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp, prev = dest + ".tmp", dest + ".prev"
    if os.path.exists(tmp) and not os.path.exists(dest):
        # Önceki çalıştırmanın PlacementError'da bıraktığı doğrulanmış yedek: ezilmeden önce mevcut yedeğin yerine
        # konur ki diğer eski yedekler gibi denetlensin (geçmiş kaybı -> kenara, bozuk -> .bozuk).
        os.replace(tmp, dest)
    # Bundle yazılırken zorla öldürülen bir çalıştırma (Görev Yöneticisi, elektrik kesintisi) git'in kilidini
    # bırakır; betik aynı anda iki kez çalıştırılmadığı için var olan kilit bayattır.
    with contextlib.suppress(FileNotFoundError):
        os.remove(tmp + ".lock")

    refs = repo_refs(repo)
    try:
        git("bundle", "create", "-q", tmp, "--all", cwd=repo)
        with open(tmp, "r+b") as f:  # git diske yazmayı zorlamaz: eski yedek silinmeden önce yenisi diskte olsun
            os.fsync(f.fileno())
        verify_bundle(tmp, refs, cwd=repo)
        new = bundle_heads(tmp, cwd=repo)
    except BaseException:
        with contextlib.suppress(OSError):  # asıl hatayı tmp silme hatasıyla örtme
            os.remove(tmp)
        raise

    res = Result(refs, "ilk", [], [])
    dest_moved = False
    if os.path.exists(dest):
        try:
            old = bundle_heads(dest, cwd=repo)
            if old != new:  # eski yedek ezilecek ya da taşınacak: başlığı okunsa bile pack'i bozuk olabilir
                verify_bundle(dest, only_refs(old), cwd=repo)
        except BackupError as e:
            res.status = "bozuk"
            res.kept.append(keep_aside(dest, dest, ".bozuk"))
            res.notes.append(f"mevcut yedek bozuk: {str(e).strip().splitlines()[-1]}")
            dest_moved = True
        else:
            without_head = {r: s for r, s in only_refs(old).items() if r != "HEAD"}
            if lost := lost_history(repo, old, new):
                res.status = "kalici"
                res.notes += lost
                res.kept.append(keep_aside(dest, dest))
                dest_moved = True
            elif without_head != {r: s for r, s in refs.items() if r != "HEAD"}:  # yalnızca checkout: kilometre taşı değil
                res.status = "prev"
                if os.path.exists(prev):  # .prev de ancak geçmişi yeni yedekte varsa ezilir
                    try:
                        prev_lost = lost_history(repo, bundle_heads(prev, cwd=repo), new)
                        tag = ""
                    except BackupError:
                        prev_lost, tag = ["okunamıyor"], ".bozuk"
                    if prev_lost:
                        res.kept.append(keep_aside(prev, dest, tag))
                        res.notes.append(f"{prev}: " + "; ".join(prev_lost))
                os.replace(dest, prev)
                dest_moved = True
            else:
                res.status = "ayni"
    try:
        os.replace(tmp, dest)
    except OSError as e:
        if not dest_moved:
            raise
        raise PlacementError(f"{e}\neski yedek taşındı ama doğrulanmış yeni bundle yerine konamadı; şu an {tmp} "
                             f"olarak duruyor. Betiği tekrar çalıştırın.") from e
    return res


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows yönlendirmesinde Türkçe karakterler
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # sys.exit(mesaj) stderr'e yazar
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default=DEFAULT_DEST, help="bundle dosyasının yolu")
    p.add_argument("--repo", default=ROOT)
    args = p.parse_args()

    try:
        dirty = git("status", "--porcelain", cwd=args.repo).splitlines()
        if dirty:
            print(f"[uyarı] {len(dirty)} commit'lenmemiş değişiklik bundle'a GİRMEZ (önce commit'leyin):")
            for line in dirty[:10]:
                print("   ", line)
        res = backup(args.repo, args.dest)
    except PlacementError as e:
        sys.exit(f"[hata] {e}")
    except (BackupError, OSError) as e:
        sys.exit(f"[hata] {e}\nmevcut yedeğe dokunulmadı.")

    dest = os.path.abspath(args.dest)
    print(f"bundle: {dest} ({os.path.getsize(dest) / 1e6:.2f} MB)")
    for ref, sha in sorted(res.refs.items()):
        print(f"  {sha[:7]}  {ref}")
    print("doğrulandı: ref eşleşmesi + temiz klon (HEAD/branch/tag) + tüm ref'ler + fsck --full")
    if res.status == "prev":
        print(f"önceki yedek (farklı ref'ler): {dest}.prev")
    elif res.status == "ayni" and os.path.exists(dest + ".prev"):
        print(f"ref'ler değişmedi; {dest}.prev olduğu gibi korundu")
    elif res.status == "kalici":
        print("[UYARI] depoda, önceki yedekte olan geçmiş artık yok:")
    elif res.status == "bozuk":
        print("[UYARI] mevcut yedek okunamadı/bozuktu (diski kontrol edin); doğrulanmış yenisiyle değiştirildi:")
    for note in res.notes:
        print("   ", note)
    for path in res.kept:
        print(f"eski yedek silinmedi, kenara alındı (bir daha dokunulmaz): {path}")


if __name__ == "__main__":
    main()
