"""FOTON v0.2 korpusu: alan-homojen, %100 Tier A lisanslı NL + kod + matematik.

    .venv\\Scripts\\python scripts/prepare_foton_corpus.py --out data/foton

Neden bu betik var (kararların gerekçesi kayıtta dursun):

* **Alan-homojen packing.** Sequence-level router'a anlamlı soru sormak için bir dizi TEK alandan
  gelmeli. Bu, alan başına ayrı `.bin` dosyasıyla YAPISAL olarak sağlanıyor: pencere bir memmap'in
  içinden alınır ve alanlar farklı memmap'lerde, yani pencere alan sınırını geçemez. Doküman
  sınırları EOS ile işaretli.
* **%100 Tier A lisans.** Her doküman kendi lisansını taşımak ZORUNDA; taşımıyorsa atılır.
  Kabul: public domain / CC0 / CC BY (share-alike YOK) / Open Parliament–Government Licence /
  Blue Oak permissive kod lisansları. Red: BY-SA, GFDL, NC, ND, lisansı olmayan.
* **Alanlar arası oran EŞİT, ve bu bilinçli bir nötrlük.** Bir alanı "X kolunun dilated modülüne
  daha çok yarar" diye ağırlıklandırmak korpusu kanıtlanmak istenen sonuca doğru eğmek olurdu.
  NL içindeki caselaw/hansard oranı da aynı gerekçeyle eşit — her kaynak TAM kendi quota'sına kırpılır.
* **Tokenizer örneği alan-dengeli.** Ölçüldü: matematikte eğitilmiş 32k tokenizer koda 1.84
  bayt/token veriyor (matematikte 4.33); dengesiz örnek alan bütçelerini bozar.
* **Tokenizasyon akış halinde.** Tokenleri Python listesinde biriktirmek 24 GB RSS'e çıkıyordu
  (Python int nesnesi 28 bayt). Her batch numpy'a çevrilip doğrudan diske yazılır.

Kaynaklar (hepsi common-pile v0.1, doküman başına lisanslı):
  NL   : caselaw_access_project_filtered (Public Domain) + uk_hansard_filtered (OPL v3.0)
  kod  : stackv2_edu_filtered (SPDX permissive, license_type == "permissive")
  mat. : arxiv_papers_filtered (CC BY / CC0 / PD payı)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.tokenizer import SentencePieceTokenizer  # noqa: E402

HF = "https://huggingface.co/datasets/common-pile"
# (alan, kaynak adı, HF deposu, dosya, indirilecek gz baytı)
SOURCES = [
    ("nl", "caselaw", "caselaw_access_project_filtered", "CAP-Dolma-0000.json.gz", 170_000_000),
    ("nl", "hansard", "uk_hansard_filtered", "uk_hansard-0000.json.gz", 170_000_000),
    ("code", "stackv2_edu", "stackv2_edu_filtered", "stack-edu-0000.json.gz", 260_000_000),
    ("math", "arxiv", "arxiv_papers_filtered", "arxiv-papers-0000.json.gz", 240_000_000),
]
BLUE_OAK_HINT = ("mit", "apache", "bsd", "isc", "unlicense", "zlib", "cc0", "0bsd", "mit-0",
                 "postgresql", "python-2.0", "ncsa", "boost", "afl", "artistic", "upl", "wtfpl")
DOC_SEP = "\x1e"


def tier_a(lic, license_type=None) -> tuple[bool, str]:
    """Tier A mı? (kabul, sınıf). Sıra önemli: share-alike kontrolü 'attribution'dan ÖNCE gelir,
    çünkü "Attribution Share-Alike" ikisini de içerir."""
    if isinstance(lic, list):
        lic = ",".join(str(x) for x in lic)
    s = (lic or "").strip()
    if not s:
        return False, "RED: lisans yok"
    low = s.lower()
    if ("share" in low and "alike" in low) or "by-sa" in low or "-sa-" in low or low.endswith("-sa"):
        return False, "RED: BY-SA (share-alike)"
    if "noncommercial" in low or "non-commercial" in low or "-nc" in low or "nc-" in low:
        return False, "RED: NC (ticari degil)"
    if "noderiv" in low or "-nd" in low or "nd-" in low:
        return False, "RED: ND (turev yok)"
    if "free documentation" in low or "gfdl" in low or "gpl" in low or "copyleft" in low:
        return False, "RED: copyleft"
    if license_type is not None and str(license_type).lower() == "permissive":
        return True, "permissive (SPDX)"
    if "public domain" in low or "cc0" in low or "creative commons zero" in low:
        return True, "PD / CC0"
    if "open parliament" in low or "open government" in low:
        return True, "OPL / OGL"
    if "attribution" in low or "cc-by" in low or "cc by" in low or "/by/" in low:
        return True, "CC BY"
    if any(h == low or low.startswith(h + "-") or low.startswith(h + " ") for h in BLUE_OAK_HINT):
        return True, "permissive (SPDX)"
    return False, f"RED: taninmayan ({s[:38]})"


def batched_docs(path: str, batch: int):
    """Düz metin dosyasını DOC_SEP'ten dokümanlara böler, `batch`lik listeler yield eder."""
    cur: list[str] = []
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == DOC_SEP:
                if cur:
                    out.append("".join(cur))
                    cur = []
                if len(out) >= batch:
                    yield out
                    out = []
            else:
                cur.append(line)
    if cur:
        out.append("".join(cur))
    if out:
        yield out


def fetch(repo: str, fname: str, nbytes: int, dst: str) -> None:
    """Gereken kadarını range isteğiyle indir (gzip akışı baştan çözülebilir). Varsa ve yeterliyse atla."""
    if os.path.exists(dst) and os.path.getsize(dst) >= nbytes * 0.98:
        print(f"  [var] {os.path.basename(dst)} ({os.path.getsize(dst)/1e6:.0f} MB)", flush=True)
        return
    url = f"{HF}/{repo}/resolve/main/{fname}"
    print(f"  [indir] {repo}/{fname} -> {nbytes/1e6:.0f} MB", flush=True)
    subprocess.run(["curl", "-sL", "--max-time", "900", "-r", f"0-{nbytes}", "-o", dst, url], check=True)
    print(f"          {os.path.getsize(dst)/1e6:.0f} MB indi", flush=True)


def extract(gz: str, out_txt: str, byte_target: int) -> dict:
    """Tier A süzgecinden geçen dokümanları düz metne yaz; lisans dağılımını döndür."""
    st = {"kept_docs": 0, "kept_bytes": 0, "dropped_docs": 0, "dropped_bytes": 0, "classes": {}}
    if os.path.exists(out_txt) and os.path.getsize(out_txt) >= byte_target * 0.95:
        st["reused"] = True
        st["kept_bytes"] = os.path.getsize(out_txt)
        return st
    with gzip.open(gz, "rt", encoding="utf-8") as f, open(out_txt, "w", encoding="utf-8") as g:
        try:
            for line in f:
                d = json.loads(line)
                md = d.get("metadata", {})
                ok, cls = tier_a(md.get("license"), md.get("license_type"))
                t = d.get("text", "").strip()
                b = len(t.encode())
                st["classes"][cls] = st["classes"].get(cls, 0) + b
                if not ok or not t:
                    st["dropped_docs"] += 1
                    st["dropped_bytes"] += b
                    continue
                g.write(t.replace("\x00", "") + "\n" + DOC_SEP + "\n")
                st["kept_docs"] += 1
                st["kept_bytes"] += b
                if st["kept_bytes"] >= byte_target:
                    break
        except (EOFError, OSError):
            pass   # range indirmenin sonu: kırpılmış gzip akışı beklenen durum
    return st


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/foton")
    p.add_argument("--cache", default="data/foton/_raw")
    p.add_argument("--vocab_size", type=int, default=32768)
    p.add_argument("--tokens_per_domain", type=int, default=100_000_000)
    p.add_argument("--val_tokens_per_domain", type=int, default=2_000_000)
    p.add_argument("--tok_sample_mb_per_domain", type=int, default=120)
    p.add_argument("--bytes_per_source", type=int, default=0)
    p.add_argument("--doc_batch", type=int, default=100)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    # --- 1. indir + Tier A suzgeci + duz metin ---
    print("== 1. Indirme ve Tier A lisans suzgeci ==", flush=True)
    per_source = args.bytes_per_source or int((args.tokens_per_domain + args.val_tokens_per_domain) * 5.5)
    domain_sources: dict[str, list[str]] = {}
    lic_report: dict[str, dict] = {}
    n_per_dom: dict[str, int] = {}
    for dom, *_ in SOURCES:
        n_per_dom[dom] = n_per_dom.get(dom, 0) + 1
    for dom, name, repo, fname, gzbytes in SOURCES:
        gz = os.path.join(args.cache, f"{name}.json.gz")
        fetch(repo, fname, gzbytes, gz)
        txt = os.path.join(args.cache, f"{name}.txt")
        st = extract(gz, txt, per_source // n_per_dom[dom])
        tot = st["kept_bytes"] + st["dropped_bytes"]
        if st.get("reused"):
            print(f"  {name:12s} [onceki cikarim, {st['kept_bytes']/1e6:.1f} MB]", flush=True)
        else:
            print(f"  {name:12s} tutulan {st['kept_docs']:8,d} dok / {st['kept_bytes']/1e6:7.1f} MB | "
                  f"atilan {st['dropped_docs']:7,d} / {st['dropped_bytes']/1e6:6.1f} MB "
                  f"(%{100*st['dropped_bytes']/max(tot,1):.2f})", flush=True)
            for cls, b in sorted(st["classes"].items(), key=lambda x: -x[1]):
                print(f"      {cls:44s} %{100*b/max(tot,1):5.2f}", flush=True)
        domain_sources.setdefault(dom, []).append(txt)
        lic_report[name] = {"domain": dom, **{k: v for k, v in st.items() if k != "classes"},
                            "license_classes_bytes": st.get("classes", {})}

    # --- 2. tokenizer: ALAN-DENGELI ornek ---
    print("\n== 2. Tokenizer (32k, alan-dengeli ornek) ==", flush=True)
    prefix = os.path.join(args.out, "tokenizer")
    if not os.path.exists(prefix + ".model"):
        sample = os.path.join(args.cache, "tok_sample.txt")
        per_dom = args.tok_sample_mb_per_domain << 20
        with open(sample, "w", encoding="utf-8") as g:
            for dom, files in domain_sources.items():
                quota = per_dom // len(files)
                for fp in files:
                    written = 0
                    with open(fp, encoding="utf-8") as f:
                        for line in f:
                            if line.strip() and line.strip() != DOC_SEP:
                                g.write(line)
                                written += len(line.encode())
                                if written >= quota:
                                    break
                print(f"  {dom}: {per_dom/1e6:.0f} MB ornek", flush=True)
        print(f"  egitiliyor (V={args.vocab_size}) ...", flush=True)
        SentencePieceTokenizer.train(sample, prefix, args.vocab_size)
    tok = SentencePieceTokenizer(prefix + ".model")
    print(f"  vocab {tok.vocab_size}", flush=True)

    # --- 3. alan basina tokenize: ayri .bin, AKIS halinde ---
    print("\n== 3. Alan basina tokenize (alan-homojen, akis) ==", flush=True)
    meta = {"tokenizer": "sentencepiece", "model_file": "tokenizer.model",
            "vocab_size": tok.vocab_size, "bos_id": tok.bos_id, "eos_id": tok.eos_id,
            "tier": "A-only", "domains": {}, "licensing": lic_report,
            "notes": "alan basina ayri .bin: bir pencere hicbir zaman alan sinirini gecemez"}
    for dom, files in domain_sources.items():
        need = args.tokens_per_domain + args.val_tokens_per_domain
        quota = need // len(files)
        tmp = os.path.join(args.out, f"{dom}_all.bin")
        bytes_seen = raw_tokens = 0
        with open(tmp, "wb") as g:
            for fp in files:
                src_got = 0
                for batch in batched_docs(fp, args.doc_batch):
                    enc = tok.encode_batch(batch)
                    a = np.concatenate([np.array(e + [tok.eos_id], dtype=np.uint16) for e in enc])
                    raw_tokens += int(a.size)
                    bytes_seen += sum(len(b.encode()) for b in batch)
                    if src_got + a.size > quota:
                        a = a[: quota - src_got]
                    a.tofile(g)
                    src_got += int(a.size)
                    del enc, a
                    if src_got >= quota:
                        break
                if src_got < quota:
                    print(f"      [UYARI] {os.path.basename(fp)}: quota {quota:,} yerine {src_got:,}", flush=True)
                print(f"      {os.path.basename(fp):22s} {src_got:12,d} token (quota {quota:,})", flush=True)
        bpt = bytes_seen / max(raw_tokens, 1)
        allm = np.memmap(tmp, dtype=np.uint16, mode="r")
        nval = args.val_tokens_per_domain
        allm[-nval:].tofile(os.path.join(args.out, f"{dom}_val.bin"))
        allm[:-nval].tofile(os.path.join(args.out, f"{dom}_train.bin"))
        ntrain = int(allm.size) - nval
        del allm
        os.remove(tmp)
        meta["domains"][dom] = {"train_tokens": ntrain, "val_tokens": nval,
                                "bytes_per_token": round(bpt, 3),
                                "sources": [os.path.basename(f) for f in files]}
        print(f"  {dom:5s} train {ntrain:12,d} | val {nval:9,d} | {bpt:.2f} bayt/token", flush=True)

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    tt = sum(d["train_tokens"] for d in meta["domains"].values())
    print(f"\ntoplam train {tt:,} token, {len(meta['domains'])} alan -> {args.out}/meta.json", flush=True)


if __name__ == "__main__":
    main()
