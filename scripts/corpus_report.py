"""FOTON korpusunun lisans ve alan raporu + doğrulama (ön-kaydın eki olarak saklanır).

    .venv\\Scripts\\python scripts/corpus_report.py --data data/foton --out experiments/foton-v02-prereg/corpus_report.json

Üç iş yapar:
  1. Ham gz kaynaklarını tarayıp DOKÜMAN BAŞINA lisans dağılımını bayt cinsinden raporlar
     (Tier A / Tier B ayrımı; kullanıcı şartı: "BY-SA payını ayrı tut").
  2. Alan istatistikleri: token sayıları, bayt/token, kaynak içi oranlar.
  3. DOĞRULAMA: alan-homojenliği yapısal olarak sağlanıyor mu, token aralıkları geçerli mi,
     tokenizer round-trip kayıpsız mı, alanlar birbirinden ayırt edilebilir mi.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.tokenizer import SentencePieceTokenizer  # noqa: E402

_spec = importlib.util.spec_from_file_location("pfc", os.path.join(ROOT, "scripts", "prepare_foton_corpus.py"))
pfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pfc)


def scan_licenses(gz: str, max_docs: int = 400_000) -> dict:
    """Ham shard'ı tarayıp doküman başına lisans dağılımını (bayt) döndürür."""
    classes: dict[str, list[int]] = {}
    raw: dict[str, list[int]] = {}
    n = 0
    try:
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            for line in f:
                if n >= max_docs:
                    break
                d = json.loads(line)
                md = d.get("metadata", {})
                ok, cls = pfc.tier_a(md.get("license"), md.get("license_type"))
                b = len(d.get("text", "").encode())
                classes.setdefault(cls, [0, 0])
                classes[cls][0] += 1
                classes[cls][1] += b
                lic = md.get("license")
                if isinstance(lic, list):
                    lic = ",".join(str(x) for x in lic)
                key = str(lic)[:70] if lic else "(yok)"
                raw.setdefault(key, [0, 0])
                raw[key][0] += 1
                raw[key][1] += b
                n += 1
    except (EOFError, OSError):
        pass
    tot = sum(v[1] for v in classes.values())
    tier_a_bytes = sum(v[1] for k, v in classes.items() if not k.startswith("RED"))
    return {"docs_scanned": n, "bytes": tot,
            "tier_a_share": tier_a_bytes / tot if tot else 0.0,
            "classes": {k: {"docs": v[0], "bytes": v[1], "share": v[1] / tot if tot else 0.0}
                        for k, v in sorted(classes.items(), key=lambda x: -x[1][1])},
            "top_raw_licenses": {k: {"docs": v[0], "share": v[1] / tot if tot else 0.0}
                                 for k, v in sorted(raw.items(), key=lambda x: -x[1][1])[:10]}}


def verify(data_dir: str, meta: dict) -> dict:
    """Korpusun ön-kayıtta iddia edilen özelliklerini fiilen sınar."""
    out: dict = {}
    tok = SentencePieceTokenizer(os.path.join(data_dir, "tokenizer.model"))
    V = tok.vocab_size
    out["vocab_size"] = V
    out["vocab_matches_meta"] = V == meta["vocab_size"]
    # 1) token aralıkları
    rng = {}
    for dom in meta["domains"]:
        for split in ("train", "val"):
            a = np.memmap(os.path.join(data_dir, f"{dom}_{split}.bin"), dtype=np.uint16, mode="r")
            rng[f"{dom}_{split}"] = {"tokens": int(a.size), "min": int(a.min()), "max": int(a.max()),
                                     "in_range": bool(a.max() < V)}
    out["ranges"] = rng
    out["all_in_range"] = all(v["in_range"] for v in rng.values())
    # 2) alan-homojenlik YAPISAL: ayrı dosyalar -> pencere alan sınırını geçemez
    out["domain_homogeneous_by_construction"] = True
    out["domain_files"] = sorted(f for f in os.listdir(data_dir) if f.endswith(".bin"))
    # 3) tokenizer round-trip: METINDEN basla. Keyfi bir token araligini decode edip yeniden
    # encode etmek BPE'de kanonik olmayan sinirlar yuzunden ayni id'leri vermez (bu tokenizer
    # hatasi degil, testin hatasi olurdu). Onemli olan ozellik: decode(encode(metin)) == metin.
    rt = {}
    for dom, d in meta["domains"].items():
        src = os.path.join(data_dir, "_raw", d["sources"][0])
        with open(src, encoding="utf-8") as f:
            text = f.read(200_000)
        text = text[text.find(chr(10)) + 1:].replace(pfc.DOC_SEP + chr(10), "")  # yarim satiri/ayiriciyi at
        ids = tok.encode(text)
        back = tok.decode(ids)
        rt[dom] = {"chars": len(text), "tokens": len(ids), "lossless": back == text,
                   "bytes_per_token": round(len(text.encode()) / max(len(ids), 1), 3),
                   "sample": text[:150]}
    out["roundtrip"] = rt
    # 4) alanlar ayırt edilebilir mi: aynı tokenizer'da alan başına en sık 8 parça
    top = {}
    for dom in meta["domains"]:
        a = np.memmap(os.path.join(data_dir, f"{dom}_train.bin"), dtype=np.uint16, mode="r")
        sample = np.asarray(a[: 2_000_000])
        cnt = np.bincount(sample, minlength=V)
        idx = np.argsort(cnt)[::-1][:10]
        top[dom] = [(tok.sp.id_to_piece(int(i)).replace("▁", "_"), int(cnt[i])) for i in idx]
    out["top_pieces_per_domain"] = top
    # 5) alanlar arası örtüşme: en sık 200 parçanın Jaccard'ı (düşük = alanlar farklı)
    sets = {}
    for dom in meta["domains"]:
        a = np.memmap(os.path.join(data_dir, f"{dom}_train.bin"), dtype=np.uint16, mode="r")
        cnt = np.bincount(np.asarray(a[: 2_000_000]), minlength=V)
        sets[dom] = set(int(i) for i in np.argsort(cnt)[::-1][:200])
    jac = {}
    doms = sorted(sets)
    for i, x in enumerate(doms):
        for y in doms[i + 1:]:
            inter = len(sets[x] & sets[y])
            jac[f"{x}~{y}"] = round(inter / len(sets[x] | sets[y]), 3)
    out["top200_jaccard"] = jac
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/foton")
    p.add_argument("--out", default="experiments/foton-v02-prereg/corpus_report.json")
    args = p.parse_args()
    with open(os.path.join(args.data, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    print("== LISANS DAGILIMI (doküman başına, bayt cinsinden) ==")
    lic: dict[str, dict] = {}
    for dom, name, repo, fname, _ in pfc.SOURCES:
        gz = os.path.join(args.data, "_raw", f"{name}.json.gz")
        if not os.path.exists(gz):
            print(f"  {name}: ham shard yok, atlandı")
            continue
        r = scan_licenses(gz)
        r["domain"] = dom
        r["repo"] = repo
        lic[name] = r
        print(f"\n  {name} ({dom}) — {r['docs_scanned']:,} doküman, {r['bytes']/1e6:.0f} MB tarandı")
        print(f"    TIER A PAYI: %{100*r['tier_a_share']:.2f}")
        for cls, v in r["classes"].items():
            print(f"      {cls:44s} %{100*v['share']:6.2f}  ({v['docs']:,} dok)")

    print("\n== ALAN ISTATISTIKLERI ==")
    print(f"  {'alan':6s} {'train token':>13s} {'val token':>10s} {'bayt/token':>11s}  kaynaklar")
    for dom, d in meta["domains"].items():
        print(f"  {dom:6s} {d['train_tokens']:13,d} {d['val_tokens']:10,d} {d['bytes_per_token']:11.2f}  "
              f"{', '.join(d['sources'])}")

    print("\n== DOGRULAMA ==")
    v = verify(args.data, meta)
    print(f"  vocab {v['vocab_size']} (meta ile uyumlu: {v['vocab_matches_meta']})")
    print(f"  tum tokenler sozluk araliginda: {v['all_in_range']}")
    print(f"  alan-homojenlik yapisal (ayri .bin): {v['domain_homogeneous_by_construction']} "
          f"-> {len(v['domain_files'])} dosya")
    for dom, r in v["roundtrip"].items():
        print(f"  round-trip {dom:5s}: kayipsiz={r['lossless']} | {r['bytes_per_token']:.2f} B/tok "
              f"| {r['sample'][:78]!r}")
    print(f"  en sik parcalar:")
    for dom, tp in v["top_pieces_per_domain"].items():
        print(f"    {dom:5s}: " + " ".join(f"{p!r}" for p, _ in tp[:8]))
    print(f"  alanlar arasi ortusme (en sik 200 parca, Jaccard): {v['top200_jaccard']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"licensing": lic, "domains": meta["domains"], "verification": v,
                   "tier": meta.get("tier")}, f, indent=1, ensure_ascii=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
