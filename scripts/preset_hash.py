"""Kampanya preset'lerinin SHA-256'sı: karar koşularından ÖNCE yazılır, koşulardan önce ve sonra doğrulanır.

    .venv\\Scripts\\python scripts/preset_hash.py --write      # experiments/foton-v02/preset_hashes.json
    .venv\\Scripts\\python scripts/preset_hash.py --verify     # çıkış kodu 0 = preset'ler yazıldığından beri değişmedi

Ön-kayıt V10: "FOTON'un config'i `src/config.py`'de preset olarak sabitlenir ve SHA-256'sı bu bölüme yazılır."
NEYİN hash'lendiği ön-kayıtta tanımlı değildi; burada sabitlendi ("Sapmalar"da da yazılı):

  * preset başına: `{"model": asdict(ModelConfig), "train": asdict(TrainConfig)}` sözlüğünün KANONİK JSON'u
    (anahtarlar sıralı, boşluksuz, UTF-8), `vocab_size` tokenizer'ın gerçek değeriyle (32768) çözülmüş.
    `src/config.py` DOSYASININ hash'i kullanılamaz: bir yorum satırı ya da sonradan eklenen bir preset onu
    değiştirir. Bir koşunun `config.json`'u da kullanılamaz: `seed`, `run_name`, `out_dir` koşudan koşuya değişir.
  * toplam: preset adı -> hash eşlemesinin kanonik JSON'unun SHA-256'sı. V10'daki yer tutucuya yazılan budur.

Dürüstlük notu (ön-kayıttaki ile aynı): bu hash, config'in hash'lendikten SONRA değişmediğini kanıtlar; H'nin
sonucunun onu etkilemediğini kanıtlamaz. O güvence donmuş spesifikasyonun config'i türetilebilir kılmasındadır.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.config import PRESETS  # noqa: E402

OUT = "experiments/foton-v02/preset_hashes.json"
VOCAB = 32768                                  # data/foton tokenizer'ı (ön-kayıt V2); meta.json ile doğrulanır
CAMPAIGN = ("foton-d", "foton-m", "foton-h", "foton-x", "foton-d-s1", "foton-m-s1", "foton-h-s1", "foton-x-s1")


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def preset_hash(name: str) -> str:
    mcfg, tcfg = PRESETS[name]
    return hashlib.sha256(canonical({"model": asdict(replace(mcfg, vocab_size=VOCAB)), "train": asdict(tcfg)})).hexdigest()


def compute() -> dict:
    per = {name: preset_hash(name) for name in CAMPAIGN}
    return {"schema": 1, "vocab_size": VOCAB, "presets": per,
            "total": hashlib.sha256(canonical(per)).hexdigest()}


def verify(path: str) -> list[str]:
    if not os.path.exists(path):
        return [f"{path} yok: preset hash'leri henüz yazılmamış"]
    with open(path, encoding="utf-8") as f:
        want = json.load(f)
    got = compute()
    bad = [f"{k}: yazılı {want['presets'].get(k, '—')[:12]} != şimdiki {v[:12]}" for k, v in got["presets"].items()
           if want["presets"].get(k) != v]
    bad += [f"{k}: dosyada var, kampanya listesinde yok" for k in want["presets"] if k not in got["presets"]]
    if not bad and want.get("total") != got["total"]:
        bad.append("toplam hash uyuşmuyor")
    return bad


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--path", default=os.path.join(ROOT, OUT))
    args = ap.parse_args()
    meta = os.path.join(ROOT, "data", "foton", "meta.json")
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as f:
            v = json.load(f).get("vocab_size")
        if v != VOCAB:
            sys.exit(f"data/foton/meta.json vocab_size = {v}, beklenen {VOCAB}")
    if args.verify:
        bad = verify(args.path)
        for b in bad:
            print("✗", b)
        print("SONUÇ:", "PRESET'LER DEĞİŞMİŞ" if bad else f"preset'ler yazılı hash'lerle aynı (toplam {compute()['total'][:16]}…)")
        sys.exit(1 if bad else 0)
    res = compute()
    for k, v in res["presets"].items():
        print(f"{v}  {k}")
    print(f"{res['total']}  TOPLAM")
    if args.write:
        if os.path.exists(args.path):
            sys.exit(f"{args.path} zaten var: preset hash'leri bir kez yazılır (değişiklik sapmadır, elle silinmeli)")
        os.makedirs(os.path.dirname(args.path), exist_ok=True)
        with open(args.path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print("->", args.path)


if __name__ == "__main__":
    main()
