"""TinyStories metnini tokenlaştırıp eğitime hazır uint16 dosyalara çevirir.

    .venv\\Scripts\\python scripts/prepare_data.py                          # SentencePiece BPE 8k (varsayılan)
    .venv\\Scripts\\python scripts/prepare_data.py --tokenizer gpt2 --out data/gpt2

Adımlar:
  1. (sp) Eğitim metninin ilk --tok_train_mb MB'ı üzerinde BPE tokenizer eğit.
     Tokenizer'ı *val* verisiyle eğitmiyoruz ki değerlendirme verisi hiçbir aşamada sızmasın.
  2. Her hikâyeyi [BOS] + tokenlar + [EOS] olarak kodla, hepsini uç uca train.bin / val.bin'e yaz.
  3. meta.json: tokenizer türü, sözlük boyutu, özel token id'leri, token sayıları.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator

import numpy as np
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.tokenizer import SentencePieceTokenizer, TiktokenTokenizer, Tokenizer  # noqa: E402

DELIM = b"<|endoftext|>"  # TinyStories dosyalarında hikâye ayırıcı


def iter_stories(path: str, max_bytes: int | None = None, pbar: tqdm | None = None) -> Iterator[str]:
    """Dosyayı 16 MB'lık parçalarla okuyup hikâyelere böler; 2.2 GB'ı tek seferde RAM'e almaz.
    Ayırıcı saf ASCII olduğundan byte seviyesinde bölmek UTF-8 karakterlerini asla ortadan kesmez."""
    read = 0
    buf = b""
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            read += len(chunk)
            if pbar is not None:
                pbar.update(len(chunk))
            parts = (buf + chunk).split(DELIM)
            buf = parts.pop()  # son parça yarım olabilir, bir sonraki okumayla birleşir
            for s in parts:
                s = s.decode("utf-8", errors="replace").strip()
                if s:
                    yield s
            if max_bytes is not None and read >= max_bytes:
                return
    s = buf.decode("utf-8", errors="replace").strip()
    if s:
        yield s


def encode_file(tok: Tokenizer, src: str, dst: str, max_bytes: int | None, batch: int = 20_000) -> int:
    """Hikâyeleri batch'ler halinde çok çekirdekli kodlayıp dst'ye ekler. Toplam token sayısını döndürür."""
    assert tok.vocab_size < 2**16, "uint16 için sözlük 65 536'dan küçük olmalı"
    total = 0
    size = min(os.path.getsize(src), max_bytes or float("inf"))
    with open(dst + ".tmp", "wb") as out, tqdm(total=size, unit="B", unit_scale=True, desc=os.path.basename(dst)) as pbar:
        stories: list[str] = []

        def flush() -> None:
            nonlocal total
            flat: list[int] = []
            for ids in tok.encode_batch(stories):
                flat.append(tok.bos_id)
                flat.extend(ids)
                if tok.eos_id != tok.bos_id:
                    flat.append(tok.eos_id)
            arr = np.asarray(flat, dtype=np.uint16)
            arr.tofile(out)
            total += arr.size
            stories.clear()

        for s in iter_stories(src, max_bytes, pbar):
            stories.append(s)
            if len(stories) >= batch:
                flush()
        if stories:
            flush()
    os.replace(dst + ".tmp", dst)
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", choices=["sp", "gpt2"], default="sp")
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--out", default=os.path.join(ROOT, "data", "sp8k"))
    p.add_argument("--raw_dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--tok_train_mb", type=int, default=100, help="tokenizer eğitimi için kullanılacak metin (MB)")
    p.add_argument("--max_train_mb", type=int, default=None, help="train.bin'i kısaltmak için (varsayılan: tümü)")
    args = p.parse_args()

    train_txt = os.path.join(args.raw_dir, "TinyStoriesV2-GPT4-train.txt")
    val_txt = os.path.join(args.raw_dir, "TinyStoriesV2-GPT4-valid.txt")
    os.makedirs(args.out, exist_ok=True)

    if args.tokenizer == "sp":
        prefix = os.path.join(args.out, "tokenizer")
        if not os.path.exists(prefix + ".model"):
            # SentencePiece satır satır "cümle" bekler; hikâyeleri paragraflarına bölüp veriyoruz.
            # Satır sonu karakteri eğitimde görülmediği için kodlamada byte_fallback ile tek bir
            # <0x0A> tokenına düşer (kayıpsız; Llama-2 tokenizer'ı da aynı şekilde davranır).
            # Ölçüm (val seti): 8k sözlükle hikâye başına ~197 token = GPT-2'nin 50k sözlüğüyle aynı.
            sample = os.path.join(args.out, "tok_sample.txt")
            with open(sample, "w", encoding="utf-8") as f:
                for story in iter_stories(train_txt, args.tok_train_mb << 20):
                    for line in story.splitlines():
                        if line.strip():
                            f.write(line + "\n")
            print(f"tokenizer eğitiliyor (BPE, V={args.vocab_size}) ...")
            SentencePieceTokenizer.train(sample, prefix, args.vocab_size)
        tok: Tokenizer = SentencePieceTokenizer(prefix + ".model")
        meta = {"tokenizer": "sentencepiece", "model_file": "tokenizer.model"}
    else:
        tok = TiktokenTokenizer("gpt2")
        meta = {"tokenizer": "tiktoken", "name": "gpt2"}

    # Kayıpsızlık kontrolü: encode -> decode aynı metni vermeli
    probe = next(iter_stories(val_txt))
    assert tok.decode(tok.encode(probe)) == probe, "tokenizer round-trip başarısız!"

    val_tokens = encode_file(tok, val_txt, os.path.join(args.out, "val.bin"), None)
    max_bytes = args.max_train_mb << 20 if args.max_train_mb else None
    train_tokens = encode_file(tok, train_txt, os.path.join(args.out, "train.bin"), max_bytes)

    meta.update(vocab_size=tok.vocab_size, bos_id=tok.bos_id, eos_id=tok.eos_id,
                train_tokens=train_tokens, val_tokens=val_tokens)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
