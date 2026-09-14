"""Tokenizer katmanı: ortak arayüz arkasında SentencePiece (kendi eğittiğimiz) veya tiktoken (GPT-2).

Neden kendi tokenizer'ımızı eğitiyoruz (varsayılan)?
  * GPT-2 BPE'nin 50 257'lik sözlüğü tüm interneti kapsamak için tasarlandı. TinyStories'in dar
    kelime dağarcığında bu tokenların çoğu hiç kullanılmaz.
  * Sözlük boyutu doğrudan parametre ve bellek maliyetidir: embedding = V x d_model.
    V=50k, d=512 -> 25.7M parametre (30M'lık modelin neredeyse tamamı!). V=8192 -> 4.2M.
  * Çıkış logit tensörü (batch x seq x V) de V ile ölçeklenir; 12 GB VRAM'de bu önemli.
  Ticari/genel amaçlı bir modelde ise daha büyük (32k-160k) ve çok dilli bir sözlük gerekir;
  Kimi K2 ~160k token kullanır.

Neden BPE + byte_fallback?
  BPE sık karakter dizilerini birleştirerek alt-kelime birimleri öğrenir. byte_fallback, sözlükte
  olmayan herhangi bir karakteri UTF-8 byte tokenlarına (<0x0A> gibi) böler -> <unk> hiç oluşmaz,
  her metin kayıpsız kodlanıp geri çözülebilir.
"""
from __future__ import annotations

import json
import os


class Tokenizer:
    vocab_size: int
    bos_id: int  # hikâye başı: üretime bu tokenla başlarız
    eos_id: int  # hikâye sonu: üretimde bu token gelince dururuz

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_file: str):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.vocab_size = self.sp.get_piece_size()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        # C++ tarafında çok çekirdekli kodlama: Python döngüsünden onlarca kat hızlı
        return self.sp.encode(texts, num_threads=os.cpu_count())

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    @staticmethod
    def train(input_file: str, model_prefix: str, vocab_size: int) -> None:
        import sentencepiece as spm

        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            model_type="bpe",
            vocab_size=vocab_size,
            byte_fallback=True,              # <unk> yerine UTF-8 byte'ları
            character_coverage=0.9995,       # nadir karakterler sözlüğe girmez -> byte'a düşer
            split_digits=True,               # sayıları rakam rakam böl (aritmetik genelleme için iyi)
            normalization_rule_name="identity",  # metni NFKC vb. ile değiştirme: kayıpsız round-trip
            remove_extra_whitespaces=False,  # boşluk/satır yapısını koru
            allow_whitespace_only_pieces=True,
            unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
            input_sentence_size=2_000_000,
            shuffle_input_sentence=True,
            num_threads=os.cpu_count(),
            minloglevel=2,
        )


class TiktokenTokenizer(Tokenizer):
    """Hazır GPT-2 BPE. Eğitim gerektirmez; tek özel tokenı <|endoftext|> hem BOS hem EOS olur."""

    def __init__(self, name: str = "gpt2"):
        import tiktoken

        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = self.enc.n_vocab
        self.bos_id = self.eos_id = self.enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self.enc.encode_ordinary_batch(texts)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


def load_tokenizer(data_dir: str) -> Tokenizer:
    """prepare_data.py'nin yazdığı meta.json'a bakarak doğru tokenizer'ı kurar."""
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    if meta["tokenizer"] == "sentencepiece":
        return SentencePieceTokenizer(os.path.join(data_dir, meta["model_file"]))
    if meta["tokenizer"] == "tiktoken":
        return TiktokenTokenizer(meta["name"])
    raise ValueError(f"bilinmeyen tokenizer: {meta['tokenizer']}")
