"""Alan-homojen çok alanlı veri yükleyici (FOTON v0.2 kampanyası).

Neden ayrı bir sınıf: sequence-level router'a anlamlı soru sormak için **bir dizi tek alandan**
gelmeli. Aynı diziye kod + mahkeme kararı + matematik doldurmak router'a cevaplanamaz bir soru sorar.

Bu, alan başına ayrı `.bin` dosyasıyla **yapısal olarak** garanti ediliyor: pencere bir memmap'in
içinden alınıyor ve alanlar farklı memmap'lerde, yani bir pencere alan sınırını geçemez. Doküman
sınırları tokenize sırasında EOS ile işaretlendi (pencere doküman sınırını geçebilir; bu standart
ve her kol için aynı).

Alan etiketi modele VERİLMEZ: router alanı içerikten çıkarmak zorunda. Etiket yalnızca
teşhis ölçütleri (routing ↔ alan ilişkisi) ve alan başına val loss için tutulur.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch


class MultiDomainDataset:
    """Alan başına memmap; karışık batch üretir ama her DİZİ tek alandan gelir.

    `get_batch` geriye (x, y) döner — mevcut `src/train.py` ile uyumlu kalsın diye. Son batch'in
    alan kimlikleri `last_domain_ids` alanında durur (teşhis için).
    """

    def __init__(self, paths: dict[str, str], block_size: int, device: str,
                 weights: dict[str, float] | None = None):
        self.names = sorted(paths)
        self.data = {k: np.memmap(paths[k], dtype=np.uint16, mode="r") for k in self.names}
        self.block_size = block_size
        self.device = device
        for k, a in self.data.items():
            if len(a) < block_size + 1:
                raise ValueError(f"{k}: {paths[k]} çok kısa ({len(a)} token)")
        if weights is None:                       # varsayılan: token payına göre (korpus eşit kurulmuş)
            tot = sum(len(a) for a in self.data.values())
            w = [len(self.data[k]) / tot for k in self.names]
        else:
            s = sum(weights[k] for k in self.names)
            w = [weights[k] / s for k in self.names]
        self.weights = torch.tensor(w, dtype=torch.float)
        self.last_domain_ids: torch.Tensor | None = None

    def __len__(self) -> int:
        return sum(len(a) for a in self.data.values())

    def domain_id(self, name: str) -> int:
        return self.names.index(name)

    def _windows(self, dom: str, starts: list[int]) -> np.ndarray:
        a = self.data[dom]
        T = self.block_size
        return np.stack([a[i : i + T + 1] for i in starts])

    def get_batch(self, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        T = self.block_size
        # Alan seçimi ve pencere konumu AYNI generator'dan: sabit seed -> her koşuda aynı batch'ler.
        dom_idx = torch.multinomial(self.weights, batch_size, replacement=True, generator=generator)
        rows = np.empty((batch_size, T + 1), dtype=np.uint16)
        for di, name in enumerate(self.names):
            sel = (dom_idx == di).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            n = int(sel.numel())
            hi = len(self.data[name]) - T
            starts = torch.randint(hi, (n,), generator=generator).tolist()
            rows[sel.numpy()] = self._windows(name, starts)
        xy = torch.from_numpy(rows.astype(np.int64))
        if self.device.startswith("cuda"):
            xy = xy.pin_memory().to(self.device, non_blocking=True)
        else:
            xy = xy.to(self.device)
        self.last_domain_ids = dom_idx.to(xy.device)
        return xy[:, :-1], xy[:, 1:]

    # ---- batch_size'dan BAĞIMSIZ dizi akışı (ön-kayıt V1'in şartı) --------------------------
    #
    # Sorun: `get_batch` generator'ı `batch_size`'a bağlı bir düzende tüketiyor (multinomial
    # batch_size örnek çeker, sonra alan başına randint). Dolayısıyla aynı seed'le mikro-batch'i
    # 32'den 16'ya çevirmek pencerelerin SAYISINI değil KİMLİĞİNİ de değiştiriyordu: hem eğitim
    # akışı hem val penceresi kümesi kayıyordu. Alanlar arası 0.9 nat fark olduğu için val
    # penceresi kümesinin kayması ölçülen loss'u kaydırır, yani `L*` bölüşüme bağlı hale gelir.
    #
    # Çözüm: akışı DİZİ başına tanımla. `tokens_per_step` sabit olduğu için adım başına dizi
    # sayısı da sabittir (batch_size × grad_accum = 64), dolayısıyla dizi akışı bölüşümden
    # bağımsız olabilir. Akış, REFERANS bölüşümün (`REF_BATCH`) ürettiği sırayı birebir tekrar
    # oynatacak şekilde tanımlanıyor — böylece dondurulmuş baseline koşuları ve `L*` DEĞİŞMEZ,
    # ama başka bir bölüşüm aynı akışı görür.
    REF_BATCH = 16

    def _stream_chunk(self, n: int, generator: torch.Generator) -> list[tuple[str, int]]:
        """`get_batch(n, g)`'nin çekeceği (alan, başlangıç) çiftlerini AYNI tüketim düzeninde üretir."""
        T = self.block_size
        dom_idx = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        slot: dict[int, tuple[str, int]] = {}
        for di, name in enumerate(self.names):          # get_batch ile aynı döngü sırası
            sel = (dom_idx == di).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            hi = len(self.data[name]) - T
            starts = torch.randint(hi, (int(sel.numel()),), generator=generator).tolist()
            for j, s in zip(sel.tolist(), starts):
                slot[j] = (name, s)
        return [slot[j] for j in range(n)]

    def sequence_stream(self, count: int, generator: torch.Generator) -> list[tuple[str, int]]:
        """Bölüşümden bağımsız dizi akışından sıradaki `count` diziyi verir.

        `count`, `REF_BATCH`'in katı olmak zorundadır; kampanyada adım başına 64 dizi çekilir
        (64 = 4 × 16), eval'de 1280 (= 80 × 16).
        """
        if count % self.REF_BATCH:
            raise ValueError(f"count={count}, REF_BATCH={self.REF_BATCH}'in katı olmalı")
        out: list[tuple[str, int]] = []
        for _ in range(count // self.REF_BATCH):
            out.extend(self._stream_chunk(self.REF_BATCH, generator))
        return out

    def windows(self, pairs: list[tuple[str, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Açıkça verilen (alan, başlangıç) çiftlerinden batch kurar — akışın tüketim ucu."""
        T = self.block_size
        rows = np.stack([self.data[d][i : i + T + 1] for d, i in pairs])
        xy = torch.from_numpy(rows.astype(np.int64))
        xy = xy.pin_memory().to(self.device, non_blocking=True) if self.device.startswith("cuda") \
            else xy.to(self.device)
        self.last_domain_ids = torch.tensor([self.domain_id(d) for d, _ in pairs], device=xy.device)
        return xy[:, :-1], xy[:, 1:]

    def frozen_val_windows(self, total: int = 1280, seed: int = 12345) -> list[tuple[str, int]]:
        """Val penceresi kümesi: sabit, bölüşümden bağımsız, `total` pencere (ön-kayıt V1-ii)."""
        return self.sequence_stream(total, torch.Generator().manual_seed(seed))

    def get_domain_batch(self, domain: str, batch_size: int,
                         generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Tek alandan batch: alan başına val loss ve teşhis ölçütleri için."""
        T = self.block_size
        hi = len(self.data[domain]) - T
        starts = torch.randint(hi, (batch_size,), generator=generator).tolist()
        xy = torch.from_numpy(self._windows(domain, starts).astype(np.int64))
        xy = xy.pin_memory().to(self.device, non_blocking=True) if self.device.startswith("cuda") \
            else xy.to(self.device)
        self.last_domain_ids = torch.full((batch_size,), self.domain_id(domain), device=xy.device)
        return xy[:, :-1], xy[:, 1:]


def multi_domain_paths(data_dir: str, split: str) -> dict[str, str]:
    """meta.json'daki alanlar için {alan: yol}. Dosya adı düzeni: <alan>_<split>.bin"""
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    out = {}
    for dom in meta["domains"]:
        p = os.path.join(data_dir, f"{dom}_{split}.bin")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        out[dom] = p
    return out
