"""Token veri yükleyici: diskteki uint16 token dizisinden rastgele pencereler örnekler.

Tasarım (nanoGPT tarzı):
  * prepare_data.py tüm hikâyeleri tokenlaştırıp tek, düz bir uint16 dizisine yazar:
        [BOS] hikâye1 [EOS] [BOS] hikâye2 [EOS] ...
    uint16 yeterli çünkü sözlük < 65 536; int64'e göre diskte/RAM'de 4 kat küçük.
  * np.memmap ile dosya RAM'e kopyalanmaz; işletim sistemi yalnızca okunan sayfaları yükler.
    Böylece GB'larca veri 32 GB RAM'i zorlamaz.
  * Her batch, dizideki rastgele konumlardan T+1 uzunluğunda pencereler: x = [i, i+T),
    y = [i+1, i+T+1). Yani model her konumda "bir sonraki token"ı tahmin eder.
  * Pencere hikâye sınırlarını aşabilir; model BOS/EOS tokenları sayesinde yeni hikâyenin
    başladığını öğrenir. (Daha titiz alternatif: belge maskeli attention - şimdilik gereksiz.)
"""
from __future__ import annotations

import os

import numpy as np
import torch


class TokenDataset:
    def __init__(self, path: str, block_size: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.device = device
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{path} çok kısa ({len(self.data)} token)")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(self, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        T = self.block_size
        ix = torch.randint(len(self.data) - T - 1, (batch_size,), generator=generator).tolist()
        chunk = np.stack([self.data[i : i + T + 1] for i in ix]).astype(np.int64)
        xy = torch.from_numpy(chunk)
        if self.device.startswith("cuda"):
            # pinned (sayfalanamaz) bellek + non_blocking: CPU->GPU kopyası GPU'yu bekletmeden,
            # asenkron yapılır; GPU önceki adımın hesabını sürdürürken veri yolda olur.
            xy = xy.pin_memory().to(self.device, non_blocking=True)
        else:
            xy = xy.to(self.device)
        return xy[:, :-1], xy[:, 1:]


def dataset_paths(data_dir: str) -> tuple[str, str]:
    return os.path.join(data_dir, "train.bin"), os.path.join(data_dir, "val.bin")
