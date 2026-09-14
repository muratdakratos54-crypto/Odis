"""Checkpoint kaydetme/yükleme.

İki tür dosya yazıyoruz:
  * last.pt : model + optimizer + adım + RNG durumları -> eğitimi kaldığı yerden sürdürmek için.
              (AdamW, parametre başına 2 moment tutar: optimizer durumu modelin ~2 katı yer kaplar.)
  * best.pt : yalnızca en iyi val loss'lu model ağırlıkları + config -> örnek üretim/analiz için.
Config'ler checkpoint'in içinde olduğundan, model mimarisi dosyadan birebir yeniden kurulabilir.
"""
from __future__ import annotations

import os
import time

import torch

from src.config import ModelConfig
from src.model import GPT


def save_atomic(obj: dict, path: str) -> None:
    """Önce geçici dosyaya yaz, sonra yerine taşı: yazma sırasında çökme/elektrik kesintisi olursa
    eski checkpoint bozulmadan kalır. fsync, işletim sisteminin önbelleğindeki verinin diske
    gerçekten yazılmasını bekler; o olmadan elektrik kesintisinde yeni dosya yarım kalabilir."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: dosya başka bir süreçte (ör. sample.py) açıksa
            time.sleep(1.0 + attempt)
    print(f"[uyarı] {path} güncellenemedi (dosya kilitli?); yeni checkpoint {tmp} içinde duruyor")


def load_model(path: str, device: str) -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt
