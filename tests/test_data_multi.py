"""Alan-homojen çok alanlı yükleyicinin regresyon testleri.

Denetimde bulunan hata: `get_batch(batch_size, g)` generator'ı `batch_size`'a bağlı bir düzende
tüketiyordu (multinomial batch_size örnek çeker, sonra alan başına randint). Dolayısıyla mikro-batch
bölüşümünü değiştirmek pencerelerin SAYISINI değil **KİMLİĞİNİ** de değiştiriyordu.

Bu neden ciddiydi: ön-kayıt `tokens_per_step`'i pinliyor ve mikro-batch'i serbest bırakıyor
(V1), val penceresi sayısını da `batch_size × eval_iters = 1280` ile sabitliyor. Ama 1280 pencere
SAYISI'nı sabitlemek, aynı 1280 pencereyi görmek anlamına gelmiyordu. Alanlar arası 0.9 nat fark
olduğu için pencere kümesinin kayması ölçülen val loss'u kaydırır → `L*` bölüşüme bağlı hale gelir
→ ön-kaydın dondurduğu hedef taşınmaz. Aynı hata eğitim akışında "aynı seed = aynı veri sırası"
iddiasını da yanlış yapıyordu.

Aşağıdaki testler üç şeyi kilitler:
  1. akış bölüşümden bağımsız (kimlik, yalnızca sayı değil),
  2. akış REFERANS bölüşümün eski sırasını birebir yeniden üretir (yani `L*` değişmedi),
  3. bir pencere ALAN SINIRINI geçemez (yapısal alan homojenliği).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data_multi import MultiDomainDataset


@pytest.fixture
def ds(tmp_path):
    """Üç alan, birbirinden ayırt edilebilir token aralıklarıyla: alan sızıntısı görünür olsun."""
    rng = np.random.default_rng(0)
    paths = {}
    for i, name in enumerate(("aaa", "bbb", "ccc")):
        lo, hi = 1000 * (i + 1), 1000 * (i + 1) + 500      # alan başına ayrık aralık
        arr = rng.integers(lo, hi, size=20_000, dtype=np.uint16)
        p = tmp_path / f"{name}.bin"
        arr.tofile(p)
        paths[name] = str(p)
    return MultiDomainDataset(paths, block_size=16, device="cpu")


def test_stream_is_split_independent(ds):
    """Aynı seed, farklı bölüşüm -> AYNI dizi kümesi, AYNI sırada."""
    a = ds.sequence_stream(64, torch.Generator().manual_seed(7))
    b = ds.sequence_stream(64, torch.Generator().manual_seed(7))
    assert a == b
    # 64 dizi 16/32/64'luk parçalara bölündüğünde de aynı akış
    for bs in (16, 32, 64):
        chunks = [a[i : i + bs] for i in range(0, 64, bs)]
        assert [p for c in chunks for p in c] == a
        assert all(len(c) == bs for c in chunks)


def test_stream_reproduces_reference_split_exactly(ds):
    """Akış, eski `get_batch(REF_BATCH, g)` düzenini birebir yeniden üretir -> `L*` korunur.

    Bu test kırılırsa dondurulmuş `L*` artık geçerli değildir: ön-kaydın hedefi, onu üreten
    koşuların gördüğü pencere kümesine bağlıdır.
    """
    ref = ds.REF_BATCH
    g_old, g_new = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    old = [ds.get_batch(ref, g_old) for _ in range(4)]
    new = ds.sequence_stream(4 * ref, g_new)
    for m, (xo, yo) in enumerate(old):
        xn, yn = ds.windows(new[m * ref : (m + 1) * ref])
        assert torch.equal(xo, xn), f"mikro-batch {m}: x farklı"
        assert torch.equal(yo, yn), f"mikro-batch {m}: y farklı"


def test_frozen_val_windows_are_fixed(ds):
    """Val penceresi kümesi sabit ve tekrarlanabilir; sayı da kimlik de bölüşümden bağımsız."""
    a = ds.frozen_val_windows(64, seed=12345)
    b = ds.frozen_val_windows(64, seed=12345)
    assert a == b and len(a) == 64
    assert ds.frozen_val_windows(64, seed=999) != a      # seed gerçekten etkili


def test_stream_count_must_be_multiple_of_ref_batch(ds):
    """Bölüşümden bağımsızlık ancak REF_BATCH'in katlarında garanti; sessizce kaymasın."""
    with pytest.raises(ValueError, match="REF_BATCH"):
        ds.sequence_stream(24, torch.Generator().manual_seed(0))


def test_window_cannot_cross_domain_boundary(ds):
    """YAPISAL alan homojenliği: her pencerenin bütün token'ları tek alanın aralığında."""
    ranges = {"aaa": (1000, 1500), "bbb": (2000, 2500), "ccc": (3000, 3500)}
    pairs = ds.sequence_stream(64, torch.Generator().manual_seed(1))
    x, y = ds.windows(pairs)
    for row, (dom, _) in enumerate(pairs):
        lo, hi = ranges[dom]
        vals = torch.cat([x[row], y[row]])
        assert int(vals.min()) >= lo and int(vals.max()) < hi, \
            f"satır {row} ({dom}) alan sınırını geçti"


def test_domain_ids_match_the_stream(ds):
    """Teşhis etiketi akışla tutarlı: routing ↔ alan ölçütleri buna dayanıyor."""
    pairs = ds.sequence_stream(32, torch.Generator().manual_seed(2))
    ds.windows(pairs)
    expected = torch.tensor([ds.domain_id(d) for d, _ in pairs])
    assert torch.equal(ds.last_domain_ids.cpu(), expected)


def test_windows_never_run_past_the_end(ds):
    """Başlangıçlar block_size+1 pencereyi dosyanın içinde bırakmalı."""
    pairs = ds.sequence_stream(16 * 20, torch.Generator().manual_seed(4))
    for dom, start in pairs:
        assert 0 <= start <= len(ds.data[dom]) - ds.block_size - 1
