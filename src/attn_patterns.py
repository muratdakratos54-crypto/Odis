"""FOTON v0.2 attention desenlerinin bağlantı aritmetiği — tek doğruluk kaynağı.

Neden ayrı bir modül: ön-kayıt ilk halinde "A/B/C'nin üçü de token başına 256 attention bağlantısı
görür, yani hesap eşitlendi" diyordu. Denetim bu cümlenin T=512 ve nedensel maske altında
**aritmetik olarak yanlış** olduğunu gösterdi: i konumundaki token yalnızca j <= i'ye bakabildiği
için desenlerin tavanı birbirinden farklıdır.

  desen                                   token/head başına ortalama nedensel bağlantı
  tam nedensel (D ve M baseline'ları)     256.50
  yerel pencere 256                       192.25
  dilated r=2, bütçe 256                  128.50   <- TAVAN; pencere büyütmek bunu aşamaz
  128 yerel + 128 strided s=4 (head-split) 88.31

Yani eşitleme hedefi 256'da tutulamazdı: dilated modülün tavanı 128.50 ve bu tavan pencereyle
değil ORANLA belirleniyor (r=2 ile geçmişin yarısı zaten erişilemez). Hedef bu yüzden dilated
desenin tavanına indirildi ve tasarım şu haliyle donduruldu:

  A: yerel, W=150                                       -> 128.17
  B: dilated, r=2, bütçe W=256                          -> 128.50
  C: head'ler ikiye bölünür: yarısı yerel W=256,
     yarısı strided s=4 W=256                           -> 128.38 (head ortalaması)

Bu modül olmasa ne bozulur: bu sayılar ön-kayıtta elle yazılmış metin olarak kalır ve bir parametre
(W, r, s, T) değiştiğinde sessizce yanlışa döner — denetimin bulduğu hatanın tam olarak kendisi.
Burada her sayı İKİ BAĞIMSIZ YOLDAN hesaplanır (kapalı form + gerçekleşen boolean maskenin satır
toplamı) ve ikisinin uyuşması `_verify_mask` ile zorunlu tutulur. Uyuşma modülün varlık sebebidir:
analitik formül ile eğitimde kullanılacak maske ayrışırsa rapor edilen "eşit hesap" iddiası çöker.

`headline()` sonuç tablosunun BAŞINA konacak satırı üretir: FOTON modülleri baseline'ların YARISI
kadar bağlantı görüyor ve kalan bu sapma FOTON'un ALEYHİNE — dipnot değil, başlık.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T_KAMPANYA = 512        # ön-kayıt `max_seq_len`; bütün başlık sayıları bu T'de
N_HEAD_KAMPANYA = 8     # `src/config.py` n_head=8; C'nin yarı-yarıya bölünmesi bu sayıda TAM

# Dondurulmuş tasarım parametreleri. Ön-kayıt bunları pinliyor; buradan değiştirmek
# `tests/test_attn_patterns.py`'deki eşitleme testini düşürür (kasıtlı: sessiz değişim olmasın).
A_LOCAL_W = 150
B_DILATION_R, B_BUDGET_W = 2, 256
C_LOCAL_W = 256
C_STRIDE_S, C_STRIDE_W = 4, 256

# A/B/C arası kabul edilen kalan eşitsizlik. Gerçekleşen sapma 0.33 bağlantı; tolerans 0.02 pay
# ile 0.35 — pay, testin kayan nokta gürültüsüne değil yalnızca parametre değişimine düşmesi için.
EQ_TOL = 0.35


# --------------------------------------------------------------------------------------------
# Tek head'in deseni
#
# Üç desen ailesinin (yerel / dilated / strided) hepsi aynı nesne: **aritmetik ofset ailesi**
# {step*k : 0 <= k < budget}, nedensel kesme ile (j = i - ofset, j >= 0). Literatürdeki "dilated
# sliding window" ile "strided/fixed attention" ofset kümesi olarak AYNI şeydir; ayrım desenin
# neyle birleştirildiğinden gelir (B tek başına, C yerel band ile yan yana). İki adı da
# koruyoruz çünkü ön-kaydın sözlüğü bu; aritmetiği tek yerde tutuyoruz çünkü hata orada çıktı.
#
# `budget` = SEÇİLEN POZİSYON SAYISI (bağlantı bütçesi), açıklık DEĞİL. Bu ayrım kritik:
# ön-kayıt B'yi "256 pozisyon, seyrekleştirilmiş: aynı bütçeyle daha uzak geçmiş" diye tanımlıyor.
# W açıklık okunursa (r=2 ile 256 pozisyonluk şeride her ikinci konum) B **96.25** verir, 128.50
# değil. Yanlış okuma eşitleme hedefini 32 bağlantı kaydırır, o yüzden yorum burada sabitlenmiştir.
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Head:
    """Tek head'in nedensel ofset deseni: step adımlı, budget pozisyonluk aritmetik aile."""

    step: int
    budget: int | None      # None = sınırsız (tam nedensel); T'de doygunluğa kendisi oturur
    label: str

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError(f"step >= 1 olmalı, {self.step} verildi")
        if self.budget is not None and self.budget < 1:
            raise ValueError(f"budget >= 1 olmalı, {self.budget} verildi")

    def effective_budget(self, T: int) -> int:
        """T'de gerçekten kullanılabilen bütçe = min(budget, ceil(T/step)).

        Doygunluk noktası budur ve tavanın kaynağı: budget ceil(T/step)'i geçtikten sonra
        pencereyi büyütmek HİÇBİR bağlantı eklemez (r=2, T=512 -> tavan 256 pozisyon = 128.50).
        """
        saturated = -(-T // self.step)          # ceil(T/step), tam sayı aritmetiğiyle
        return saturated if self.budget is None else min(self.budget, saturated)

    def offsets(self, T: int) -> np.ndarray:
        """Desenin T'de kullandığı geri ofsetler (0 = kendisi)."""
        return np.arange(self.effective_budget(T), dtype=np.int64) * self.step

    def counts(self, T: int) -> np.ndarray:
        """Konum başına bağlantı sayısı, KAPALI FORM: min(i // step + 1, budget).

        Maskeden türetilmiyor — maskenin doğrulanacağı bağımsız referans bu.
        """
        i = np.arange(T, dtype=np.int64)
        return np.minimum(i // self.step + 1, self.effective_budget(T))

    def avg(self, T: int) -> float:
        """Konumlar üzerinden ortalama bağlantı, KAPALI FORM (counts toplamından bağımsız).

        Üçüncü bağımsız yol olarak duruyor: `self_check` bunu counts toplamıyla ve maske satır
        toplamıyla karşılaştırır. Tavan iddiası ("r=2 128.50'yi aşamaz") böylece tarama değil
        formül sonucudur: B doygunluğa oturduktan sonra ortalama sabittir.
        """
        B = self.effective_budget(T)
        n = min(T, self.step * B)               # bu i'ye kadar sayım artar, sonrası sabit B
        q, r = divmod(n, self.step)
        total = self.step * q * (q + 1) // 2 + r * (q + 1)
        total += (T - n) * B
        return total / T

    def max_span(self, T: int) -> int:
        """Gerçekleşen en uzak geri erişim (ofset cinsinden). Ön-kaydın 'etkili alım alanı' ekseni."""
        return self.step * (self.effective_budget(T) - 1)

    def distinct_reachable(self, T: int) -> int:
        """Son konumdaki sorgunun eriştiği AYRIK pozisyon sayısı = doygun bütçe.

        `max_span` ile birlikte okunur: B 510 ofset geriye uzanıp 256 pozisyon görür (yarım
        yoğunluk), A 149 geriye uzanıp 150 pozisyon görür (tam yoğunluk). Ön-kaydın
        'eşitlenemeyen eksen 2'si tam olarak bu fark.
        """
        return self.effective_budget(T)

    def mask(self, T: int) -> np.ndarray:
        """[T, T] boolean maske (True = izinli), attention maskesi olarak kullanılabilir biçimde.

        Kurulduğu anda satır toplamları kapalı form `counts` ile karşılaştırılır.
        """
        i = np.arange(T, dtype=np.int64)[:, None]
        j = np.arange(T, dtype=np.int64)[None, :]
        off = i - j
        m = (off >= 0) & (off % self.step == 0) & (off // self.step < self.effective_budget(T))
        _verify_mask(m, self.counts(T), f"{self.label} (T={T})")
        return m


def _verify_mask(mask: np.ndarray, counts: np.ndarray, who: str) -> None:
    """Maske ile kapalı formun uyuşmasını ZORUNLU kılar.

    Çıplak `assert` değil: `python -O` assert'leri atar ve bu kontrolün atlanması modülün tek
    işini (iki yolun uyuşması) sessizce ortadan kaldırır. Nedensellik de burada kontrol edilir,
    çünkü j > i sızıntısı satır toplamını da bozar ama önce ADIYLA görülmesi gerekir.
    """
    T = mask.shape[0]
    upper = np.triu(np.ones((T, T), dtype=bool), k=1)
    if bool((mask & upper).any()):
        raise AssertionError(f"{who}: maske nedensel değil, j > i girdisi var")
    rows = mask.sum(axis=1)
    if not np.array_equal(rows, counts):
        bad = int(np.flatnonzero(rows != counts)[0])
        raise AssertionError(
            f"{who}: maske satır toplamı kapalı formla uyuşmuyor "
            f"(i={bad}: maske {int(rows[bad])}, formül {int(counts[bad])})")


# --- parametreli kurucular -------------------------------------------------------------------

def full() -> "Pattern":
    """Tam nedensel attention: D ve M baseline'larının gördüğü desen (T=512'de 256.50)."""
    return Pattern("full", (Head(1, None, "tam nedensel"),),
                   "D ve M baseline'ları: kısıt yok")


def local(W: int) -> "Pattern":
    """Bitişik yerel pencere: son W pozisyon."""
    return Pattern(f"local(W={W})", (Head(1, W, f"yerel W={W}"),), "bitişik pencere")


def dilated(r: int, W: int) -> "Pattern":
    """Dilated: r adımla seçilmiş W pozisyon (açıklık r*(W-1)+1, bağlantı bütçesi W)."""
    return Pattern(f"dilated(r={r},W={W})", (Head(r, W, f"dilated r={r} W={W}"),),
                   "aynı bütçeyle daha uzak geçmiş")


def strided(s: int, W: int) -> "Pattern":
    """Strided: s adımla seçilmiş W pozisyon. Ofset ailesi olarak `dilated` ile aynı (bkz. yorum)."""
    return Pattern(f"strided(s={s},W={W})", (Head(s, W, f"strided s={s} W={W}"),),
                   "Sparse Transformer'ın strided bileşeni")


def head_split(name: str, *heads: Head, note: str = "") -> "Pattern":
    """Head'leri eşit parçalara bölen heterojen desen (C böyle kurulur)."""
    return Pattern(name, tuple(heads), note or "head'ler desene bölünür")


@dataclass(frozen=True)
class Pattern:
    """Bir modülün attention deseni: head grupları + raporlanacak türev ölçüler.

    `heads` uzunluğu 1 ise bütün head'ler aynı desene sahiptir. C'de 2'dir ve head'ler EŞİT
    parçalara bölünür — bu, ön-kaydın "eşitlenemeyen eksen 5"i: A ve B head düzeyinde homojen,
    C değil. BİLEREK böyle; head içi heterojenlik C'nin tanımının parçası, o yüzden eşitleme
    iddiası head ORTALAMASI üzerinden kurulur ve head başına değerler ayrıca raporlanır.
    """

    name: str
    heads: tuple[Head, ...]
    note: str = ""

    def head_groups(self, n_heads: int = N_HEAD_KAMPANYA) -> list[tuple[Head, int]]:
        """(head deseni, kaç head) listesi. Tam bölünmezse hata verir.

        Neden hata: C'nin "yarısı yerel, yarısı strided" iddiası n_head bölünmediğinde YANLIŞ
        olur (7 head'de 4/3 = 0.5 değil) ve head ortalaması sessizce kayar. n_head=8'de tamdır.
        """
        k = len(self.heads)
        if n_heads % k:
            raise ValueError(f"{self.name}: n_heads={n_heads}, {k} desene eşit bölünemiyor")
        return [(h, n_heads // k) for h in self.heads]

    def head_avgs(self, T: int = T_KAMPANYA) -> list[float]:
        """Head deseni başına ortalama bağlantı — C'nin heterojenliği burada görünür."""
        return [h.avg(T) for h in self.heads]

    def avg(self, T: int = T_KAMPANYA, n_heads: int = N_HEAD_KAMPANYA) -> float:
        """Head'ler üzerinden ağırlıklı ortalama bağlantı: modülün eşitleme iddiasındaki sayı."""
        groups = self.head_groups(n_heads)
        return sum(h.avg(T) * c for h, c in groups) / n_heads

    def counts(self, T: int = T_KAMPANYA) -> list[np.ndarray]:
        """Head deseni başına konum-bazlı bağlantı sayıları (kapalı form)."""
        return [h.counts(T) for h in self.heads]

    def masks(self, T: int = T_KAMPANYA, n_heads: int = N_HEAD_KAMPANYA) -> np.ndarray:
        """[n_heads, T, T] boolean maske; head'ler `head_groups` sırasında tekrarlanır."""
        blocks = [np.broadcast_to(h.mask(T), (c, T, T)) for h, c in self.head_groups(n_heads)]
        return np.concatenate(blocks, axis=0)

    def summary(self, T: int = T_KAMPANYA, n_heads: int = N_HEAD_KAMPANYA) -> dict:
        """Rapor satırı: ortalama, head başına ortalama, en uzak erişim, ayrık pozisyon sayısı.

        `max_span` head'ler üzerinden MAKSİMUMdur ve heterojen desende tek başına yanıltır:
        C'nin strided yarısı 508 geriye uzanırken yerel yarısı yalnızca 255'e uzanır, yani tek
        sayı C'nin alım alanını OLDUĞUNDAN GENİŞ gösterir — FOTON'un LEHİNE sapan bir özet.
        `head_max_spans` bu yüzden head başına ayrı döner ve tabloda ayrı satır olarak basılır.
        """
        return {"name": self.name, "note": self.note, "T": T, "n_heads": n_heads,
                "avg": self.avg(T, n_heads), "head_avgs": self.head_avgs(T),
                "head_labels": [h.label for h in self.heads],
                "max_span": max(h.max_span(T) for h in self.heads),
                "head_max_spans": [h.max_span(T) for h in self.heads],
                "distinct_reachable": [h.distinct_reachable(T) for h in self.heads]}


# --- kayıt: ön-kayıtta donmuş dört desen -----------------------------------------------------
PATTERNS: dict[str, Pattern] = {
    "full": full(),
    "A": local(A_LOCAL_W),
    "B": dilated(B_DILATION_R, B_BUDGET_W),
    "C": head_split("C",
                    Head(1, C_LOCAL_W, f"yerel W={C_LOCAL_W}"),
                    Head(C_STRIDE_S, C_STRIDE_W, f"strided s={C_STRIDE_S} W={C_STRIDE_W}"),
                    note="head'ler ikiye bölünür (bilerek head düzeyinde heterojen)"),
}
FOTON_MODULES = ("A", "B", "C")     # X kolunun modülleri; H'de üçü de aynı desen


def per_head_avg(name: str = "C", T: int = T_KAMPANYA) -> list[float]:
    """Adlandırılmış desenin head başına ortalaması. C için iki değer döner (192.25 / 64.50)."""
    return PATTERNS[name].head_avgs(T)


def equalization(T: int = T_KAMPANYA, n_heads: int = N_HEAD_KAMPANYA) -> dict:
    """A/B/C arası KALAN eşitsizlik: ön-kaydın "eşitlendi" dediği eksenin gerçek hâli."""
    per = {k: PATTERNS[k].avg(T, n_heads) for k in FOTON_MODULES}
    vals = list(per.values())
    spread = max(vals) - min(vals)
    mean = sum(vals) / len(vals)
    return {"per_module": per, "mean": mean, "spread": spread,
            "spread_pct": 100.0 * spread / mean, "tol": EQ_TOL, "within_tol": spread <= EQ_TOL}


def baseline_gap(T: int = T_KAMPANYA, n_heads: int = N_HEAD_KAMPANYA) -> dict:
    """FOTON modülleri ile D/M baseline'ları arasındaki bağlantı açığı — YÖNÜYLE birlikte.

    Yön önemli: açık FOTON'un ALEYHİNE. `favors` alanı bunu makine-okunur tutar ki rapor
    üreteci yönü ters yazamasın.
    """
    foton = equalization(T, n_heads)["mean"]
    base = PATTERNS["full"].avg(T, n_heads)
    return {"foton_mean": foton, "baseline": base, "ratio": foton / base,
            "deficit": base - foton, "favors": "baseline"}


def headline(T: int = T_KAMPANYA, n_heads: int = N_HEAD_KAMPANYA) -> str:
    """Sonuç tablosunun BAŞINA konacak satır(lar) — dipnota değil.

    Neden zorunlu: FOTON modülleri baseline'ların yarısı kadar bağlantı görüyor. X kazanırsa
    sonuç daha güçlü; kaybederse okuyan kişi yarım bağlantıyla yarıştığını bilmek zorunda.
    """
    g, e = baseline_gap(T, n_heads), equalization(T, n_heads)
    return (
        f"EŞİT DEĞİL — YÖN: FOTON modülleri (A/B/C) token/head başına ortalama "
        f"{g['foton_mean']:.2f} nedensel bağlantı görüyor; D ve M baseline'ları {g['baseline']:.2f} "
        f"görüyor (%{100 * g['ratio']:.1f}, açık {g['deficit']:.2f} bağlantı).\n"
        f"Kalan bu eşitsizlik FOTON'un ALEYHİNE: X ve H kolları baseline'ların YARISI kadar "
        f"bağlantıyla yarışıyor. X kazanırsa sonuç daha güçlüdür; kaybederse yarım bağlantıyla "
        f"kaybettiği sonuçla birlikte okunmalıdır.\n"
        f"A/B/C arası kalan sapma en çok {e['spread']:.2f} bağlantı (%{e['spread_pct']:.2f}) — "
        f"'eşitlendi' denilen eksen budur, 256 değil."
    )


def self_check(T: int = T_KAMPANYA, n_heads: int = N_HEAD_KAMPANYA) -> dict:
    """Üç bağımsız yolun uyuştuğunu doğrular: kapalı form ortalama / counts toplamı / maske.

    Modülün varlık sebebi bu uyuşma; `__main__` ve testler bunu çağırır, tablo ancak bu geçerse
    basılır. Geçmezse rapor edilen "eşit hesap" iddiası dayanaksızdır.
    """
    out = {}
    for name, pat in PATTERNS.items():
        for h in pat.heads:
            c = h.counts(T)
            from_counts = float(c.sum()) / T
            if abs(from_counts - h.avg(T)) > 1e-12:
                raise AssertionError(
                    f"{name}/{h.label}: kapalı form ortalama {h.avg(T)} != counts toplamı {from_counts}")
            rows = h.mask(T).sum(axis=1)        # `mask` kendi içinde counts ile karşılaştırır
            if not np.array_equal(rows, c):
                raise AssertionError(f"{name}/{h.label}: maske satır toplamı counts ile uyuşmadı")
        out[name] = pat.avg(T, n_heads)
    return out


def _table(T: int, n_heads: int) -> str:
    """Denetimin tablosu + donmuş tasarım tablosu, HESAPLANARAK (elle yazılmış sayı yok)."""
    why = [("tam nedensel (D ve M baseline'ları)", full()),
           ("yerel pencere 256", local(256)),
           ("dilated r=2, bütçe 256", dilated(2, 256)),
           ("128 yerel + 128 strided s=4 (head-split)",
            head_split("128+128", Head(1, 128, "yerel W=128"), Head(4, 128, "strided s=4 W=128")))]
    lines = [f"Neden hedef 256'da tutulamadı (T={T}, nedensel):", ""]
    lines.append(f"  {'desen':42s} {'ort. bağlantı':>14s}")
    for label, pat in why:
        lines.append(f"  {label:42s} {pat.avg(T, n_heads):14.2f}")
    lines += ["", f"Donmuş tasarım (n_head={n_heads}):", "",
              f"  {'modül':6s} {'desen':34s} {'ort.':>8s} {'en uzak':>8s} {'ayrık':>7s}"]
    for key in FOTON_MODULES:
        s = PATTERNS[key].summary(T, n_heads)
        lines.append(f"  {key:6s} {' + '.join(s['head_labels']):34s} {s['avg']:8.2f} "
                     f"{s['max_span']:8d} {'/'.join(str(d) for d in s['distinct_reachable']):>7s}")
        if len(s["head_avgs"]) > 1:             # C: head başına ortalama ayrıca raporlanır
            per = ", ".join(f"{lab} = {v:.2f}" for lab, v in
                            zip(s["head_labels"], s["head_avgs"]))
            lines.append(f"  {'':6s} head başına: {per}")
    base = PATTERNS["full"].summary(T, n_heads)
    lines.append(f"  {'D/M':6s} {base['head_labels'][0]:34s} {base['avg']:8.2f} "
                 f"{base['max_span']:8d} {base['distinct_reachable'][0]:7d}")
    return "\n".join(lines)


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows konsolu cp1252
    T, n_heads = T_KAMPANYA, N_HEAD_KAMPANYA
    self_check(T, n_heads)                      # tablo ancak üç yol uyuşursa basılır
    print(headline(T, n_heads))
    print()
    print(_table(T, n_heads))
    print()
    e = equalization(T, n_heads)
    print(f"eşitleme: sapma {e['spread']:.4f} bağlantı (%{e['spread_pct']:.4f}), "
          f"tolerans {e['tol']} -> {'geçti' if e['within_tol'] else 'GEÇMEDİ'}")
    print("üç bağımsız yol (kapalı form / konum toplamı / maske satır toplamı) uyuştu.")


if __name__ == "__main__":
    main()
