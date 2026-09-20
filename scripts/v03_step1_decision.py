"""v03 ADIM 1 — karar kuralının TEK kopyası (veri görülmeden yazıldı). Belge: experiments/foton-v03-step1/decision_rule.md

Saf fonksiyonlar; hem ulaşılabilirlik sınavı (`v03_step1_reachability.py`) hem analiz (`v03_step1_analyze.py`) BUNLARI
çağırır — kural iki yerde iki ayrı biçimde yazılmasın. Bütün girdiler SEED KİMLİĞİYLE anahtarlı eşlemelerdir (liste kabul
edilmez): bu depoda konuma göre eşleme hatası üç kez doğdu; yanlış kullanım çalışmasın, patlasın.

PUAN ÇEVİRİMİ — tek ve tutarlı: `puan = 100·(1 − exp(−Δ/0.3346))`, ön-kaydın V9'da çivilediği L* yerel eğimi (v02'nin
"1.34 puan"ı ve "X, D'nin ~14 puan gerisinde" okuması da bu çevirimdir). Kural NAT cinsinden BAĞLAYICIDIR; puan okunuştur.
Dürüstlük notu: ölçüm adım 2188'de (bütçenin yarısı) yapılır ve orada eğrinin yerel eğimi daha diktir (v02 X koşularında
−0.44 … −0.49): aynı nat farkı o noktada daha AZ compute eder (0.01019 nat ≈ 2.1–2.3, 0.01716 nat ≈ 3.4–3.8 yerel puan).
Yani nat eşikleri "yerel puan" cinsinden adlarından daha GEVŞEKTİR; analiz doğrudan token-oranı okumasını da raporlar.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

SLOPE = -0.3346
SEEDS = (101, 102, 103)


def nat_of_points(points: float) -> float:
    return -abs(SLOPE) * math.log(1.0 - points / 100.0)


def points_of_nat(nat: float) -> float:
    """İşaretli: negatif fark negatif puan okur (|Δ| üzerinden, simetrik)."""
    return math.copysign(100.0 * (1.0 - math.exp(-abs(nat) / abs(SLOPE))), nat)


A_POINTS, B_POINTS = 3.0, 5.0
A_NAT, B_NAT = nat_of_points(A_POINTS), nat_of_points(B_POINTS)          # 0.010192 / 0.017163 nat
PRE_MEAN_FIXED_NAT = 0.015        # ön koşul: |mean_fixed(X-iid) − mean_fixed(v02-X)|, seed başına
PRE_MEAN_SHIFT_NAT = 0.0075       # ön koşul: üç seed'in İŞARETLİ ortalama kayması (seed-başı bacak sistematik kaymaya kördür:
#                                   (a) eşiği kadar, 0.010 nat'lık bir kaymayı çoğu zaman "temiz" sayardı). 3·σ/√3, σ = 0.0044
#                                   (v02'nin aynı-seed'li 3 x 3 tablosunun artık sd'si).
PRE_CEILING_POINTS = 0.4          # ön koşul: |oracle tavanı(X-iid) − oracle tavanı(v02-X)|, seed başına
MASK_VS_WEIGHT_NAT = 0.005        # teşhis: |val(H-dom) − val(X-dom)| bunun altındaysa "grafik farkı ölçülemedi"
# (b) kapısı bütünüyle YENİ D koşularına dayanır; onların akla-yatkınlık denetimi: dondurulmuş D referansı (seed 11..55, n = 5)
# adım 2000'de 3.1198 ± 0.0027. Yeni D koşuları (seed 101..103) başka seed ve veri sırası görür: eşli değil, akla-yatkınlık.
D_REF_STEP, D_REF_VAL, D_REF_TOL = 2000, 3.1198, 0.012     # 0.012 = 4 · 0.0027 · sqrt(1 + 1/5)
# AYNI TAKVİM KESRİ: referans D koşuları tam bütçeli, yeni D'ler tarama bütçeli — ama `stop_step` yalnızca koşuyu KESER; LR takvimi
# iki durumda da max_steps = 4375 üzerinden hesaplanır. Adım 2000'de loglanan LR referans koşularda da tarama koşularında da
# bit düzeyinde aynıdır (aşağıdaki sabit); analiz bunu 12 koşunun her birinde VERİDEN denetler. Tolerans takvim farkı İÇERMEZ.
LR_AT_D_REF_STEP = 0.0006469248450683149
D_REF_MEAN_TOL = 0.006            # üç yeni D'nin İŞARETLİ ortalama sapması: 3 · 0.0027 · sqrt(1/3 + 1/5). Seed-başı bacak tek başına
#                                   0.010 nat'lık sistematik bir D kaymasını (kapıyı 0.010 nat şişirir / söndürür) ~%46 "akla yatkın" sayardı.


def _num(v, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} sonlu bir sayı olmalı, verilen {v!r}")
    return float(v)


def precondition_seed_leg(mean_fixed_diff_nat, ceiling_diff_points) -> bool:
    """Ön koşulun SEED-BAŞI bacağı — TEK kopya: hem `precondition()` (üç seed + işaretli ortalama) hem de seed 101 bittiğinde
    bakılan geçerlilik kapısı (`v03_step1_gate.py`) BUNU çağırır."""
    dm, dc = _num(mean_fixed_diff_nat, "mean_fixed farkı"), _num(ceiling_diff_points, "tavan farkı")
    return abs(dm) <= PRE_MEAN_FIXED_NAT and abs(dc) <= PRE_CEILING_POINTS


def d_sanity_seed_leg(val_d_at_ref_step) -> bool:
    """D akla-yatkınlık denetiminin SEED-BAŞI bacağı — TEK kopya (`d_sanity()` ve seed-101 kapısı)."""
    dev = _num(val_d_at_ref_step, f"val(D, adım {D_REF_STEP})") - D_REF_VAL
    return abs(dev) <= D_REF_TOL


def _by_seed(x, name: str) -> dict[int, float]:
    if not isinstance(x, Mapping):
        raise TypeError(f"{name}: seed kimliğiyle anahtarlı bir eşleme olmalı (liste / dizi kabul edilmez), verilen {type(x).__name__}")
    if set(x) != set(SEEDS):
        raise ValueError(f"{name}: anahtarlar tam olarak {SEEDS} olmalı, verilen {sorted(x, key=str)}")
    out = {}
    for s in SEEDS:
        v = x[s]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name}[{s}] sonlu bir sayı olmalı, verilen {v!r}")
        out[s] = float(v)
    return out


def paired_difference(a, b, name: str) -> dict[int, float]:
    """a[s] − b[s], SEED KİMLİĞİYLE. (δ = val(X-iid) − val(X-dom); δ_D = val(D) − val(X-dom): pozitif = X-dom daha iyi.)"""
    a, b = _by_seed(a, f"{name}: a"), _by_seed(b, f"{name}: b")
    return {s: a[s] - b[s] for s in SEEDS}


def _threshold_rule(diff, threshold_nat: float, name: str) -> dict:
    d = _by_seed(diff, name)
    mean = sum(d.values()) / len(d)
    all_positive = all(v > 0 for v in d.values())
    return {"per_seed_nat": d, "mean_nat": mean, "mean_points": points_of_nat(mean), "threshold_nat": threshold_nat,
            "all_seeds_positive": all_positive, "passed": bool(mean >= threshold_nat and all_positive)}


def question_a(delta) -> dict:
    """MEKANİZMA — uzmanlaşma oluyor mu: ort_s δ ≥ 3 puan (0.010192 nat) VE üç seed'de de δ > 0."""
    return _threshold_rule(delta, A_NAT, "delta")


def gate_b(delta_d) -> dict:
    """GEÇERLİLİK KAPISI — X-dom ≥ eşli D + 5 puan: ort_s δ_D ≥ 0.017163 nat VE üç seed'de de δ_D > 0."""
    return _threshold_rule(delta_d, B_NAT, "delta_D")


def precondition(mean_fixed_iid, mean_fixed_v02, ceiling_points_iid, ceiling_points_v02) -> dict:
    """X-iid null'ı temiz mi: zorlanmış yol, aynı seed'li v02-X'e göre bir şey değiştirmemiş olmalı (HER seed'de)."""
    dm = paired_difference(mean_fixed_iid, mean_fixed_v02, "mean_fixed")
    dc = paired_difference(ceiling_points_iid, ceiling_points_v02, "oracle tavanı")
    shift = sum(dm.values()) / len(dm)                       # İŞARETLİ: pozitif = X-iid, v02-X'ten kötü (delta'yı şişirir)
    ok = all(precondition_seed_leg(dm[s], dc[s]) for s in SEEDS)
    ok = ok and abs(shift) <= PRE_MEAN_SHIFT_NAT
    return {"mean_fixed_diff_nat": dm, "mean_shift_nat": shift, "ceiling_diff_points": dc,
            "limits": [PRE_MEAN_FIXED_NAT, PRE_MEAN_SHIFT_NAT, PRE_CEILING_POINTS], "clean": bool(ok)}


def d_sanity(val_d_at_ref_step) -> dict:
    """Yeni D koşuları akla yatkın mı: HER seed'de |val(D, adım 2000) − 3.1198| ≤ 0.012 VE üç seed'in işaretli ortalama sapması
    ≤ 0.006. Aşılırsa (b) OKUNMAZ; (a) okunur."""
    v = _by_seed(val_d_at_ref_step, f"val(D, adım {D_REF_STEP})")
    dev = {s: v[s] - D_REF_VAL for s in SEEDS}
    mean_dev = sum(dev.values()) / len(dev)
    ok = all(d_sanity_seed_leg(v[s]) for s in SEEDS) and abs(mean_dev) <= D_REF_MEAN_TOL
    return {"deviation_nat": dev, "mean_deviation_nat": mean_dev, "reference": [D_REF_STEP, D_REF_VAL, D_REF_TOL, D_REF_MEAN_TOL],
            "clean": bool(ok)}


NEXT_STEP = {
    "blocked": "KARAR OKUNMAZ: X-iid aynı seed'li v02-X'ten sapıyor — zorlanmış yol bir şey değiştirmiş; önce o anlaşılır.",
    (False, False): "(a) HAYIR → yönlendirilen katman payı 2/8 → 4/8, AYNI ölçüm, AYNI kural. O da HAYIR ise dizi-düzeyi prefix "
                    "yönlendirme bu ölçekte tükenmiş sayılır; uzun bağlam YENİ hipotez olarak ayrı ön-kayıtla açılır.",
    (False, None): "(a) HAYIR → yönlendirilen katman payı 2/8 → 4/8, AYNI ölçüm, AYNI kural. O da HAYIR ise dizi-düzeyi prefix "
                   "yönlendirme bu ölçekte tükenmiş sayılır; uzun bağlam YENİ hipotez olarak ayrı ön-kayıtla açılır. ((b) okunamadı: "
                   "D koşuları akla yatkınlık denetimini geçmedi; (a) HAYIR iken sonraki adımı değiştirmez, ama D koşuları incelenir.)",
    (False, True): "BEKLENMEYEN — KARAR OKUNMAZ: (b) geçildi ama (a) geçilmedi. δ − δ_D = val(X-iid) − val(D) olduğundan bu ancak "
                   "X-iid eşli D'yi yakalamış ya da geçmişse olur ((a) ortalama şartından düştüyse X-iid, D'den en az 0.007 nat İYİ "
                   "demektir) ve v02'nin 'X, D'nin 0.050 nat gerisinde' bulgusuyla çelişir. Önce D koşuları ve eşleme denetlenir, "
                   "sahibe sorulur; 4/8'e de router'a da GEÇİLMEZ.",
    (True, None): "(a) EVET, ama (b) OKUNAMADI: D koşuları akla yatkınlık denetimini geçmedi. Önce D koşuları incelenir / yeniden "
                  "koşulur; router deneyi kapı okunmadan AÇILMAZ.",
    (True, True): "(a) EVET ve X-dom ≥ D + 5 (eşli) → tavuk-yumurta doğrulandı VE seyreklik maliyeti kapanıyor: v03 asıl deneyi "
                  "(alan etiketiyle ısıtılıp öğrenilen router'a devredilen yönlendirme) yazılır.",
    (True, False): "(a) EVET ama X-dom < D + 5 → uzmanlaşma var, maliyeti ödemiyor: ÖNCE 4/8, router SONRA. Zorlanmış tavan "
                   "D + 5'i geçmiyorsa öğrenilmiş router hiç geçemez.",
}


def decide(delta, delta_d, pre: Mapping, d_ok: Mapping) -> dict:
    """Haritanın tamamı. `pre` = `precondition()`, `d_ok` = `d_sanity()` çıktısı. Ön koşul kirliyse karar OKUNMAZ; D akla
    yatkın değilse yalnızca (b) okunmaz. İki sayı her durumda raporlanır."""
    a, b = question_a(delta), gate_b(delta_d)
    for name, x in (("pre", pre), ("d_ok", d_ok)):
        if not isinstance(x, Mapping) or not isinstance(x.get("clean"), bool):
            raise TypeError(f"{name}: precondition() / d_sanity() çıktısı olmalı")
    key = "blocked" if not pre["clean"] else (a["passed"], b["passed"] if d_ok["clean"] else None)
    return {"question_a": a, "gate_b": b, "precondition_clean": pre["clean"], "d_sanity_clean": d_ok["clean"],
            "outcome": key if key == "blocked" else list(key), "next_step": NEXT_STEP[key]}


def mask_vs_weight(val_h_dom, val_x_dom) -> dict:
    """TEŞHİS (karar değil): val(H-dom) − val(X-dom). Pozitif = (A, B, C) maske kümesi, BU eşlemeyle (code→A, math→B, nl→C),
    (A, A, A)'dan iyi. "Heterojenlik yararlı" diye OKUNAMAZ: fark = [val(H-iid) − val(X-iid)] + [δ_X − δ_H] ve ilk terim
    bilinmiyor (H hiç koşmadı; v02'nin genelci modüllerinde maske A ≈0.001 nat daha iyiydi). Maskenin UZMANLAŞMAYA katkısı
    ancak H-iid ile ölçülür. "Ölçülebilir" = |ortalama| ≥ 0.005 nat VE üç seed'de aynı işaret (yalnız ortalama şartı, gerçek
    fark 0 iken σ = 0.005–0.008'de %8–28 sahte "ölçülebilir" verirdi)."""
    d = paired_difference(val_h_dom, val_x_dom, "H-dom − X-dom")
    mean = sum(d.values()) / len(d)
    same_sign = all(v > 0 for v in d.values()) or all(v < 0 for v in d.values())
    return {"per_seed_nat": d, "mean_nat": mean, "same_sign_in_every_seed": same_sign,
            "measurable": bool(abs(mean) >= MASK_VS_WEIGHT_NAT and same_sign)}
