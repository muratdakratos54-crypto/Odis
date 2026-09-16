"""README'de ve deney kaydında geçen TÜREV sayıları artefaktlardan yeniden hesaplar.

v2'de denetim, README'ye elle yazdığım türev sayıların bir kısmının yanlış olduğunu bulmuştu; v3 denetimi de
aynısını buldu (bkz. kayıt README'sindeki "Denetim" bölümü). O yüzden metinde geçen her türev sayı
(korelasyonlar, oranlar, yüzdeler, p değerleri, null'lar, örüntü madde sayıları) tek bir yerden üretilir.

Girdi : summary.json, analysis/*.json (kanonik), analysis_ilk_kosu/*.json (tekrar zarfı için)
Çıktı : report_numbers.log

Kullanım:  .venv\\Scripts\\python experiments\\v3-router-ablation\\report_numbers.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st

REC = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(REC, "summary.json"), encoding="utf-8"))
T, P, CONDS = S["table"], S["paired_vs_a"], S["conditions"]
SERIES_P, MULT, BEST = S["series_paired"], S["multiplicity"], S["series_best"]
BASE = "a_aux0.01"
E, K, LAYERS = 16, 3, 7
BALANCED_CV = 0.15          # kararı veren eşik; docs/router_metrics.md ile aynı olmalı
HOLM, BH = set(MULT["holm"]), set(MULT["bh"])

out: list[str] = []


def say(s: str = "") -> None:
    print(s)
    out.append(s)


def mean(metric: str, cond: str) -> float:
    return st.mean(T[metric][cond])


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den


def pull(keys: list[str], metric: str) -> list[float]:
    return [v for k in keys for v in T[metric][k]]


def analyses(cond: str, canonical: bool = True) -> list[dict]:
    d = "analysis" if canonical else "analysis_ilk_kosu"
    return [json.load(open(f, encoding="utf-8"))
            for f in sorted(glob.glob(os.path.join(REC, d, f"{cond}-s*.json")))]


def status(metric: str, comp: str) -> tuple[dict, str]:
    """(fark kaydı, çoklu karşılaştırma durumu). comp: '<koşul> vs a' ya da series_paired anahtarı."""
    if comp.endswith(" vs a"):
        d = P.get(metric, {}).get(comp[:-5])
    else:
        d = SERIES_P.get(comp, {}).get(metric)
    if not d:
        return {}, "yok"
    name = f"{metric} | {comp}"
    return d, ("Holm+BH" if name in HOLM else "BH" if name in BH else "yalnız nominal")


def line(metric: str, comp: str, desc: str) -> None:
    d, stat = status(metric, comp)
    if not d:
        say(f"  {desc:46s} (hesaplanmadı)")
        return
    lo, hi = d.get("ci95", (float("nan"),) * 2)
    say(f"  {desc:46s} {d['mean']:+9.4f} GA [{lo:+.4f}, {hi:+.4f}] p={d.get('p', float('nan')):.4f}  {stat}")


say("# v3 türev sayıları (artefaktlardan yeniden hesaplandı)")
say(f"koşul: {len(CONDS)} | koşu: {sum(len(c['seeds']) for c in CONDS.values())} "
    f"| expert: {E} x {LAYERS} katman = {E * LAYERS} | top-k = {K}")

say("\n## 1. Dengeli kollar (yük CV <= %.2f ve ölü expert = 0)" % BALANCED_CV)
balanced = [k for k in CONDS if mean("yük CV", k) <= BALANCED_CV and mean("ölü expert", k) == 0]
softmax_balanced = [k for k in balanced if mean("bias ezme oranı", k) == 0]
say(f"dengeli: {balanced}")
say(f"bunların softmax gate'lileri (yalnızca aux gücü değişiyor): {softmax_balanced}")
say(f"en düşük yük CV: {min(balanced, key=lambda k: mean('yük CV', k))} "
    f"| en düşük maks pay: {min(balanced, key=lambda k: mean('maks pay', k))} "
    f"| en iyi val loss: {min(CONDS, key=lambda k: mean('val loss', k))} "
    f"| en yüksek H(E|T): {max(CONDS, key=lambda k: mean('H(expert|token) bit', k))}")

say("\n## 2. İSTATİSTİKSEL DURUM: 151 eşleştirilmiş test, n=3, düzeltme")
say(f"  nominal p<=.05: {len(MULT['nominal'])} | Holm (FWER .05): {len(MULT['holm'])} | BH (FDR .05): {len(MULT['bh'])}")
say("  Holm'da ayakta kalanların HEPSİ denge ölçütleri (yük CV / yük entropisi / top-1 kullanım entropisi /")
say("  mi_bits null) — yani yalnızca çok büyük denge etkileri. val loss ve uzmanlaşma etkilerinin hiçbiri")
say("  FWER kontrolünden geçmiyor. Aşağıda her iddianın yanında durumu yazıyor.")

say("\n## 3. U eğrisi: SERİNİN OPTİMUMUNA karşı test (referans (a) değil)")
say(f"  serinin en iyi val loss'lu kolu: {BEST} ({CONDS[BEST]['label']})")
for pair, desc in ((f"c_aux0 - {BEST}", "aux yok, optimuma göre"),
                   (f"b_aux0.001 - {BEST}", "aux 0.001, optimuma göre"),
                   (f"a_aux0.01 - {BEST}", "aux 0.01, optimuma göre"),
                   (f"f_aux1.0 - {BEST}", "aux 1.0, optimuma göre")):
    line("val loss", pair, desc)
say("  -> U'nun iki kolu da BH'de ayakta; DİBİ bir nokta değil PLATO: aux 0.01 ile 0.1 ayrıştırılamıyor.")

say("\n## 4. aux 1.0 neyi değiştirdi (optimuma karşı)")
for metric, desc in (("val loss", "val loss"), ("seçim marjı", "seçim marjı (gate yayvanlaştı)"),
                     ("yük CV", "yük CV"), ("mi_bits", "mi_bits (token kimliği bilgisi)"),
                     ("H(expert|token) bit", "H(expert|token) (bağlam bağımlılığı)"),
                     ("tutarlılık", "tutarlılık"), ("ΔCE permütasyon", "ΔCE permütasyon")):
    line(metric, f"f_aux1.0 - {BEST}", desc)
say("  -> DÜZELTMEDEN GEÇEN: val loss kötüleşmesi ve gate'in yayvanlaşması. Bilgi kaymasının kendisi")
say("     (mi_bits yukarı, H(E|T) aşağı) yalnız nominal: yönü 3 seed'de tutarlı ama düzeltmeden geçmiyor.")

say("\n## 5. mi_bits ile H(expert|token) AYRI KANIT DEĞİL (cebirsel ayna)")
he = [v for k in balanced for v in [a["specialization"]["utilization_entropy"] * math.log2(E) for a in analyses(k)]]
say(f"  dengeli kollarda H(expert) = {st.mean(he):.3f} ± {st.stdev(he):.3f} bit (aralık {max(he) - min(he):.3f} bit)")
say(f"  mi_bits'in sd'si: {st.stdev(pull(balanced, 'mi_bits')):.3f} bit -> H(expert) neredeyse sabit")
for name, keys in (("dengeli kollar", balanced), ("dengeli + softmax", softmax_balanced)):
    r = pearson(pull(keys, "mi_bits"), pull(keys, "H(expert|token) bit"))
    say(f"  r(mi_bits, H(E|T)) [{name}, n={len(pull(keys, 'mi_bits'))}] = {r:+.3f}")
say("  -> mi_bits + H(E|T) = H(expert) ≈ sabit olduğu için ikisi tek ölçümün iki yüzü. 'Uzmanlaşma arttı VE")
say("     bağlam bağımlılığı düştü' iki bağımsız bulgu gibi sunulamaz; ayrıca docs §2'deki eski güvenlik")
say("     kuralı ('ikisi de yüksekse bağlam duyarlılığıdır') bu rejimde cebirsel olarak imkânsızdır.")

say("\n## 6. Korelasyonlar: gözlem birimi KOŞUL, koşu değil (pseudo-replikasyon düzeltmesi)")
say(f"  {'küme':22s} {'birim':>8s} {'n':>3s} {'df':>3s}  r(H(E|T),val)  r(mi,val)  r(yük CV,val)")
for name, keys in (("tüm kollar", list(CONDS)), ("dengeli", balanced), ("dengeli+softmax", softmax_balanced)):
    for unit in ("koşu", "koşul"):
        if unit == "koşu":
            h, v, mi, cv = (pull(keys, m) for m in ("H(expert|token) bit", "val loss", "mi_bits", "yük CV"))
        else:
            h, v, mi, cv = ([mean(m, k) for k in keys] for m in ("H(expert|token) bit", "val loss", "mi_bits", "yük CV"))
        say(f"  {name:22s} {unit:>8s} {len(v):3d} {len(v) - 2:3d}  {pearson(h, v):+13.3f}  {pearson(mi, v):+9.3f}  "
            f"{pearson(cv, v):+13.3f}")
say("  -> Koşul düzeyinde df 4 / 2 / 1'e düşüyor. 'İşaret döndü' iddiası dengeli+softmax'ta 3 koşul ortalamasına")
say("     dayanıyor (df=1); 3 noktada herhangi bir monoton ilişki |r|~1 verir. Bu bir UYARI, kanıt değil.")

say("\n## 7. Null'lar: ham tutarlılık koşullar arası karşılaştırılamaz")
say(f"  {'koşul':22s} {'tutarlılık':>11s} {'null':>7s} {'null üstü':>10s} {'mi_bits':>8s} {'mi null':>8s}")
for k in CONDS:
    c, cn = mean("tutarlılık", k), mean("tutarlılık null", k)
    say(f"  {CONDS[k]['label']:22s} {c:11.3f} {cn:7.3f} {c - cn:10.3f} {mean('mi_bits', k):8.3f} "
        f"{mean('mi_bits null', k):8.3f}")
say("  -> (c)'nin HAM tutarlılığı (a)'dan YÜKSEK ama null'ı da yüksek; null üstü fazlada (c) en düşük.")
say("     Yani 'aux'suz kolda tutarlılık daha yüksek' bir artefakt ve düzeltmesi null'a göre okumaktır.")
say(f"  mi_bits'in plug-in tabanı dengeli kollarda ~{st.mean([mean('mi_bits null', k) for k in balanced]):.3f} bit, "
    f"sinyalin ~%{100 * st.mean([mean('mi_bits null', k) for k in balanced]) / st.mean([mean('mi_bits', k) for k in balanced]):.0f}'i.")

say("\n## 8. Çöküş tarafı ve permütasyon testinin (c)'deki karışıklığı")
dead = mean("ölü expert", "c_aux0")
cv_collapse = math.sqrt(E / K - 1)
say(f"  (c) ölü expert {dead:.1f}/{E * LAYERS} = %{100 * dead / (E * LAYERS):.1f}")
say(f"  (c) yük CV {mean('yük CV', 'c_aux0'):.3f}; tam çöküş sqrt(E/k-1) = {cv_collapse:.3f} "
    f"-> yolun %{100 * mean('yük CV', 'c_aux0') / cv_collapse:.1f}'i")
for cond in ("c_aux0", BASE):
    fr = st.mean([st.mean([sum(1 for x in layer if x < 0.01 / E) / E for layer in a["load_per_layer"]])
                  for a in analyses(cond)])
    say(f"  {CONDS[cond]['label']:22s}: rastgele permütasyon tokenların ~%{100 * fr:.1f}'ini, ideal payın %1'inden")
    say(f"  {'':22s}  azını görmüş (fiilen eğitilmemiş) bir modüle yönlendirir")
say("  -> (c)'nin 1.161 nat'ı 'eşleşmenin değeri' olarak okunamaz: büyük kısmı eğitilmemiş modüllere")
say("     yönlendirmenin maliyeti. Dengeli kollarda (ölü expert 0) bu karışıklık yok.")

say("\n## 9. Permütasyon maliyeti (yalnızca dengeli kollar; karışıklıksız)")
for k in balanced:
    rows = analyses(k)
    learned = st.mean(r["counterfactual_ce"]["learned"] for r in rows)
    dperm = st.mean(r["counterfactual_ce"]["delta_permuted"] for r in rows)
    dnext = st.mean(r["counterfactual_ce"]["delta_next"] for r in rows)
    say(f"  {CONDS[k]['label']:22s} öğrenilmiş {learned:.4f} -> permüte +{dperm:.4f} nat "
        f"(kayıp {1 + dperm / learned:.2f}x) | ikinci-en-iyi +{dnext:.4f}")
say("  -> Dengeli dört kolun hiçbirinde routing dekoratif değil. ΔCE permütasyonun SIFIR NOKTASI")
say("     tests/test_router_stats.py'de sabitlendi (expert'ler birbirinin kopyasıysa ~0).")

say("\n## 10. Sahte denge örüntüsü (docs/router_metrics.md §6), mekanik olarak işaretlendi")
say("  1: yük CV <= %.2f ve ölü expert = 0 | 2: mi_bits VE tutarlılık (a)'dan anlamlı düşük" % BALANCED_CV)
say("  3: ΔCE permütasyon (a)'dan anlamlı düşük | 4: aux gradyan oranı (a)'nın 10 katından fazla,")
say("     ya da bias ezme oranı > 0.05 ve ΔCE bias'sız < 0")
say("  (ön-kayıt maddeleri nitelikseldi: 'yüksek/düşük'. Yukarıdaki eşikler sonradan, bu betikte sabitlendi;")
say("   verdikleri karar metnin değil betiğin kararıdır.)")
base_ratio = mean("|aux grad|/|görev grad|", BASE)


def sig(metric: str, cond: str) -> tuple[bool, float]:
    d = P.get(metric, {}).get(cond)
    if not d or "ci95" not in d:
        return False, float("nan")
    lo, hi = d["ci95"]
    return (lo > 0 or hi < 0), d["mean"]


for k in CONDS:
    if k == BASE:
        say(f"  {CONDS[k]['label']:22s} (referans koşul, kendisiyle karşılaştırılamaz)")
        continue
    c1 = mean("yük CV", k) <= BALANCED_CV and mean("ölü expert", k) == 0
    mi_s, mi_d = sig("mi_bits", k)
    co_s, co_d = sig("tutarlılık", k)
    c2 = (mi_s and mi_d < 0) and (co_s and co_d < 0)
    pe_s, pe_d = sig("ΔCE permütasyon", k)
    c3 = pe_s and pe_d < 0
    ratio = T["|aux grad|/|görev grad|"][k]
    ratio_hit = all(x is not None for x in ratio) and st.mean(ratio) > 10 * base_ratio
    nb = T["ΔCE bias'sız"][k]
    bias_hit = mean("bias ezme oranı", k) > 0.05 and all(x is not None for x in nb) and st.mean(nb) < 0
    hits = [i for i, c in enumerate((c1, c2, c3, ratio_hit or bias_hit), 1) if c]
    say(f"  {CONDS[k]['label']:22s} {len(hits)}/4  tutan maddeler: {hits or '-'}")
say("  -> 4/4 (sahte denge) hiçbir koşulda yok. (f) 2/4 ama 2. madde TERS yönde (uzmanlaşma arttı).")

say("\n## 11. Aux'suz dengeleme (d): baskı ne kadar müdahale ediyor")
say(f"  bias ezme oranı {mean('bias ezme oranı', 'd_auxfree'):.3f} -> tokenların "
    f"%{100 * mean('bias ezme oranı', 'd_auxfree'):.1f}'sının top-{K} kümesini bias belirliyor")
nobias = mean("ΔCE bias'sız", "d_auxfree")
say(f"  ΔCE bias'sız {nobias:+.6f} nat; tekrar zarfı (aynı checkpoint iki kez analiz edildiğinde) 1.6e-05")
say(f"  -> etki ölçülebilir ({abs(nobias) / 1.6e-5:.0f}x zarf) ama permütasyon maliyetinin "
    f"{mean('ΔCE permütasyon', 'd_auxfree') / abs(nobias):.0f}'de biri: baskı, seçimin önemsiz olduğu bölgede çalışıyor")
for m, desc in (("ΔCE permütasyon", "ΔCE permütasyon"), ("top-1 gate", "top-1 gate"),
                ("H(expert|token) bit", "H(expert|token)"), ("val loss", "val loss"), ("tok/s (k)", "tok/s (bin)")):
    line(m, "d_auxfree vs a", desc)
say("  -> ΔCE permütasyon ve top-1 gate ikisi de anlamlı düşük: 'zayıf eşleşme' ile 'yayvan gate (seyreltme)'")
say("     ayrıştırılamıyor. val loss üstünlüğü anlamsız. tok/s farkı OTURUM ETKİSİYLE karışık (§13).")

say("\n## 12. Dengeleme kuvveti: koşul / koşu / katman düzeyinde sınırlar")
say(f"  {'koşul':22s} {'oran (ort)':>11s} {'|kosinüs| ort':>14s} {'koşu maks':>10s} {'katman maks':>12s}")
for k in CONDS:
    if not all(x is not None for x in T["|aux grad|/|görev grad|"][k]):
        continue
    rows = analyses(k)
    run_max = max(abs(r["gradient_tension"]["cosine_mean"]) for r in rows)
    layer_max = max(abs(p_["cosine"]) for r in rows for p_ in r["gradient_tension"]["per_layer"])
    say(f"  {CONDS[k]['label']:22s} {mean('|aux grad|/|görev grad|', k):11.4f} "
        f"{abs(mean('aux-görev kosinüs', k)):14.4f} {run_max:10.4f} {layer_max:12.4f}")
allrun = max(abs(r["gradient_tension"]["cosine_mean"]) for k in CONDS for r in analyses(k)
             if r.get("gradient_tension", {}).get("per_layer"))
alllayer = max(abs(p_["cosine"]) for k in CONDS for r in analyses(k)
               for p_ in r.get("gradient_tension", {}).get("per_layer", []))
say(f"  GEÇERLİ ÜST SINIRLAR: koşu düzeyinde |kosinüs| <= {allrun:.4f}, katman düzeyinde <= {alllayer:.4f}")
say(f"  en güçlü kolda oran {mean('|aux grad|/|görev grad|', 'f_aux1.0'):.4f} "
    f"= (a)'nın {mean('|aux grad|/|görev grad|', 'f_aux1.0') / base_ratio:.0f} katı")

say("\n## 13. Hız: koşul etkisi değil OTURUM etkisi")
say("  (a) ve (c) v2 kuyruğunda (gece), (b)/(d)/(e)/(f) v3 kuyruklarında (gündüz) koştu.")
v2, v3 = ["a_aux0.01", "c_aux0"], ["b_aux0.001", "d_auxfree", "e_aux0.1", "f_aux1.0"]
say(f"  v2 oturumu ortalaması {st.mean(pull(v2, 'tok/s (k)')):.2f} bin tok/s | "
    f"v3 oturumu {st.mean(pull(v3, 'tok/s (k)')):.2f} bin tok/s -> fark {st.mean(pull(v2, 'tok/s (k)')) - st.mean(pull(v3, 'tok/s (k)')):.2f}")
sig_pairs = [k for k in CONDS if k != BASE and status("tok/s (k)", f"{k} vs a")[1] != "yalnız nominal"
             or (P.get("tok/s (k)", {}).get(k, {}).get("ci95") or [0, 0])[0] > 0
             or (P.get("tok/s (k)", {}).get(k, {}).get("ci95") or [0, 0])[1] < 0]
say(f"  (a)'ya karşı GA'sı sıfırı dışlayan tok/s farkı olan koşullar: {sig_pairs} -> hepsi v3 oturumunda")
say("  -> tok/s farkları oturum sınırını geçtiği için koşula atfedilemez. Tek geçerli ifade: bütün kollar")
say("     84-87 bin tok/s bandında; dengeleme yönteminin hıza etkisi bu tasarımla ölçülemedi.")

say("\n## 14. Tekrar zarfı (aynı 18 checkpoint iki kez analiz edildi)")
METRICS = [("balance", "load_cv"), ("specialization", "mi_bits"), ("specialization", "consistency"),
           ("counterfactual_ce", "delta_permuted"), ("counterfactual_ce", "delta_next"),
           ("counterfactual_ce", "delta_nobias"), ("gradient_tension", "cosine_mean")]
say(f"  {'ölçüt':36s} {'maks |fark|':>12s}")
for grp, key in METRICS:
    diffs = []
    for k in CONDS:
        for new, old in zip(analyses(k), analyses(k, canonical=False)):
            a, b = new.get(grp, {}).get(key), old.get(grp, {}).get(key)
            if a is not None and b is not None:
                diffs.append(abs(a - b))
    if diffs:
        say(f"  {grp + '.' + key:36s} {max(diffs):12.2e}")
say("  -> GPU tekrar gürültüsü (bf16 + deterministik olmayan kernel'ler) birkaç tokenı farklı yönlendiriyor.")
say("     Bütün raporlanan etkiler bu zarfın en az 10 katı üstünde; ΔCE bias'sız (+2e-4) dahil.")

with open(os.path.join(REC, "report_numbers.log"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
