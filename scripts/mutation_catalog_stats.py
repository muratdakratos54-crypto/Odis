"""Karar kuralı (`src/stats.py`) mutantlarının kataloğu — `scripts/mutation_guards.py` bunu içe aktarır.

Ayrı dosya olmasının sebebi okunurluk: her satır "şu bozulursa" der ve bir denetçi listeyi koda
bakmadan okuyabilir. STATS_MUTANTS pytest guard'ıyla, REACH_MUTANTS ulaşılabilirlik taramasının
assertion'larıyla koşulur (aynı bozulma iki ayrı guard'a karşı denenir).
"""
S = "src/stats.py"
G_STATS = "G_STATS"
G_REACH = "G_REACH"

EQ_S27 = ("EŞDEĞER MUTANT — hiçbir test öldüremez, çünkü davranış değişmiyor. Kanıt: kesinlik "
          "ölçütü bu daldan ÖNCE koşar ve karar ancak `(lo > 0 ve s̄ >= 5)` ya da `hi < 5` ise "
          "belirlidir. `s̄ >= 5` iken `hi >= 5` olduğundan ikinci yol kapalıdır; yani bu dala "
          "`s̄ >= 5` ile gelen her örnekte `lo > 0` ZATEN sağlanmıştır. Sansürlü durumda ölçüt "
          "sansürsüz alt örneğe bakar ama o zaman tam örnek ortalaması ikame yüzünden ~−25'tir "
          "ve bu dala hiç girilmez. Koşul yedekli savunmadır; silinmesi satırı değiştirmez.")

UB = "        return min(arith, (len(unc) * ci95(unc)[2] + sum(cens)) / n)"
ORACLE = "    oracle_ok = all(g < oracle_threshold for g in gains)"
HOLM_IF = '    if row1 in ("güçlü başarı", "başarılı") and "birincil-1" not in passed:'

STATS_MUTANTS = [
    dict(id="S01", what="paired(): seed KİMLİĞİ yerine liste KONUMU",
         old="    if seeds_a is not None and seeds_b is not None:\n        pos = {s: i",
         new="    if False:\n        pos = {s: i"),
    dict(id="S02", what="t kritik değeri n'den bağımsız sabit (v3 hatası)",
         old="    if df in T975:\n        return T975[df]", new="    if df in T975:\n        return T975[2]"),
    dict(id="S03", what="n_star: sürdürülen kesişim yerine İLK (erken sarkma) kesişim",
         old="    (a, ya), (b, yb) = pts[above[-1]], pts[above[-1] + 1]",
         new="    j0 = next(j for j, (_, y) in enumerate(pts) if y <= target)\n"
             "    (a, ya), (b, yb) = pts[j0 - 1], pts[j0]"),
    dict(id="S04", what="savings(): baseline KONUMLA bulunuyor",
         old="        fb = f_base[pos[s]]", new="        fb = f_base[len(s_i)]"),
    dict(id="S05", what="savings(): seed kümesi uyuşmazlığı sessiz",
         old="    if sorted(seeds_f) != sorted(seeds_b):", new="    if False:"),
    dict(id="S06", what="decide(): n ∈ {5,10} şartı kapalı (2 seed'le başarı)", nth=2,
         old="    if strict and n not in REQUIRED_N:", new="    if False:"),
    dict(id="S07", what="decide(): 'güçlü başarı' iki baseline'ı da İSTEMİYOR",
         old="and wmin > 0 and w_complete:", new="and wmin > 0:"),
    dict(id="S08", what="decide(): çelişki etiketi GA yerine yalnız mean > 0",
         old='                if lo > CI_EPS:\n                    extra.append("FLOP kazancı var (GA tamamen > 0)',
         new='                if mean > 0:\n                    extra.append("FLOP kazancı var (GA tamamen > 0)'),
    dict(id="S09", what="decide(): n=10'da terminal satır yok",
         old="        if n >= max(REQUIRED_N) or not seed_right_available:", new="        if not seed_right_available:"),
    dict(id="S10", what="decide(): ≥2 sansür -> 'başarısız' kuralı kapalı",
         old="    if n_cens >= 2:", new="    if n_cens >= 99:"),
    dict(id="S11", what="kesinlik ölçütü (tripwire) tamamen kapalı",
         old="    return hi >= SUCCESS", new="    return False"),
    dict(id="S12", what="ub95: sansürlü ikame GÖZLEM gibi düz t-aralığına giriyor (2. denetimin hatası)",
         old=UB, new="        return ci95(savings)[2]"),
    dict(id="S13", what="ub95: yalnızca sansürsüz ALT örneğin aralığı (3. denetimin hatası: 3/5 sansür terk edilemez)",
         old=UB, new="        return ci95(unc)[2]"),
    dict(id="S14", what="abandon(): tek ölçek yolunda 'hak kullanılmadı' ön-koşulu kapalı",
         old="        if escalation_used:\n            why.append(", new="        if False:\n            why.append("),
    dict(id="S15", what="abandon(): oracle şartı bağlamıyor", old=ORACLE, new="    oracle_ok = True"),
    dict(id="S16", what="contribution(): Holm düzeyi (.025) yerine .05 GA",
         old="    h = t_crit_alpha(n - 1, alpha) * sd / math.sqrt(n)", new="    h = t_crit(n - 1) * sd / math.sqrt(n)"),
    dict(id="S17", what="contribution(): 'küçük' durumu hiç oluşmuyor (dört durum çöküyor)",
         old="    real, small = lo > CI_EPS, hi < mde", new="    real, small = lo > CI_EPS, False"),
    dict(id="S18", what="decide_primary(): Holm satır adını HİÇ düşürmüyor", old=HOLM_IF, new="    if False:"),
    dict(id="S19", what="multiplicity(): pinli aile büyüklüğü yok sayılıyor",
         old="    m = family_size if family_size is not None else m_eff", new="    m = m_eff"),
    dict(id="S20", what="_wbar(): alakasız anahtar kabul ediliyor", old="        if extra:", new="        if False:"),
    dict(id="S21", what="decide(): 1 sansürlü seed varken başarı BLOKLANMIYOR",
         old='    if n_cens == 1 and row in ("güçlü başarı", "başarılı"):', new="    if False:"),
    dict(id="S22", what="başarı eşiği 5 -> 4 puan",
         old="SUCCESS = 5.0           # puan", new="SUCCESS = 4.0           # puan"),
    dict(id="S23", what="'güçlü başarı' eşiği 10 -> 9 puan",
         old="STRONG = 10.0           # puan", new="STRONG = 9.0            # puan"),
    dict(id="S24", what="'nötr' bandı ±2 -> ±1 puan",
         old="NEUTRAL = 2.0           # puan", new="NEUTRAL = 1.0           # puan"),
    dict(id="S25", what="birincil-2'nin MDE çıtası 1.88 -> 1.00 puan",
         old="MDE_PRIMARY2 = 1.88      # puan", new="MDE_PRIMARY2 = 1.00      # puan"),
    dict(id="S26", what="Holm aile büyüklüğü 2 -> 1 (düzeltme fiilen kapalı)",
         old="PRIMARY_FAMILY = 2       # birincil-1", new="PRIMARY_FAMILY = 1       # birincil-1"),
    dict(id="S27", what="decide(): 'başarılı' dalı GA alt > 0 şartını aramıyor",
         old="    elif mean >= SUCCESS and lo > CI_EPS:", new="    elif mean >= SUCCESS:", equivalent=EQ_S27),
    # --- üçüncü denetim turunun bulduğu davranışlar (S28, S37, S38, S39 o turda HAYATTA KALAN mutantlardı) ---
    dict(id="S28", what="abandon(): iki ölçekte yalnızca SON ölçeğe bakıyor ([başarılı, başarısız] terk)",
         old="    if not all(u < SUCCESS for u in ubs):", new="    if not ubs[-1] < SUCCESS:"),
    dict(id="S29", what="abandon(): iki ölçek yolu hiç kullanılmıyor",
         old="    two = len(scale_results) >= 2", new="    two = False"),
    dict(id="S30", what="abandon(): tek ölçek yolu 'nötr'den de açılıyor",
         old='        if r["row"] != "başarısız":', new="        if False:"),
    dict(id="S31", what="abandon(): oracle ölçeklerden BİRİNDE geçse yetiyor",
         old=ORACLE, new="    oracle_ok = any(g < oracle_threshold for g in gains)"),
    dict(id="S32", what="abandon(): keşifsel (strict=False) sonuç terk kararını sürebiliyor",
         old='        if not r.get("strict", False) or r.get("n") not in REQUIRED_N:', new="        if False:"),
    dict(id="S33", what="decide(): seed hakkı harcanmışken yine 'seed 10'a çıkar' diyor",
         old="        if n >= max(REQUIRED_N) or not seed_right_available:", new="        if n >= max(REQUIRED_N):"),
    dict(id="S34", what="_wbar(): NaN w̄ kabul ediliyor (sıra-bağımlı 'güçlü başarı')",
         old="    if bad:", new="    if False:"),
    dict(id="S35", what="decide_primary(): iki endpoint farklı n'de hesaplanabiliyor",
         old="    if len(s_xm) != len(s_hm):", new="    if False:"),
    dict(id="S36", what="decide_primary(): sansürlü seed birincil-2'de ikameyle farka giriyor",
         old="    if any(mx) or any(mh):", new="    if False:"),
    dict(id="S37", what="decide_primary(): Holm yalnızca 'başarılı'yı düşürüyor, 'güçlü başarı'yı değil",
         old=HOLM_IF, new='    if row1 in ("başarılı",) and "birincil-1" not in passed:'),
    dict(id="S38", what="decide_primary(): contribution()'a α/2 yerine α iletiliyor",
         old="        c2 = contribution(diffs, alpha=alpha / PRIMARY_FAMILY, strict=strict)",
         new="        c2 = contribution(diffs, alpha=alpha, strict=strict)"),
    dict(id="S39", what="Holm adım adım DURMUYOR (break -> continue)",
         old="        else:\n            break", new="        else:\n            continue"),
    dict(id="S40", what="decide(): erken dönüşlerde B8(i) 'LEHİNE yanlı' notu atlanıyor",
         old="        if n_cens >= 1:\n            extra.append(", new="        if False:\n            extra.append("),
    dict(id="S41", what="contribution(): bağlayıcı durum 3 seed'den çıkabiliyor", nth=1,
         old="    if strict and n not in REQUIRED_N:", new="    if False:"),
]

REACH_MUTANTS = [
    dict(id="R01", what="(S28) tarama '[başarılı, başarısız] terk' regresyonunu YAKALAMALI",
         old="    if not all(u < SUCCESS for u in ubs):", new="    if not ubs[-1] < SUCCESS:"),
    dict(id="R02", what="(S12) tarama düz t-aralığıyla sansürde ulaşılamazlığı YAKALAMALI",
         old=UB, new="        return ci95(savings)[2]"),
    dict(id="R03", what="(S15) tarama oracle'ın bağlamadığını YAKALAMALI", old=ORACLE, new="    oracle_ok = True"),
    dict(id="R04", what="(S13) tarama 3/5 sansür deliğini ve monotonluk ihlalini YAKALAMALI",
         old=UB, new="        return ci95(unc)[2]"),
    dict(id="R05", what="(S30) tarama 'nötr'den tek ölçekle terki YAKALAMALI",
         old='        if r["row"] != "başarısız":', new="        if False:"),
    dict(id="R06", what="(S31) tarama oracle'ın ölçek BAŞINA bağlamadığını YAKALAMALI",
         old=ORACLE, new="    oracle_ok = any(g < oracle_threshold for g in gains)"),
]
