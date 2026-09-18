"""Karar kuralı (`src/stats.py`) mutantlarının kataloğu — `scripts/mutation_guards.py` bunu içe aktarır.

Ayrı dosya olmasının sebebi okunurluk: her satır "şu bozulursa" der ve bir denetçi listeyi koda
bakmadan okuyabilir. STATS_MUTANTS pytest guard'ıyla, REACH_MUTANTS ulaşılabilirlik taramasının
assertion'larıyla koşulur (aynı bozulma iki ayrı guard'a karşı denenir).

E.. bloğu KONUMA GÖRE EŞLEME sınıfıdır. O hata üç ayrı fonksiyonda üç kez çıktı ve her seferinde
fonksiyon yamalandı; yapısal düzeltme veriyi seed kimliğiyle anahtarlı eşlemeye taşıdı. Buradaki
mutantlar o yapıyı geri bozar: eşlemeyi ekleme sırasına göre yapan, bir kaydıran, listeyi sessizce
kabul eden, anahtar kümesi denetimini kapatan. Yakalanmazsa guard yok demektir.
"""
S = "src/stats.py"
G_STATS = "G_STATS"
G_REACH = "G_REACH"

EQ_S27 = ("EŞDEĞER MUTANT — hiçbir test öldüremez, çünkü davranış değişmiyor. Kanıt: kesinlik "
          "ölçütü bu daldan ÖNCE koşar ve karar ancak `(lo > 0 ve s̄ >= 5)` ya da `ub95 < 5` ise "
          "belirlidir. Her zaman `s̄ <= ub95`: formül kolunda hızlı seed'lerin GA üst sınırı kendi "
          "ortalamalarından büyüktür ve sabit değerler aynen girer; tavan kolunda her hızlı değer "
          "<= 100'dür (`Obs` s > 100'ü reddeder). Dolayısıyla `s̄ >= 5` iken `ub95 >= 5`, ikinci "
          "yol kapalıdır; yani bu dala `s̄ >= 5` ile gelen her örnekte `lo > 0` ZATEN sağlanmıştır. "
          "(>= 2 sansür dalı bu daldan önce döner.) Koşul yedekli savunmadır; silinmesi satırı değiştirmez.")

UB = "        return min(arith, (len(fast) * ci95(fast)[2] + sum(fixed)) / n)"
UB_PLAIN = "        return ci95([o.s for o in tab.values()])[2]"
UB_SPLIT = ('    fast = [o.s for o in tab.values() if o.status == "hızlı"]\n'
            '    fixed = [o.s for o in tab.values() if o.status != "hızlı"]\n'
            "    arith = ")
UB_SPLIT_SLOW_ORDINARY = ('    fast = [o.s for o in tab.values() if o.status != "sansürlü"]\n'
                          '    fixed = [o.s for o in tab.values() if o.status == "sansürlü"]\n'
                          "    arith = ")
PRECISION = "    if indeterminate(mean, lo, ub):"
ORACLE = "    oracle_ok = all(g < oracle_threshold for g in gains)"
HOLM_IF = '    if row1 in ("güçlü başarı", "başarılı") and "birincil-1" not in passed:'
GATE = "    if not isinstance(x, Mapping):\n        raise TypeError("
DIFFS = "    diffs = [x[s].s - h[s].s for s in seeds]"

STATS_MUTANTS = [
    # ------------------------------------------------------------------ E: konuma göre EŞLEME sınıfı
    dict(id="E01", what="paired(): seed KİMLİĞİ yerine ekleme SIRASI (v3 hatası)",
         old="    d = [fa[s] - fb[s] for s in matched]",
         new="    d = [u - v for u, v in zip(a.values(), b.values())][:len(matched)]"),
    dict(id="E02", what="savings(): baseline KİMLİKLE değil ekleme SIRASIYLA bulunuyor (1. denetimin hatası)",
         old="        obs[s] = Obs(100.0 * (1.0 - ff[s] / fb[s]), stat[s])",
         new="        obs[s] = Obs(100.0 * (1.0 - ff[s] / list(f_base.values())[len(obs)]), stat[s])"),
    dict(id="E03", what="savings(): seed DURUMU kimlikle değil ekleme sırasıyla eşleniyor (maske kayması)",
         old="        obs[s] = Obs(100.0 * (1.0 - ff[s] / fb[s]), stat[s])",
         new="        obs[s] = Obs(100.0 * (1.0 - ff[s] / fb[s]), list(stat.values())[len(obs)])"),
    dict(id="E04", what="contribution(): X ile H ekleme SIRASIYLA eşleniyor (4. denetimin hatası)",
         old=DIFFS, new="    diffs = [u.s - v.s for u, v in zip(s_x.values(), s_h.values())]"),
    dict(id="E05", what="contribution(): eşleme BİR KAYDIRILMIŞ (X-s11 ile H-s22 ...)",
         old=DIFFS, new="    diffs = [x[s].s - h[seeds[(i + 1) % n]].s for i, s in enumerate(seeds)]"),
    dict(id="E06", what="abandon(): oracle kazançları ölçekle KİMLİKLE değil ekleme sırasıyla eşleniyor",
         old="    gains = [float(gains_in[k]) for k in used]",
         new="    gains = [float(g) for g in gains_in.values()]"),
    dict(id="E07", what="abandon(): ölçek sonuçları ölçek SIRASINA değil ekleme sırasına göre okunuyor",
         old="    order = [k for k in SCALES if k in res]", new="    order = [k for k in res if k in SCALES]"),
    dict(id="E08", what="kapı KAPALI: liste sessizce `dict(enumerate(...))` sayılıyor (konumlu girdi çalışıyor)",
         old=GATE, new="    if not isinstance(x, Mapping):\n        return dict(enumerate(x))\n    if False:\n        raise TypeError("),
    dict(id="E09", what="kapı yok: eşleme olmayan girdi denetlenmiyor",
         old="    if not isinstance(x, Mapping):", new="    if False:"),
    dict(id="E10", what="anahtar kümesi denetimi kapalı (eksik/fazla seed, farklı n'de iki endpoint)",
         old="    if set(a) != set(b):", new="    if False:"),
    dict(id="E11", what="kanonik sıra yok: çıktı tablonun ekleme sırasına bağlı",
         old="    try:\n        return sorted(keys)", new="    try:\n        return list(keys)"),
    dict(id="E12", what="keyed(): tekrar eden seed sessizce eziliyor (yeniden koşulan seed'in iki kaydı)",
         old="        if k in out:", new="        if False:"),
    dict(id="E13", what="Obs: tanımsız durum kabul ediliyor ('censored' yazım hatası 'hızlı' gibi davranır)",
         old="        if self.status not in STATUSES:", new="        if False:"),
    dict(id="E14", what="Obs: +100 puanı aşan tasarruf kabul ediliyor (birim hatası; S27'nin eşdeğerlik kanıtı buna dayanıyor)",
         old="        if s > S_MAX:", new="        if False:"),
    dict(id="E15", what="abandon(): ölçek anahtarı denetimi kapalı (S0'sız S1, tanımsız ölçek)",
         old="    if unknown or tuple(order) != SCALES[:len(order)]:", new="    if False:"),
    dict(id="E16", what="abandon(): oracle kazancı ölçek BAŞINA ve aynı ölçeklerle verilmek zorunda değil",
         old="    if set(gains_in) != set(res):", new="    if False:"),
    # ------------------------------------------------------------------ S: karar kuralı
    dict(id="S02", what="t kritik değeri n'den bağımsız sabit (v3 hatası)",
         old="    if df in T975:\n        return T975[df]", new="    if df in T975:\n        return T975[2]"),
    dict(id="S03", what="n_star: sürdürülen kesişim yerine İLK (erken sarkma) kesişim",
         old="    (a, ya), (b, yb) = pts[above[-1]], pts[above[-1] + 1]",
         new="    j0 = next(j for j, (_, y) in enumerate(pts) if y <= target)\n"
             "    (a, ya), (b, yb) = pts[j0 - 1], pts[j0]"),
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
    dict(id="S12", what="ub95: başarısız seed GÖZLEM gibi düz t-aralığına giriyor (2. denetimin hatası)",
         old=UB, new=UB_PLAIN),
    dict(id="S13", what="ub95: yalnızca hızlı ALT örneğin aralığı (3. denetimin hatası: 3/5 sansür terk edilemez)",
         old=UB, new="        return ci95(fast)[2]"),
    dict(id="S14", what="abandon(): tek ölçek yolunda 'hak kullanılmadı' ön-koşulu kapalı",
         old="        if escalation_used:\n            why.append(", new="        if False:\n            why.append("),
    dict(id="S15", what="abandon(): oracle şartı bağlamıyor", old=ORACLE, new="    oracle_ok = True"),
    dict(id="S16", what="contribution(): Holm düzeyi (.025) yerine .05 GA",
         old="    hw = t_crit_alpha(n - 1, alpha) * sd / math.sqrt(n)", new="    hw = t_crit(n - 1) * sd / math.sqrt(n)"),
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
         old="    two = len(order) >= 2", new="    two = False"),
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
    dict(id="S36", what="contribution(): başarısız seed birincil-2'de farka giriyor ('ölçemedik' kuralı kapalı)",
         old="    if failed:", new="    if False:"),
    dict(id="S37", what="decide_primary(): Holm yalnızca 'başarılı'yı düşürüyor, 'güçlü başarı'yı değil",
         old=HOLM_IF, new='    if row1 in ("başarılı",) and "birincil-1" not in passed:'),
    dict(id="S38", what="decide_primary(): contribution()'a α/2 yerine α iletiliyor",
         old="    c2 = contribution(x, h, alpha=alpha / PRIMARY_FAMILY, strict=strict)",
         new="    c2 = contribution(x, h, alpha=alpha, strict=strict)"),
    dict(id="S39", what="Holm adım adım DURMUYOR (break -> continue)",
         old="        else:\n            break", new="        else:\n            continue"),
    dict(id="S40", what="decide(): erken dönüşlerde B8(i) 'LEHİNE yanlı' notu atlanıyor",
         old="        if n_cens >= 1:\n            extra.append(", new="        if False:\n            extra.append("),
    dict(id="S41", what="contribution(): bağlayıcı durum 3 seed'den çıkabiliyor", nth=1,
         old="    if strict and n not in REQUIRED_N:", new="    if False:"),
    # --- dördüncü denetim turu: ub95 tavanı, yavaş seed, tek biçimli kesinlik ölçütü ---
    dict(id="S42", what="ub95: aritmetik tavan yalnızca n_h < 2'de ([−62.5, −125, C, C, C] -> +31.4)",
         old=UB, new="        return (len(fast) * ci95(fast)[2] + sum(fixed)) / n"),
    dict(id="S43", what="ub95: yavaş seed sıradan GÖZLEM sayılıyor (4. denetimin hatası: terminal 'yetersiz kesinlik')",
         old=UB_SPLIT, new=UB_SPLIT_SLOW_ORDINARY),
    dict(id="S44", what="decide(): kesinlik ölçütü ub95 yerine düz GA üst sınırına bakıyor",
         old=PRECISION, new="    if indeterminate(mean, lo, hi):"),
    dict(id="S45", what="decide(): yavaş seed B8'in sansür sayımına giriyor (2 yavaş -> doğrudan 'başarısız')",
         old='    n_cens = sum(o.status == "sansürlü" for o in tab.values())',
         new='    n_cens = sum(o.status != "hızlı" for o in tab.values())'),
    dict(id="S46", what="resolve_n_star(): 2x koşu ZORUNLU değil (1x kesmeyince doğrudan sansür)",
         old="    if curve_2x is None:\n        raise ValueError(",
         new='    if curve_2x is None:\n        return 2.0 * float(budget_tokens), "sansürlü"\n    if False:\n        raise ValueError('),
    dict(id="S47", what="resolve_n_star(): 2x'te ulaşan seed 'hızlı' etiketleniyor",
         old='        return n2, "yavaş"', new='        return n2, "hızlı"'),
    dict(id="S48", what="resolve_n_star(): 1x kesse bile 2x eğrisi okunuyor (analist eğri seçebiliyor)",
         old="    if n1 is not None:\n        return n1, \"hızlı\"",
         new="    if n1 is not None and curve_2x is None:\n        return n1, \"hızlı\""),
    dict(id="S49", what="contribution(): 'ölçemedik' kuralı yalnızca SANSÜRE bakıyor, yavaş seed farka giriyor",
         old='    failed = [s for s in seeds if x[s].status != "hızlı" or h[s].status != "hızlı"]',
         new='    failed = [s for s in seeds if x[s].status == "sansürlü" or h[s].status == "sansürlü"]'),
    dict(id="S50", what="decide(): yavaş seed notu rapora düşmüyor",
         old="        if n_slow >= 1:\n            extra.append(", new="        if False:\n            extra.append("),
]

REACH_MUTANTS = [
    dict(id="R01", what="(S28) tarama '[başarılı, başarısız] terk' regresyonunu YAKALAMALI",
         old="    if not all(u < SUCCESS for u in ubs):", new="    if not ubs[-1] < SUCCESS:"),
    dict(id="R02", what="(S12) tarama düz t-aralığıyla başarısız seed'de ulaşılamazlığı YAKALAMALI",
         old=UB, new=UB_PLAIN),
    dict(id="R03", what="(S15) tarama oracle'ın bağlamadığını YAKALAMALI", old=ORACLE, new="    oracle_ok = True"),
    dict(id="R04", what="(S13) tarama 3/5 sansür deliğini ve monotonluk ihlalini YAKALAMALI",
         old=UB, new="        return ci95(fast)[2]"),
    dict(id="R05", what="(S30) tarama 'nötr'den tek ölçekle terki YAKALAMALI",
         old='        if r["row"] != "başarısız":', new="        if False:"),
    dict(id="R06", what="(S31) tarama oracle'ın ölçek BAŞINA bağlamadığını YAKALAMALI",
         old=ORACLE, new="    oracle_ok = any(g < oracle_threshold for g in gains)"),
    dict(id="R07", what="(S42) tarama aritmetik tavanın kalktığını YAKALAMALI",
         old=UB, new="        return (len(fast) * ci95(fast)[2] + sum(fixed)) / n"),
    dict(id="R08", what="(S43) tarama yavaş seed'in sıradan gözlem sayıldığını YAKALAMALI",
         old=UB_SPLIT, new=UB_SPLIT_SLOW_ORDINARY),
    dict(id="R09", what="(S44) tarama kesinlik ölçütünün düz GA'ya döndüğünü YAKALAMALI",
         old=PRECISION, new="    if indeterminate(mean, lo, hi):"),
]
