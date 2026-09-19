"""07 raporunun sayılarını ÜRETEN teşhis betiklerinin mutant kataloğu — `scripts/mutation_guards.py --suite 07`.

Ön-kayıt V13: yakaladığı kanıtlanmamış guard, guard değildir. Bu betikler karar kuralı DEĞİLDİR, ama 07 raporunun
iddiaları ("sinyal var, hedef yok", oracle tavanı 1.34 puan, eşit token açığı) onların çıktısıdır: sessizce yanlış
hesaplayan bir teşhis, yanlış bir sonuç cümlesi demektir. Kayıt: `experiments/foton-v02/mutation_log_07.md`.
"""
RD, GD, PD = "scripts/router_diag.py", "scripts/router_grad_diag.py", "scripts/router_probe_diag.py"
OC, SC, TO = "scripts/oracle_ceiling.py", "scripts/screening_context.py", "scripts/topology_overlap.py"

G_RD, G_GD, G_PD = "tests/test_router_diag.py", "tests/test_router_grad_diag.py", "tests/test_router_probe_diag.py"
G_OC, G_SC, G_TO = "tests/test_oracle_ceiling.py", "tests/test_screening_context.py", "tests/test_topology_overlap.py"

MUTANTS_07 = [
    # ------------------------------------------------------------------ router sebep incelemesi
    dict(id="Z01", guard=G_RD, file=RD, what="modül değiştirme oranı TERS (aynı kalanların oranı okunuyor)",
         old="else float((prev != choice).mean())", new="else float((prev == choice).mean())"),
    dict(id="Z02", guard=G_RD, file=RD, what="'router = en iyi modül' en KÖTÜ modülle karşılaştırıyor",
         old="    best = sm.argmin(1)\n    top2", new="    best = sm.argmax(1)\n    top2"),
    dict(id="Z03", guard=G_RD, file=RD, what="hedef durağanlığı ölçüsü TERS (en iyi modülü DEĞİŞEN dizilerin oranı)",
         old='float((prev_best == best).mean()) if same_n else None', new='float((prev_best != best).mean()) if same_n else None'),
    dict(id="Z04", guard=G_RD, file=RD, what="I(alan; seçim) bağımsızlık çarpımı yerine ortak dağılımın kendisine bölünüyor (hep 0)",
         old="    return float((p[nz] * np.log2(p[nz] / (pa @ pb)[nz])).sum())", new="    return float((p[nz] * np.log2(p[nz] / p[nz])).sum())"),
    dict(id="Z05", guard=G_GD, file=GD, what="sinyal 'en iyi modülü gösteriyor' ölçüsü en POZİTİF gradyana bakıyor",
         old='"grad_points_to_best": float((g.argmin(1) == loss.argmin(1)).mean()),',
         new='"grad_points_to_best": float((g.argmax(1) == loss.argmin(1)).mean()),'),
    dict(id="Z06", guard=G_GD, file=GD, what="gate yaprak tensörü modül çıktısına değil residual'ın tamamına çarpılıyor (başka bir türev)",
         old="        x = x + gate.to(x.dtype)[:, None, None] * (y - x)", new="        x = gate.to(x.dtype)[:, None, None] * y"),
    dict(id="Z07", guard=G_PD, file=PD, what="permütasyon null'ı etiketleri karıştırmıyor (null = gerçek doğruluk)",
         old="    null = [cv_accuracy(x, rng.permutation(y), n_cls, seed=seed) for _ in range(n_perm)]",
         new="    null = [cv_accuracy(x, y, n_cls, seed=seed) for _ in range(n_perm)]"),
    dict(id="Z08", guard=G_PD, file=PD, what="prob değerlendirme katını EĞİTİMDE de görüyor (sızıntı: ezber doğruluğu)",
         old="        tr = np.setdiff1d(order, te)", new="        tr = order"),
    # ------------------------------------------------------------------ oracle tavanı
    dict(id="Z09", guard=G_OC, file=OC, what="puana çevirme eğime BÖLMÜYOR, çarpıyor",
         old="    return 100.0 * (1.0 - math.exp(-delta_nat / abs(slope)))", new="    return 100.0 * (1.0 - math.exp(-delta_nat * abs(slope)))"),
    dict(id="Z10", guard=G_OC, file=OC, what="taban en iyi değil en KÖTÜ sabit modül (tavan şişer)",
         old="    best_m = int(fixed.argmin())", new="    best_m = int(fixed.argmax())"),
    dict(id="Z11", guard=G_OC, file=OC, what="ham döküm ↔ loglanan eğri sağlaması kapalı (yanlış dosya sessizce okunur)",
         old='        if abs(c["oracle"] - v["val_loss_oracle"]) > 1e-4 or abs(c["routed"] - v["val_loss"]) > 1e-4:', new="        if False:"),
    dict(id="Z12", guard=G_OC, file=OC, what="ikiye-bölmeli alan politikası modülü DEĞERLENDİRME yarısında seçiyor (sızıntı)",
         old="        m = sm[f].mean(0).argmin() if f.any() else sm[fit].mean(0).argmin()", new="        m = sm[t].mean(0).argmin()"),
    dict(id="Z13", guard=G_OC, file=OC, what="null'ı aşan kısmın işareti ters",
         old='        c["structured_gain_over_null"] = v["val_loss_oracle_null"] - c["oracle"]',
         new='        c["structured_gain_over_null"] = c["oracle"] - v["val_loss_oracle_null"]'),
    dict(id="Z14", guard=G_OC, file=OC, what="oracle dizi başına en küçüğü değil ortalamayı alıyor (tavan = 0'a yakın)",
         old="    seq_min = sm.min(1)", new="    seq_min = sm.mean(1)"),
    dict(id="Z20", guard=G_OC, file=OC, what="token-düzeyi ham ölçü prefix pozisyonlarını da sayıyor (ölçüt 64..511 değil 0..511)",
         old="    x = np.asarray(ce_tok[:, :, k:], dtype=np.float64)", new="    x = np.asarray(ce_tok[:, :, :], dtype=np.float64)"),
    dict(id="Z21", guard=G_OC, file=OC, what="koşular arası örtüşmenin şans düzeyi marjinallerden değil 1/M'den (tek modül hep kazanırsa sahte 'tekrarlanan tercih')",
         old='"expected_if_independent": float(px @ py)})', new='"expected_if_independent": 1.0 / m})'),
    dict(id="Z22", guard=G_OC, file=OC, what="puan cinsinden seyir her noktada kendi yerel eğimini değil SON eğimi kullanıyor",
         old="            sl = local_slope(upto)", new="            sl = local_slope(ordered)"),
    # ------------------------------------------------------------------ eşit token bağlam sayıları
    dict(id="Z23", guard=G_SC, file=SC, what="eşit-kayıp token oranı TERS (baseline / X)",
         old="            out[str(s)] = None if b is None else s / b", new="            out[str(s)] = None if b is None else b / s"),
    dict(id="Z24", guard=G_SC, file=SC, what="eğrinin ulaşmadığı kayıp için oran UYDURULUYOR (None yerine son adım)",
         old="    return None\n\n\ndef equal_loss_token_ratio", new="    return float(pts[-1][0])\n\n\ndef equal_loss_token_ratio"),
    dict(id="Z15", guard=G_SC, file=SC, what="maskesiz ölçütle loglanmış koşu kabul ediliyor",
         old="    if skips != {64}:", new="    if False:"),
    dict(id="Z16", guard=G_SC, file=SC, what="'X − D' farkı M'e karşı hesaplanıyor",
         old='            row[g]["minus_D"] = row[g]["mean"] - row["D"]["mean"]', new='            row[g]["minus_D"] = row[g]["mean"] - row["M"]["mean"]'),
    # ------------------------------------------------------------------ grafik örtüşmesi
    dict(id="Z17", guard=G_TO, file=TO, what="çift sayımı nedensel kullanılabilirliği yok sayıyor (her ofset T sorguda sayılıyor)",
         old="    return int((T - np.flatnonzero(offsets)).sum())", new="    return int(len(np.flatnonzero(offsets)) * T)"),
    dict(id="Z18", guard=G_TO, file=TO, what="Jaccard birleşime değil ilk modüle bölünüyor",
         old='        out["pairs"][f"{a}-{b}"] = {"jaccard": inter / union,', new='        out["pairs"][f"{a}-{b}"] = {"jaccard": inter / pair_count(unions[a]),'),
    dict(id="Z19", guard=G_TO, file=TO, what="modül grafiği head birleşimi yerine yalnızca İLK head (C'nin strided yarısı kaybolur)",
         old="    for head, _ in ap.PATTERNS[name].head_groups():", new="    for head, _ in ap.PATTERNS[name].head_groups()[:1]:"),
]
