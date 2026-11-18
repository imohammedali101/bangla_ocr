def edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]

def seq_scores(pred, true):
    ov = sum(sum((Counter(p) & Counter(t)).values()) for p, t in zip(pred, true)); np_, nt = sum(map(len, pred)), sum(map(len, true))
    pr, rc = ov / max(np_, 1), ov / max(nt, 1)
    return dict(exact=float(np.mean([p == t for p, t in zip(pred, true)])), cer=float(sum(edit(p, t) for p, t in zip(pred, true)) / max(nt, 1)),
                f1=float(2 * pr * rc / max(pr + rc, 1e-9)), n=len(true))

tier = lambda ids: 'A' if isA(ids) else 'B'
ne = 200 if CFG['FAST'] else 1000; R = {}

# real glyphs: upright (as stored) and in their original slant (the stored glyphs were deslanted; a canvas drawing is not)
tg_ = np.random.default_rng(1).choice(te, min(len(te), 20000), replace=False); yt = Y[tg_]
def ece_(conf, ok, n=15):
    b = (conf * n).astype(int).clip(max=n - 1)
    return float(sum(abs(conf[b == i].mean() - ok[b == i].mean()) * (b == i).mean() for i in range(n) if (b == i).any()))
def glyph_probs(imgs):
    lg, ax = run(model, imgs), run_aux(model, imgs); pc = np.stack([char_probs(softmax(l, Tc))[1] for l in lg]); pa = softmax(np.stack(ax), Ta)
    return dict(ctc=pc, head=pa, blend=(1 - wa) * pc + wa * pa)
def gscore(P, y):
    ok = P.argmax(1) == y
    return dict(top1=float(ok.mean()), top3=float((np.argsort(-P, 1)[:, :3] == y[:, None]).any(1).mean()), macro_f1=macro_f1(P.argmax(1), y), ece=ece_(P.max(1), ok), n=len(y))
PU = glyph_probs([prep_line(X[i]) for i in tg_]); PS = glyph_probs([prep_line(reslant(X[i], float(SK[i]))) for i in tg_])
Pc = PU['blend']; top1 = Pc.argmax(1) == yt
R['Real glyphs (84 classes)'] = gscore(Pc, yt)
R['  CTC readout only'] = gscore(PU['ctc'], yt); R['  classifier head only'] = gscore(PU['head'], yt); R['  original slant (reconstructed)'] = gscore(PS['blend'], yt)
qd = df.quality.to_numpy()[tg_]; grp = lambda a, b: float(top1[(yt >= a) & (yt < b)].mean())
extra = dict(basic=grp(0, 50), numerals=grp(50, 60), compound=grp(60, 84), q_high=float(top1[qd >= 3].mean()) if (qd >= 3).any() else None,
             q_mid=float(top1[(qd > 1.5) & (qd < 3)].mean()) if ((qd > 1.5) & (qd < 3)).any() else None, q_low=float(top1[qd <= 1.5].mean()) if (qd <= 1.5).any() else None)
cm = np.bincount(yt * 84 + Pc.argmax(1), minlength=84 * 84).reshape(84, 84); np.fill_diagonal(cm, 0)
confused = [(CLASSES[i // 84], CLASSES[i % 84], int(cm.flat[i])) for i in np.argsort(-cm.flatten())[:10]]
per_writer = pd.Series(top1).groupby(WR[tg_]).mean()

# sequences
E = Synth(te, seed=5); en = lambda kind, n, eng=None: [(eng or E).sample(kind) for _ in range(n)]
def evaluate(samples, mode, lex=None, bonus_=0.):
    P = [softmax(l, T) for l in run(model, [s[0] for s in samples])]
    return [(rank(candidates(p, mode, lex), lex, bonus_, 1) or [((), 0.)])[0][0] for p in P], [tuple(s[1]) for s in samples]
nums, lexw, pw = en('num', ne), en('lex', ne, Synth(te, seed=6, pool=LEX_TEST)), en('pseudo', ne)
R['Numbers (1–8 digits)'] = seq_scores(*evaluate(nums, 'number'))
for a, b, nm in ((1, 4, '1–4'), (5, 8, '5–8')):
    sub = [s for s in nums if a <= len(s[1]) <= b]; R[f'  numbers {nm} digits'] = seq_scores(*evaluate(sub, 'number')) if sub else {}
pr_, tr_ = evaluate(lexw, 'word', LEXS, bonus); R['Held-out words, lexicon decoding'] = seq_scores(pr_, tr_)
for tn in 'AB':
    ix = [i for i, t in enumerate(tr_) if tier(t) == tn]; R[f'  tier {tn} (lexicon)'] = seq_scores([pr_[i] for i in ix], [tr_[i] for i in ix]) if ix else {}
pr_f, _ = evaluate(lexw, 'word'); R['Held-out words, free reading'] = seq_scores(pr_f, tr_)
pp_, tp_ = evaluate(pw, 'word'); R['Pseudo-words, free reading'] = seq_scores(pp_, tp_)
for tn in 'AB':
    ix = [i for i, t in enumerate(tp_) if tier(t) == tn]; R[f'  tier {tn} (pseudo)'] = seq_scores([pp_[i] for i in ix], [tp_[i] for i in ix]) if ix else {}

report = pd.DataFrame(R).T[['top1', 'top3', 'macro_f1', 'ece', 'exact', 'cer', 'f1', 'n']]; print(report.round(4).fillna('').to_string())
print('\nGlyph accuracy by group / form quality:', {k: (round(v, 4) if v is not None else None) for k, v in extra.items()})
print('Per-writer accuracy percentiles (5/25/50/75/95):', per_writer.quantile([.05, .25, .5, .75, .95]).round(3).tolist()); print('Top confusions (true → predicted, count):', confused)
TARGETS = [('Real glyph top-1 ≥ 96.5%', R['Real glyphs (84 classes)']['top1'] >= .965), ('Real glyph top-3 ≥ 99%', R['Real glyphs (84 classes)']['top3'] >= .99),
           ('Real glyph macro-F1 ≥ 0.96', R['Real glyphs (84 classes)']['macro_f1'] >= .96), ('Numbers exact ≥ 97%', R['Numbers (1–8 digits)']['exact'] >= .97),
           ('Numbers CER ≤ 1%', R['Numbers (1–8 digits)']['cer'] <= .01), ('Held-out words (lexicon) exact ≥ 95%', R['Held-out words, lexicon decoding']['exact'] >= .95),
           ('Free-reading word CER ≤ 3%', R['Pseudo-words, free reading']['cer'] <= .03)]
for k, ok in TARGETS: print('PASS' if ok else 'FAIL', k)

fig, ax = plt.subplots(1, 2, figsize=(11, 3)); ax[0].hist(per_writer, 20); ax[0].set_title('accuracy per test writer')
ax[1].bar(['A', 'B'], [R[f'  tier {t} (pseudo)'].get('exact', 0) for t in 'AB']); ax[1].set_title('pseudo-word exact match by tier'); plt.show()
json.dump(dict(results=R, glyph_groups=extra, calibration=CAL, accepted_coverage=dict(char=cov_char, number=cov_num, word=cov_word), confusions=confused,
               per_writer_quantiles=per_writer.quantile([.05, .25, .5, .75, .95]).tolist(), targets={k: bool(ok) for k, ok in TARGETS}), open(REL / 'metrics.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
(REL / 'report.html').write_text('<meta charset="utf-8"><h2>Test results (unseen writers)</h2>' + report.round(4).to_html()
                                 + '<ul>' + ''.join(f'<li>{"PASS" if ok else "FAIL"} {k}</li>' for k, ok in TARGETS) + '</ul>', encoding='utf-8')