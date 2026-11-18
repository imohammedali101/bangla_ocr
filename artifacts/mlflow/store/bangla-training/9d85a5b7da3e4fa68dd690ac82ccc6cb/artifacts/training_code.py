class Block(nn.Module):
    def __init__(s, i, o, st):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(i, o, 3, st, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
        s.sc = nn.Identity() if st == 1 and i == o else nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o))
    def forward(s, x): return F.relu(s.f(x) + s.sc(x))

class CRNN(nn.Module):
    """(B,1,64,W) -> (B,W/4,V) logits"""
    def __init__(s, nv=len(VOCAB)):
        super().__init__()
        s.cnn = nn.Sequential(nn.Conv2d(1, 24, 3, 1, 1, bias=False), nn.BatchNorm2d(24), nn.ReLU(True), nn.Conv2d(24, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
                              Block(32, 32, 1), Block(32, 64, 2), Block(64, 128, (2, 1)), Block(128, 192, (2, 1)))
        s.proj = nn.Sequential(nn.Conv1d(768, 256, 1), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(.1))
        s.rnn = nn.GRU(256, 128, 2, batch_first=True, bidirectional=True, dropout=.1)
        s.fc, s.aux = nn.Linear(256, nv), nn.Linear(256, 84)
    def feats(s, x): return s.rnn(s.proj(s.cnn(x).flatten(1, 2)).transpose(1, 2))[0]
    def forward(s, x): return s.fc(s.feats(x))

def seq_loss(m, b):
    x, tg, tl, il, sp = (v.to(DEV, non_blocking=True) for v in b)
    h = m.feats(x); lp = m.fc(h).float().log_softmax(2)
    loss = F.ctc_loss(lp.transpose(0, 1), tg, il, tl, zero_infinity=True)
    if len(sp):
        t = torch.arange(h.shape[1], device=DEV); msk = ((t >= sp[:, 2:3]) & (t <= sp[:, 3:4])).float()[..., None]
        hp = (h[sp[:, 0]].float() * msk).sum(1) / msk.sum(1).clamp(min=1)
        loss = loss + .3 * F.cross_entropy(m.aux(hp), sp[:, 1], label_smoothing=.1)
    return loss

@torch.no_grad()
def run(model, imgs, bs=128):
    """Raw logits (T, V) for each image; sorted by width to minimise padding."""
    model.eval(); res = [None] * len(imgs); order = np.argsort([im.shape[1] for im in imgs])
    for j in range(0, len(imgs), bs):
        b = order[j:j + bs]; x = torch.zeros(len(b), 1, 64, max(imgs[k].shape[1] for k in b))
        for r, k in enumerate(b): x[r, 0, :, :imgs[k].shape[1]] = torch.from_numpy(imgs[k]) / 255.
        with torch.autocast(**AMP): lg = model(x.to(DEV)).float().cpu().numpy()
        for r, k in enumerate(b): res[k] = lg[r, :imgs[k].shape[1] // 4]
    return res

def macro_f1(pred, y, n=84):
    cm = np.bincount(y * n + pred, minlength=n * n).reshape(n, n).astype(float); has = cm.sum(1) > 0
    return float((2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1))[has].mean())

# validation: synthetic mix (exact match, greedy) + REAL validation glyphs (top-1) -> the checkpoint score
VAL = [Synth(va, seed=1).sample() for _ in range(200 if CFG['FAST'] else 1000)]
vg = np.random.default_rng(0).choice(va, min(len(va), 3000), replace=False); VG = [prep_line(X[i]) for i in vg]
def score(m):
    ex = np.mean([greedy(softmax(l)) == tuple(v[1]) for l, v in zip(run(m, [v[0] for v in VAL]), VAL)])
    gl = np.mean([char_probs(softmax(l))[0].argmax() == Y[i] for l, i in zip(run(m, VG), vg)]); return .5 * ex + .5 * gl

def fit(name, model, epochs, steps, batch_fn, loss_fn, score_fn, lr):
    """AdamW + one-cycle + AMP + EMA. Resumes from the last epoch; returns the best EMA weights."""
    model.to(DEV); best_p, ck_p = OUT / f'{name}.pt', OUT / 'ckpt' / f'{name}.ckpt'
    if best_p.exists(): model.load_state_dict(torch.load(best_p, map_location=DEV)); return model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps, pct_start=.05)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(.999), use_buffers=True)
    scaler, ep0, best = torch.amp.GradScaler(enabled=AMP['enabled']), 0, -1.
    if ck_p.exists():
        ck = torch.load(ck_p, map_location=DEV, weights_only=False)
        for k, o in dict(model=model, opt=opt, sch=sch, ema=ema, scaler=scaler).items(): o.load_state_dict(ck[k])
        ep0, best = ck['ep'], ck['best']
    for ep in range(ep0, epochs):
        model.train(); t0, tot = time.time(), 0.
        for it in range(steps):
            with torch.autocast(**AMP): loss = loss_fn(model, batch_fn(ep, it))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.); scaler.step(opt); scaler.update(); sch.step()
            ema.update_parameters(model); tot += loss.detach()
        s = score_fn(ema.module.eval())
        if s > best: best = s; torch.save(ema.module.state_dict(), OUT / 'ckpt' / f'{name}_best.pt')
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), ema=ema.state_dict(), scaler=scaler.state_dict(), ep=ep + 1, best=best), ck_p)
        print(f'{name} {ep + 1}/{epochs} loss {float(tot) / steps:.4f} val-score {s:.4f} best {best:.4f} {time.time() - t0:.0f}s', flush=True)
    model.load_state_dict(torch.load(OUT / 'ckpt' / f'{name}_best.pt', map_location=DEV)); torch.save(model.state_dict(), best_p)
    return model.eval()

model = CRNN(); print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters')
model = fit('model3', model, CFG['EP'], CFG['STEPS'], SeqData(TRAIN_IDS), seq_loss, score, CFG['LR'])