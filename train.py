#!/usr/bin/env python
"""train.py - train and test the three Bangla handwriting models; every step is recorded in MLflow.

    Model 1  Lekha-ResNet   one glyph in, 85 scores out  (the accurate character model; teacher of Model 2)
    Model 2  Lekha-Mobile   same job, much smaller       (MobileNetV3, distilled from Model 1)
    Model 3  Lekha-CRNN     a whole drawing in           (CNN + BiGRU + CTC: characters, numbers and WHOLE WORDS)

Commands
    python train.py                                                            # = extract (the default): notebooks in notebooks/ and ONNX files in models/ -> MLflow
    python train.py extract --v5 notebooks/bangla-hwr-v5.ipynb --model3-train notebooks/bangla-hwr-model3-f.ipynb --model3-test notebooks/bangla-hwr-model3-eval.ipynb
                                                                               # LOCAL, light: put what the executed notebooks trained and tested into MLflow
    python train.py predict drawing.png --expect মানুষ                         # LOCAL, light: run the registered models on a drawing, with timing (see predict_cli.py)
    python train.py train --data /path/to/BanglaLekha-Isolated                 # train all three models
    python train.py train --data ... --model crnn                              # one model (resnet | mobilenet | crnn, or a,b list)
    python train.py train --data ... --profile quick                           # fewer epochs (full | quick | smoke)
    python train.py train --data ... --reuse /path/to/old/output               # reuse model1/2/3.pt and the data cache found there
    python train.py test  --data /path/to/BanglaLekha-Isolated                 # test the saved ONNX models, log to MLflow
    python train.py import-notebook --model resnet --notebook bangla-hwr-v5.ipynb --onnx model1_resnet.onnx --reuse ...
                                                                               # record a model trained in a Kaggle notebook (see below)
    python train.py train --profile smoke --data ...                           # 2-minute smoke test in a scratch folder

Folders (fixed, see bangla_common.py): models/ (ONNX + config), artifacts/mlflow/ (MLflow), work/ (cache, checkpoints).
Look at the runs:  mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 0.0.0.0 --port 5000
"""
import argparse
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


if len(sys.argv) == 1 or sys.argv[1] == 'extract':      # `python train.py` = extract: only needs mlflow, numpy, pandas, matplotlib (no torch, no GPU, no dataset)
    from notebook_extract import main as _extract
    sys.exit(_extract(sys.argv[2:] if len(sys.argv) > 1 else []))
if len(sys.argv) > 1 and sys.argv[1] == 'predict':      # run the registered ONNX models on drawings: numpy, OpenCV, ONNX Runtime only
    from predict_cli import main as _predict
    sys.exit(_predict(sys.argv[2:]))


def check_dependencies():
    missing = []
    for mod, pip in (('torch', 'torch'), ('timm', 'timm'), ('mlflow', 'mlflow'), ('onnx', 'onnx'), ('onnxruntime', 'onnxruntime'),
                     ('cv2', 'opencv-python-headless'), ('skimage', 'scikit-image'), ('pandas', 'pandas'), ('matplotlib', 'matplotlib')):
        try:
            __import__(mod)
        except Exception as e:                                         # ImportError, or a version clash such as protobuf
            missing.append(f'{pip} ({type(e).__name__})')
    if missing:
        sys.exit('Missing or broken packages: ' + ', '.join(missing) + '\nFix: pip install -q -r requirements-train.txt   (training from scratch only; on Kaggle: turn Internet ON)')


check_dependencies()
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.morphology import skeletonize

import bangla_common as bc
from bangla_common import *          # labels, prep/prep_line, decoding helpers (see bangla_common.py)

# ------------------------------------------------------------------------------------------------ settings
PROFILES = {   # epochs per model, steps per CRNN epoch, batch sizes, share of writers used
    'full':  dict(epochs=dict(resnet=60, mobilenet=80, crnn=40), steps=1500, bs=256, bs_crnn=64),
    'quick': dict(epochs=dict(resnet=15, mobilenet=20, crnn=10), steps=1500, bs=256, bs_crnn=64),
    'smoke': dict(epochs=dict(resnet=2, mobilenet=2, crnn=2), steps=20, bs=64, bs_crnn=16),
}
KEYS = {'resnet': 'model1', 'mobilenet': 'model2', 'crnn': 'model3'}
ONNX_FILES = {'model1': 'model1_resnet.onnx', 'model2': 'model2_mobilenet.onnx', 'model3': 'model3_crnn.onnx'}
RUN_NAMES = {'model1': 'model1-resnet', 'model2': 'model2-mobilenet', 'model3': 'model3-crnn'}
CFG = dict(SEED=42, BS=64, BS_CHAR=256, EP=40, STEPS=1500, WORKERS=4, LR=2e-3, CLEAN=True, FAST=False)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = dict(device_type=DEV, dtype=torch.float16, enabled=DEV == 'cuda')
OUT = None            # scratch folder (cache, checkpoints), set in setup()
REUSED = set()        # cache/weight files copied in from --reuse in this invocation
KW = {'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}


def adir(key):
    """Per-model folder for the files logged as MLflow artifacts (so they get the names curves.png, confusion_matrix.png, ...)."""
    d = OUT / 'artifacts' / key; d.mkdir(parents=True, exist_ok=True); return d


def setup(args):
    """Fix the folders, seeds and CFG for this invocation."""
    global OUT
    prof = PROFILES[args.profile]
    if args.profile == 'smoke':                       # scratch copy of the whole project layout: never touches real models/runs
        bc.set_home(Path(__file__).resolve().parent / 'work' / 'smoke')
        CFG.update(FAST=True)
    OUT = bc.BASE_DIR / 'work'
    for d in (OUT / 'ckpt', bc.MODELS_DIR, bc.MLFLOW_DIR):
        d.mkdir(parents=True, exist_ok=True)
    CFG.update(BS=prof['bs_crnn'], BS_CHAR=prof['bs'], STEPS=prof['steps'], WORKERS=args.workers if args.workers >= 0 else (0 if sys.platform == 'win32' else min(4, os.cpu_count() or 1)))   # Windows starts workers with spawn, which cannot see the data globals: 0 there
    random.seed(CFG['SEED']); np.random.seed(CFG['SEED']); torch.manual_seed(CFG['SEED'])
    torch.backends.cudnn.benchmark = True
    if args.reuse:                                    # take cache + weights from an earlier run (for example a Kaggle output)
        for f in ('data.npz', 'meta.csv', 'cl_flags.npy', 'model1.pt', 'model2.pt', 'model3.pt'):
            src = next(iter(Path(args.reuse).rglob(f)), None)
            if src is not None and not (OUT / f).exists():
                shutil.copy(src, OUT / f); REUSED.add(f)
                print(f'reused {f} from {src.parent}')
    print(f'device {DEV} | profile {args.profile} | project {bc.BASE_DIR}')


# ------------------------------------------------------------------------------------------------ lexicon
def isA(ids): return not any(VOCAB[t] in KARS for t in ids)       # tier A: dataset glyphs only, tier B: needs vowel signs


def build_lexicon(extra_file=None):
    words = list(DEFAULT_WORDS) + (Path(extra_file).read_text(encoding='utf-8').split() if extra_file else [])
    return sorted({N(w) for w in words if '\u0981' not in w and to_visual(w)})    # ঁ sits above a letter: not renderable as a glyph


LEX = build_lexicon()
LEXS = make_lex(LEX)                                                  # decoding lexicon = all words
LEX_TEST, LEX_VAL = LEX[0::5], LEX[2::10]                             # held out: the model never trains on these words
LEX_TRAIN = [w for w in LEX if w not in set(LEX_TEST) | set(LEX_VAL)]


# ------------------------------------------------------------------------------------------------ data
def find_assets(data_dir):
    root = next((Path(r) for r, d, _ in os.walk(data_dir) if {'1', '84'} <= set(d)), None)
    if root is None:
        sys.exit(f'No folder with the class folders 1..84 under {data_dir}. Pass --data /path/to/BanglaLekha-Isolated')
    xl = next((p for b in (root, root.parent, root.parent.parent) for p in b.glob('*.xlsx') if 'form' in p.name.lower()), None)
    return root, xl


def quality_map(xl):
    """Form-based Marking.xlsx -> {form serial: average mark (0-5)}; {} if unavailable."""
    try:
        t = pd.read_excel(xl); t.columns = [str(c).strip() for c in t.columns]
        f = next(c for c in t.columns if 'form' in c.lower()); a = next(c for c in t.columns if 'average' in c.lower())
        return dict(zip(pd.to_numeric(t[f], errors='coerce').fillna(-1).astype(int), pd.to_numeric(t[a], errors='coerce')))
    except Exception as e:
        print('form-quality table unavailable:', e)
        return {}


def load(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE); g = ink(g if g is not None else np.zeros((64, 64), np.uint8))
    c, sk = prep(g, info=True)
    return c, sk, (g > 0).mean()


gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEV)

def cnn():
    cbr = lambda i, o: [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2)]
    return nn.Sequential(*cbr(1, 32), *cbr(32, 64), *cbr(64, 128), *cbr(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 84)).to(DEV)

def train_cnn(ids, epochs, lr=2e-3):
    m, Xg, yg = cnn(), gpu(X[ids]), gpu(Y[ids]); bs = min(256, max(8, len(ids) // 4)); n = len(ids) // bs
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=.05); sc = torch.amp.GradScaler(enabled=AMP['enabled'])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * n)
    for _ in range(epochs):
        m.train(); p = torch.randperm(len(ids), device=DEV)
        for i in range(n):
            j = p[i * bs:(i + 1) * bs]
            with torch.autocast(**AMP): loss = F.cross_entropy(m(Xg[j][:, None].float() / 255), yg[j])
            opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step()
    return m.eval()

@torch.no_grad()
def cnn_probs(m, A, bs=1024):
    return torch.cat([m(gpu(A[j:j + bs])[:, None].float() / 255).softmax(1).cpu() for j in range(0, len(A), bs)])

# V5: counterfactual glyphs, appended to the glyph bank under their new class
def split_dot(g):
    n, lab, st, cen = cv2.connectedComponentsWithStats((g > 64).astype(np.uint8), 8)
    if n < 3: return None
    k = 1 + st[1:, 4].argmin()
    if st[k, 4] > .08 * st[1:, 4].sum() or cen[k, 1] < .55 * g.shape[0]: return None
    return g * (lab != k), g * (lab == k)

def counterfactuals(cap=1500, rng=np.random.default_rng(1)):
    out = []
    for dotted, plain in [('ড়', 'ড'), ('ঢ়', 'ঢ'), ('য়', 'য'), ('র', 'ব')]:
        cd, cp = CLASSES.index(N(dotted)), CLASSES.index(N(plain)); dots = []
        for i in trc[Y[trc] == cd][:cap]:
            r = split_dot(X[i])
            if r: out.append((prep(r[0]), cp, i)); dots.append(crop(r[1]))
        for i in trc[Y[trc] == cp][:cap if dots else 0]:
            g, d = X[i].copy(), dots[rng.integers(len(dots))]; ys_, xs_ = np.nonzero(g)
            y0 = min(ys_.max() + 3, 64 - d.shape[0]); x0 = int(np.clip(xs_.mean() - d.shape[1] / 2, 0, 64 - d.shape[1]))
            np.maximum(g[y0:y0 + d.shape[0], x0:x0 + d.shape[1]], d, out=g[y0:y0 + d.shape[0], x0:x0 + d.shape[1]]); out.append((prep(g), cd, i))
    return out


def prepare_data(args):
    """Load (or build and cache) the glyph bank, split by writer, clean (V1), add counterfactual glyphs (V5).
    Fills the module globals used by the generator and the models: X S SW SK Y WR QW df tr va te trc TRAIN_IDS."""
    global df, X, SK, S, SW, Y, WR, QW, tr, va, te, trc, TRAIN_IDS, n0
    if (OUT / 'data.npz').exists() and (OUT / 'meta.csv').exists():
        D = dict(np.load(OUT / 'data.npz')); df = pd.read_csv(OUT / 'meta.csv', dtype={'writer': str})
        print('data cache found: no dataset needed')
    else:
        root, xl = find_assets(args.data or '/kaggle/input')
        QM = quality_map(xl) if xl else {}
        rows = []
        for f in root.glob('*/*'):
            if not (f.parent.name.isdigit() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')): continue
            p, k = f.stem.split('_'), int(f.parent.name)
            if len(p) >= 7 and p[6].isdigit() and int(p[6]) != k: continue          # file name and folder must agree
            rows.append((str(f), k - 1, '_'.join(p[:6]) if len(p) >= 6 else f.stem, *(p[i] if len(p) > i else '' for i in (0, 2, 3, 5))))
        df = pd.DataFrame(rows, columns=['path', 'y', 'writer', 'district', 'gender', 'age', 'form'])
        df['quality'] = pd.to_numeric(df.form, errors='coerce').map(QM).fillna(3.0)
        print(f'{len(df)} images · {df.writer.nunique()} writers · {df.y.nunique()} classes · quality table: {len(QM)} forms')
        if args.profile == 'smoke':
            wr = df.writer.drop_duplicates(); df = df[df.writer.isin(wr.sample(min(len(wr), max(30, len(wr) // 25)), random_state=0))].reset_index(drop=True)
        with ThreadPoolExecutor(16) as ex: R = list(ex.map(load, df.path, chunksize=256))
        X, SK, INK = (np.stack([r[i] for r in R]) for i in range(3)); del R
        key = pd.Series([hash((int(y), w, (cv2.resize(x, (32, 32), interpolation=cv2.INTER_AREA) > 64).tobytes())) for y, w, x in zip(df.y, df.writer, X)])
        keep = ((INK > .01) & ~key.duplicated()).to_numpy()                          # V1: blanks and duplicates within (class, writer)
        print(f'V1: removed {(INK <= .01).sum()} blanks and {int((~keep).sum() - (INK <= .01).sum())} duplicates')
        df, X, SK = df[keep].reset_index(drop=True), X[keep], SK[keep]
        w = df.writer.drop_duplicates().sample(frac=1, random_state=CFG['SEED']).to_numpy()
        df['split'] = df.writer.map({k: 'train' if i < .8 * len(w) else 'val' if i < .9 * len(w) else 'test' for i, k in enumerate(w)})   # by WRITER
        with ThreadPoolExecutor(16) as ex: S = np.stack(list(ex.map(lambda x: skeletonize(x > 64), X, chunksize=256))).astype(np.uint8)
        D = dict(X=X, SK=SK, S=S, SW=(X > 64).sum((1, 2)) / np.maximum(S.sum((1, 2)), 1)); np.savez(OUT / 'data.npz', **D); df.to_csv(OUT / 'meta.csv', index=False)
    X, SK, S, SW, Y = D['X'], D['SK'], D['S'], D['SW'], df.y.to_numpy()
    WR = pd.factorize(df.writer)[0]; QW = pd.Series(df.quality.to_numpy()).groupby(WR).first().to_dict()
    idx = {s: np.flatnonzero(df.split.to_numpy() == s) for s in ('train', 'val', 'test')}
    tr, va, te = idx['train'], idx['val'], idx['test']
    print({k: len(v) for k, v in idx.items()}, '| writers:', df.groupby('split').writer.nunique().to_dict())
    # V1: likely mislabels, from out-of-fold predictions of a throw-away CNN (confident learning)
    if (OUT / 'cl_flags.npy').exists(): flag = np.load(OUT / 'cl_flags.npy')
    elif CFG['CLEAN']:
        half = np.random.default_rng(0).random(len(tr)) < .5; P = np.zeros((len(tr), 84), np.float32)
        for f in (0, 1):
            m = train_cnn(tr[half != f], 1 if CFG['FAST'] else 4); P[half == f] = cnn_probs(m, X[tr[half == f]]).numpy()
        flag = (P.argmax(1) != Y[tr]) & (P.max(1) > .9) & (P[np.arange(len(tr)), Y[tr]] < .05); np.save(OUT / 'cl_flags.npy', flag)
    else: flag = np.zeros(len(tr), bool)
    print(f'V1: {flag.sum()} likely mislabels removed from train'); trc = tr[~flag]
    # V5: counterfactual glyphs, appended to the glyph bank under their new class
    cf = counterfactuals(); n0 = len(X)
    if cf:
        Xc = np.stack([c[0] for c in cf]); src = np.array([c[2] for c in cf]); Sc = np.stack([skeletonize(x > 64) for x in Xc]).astype(np.uint8)
        X, S = np.concatenate([X, Xc]), np.concatenate([S, Sc]); Y, WR = np.r_[Y, [c[1] for c in cf]], np.r_[WR, WR[src]]
        SW, SK = np.r_[SW, SW[src]], np.r_[SK, SK[src]]
    TRAIN_IDS = np.r_[trc, np.arange(n0, len(X))]
    print(f'V5: {len(X) - n0} counterfactual glyphs · glyph bank: {len(TRAIN_IDS)} train glyphs')


# ------------------------------------------------------------------------------------------------ Models 1 and 2: character data
def rejects(n=2000, rng=np.random.default_rng(2)):
    """V7: 'not a character' glyphs: fragments, two glyphs on top of each other, scribbles, dots, blanks."""
    out = []
    for _ in range(n):
        a, k = X[trc[rng.integers(len(trc))]], rng.integers(5)
        if k == 0: h, w = rng.integers(14, 40, 2); y, x = rng.integers(0, 64 - h), rng.integers(0, 64 - w); g = a[y:y + h, x:x + w]
        elif k == 1: g = np.maximum(a, np.roll(X[trc[rng.integers(len(trc))]], int(rng.integers(-24, 24)), 1))
        elif k == 2:
            g = np.zeros((64, 64), np.uint8); p = (32 + np.cumsum(rng.normal(0, 7, (int(rng.integers(3, 12)), 2)), 0)).clip(2, 61)
            cv2.polylines(g, [p.astype(np.int32)], False, 255, int(rng.integers(2, 5)))
        elif k == 3:
            g = np.zeros((64, 64), np.uint8)
            for _ in range(rng.integers(1, 5)): cv2.circle(g, tuple(int(v) for v in rng.integers(4, 60, 2)), int(rng.integers(1, 4)), 255, -1)
        else: g = np.zeros((64, 64), np.uint8)
        out.append(prep(g))
    return np.stack(out)


def augment(x, sk):
    """GPU augmentation: V3 canvas re-draw, V4 writer slant + pen width, rotation, scale, shift, elastic warp, erasing, noise."""
    B, d = len(x), x.device; u = lambda *s: torch.rand(*s, device=d)
    k = random.choice((3, 5))                                                  # V3: brush size for this batch
    canvas = F.avg_pool2d(F.max_pool2d(sk, k, 1, k // 2), 3, 1, 1).clamp(0, 1)
    x = torch.where(u(B, 1, 1, 1) < .3, canvas, x)
    st = WSTYLE[torch.randint(len(WSTYLE), (B,), device=d)]                    # V4
    pw = st[:, 1].view(B, 1, 1, 1)
    x = torch.where(pw > 0, F.max_pool2d(x, 3, 1, 1), torch.where(pw < 0, -F.max_pool2d(-x, 3, 1, 1), x))
    a, s = (u(B) * 2 - 1) * math.radians(10), .85 + u(B) * .25
    t = (u(2, B) * 2 - 1) * .16
    th = torch.stack([torch.stack([a.cos() / s, -a.sin() / s + st[:, 0], t[0]], 1), torch.stack([a.sin() / s, a.cos() / s, t[1]], 1)], 1)
    g = F.affine_grid(th, x.shape, align_corners=False)
    g = g + F.interpolate(torch.randn(B, 2, 4, 4, device=d) * .03, size=x.shape[2:], mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
    x = F.grid_sample(x, g, align_corners=False)
    c = torch.randint(0, 64, (2, B, 1, 1), device=d); r = torch.randint(3, 10, (B, 1, 1), device=d); ar = torch.arange(64, device=d)
    box = ((ar.view(1, -1, 1) - c[0]).abs() < r) & ((ar.view(1, 1, -1) - c[1]).abs() < r) & (u(B, 1, 1) < .3)
    return (x.masked_fill(box[:, None], 0) + torch.randn_like(x) * .03 * u(B, 1, 1, 1)).clamp(0, 1)


class CharData:
    """Batches for the character models; tracks the loss of every sample for hard-example mining."""
    def __init__(s, X, Sk, y, epochs, hard_from=None):
        s.X, s.S, s.y, s.ep, s.hard = X, Sk, y, epochs, hard_from
        s.n, s.loss = len(y), torch.zeros(len(y), device=DEV); s.steps = max(1, s.n // CFG['BS_CHAR'])
    def __call__(s, ep, it):
        bs = CFG['BS_CHAR']
        if it == 0:
            hard = s.hard is not None and ep >= s.hard
            s.order = torch.multinomial(1 + (s.loss >= s.loss.quantile(.9)).float(), s.n, True) if hard else torch.randperm(s.n, device=DEV)
        i = s.order[it * bs:(it + 1) * bs]; x, y = s.X[i][:, None].float() / 255, s.y[i]
        x = augment(x, s.S[i][:, None].float())
        if ep < .9 * s.ep and random.random() < .3:                              # CutMix, off for the last 10%
            lam, p = np.random.beta(1, 1), torch.randperm(len(i), device=DEV); h = int(64 * math.sqrt(1 - lam))
            cy, cx = np.random.randint(64, size=2); y0, y1, x0, x1 = max(cy - h // 2, 0), min(cy + h // 2, 64), max(cx - h // 2, 0), min(cx + h // 2, 64)
            x[..., y0:y1, x0:x1] = x[p][..., y0:y1, x0:x1]
            return i, x, y, y[p], 1 - (y1 - y0) * (x1 - x0) / 4096
        return i, x, y, y, 1.


def char_loss(data, teacher=None, T=4.):
    def f(m, b):
        i, x, ya, yb, lam = b; out = m(x)
        l = lam * F.cross_entropy(out, ya, label_smoothing=.1, reduction='none') + (1 - lam) * F.cross_entropy(out, yb, label_smoothing=.1, reduction='none')
        data.loss[i] = l.detach().float(); l = l.mean()
        if teacher is None: return l
        with torch.no_grad(): t = teacher(x)
        return .5 * l + .5 * T * T * F.kl_div(F.log_softmax(out.float() / T, 1), F.softmax(t.float() / T, 1), reduction='batchmean')
    return f


def build_char_training():
    """Training tensors for Models 1 and 2 (kept on the GPU): glyph bank (V1-V5) + reject glyphs (V7)."""
    global WSTYLE
    ws = pd.DataFrame(dict(w=WR[trc], sk=SK[trc], sw=SW[trc])).groupby('w').median()           # V4: writer styles
    pen = np.select([ws.sw > ws.sw.median() * 1.15, ws.sw < ws.sw.median() * .85], [1, -1], 0)
    WSTYLE = gpu(np.c_[ws.sk.to_numpy(), pen].astype(np.float32))
    Xr = rejects()
    Sr = np.stack([skeletonize(x > 64) for x in Xr]).astype(np.uint8)
    ids = TRAIN_IDS
    Xt, St, Yt = np.concatenate([X[ids], Xr]), np.concatenate([S[ids], Sr]), np.r_[Y[ids], np.full(len(Xr), REJ)]
    if CFG['FAST']:                                               # smoke test only: a small slice, so it finishes in minutes on a CPU
        k = np.random.default_rng(0).permutation(len(Yt))[:768]; Xt, St, Yt = Xt[k], St[k], Yt[k]
    return gpu(Xt), gpu(St), gpu(Yt)


def body(strides, w=(32, 64, 128, 256)):
    """ResNet-18 layout at half width; strides per stage."""
    L, c = [nn.Conv2d(1, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True)], w[0]
    for o, st in zip(w, strides): L += [Block(c, o, st), Block(o, o, 1)]; c = o
    return nn.Sequential(*L)


def make_resnet():
    return nn.Sequential(body((1, 2, 2, 2)), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.2), nn.Linear(256, NCLS))


def make_mobilenet():
    import timm
    m = timm.create_model('mobilenetv3_small_100', pretrained=False, in_chans=1, num_classes=NCLS, drop_rate=.2)
    m.conv_stem.stride = (1, 1)                                                # keep the detail of 64x64 glyphs
    return m


@torch.no_grad()
def logits(m, A, bs=1024):
    m.eval(); out = []
    for j in range(0, len(A), bs):
        with torch.autocast(**AMP): out.append(m(torch.as_tensor(A[j:j + bs], device=DEV)[:, None].float() / 255).float().cpu())
    return torch.cat(out)


AA, I, II, U, UU, RI, E_, AI, AU = KARS
BELOW = {U, UU, RI}
SIGNS = ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
CONS = [N(c) for c in CONSONANTS[:35]]; CONJ = [N(c) for c in COMPOUNDS]; MODS = [N(c) for c in 'ংঃৎ']; VOW = [N(c) for c in VOWELS]
MIX = {0: dict(num=.30, char=.20, lex=.25, pseudo=.20, rej=.05), 1: dict(num=.25, char=.18, lex=.22, pseudo=.30, rej=.05),
       2: dict(num=.22, char=.20, lex=.20, pseudo=.33, rej=.05)}                    # curriculum: simple -> full
LENP = [.06, .18, .20, .20, .14, .10, .07, .05]                                     # number length 1..8
BRUSH = {k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) for k in (2, 3, 4)}
EL = np.array(KARS)

def reslant(g, sk):
    """Re-apply a glyph's own measured slant (inverse of the deslanting in `prep`): approximates the original, un-normalised image."""
    h, w = g.shape; pad = int(abs(sk) * h) + 1
    return cv2.warpAffine(g, np.float32([[1, sk, pad - sk * h / 2], [0, 1, 0]]), (w + 2 * pad, h))

def kar_strokes(k, hw):
    """Vowel-sign strokes in units of glyph height. x: from the sign's slot (from the host for signs below); y: 0 = headline, 1 = host bottom."""
    e = [(.25, 0), (.08, .25), (.22, .5), (.08, .75), (.25, .95)]
    return {AA: [[(.15, 0), (.15, 1)]], I: [[(.15, 1), (.15, 0), (.25, -.35), (.3 + .5 * hw, -.42), (.3 + .9 * hw, -.15)]],
            II: [[(.15, 1), (.15, 0), (.05, -.35), (-.5 * hw, -.42), (-.9 * hw, -.15)]], E_: [e],
            AI: [e, [(.05, -.05), (.15, -.3), (.3, -.18)]], AU: [[(.15, 1), (.15, 0)], [(.15, 0), (.25, -.3), (.35, -.12)]],
            U: [[(.5 * hw, .95), (.35 * hw, 1.2), (.55 * hw, 1.38), (.75 * hw, 1.18)]],
            UU: [[(.5 * hw, .95), (.3 * hw, 1.18), (.6 * hw, 1.3), (.35 * hw, 1.48), (.7 * hw, 1.45)]],
            RI: [[(.55 * hw, .95), (.4 * hw, 1.15), (.65 * hw, 1.28), (.45 * hw, 1.45)]]}[k]

def chaikin(p, n=2):
    for _ in range(n):
        q, r = .75 * p[:-1] + .25 * p[1:], .25 * p[:-1] + .75 * p[1:]
        p = np.vstack([p[:1], np.stack([q, r], 1).reshape(-1, 2), p[-1:]])
    return p

def headline(g):
    rows = (g > 64).sum(1); r = int(rows[:max(1, int(.35 * len(rows)))].argmax()); return r, rows[r] > .5 * g.shape[1]

def pseudo(rng, level):
    """Random but valid syllable pattern: [vowel | consonant/conjunct (+ vowel sign)]* (+ ং ঃ ৎ). Forces glyph reading, not word memorising."""
    out = ''
    for k in range(int(rng.integers(1, 5))):
        if k == 0 and rng.random() < .12: out += VOW[rng.integers(len(VOW))]; continue
        out += CONJ[rng.integers(24)] if level >= 2 and rng.random() < .15 else CONS[rng.integers(len(CONS))]
        if level >= 1 and rng.random() < .55: out += SIGNS[rng.integers(len(SIGNS))]
    return out + (MODS[rng.integers(3)] if rng.random() < .1 else '')

class Synth:
    """sample(kind=None) -> (image, token ids, spans [(class, frame0, frame1)], text, kind)"""
    def __init__(s, ids, seed=0, level=2, aug=True, pool=None):
        s.by, s.cls = {}, {}
        for i in ids: s.by.setdefault(WR[i], {}).setdefault(Y[i], []).append(i); s.cls.setdefault(Y[i], []).append(i)
        s.writers, s.level, s.aug, s.rng = list(s.by), level, aug, np.random.default_rng(seed)
        g = lambda a: pd.Series(a[ids]).groupby(WR[ids]).median()
        s.pen, s.slant = g(SW).to_dict(), g(SK).to_dict(); s.style = [(s.slant[w], s.pen[w] / np.median(list(s.pen.values()))) for w in s.writers]
        s.clean = [w for w in s.writers if QW.get(w, 3.) >= 2.5] or s.writers
        s.pool = pool or LEX_TRAIN; s.poolA = [w for w in s.pool if isA([TID[t] for t in to_visual(w)])] or s.pool

    def sample(s, kind=None):
        for _ in range(5):
            try: return s._sample(kind)
            except Exception as e: s.fails, s.last = getattr(s, 'fails', 0) + 1, repr(e)
        raise RuntimeError(f'sample() failed 5 times in a row: {s.last}')

    def _sample(s, kind=None):
        r, m = s.rng, MIX[s.level]; kind = kind or r.choice(list(m), p=list(m.values()))
        if kind == 'rej': return (*s.reject(), '', kind)
        if kind == 'num': text = ''.join(r.choice(DIGITS, int(r.choice(8, p=LENP)) + 1))
        elif kind == 'char':
            c = int(r.integers(84)); text = CLASSES[c]
            if r.random() < .5: return (*s.direct(c), text, kind)                  # half: a glyph exactly as the glyph test feeds it
        elif kind == 'lex': pool = s.poolA if s.level == 0 else s.pool; text = pool[r.integers(len(pool))]
        else: text = pseudo(r, s.level)
        return (*s.render(text), text, kind)

    def direct(s, c):
        """Single real glyph as the glyph test feeds it: tight crop, upright or in its original slant, usually lightly augmented."""
        r = s.rng; w = s.writers[r.integers(len(s.writers))]
        L = s.by[w].get(c) or s.cls.get(c) or s.cls[next(iter(s.cls))]; i = L[r.integers(len(L))]
        g = cv2.GaussianBlur(cv2.dilate(S[i] * 255, BRUSH[int(r.choice((2, 3, 4)))]), (3, 3), 0) if s.aug and r.random() < .3 else X[i]
        if r.random() < .5: g = reslant(g, float(SK[i]))
        img = prep_line(g); img = s.finish(img) if r.random() < .7 else img
        return img, [c + 1], [(c, 0, img.shape[1] // 4 - 1)]

    def glyph(s, w, c, h):
        r, L = s.rng, s.by[w].get(c) or s.cls.get(c) or s.cls[next(iter(s.cls))]; i = L[r.integers(len(L))]
        g = cv2.GaussianBlur(cv2.dilate(S[i] * 255, BRUSH[int(r.choice((2, 3, 4)))]), (3, 3), 0) if s.aug and r.random() < .3 else X[i]   # V3
        g = crop(g); g = g if g is not None else crop(X[i]); return i, cv2.resize(g, (max(4, min(int(1.8 * h), round(g.shape[1] * h / g.shape[0]))), h), interpolation=cv2.INTER_AREA)

    def aa(s, w, h):  # V6: the real া stroke = right part of the same writer's আ
        L = s.by[w].get(CLASSES.index(N('আ'))); g = crop(X[L[0]]) if L else None; g = crop(g[:, int(.68 * g.shape[1]):]) if g is not None else None
        return None if g is None or g.shape[1] > .5 * g.shape[0] else cv2.resize(g, (max(2, round(g.shape[1] * h / g.shape[0])), h))

    def render(s, text):
        r, toks = s.rng, to_visual(text) or ['ক']; is_num = all(k in DIGITS for k in toks)
        pool = s.clean if s.level == 0 and r.random() < .7 else s.writers; w = pool[r.integers(len(pool))]
        h, HL = int(r.integers(38, 50)), 44
        t = max(2, round(s.pen.get(w, 4) * h / 52 * r.uniform(.8, 1.2))); lo = -.12 if s.level >= 2 and not is_num else .02
        items, x, prev = [], 40, None
        for k in toks:
            if k in BELOW: items.append(dict(k=k, x0=x, x1=x, m=False)); continue
            if k in KARS:
                aa = s.aa(w, h) if k == AA and r.random() < .5 else None; wd = aa.shape[1] if aa is not None else int(.3 * h)
                items.append(dict(k=k, x0=x, x1=x + wd, m=True, **({} if aa is None else dict(img=aa, y0=HL)))); x += wd
            else:
                gi, g = s.glyph(w, TID[k] - 1, h); rr, m = headline(g); m = m and k not in DIGITS
                items.append(dict(k=k, img=g, gi=gi, x0=x, x1=x + g.shape[1], y0=HL - rr if m else HL, m=m)); x += g.shape[1]
            x += int(h * (max(.08, r.uniform(lo, .2)) if k == prev else r.uniform(lo, .2))); prev = k
        cv = np.zeros((140, x + 40), np.uint8)
        for it in items:
            if 'img' in it: g = it['img']; sl = cv[it['y0']:it['y0'] + g.shape[0], it['x0']:it['x0'] + g.shape[1]]; np.maximum(sl, g, out=sl)
        for j, it in enumerate(items):                                       # procedural vowel signs
            if it['k'] not in KARS or 'img' in it: continue
            near = range(j + 1, len(items)) if it['k'] in PRE else range(j - 1, -1, -1)
            host = next((items[q] for q in near if 'gi' in items[q]), None)
            if host is None: continue
            hw, hb = (host['x1'] - host['x0']) / h, host['y0'] + host['img'].shape[0] - HL; ox = host['x0'] if it['k'] in BELOW else it['x0']
            for st in kar_strokes(it['k'], hw):
                p = np.array(st, float); p = np.c_[ox + p[:, 0] * h, HL + p[:, 1] * hb] + r.normal(0, .04 * h, (len(p), 2))
                cv2.polylines(cv, [chaikin(p).round().astype(np.int32)], False, 255, t, cv2.LINE_AA)
        flat = [it for it in items if it['k'] not in BELOW]                  # joined headline
        for a, b in zip(flat, flat[1:]):
            if a['m'] and b['m']: y = HL + int(r.integers(-1, 2)); cv2.line(cv, (a['x0'], y), (b['x1'], y), 255, t, cv2.LINE_AA)
        sl_ = r.choice([st[0] for st in s.style]) * r.uniform(.7, 1.3) + r.normal(0, .05) if s.aug else 0.   # V4: another writer's slant
        an = math.radians(r.uniform(-3, 3)) if s.aug else 0.
        A = np.float32([[math.cos(an), -math.sin(an) + sl_, 0], [math.sin(an), math.cos(an), 0]])
        A[:, 2] = np.float32([cv.shape[1] / 2, 70]) - A[:, :2] @ np.float32([cv.shape[1] / 2, 70]); cv = cv2.warpAffine(cv, A, cv.shape[::-1])
        ys, xs = np.nonzero(cv > 50)
        if not len(xs): return s.reject()
        img = prep_line(cv); sx = img.shape[1] / (xs.max() - xs.min() + 9)
        fx = lambda v: int(((A[0, 0] * v + A[0, 1] * (HL + h / 2) + A[0, 2]) - xs.min() + 4) * sx / 4)
        Tn = img.shape[1] // 4; spans = []
        for it in items:
            if 'gi' in it: a = int(np.clip(fx(it['x0']), 0, Tn - 1)); spans.append((TID[it['k']] - 1, a, int(np.clip(fx(it['x1']) - 1, a, Tn - 1))))
        return s.finish(img), [TID[k] for k in toks], spans

    def finish(s, img):
        """Whole-word augmentation: elastic warp, stroke width, blur, noise, contrast (V4 pen width via the style table)."""
        if not s.aug: return img
        r, (H, W) = s.rng, img.shape; img = img.astype(np.float32)
        if r.random() < .5:
            f = cv2.resize(r.normal(0, 1, (3, max(4, W // 40), 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) * r.uniform(.5, 2.)
            xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)); img = cv2.remap(img, xs + f[..., 0], ys + f[..., 1], cv2.INTER_LINEAR)
        pen = s.style[r.integers(len(s.style))][1]
        if pen > 1.15 and r.random() < .6: img = cv2.dilate(img, BRUSH[2])
        elif pen < .85 and r.random() < .6: img = cv2.erode(img, BRUSH[2])
        if r.random() < .4: img = cv2.GaussianBlur(img, (3, 3), r.uniform(.4, 1.))
        if r.random() < .3: img = img + r.normal(0, r.uniform(2, 12), img.shape)
        return np.clip(img * r.uniform(.75, 1.), 0, 255).astype(np.uint8)

    def reject(s):
        """V7: scribbles, glyph fragments, dots, blanks -> empty label."""
        r, kind, W = s.rng, int(s.rng.integers(4)), int(s.rng.integers(32, 260)); g = np.zeros((64, W), np.uint8)
        if kind == 0:
            for _ in range(int(r.integers(1, 4))):
                p = (np.cumsum(r.normal(0, 7, (int(r.integers(4, 14)), 2)), 0) + [r.uniform(8, W - 8), 32]).clip(2, [W - 2, 61])
                cv2.polylines(g, [p.astype(np.int32)], False, 255, int(r.integers(2, 5)))
        elif kind == 1:
            a = X[TRAIN_IDS[r.integers(len(TRAIN_IDS))]]; h_, w_ = r.integers(14, 36, 2); y, x = r.integers(0, 64 - h_), r.integers(0, 64 - w_)
            f = crop(a[y:y + h_, x:x + w_]); f = (f if f is not None else a[:8, :8])[:, :W]; g[:f.shape[0], :f.shape[1]] = f   # clip to the canvas width
        elif kind == 2:
            for _ in range(int(r.integers(1, 5))): cv2.circle(g, (int(r.integers(4, W - 4)), int(r.integers(4, 60))), int(r.integers(1, 4)), 255, -1)
        return s.finish(prep_line(g)), [], []

def collate(b):
    imgs, tg, sp = zip(*[(z[0], z[1], z[2]) for z in b])
    x = torch.zeros(len(b), 1, 64, max(i.shape[1] for i in imgs))
    for k, i in enumerate(imgs): x[k, 0, :, :i.shape[1]] = torch.from_numpy(i) / 255.
    spans = torch.tensor([(k, *q) for k, rs in enumerate(sp) for q in rs], dtype=torch.long).view(-1, 4)
    return x, torch.tensor([t for ts in tg for t in ts], dtype=torch.long), torch.tensor([len(t) for t in tg]), torch.tensor([i.shape[1] // 4 for i in imgs]), spans

class Stream(torch.utils.data.IterableDataset):
    """Infinite sample stream. Samples are grouped by width, so a batch carries almost no padding: the shipped model reads unpadded images."""
    def __init__(s, ids): s.syn = Synth(ids)
    def __iter__(s):
        s.syn.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32); bs = CFG['BS']
        while True:
            pool = sorted((s.syn.sample() for _ in range(bs * 8)), key=lambda z: z[0].shape[1])
            for j in s.syn.rng.permutation(8): yield from pool[j * bs:(j + 1) * bs]

class SeqData:
    """Curriculum by epoch: numbers + clean writers + simple words -> vowel signs -> overlaps, conjuncts, all writers."""
    def __init__(s, ids): s.ds, s.level = Stream(ids), None
    def __call__(s, ep, it):
        lv = 0 if ep < .125 * CFG['EP'] else 1 if ep < .375 * CFG['EP'] else 2
        if lv != s.level:
            s.level = s.ds.syn.level = lv
            s.it = iter(torch.utils.data.DataLoader(s.ds, CFG['BS'], num_workers=CFG['WORKERS'], collate_fn=collate, pin_memory=DEV == 'cuda'))
        return next(s.it)

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

def _batches(imgs, bs):
    order = np.argsort([im.shape[1] for im in imgs])                          # sorted by width: minimal padding
    for j in range(0, len(imgs), bs):
        b = order[j:j + bs]; x = torch.zeros(len(b), 1, 64, max(imgs[k].shape[1] for k in b))
        for r, k in enumerate(b): x[r, 0, :, :imgs[k].shape[1]] = torch.from_numpy(imgs[k]) / 255.
        yield b, x.to(DEV)

@torch.no_grad()
def run(model, imgs, bs=128):
    """Raw CTC logits (T, V) per image."""
    model.eval(); res = [None] * len(imgs)
    for b, x in _batches(imgs, bs):
        with torch.autocast(**AMP): lg = model(x).float().cpu().numpy()
        for r, k in enumerate(b): res[k] = lg[r, :imgs[k].shape[1] // 4]
    return res

@torch.no_grad()
def run_aux(model, imgs, bs=128):
    """Classifier-head logits (84,) per image: the head averages all frames of the image (a single glyph)."""
    model.eval(); res = [None] * len(imgs)
    for b, x in _batches(imgs, bs):
        with torch.autocast(**AMP): h = model.feats(x).float()
        n = torch.tensor([imgs[k].shape[1] // 4 for k in b], device=DEV); m = (torch.arange(h.shape[1], device=DEV)[None] < n[:, None]).float()[..., None]
        lg = model.aux((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
        for r, k in enumerate(b): res[k] = lg[r]
    return res

def macro_f1(pred, y, n=85):
    cm = np.bincount(y * n + pred, minlength=n * n).reshape(n, n).astype(float); has = cm.sum(1) > 0
    return float((2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1))[has].mean())


def fit(name, model, epochs, steps, batch_fn, loss_fn, score_fn, lr, on_epoch=None):
    """AdamW + one-cycle + AMP + EMA. Resumes from the last finished epoch; returns the best EMA weights."""
    model.to(DEV); best_p, ck_p = OUT / f'{name}.pt', OUT / 'ckpt' / f'{name}.ckpt'
    if best_p.exists(): model.load_state_dict(torch.load(best_p, map_location=DEV)); print(f'{name}: found {best_p}, training skipped (use --retrain to train again)'); return model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps, pct_start=.05)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(.999), use_buffers=True)
    scaler, ep0, best = torch.amp.GradScaler(enabled=AMP['enabled']), 0, -1.
    if ck_p.exists():
        ck = torch.load(ck_p, map_location=DEV, weights_only=False)
        for k, o in dict(model=model, opt=opt, sch=sch, ema=ema, scaler=scaler).items(): o.load_state_dict(ck[k])
        ep0, best = ck['ep'], ck['best']; print(f'{name}: resuming after epoch {ep0}')
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
        if on_epoch: on_epoch(ep + 1, float(tot) / steps, s, sch.get_last_lr()[0], time.time() - t0)
    model.load_state_dict(torch.load(OUT / 'ckpt' / f'{name}_best.pt', map_location=DEV)); torch.save(model.state_dict(), best_p)
    return model.eval()


def threshold(conf, ok, target=.99):
    """Lowest confidence threshold whose accepted answers reach the target precision (and the share of inputs accepted)."""
    conf, ok = np.asarray(conf), np.asarray(ok)
    for t in np.arange(.2, 1, .005):
        a = conf >= t
        if a.sum() >= 10 and ok[a].mean() >= target: return float(t), float(a.mean())
    return 1., 0.

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

def ece_(conf, ok, n=15):
    b = (conf * n).astype(int).clip(max=n - 1)
    return float(sum(abs(conf[b == i].mean() - ok[b == i].mean()) * (b == i).mean() for i in range(n) if (b == i).any()))


# ------------------------------------------------------------------------------------------------ the three training flows
def load_weights(name, model):
    p = OUT / f'{name}.pt'
    model.load_state_dict(torch.load(p, map_location=DEV)); return model.to(DEV).eval()


def train_char_model(key, args, on_epoch):
    """Models 1 and 2. Returns (model, parameters to log)."""
    ep = args.epochs_for['resnet' if key == 'model1' else 'mobilenet']
    XT, ST, YT = build_char_training()
    k = torch.randperm(len(YT), device=DEV)[:16]                                # picture of the engineered data, logged as an artifact
    show = augment(XT[k, None].float() / 255, ST[k, None].float()).cpu().numpy()[:, 0]
    fig = plt.figure(figsize=(12, 2)); plt.imshow(np.vstack([np.hstack(XT[k].cpu().numpy() / 255), np.hstack(show)]), cmap='gray'); plt.axis('off')
    plt.title('top: engineered glyphs (V1-V5, V7) · bottom: after augmentation (V3, V4, noise)'); fig.savefig(adir(key) / 'samples.png', bbox_inches='tight', dpi=110); plt.close(fig)
    teacher = None
    if key == 'model2':
        if not (OUT / 'model1.pt').exists(): sys.exit('Model 2 is distilled from Model 1: train resnet first (or --reuse a folder with model1.pt).')
        teacher = load_weights('model1', make_resnet())
    model = make_resnet() if key == 'model1' else make_mobilenet()
    data = CharData(XT, ST, YT, ep, hard_from=int(ep * 2 / 3) if key == 'model1' else None)
    f1 = lambda m: macro_f1(logits(m, X[va]).argmax(1).numpy(), Y[va])
    model = fit(key, model, ep, data.steps, data, char_loss(data, teacher), f1, CFG['LR'], on_epoch)
    return model, dict(epochs=ep, steps_per_epoch=data.steps, batch_size=CFG['BS_CHAR'], train_samples=len(YT), teacher='model1' if teacher else 'none',
                       augmentation='V3 canvas redraw, V4 writer style, rotation/scale/shift, elastic, erasing, noise, CutMix' + (', hard-example mining' if key == 'model1' else ''))


def prepare_crnn_val():
    """Validation sequences and real validation glyphs for Model 3 (checkpoint score and calibration)."""
    global VAL, vg, VG
    VAL = [Synth(va, seed=1).sample() for _ in range(200 if CFG['FAST'] else 1000)]
    vg = np.random.default_rng(0).choice(va, min(len(va), 3000), replace=False); VG = [prep_line(X[i]) for i in vg]


def train_crnn(key, args, on_epoch):
    """Model 3. Returns (model, parameters to log)."""
    ep = args.epochs_for['crnn']; CFG['EP'] = ep
    prepare_crnn_val()
    def score(m):
        ex = np.mean([greedy(softmax(l)) == tuple(v[1]) for l, v in zip(run(m, [v[0] for v in VAL]), VAL)])
        gl = np.mean([char_probs(softmax(l))[0].argmax() == Y[i] for l, i in zip(run(m, VG), vg)]); return .5 * ex + .5 * gl
    demo = Synth(TRAIN_IDS, seed=3); fig = plt.figure(figsize=(12, 5))                          # picture of the synthetic words, with glyph spans
    for k, txt in enumerate(['জল', 'মানুষ', 'বিড়াল', '১২৩৪৫', 'কবিতা', 'নৌকা']):
        img, ids, sp = demo.render(txt); plt.subplot(3, 2, k + 1); plt.imshow(img, cmap='gray'); plt.axis('off'); plt.title(f'sample {k + 1}', fontsize=8)
        for c, a, b in sp: plt.axvspan(a * 4, (b + 1) * 4, ymin=0, ymax=.08, color='C1')
    plt.tight_layout(); fig.savefig(adir(key) / 'samples.png', bbox_inches='tight', dpi=110); plt.close(fig)
    model = fit(key, CRNN(), ep, CFG['STEPS'], SeqData(TRAIN_IDS), seq_loss, score, CFG['LR'], on_epoch)
    return model, dict(epochs=ep, steps_per_epoch=CFG['STEPS'], batch_size=CFG['BS'], train_samples=ep * CFG['STEPS'] * CFG['BS'], teacher='none (per-glyph classifier head on exact spans)',
                       augmentation='V3 canvas redraw, V4 writer style, V6 vowel signs, V7 reject, V8 same-writer sequences, elastic/pen/blur/noise',
                       curriculum='numbers + simple words -> vowel signs -> overlaps, conjuncts, random syllables', lexicon_train=len(LEX_TRAIN))


# ------------------------------------------------------------------------------------------------ calibration
def calibrate_classifier(model):
    """Temperature (NLL on validation glyphs) and the confidence needed for >=99% precision on accepted glyphs."""
    L, yv = logits(model, X[va]), Y[va]
    nll = lambda T: float(F.cross_entropy(L / T, torch.as_tensor(yv)))
    Ts = np.round(np.arange(.4, 2.51, .1), 2); T = float(min(Ts, key=nll))
    P = (L / T).softmax(1).numpy(); tau, cov = threshold(P.max(1), P.argmax(1) == yv)
    print(f'calibration: T={T}  tau_char={tau:.3f}  (accepted {cov:.1%} of validation glyphs at >=99% precision)')
    return dict(T=T, tau=dict(char=tau, low=.3), accepted_val=cov)


def calibrate_crnn(model):
    """Temperatures (sequence, single glyph, classifier head), blend weight, lexicon bonus and thresholds, all on VALIDATION writers."""
    def seqs(engine, kind, n):
        out = [engine.sample(kind) for _ in range(n)]
        return [o[0] for o in out], [tuple(o[1]) for o in out]
    nv = 150 if CFG['FAST'] else 600
    (v_num, t_num), (v_lex, t_lex), (v_ps, t_ps) = seqs(Synth(va, seed=11), 'num', nv), seqs(Synth(va, seed=12, pool=LEX_VAL), 'lex', nv), seqs(Synth(va, seed=13), 'pseudo', nv)
    LG = {k: run(model, v) for k, v in dict(num=v_num, lex=v_lex, ps=v_ps).items()}; truth = dict(num=t_num, lex=t_lex, ps=t_ps)
    nll = lambda T: np.mean([-np.log(ctc_prob(softmax(l, T), t) + 1e-12) / max(1, len(t)) for k in LG for l, t in zip(LG[k][:150], truth[k][:150])])
    Ts = np.round(np.arange(.4, 2.51, .1), 2); T = float(min(Ts, key=nll))
    LGg, AXg = run(model, VG), run_aux(model, VG); yv_ = Y[vg]
    p_ctc = lambda t: np.stack([char_probs(softmax(l, t))[1] for l in LGg]); p_aux = lambda t: softmax(np.stack(AXg), t)
    nll_ = lambda p: float(-np.log(p[np.arange(len(p)), yv_] + 1e-12).mean())
    Tc = float(min(Ts, key=lambda t: nll_(p_ctc(t)))); Ta = float(min(Ts, key=lambda t: nll_(p_aux(t))))
    wa = float(min(np.linspace(0, 1, 5), key=lambda w: nll_((1 - w) * p_ctc(Tc) + w * p_aux(Ta)))); pg = (1 - wa) * p_ctc(Tc) + wa * p_aux(Ta)
    C = {k: [candidates(softmax(l, T), 'word', LEXS) for l in LG[k]] for k in ('lex', 'ps')}
    acc = lambda k, b: np.mean([(rank(c, LEXS, b, 1) or [((), 0)])[0][0] == t for c, t in zip(C[k], truth[k])])
    bonus = float(max([0., 1., 2., 3., 4., 6., 8., 10.], key=lambda b: acc('lex', b) + acc('ps', b)))
    rn = [rank(candidates(softmax(l, T), 'number'), None, 0, 1) or [((), 0.)] for l in LG['num']]
    tau_num, cov_num = threshold([r[0][1] for r in rn], [r[0][0] == t for r, t in zip(rn, truth['num'])])
    rw = [rank(c, LEXS, bonus, 1) or [((), 0.)] for c in C['lex'] + C['ps']]
    tau_word, cov_word = threshold([r[0][1] for r in rw], [r[0][0] == t for r, t in zip(rw, truth['lex'] + truth['ps'])])
    tau_char, cov_char = threshold(pg.max(1), pg.argmax(1) == yv_)
    print(f'calibration: T={T} Tc={Tc} Ta={Ta} wa={wa} bonus={bonus} | accepted at >=99% precision: characters {cov_char:.1%} numbers {cov_num:.1%} words {cov_word:.1%}')
    return dict(T=T, Tc=Tc, Ta=Ta, wa=wa, bonus=bonus, tau=dict(ink=.3, char=tau_char, number=tau_num, word=tau_word),
                accepted_val=dict(char=cov_char, number=cov_num, word=cov_word))


# ------------------------------------------------------------------------------------------------ ONNX export, config
def export_onnx(key, model):
    """Write models/<file>.onnx (batch 1 for the CRNN, dynamic width) and check it against PyTorch. Returns (path, parity)."""
    path = bc.MODELS_DIR / ONNX_FILES[key]; m = model.float().cpu().eval()
    if key == 'model3':
        class Exp(nn.Module):
            """ONNX graph: CTC logits 'y' and classifier-head logits 'a' (mean over all frames) from one backbone pass."""
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x): h = s.m.feats(x); return s.m.fc(h), s.m.aux(h.mean(1))
        torch.onnx.export(Exp(m).eval(), torch.zeros(1, 1, 64, 256), path, input_names=['x'], output_names=['y', 'a'], dynamic_axes={'x': {3: 'w'}, 'y': {1: 't'}}, opset_version=17, **KW)
    else:
        torch.onnx.export(m, torch.zeros(1, 1, 64, 64), path, input_names=['x'], output_names=['y'], dynamic_axes={'x': {0: 'n'}, 'y': {0: 'n'}}, opset_version=17, **KW)
    s = bc._session(path); agree = []
    if key == 'model3':
        ex = [Synth(te, seed=31).sample('lex') for _ in range(30)]
        for im, ids, *_ in ex:
            with torch.no_grad(): ref = m(torch.from_numpy(im)[None, None].float() / 255).numpy()[0]
            agree.append(greedy(softmax(ref)) == greedy(softmax(s.run(None, {'x': im[None, None].astype(np.float32) / 255})[0][0])))
    else:
        A = X[te[:200]]
        with torch.no_grad(): ref = m(torch.from_numpy(A)[:, None].float() / 255).argmax(1).numpy()
        agree = list(ref == s.run(None, {'x': A[:, None].astype(np.float32) / 255})[0].argmax(1))
    parity = float(np.mean(agree)); print(f'{path.name}: {path.stat().st_size / 1e6:.1f} MB, ONNX vs PyTorch agreement {parity:.3f}')
    return path, parity


def update_config(key, entry):
    p = bc.MODELS_DIR / 'config.json'; cfg = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'models': {}}
    cfg['models'][key] = entry; p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding='utf-8')


def write_labels():
    lab = dict(classes=CLASSES, class_folders={i + 1: c for i, c in enumerate(CLASSES)}, reject_index_models_1_2=REJ, vocab=VOCAB, vowel_signs=KARS)
    for p in (bc.MODELS_DIR / 'labels.json', bc.BASE_DIR / 'labels.json'): p.write_text(json.dumps(lab, indent=1, ensure_ascii=False), encoding='utf-8')
    (bc.MODELS_DIR / 'lexicon.txt').write_text('\n'.join(LEX), encoding='utf-8')


def export_samples():
    """Synthetic sample drawings for the app (clearly synthetic: real glyphs of unseen writers composed into words and numbers)."""
    d = bc.MODELS_DIR / 'samples'; d.mkdir(exist_ok=True); eng = Synth(te, seed=21); meta = {}
    wanted = [w for w in ('জল', 'মানুষ', 'কলম', 'নদী', 'ফুল', 'বাড়ি') if to_visual(w)]
    for n, txt in enumerate(wanted): meta[f'word_{n + 1:02d}.png'] = dict(text=txt, kind='word'); cv2.imwrite(str(d / f'word_{n + 1:02d}.png'), eng.render(txt)[0])
    for n, txt in enumerate(['০৮৪৫৬২', '১২৩৪', '৯০৭']): meta[f'number_{n + 1:02d}.png'] = dict(text=txt, kind='number'); cv2.imwrite(str(d / f'number_{n + 1:02d}.png'), eng.render(txt)[0])
    for n, i in enumerate(np.random.default_rng(3).choice(te, 4, replace=False)):
        meta[f'char_{n + 1:02d}.png'] = dict(text=CLASSES[Y[i]], kind='char'); cv2.imwrite(str(d / f'char_{n + 1:02d}.png'), X[i])
    (d / 'samples.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding='utf-8')


# ------------------------------------------------------------------------------------------------ evaluation of the EXPORTED models
def to_ids(text):
    return tuple(TID[t] for t in (to_visual(text) or []))


def evaluate_models(keys, n_glyph, n_seq):
    """Test the saved ONNX models exactly as the app runs them, on writers never seen in training.
    Returns {key: flat metrics}, {key: nested report}, {key: (confusion matrix, top confusions)}."""
    models = bc.load_models(keys)
    gi = np.random.default_rng(1).choice(te, min(len(te), n_glyph), replace=False); yt = Y[gi]
    E = Synth(te, seed=5); nums = [E.sample('num') for _ in range(n_seq)]; pws = [E.sample('pseudo') for _ in range(n_seq)]
    lexw = [Synth(te, seed=6, pool=LEX_TEST).sample('lex') for _ in range(n_seq)]
    qd = df.quality.to_numpy()[gi]; flat, report, extras = {}, {}, {}
    for key, m in models.items():
        t0 = time.time(); P = np.stack([m.glyph_probs(X[i]) for i in gi]); ok = P.argmax(1) == yt
        g = dict(top1=float(ok.mean()), top3=float((np.argsort(-P, 1)[:, :3] == yt[:, None]).any(1).mean()), macro_f1=macro_f1(P.argmax(1), yt, NCLS), ece=ece_(P.max(1), ok), n=len(yt))
        grp = lambda a, b: float(ok[(yt >= a) & (yt < b)].mean())
        by = dict(basic=grp(0, 50), numerals=grp(50, 60), compound=grp(60, 84), q_high=float(ok[qd >= 3].mean()) if (qd >= 3).any() else None,
                  q_low=float(ok[qd <= 1.5].mean()) if (qd <= 1.5).any() else None)
        cm = np.bincount(yt * NCLS + P.argmax(1), minlength=NCLS * NCLS).reshape(NCLS, NCLS); off = cm.copy(); np.fill_diagonal(off, 0)
        top = [(CLASSES[i // NCLS] if i // NCLS < 84 else '-', CLASSES[i % NCLS] if i % NCLS < 84 else 'reject', int(off.flat[i])) for i in np.argsort(-off.flatten())[:10]]
        def seqeval(samples, mode):
            pred = [to_ids(m.predict(s[0], mode)['text']) for s in samples]; return seq_scores(pred, [tuple(s[1]) for s in samples])
        num, lex, ps = seqeval(nums, 'number'), seqeval(lexw, 'word'), seqeval(pws, 'word')
        lat = {k: float(np.median([timed(m, s[0], mode, 1)[1] for s in smp[:25]])) for k, mode, smp in (('char', 'char', [(X[i],) for i in gi[:25]]), ('number', 'number', nums), ('word', 'word', lexw))}
        info = bc.load_config()['models'][key]
        valid = info.get('split_valid', True)                       # False: the model may have seen these writers, so the numbers are a sanity check, not a test
        pre = 'test_' if valid else 'sanity_'
        report[key] = dict(glyph=g, glyph_by_group=by, numbers=num, heldout_words=lex, pseudo_words=ps, latency_ms=lat, top_confusions=top, params=info.get('params'), size_mb=info.get('size_mb'), split_valid=valid)
        flat[key] = {pre + 'char_top1': g['top1'], pre + 'char_top3': g['top3'], pre + 'char_macro_f1': g['macro_f1'], pre + 'char_ece': g['ece'],
                     pre + 'number_exact': num['exact'], pre + 'number_cer': num['cer'], pre + 'number_f1': num['f1'],
                     pre + 'word_exact': lex['exact'], pre + 'word_cer': lex['cer'], pre + 'pseudo_exact': ps['exact'], pre + 'pseudo_cer': ps['cer'],
                     'latency_ms_char': lat['char'], 'latency_ms_number': lat['number'], 'latency_ms_word': lat['word'], 'eval_seconds': time.time() - t0}
        extras[key] = (cm, top)
        print(f"{bc.MODEL_TITLES[key]}{'' if valid else ' [SANITY CHECK, not a test: these writers may have been in its training set]'}: glyph top-1 {g['top1']:.4f} top-3 {g['top3']:.4f} | numbers {num['exact']:.3f} | held-out words {lex['exact']:.3f} | pseudo-words {ps['exact']:.3f} | "
              f"{lat['char']:.1f}/{lat['number']:.1f}/{lat['word']:.1f} ms (char/number/word)")
    return flat, report, extras


# ------------------------------------------------------------------------------------------------ MLflow artifacts
def plot_curves(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2)); h = pd.DataFrame(hist)
    if len(h):
        ax[0].plot(h.epoch, h.train_loss); ax[0].set_title('training loss'); ax[1].plot(h.epoch, h.val_score); ax[1].set_title('validation score'); [a.set_xlabel('epoch') for a in ax]
    else:
        for a in ax: a.text(.5, .5, 'weights reused: no training in this run', ha='center'); a.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_confusion(cm, path, title):
    fig = plt.figure(figsize=(6.5, 6)); n = cm / np.maximum(cm.sum(1, keepdims=True), 1); plt.imshow(n, cmap='viridis', vmin=0, vmax=1)
    plt.xlabel('predicted class (index; 85 = reject)'); plt.ylabel('true class (index)'); plt.title(title, fontsize=9); plt.colorbar(fraction=.046)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def model_card(key, params, report, note=''):
    r = report; what = 'unseen writers' if r.get('split_valid', True) else "sanity check on this project's split, NOT a test: writers may have been in training"
    return (f"# {bc.MODEL_TITLES[key]}\n\n{note}{bc.MODEL_ROLES[key]}\n\n- parameters: {r['params']:,}\n- ONNX size: {r['size_mb']:.1f} MB\n"
            f"- single glyph ({what}): top-1 {r['glyph']['top1']:.2%}, top-3 {r['glyph']['top3']:.2%}, macro-F1 {r['glyph']['macro_f1']:.3f}\n"
            f"- numbers (1-8 digits): exact {r['numbers']['exact']:.2%}\n- {'held-out lexicon words' if r.get('heldout_words_valid', True) else 'lexicon words (the model SAW these in training: not held out)'}: exact {r['heldout_words']['exact']:.2%}\n"
            f"- random syllable words: exact {r['pseudo_words']['exact']:.2%}\n- latency on one CPU thread: {r['latency_ms']['char']:.1f} ms (char), {r['latency_ms']['word']:.1f} ms (word)\n\n"
            f"Training parameters: {json.dumps(params, ensure_ascii=False)}\n\nLimits: the dataset has no vowel signs and no word images; words are built from real glyphs "
            f"(vowel signs are drawn procedurally). Real handwriting of whole words has not been measured here.\n")


def run_model(key, args):
    """One MLflow training run for one model: train, calibrate, export, test the exported file, log everything."""
    mlflow = bc.mlflow_setup(); mlflow.set_experiment(experiment_id=bc.get_experiment_id(mlflow, bc.TRAIN_EXPERIMENT))
    hist, t0 = [], time.time()
    def on_epoch(ep, loss, s, lr, secs):
        hist.append(dict(epoch=ep, train_loss=loss, val_score=s, lr=lr)); mlflow.log_metrics(dict(train_loss=loss, val_score=s, lr=lr, epoch_seconds=secs), step=ep)
    with mlflow.start_run(run_name=RUN_NAMES[key]) as run:
        mlflow.set_tags(dict(model_key=key, role=bc.MODEL_ROLES[key], profile=args.profile, framework='pytorch -> onnx'))
        mlflow.log_params(dict(model=bc.MODEL_TITLES[key], device=DEV, seed=CFG['SEED'], lr=CFG['LR'], optimizer='AdamW (wd 0.01)', schedule='one-cycle (5% warm-up)',
                               ema=0.999, label_smoothing=0.1, mixed_precision=AMP['enabled'], profile=args.profile, dataset='BanglaLekha-Isolated',
                               split='by writer 80/10/10', n_train_glyphs=len(TRAIN_IDS), n_val_glyphs=len(va), n_test_glyphs=len(te),
                               writers_train=int(df[df.split == 'train'].writer.nunique()), mislabels_removed=int(len(tr) - len(trc))))
        reused = (OUT / f'{key}.pt').exists() and not args.retrain
        mlflow.set_tag('weights_source', 'reused (found in work/ or via --reuse)' if reused else 'trained in this run')
        if args.retrain:
            for p in [OUT / f'{key}.pt', OUT / 'ckpt' / f'{key}.ckpt', OUT / 'ckpt' / f'{key}_best.pt']: p.unlink(missing_ok=True)
        model, params = (train_crnn if key == 'model3' else train_char_model)(key, args, on_epoch)
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        n_params = sum(p.numel() for p in model.parameters()); mlflow.log_metric('n_params', n_params)
        mlflow.log_metric('train_seconds', time.time() - t0)
        entry = (calibrate_crnn if key == 'model3' else calibrate_classifier)(model)
        path, parity = export_onnx(key, model)
        entry.update(file=path.name, params=int(n_params), size_mb=path.stat().st_size / 1e6, onnx_parity=parity); update_config(key, entry)
        write_labels(); export_samples()
        flat, report, extras = evaluate_models([key], args.eval_glyphs, args.eval_seqs)
        mlflow.log_metrics(flat[key]); mlflow.log_metrics(dict(model_size_mb=entry['size_mb'], onnx_parity=parity))
        mlflow.log_dict(entry, 'calibration.json'); mlflow.log_dict(report[key], 'report.json')
        mlflow.log_artifact(str(path)); mlflow.log_artifact(str(bc.MODELS_DIR / 'config.json')); mlflow.log_artifact(str(bc.MODELS_DIR / 'labels.json'))
        plot_curves(hist, adir(key) / 'curves.png'); plot_confusion(extras[key][0], adir(key) / 'confusion_matrix.png', f'{bc.MODEL_TITLES[key]}: confusion (test glyphs)')
        for name in ('curves.png', 'confusion_matrix.png', 'samples.png'): mlflow.log_artifact(str(adir(key) / name))
        mlflow.log_text(model_card(key, params, report[key]), 'model_card.md'); mlflow.log_dict(dict(top_confusions=extras[key][1]), 'top_confusions.json')
        print(f'MLflow run {run.info.run_id} ({RUN_NAMES[key]}) logged.')
    return flat[key], report[key]


def log_comparison(flat, report):
    """One 'compare-all' run in the evaluation experiment: a table and a chart of the three models side by side."""
    mlflow = bc.mlflow_setup(); mlflow.set_experiment(experiment_id=bc.get_experiment_id(mlflow, bc.EVAL_EXPERIMENT))
    rows = [dict(model=bc.MODEL_TITLES[k], params=report[k]['params'], size_mb=report[k]['size_mb'], **{re.sub(r'^(test|sanity)_', '', m): v for m, v in flat[k].items() if m != 'eval_seconds'}) for k in flat]
    df_ = pd.DataFrame(rows)
    with mlflow.start_run(run_name='compare-all'):
        mlflow.set_tags(dict(kind='comparison', models=','.join(flat)))
        for k in flat: mlflow.log_metrics({f'{k}_{m}': v for m, v in flat[k].items()})
        mlflow.log_text(df_.to_csv(index=False), 'comparison.csv'); mlflow.log_dict(report, 'comparison.json')
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4)); lab = [r['model'].split(' · ')[0] for r in rows]
        ax[0].bar(lab, df_['char_top1']); ax[0].set_title('single glyph top-1'); ax[0].set_ylim(0, 1)
        ax[1].bar(lab, df_['number_exact']); ax[1].set_title('numbers exact match'); ax[1].set_ylim(0, 1)
        ax[2].bar(lab, df_['latency_ms_word']); ax[2].set_title('time per word image (ms, 1 CPU thread)')
        fig.tight_layout(); fig.savefig(adir('comparison') / 'comparison.png', dpi=110); plt.close(fig); mlflow.log_artifact(str(adir('comparison') / 'comparison.png'))
    print(df_.round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------------ models trained in the Kaggle notebooks
# The three models were trained in notebooks (bangla-hwr-v5.ipynb for Models 1 and 2, bangla-hwr-model3-f.ipynb for Model 3). train.py
# does not pretend otherwise: `import-notebook` reads what the executed notebook REALLY printed (epoch lines, calibration, reported test
# results), takes the model file the notebook produced, re-tests the exported file here, and records one MLflow run that is tagged
# source=notebook with the notebook's file name.
RX_V5 = re.compile(r'\[(?P<m>model1|model2)\] Ep (?P<ep>\d+)/(?P<tot>\d+) \| LR=(?P<lr>[0-9.eE+-]+) \| Loss=(?P<loss>[0-9.eE+-]+) \| Val Acc=(?P<acc>[0-9.]+)% \| Val F1=(?P<f1>[0-9.]+)%')
RX_F = re.compile(r'model3 (?P<ep>\d+)/(?P<tot>\d+) loss (?P<loss>[0-9.]+) val-score (?P<s>[0-9.]+) best (?P<best>[0-9.]+) (?P<sec>\d+)s')
RX_CAL = re.compile(r'Calibration Complete: T1=(?P<t1>[0-9.]+), T2=(?P<t2>[0-9.]+) \| Weights=(?P<w1>[0-9.]+)/(?P<w2>[0-9.]+) \| Accept>=(?P<acc>[0-9.]+)')


def notebook_text(path):
    """The printed (stream) output of every code cell of an executed notebook, one string per cell."""
    nb = json.loads(Path(path).read_text(encoding='utf-8'))
    return [''.join(''.join(o.get('text', '')) for o in c.get('outputs', []) if o.get('output_type') == 'stream') for c in nb['cells'] if c['cell_type'] == 'code']


def parse_training_log(key, text):
    """The epoch lines a notebook printed -> ([(epoch, metrics)], total epochs, the lines themselves)."""
    hist, lines, total = [], [], 0
    if key == 'model3':
        for m in RX_F.finditer(text):
            hist.append((int(m['ep']), dict(train_loss=float(m['loss']), val_score=float(m['s']), epoch_seconds=float(m['sec'])))); lines.append(m.group(0)); total = int(m['tot'])
    else:
        for m in RX_V5.finditer(text):
            if m['m'] != key: continue
            hist.append((int(m['ep']), dict(train_loss=float(m['loss']), val_score=float(m['f1']) / 100, val_accuracy=float(m['acc']) / 100, lr=float(m['lr']))))
            lines.append(m.group(0)); total = int(m['tot'])
    return hist, total, lines


def parse_table(text):
    """Metrics from the results table a notebook printed (Model 3): {name: value}."""
    out, num = {}, r'([0-9.]+)'
    m = re.search(r'^Real glyphs \(84 classes\)\s+' + r'\s+'.join([num] * 4), text, re.M)
    if m: out.update(zip(('char_top1', 'char_top3', 'char_macro_f1', 'char_ece'), map(float, m.groups())))
    for label, name in ((r'Numbers \(1.8 digits\)', 'number'), (r'Pseudo-words, free reading', 'pseudo'), (r'(?:Held-out|Real) words, lexicon decoding', 'lexicon_words')):
        m = re.search(r'^' + label + r'\s+' + r'\s+'.join([num] * 4), text, re.M)
        if m: out.update({f'{name}_exact': float(m.group(1)), f'{name}_cer': float(m.group(2)), f'{name}_f1': float(m.group(3))})
    return out


def parse_reported(key, texts, eval_texts):
    """What the notebook itself reported as test results: ({metric: value} named notebook_*, the verbatim text for an artifact)."""
    rep, shown = {}, []
    if key in ('model1', 'model2'):
        allt = '\n'.join(texts); i = allt.find('Evaluation Complete:')
        try:
            blob, _ = json.JSONDecoder().raw_decode(allt[allt.index('{', i):])
            blk = blob.get({'model1': 'model1_resnet', 'model2': 'model2_mobile'}[key], {})
            for k, v in blk.items(): rep['notebook_cpu_ms' if k == 'cpu_ms' else f'notebook_test_{k}'] = v / 100 if k in ('char_top1', 'char_top3', 'ece') else v
            shown.append('Evaluation Complete: ' + json.dumps(blob, indent=1))
        except Exception as e:
            shown.append(f'(could not read the evaluation block: {e})')
    else:
        for name, group in (('training notebook', texts), ('evaluation notebook', eval_texts)):
            tab = next((t for t in group if 'macro_f1' in t), '')
            if not tab: continue
            shown.append(f'--- printed by the {name} ---\n{tab}')
            for k, v in parse_table(tab).items():                       # the evaluation notebook, read last, wins
                if k.startswith('lexicon_words'): rep['notebook_' + k.replace('lexicon_words', 'words') + '_SEEN_in_training'] = v
                else: rep[f'notebook_test_{k}'] = v
    return rep, '\n\n'.join(shown)


def import_notebook(args):
    """One MLflow run for a model that a notebook trained: the notebook's real log + a fresh test of the exported file."""
    key, nb = KEYS[args.model], Path(args.notebook)
    texts = notebook_text(nb); ev_texts = notebook_text(args.eval_notebook) if args.eval_notebook else []
    hist, total, lines = parse_training_log(key, '\n'.join(texts))
    if not hist: sys.exit(f'No epoch lines for {key} found in {nb.name}. Is it the executed notebook that trained this model?')
    reported, shown = parse_reported(key, texts, ev_texts)
    split_valid = False       # Models 1 and 2 (v5) used their own writer split, which this project's cache does not reproduce
    if key == 'model3':
        src = Path(args.weights) if args.weights else OUT / 'model3.pt'
        if not src.exists(): sys.exit('Model 3 weights not found: pass --weights /path/to/model3.pt (or --reuse the notebook output folder).')
        # Model 3 was trained on the split in the meta.csv next to its weights. If that file is byte-identical to the cache in use,
        # the test writers here are unseen by the model; --same-split asserts it when the weights live in another folder.
        sibling = src.parent / 'meta.csv'
        split_valid = bool(args.same_split or (sibling.exists() and (OUT / 'meta.csv').exists() and sibling.read_bytes() == (OUT / 'meta.csv').read_bytes()))
        if src.resolve() != (OUT / 'model3.pt').resolve(): shutil.copy(src, OUT / 'model3.pt')
        prepare_crnn_val(); model = load_weights('model3', CRNN()); n_params = sum(p.numel() for p in model.parameters())
        entry = calibrate_crnn(model); path, parity = export_onnx(key, model)
        entry['calibration_source'] = 'recomputed by train.py on validation writers'
    else:
        if not args.onnx or not Path(args.onnx).exists(): sys.exit('--onnx /path/to/the_onnx_file_the_notebook_exported is required for resnet / mobilenet.')
        import onnx
        n_params = int(sum(int(np.prod(t.dims)) for t in onnx.load(args.onnx).graph.initializer))
        path = bc.MODELS_DIR / ONNX_FILES[key]; shutil.copy(args.onnx, path); parity = None
        cal = RX_CAL.search('\n'.join(texts))
        entry = dict(T=float(cal['t1' if key == 'model1' else 't2']) if cal else 1.0, tau=dict(char=float(cal['acc']) if cal else .9, low=.3), preprocess='v5',
                     calibration_source='read from the notebook log' if cal else 'default: no calibration line in the notebook log')
    entry.update(file=path.name, params=int(n_params), size_mb=path.stat().st_size / 1e6, onnx_parity=parity, source=f'notebook:{nb.name}', split_valid=split_valid)
    update_config(key, entry); write_labels(); export_samples()
    flat, report, extras = evaluate_models([key], args.eval_glyphs, args.eval_seqs)
    if key == 'model3':       # that notebook trained on EVERY lexicon word, so the 'held-out' words here were seen in training: say so in the metric names
        pre = 'test_' if split_valid else 'sanity_'
        for m in ('word_exact', 'word_cer'): flat[key][f'seen_in_training_{m}'] = flat[key].pop(pre + m)
        report[key]['heldout_words_valid'] = False
    mlflow = bc.mlflow_setup(); mlflow.set_experiment(experiment_id=bc.get_experiment_id(mlflow, bc.TRAIN_EXPERIMENT))
    a = adir(key)
    note = (f"**Trained in the Kaggle notebook `{nb.name}`, not with train.py.** The epoch curve below is what that notebook printed; "
            f"the test numbers are re-measured here on the exported file ({'unseen writers' if split_valid else 'a sanity check: this model used a different writer split'}).\n\n")
    with mlflow.start_run(run_name=RUN_NAMES[key]) as run:
        mlflow.set_tags(dict(model_key=key, role=bc.MODEL_ROLES[key], profile=args.profile, framework='pytorch -> onnx', source='notebook', notebook=nb.name,
                             trained_with='Kaggle notebook, NOT train.py', weights_source='imported from the notebook output', split_valid=str(split_valid)))
        mlflow.log_params(dict(model=bc.MODEL_TITLES[key], dataset='BanglaLekha-Isolated', source_notebook=nb.name, evaluation_notebook=Path(args.eval_notebook).name if args.eval_notebook else 'none',
                               epochs=total, epochs_printed_by_notebook=len(hist), calibration_source=entry['calibration_source'], preprocessing=entry.get('preprocess', 'default'),
                               n_glyphs_measured_here=report[key]['glyph']['n']))
        for ep, m in hist: mlflow.log_metrics(m, step=ep)
        mlflow.log_metrics(dict(n_params=n_params, model_size_mb=entry['size_mb'])); mlflow.log_metrics(reported); mlflow.log_metrics(flat[key])
        mlflow.log_dict(entry, 'calibration.json'); mlflow.log_dict(report[key], 'report.json')
        mlflow.log_artifact(str(path)); mlflow.log_artifact(str(bc.MODELS_DIR / 'config.json')); mlflow.log_artifact(str(bc.MODELS_DIR / 'labels.json'))
        plot_curves([dict(epoch=ep, **m) for ep, m in hist], a / 'curves.png')
        plot_confusion(extras[key][0], a / 'confusion_matrix.png', f"{bc.MODEL_TITLES[key]}: confusion ({'test glyphs' if split_valid else 'sanity check, not a test'})")
        for name in ('curves.png', 'confusion_matrix.png'): mlflow.log_artifact(str(a / name))
        mlflow.log_text('\n'.join(lines) + '\n', 'notebook_training_log.txt'); mlflow.log_text(shown or '(the notebook printed no results table)', 'notebook_reported_results.txt')
        mlflow.log_text(model_card(key, dict(epochs=total, notebook=nb.name), report[key], note), 'model_card.md'); mlflow.log_dict(dict(top_confusions=extras[key][1]), 'top_confusions.json')
        print(f'MLflow run {run.info.run_id} ({RUN_NAMES[key]}, source=notebook:{nb.name}) logged: {len(hist)} epoch points, {len(reported)} notebook-reported metrics.')


# ------------------------------------------------------------------------------------------------ commands
def parse_models(text):
    ks = [k.strip() for k in text.split(',') if k.strip()]
    keys = list(KEYS) if 'all' in ks else ks
    bad = [k for k in keys if k not in KEYS]
    if bad: sys.exit(f'unknown model {bad}; use all, resnet, mobilenet, crnn')
    return [KEYS[k] for k in sorted(keys, key=list(KEYS).index)]        # resnet first: Model 2 needs Model 1


def main():
    ap = argparse.ArgumentParser(description='Train and test the three Bangla handwriting models (MLflow tracked).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('train', 'test'):
        p = sub.add_parser(name)
        p.add_argument('--data', help='BanglaLekha-Isolated folder (searched recursively for the class folders 1..84)')
        p.add_argument('--model', default='all', help='all | resnet | mobilenet | crnn (comma list allowed)')
        p.add_argument('--profile', default='full', choices=PROFILES)
        p.add_argument('--reuse', help='folder of an earlier run: reuse its data cache and model1/2/3.pt')
        p.add_argument('--workers', type=int, default=-1, help='data-generation workers for Model 3 (-1 = automatic: up to 4, none on Windows)')
        p.add_argument('--eval-glyphs', type=int, default=5000); p.add_argument('--eval-seqs', type=int, default=1000)
    p = sub.add_parser('import-notebook', help='record a model trained in a Kaggle notebook as an MLflow run (its real log, tagged source=notebook)')
    p.add_argument('--model', required=True, choices=list(KEYS)); p.add_argument('--notebook', required=True, help='the EXECUTED .ipynb that trained the model')
    p.add_argument('--onnx', help='resnet / mobilenet: the ONNX file the notebook exported'); p.add_argument('--weights', help='crnn: model3.pt from the notebook output')
    p.add_argument('--eval-notebook', help='executed .ipynb that tested the model (optional)')
    p.add_argument('--same-split', action='store_true', help='crnn: assert that the data cache (meta.csv) is the one the model was trained with (checked automatically when meta.csv sits next to --weights)')
    p.add_argument('--data'); p.add_argument('--reuse'); p.add_argument('--profile', default='full', choices=PROFILES); p.add_argument('--workers', type=int, default=-1)
    p.add_argument('--eval-glyphs', type=int, default=5000); p.add_argument('--eval-seqs', type=int, default=1000)
    t = sub.choices['train']
    t.add_argument('--epochs-resnet', type=int); t.add_argument('--epochs-mobilenet', type=int); t.add_argument('--epochs-crnn', type=int)
    t.add_argument('--retrain', action='store_true', help='ignore existing weights in work/ and train again')
    t.add_argument('--no-clean', action='store_true', help='skip the confident-learning label cleaning')
    a = ap.parse_args()
    a.retrain = getattr(a, 'retrain', False)
    if a.profile == 'smoke': a.eval_glyphs, a.eval_seqs = min(a.eval_glyphs, 200), min(a.eval_seqs, 60)
    setup(a); CFG['CLEAN'] = not getattr(a, 'no_clean', False)
    keys = parse_models(a.model)
    prof = PROFILES[a.profile]['epochs']
    a.epochs_for = {k: (getattr(a, f'epochs_{k}', None) or prof[k]) for k in prof}
    prepare_data(a); write_labels()
    if a.cmd == 'train':
        res = {k: run_model(k, a) for k in keys}
        if len(keys) > 1: log_comparison({k: r[0] for k, r in res.items()}, {k: r[1] for k, r in res.items()})
    elif a.cmd == 'import-notebook':
        import_notebook(a)
    else:
        missing = [k for k in keys if k not in bc.load_config()['models']]
        if missing: sys.exit(f'No trained model for {missing} in {bc.MODELS_DIR}. Run: python train.py train --data ... first.')
        flat, report, extras = evaluate_models(keys, a.eval_glyphs, a.eval_seqs)
        mlflow = bc.mlflow_setup(); mlflow.set_experiment(experiment_id=bc.get_experiment_id(mlflow, bc.EVAL_EXPERIMENT))
        for k in keys:
            with mlflow.start_run(run_name=f'test-{RUN_NAMES[k]}'):
                mlflow.set_tags(dict(model_key=k, kind='test')); mlflow.log_metrics(flat[k]); mlflow.log_dict(report[k], 'report.json')
                plot_confusion(extras[k][0], adir(k) / 'confusion_matrix_test.png', f'{bc.MODEL_TITLES[k]}: confusion (test glyphs)'); mlflow.log_artifact(str(adir(k) / 'confusion_matrix_test.png'))
        if len(keys) > 1: log_comparison(flat, report)
    print('done. Models:', bc.MODELS_DIR, '| MLflow:', bc.MLFLOW_DIR)


if __name__ == '__main__':
    main()
