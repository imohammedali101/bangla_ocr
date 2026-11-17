"""bangla_common.py - shared by train.py and app.py.

Labels, image preprocessing, CTC decoding, the three ONNX models behind one interface, glyph segmentation (for the
two character models), timing and the MLflow helpers.  Inference needs only numpy, OpenCV and ONNX Runtime.

Fixed folders (they never depend on where Python is started; BANGLA_HOME overrides the project folder):
    models/            the three ONNX models, config.json, lexicon.txt, labels.json, samples/
    artifacts/mlflow/  mlflow.db (runs, parameters, metrics) and store/ (artifact files)
"""
import json
import os
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

try:                                  # OpenCV is only needed to read images; notebook extraction and MLflow helpers work without it
    import cv2
except ImportError:
    cv2 = None
import numpy as np

# --------------------------------------------------------------------------------------------- fixed paths
BASE_DIR = MODELS_DIR = MLFLOW_DIR = MLFLOW_STORE = None
TRACKING_URI = ''
TRAIN_EXPERIMENT, EVAL_EXPERIMENT, SERVE_EXPERIMENT = 'bangla-training', 'bangla-evaluation', 'bangla-serving'


def set_home(path):
    """Point every fixed path at one project folder (train.py --profile smoke uses a scratch copy)."""
    global BASE_DIR, MODELS_DIR, MLFLOW_DIR, MLFLOW_STORE, TRACKING_URI
    BASE_DIR = Path(path).resolve()
    MODELS_DIR = BASE_DIR / 'models'
    MLFLOW_DIR = BASE_DIR / 'artifacts' / 'mlflow'
    MLFLOW_STORE = MLFLOW_DIR / 'store'
    TRACKING_URI = f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}"      # forward slashes also on Windows: sqlite:///C:/Users/...


set_home(os.environ.get('BANGLA_HOME', Path(__file__).resolve().parent))

# --------------------------------------------------------------------------------------------- the three models
MODEL_KEYS = ('model1', 'model2', 'model3')
MODEL_TITLES = {'model1': 'Model 1 · Lekha-ResNet', 'model2': 'Model 2 · Lekha-Mobile', 'model3': 'Model 3 · Lekha-CRNN'}
MODEL_ROLES = {
    'model1': 'Reads one glyph at a time (ResNet). Words and numbers are read by cutting the drawing into glyphs; it has no class for vowel signs.',
    'model2': 'Same job as Model 1 with a much smaller network (MobileNetV3, distilled from Model 1).',
    'model3': 'Reads a whole drawing in one pass (CNN + BiGRU + CTC): characters, numbers and whole words, including vowel signs.',
}

# --------------------------------------------------------------------------------------------- labels, text, decoding
N = lambda s: unicodedata.normalize('NFC', s)
VOWELS = 'অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ'.split()
CONSONANTS = 'ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ট ঠ ড ঢ ণ ত থ দ ধ ন প ফ ব ভ ম য র ল শ ষ স হ ড় ঢ় য় ৎ ং ঃ ঁ'.split()
DIGITS = list('০১২৩৪৫৬৭৮৯')
COMPOUNDS = 'ক্ষ জ্ঞ ঞ্চ ঞ্ছ ঞ্জ ত্ত ত্র দ্ধ দ্ব ন্ত ন্দ ন্ধ ম্প ম্ফ ম্ব ম্ভ ম্ম ল্ক ল্গ ল্প ল্ট শ্চ ষ্ঠ স্ত'.split()
CLASSES = [N(c) for c in VOWELS + CONSONANTS + DIGITS + COMPOUNDS]       # folder k  ->  CLASSES[k-1]
assert len(CLASSES) == 84 == len(set(CLASSES))
KARS = ['\u09be', '\u09bf', '\u09c0', '\u09c1', '\u09c2', '\u09c3', '\u09c7', '\u09c8', '\u09d7']  # া ি ী ু ূ ৃ ে ৈ ৗ
PRE = {'\u09bf', '\u09c7', '\u09c8'}                 # ি ে ৈ are written before their consonant
VOCAB = ['<blank>'] + CLASSES + KARS                 # CTC tokens; class c has token id c+1
TID = {t: i for i, t in enumerate(VOCAB)}
DIG = [0] + [TID[d] for d in DIGITS]
_TOKS = sorted(VOCAB[1:], key=len, reverse=True)


def to_visual(word):
    """Unicode word -> tokens in writing order (ো = ে+া, ৌ = ে+ৗ). None if it uses symbols the model cannot write."""
    s, out, i = N(word).replace('\u09cb', '\u09c7\u09be').replace('\u09cc', '\u09c7\u09d7'), [], 0
    while i < len(s):
        t = next((t for t in _TOKS if s.startswith(t, i)), None)
        if t is None: return None
        out.insert(len(out) - 1, t) if t in PRE and out and out[-1] not in KARS else out.append(t)
        i += len(t)
    return out


def from_visual(toks):
    """Tokens in writing order -> Unicode text."""
    t, i = list(toks), 0
    while i < len(t) - 1:
        if t[i] in PRE and t[i + 1] not in KARS: t[i], t[i + 1] = t[i + 1], t[i]; i += 2
        else: i += 1
    return N(''.join(t).replace('\u09c7\u09be', '\u09cb').replace('\u09c7\u09d7', '\u09cc'))


def ink(g):
    """Gray/BGR/RGBA image -> uint8, bright strokes on black, faint paper noise removed."""
    g = np.asarray(g)
    if g.ndim == 3:
        g = g[..., 3] if g.shape[2] == 4 and g[..., 3].min() < 255 else cv2.cvtColor(g[..., :3], cv2.COLOR_BGR2GRAY)
    g = g.astype(np.uint8)
    if g.mean() > 127: g = 255 - g
    return np.where(g > 50, g, 0).astype(np.uint8)


def crop(g):
    ys, xs = np.nonzero(g)
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(xs) else None


def prep(g, size=64, fit=52, info=False):
    """One character -> canonical 64x64: cropped, deslanted, longest side 52, centred (used to build the glyph bank)."""
    g = crop(ink(g)); z = np.zeros((size, size), np.uint8)
    if g is None or (g > 0).sum() < 5: return (z, 0.) if info else z
    m = cv2.moments(g); sk = float(np.clip(m['mu11'] / (m['mu02'] + 1e-6), -.4, .4))
    h, w = g.shape; pad = int(abs(sk) * h) + 1
    g = crop(cv2.warpAffine(g, np.float32([[1, -sk, pad + sk * h / 2], [0, 1, 0]]), (w + 2 * pad, h)))
    h, w = g.shape; s = fit / max(h, w)
    g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    y, x = (size - g.shape[0]) // 2, (size - g.shape[1]) // 2; z[y:y + g.shape[0], x:x + g.shape[1]] = g
    return (z, sk) if info else z


def prep_line(g, H=64, maxw=512):
    """Model input for ANY drawing (character, number, word): height 64, width a multiple of 4 (16..512)."""
    g = crop(ink(g))
    if g is None: return np.zeros((H, 16), np.uint8)
    g = cv2.copyMakeBorder(g, 4, 4, 4, 4, cv2.BORDER_CONSTANT)
    w = min(maxw, max(16, round(g.shape[1] * H / g.shape[0])))
    return cv2.resize(g, (w - w % 4, H), interpolation=cv2.INTER_AREA)


def softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / T; z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def greedy(P):
    """Best-path CTC decoding of (T, V) probabilities; token 0 is the blank."""
    a = np.asarray(P).argmax(-1); keep = (a != 0) & np.r_[True, a[1:] != a[:-1]]
    return tuple(a[keep].tolist())


def beam(P, prefixes=None, words=None, width=10, prune=1e-3):
    """CTC prefix beam search, optionally limited to a lexicon. Returns [(tokens, relative mass)], best first."""
    beams = {(): (1.0, 0.0)}
    for p in P:
        nxt, cs = {}, [c for c in np.nonzero(p > prune)[0] if c]
        def add(k, b=0., nb=0.):
            ob, onb = nxt.get(k, (0., 0.)); nxt[k] = (ob + b, onb + nb)
        for pre, (pb, pnb) in beams.items():
            add(pre, b=(pb + pnb) * p[0])
            for c in cs:
                new = pre + (c,)
                if pre and pre[-1] == c: add(pre, nb=pnb * p[c]); ext = pb * p[c]
                else: ext = (pb + pnb) * p[c]
                if prefixes is None or new in prefixes: add(new, nb=ext)
        best = sorted(nxt.items(), key=lambda kv: -sum(kv[1]))[:width]
        z = sum(sum(v) for _, v in best) or 1.
        beams = {k: (b / z, nb / z) for k, (b, nb) in best}
    return sorted(((k, b + nb) for k, (b, nb) in beams.items() if words is None or k in words), key=lambda kv: -kv[1])


def ctc_prob(P, lab):
    """True probability (all alignments) that the model reads exactly `lab`."""
    if not lab: return float(np.prod(P[:, 0]))
    ext = np.zeros(2 * len(lab) + 1, np.int64); ext[1::2] = lab
    skip = np.zeros(len(ext), bool); skip[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])
    a = np.zeros(len(ext)); a[0] = P[0, 0]; a[1] = P[0, ext[1]]
    for t in range(1, len(P)):
        n = a.copy(); n[1:] += a[:-1]; n[2:] += np.where(skip[2:], a[:-2], 0.); a = n * P[t, ext]
    return float(a[-1] + a[-2])


def make_lex(words):
    ids = [tuple(TID[t] for t in v) for v in map(to_visual, words) if v]
    return set(ids), {w[:i] for w in ids for i in range(1, len(w) + 1)}


def candidates(P, mode, lex=None):
    """Candidate readings with their TRUE probabilities. mode: 'number' (digits only) or 'word'."""
    if mode == 'number':
        Q = np.zeros_like(P); Q[:, DIG] = P[:, DIG]; Q /= Q.sum(1, keepdims=True)
        c = [k for k, _ in beam(Q, width=8)]
    else:
        c = [k for k, _ in beam(P, width=10)]
        if lex: c += [k for k, _ in beam(P, lex[1], lex[0], width=10)[:4]]
    return [(k, ctc_prob(P, k)) for k in dict.fromkeys(c) if k]


def rank(cands, lex=None, bonus=3.0, k=3):
    """Best-first: log-probability, plus `bonus` for lexicon words (a soft lexicon prior, never a hard limit)."""
    key = lambda c: np.log(c[1] + 1e-12) + (bonus if lex and c[0] in lex[0] else 0.)
    return sorted(cands, key=key, reverse=True)[:k]


def char_probs(P):
    """Single glyph: peak evidence per class (84,) and the normalised distribution."""
    S = P[:, 1:1 + len(CLASSES)].max(0); return S, S / S.sum()

REJ = 84          # Models 1 and 2 have 85 outputs: the 84 classes + "not a character"
NCLS = 85
DEFAULT_WORDS = '''জল ঘর কলম বই ফল পথ মন বন রথ নখ দশ শত চল বল দল সরল গরম নরম তরল কমল সফল নকল ফসল বদল শহর খবর নজর বছর
অমর অলস ঈগল উৎস একক ওজন ঔষধ কদম গজল চরণ জনম তপন দমন নয়ন পবন ভবন মরণ যখন লবণ শয়ন সড়ক হরণ অঞ্চল সুন্দর বন্দর
মানুষ কথা বাড়ি পাখি নদী আকাশ বাতাস সকাল বিকাল রাত দিন ফুল পাতা মাটি পানি আলো ছায়া মেঘ তারা সূর্য সাগর পাহাড় গাছ
ভালো মন্দ সুখ দুঃখ হাসি গান কবিতা ছবি দেশ ভাষা বাংলা জীবন সময় কাজ খেলা খাবার ভাত মাছ দুধ চিনি মধু তেল নুন
জগৎ আম আজ আর এক ইট উট ঋণ ঈদ মা বাবা ভাই বোন নৌকা দিনরাত লাল নীল কালো সাদা হলুদ সবুজ বড় ছোট কুকুর বিড়াল গরু ছাগল
হাতি বাঘ সিংহ সাগর শীত জামা জুতা টাকা বাজার দোকান গাড়ি রেল খাতা কাগজ নাচ মাঠ বাগান পুকুর ঘড়ি চাবি দরজা জানালা টেবিল
চেয়ার বালিশ সমাজ পরিবার ভালোবাসা আশা সোনা রূপা লোহা মাথা চোখ কান নাক মুখ পা হাত পেট বুক শরীর রোগ শিশু রাজা রানি কবি
উপহার বছর মাস ঢাকা খুলনা রাজশাহী সিলেট বরিশাল রংপুর মেলা পিঠা চা ঢোল ছুটি চিঠি ডাক পিতা মাতা ঘুম'''.split()


# --------------------------------------------------------------------------------------------- glyph segmentation
def segment_glyphs(img):
    """Cut a drawn number or word into glyph crops, left to right (the character models read one glyph at a time).

    Removes the headline (matra) that joins the letters, groups the connected strokes, merges parts that share the same
    columns (vowel signs below, dots) and gives every crop its own piece of headline back.
    Returns [(crop, (x0, y0, x1, y1))] in the coordinates of the tightly cropped drawing."""
    g = crop(ink(img))
    if g is None:
        return []
    H, W = g.shape
    b = (g > 60).astype(np.uint8)
    work, band = b.copy(), None
    if W > 1.3 * H:                                              # looks like a word: find a long horizontal headline
        rows = b.sum(1)
        r = int(rows[:max(2, int(.45 * H))].argmax())
        if rows[r] > .55 * W:
            dt = cv2.distanceTransform(b, cv2.DIST_L2, 3)
            w = max(2, int(round(2 * np.percentile(dt[dt > 0], 90))))
            band = (max(0, r - w // 2 - 1), min(H, r + w // 2 + 2))
            work[band[0]:band[1]] = 0
    n, _, st, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    comps = sorted((int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2]), int(st[i, 1] + st[i, 3]), int(st[i, 4])) for i in range(1, n))
    if not comps:
        return [(g, (0, 0, W, H))]
    big = max(c[4] for c in comps)
    groups = []
    for x0, y0, x1, y1, a in comps:
        if groups:
            gx0, gy0, gx1, gy1 = groups[-1]
            overlap = min(x1, gx1) - max(x0, gx0)
            if overlap > .3 * min(x1 - x0, gx1 - gx0) or (a < .04 * big and x0 < gx1):   # shares its columns, or a stray dot
                groups[-1] = [min(gx0, x0), min(gy0, y0), max(gx1, x1), max(gy1, y1)]
                continue
        if a >= .02 * big or not groups:                          # tiny specks with no neighbour are dropped
            groups.append([x0, y0, x1, y1])
    out = []
    for x0, y0, x1, y1 in groups:
        top = min(y0, band[0]) if band else y0                    # take the glyph's own headline back
        c = crop(g[top:y1, x0:x1])
        if c is not None:
            out.append((c, (x0, top, x1, y1)))
    return out or [(g, (0, 0, W, H))]


# --------------------------------------------------------------------------------------------- v5 glyph normalisation
def _skeleton_v5(bw):
    img = (bw > 127).astype(np.uint8) * 255
    skel, elem = np.zeros_like(img), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(32):
        eroded = cv2.erode(img, elem)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, cv2.dilate(eroded, elem)))
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def prep_v5(img, size=64):
    """The glyph normalisation of the v5 notebook, which its Models 1 and 2 were trained on: Otsu binary, longest side 48 px,
    centred, deslanted, strokes redrawn as a 3 px skeleton, centred by mass. Used when config.json says preprocess = "v5"."""
    _, bw = cv2.threshold(ink(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(bw > 30)
    if len(xs) < 10:
        return np.zeros((size, size), np.uint8)
    c = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = c.shape
    scale = 48.0 / max(h, w, 1)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    canvas = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
    m = cv2.moments((canvas > 64).astype(np.uint8), binaryImage=True)
    if abs(m['mu02']) > 1e-3:
        skew = np.clip(m['mu11'] / m['mu02'], -0.4, 0.4)
        canvas = cv2.warpAffine(canvas, np.float32([[1, skew, -0.5 * size * skew], [0, 1, 0]]), (size, size),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    redrawn = cv2.dilate(_skeleton_v5(canvas), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    mc = cv2.moments((redrawn > 0).astype(np.uint8), binaryImage=True)
    if mc['m00'] > 0:
        cx, cy = mc['m10'] / mc['m00'], mc['m01'] / mc['m00']
        redrawn = cv2.warpAffine(redrawn, np.float32([[1, 0, size / 2.0 - cx], [0, 1, size / 2.0 - cy]]), (size, size), borderValue=0)
    return redrawn


# --------------------------------------------------------------------------------------------- the models
def _session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get('ORT_THREADS', 1))      # one thread: comparable, predictable timings
    return ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])


def _result(text='', top3=(), confidence=0., decision='reject', mode='char', glyphs=0):
    return dict(text=text, top3=list(top3), confidence=float(confidence), decision=decision, mode=mode, glyphs=glyphs)


class ClassifierModel:
    """Models 1 and 2: one 64x64 glyph in, 85 scores out. Words and numbers are read glyph by glyph after segmentation."""
    kind = 'classifier'

    def __init__(self, key, cfg, models_dir=None):
        self.key, self.c, self.title = key, cfg, MODEL_TITLES[key]
        self.s = _session(Path(models_dir or MODELS_DIR) / cfg['file'])
        self.inp = self.s.get_inputs()[0].name                               # 'x' here, 'input' in the v5 notebook's export
        self.pre = prep_v5 if cfg.get('preprocess') == 'v5' else prep

    def probs(self, part):
        """Calibrated probabilities (85,) for one glyph (any drawing of one glyph)."""
        x = self.pre(part)[None, None].astype(np.float32) / 255
        return softmax(self.s.run(None, {self.inp: x})[0][0], self.c['T'])

    def glyph_probs(self, img):
        return self.probs(img)

    def _glyph(self, part, digits_only=False):
        p = self.probs(part)
        if digits_only:
            q = np.zeros_like(p)
            q[50:60] = p[50:60]
            p = q / max(q.sum(), 1e-9)
        order = [int(i) for i in np.argsort(-p) if i != REJ]
        cls = REJ if (not digits_only and int(p.argmax()) == REJ) else order[0]
        return dict(cls=cls, conf=float(p[cls]), p=p, order=order)

    def predict(self, img, mode='auto'):
        g = crop(ink(img))
        if g is None:
            return _result(mode='char' if mode == 'auto' else mode)
        parts = [g] if mode == 'char' else [c for c, _ in segment_glyphs(g)] or [g]
        res = [self._glyph(p, digits_only=(mode == 'number')) for p in parts]
        kept = [r for r in res if r['cls'] != REJ]                 # what the model calls "not a character" is dropped
        if mode == 'auto':
            mode = 'char' if len(parts) == 1 else 'number' if kept and all(CLASSES[r['cls']] in DIGITS for r in kept) else 'word'
        if not kept:
            return _result(mode=mode, glyphs=len(parts))
        tau = self.c['tau']['char']
        accept = all(r['conf'] >= tau for r in kept) and len(kept) == len(res)
        text = ''.join(CLASSES[r['cls']] for r in kept)
        if len(kept) == 1:
            r = kept[0]
            top3 = [(CLASSES[i], float(r['p'][i])) for i in r['order'][:3]]
            return _result(text, top3, r['conf'], 'accept' if accept else 'suggest', mode, len(parts))
        weak = min(range(len(kept)), key=lambda i: kept[i]['conf'])      # alternatives: swap the least certain glyph
        rest = float(np.prod([r['conf'] for i, r in enumerate(kept) if i != weak]))
        top3 = [(text, rest * kept[weak]['conf'])]
        for alt in kept[weak]['order'][1:3]:
            sw = [CLASSES[r['cls']] if i != weak else CLASSES[alt] for i, r in enumerate(kept)]
            top3.append((''.join(sw), rest * float(kept[weak]['p'][alt])))
        return _result(text, top3, min(r['conf'] for r in kept), 'accept' if accept else 'suggest', mode, len(parts))


class SequenceModel:
    """Model 3: a whole drawing in, a sequence of tokens out (CTC). Reads characters, numbers and whole words."""
    kind = 'sequence'

    def __init__(self, key, cfg, models_dir=None):
        d = Path(models_dir or MODELS_DIR)
        self.key, self.c, self.title = key, cfg, MODEL_TITLES[key]
        self.s = _session(d / cfg['file'])
        words = (d / 'lexicon.txt').read_text(encoding='utf-8').split() if (d / 'lexicon.txt').exists() else DEFAULT_WORDS
        self.lex = make_lex(words)

    def raw(self, img):
        """One forward pass: CTC logits (T, V) and classifier-head logits (84,)."""
        o = self.s.run(None, {'x': prep_line(img)[None, None].astype(np.float32) / 255})
        return o[0][0], (o[1][0] if len(o) > 1 else None)             # the first release had no classifier head: CTC readout only

    def glyph_probs(self, img):
        lg, ax = self.raw(img)
        c = self.c
        wa = c.get('wa', 0.) if ax is not None else 0.
        p = (1 - wa) * char_probs(softmax(lg, c['Tc']))[1] + (wa * softmax(ax, c.get('Ta', 1.)) if wa else 0.)
        return np.r_[p, 0.]                                           # 85 entries, like the character models

    def predict(self, img, mode='auto'):
        c = self.c
        if crop(ink(img)) is None:
            return _result(mode='char' if mode == 'auto' else mode)
        lg, ax = self.raw(img)
        P = softmax(lg, c['T'])
        free = greedy(P)
        if not free:
            return _result(mode='char' if mode == 'auto' else mode)
        if mode == 'auto':
            mode = 'char' if len(free) == 1 else 'number' if all(VOCAB[t] in DIGITS for t in free) else 'word'
        if mode == 'char':
            S, p1 = char_probs(softmax(lg, c['Tc']))
            wa = c.get('wa', 0.) if ax is not None else 0.
            p = (1 - wa) * p1 + (wa * softmax(ax, c.get('Ta', 1.)) if wa else 0.)
            top = np.argsort(-p)[:3]
            conf = float(p[top[0]])
            dec = 'reject' if S.max() < c['tau']['ink'] else 'accept' if conf >= c['tau']['char'] else 'suggest'
            return _result(CLASSES[top[0]], [(CLASSES[i], float(p[i])) for i in top], conf, dec, mode, 1)
        lex = self.lex if mode == 'word' else None
        r = rank(candidates(P, mode, lex), lex, c['bonus'])
        if not r:
            return _result(mode=mode)
        txt = lambda k: from_visual([VOCAB[i] for i in k])
        best, conf = r[0]
        ok = conf >= c['tau'][mode] and (mode == 'number' or best in self.lex[0] or greedy(P) == best)
        return _result(txt(best), [(txt(k), float(s)) for k, s in r], conf, 'accept' if ok else 'suggest', mode, len(free))


def load_config(models_dir=None):
    p = Path(models_dir or MODELS_DIR) / 'config.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'models': {}}


def load_models(keys=None, models_dir=None):
    """{'model1': ..., 'model2': ..., 'model3': ...} for every model that has been trained; each is warmed up once."""
    cfg = load_config(models_dir)['models']
    out = {}
    for k in (keys or MODEL_KEYS):
        if k in cfg and (Path(models_dir or MODELS_DIR) / cfg[k]['file']).exists():
            out[k] = (SequenceModel if k == 'model3' else ClassifierModel)(k, cfg[k], models_dir)
            out[k].predict(np.zeros((64, 128), np.uint8) + 255 * (np.arange(128) % 16 < 3).astype(np.uint8), 'char')   # warm-up
    return out


def timed(model, img, mode='auto', repeats=3):
    """(result, median milliseconds) for the whole pipeline: preprocessing, network, decoding."""
    res, ts = None, []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        res = model.predict(img, mode)
        ts.append((time.perf_counter() - t0) * 1000)
    return res, float(np.median(ts))


def list_samples(models_dir=None):
    """[(file name, expected text, kind)] of the synthetic sample drawings written by train.py."""
    d = Path(models_dir or MODELS_DIR) / 'samples'
    p = d / 'samples.json'
    if not p.exists():
        return []
    meta = json.loads(p.read_text(encoding='utf-8'))
    return [(f, m['text'], m['kind']) for f, m in meta.items() if (d / f).exists()]


# --------------------------------------------------------------------------------------------- MLflow helpers
def rebase_artifact_uris():
    """Runs recorded on another machine (for example Kaggle) keep that machine's absolute artifact paths in mlflow.db.
    Rewrite them to this project folder so the artifacts open here, and inside Docker."""
    db = MLFLOW_DIR / 'mlflow.db'
    if not db.exists():
        return 0
    new_root = BASE_DIR.as_uri()
    pat = re.compile(r'^(?:file://)?(?P<pre>.*?)/artifacts/mlflow/store(?P<rest>.*)$')
    changed = 0
    con = sqlite3.connect(db)
    try:
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        for table, col in (('experiments', 'artifact_location'), ('runs', 'artifact_uri'), ('logged_models', 'artifact_location'),
                           ('model_versions', 'source')):
            if table not in tables:
                continue
            for rowid, val in con.execute(f'select rowid, {col} from {table}').fetchall():
                m = pat.match(val or '')
                if not m:
                    continue
                new = f"{new_root}/artifacts/mlflow/store{m.group('rest')}"
                if new != val:
                    con.execute(f'update {table} set {col}=? where rowid=?', (new, rowid))
                    changed += 1
        con.commit()
    finally:
        con.close()
    return changed


def mlflow_setup():
    """Point MLflow at the fixed store (creating the folder, fixing moved paths) and return the module."""
    import mlflow
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    rebase_artifact_uris()
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        mlflow.search_experiments(max_results=1)
    except Exception as e:                       # a database only opens with the MLflow version that wrote it
        if 'database schema' in str(e):
            raise RuntimeError(f"The MLflow database in {MLFLOW_DIR} was written by a different MLflow version than the installed {mlflow.__version__}. "
                               "Delete the folder artifacts/mlflow and run  python train.py  again (requirements.txt pins one MLflow version for this reason).") from e
        raise
    return mlflow


def get_experiment_id(mlflow, name):
    """Create the experiment once, with its artifacts under the fixed store folder."""
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(name, artifact_location=(MLFLOW_STORE / name).as_uri())
