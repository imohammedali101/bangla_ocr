"""predict_cli.py - `python train.py predict`: run the registered models on drawings, with timing, and score them if you know the answers.

    python train.py predict drawing.png                         # all three models on one image
    python train.py predict drawing.png --expect মানুষ           # ... and say which models got it right
    python train.py predict my_drawings/ --mode word            # every png/jpg in a folder
    python train.py predict my_drawings/ --models model3        # one model only

In a folder a file named  <expected text>_<anything>.png  (for example  মানুষ_1.png,  ০৮৪৫৬২_2.png,  ক_3.png)  is scored against its name,
so a folder of your own drawings is a real test set. Needs only numpy, OpenCV and ONNX Runtime (requirements.txt): no torch, no dataset.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import bangla_common as bc

IMG = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def read_image(path):
    """Read a drawing (also from a path with non-ASCII characters, which cv2.imread cannot open on Windows). Alpha is kept."""
    if bc.cv2 is None:
        sys.exit('OpenCV is missing: pip install opencv-python-headless  (or: pip install -r requirements.txt)')
    img = bc.cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), bc.cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'cannot read {path}')
    return img


def expected_from_name(path):
    stem = Path(path).stem
    return bc.N(stem.rsplit('_', 1)[0]) if '_' in stem else None


def main(argv):
    ap = argparse.ArgumentParser(prog='train.py predict', description='Run the registered models on drawings (timing, and accuracy when the answers are known).')
    ap.add_argument('paths', nargs='+', help='image files or folders')
    ap.add_argument('--models', default='all', help='all, or a comma list of model1,model2,model3')
    ap.add_argument('--mode', default='auto', choices=['auto', 'word', 'number', 'char'], help='what is drawn (default: detect)')
    ap.add_argument('--repeats', type=int, default=3, help='timing: median of this many runs (default 3)')
    ap.add_argument('--expect', help='the right answer, for a single image')
    ap.add_argument('--home', help='project folder (default: this folder)')
    a = ap.parse_args(argv)
    if a.home:
        bc.set_home(a.home)
    try:
        sys.stdout.reconfigure(encoding='utf-8')                  # Bengali text in a Windows console
    except Exception:
        pass
    files = []
    for p in map(Path, a.paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG) if p.is_dir() else [p]
    if not files:
        sys.exit('no images found')
    keys = None if a.models == 'all' else [k.strip() for k in a.models.split(',')]
    try:
        models = bc.load_models(keys)
    except ImportError as e:
        sys.exit(f'{e}\nInstall the requirements:  pip install -r requirements.txt')
    if not models:
        sys.exit(f'No models in {bc.MODELS_DIR}. Register them first: python train.py extract --onnx model1=... --onnx model2=... --onnx model3=...')
    single = len(files) == 1
    rows = []
    print(f'{"image":<24}{"model":<26}{"expected":<12}{"predicted":<14}{"":<3}{"conf":>6}  {"decision":<9}{"read as":<9}{"ms":>7}')
    for f in files:
        img = read_image(f)
        exp = bc.N(a.expect) if (a.expect and single) else expected_from_name(f)
        for key, m in models.items():
            res, ms = bc.timed(m, img, a.mode, a.repeats)
            ok = None if exp is None else res['text'] == exp
            rows.append(dict(model=key, title=m.title, ok=ok, ms=ms))
            mark = ' ' if ok is None else ('✓' if ok else '✗')
            print(f'{f.name[:23]:<24}{m.title:<26}{(exp or "-"):<12}{(res["text"] or "-"):<14}{mark:<3}{res["confidence"]:>6.1%}  {res["decision"]:<9}{res["mode"]:<9}{ms:>7.1f}')
    print('\nsummary (median time over all images; accuracy only where the answer is known):')
    for key, m in models.items():
        mine = [r for r in rows if r['model'] == key]; scored = [r for r in mine if r['ok'] is not None]
        acc = f'{sum(r["ok"] for r in scored)}/{len(scored)} correct = {sum(r["ok"] for r in scored) / len(scored):.1%}' if scored else 'no answers known'
        print(f'  {m.title:<26}{acc:<32}median {np.median([r["ms"] for r in mine]):.1f} ms')
    return 0
