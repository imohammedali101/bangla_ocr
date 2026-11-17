"""make_submission.py - build the source-code-only zip the assignment asks for.

    python make_submission.py                       # -> bangla-ocr-assignment-source.zip
    python make_submission.py --with-screenshots    # also put screenshots/*.png into the zip

Only the files listed below go in. Datasets, trained models (models/), MLflow runs (artifacts/), work/, caches,
virtual environments and other generated files are never included. The GitHub repository, in contrast, DOES contain
models/ and artifacts/ (docker build needs them), see README.md.
"""
import argparse
import zipfile
from pathlib import Path

SOURCE = ['train.py', 'notebook_extract.py', 'predict_cli.py', 'app.py', 'bangla_common.py', 'make_submission.py', 'requirements.txt', 'requirements-train.txt',
          'Dockerfile', 'docker-compose.yml', '.dockerignore', '.gitignore', 'README.md', 'labels.json']
MAX_BYTES = 2_000_000          # source code is small; anything bigger means something generated slipped in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-screenshots', action='store_true')
    ap.add_argument('--out', default='bangla-ocr-assignment-source.zip')
    a = ap.parse_args()
    base = Path(__file__).resolve().parent
    files = [f for f in SOURCE if (base / f).exists()]
    if a.with_screenshots:
        files += sorted(str(p.relative_to(base)) for p in (base / 'screenshots').glob('*.png'))
    missing = [f for f in SOURCE if not (base / f).exists()]
    if missing:
        print('not found (skipped):', ', '.join(missing))
    total = sum((base / f).stat().st_size for f in files)
    if total > MAX_BYTES:
        raise SystemExit(f'{total / 1e6:.1f} MB is too large for a source-only zip; check the list')
    with zipfile.ZipFile(base / a.out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(base / f, f'bangla-ocr-assignment/{f}')
    print(f'wrote {a.out}: {len(files)} files, {total / 1e3:.0f} KB uncompressed')
    for f in files:
        print('  ', f)


if __name__ == '__main__':
    main()
