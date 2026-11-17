"""notebook_extract.py - `python train.py extract`: turn the executed Kaggle notebooks into MLflow runs, locally.

Nothing is trained or re-computed here. The three notebooks already did the training and the testing; this reads what they
recorded and puts it into MLflow, so the UI shows it:

    bangla-hwr-v5.ipynb                  -> Model 1 (ResNet), Model 2 (MobileNet) and the baseline CNN: settings, epoch curves, calibration,
                                            test results, export sizes, architecture, the code that trained them, the printed logs
    bangla-hwr-model3-f.ipynb            -> Model 3 (CRNN) training: settings, 40-epoch curve, data statistics, calibration, the figures it displayed
    bangla-hwr-model3-eval.ipynb         -> Model 3 testing: results table, per-group / per-writer accuracy, confusions, PASS/FAIL, export, the figures

Every run is tagged source=notebook with the notebook's file name: the record says where the numbers came from.
Needs only: mlflow, numpy, pandas, matplotlib (no torch, no GPU, no dataset, no model files). Optional --onnx files are attached to the runs
and registered in models/ so the app can use them.
"""
import argparse
import ast
import base64
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import bangla_common as bc

TRAINER = 'train.py extract'
STD_FILES = {'model1': 'model1_resnet.onnx', 'model2': 'model2_mobilenet.onnx', 'model3': 'model3_crnn.onnx'}
RX_V5 = re.compile(r'\[(?P<m>baseline|model1|model2)\] Ep (?P<ep>\d+)/(?P<tot>\d+) \| LR=(?P<lr>[0-9.eE+-]+) \| Loss=(?P<loss>[0-9.eE+-]+) \| Val Acc=(?P<acc>[0-9.]+)% \| Val F1=(?P<f1>[0-9.]+)%')
RX_F = re.compile(r'model3 (?P<ep>\d+)/(?P<tot>\d+) loss (?P<loss>[0-9.]+) val-score (?P<s>[0-9.]+) best (?P<best>[0-9.]+) (?P<sec>\d+)s')
RX_CAL5 = re.compile(r'Calibration Complete: T1=(?P<t1>[0-9.]+), T2=(?P<t2>[0-9.]+) \| Weights=(?P<w1>[0-9.]+)/(?P<w2>[0-9.]+) \| Accept>=(?P<acc>[0-9.]+)')
RX_EXPORT5 = re.compile(r'\[(?P<m>model\d_\w+)\] FP32=(?P<fp>[0-9.]+)MB \(Parity=(?P<fpp>[0-9.]+)%\) \| INT8=(?P<i8>[0-9.]+)MB \(Parity=(?P<i8p>[0-9.]+)%, Ships=(?P<ship>True|False)\)')


# ------------------------------------------------------------------------------------------------ reading a notebook
class Notebook:
    """An executed notebook: for every code cell its source, printed output, execute-results and embedded figures."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            sys.exit(f'notebook not found: {path}\nPut the three executed notebooks in the notebooks/ folder, or pass --v5 / --model3-train / --model3-test.')
        nb = json.loads(self.path.read_text(encoding='utf-8'))
        self.name, self.cells, heading = self.path.name, [], ''
        for i, c in enumerate(nb['cells']):
            src = ''.join(c['source'])
            if c['cell_type'] == 'markdown':
                m = re.search(r'^#+\s*(.+)$', src, re.M)
                heading = m.group(1).strip() if m else heading
                continue
            outs = c.get('outputs', [])
            self.cells.append(dict(
                index=i, source=src, heading=heading,
                stdout=''.join(''.join(o['text']) for o in outs if o.get('output_type') == 'stream' and o.get('name') == 'stdout'),
                result=''.join(''.join(o['data'].get('text/plain', '')) for o in outs if o.get('output_type') == 'execute_result'),
                images=[base64.b64decode(o['data']['image/png']) for o in outs if o.get('output_type') in ('display_data', 'execute_result') and 'image/png' in o.get('data', {})]))
        if not any(c['stdout'] for c in self.cells):
            sys.exit(f'{self.name} has no saved outputs. Use the EXECUTED notebook (the copy downloaded after the run), not a clean one.')

    def find(self, pattern, field='stdout'):
        """First code cell whose `field` matches the regex (None if none)."""
        return next((c for c in self.cells if re.search(pattern, c[field])), None)

    def text(self):
        return '\n'.join(c['stdout'] for c in self.cells)


def parse_cfg(source):
    """The settings dict a notebook assigned to CFG (constants only), as {name: value}."""
    def lit(node):
        try:
            return ast.literal_eval(node)
        except Exception:
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'dict':
                return {k.arg: lit(k.value) for k in node.keywords}
            if isinstance(node, ast.Dict):
                return {lit(k): lit(v) for k, v in zip(node.keys, node.values)}
            return None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == 'CFG' for t in node.targets):
            cfg = lit(node.value)
            return {k: v for k, v in (cfg or {}).items() if v is not None and not (isinstance(v, str) and v.startswith('/kaggle'))}
    return {}


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        out.update(flat(v, f'{prefix}{k}.') if isinstance(v, dict) else {f'{prefix}{k}': v})
    return out


def num_row(line):
    """('label', [numbers]) for a results-table row such as 'Numbers (1-8 digits)   0.938  0.0214  0.9807  1000.0'."""
    m = re.match(r'^\s*(?P<label>\S.*?)\s{2,}(?P<nums>[0-9.]+(?:\s+[0-9.]+)*)\s*$', line.replace('\u2013', '-'))
    return (m['label'].strip(), [float(x) for x in m['nums'].split()]) if m else None


ROW_PREFIX = {'Real glyphs (84 classes)': 'char', 'CTC readout only': 'char_ctc', 'classifier head only': 'char_head', 'original slant (reconstructed)': 'char_slant',
              'Numbers (1-8 digits)': 'number', 'numbers 1-4 digits': 'number_1to4', 'numbers 5-8 digits': 'number_5to8',
              'Held-out words, lexicon decoding': 'SEEN_word', 'tier A (lexicon)': 'SEEN_word_tierA', 'tier B (lexicon)': 'SEEN_word_tierB',
              'Held-out words, free reading': 'SEEN_word_free', 'Pseudo-words, free reading': 'pseudo', 'tier A (pseudo)': 'pseudo_tierA', 'tier B (pseudo)': 'pseudo_tierB'}


def parse_results_table(text):
    """The results table Model 3's notebook printed -> {metric: value}. Words from the lexicon are named SEEN_...: that model trained on them."""
    out = {}
    for line in text.split('\n'):
        row = num_row(line)
        if not row or row[0] not in ROW_PREFIX:
            continue
        p, nums = ROW_PREFIX[row[0]], row[1]
        names = ('top1', 'top3', 'macro_f1', 'ece', 'n') if len(nums) == 5 else ('exact', 'cer', 'f1', 'n')
        pre = ('seen_in_training_' + p[5:]) if p.startswith('SEEN_') else p
        out.update({f'{"test_" if not pre.startswith("seen") else ""}{pre}_{n}': v for n, v in zip(names, nums) if n != 'n'})
    return out


def parse_literal(text, pattern):
    m = re.search(pattern, text, re.S)
    try:
        return ast.literal_eval(m.group(1)) if m else None
    except Exception:
        return None


def slug(s, n=44):
    return re.sub(r'[^a-z0-9]+', '-', s.lower().replace('\u00b7', ' ')).strip('-')[:n] or 'figure'


# ------------------------------------------------------------------------------------------------ what each notebook yields
def runs_from_v5(nb):
    """Baseline, Model 1 and Model 2 from the v5 notebook."""
    cfg_cell = next(c for c in nb.cells if 'CFG = {' in c['source'])
    cfg = parse_cfg(cfg_cell['source'])
    train_cell = nb.find(RX_V5.pattern)
    if train_cell is None:
        sys.exit(f'{nb.name}: no "[model1] Ep ../.." lines found. Is this the v5 notebook, executed?')
    data = '\n'.join(c['stdout'] for c in nb.cells if re.search(r'Writer-Independent Split Ready|Engineered Dataset Cached|Loaded Form-based|Device:', c['stdout']))
    dataset = {}
    if (m := re.search(r'Train=(\d+) forms \| Val=(\d+) \| Test=(\d+) \| Total Images=(\d+)', data)):
        dataset.update(forms_train=int(m[1]), forms_val=int(m[2]), forms_test=int(m[3]), images_total=int(m[4]))
    if (m := re.search(r'V1 Clean: Dropped (\d+) blanks, (\d+) duplicates, (\d+)', data)):
        dataset.update(v1_blanks=int(m[1]), v1_duplicates=int(m[2]), v1_mislabels=int(m[3]))
    if (m := re.search(r'Engineered Dataset Cached: Train=(\d+), Val=(\d+), Test=(\d+)', data)):
        dataset.update(engineered_train=int(m[1]), engineered_val=int(m[2]), engineered_test=int(m[3]))
    cal_cell = nb.find('Calibration Complete')
    cal = RX_CAL5.search(cal_cell['stdout']) if cal_cell else None
    ev_cell = nb.find('Evaluation Complete:')
    ev = {}
    if ev_cell:
        t = ev_cell['stdout']
        ev, _ = json.JSONDecoder().raw_decode(t[t.index('{', t.index('Evaluation Complete:')):])
    exp_cell = nb.find(r'\[model1_resnet\] FP32')
    exports = {m['m']: m for m in RX_EXPORT5.finditer(exp_cell['stdout'])} if exp_cell else {}
    out = []
    for key, name, blockname, role in (('baseline', 'baseline-cnn', 'baseline', 'baseline (not one of the three models)'), ('model1', 'model1-resnet', 'model1_resnet', bc.MODEL_ROLES['model1']),
                                       ('model2', 'model2-mobilenet', 'model2_mobile', bc.MODEL_ROLES['model2'])):
        epochs, lines, total = [], [], 0
        for m in RX_V5.finditer(train_cell['stdout']):
            if m['m'] == key:
                epochs.append((int(m['ep']), dict(train_loss=float(m['loss']), val_score=float(m['f1']) / 100, val_accuracy=float(m['acc']) / 100, lr=float(m['lr']))))
                lines.append(m.group(0)); total = int(m['tot'])
        blk = ev.get(blockname, {})
        metrics = {'test_' + ('char_ece' if k == 'ece' else k): (v / 100 if k in ('char_top1', 'char_top3', 'ece') else v) for k, v in blk.items() if k != 'cpu_ms'}   # v5 calls it 'ece'
        if 'cpu_ms' in blk:
            metrics['cpu_ms_in_notebook'] = blk['cpu_ms']
        if key != 'baseline':
            e = exports.get(blockname)
            if e:
                metrics.update(model_size_mb=float(e['fp']), onnx_parity=float(e['fpp']) / 100, int8_size_mb=float(e['i8']), int8_parity=float(e['i8p']) / 100)
            metrics['T'] = float(cal['t1' if key == 'model1' else 't2']) if cal else None
            metrics = {k: v for k, v in metrics.items() if v is not None}
        metrics.update(final_val_f1=epochs[-1][1]['val_score'], final_val_accuracy=epochs[-1][1]['val_accuracy'], final_train_loss=epochs[-1][1]['train_loss'])
        params = {**{f'cfg.{k}': v for k, v in flat(cfg).items() if k not in ('KAGGLE_DATASET_PATHS',)}, **{f'data.{k}': v for k, v in dataset.items()},
                  'epochs': total, 'epochs_printed_by_notebook': len(epochs), 'notebook_device': 'Kaggle GPU (cuda)',
                  'learning_rate_note': 'LR 1e-5 in this notebook (too low for AdamW from scratch)'}
        if key != 'baseline' and cal:
            params.update(calibration_T=metrics['T'], calibration_accept_threshold=float(cal['acc']), calibration_ensemble_weights=f"{cal['w1']}/{cal['w2']}")
        texts = {'training_log.txt': '\n'.join(lines) + '\n',
                 'dataset_and_split_log.txt': data,
                 'test_report.txt': ev_cell['stdout'] if ev_cell else '',
                 'calibration.txt': cal_cell['stdout'] if cal_cell else '',
                 'export_and_inference_log.txt': (exp_cell['stdout'] if exp_cell else ''),
                 'training_code.py': train_cell['source'], 'evaluation_code.py': ev_cell['source'] if ev_cell else ''}
        if key == 'model2' and train_cell['result']:
            texts['model_architecture.txt'] = train_cell['result']
        out.append(dict(key=key, name=name, role=role, source=nb.name, epochs=epochs, total=total, params=params, metrics=metrics, texts=texts, images={}, notebooks=[nb],
                        exports=exports.get(blockname), cal=dict(T=metrics.get('T'), acc=float(cal['acc'])) if (cal and key != 'baseline') else None))
    return out


def run_from_model3(train_nb, test_nb):
    """Model 3: training from the f notebook, testing from the eval notebook."""
    cfg = parse_cfg(next(c for c in train_nb.cells if 'CFG = dict(' in c['source'])['source'])
    ep_cell = train_nb.find(RX_F.pattern)
    if ep_cell is None:
        sys.exit(f'{train_nb.name}: no "model3 N/E loss .." epoch lines found. Is it the executed Model 3 training notebook?')
    epochs, lines, total = [], [], 0
    for m in RX_F.finditer(ep_cell['stdout']):
        epochs.append((int(m['ep']), dict(train_loss=float(m['loss']), val_score=float(m['s']), best_val_score=float(m['best']), epoch_seconds=float(m['sec']))))
        lines.append(m.group(0)); total = int(m['tot'])
    data = '\n'.join(c['stdout'] for c in train_nb.cells if re.search(r'images .+ writers|V1: |V5: |\d+\.\d+ M parameters|\{.train.: ', c['stdout']))
    p = {}
    if (m := re.search(r'(\d+) images \S+ (\d+) writers', data)):
        p.update(images=int(m[1]), writers=int(m[2]))
    if (m := re.search(r"V1: removed (\d+) blanks and (\d+) duplicates", data)):
        p.update(v1_blanks=int(m[1]), v1_duplicates=int(m[2]))
    if (m := re.search(r"\{'train': (\d+), 'val': (\d+), 'test': (\d+)\}", data)):
        p.update(glyphs_train=int(m[1]), glyphs_val=int(m[2]), glyphs_test=int(m[3]))
    if (m := re.search(r"writers: \{'test': (\d+), 'train': (\d+), 'val': (\d+)\}", data)):
        p.update(writers_test=int(m[1]), writers_train=int(m[2]), writers_val=int(m[3]))
    if (m := re.search(r'V1: (\d+) likely mislabels', data)):
        p.update(v1_mislabels=int(m[1]))
    if (m := re.search(r'V5: (\d+) counterfactual glyphs\D+(\d+) train glyphs', data)):
        p.update(v5_counterfactual=int(m[1]), glyph_bank_train=int(m[2]))
    nparams = float(m[1]) * 1e6 if (m := re.search(r'(\d+\.\d+) M parameters', data)) else None

    ev_cell = test_nb.find(r'macro_f1')
    if ev_cell is None:
        sys.exit(f'{test_nb.name}: no results table found. Is it the executed Model 3 evaluation notebook?')
    table = parse_results_table(ev_cell['stdout'])
    metrics = {k: v for k, v in table.items()}
    groups = parse_literal(ev_cell['stdout'], r'Glyph accuracy by group / form quality: (\{.*?\})')
    metrics.update({f'test_acc_{k}': v for k, v in (groups or {}).items()})
    pw = parse_literal(ev_cell['stdout'], r'Per-writer accuracy percentiles \(5/25/50/75/95\): (\[.*?\])')
    metrics.update({f'test_writer_acc_p{q:02d}': v for q, v in zip((5, 25, 50, 75, 95), pw or [])})
    conf = parse_literal(ev_cell['stdout'], r'Top confusions[^\n]*?: (\[.*\])')
    checks = re.findall(r'^(PASS|FAIL) (.+)$', ev_cell['stdout'], re.M)
    metrics.update(targets_passed=sum(1 for s, _ in checks if s == 'PASS'), targets_total=len(checks))
    cal_cell = test_nb.find(r'head weight wa')
    cal_txt = cal_cell['stdout'] if cal_cell else ''
    cal = parse_literal(cal_txt, r"(\{'T': .*?\}\})")
    acc = re.search(r'characters ([0-9.]+)%.*?numbers ([0-9.]+)%.*?words ([0-9.]+)%', cal_txt)
    vg = re.search(r'val glyph top-1: CTC ([0-9.]+).*?head ([0-9.]+).*?blend ([0-9.]+)', cal_txt)
    if acc:
        metrics.update(accepted_char=float(acc[1]) / 100, accepted_number=float(acc[2]) / 100, accepted_word=float(acc[3]) / 100)
    if vg:
        metrics.update(val_glyph_top1_ctc=float(vg[1]), val_glyph_top1_head=float(vg[2]), val_glyph_top1_blend=float(vg[3]))
    exp_cell = test_nb.find(r'ONNX vs PyTorch greedy agreement')
    exp_txt = exp_cell['stdout'] if exp_cell else ''
    shipped = {}
    if (m := re.search(r'agreement ([0-9.]+) \S+ (\S+): exact ([0-9.]+), ([0-9.]+) ms \S+ (\S+): exact ([0-9.]+), ([0-9.]+) ms .*?ship (\S+) \(([0-9.]+) MB\)', exp_txt)):
        metrics.update(onnx_parity=float(m[1]), onnx_fp32_exact_sample=float(m[3]), onnx_fp32_ms=float(m[4]), onnx_int8_exact_sample=float(m[6]), onnx_int8_ms=float(m[7]), model_size_mb=float(m[9]))
        shipped = dict(file=m[8], fp32_ms=float(m[4]), int8_ms=float(m[7]))
    for label, name in (('real glyphs', 'shipped_char_top1'), ('numbers', 'shipped_number_exact'), ('lexicon words', 'shipped_word_exact_SEEN')):
        if (m := re.search(rf'^{label}\s*: ([0-9.]+)', exp_txt, re.M)):
            metrics[name] = float(m[1])
    metrics.update(final_val_score=epochs[-1][1]['val_score'], best_val_score=max(e[1]['best_val_score'] for e in epochs), final_train_loss=epochs[-1][1]['train_loss'],
                   train_seconds=sum(e[1]['epoch_seconds'] for e in epochs))
    if nparams:
        metrics['n_params'] = nparams
    train_cal = train_nb.find(r'lexicon bonus')
    train_eval = train_nb.find(r'macro_f1')
    params = {**{f'cfg.{k}': v for k, v in flat(cfg).items() if k not in ('DATA', 'OUT', 'WORDS', 'GOLDEN')}, **{f'data.{k}': v for k, v in p.items()},
              'epochs': total, 'epochs_printed_by_notebook': len(epochs), 'notebook_device': 'Kaggle GPU (cuda)',
              'training_notebook': train_nb.name, 'testing_notebook': test_nb.name,
              'note_lexicon_words': 'the trained model saw every lexicon word: lexicon-word scores are NOT held-out, random-syllable words are the honest figure'}
    if cal:
        params.update({f'calibration.{k}': v for k, v in flat(cal).items()})
    images = {}
    for tag, nb, desc in (('train', train_nb, 'figure displayed by the training notebook'), ('test', test_nb, 'figure displayed by the testing notebook')):
        for c in nb.cells:
            for n, png in enumerate(c['images']):
                images[f'fig_{tag}_{c["index"]:02d}_{slug(c["heading"])}{"" if n == 0 else f"_{n + 1}"}.png'] = png
    texts = {'training_log.txt': '\n'.join(lines) + '\n',
             'dataset_and_split_log.txt': data,
             'calibration.txt': ('--- printed by the training notebook ---\n' + (train_cal['stdout'] if train_cal else '') + '\n--- printed by the testing notebook ---\n' + cal_txt),
             'test_report.txt': ev_cell['stdout'],
             'training_notebook_test_report.txt': train_eval['stdout'] if train_eval else '',
             'export_and_inference_log.txt': exp_txt,
             'training_code.py': ep_cell['source'], 'evaluation_code.py': ev_cell['source'],
             'top_confusions.json': json.dumps([dict(true=a, predicted=b, count=n) for a, b, n in (conf or [])], ensure_ascii=False, indent=1),
             'targets.txt': '\n'.join(f'{s} {t}' for s, t in checks)}
    return dict(key='model3', name='model3-crnn', role=bc.MODEL_ROLES['model3'], source=f'{train_nb.name} (training) + {test_nb.name} (testing)', epochs=epochs, total=total, params=params,
                metrics=metrics, texts=texts, images=images, notebooks=[train_nb, test_nb], exports=shipped, cal=cal, nparams=nparams, checks=checks, conf=conf)


# ------------------------------------------------------------------------------------------------ MLflow
def curves_png(epochs, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3)); h = pd.DataFrame([dict(epoch=e, **m) for e, m in epochs])
    mk = 'o-' if len(h) <= 12 else '-'
    ax[0].plot(h.epoch, h.train_loss, mk); ax[0].set_title('training loss')
    ax[1].plot(h.epoch, h.val_score, mk); ax[1].set_title('validation score' + (' (macro-F1)' if 'val_accuracy' in h else '')); ax[1].set_ylim(0, 1)
    for a in ax:
        a.set_xlabel('epoch'); a.grid(alpha=.3)
    fig.suptitle(f'{title}: as printed by the notebook ({len(h)} epoch lines)', fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def card(r):
    m = r['metrics']; g = lambda k, f='{:.2%}': (f.format(m[k]) if k in m else 'n/a')
    lines = [f"# {bc.MODEL_TITLES.get(r['key'], 'Baseline CNN')}", '',
             f"**Trained in a Kaggle notebook, not with train.py** ({r['source']}). This run was created by `{TRAINER}`: it records what the executed notebook printed and displayed; nothing was retrained.", '',
             r['role'], '', f"- epochs: {r['total']} ({len(r['epochs'])} printed by the notebook)", f"- final validation score: {g('final_val_f1') if 'final_val_f1' in m else g('final_val_score')}"]
    lines += [f"- single glyph on the notebook's held-out writers: top-1 {g('test_char_top1')}, top-3 {g('test_char_top3')}, calibration error {g('test_char_ece', '{:.2%}')}"]
    if r['key'] == 'model3':
        lines += [f"- numbers (1-8 digits): exact {g('test_number_exact')}", f"- random-syllable words (unseen words): exact {g('test_pseudo_exact')}",
                  f"- lexicon words: exact {g('seen_in_training_word_exact')} (NOT held out: this model saw every word of the list in training)",
                  f"- glyph accuracy: clear forms {g('test_acc_q_high')}, low-legibility forms {g('test_acc_q_low')}; compound characters {g('test_acc_compound')}",
                  f"- ONNX: {g('model_size_mb', '{:.1f}')} MB, matches PyTorch on {g('onnx_parity')} of greedy readings; targets passed {int(m.get('targets_passed', 0))}/{int(m.get('targets_total', 0))}"]
    elif r['key'] != 'baseline':
        lines += [f"- ONNX (fp32): {g('model_size_mb', '{:.2f}')} MB, parity {g('onnx_parity')}; reading time in the notebook {g('cpu_ms_in_notebook', '{:.1f}')} ms per glyph"]
    if r['key'] in ('model1', 'model2'):
        lines += ['', 'Limits: trained with a learning rate of 1e-5, which is too low for AdamW from scratch; it cannot read vowel signs (no class for them). '
                      'The notebook printed only every 12th/16th epoch, so the curve has 5 points.']
    return '\n'.join(lines) + '\n'


def log_run(mlflow, exp_id, r, nb_paths):
    mlflow.set_experiment(experiment_id=exp_id)
    clean = lambda k: re.sub(r'[^0-9A-Za-z_\-. /]', '_', str(k))[:240]
    with tempfile.TemporaryDirectory() as td, mlflow.start_run(run_name=r['name']) as run:
        td = Path(td)
        mlflow.set_tags(dict(model_key=r['key'], role=r['role'][:250], source='notebook', notebook=r['source'][:250], trained_with='Kaggle notebook, NOT train.py',
                             extracted_by=TRAINER, weights_source='trained in the notebook'))
        mlflow.log_params({clean(k): str(v)[:480] for k, v in r['params'].items()})
        for step, m in r['epochs']:
            mlflow.log_metrics(m, step=step)
        mlflow.log_metrics({clean(k): float(v) for k, v in r['metrics'].items() if v == v})
        curves_png(r['epochs'], td / 'curves.png', r['name']); mlflow.log_artifact(str(td / 'curves.png'))
        for name, png in r['images'].items():
            (td / name).write_bytes(png); mlflow.log_artifact(str(td / name))
        for name, text in r['texts'].items():
            if text and text.strip():
                mlflow.log_text(text, name)
        mlflow.log_text(card(r), 'model_card.md')
        for nb in r['notebooks']:
            mlflow.log_artifact(str(nb.path), 'source_notebook')
        if r.get('onnx'):
            mlflow.log_artifact(str(r['onnx']), 'model')
        return run.info.run_id


def delete_previous(mlflow, exp_name):
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        return 0
    old = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"tags.extracted_by = '{TRAINER}'")
    for rid in old['run_id'] if len(old) else []:
        mlflow.delete_run(rid)
    return len(old)


def log_comparison(mlflow, exp_id, runs):
    mlflow.set_experiment(experiment_id=exp_id)
    rows = []
    for r in runs:
        m = r['metrics']
        rows.append({'model': bc.MODEL_TITLES.get(r['key'], 'Baseline CNN'), 'notebook': r['source'], 'glyph_top1': m.get('test_char_top1'), 'glyph_top3': m.get('test_char_top3'), 'ece': m.get('test_char_ece'),
                     'numbers_exact': m.get('test_number_exact'), 'random_words_exact': m.get('test_pseudo_exact'), 'onnx_fp32_mb': m.get('model_size_mb'),
                     'cpu_ms_in_notebook': m.get('cpu_ms_in_notebook', m.get('onnx_fp32_ms'))})
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as td, mlflow.start_run(run_name='compare-all'):
        td = Path(td)
        mlflow.set_tags(dict(kind='comparison', source='notebook', extracted_by=TRAINER,
                             note="each model was tested by its own notebook on its own held-out writers: Models 1 and 2 (and the baseline) on the v5 split, Model 3 on the project split"))
        for r in runs:
            mlflow.log_metrics({f'{r["key"]}_{k}': v for k, v in r['metrics'].items() if k.startswith('test_char') or k in ('model_size_mb', 'cpu_ms_in_notebook')})
        mlflow.log_text(df.to_csv(index=False), 'comparison.csv'); mlflow.log_dict(rows, 'comparison.json')
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4)); lab = [r['model'].split(' · ')[0] for r in rows]
        ax[0].bar(lab, df['glyph_top1'].fillna(0)); ax[0].set_ylim(.4, 1); ax[0].set_title('single glyph top-1 (own notebook split)')
        ax[1].bar(lab, df['onnx_fp32_mb'].fillna(0)); ax[1].set_title('ONNX size, fp32 (MB)')
        ax[2].bar(lab, df['cpu_ms_in_notebook'].fillna(0)); ax[2].set_title('CPU ms in the notebook (not comparable across notebooks)')
        fig.tight_layout(); fig.savefig(td / 'comparison.png', dpi=110); plt.close(fig); mlflow.log_artifact(str(td / 'comparison.png'))
    return df


# ------------------------------------------------------------------------------------------------ registering the notebook-made ONNX files for the app
def register_onnx(runs, specs, lexicon):
    """Copy the ONNX files the notebooks exported into models/ and describe them in models/config.json (calibration read from the notebook logs)."""
    by = {r['key']: r for r in runs}
    bc.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = bc.MODELS_DIR / 'config.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists() else {'models': {}}
    for spec in specs:
        key, _, path = spec.partition('=')
        if key not in STD_FILES:
            sys.exit(f'--onnx expects model1|model2|model3=FILE.onnx (got {spec!r})')
        if not Path(path).exists():
            sys.exit(f'file not found for {key}: {path!r}\nReplace the example path with the real location of the ONNX file you downloaded, for example (Windows):\n'
                     f'  --onnx {key}="C:\\Users\\you\\Downloads\\{STD_FILES[key] if key == "model3" else ("model1_resnet.onnx" if key == "model1" else "model2_mobile.onnx")}"\n(keep the quotes if the path contains spaces)')
        dst = bc.MODELS_DIR / STD_FILES[key]
        if Path(path).resolve() != dst.resolve():           # the file may already be in models/
            shutil.copy(path, dst)
        size = dst.stat().st_size / 1e6
        estimated = False
        try:
            import onnx
            import numpy as np
            n_params = int(sum(int(np.prod(t.dims)) for t in onnx.load(str(dst)).graph.initializer))
        except Exception:
            n_params, estimated = by[key].get('nparams') or int(dst.stat().st_size / 4), True      # no onnx package: Model 3's printed count, else ~4 bytes per fp32 weight
        if key == 'model3':
            c = by[key]['cal'] or {}
            entry = dict(T=c.get('T', 1.1), Tc=c.get('Tc', 1.1), Ta=c.get('Ta', 1.0), wa=c.get('wa', 0.0), bonus=c.get('bonus', 3.0), tau=c.get('tau', dict(ink=.3, char=.95, number=.95, word=.9)))
        else:
            c = by[key]['cal'] or {}
            entry = dict(T=c.get('T') or 1.0, tau=dict(char=c.get('acc', .9), low=.3), preprocess='v5')
        entry.update(file=dst.name, params=n_params, size_mb=size, source=f'notebook: {by[key]["source"]}', split_valid=False, params_estimated=estimated)
        cfg['models'][key] = entry
        by[key]['onnx'] = dst
        print(f'registered {key}: {dst.name} ({size:.1f} MB) in {cfg_path}')
    cfg_path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding='utf-8')
    lab = dict(classes=bc.CLASSES, class_folders={i + 1: c for i, c in enumerate(bc.CLASSES)}, reject_index_models_1_2=bc.REJ, vocab=bc.VOCAB, vowel_signs=bc.KARS)
    for p in (bc.MODELS_DIR / 'labels.json', bc.BASE_DIR / 'labels.json'):
        p.write_text(json.dumps(lab, indent=1, ensure_ascii=False), encoding='utf-8')
    if lexicon:
        shutil.copy(lexicon, bc.MODELS_DIR / 'lexicon.txt')
    elif not (bc.MODELS_DIR / 'lexicon.txt').exists():
        (bc.MODELS_DIR / 'lexicon.txt').write_text('\n'.join(sorted({bc.N(w) for w in bc.DEFAULT_WORDS if bc.to_visual(w)})), encoding='utf-8')


# ------------------------------------------------------------------------------------------------ command
def main(argv):
    ap = argparse.ArgumentParser(prog='train.py extract', description='Put what the executed Kaggle notebooks trained and tested into MLflow (no training, no GPU, no dataset).')
    nbdir = Path(__file__).resolve().parent / 'notebooks'
    ap.add_argument('--v5', default=str(nbdir / 'bangla-hwr-v5.ipynb'), help='executed bangla-hwr-v5.ipynb: Model 1, Model 2 and the baseline (default: notebooks/)')
    ap.add_argument('--model3-train', default=str(nbdir / 'bangla-hwr-model3-f.ipynb'), help='executed bangla-hwr-model3-f.ipynb: Model 3 training (default: notebooks/)')
    ap.add_argument('--model3-test', default=str(nbdir / 'bangla-hwr-model3-eval.ipynb'), help='executed bangla-hwr-model3-eval.ipynb: Model 3 testing (default: notebooks/)')
    ap.add_argument('--onnx', action='append', default=[], metavar='KEY=PATH', help='optional, repeatable: model1|model2|model3=/path/to/file.onnx (attached to the run and registered for the app)')
    ap.add_argument('--lexicon', help='optional: lexicon.txt that came with the Model 3 release')
    ap.add_argument('--no-baseline', action='store_true', help='do not create the baseline-cnn run')
    ap.add_argument('--keep-old', action='store_true', help='keep runs from a previous extraction (default: replace them, so the UI shows one set)')
    ap.add_argument('--home', help='project folder (default: this folder; BANGLA_HOME also works)')
    a = ap.parse_args(argv)
    if a.home:
        bc.set_home(a.home)
    v5, f, ev = Notebook(a.v5), Notebook(a.model3_train), Notebook(a.model3_test)
    runs = [r for r in runs_from_v5(v5) if not (a.no_baseline and r['key'] == 'baseline')] + [run_from_model3(f, ev)]
    runs.sort(key=lambda r: {'model1': 0, 'model2': 1, 'model3': 2, 'baseline': 3}[r['key']])
    if not a.onnx:                                           # no --onnx given: use the files already in models/
        for key, names in (('model1', ('model1_resnet.onnx',)), ('model2', ('model2_mobilenet.onnx', 'model2_mobile.onnx')), ('model3', ('model3_crnn.onnx',))):
            f = next((bc.MODELS_DIR / n for n in names if (bc.MODELS_DIR / n).exists()), None)
            if f is not None:
                a.onnx.append(f'{key}={f}')
        missing = [k for k in STD_FILES if not any(x.startswith(k + '=') for x in a.onnx)]
        print(f"ONNX files found in {bc.MODELS_DIR}: {[x.split('=')[0] for x in a.onnx] or 'none'}" + (f" | missing: {missing}: put them there (see README) and run again, or the app will not offer those models" if missing else ''))
    if a.onnx:
        register_onnx(runs, a.onnx, a.lexicon)
    mlflow = bc.mlflow_setup()
    if not a.keep_old:
        n = delete_previous(mlflow, bc.TRAIN_EXPERIMENT) + delete_previous(mlflow, bc.EVAL_EXPERIMENT)
        if n:
            print(f'replaced {n} run(s) from a previous extraction')
    train_id = bc.get_experiment_id(mlflow, bc.TRAIN_EXPERIMENT)
    for r in runs:
        rid = log_run(mlflow, train_id, r, None)
        print(f"logged {r['name']:<18} from {r['source']}: {len(r['epochs'])} epoch points, {len(r['metrics'])} metrics, {len(r['images'])} figures  (run {rid[:8]})")
    df = log_comparison(mlflow, bc.get_experiment_id(mlflow, bc.EVAL_EXPERIMENT), runs)
    print('\ncomparison (each model on its own notebook\'s held-out writers):'); print(df.round(4).to_string(index=False))
    print(f'\nMLflow store: {bc.MLFLOW_DIR}\nSee it:  mlflow ui --backend-store-uri {bc.TRACKING_URI} --port 5000   ->  http://localhost:5000')
    return 0
