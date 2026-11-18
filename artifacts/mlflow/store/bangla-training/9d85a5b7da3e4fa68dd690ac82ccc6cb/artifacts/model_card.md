# Model 3 · Lekha-CRNN

**Trained in a Kaggle notebook, not with train.py** (bangla-hwr-model3-f.ipynb (training) + bangla-hwr-model3-eval.ipynb (testing)). This run was created by `train.py extract`: it records what the executed notebook printed and displayed; nothing was retrained.

Reads a whole drawing in one pass (CNN + BiGRU + CTC): characters, numbers and whole words, including vowel signs.

- epochs: 40 (40 printed by the notebook)
- final validation score: 96.93%
- single glyph on the notebook's held-out writers: top-1 92.86%, top-3 98.00%, calibration error 0.67%
- numbers (1-8 digits): exact 93.80%
- random-syllable words (unseen words): exact 79.10%
- lexicon words: exact 97.00% (NOT held out: this model saw every word of the list in training)
- glyph accuracy: clear forms 97.31%, low-legibility forms 83.69%; compound characters 90.14%
- ONNX: 6.9 MB, matches PyTorch on 100.00% of greedy readings; targets passed 1/7
