# Model 1 · Lekha-ResNet

**Trained in a Kaggle notebook, not with train.py** (bangla-hwr-v5.ipynb). This run was created by `train.py extract`: it records what the executed notebook printed and displayed; nothing was retrained.

Reads one glyph at a time (ResNet). Words and numbers are read by cutting the drawing into glyphs; it has no class for vowel signs.

- epochs: 60 (5 printed by the notebook)
- final validation score: 92.84%
- single glyph on the notebook's held-out writers: top-1 93.37%, top-3 98.22%, calibration error 0.78%
- ONNX (fp32): 10.92 MB, parity 100.00%; reading time in the notebook 16.4 ms per glyph

Limits: trained with a learning rate of 1e-5, which is too low for AdamW from scratch; it cannot read vowel signs (no class for them). The notebook printed only every 12th/16th epoch, so the curve has 5 points.
