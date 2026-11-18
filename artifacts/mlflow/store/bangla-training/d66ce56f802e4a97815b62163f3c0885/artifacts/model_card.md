# Model 2 · Lekha-Mobile

**Trained in a Kaggle notebook, not with train.py** (bangla-hwr-v5.ipynb). This run was created by `train.py extract`: it records what the executed notebook printed and displayed; nothing was retrained.

Same job as Model 1 with a much smaller network (MobileNetV3, distilled from Model 1).

- epochs: 80 (5 printed by the notebook)
- final validation score: 89.55%
- single glyph on the notebook's held-out writers: top-1 90.53%, top-3 97.37%, calibration error 0.98%
- ONNX (fp32): 6.13 MB, parity 100.00%; reading time in the notebook 10.1 ms per glyph

Limits: trained with a learning rate of 1e-5, which is too low for AdamW from scratch; it cannot read vowel signs (no class for them). The notebook printed only every 12th/16th epoch, so the curve has 5 points.
