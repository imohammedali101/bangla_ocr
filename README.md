# Bangla Handwritten Word Recognition (Assignment 04)

**Name:** `<<< REPLACE WITH YOUR FULL NAME >>>`
**Student ID:** `<<< REPLACE WITH YOUR STUDENT ID >>>`

Draw a Bangla word such as **মানুষ** or **জল** on a canvas and the app predicts the **whole word**. The same app reads numbers (for example ০৮৪৫৬২) and single characters,
lets you pick one of three models, shows how long each takes, and can run all three on the same drawing to compare them. What the models did during training and testing is
recorded in MLflow, and everything runs in one Docker image.

## How the models were produced (read this first)

The three models were developed, **trained and tested in Kaggle notebooks**, and they are not retrained here:

| Model | Trained and tested in | What the notebook did |
|---|---|---|
| 1 · Lekha-ResNet | `bangla-hwr-v5.ipynb` | 60 epochs (learning rate 1e-5, too low for AdamW from scratch), calibration, test, ONNX export |
| 2 · Lekha-Mobile | `bangla-hwr-v5.ipynb` | 80 epochs, same recipe |
| (baseline CNN) | `bangla-hwr-v5.ipynb` | 20 epochs on raw data, kept for the comparison |
| 3 · Lekha-CRNN | `bangla-hwr-model3-f.ipynb` (training), `bangla-hwr-model3-eval.ipynb` (testing) | 40 epochs × 1,500 steps, learning rate 2e-3, calibration, test on unseen writers, ONNX export |

`python train.py` reads those three executed notebooks and records in MLflow what they printed and displayed: settings, epoch curves, data statistics, calibration, test results,
the figures they drew, the code that trained and tested the models. It trains nothing and needs no GPU, no dataset and no Kaggle. Every run it creates is tagged `source = notebook`
with the notebook's file name, and the app's MLflow tab says so. (`python train.py train ...` can also train from scratch; that needs `requirements-train.txt` and is only verified on a tiny fake dataset.)

## Run it: one environment, every command

Needs Python 3.10 or newer (checked with Windows packages for 3.12, 3.13 and 3.14) and nothing else.

**1. Install, once**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # Linux / macOS: source venv/bin/activate   (if PowerShell blocks it: Set-ExecutionPolicy -Scope Process Bypass)
pip install -r requirements.txt
```

**2. Put the files in place**

* `notebooks/`: the three executed notebooks, named `bangla-hwr-v5.ipynb`, `bangla-hwr-model3-f.ipynb`, `bangla-hwr-model3-eval.ipynb`.
* `models/`: the files the notebooks exported (download them from the Kaggle outputs; no renaming): from the v5 run's `release/` folder `model1_resnet.onnx` and `model2_mobile.onnx` (the fp32 files, not int8);
  from the Model 3 testing run's `release/` folder `model3_crnn.onnx` and `lexicon.txt`.

**3. Create the MLflow runs** (this is the assignment's `python train.py`)

```powershell
python train.py
```

It finds the ONNX files in `models/`, writes `models/config.json` (calibration read from the notebook logs) and `labels.json`, and creates the runs in `artifacts/mlflow/`. Running it again replaces its earlier runs.

**4. Look at MLflow**

```powershell
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --port 5000
```

Open http://localhost:5000 and the experiment **`bangla-training`**: four runs (`model1-resnet`, `model2-mobilenet`, `model3-crnn`, `baseline-cnn`), each with parameters, the epoch curve, test metrics, and artifacts
(`curves.png`, `model_card.md`, `training_log.txt`, `test_report.txt`, `calibration.txt`, the training and evaluation code, the source notebook; for Model 3 also six figures, `top_confusions.json`, `targets.txt`).
The experiment **`bangla-evaluation`** holds a `compare-all` run. Save a screenshot of the experiment page as `screenshots/mlflow_experiment.png`.

**5. Test the models and run the app**

```powershell
python train.py predict "C:\path\to\drawing.png"            # all three models on one image, with timing
python train.py predict my_drawings                         # a folder; files named  <what it says>_<n>.png  (for example জল_1.png) are scored
python -m streamlit run app.py                              # the app: http://localhost:8501
```

In the app choose a model in the dropdown (or **All 3 models: compare them**), draw one word large on the canvas, press **Predict**. Save a screenshot of a predicted word as `screenshots/streamlit_app.png`.
(`$env:PYTHONUTF8=1` first if Bengali text looks garbled in the terminal.)

**6. Docker** (Docker Desktop must be running; run step 3 first, the image contains `models/` and `artifacts/mlflow/`)

```powershell
docker build -t bangla-ocr-app:0.1 .
docker run -p 8501:8501 bangla-ocr-app:0.1                  # http://localhost:8501
docker compose up -d                                        # optional: the app and the MLflow UI (http://localhost:5000)
```

**7. Submit** (deadline 17 November 2026, no extension)

1. Fill in your name and student ID at the top of this file.
2. Put your repository link in `github_link.txt`: `https://github.com/YOUR_USER_NAME/YOUR_PROJECT_NAME`.
3. Create a **public** repository on GitHub with that name, then in the project folder:
   ```powershell
   git init
   git add .
   git commit -m "Assignment 04"
   git branch -M main
   git remote add origin https://github.com/YOUR_USER_NAME/YOUR_PROJECT_NAME.git
   git push -u origin main
   ```
   `models/`, `artifacts/`, `notebooks/` and `screenshots/` must be in the repository (`.gitignore` already leaves out `venv/`, `work/` and `*.zip`). Open the repository in a private browser window to check it is public.
4. Build the source-only zip: `python make_submission.py` creates `bangla-ocr-assignment-source.zip` (code only: no models, no MLflow runs, no notebooks, no environment).
5. Submit through https://forms.gle/h8XGbmRYyYHmYeSp6 with what the assignment lists: the zip, the two screenshots and `github_link.txt` (and the repository link).

## How the app works

A **Model** dropdown (Model 1, Model 2, Model 3, or **All 3 models: compare them**; default Model 3). Every prediction shows the **time taken** (median of N runs: preprocessing, network and decoding), the confidence,
the decision (accept / suggest / reject) and the top-3. *All 3 models* runs Model 1, then 2, then 3 on the same drawing and shows each prediction, confidence, time, parameters and size, which model was fastest and most confident,
and whether they agree. Input: draw, upload an image, or a synthetic sample. The **MLflow** tab shows the training runs, every figure and log of the selected run, and the predictions logged by the app.

## Project structure

```
bangla-ocr-assignment/
|-- train.py                 python train.py = notebooks -> MLflow;  also: predict, train, test, import-notebook
|-- notebook_extract.py      the notebook -> MLflow step
|-- predict_cli.py           run the models on drawings from the command line
|-- app.py                   Streamlit app
|-- bangla_common.py         labels, preprocessing, decoding, the three ONNX models, MLflow helpers
|-- requirements.txt         the one environment (also used by the Docker image)
|-- requirements-train.txt   only for training from scratch
|-- Dockerfile  docker-compose.yml  .dockerignore  .gitignore
|-- labels.json              the 84 classes, the CTC tokens, the vowel signs
|-- notebooks/               the three executed Kaggle notebooks
|-- models/                  ONNX files, config.json, lexicon.txt
|-- artifacts/mlflow/        mlflow.db (runs) and store/ (artifacts)
|-- screenshots/             streamlit_app.png, mlflow_experiment.png
`-- make_submission.py       builds the source-only zip
```

The assignment's example lists `models/model.keras`. The models here come from PyTorch notebooks and are ONNX files in `models/`; the other folders are as in the example.

## Approach (what the notebooks did)

**Data.** BanglaLekha-Isolated: 166,058 usable images of 84 classes (50 basic characters, 10 digits, 24 conjuncts) from about 2,000 writers, split **by writer** (80 / 10 / 10), so every test number is for handwriting the models never saw.
The dataset has no word images and no vowel signs, so words are built from the isolated glyphs: same-writer glyphs on a joined headline (matra), vowel signs added (া cut from the writer's আ, the others drawn), labels in writing order
(ি before its consonant), plus dataset engineering (cleaning, canonical glyphs, canvas-style redraw, writer-style transfer, counterfactual dot glyphs, "not a character" samples).

**Models.** Models 1 and 2 classify one 64×64 glyph (85 outputs: 84 classes + "not a character"); words and numbers are read by cutting the drawing into glyphs, which cannot see vowel signs. Model 3 (CNN + BiGRU + CTC) reads a
whole drawing in one pass, including vowel signs, and decodes words with a soft word-list preference. All three are temperature-calibrated.

## Results

As reported by the notebooks (each on its own held-out writers; the real printed numbers, also in MLflow):

| Model | glyph top-1 | glyph top-3 | calibration error | other |
|---|---|---|---|---|
| Baseline CNN (v5) | 50.25% | | | 20 epochs on raw data |
| 1 · Lekha-ResNet (v5) | 93.37% | 98.22% | 0.78% | 16.4 ms per glyph in the notebook; ONNX 10.92 MB |
| 2 · Lekha-Mobile (v5) | 90.53% | 97.37% | 0.98% | 10.1 ms per glyph in the notebook; ONNX 6.13 MB |
| 3 · Lekha-CRNN (model3-f / eval) | 92.86% | 98.00% | 0.67% | numbers 93.8% exact; random-syllable words 79.1%; macro-F1 0.928; ONNX 6.9 MB; 8.7 ms in the eval notebook |

Model 3's lexicon-word score (97.0%) is **not** a held-out result, because that model trained on every word of the list; the random-syllable score (79.1%) is the honest figure for unseen words.
The v5 notebook printed only every 12th / 16th epoch, so the curves of Models 1 and 2 have 5 points.

## Limitations

* **Vowel signs are synthetic.** The dataset has no vowel signs; their appearance in training comes from a generator. Real handwritten words, especially vowel signs, have **not** been measured.
* **Glyph accuracy is the ceiling.** On unseen writers a glyph is read correctly about 93% of the time (top-3 about 98%); legible forms do far better (97%) than low-legibility ones (84%). Look-alikes (ত/৩, ন্ধ/ক্ষ, ঞ্জ/ঞ্ছ) are the commonest errors.
* **Models 1 and 2 were trained with a learning rate that was too low** and cannot read vowel signs; they cut words into glyphs with a simple rule, so words such as মানুষ are expected to go wrong with them. They are here for the comparison.
* **Vocabulary.** The built-in word list is small (189 words). Words outside it can still be read but are less reliable; replace `models/lexicon.txt` for a bigger list.
* Draw large, in the middle of the canvas, one word at a time. On Windows, `python train.py train` (from scratch) runs without worker processes and is slow: train on Linux, WSL or a cloud GPU.
