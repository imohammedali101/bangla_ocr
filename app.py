"""Streamlit app: draw a Bangla word, number or character and read it with one of three models, or compare all three.

    streamlit run app.py                         # the app
    START_MLFLOW_UI=1 streamlit run app.py       # the app and the full MLflow UI (http://localhost:5000)

Predict tab   draw (or upload / pick a sample), choose the model, see the prediction and the time it took.
              "All 3 models" runs Model 1, Model 2 and Model 3 one after the other on the same drawing and compares them.
MLflow tab    training runs, the artifacts of each model (curves, confusion matrix, model card) and the predictions logged here.
"""
import html
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import bangla_common as bc

st.set_page_config(page_title="Bangla handwriting recognizer", page_icon="✍️", layout="wide")
try:
    mlflow = bc.mlflow_setup()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# Streamlit renamed these options (use_container_width / use_column_width -> width="stretch"); work with old and new versions
_NEW_API = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50)
FILL = {"width": "stretch"} if _NEW_API else {"use_container_width": True}
IMG_FILL = {"width": "stretch"} if _NEW_API else {"use_column_width": True}


@st.cache_resource
def start_mlflow_ui():
    """Optional: the full MLflow UI next to the app, on the same fixed store (START_MLFLOW_UI=1)."""
    return subprocess.Popen([sys.executable, "-m", "mlflow", "server", "--host", os.environ.get("MLFLOW_UI_HOST", "127.0.0.1"),
                             "--port", os.environ.get("MLFLOW_UI_PORT", "5000"), "--backend-store-uri", bc.TRACKING_URI,
                             "--artifacts-destination", bc.MLFLOW_STORE.as_uri()])


if os.environ.get("START_MLFLOW_UI") == "1":
    start_mlflow_ui()


@st.cache_resource(show_spinner="Loading the models...")
def get_models():
    return bc.load_models()


models = get_models()
if not models:
    st.error(f"No trained models in `{bc.MODELS_DIR}`. Run `python train.py train --data /path/to/BanglaLekha-Isolated` first.")
    st.stop()
cfg = bc.load_config()["models"]

# ------------------------------------------------------------------------------------------------ sidebar
CHOICES = {f"{m.title}": k for k, m in models.items()}
if len(models) > 1:
    CHOICES["All 3 models: compare them" if len(models) == 3 else f"All {len(models)} models: compare them"] = "all"
ORDER = list(CHOICES)
default = ORDER.index(next(t for t, k in CHOICES.items() if k == "model3")) if "model3" in models else 0
MODES = {"Auto-detect": "auto", "Whole word": "word", "Number": "number", "Single character": "char"}

with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Model", ORDER, index=default, help="Model 3 reads whole words; Models 1 and 2 read one glyph at a time.")
    mode = MODES[st.selectbox("What is drawn?", list(MODES))]
    source = st.radio("Input", ["Draw on the canvas", "Upload an image", "Synthetic sample"])
    stroke = st.slider("Pen width (canvas)", 4, 30, 12, help="Draw large; the pen should be about a tenth of the letter height.")
    repeats = st.number_input("Timing repeats (median)", 1, 20, 3, help="Each model is run this many times; the median time is shown.")
    log_runs = st.checkbox("Log predictions to MLflow", True)
    st.caption(f"Models: `{bc.MODELS_DIR}`  \nMLflow: `{bc.MLFLOW_DIR}`")

# ------------------------------------------------------------------------------------------------ helpers
def log_single(key, res, ms, src):
    mlflow.set_experiment(experiment_id=bc.get_experiment_id(mlflow, bc.SERVE_EXPERIMENT))
    with mlflow.start_run(run_name=f"predict-{key}"):
        mlflow.log_params(dict(model=key, mode=res["mode"], predicted=(res["text"] or "-")[:250], decision=res["decision"], source=src))
        mlflow.log_metrics(dict(latency_ms=ms, confidence=res["confidence"]))


def log_compare(rows, img, src):
    mlflow.set_experiment(experiment_id=bc.get_experiment_id(mlflow, bc.SERVE_EXPERIMENT))
    with mlflow.start_run(run_name="compare-all"):
        mlflow.set_tag("kind", "comparison"); mlflow.log_param("source", src)
        for r in rows:
            mlflow.log_param(f"predicted_{r['key']}", (r["text"] or "-")[:250])
            mlflow.log_metrics({f"latency_ms_{r['key']}": r["ms"], f"confidence_{r['key']}": r["confidence"]})
        mlflow.log_dict(rows, "comparison.json")
        mlflow.log_image(cv2.resize(bc.ink(img), None, fx=.5, fy=.5) if max(img.shape[:2]) > 800 else bc.ink(img), "input.png")


def agreement(rows):
    texts = [r["text"] for r in rows]
    if len(set(texts)) == 1:
        return "All models give the same answer." if texts[0] else "No model could read this drawing."
    groups = {}
    for r in rows:
        groups.setdefault(r["text"] or "—", []).append(r["model"].split(" · ")[0])
    return "The models disagree: " + "; ".join(f"{t} ← {', '.join(ms)}" for t, ms in groups.items())


def big(text):
    st.markdown(f"<div style='font-size:60px;line-height:1.2'>{html.escape(text) if text else '—'}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------ tabs
tab_predict, tab_mlflow, tab_models = st.tabs(["Predict", "MLflow", "About the models"])

with tab_predict:
    st.title("✍️ Bangla handwriting recognizer")
    st.caption("Draw one word (for example মানুষ or জল), a number, or a single character. Draw large and in the middle.")
    if "canvas_n" not in st.session_state:
        st.session_state.canvas_n = 0
    col_in, col_out = st.columns([3, 2])
    image, expected = None, None
    compare = None
    with col_in:
        if source == "Draw on the canvas":
            canvas = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=stroke, stroke_color="#FFFFFF", background_color="#000000",
                               height=260, width=640, drawing_mode="freedraw", key=f"canvas_{st.session_state.canvas_n}")
            if canvas.image_data is not None:
                image = canvas.image_data.astype("uint8")
            if st.button("Clear canvas"):
                st.session_state.canvas_n += 1
                st.rerun()
        elif source == "Upload an image":
            up = st.file_uploader("PNG or JPG: dark ink on light paper, or light ink on dark", type=["png", "jpg", "jpeg"])
            if up is not None:
                image = np.array(Image.open(up).convert("RGBA"))
                st.image(image, caption="uploaded image", **IMG_FILL)
        else:
            samples = bc.list_samples()
            if not samples:
                st.info("No samples found in models/samples.")
            else:
                pick = st.selectbox("Sample (synthetic: real glyphs of unseen writers composed into a drawing)", samples, format_func=lambda s: f"{s[2]}: {s[1]}")
                image = cv2.imread(str(bc.MODELS_DIR / "samples" / pick[0]), cv2.IMREAD_GRAYSCALE)
                expected = pick[1]
                st.image(image, caption=f"sample, expected: {expected}", **IMG_FILL)
        go = st.button("Predict", type="primary", **FILL)

    with col_out:
        if not go:
            st.info("Draw on the left, choose a model in the sidebar, then press Predict.")
        elif image is None or bc.crop(bc.ink(image)) is None:
            st.warning("Nothing to read. Draw something first.")
        elif CHOICES[choice] != "all":
            key = CHOICES[choice]
            res, ms = bc.timed(models[key], image, mode, repeats)
            st.subheader(models[key].title)
            big(res["text"])
            st.caption(f"{res['decision']} · confidence {res['confidence']:.1%} · read as `{res['mode']}` ({res['glyphs']} glyph(s))")
            c1, c2 = st.columns(2)
            c1.metric("Time taken", f"{ms:.1f} ms", help=f"median of {int(repeats)} run(s): preprocessing, network and decoding")
            c2.metric("Decision", res["decision"], help="accept: confident; suggest: check the alternatives; reject: could not read")
            if res["top3"]:
                st.dataframe(pd.DataFrame([dict(rank=i + 1, text=t, probability=f"{p:.1%}") for i, (t, p) in enumerate(res["top3"])]),
                             hide_index=True, **FILL)
            if expected is not None:
                (st.success if res["text"] == expected else st.error)(f"Expected `{expected}`: {'correct' if res['text'] == expected else 'wrong'}")
            if log_runs:
                log_single(key, res, ms, source)
        else:
            rows = []
            progress = st.progress(0.0, text="Running the models one after the other...")
            for n, key in enumerate(k for k in bc.MODEL_KEYS if k in models):          # sequentially: Model 1, Model 2, Model 3
                res, ms = bc.timed(models[key], image, mode, repeats)
                rows.append(dict(key=key, model=models[key].title, text=res["text"], confidence=res["confidence"], decision=res["decision"], ms=ms,
                                 mode=res["mode"], glyphs=res["glyphs"], params_m=cfg[key]["params"] / 1e6, size_mb=cfg[key]["size_mb"]))
                progress.progress((n + 1) / len([k for k in bc.MODEL_KEYS if k in models]), text=f"{models[key].title}: {ms:.1f} ms")
            progress.empty()
            compare = (rows, expected)                                   # shown below, across the full page width

    if compare is not None:                                               # the three-model comparison: full width, so every column of the table is visible
        rows, expected = compare
        st.subheader("All models, same drawing")
        table = pd.DataFrame([{"Model": r["model"], "Prediction": r["text"] or "—", "Confidence": f"{r['confidence']:.1%}", "Decision": r["decision"],
                               "Time (ms)": round(r["ms"], 1), "Read as": f"{r['mode']} ({r['glyphs']} glyph)", "Params (M)": round(r["params_m"], 2),
                               "Size (MB)": round(r["size_mb"], 1)} | ({"Correct?": "✅" if r["text"] == expected else "❌"} if expected is not None else {}) for r in rows])
        st.dataframe(table, hide_index=True, **FILL)
        fastest, surest = min(rows, key=lambda r: r["ms"]), max(rows, key=lambda r: r["confidence"])
        st.markdown(f"**Fastest:** {fastest['model']} ({fastest['ms']:.1f} ms) · **Most confident:** {surest['model']} ({surest['confidence']:.1%}) · {agreement(rows)}")
        st.bar_chart(pd.DataFrame({"time (ms)": [r["ms"] for r in rows]}, index=[r["model"].split(" · ")[0] for r in rows]))
        st.caption("Models 1 and 2 read one glyph at a time: the drawing is cut into glyphs and they have no class for vowel signs, "
                   "so words with vowel signs (such as মানুষ) are expected to go wrong. Model 3 reads the whole drawing.")
        if log_runs:
            log_compare(rows, image, source)

with tab_mlflow:
    client = mlflow.MlflowClient()
    st.subheader("Training runs (one per model)")
    train_exp = mlflow.get_experiment_by_name(bc.TRAIN_EXPERIMENT)
    if train_exp is None:
        st.info("No training runs yet. Run `python train.py train --data ...`.")
    else:
        runs = mlflow.search_runs(experiment_ids=[train_exp.experiment_id], order_by=["start_time DESC"])
        if "tags.source" not in runs:
            runs["tags.source"] = None
        runs["tags.source"] = runs["tags.source"].fillna("train.py")           # runs recorded from a notebook are tagged source=notebook
        cols = {"tags.mlflow.runName": "run", "tags.source": "source", "params.epochs": "epochs",
                "metrics.test_char_top1": "glyph top-1 (test)", "metrics.test_char_top3": "glyph top-3 (test)", "metrics.test_number_exact": "numbers (test)",
                "metrics.test_pseudo_exact": "random words (test)", "metrics.sanity_char_top1": "glyph top-1 (sanity, not a test)",
                "metrics.notebook_test_char_top1": "glyph top-1 (notebook reported)", "metrics.model_size_mb": "size MB",
                "metrics.cpu_ms_in_notebook": "ms in notebook", "metrics.latency_ms_word": "ms / word image"}
        st.dataframe(runs[[c for c in cols if c in runs.columns]].rename(columns=cols), hide_index=True, **FILL)
        st.caption("source = notebook: the model was trained and tested in a Kaggle notebook; this project only recorded what that notebook printed and displayed "
                   "(each model on its own held-out writers). 'sanity' numbers are measured on this project's split, which that model may have seen, so they are not test results.")
        names = list(runs["tags.mlflow.runName"]) if "tags.mlflow.runName" in runs else []
        if names:
            pick = st.selectbox("Artifacts of run", names)
            rid = runs.loc[runs["tags.mlflow.runName"] == pick, "run_id"].iloc[0]
            pngs = sorted((x.path for x in client.list_artifacts(rid) if x.path.lower().endswith(".png")), key=lambda n: (n != "curves.png", n))
            for row in range(0, len(pngs), 3):                                   # every figure of the run: curves, confusion matrix, notebook figures ...
                for slot, art in zip(st.columns(3), pngs[row:row + 3]):
                    try:
                        slot.image(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path=art), caption=art, **IMG_FILL)
                    except Exception:
                        slot.info(f"{art} not available")
            try:
                st.markdown(open(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="model_card.md"), encoding="utf-8").read())
            except Exception:
                pass
            for art, title in (("training_log.txt", "Training log printed by the notebook"), ("notebook_training_log.txt", "Training log printed by the notebook"),
                               ("test_report.txt", "Test report printed by the notebook"), ("calibration.txt", "Calibration printed by the notebook")):
                try:
                    txt = open(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path=art), encoding="utf-8").read()
                    with st.expander(f"{title} ({art})"):
                        st.code(txt)
                except Exception:
                    pass
            st.caption("All artifacts of the run: " + ", ".join(a.path for a in client.list_artifacts(rid)))
    eval_exp = mlflow.get_experiment_by_name(bc.EVAL_EXPERIMENT)
    if eval_exp is not None:
        st.subheader("Comparison of the three models (latest test)")
        cmp_runs = mlflow.search_runs(experiment_ids=[eval_exp.experiment_id], filter_string="tags.mlflow.runName = 'compare-all'", order_by=["start_time DESC"], max_results=1)
        if len(cmp_runs):
            try:
                st.dataframe(pd.read_csv(mlflow.artifacts.download_artifacts(run_id=cmp_runs.iloc[0]["run_id"], artifact_path="comparison.csv")), hide_index=True, **FILL)
                st.image(mlflow.artifacts.download_artifacts(run_id=cmp_runs.iloc[0]["run_id"], artifact_path="comparison.png"), **IMG_FILL)
            except Exception:
                st.info("comparison artifacts not available")
    st.subheader("Predictions logged by this app")
    serve_exp = mlflow.get_experiment_by_name(bc.SERVE_EXPERIMENT)
    if serve_exp is None:
        st.info("No predictions yet. Use the Predict tab.")
    else:
        preds = mlflow.search_runs(experiment_ids=[serve_exp.experiment_id], order_by=["start_time DESC"], max_results=20)
        cols = ["start_time", "tags.mlflow.runName", "params.mode", "params.predicted", "params.decision", "metrics.latency_ms", "metrics.confidence"]
        st.dataframe(preds[[c for c in cols if c in preds.columns]], hide_index=True, **FILL)
    st.caption(f"Full MLflow UI: `mlflow ui --backend-store-uri {bc.TRACKING_URI}` or start the app with START_MLFLOW_UI=1 (http://localhost:5000).")

with tab_models:
    st.subheader("The three models")
    st.dataframe(pd.DataFrame([{"Model": m.title, "Reads": bc.MODEL_ROLES[k], "Params (M)": round(cfg[k]["params"] / 1e6, 2), "Size (MB)": round(cfg[k]["size_mb"], 1),
                                "Temperature": cfg[k]["T"]} for k, m in models.items()]), hide_index=True, **FILL)
    st.markdown("""
**How a drawing becomes text**
- *Model 3* resizes the drawing to height 64 and reads it in one pass with a CNN + BiGRU + CTC network, then decodes the sequence
  (words are matched against a word list as a soft preference, numbers use digits only).
- *Models 1 and 2* read one 64×64 glyph. For a word or a number the drawing is cut into glyphs first (the joining headline is removed, strokes are grouped).

**Limits you should know**
- The dataset has 84 isolated glyphs and no vowel signs or word images. Words are built from real glyphs; vowel signs are drawn procedurally.
  Reading real handwritten words has not been measured.
- Glyph accuracy on writers never seen in training is about 93% top-1 for the best model (see the MLflow tab for this installation's numbers).
""")
