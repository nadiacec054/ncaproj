
import time
from pathlib import Path

import joblib
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from preprocessing import preprocess_image
from calculator import solve_expression


MODEL_PATH = Path("models/symbol_classifier.joblib")
CNN_MODEL_PATH = Path("models/symbol_classifier_cnn.keras")
CNN_CLASSES_PATH = Path("models/symbol_classes.json")
@st.cache_resource
def load_cnn_model():
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found: {CNN_MODEL_PATH}"
        )

    return tf.keras.models.load_model(CNN_MODEL_PATH)


@st.cache_data
def load_cnn_classes():
    with open(CNN_CLASSES_PATH, "r") as file:
        return json.load(file)


try:
    cnn_model = load_cnn_model()
    cnn_classes = load_cnn_classes()
except Exception as error:
    st.error(f"Could not load the CNN model: {error}")
    st.stop()
DEFAULT_DATA_DIR = "/Users/nadiacecil/Downloads/HASYv2"

st.set_page_config(
    page_title="Math Symbol Classifier",
    page_icon="➗",
    layout="wide",
)


# -------------------- DARK THEME --------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #10131b;
        color: #f1f5f9;
    }

    [data-testid="stHeader"] {
        background-color: #10131b;
    }

    [data-testid="stSidebar"] {
        background-color: #171c28;
    }

    [data-testid="stMetric"] {
        background-color: #1b2230;
        border: 1px solid #30394b;
        padding: 16px;
        border-radius: 12px;
    }

    div.stButton > button {
        background-color: #6478f5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1rem;
    }

    div.stButton > button:hover {
        background-color: #8290ff;
        color: white;
    }

    [data-testid="stTabs"] button {
        color: #cbd5e1;
    }

    [data-testid="stFileUploader"] {
        background-color: #171c28;
        border-radius: 10px;
    }

    div[data-baseweb="input"],
    div[data-baseweb="textarea"] {
        background-color: #171c28;
    }

    hr {
        border-color: #30394b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- MODEL --------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


DISPLAY_SYMBOLS = {
    r"\times": "×",
    r"\div": "÷",
    r"\ast": "*",
    r"\neq": "≠",
    r"\pi": "π",
    r"\sum": "∑",
}


def display_symbol(symbol):
    symbol = str(symbol).strip()
    return DISPLAY_SYMBOLS.get(symbol, symbol)




def get_cnn_predictions(image, top_k=3):
    """Crop and centre a handwritten symbol before CNN prediction."""
    image = image.convert("L")

    # Find the dark pixels that make up the symbol.
    image_array = np.asarray(image)
    mask = image_array < 200

    if not mask.any():
        raise ValueError("Please draw a symbol first.")

    rows, cols = np.where(mask)
    left, right = cols.min(), cols.max() + 1
    top, bottom = rows.min(), rows.max() + 1

    # Crop to the symbol.
    image = image.crop((left, top, right, bottom))

    # Preserve proportions and centre the symbol on a white square.
    image.thumbnail((24, 24), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (32, 32), color=255)
    x = (32 - image.width) // 2
    y = (32 - image.height) // 2
    canvas.paste(image, (x, y))

    # Match the CNN's expected input shape and pixel range.
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = array[np.newaxis, ..., np.newaxis]

    start_time = time.perf_counter()
    probabilities = cnn_model.predict(array, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    top_indices = np.argsort(probabilities)[::-1][:top_k]

    alternatives = [
        {
            "symbol": cnn_classes[int(i)],
            "display": display_symbol(cnn_classes[int(i)]),
            "score": float(probabilities[i] * 100),
        }
        for i in top_indices
    ]

    predicted_label = cnn_classes[int(np.argmax(probabilities))]

    return {
        "prediction": predicted_label,
        "display": display_symbol(predicted_label),
        "alternatives": alternatives,
        "inference_ms": elapsed_ms,
    }

def get_features(image):
    """Use the same preprocessing function as the trained model."""
    features = np.asarray(
        preprocess_image(image),
        dtype=np.float32,
    )
    return features.reshape(1, -1)


def get_top_predictions(image, top_k=3):
    """
    Return the model prediction and its top alternatives.

    The displayed scores are relative scores derived from SVM decision
    values. They are not calibrated probabilities.
    """
    features = get_features(image)

    start_time = time.perf_counter()
    predicted_label = model.predict(features)[0]
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    alternatives = []

    if hasattr(model, "decision_function"):
        decision_scores = np.asarray(
            model.decision_function(features)
        )

        classes = np.asarray(model.classes_)

        if decision_scores.ndim == 1:
            # Binary classification.
            decision_scores = np.array(
                [-decision_scores[0], decision_scores[0]]
            ).reshape(1, -1)

        scores = decision_scores.reshape(-1)

        if len(scores) == len(classes):
            shifted_scores = scores - np.max(scores)
            weights = np.exp(
                np.clip(shifted_scores, -50, 0)
            )
            relative_scores = weights / weights.sum()

            ranking = np.argsort(scores)[::-1][:top_k]

            alternatives = [
                {
                    "symbol": str(classes[index]),
                    "display": display_symbol(classes[index]),
                    "score": float(relative_scores[index] * 100),
                }
                for index in ranking
            ]

    # Always include a prediction even if decision scores are unavailable.
    if not alternatives:
        alternatives = [{
            "symbol": str(predicted_label),
            "display": display_symbol(predicted_label),
            "score": None,
        }]

    return {
        "prediction": str(predicted_label),
        "display": display_symbol(predicted_label),
        "alternatives": alternatives,
        "inference_ms": elapsed_ms,
    }


def show_prediction(result):
    st.success(f"Predicted symbol: {result['display']}")

    st.markdown("#### Top predictions")

    for item in result["alternatives"]:
        if item["score"] is None:
            st.write(item["display"])
        else:
            st.write(
                f"**{item['display']}** — "
                f"{item['score']:.2f}% relative score"
            )
            st.progress(
                min(1.0, max(0.0, item["score"] / 100))
            )

    st.caption(
        "Relative scores are based on SVM decision values. "
        "They are not calibrated probabilities of correctness."
    )
    st.caption(
        f"Inference time: {result['inference_ms']:.3f} ms "
        "(model prediction only)"
    )


# -------------------- HEADER --------------------

st.title("Math Symbol Classifier")
st.write(
    "Recognise handwritten mathematical symbols, draw expressions, "
    "evaluate calculations, and evaluate the classifier."
)

st.caption("Machine learning • Image processing • Mathematical evaluation")
model_choice = st.radio(
    "Choose classifier",
    ["CNN", "SVM"],
    horizontal=True,
)

predict_symbol = (
    get_cnn_predictions
    if model_choice == "CNN"
    else get_top_predictions
)

upload_tab, draw_tab, expression_tab, type_tab, evaluation_tab = (
    st.tabs([
        "Upload Image",
        "Draw Symbol",
        "Draw Expression",
        "Calculator",
        "Model Evaluation",
    ])
)


# -------------------- UPLOAD IMAGE --------------------

with upload_tab:
    st.subheader("Recognise a handwritten symbol")

    uploaded_file = st.file_uploader(
        "Upload an image of one symbol",
        type=["png", "jpg", "jpeg"],
        key="uploaded_symbol",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            width=300,
        )

        if st.button(
            "Recognise Symbol",
            key="upload_predict",
        ):
            try:
                result = predict_symbol(image)
                show_prediction(result)
            except Exception as error:
                st.error(f"Prediction failed: {error}")


# -------------------- DRAW ONE SYMBOL --------------------

with draw_tab:
    st.subheader("Draw a single symbol")
    st.write("Draw one digit or mathematical symbol in the box.")

    canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=8,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=250,
        width=400,
        drawing_mode="freedraw",
        key="single_symbol_canvas",
    )

    if st.button(
        "Recognise Symbol",
        key="draw_predict",
    ):
        if canvas.image_data is None:
            st.warning("Please draw a symbol first.")
        else:
            try:
                image = Image.fromarray(
                    canvas.image_data.astype("uint8")
                ).convert("RGB")

                if np.asarray(image).mean() > 254.5:
                    st.warning("Please draw a symbol first.")
                else:
                    result = predict_symbol(image)
                    show_prediction(result)
            except Exception as error:
                st.error(f"Prediction failed: {error}")


# -------------------- DRAW AN EXPRESSION --------------------

with expression_tab:
    st.subheader("Recognise and calculate an expression")

    st.write(
        "Draw an expression from left to right. Leave gaps between "
        "symbols and keep the expression on one line."
    )

    expression_canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=6,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=250,
        width=650,
        drawing_mode="freedraw",
        key="expression_canvas_v2",
    )

    if st.button(
        "Recognise and Calculate",
        key="expression_predict_v2",
    ):
        if expression_canvas.image_data is None:
            st.warning("Please draw an expression first.")
        else:
            image = Image.fromarray(
                expression_canvas.image_data.astype("uint8")
            ).convert("RGB")

            pixels = np.asarray(image.convert("L"))
            mask = pixels < 200

            if not mask.any():
                st.warning("Please draw an expression first.")
            else:
                active_columns = np.where(mask.any(axis=0))[0]

                regions = []
                start = previous = active_columns[0]

                for column in active_columns[1:]:
                    if column > previous + 1:
                        regions.append((start, previous + 1))
                        start = column
                    previous = column

                regions.append((start, previous + 1))

                # Merge regions separated by very small horizontal gaps.
                merged_regions = []

                for region in regions:
                    if (
                        merged_regions
                        and region[0] - merged_regions[-1][1] <= 3
                    ):
                        merged_regions[-1] = (
                            merged_regions[-1][0],
                            region[1],
                        )
                    else:
                        merged_regions.append(region)

                recognised_symbols = []
                segmented_preview = image.copy()
                draw = ImageDraw.Draw(segmented_preview)

                for index, (x_start, x_end) in enumerate(
                    merged_regions, start=1
                ):
                    region_mask = mask[:, x_start:x_end]
                    active_rows = np.where(
                        region_mask.any(axis=1)
                    )[0]

                    if len(active_rows) == 0:
                        continue

                    y_start = int(active_rows[0])
                    y_end = int(active_rows[-1] + 1)

                    # Add a little padding around each segmented symbol.
                    padding = 5
                    left = max(0, x_start - padding)
                    right = min(image.width, x_end + padding)
                    top = max(0, y_start - padding)
                    bottom = min(image.height, y_end + padding)

                    symbol_image = image.crop(
                        (left, top, right, bottom)
                    )

                    try:
                        result = predict_symbol(symbol_image)
                        recognised_symbols.append(
                            display_symbol(result["prediction"])
                        )

                        draw.rectangle(
                            (left, top, right - 1, bottom - 1),
                            outline="#E53935",
                            width=2,
                        )
                        draw.text(
                            (left, max(0, top - 15)),
                            str(index),
                            fill="#E53935",
                        )

                    except Exception as error:
                        st.warning(
                            f"Could not classify region {index}: {error}"
                        )

                st.markdown("#### Segmentation preview")
                st.image(
                    segmented_preview,
                    caption="Red boxes show detected symbol regions",
                    use_container_width=True,
                )

                recognised_expression = "".join(
                    recognised_symbols
                )

                st.markdown("#### Recognition result")
                st.code(
                    recognised_expression or "No symbols recognised",
                    language=None,
                )

                if recognised_expression:
                    try:
                        answer = solve_expression(
                            recognised_expression
                        )
                        st.success(f"Calculated answer: {answer}")
                    except (ValueError, TypeError, SyntaxError) as error:
                        st.error(
                            "The recognised expression could not be "
                            f"evaluated: {error}"
                        )

                st.caption(
                    "Segmentation uses horizontal gaps between marks. "
                    "Touching symbols, stacked fractions, and multi-level "
                    "expressions may require more advanced segmentation."
                )


# -------------------- TYPED CALCULATOR --------------------

with type_tab:
    st.subheader("Mathematical calculator")
    st.write("Type an expression and calculate its answer.")

    expression = st.text_input(
        "Enter an expression",
        placeholder="Example: 5 * 4 + 2",
        key="typed_expression",
    )

    if st.button("Calculate", key="calculate_expression"):
        try:
            answer = solve_expression(expression)
            st.success(f"Answer: {answer}")
        except (ValueError, TypeError, SyntaxError) as error:
            st.error(str(error))


# -------------------- MODEL EVALUATION --------------------

with evaluation_tab:
    st.subheader("Evaluate the trained classifier")

    st.write(
        "Upload a labelled CSV containing image paths and true symbol "
        "labels. This evaluates the existing model on those images."
    )

    st.info(
        "For a valid test evaluation, use held-out images that were not "
        "used to train the model. Evaluating on training images will "
        "overestimate performance."
    )

    csv_file = st.file_uploader(
        "Upload labelled test CSV",
        type=["csv"],
        key="evaluation_csv",
    )

    data_dir = st.text_input(
        "Dataset root directory",
        value=DEFAULT_DATA_DIR,
        key="evaluation_data_dir",
    )

    if csv_file is not None:
        try:
            evaluation_df = pd.read_csv(csv_file)

            st.write("CSV columns:", ", ".join(evaluation_df.columns))

            if not {"path", "latex"}.issubset(evaluation_df.columns):
                st.error(
                    "The CSV must contain columns named 'path' and "
                    "'latex'. The HASYv2 labels CSV uses these columns."
                )
            elif st.button(
                "Run Evaluation",
                key="run_evaluation",
            ):
                y_true = []
                y_pred = []
                top3_correct = []
                inference_times = []
                errors = []

                progress = st.progress(0)

                for index, row in evaluation_df.iterrows():
                    image_path = Path(data_dir) / str(row["path"])

                    try:
                        with Image.open(image_path) as opened:
                            image = opened.convert("RGB")

                        result = get_top_predictions(image)

                        y_true.append(str(row["latex"]))
                        y_pred.append(result["prediction"])
                        inference_times.append(result["inference_ms"])

                        top_symbols = [
                            item["symbol"]
                            for item in result["alternatives"]
                        ]
                        top3_correct.append(
                            str(row["latex"]) in top_symbols
                        )

                    except Exception as error:
                        errors.append(
                            f"{image_path}: {error}"
                        )

                    progress.progress(
                        (index + 1) / len(evaluation_df)
                    )

                if not y_true:
                    st.error(
                        "No images could be evaluated. Check the dataset "
                        "directory and paths in the CSV."
                    )
                else:
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(
                        y_true, y_pred, average="macro",
                        zero_division=0,
                    )
                    recall = recall_score(
                        y_true, y_pred, average="macro",
                        zero_division=0,
                    )
                    macro_f1 = f1_score(
                        y_true, y_pred, average="macro",
                        zero_division=0,
                    )
                    top3_accuracy = float(np.mean(top3_correct))

                    st.markdown("#### Evaluation metrics")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{accuracy:.2%}")
                    c2.metric("Macro precision", f"{precision:.2%}")
                    c3.metric("Macro recall", f"{recall:.2%}")

                    c4, c5, c6 = st.columns(3)
                    c4.metric("Macro F1", f"{macro_f1:.2%}")
                    c5.metric("Top-3 accuracy", f"{top3_accuracy:.2%}")
                    c6.metric(
                        "Mean inference",
                        f"{np.mean(inference_times):.2f} ms",
                    )

                    labels = sorted(
                        set(y_true) | set(y_pred)
                    )

                    st.markdown("#### Classification report")

                    report = classification_report(
                        y_true,
                        y_pred,
                        labels=labels,
                        target_names=[
                            display_symbol(label)
                            for label in labels
                        ],
                        output_dict=True,
                        zero_division=0,
                    )

                    st.dataframe(
                        pd.DataFrame(report).transpose().round(4),
                        use_container_width=True,
                    )

                    st.markdown("#### Confusion matrix")

                    matrix = confusion_matrix(
                        y_true,
                        y_pred,
                        labels=labels,
                    )

                    matrix_df = pd.DataFrame(
                        matrix,
                        index=[
                            display_symbol(label)
                            for label in labels
                        ],
                        columns=[
                            display_symbol(label)
                            for label in labels
                        ],
                    )

                    st.dataframe(
                        matrix_df,
                        use_container_width=True,
                    )

                    if errors:
                        st.warning(
                            f"{len(errors)} image(s) could not be "
                            "evaluated. See details below."
                        )
                        with st.expander("Evaluation errors"):
                            for message in errors[:30]:
                                st.write(message)

                    st.download_button(
                        "Download classification report",
                        data=pd.DataFrame(report).transpose().to_csv(),
                        file_name="classification_report.csv",
                        mime="text/csv",
                    )

        except Exception as error:
            st.error(f"Could not read the evaluation CSV: {error}")
