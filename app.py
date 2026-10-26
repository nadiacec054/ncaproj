
import streamlit as st
import numpy as np
import joblib

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from preprocessing import preprocess_image
from calculator import solve_expression


st.set_page_config(
    page_title="Math Symbol Classifier",
    page_icon="➗",
    layout="wide",
)

st.title("Math Symbol Classifier")
st.write(
    "Recognise handwritten symbols, draw expressions, "
    "or type an expression to calculate its answer."
)


@st.cache_resource
def load_model():
    return joblib.load("models/symbol_classifier.joblib")


model = load_model()

DISPLAY_SYMBOLS = {
    r"\times": "×",
    r"\div": "÷",
    r"\ast": "*",
    r"\neq": "≠",
    r"\pi": "π",
    r"\sum": "∑",
}


def display_symbol(prediction):
    prediction = str(prediction).strip()
    return DISPLAY_SYMBOLS.get(prediction, prediction)


def predict_symbol(image):
    features = preprocess_image(image).reshape(1, -1)
    prediction = model.predict(features)[0]
    return display_symbol(prediction)


upload_tab, draw_tab, expression_tab, type_tab = st.tabs(
    [
        "Upload Image",
        "Draw Symbol",
        "Draw Expression",
        "Type Expression",
    ]
)


with upload_tab:
    st.subheader("Upload a handwritten symbol")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        key="uploaded_symbol",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")

        st.image(
            image,
            caption="Uploaded image",
            width=250,
        )

        if st.button("Recognise Symbol", key="upload_predict"):
            result = predict_symbol(image)
            st.success(f"Predicted symbol: {result}")


with draw_tab:
    st.subheader("Draw a single symbol")
    st.write("Draw one digit or mathematical symbol.")

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

    if st.button("Recognise Symbol", key="draw_predict"):
        if canvas.image_data is None:
            st.warning("Please draw a symbol first.")
        else:
            image = Image.fromarray(
                canvas.image_data.astype("uint8")
            ).convert("L")

            result = predict_symbol(image)
            st.success(f"Predicted symbol: {result}")


with expression_tab:
    st.subheader("Draw a mathematical expression")

    st.write(
        "Draw an expression from left to right. Leave clear gaps "
        "between symbols and keep everything on one line."
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
            ).convert("L")

            pixels = np.array(image)
            mask = pixels < 200

            if not mask.any():
                st.warning("Please draw an expression first.")
            else:
                active_columns = np.where(mask.any(axis=0))[0]

                regions = []
                start = active_columns[0]
                previous = active_columns[0]

                for column in active_columns[1:]:
                    if column > previous + 1:
                        regions.append((start, previous + 1))
                        start = column

                    previous = column

                regions.append((start, previous + 1))

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

                for x_start, x_end in merged_regions:
                    region_mask = mask[:, x_start:x_end]
                    active_rows = np.where(
                        region_mask.any(axis=1)
                    )[0]

                    if len(active_rows) == 0:
                        continue

                    y_start = active_rows[0]
                    y_end = active_rows[-1] + 1

                    symbol_image = image.crop(
                        (x_start, y_start, x_end, y_end)
                    )

                    prediction = predict_symbol(symbol_image)
                    recognised_symbols.append(prediction)

                recognised_expression = "".join(
                    recognised_symbols
                )

                st.write(
                    "Recognised expression:",
                    recognised_expression,
                )

                try:
                    answer = solve_expression(
                        recognised_expression
                    )
                    st.success(f"Answer: {answer}")
                except (ValueError, TypeError, SyntaxError) as error:
                    st.error(
                        f"Could not evaluate the recognised "
                        f"expression: {error}"
                    )


with type_tab:
    st.subheader("Mathematical Calculator")
    st.write("Type an expression and calculate the answer.")

    expression = st.text_input(
        "Enter an expression",
        placeholder="Example: 5 × 4 + 2",
        key="typed_expression",
    )

    if st.button("Calculate", key="calculate_expression"):
        try:
            answer = solve_expression(expression)
            st.success(f"Answer: {answer}")
        except (ValueError, TypeError, SyntaxError) as error:
            st.error(str(error))
