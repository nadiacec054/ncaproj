
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

MODEL_PATH = Path("models/symbol_classifier.joblib")


def predict_symbol(image_path):
    model = joblib.load(MODEL_PATH)

    with Image.open(image_path) as image:
        image = image.convert("L")
        pixels = np.array(image, dtype=np.float32) / 255.0

    features = pixels.flatten().reshape(1, -1)
    prediction = model.predict(features)[0]

    return prediction


if __name__ == "__main__":
    image_path = input("Enter the path to a symbol image: ").strip()
    predict_symbol(image_path)
