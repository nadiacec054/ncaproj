
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image

MODEL_PATH = Path("models/symbol_classifier.joblib")


def load_model():
    """Load the trained SVM model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the model first."
        )
    return joblib.load(MODEL_PATH)


def extract_features(image_path):
    """Convert an image into the feature format expected by the model."""
    with Image.open(image_path) as image:
        image = image.convert("L")
        pixels = np.asarray(image, dtype=np.float32) / 255.0

    return pixels.flatten().reshape(1, -1)


def predict_symbol(image_path, model=None, top_k=3):
    """
    Predict a symbol and return the top alternatives.

    Decision scores are relative SVM scores, not calibrated probabilities.
    """
    if model is None:
        model = load_model()

    features = extract_features(image_path)
    start_time = time.perf_counter()

    prediction = model.predict(features)[0]

    result = {
        "image": str(image_path),
        "prediction": str(prediction),
        "alternatives": [],
        "inference_time_ms": 0.0,
    }

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features))

        if scores.ndim == 1:
            # Binary SVM: convert its single margin into two ranking scores.
            scores = np.array([[-scores[0], scores[0]]])

        scores = scores.reshape(-1)
        classes = np.asarray(model.classes_)

        if len(scores) == len(classes):
            # Convert scores into relative ranking weights for display.
            # These are NOT calibrated probabilities.
            shifted = scores - np.max(scores)
            weights = np.exp(np.clip(shifted, -50, 0))
            relative_scores = weights / weights.sum()

            order = np.argsort(scores)[::-1][:max(1, top_k)]

            result["alternatives"] = [
                {
                    "symbol": str(classes[i]),
                    "relative_score_percent": round(
                        float(relative_scores[i] * 100), 2
                    ),
                }
                for i in order
            ]

    result["inference_time_ms"] = round(
        (time.perf_counter() - start_time) * 1000, 3
    )

    return result


def predict_batch(image_paths, model=None, top_k=3):
    """Predict symbols in multiple image files using one loaded model."""
    if model is None:
        model = load_model()

    results = []

    for image_path in image_paths:
        try:
            results.append(
                predict_symbol(image_path, model=model, top_k=top_k)
            )
        except Exception as error:
            results.append({
                "image": str(image_path),
                "error": str(error),
            })

    return results


if __name__ == "__main__":
    model = load_model()
    image_path = input("Enter the path to a symbol image: ").strip()

    try:
        result = predict_symbol(image_path, model=model)

        print(f"\nPredicted symbol: {result['prediction']}")
        print("\nTop predictions:")

        for item in result["alternatives"]:
            print(
                f"  {item['symbol']}: "
                f"{item['relative_score_percent']}% relative score"
            )

        print(f"\nInference time: {result['inference_time_ms']} ms")

    except Exception as error:
        print(f"Prediction failed: {error}")
