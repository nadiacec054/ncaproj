
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_image

DATA_DIR = Path("/Users/nadiacecil/Downloads/HASYv2")
CSV_PATH = DATA_DIR / "hasy-data-labels.csv"
MODEL_PATH = Path("models/symbol_classifier.joblib")
OUTPUT_DIR = Path("evaluation_results")


def main():
    print("Loading SVM model...")
    model = joblib.load(MODEL_PATH)

    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    df["latex"] = df["latex"].astype(str)

    supported_classes = {str(c) for c in model.classes_}
    df = df[df["latex"].isin(supported_classes)]
    df = df.drop_duplicates(subset=["path"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["latex"],
    )

    print(f"Evaluation images selected: {len(test_df)}")
    features, labels, paths = [], [], []

    for row in test_df.itertuples(index=False):
        try:
            with Image.open(DATA_DIR / row.path) as image:
                features.append(preprocess_image(image))
            labels.append(row.latex)
            paths.append(row.path)
        except Exception as exc:
            print(f"Skipping {row.path}: {exc}")

    if not features:
        raise RuntimeError("No test images could be loaded.")

    X_test = np.asarray(features)
    y_test = np.asarray(labels)

    print("Running predictions...")
    start = time.perf_counter()
    predictions = model.predict(X_test)
    elapsed = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "model": "SVM",
        "evaluation_images": len(y_test),
        "classes": len(model.classes_),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "average_inference_ms": float(
            elapsed * 1000 / len(y_test)
        ),
        "note": (
            "Preliminary results: the original training split "
            "has not been verified, so test-set overlap is possible."
        ),
    }

    with open(OUTPUT_DIR / "metrics.json", "w") as file:
        json.dump(results, file, indent=2)

    pd.DataFrame({
        "image_path": paths,
        "actual": y_test,
        "predicted": predictions,
        "correct": y_test == predictions,
    }).to_csv(
        OUTPUT_DIR / "per_image_results.csv", index=False
    )

    print("\n--- SVM Evaluation ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Average inference time: {elapsed * 1000 / len(y_test):.3f} ms")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")
    print("\nThese results are preliminary until training/test overlap is checked.")


if __name__ == "__main__":
    main()
