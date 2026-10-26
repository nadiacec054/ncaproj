
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_image

CSV_PATH = "/Users/nadiacecil/Downloads/HASYv2/hasy-data-labels.csv"
DATA_DIR = Path("/Users/nadiacecil/Downloads/HASYv2")
MODEL_PATH = Path("models/symbol_classifier.joblib")

MIN_SAMPLES = 20

# Requested symbols and their possible HASYv2 labels.
requested_groups = {
    "Digits": [str(n) for n in range(10)],
    "Arithmetic operators": [
        "+", "-", r"\times", r"\div", r"\ast", "*", "/", "%"
    ],
    "Relational operators": [
        "<", ">", "=", r"\neq", "≠"
    ],
    "Special symbols": [
        r"\sum", r"\pi"
    ],
}

df = pd.read_csv(CSV_PATH)
df["latex"] = df["latex"].astype(str)
available_labels = set(df["latex"].unique())

selected_labels = set()

print("\nChecking requested symbols...\n")

for category, candidates in requested_groups.items():
    found = [label for label in candidates if label in available_labels]
    missing = [label for label in candidates if label not in available_labels]

    selected_labels.update(found)

    print(f"{category}:")
    print("  Available:", ", ".join(found) if found else "None")
    if missing:
        print("  Not found:", ", ".join(missing))

df = df[df["latex"].isin(selected_labels)].copy()

# Remove classes with too few examples.
counts = df["latex"].value_counts()
rare_labels = counts[counts < MIN_SAMPLES].index.tolist()

if rare_labels:
    print("\nExcluded classes with fewer than", MIN_SAMPLES, "images:")
    print(", ".join(rare_labels))

df = df[~df["latex"].isin(rare_labels)].copy()

if df.empty or df["latex"].nunique() < 2:
    raise ValueError(
        "Not enough eligible classes. Check dataset labels and sample counts."
    )

print("\nFinal number of classes:", df["latex"].nunique())
print("Total images:", len(df))
print("\nImages per class:")
print(df["latex"].value_counts().sort_index().to_string())


images = []
labels = []

for _, row in df.iterrows():
    image_path = DATA_DIR / row["path"]

    with Image.open(image_path) as image:
        features = preprocess_image(image)

    images.append(features)
    labels.append(row["latex"])


X = np.array(images)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = make_pipeline(
    StandardScaler(),
    SVC(class_weight="balanced", random_state=42),
)

print("\nTraining model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nTest accuracy:", accuracy_score(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nUpdated model saved to {MODEL_PATH}")
print("Number of trained classes:", len(model.classes_))
print("Exact trained labels:", list(model.classes_))
