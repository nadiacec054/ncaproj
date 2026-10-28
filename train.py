
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
    f1_score,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CSV_PATH = Path("/Users/nadiacecil/Downloads/HASYv2/hasy-data-labels.csv")
DATA_DIR = Path("/Users/nadiacecil/Downloads/HASYv2")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "symbol_classifier_cnn.keras"
CLASSES_PATH = MODEL_DIR / "symbol_classes.json"
REPORT_PATH = MODEL_DIR / "cnn_evaluation.json"

IMAGE_SIZE = 32
MIN_SAMPLES = 20
RANDOM_STATE = 42
EPOCHS = 25
BATCH_SIZE = 64

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# --------------------------------------------------
# SYMBOLS TO RECOGNIZE
# --------------------------------------------------

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


# --------------------------------------------------
# IMAGE PREPROCESSING
# Keep this identical in predict.py later.
# --------------------------------------------------


def preprocess_cnn_image(image):
    """Crop a symbol and centre it on a 32x32 white canvas."""
    image = image.convert("L")

    # HASYv2 and the drawing canvas use dark symbols on a light background.
    image_array = np.asarray(image)
    mask = image_array < 200

    if not mask.any():
        raise ValueError("Image contains no visible symbol.")

    rows, cols = np.where(mask)
    left, right = cols.min(), cols.max() + 1
    top, bottom = rows.min(), rows.max() + 1

    image = image.crop((left, top, right, bottom))
    image.thumbnail((24, 24), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (32, 32), color=255)
    x = (32 - image.width) // 2
    y = (32 - image.height) // 2
    canvas.paste(image, (x, y))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return array[..., np.newaxis]

# --------------------------------------------------
# LOAD AND FILTER DATASET
# --------------------------------------------------

print("\nLoading HASYv2 labels...")

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

df = pd.read_csv(CSV_PATH)
df["latex"] = df["latex"].astype(str)

available_labels = set(df["latex"].unique())
selected_labels = set()

print("\nChecking requested symbols:\n")

for category, candidates in requested_groups.items():
    found = [label for label in candidates if label in available_labels]
    missing = [label for label in candidates if label not in available_labels]

    selected_labels.update(found)

    print(f"{category}:")
    print("  Available:", ", ".join(found) if found else "None")

    if missing:
        print("  Not found:", ", ".join(missing))

df = df[df["latex"].isin(selected_labels)].copy()

counts = df["latex"].value_counts()
rare_labels = counts[counts < MIN_SAMPLES].index.tolist()

if rare_labels:
    print(f"\nExcluding classes with fewer than {MIN_SAMPLES} images:")
    print(", ".join(rare_labels))

df = df[~df["latex"].isin(rare_labels)].copy()

# Remove repeated paths to prevent the same image appearing twice.
df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)

if df.empty or df["latex"].nunique() < 3:
    raise ValueError(
        "At least three eligible symbol classes are needed "
        "for this training and top-three evaluation setup."
    )

print("\nNumber of classes:", df["latex"].nunique())
print("Total unique images:", len(df))
print("\nImages per class:")
print(df["latex"].value_counts().sort_index().to_string())


# --------------------------------------------------
# LOAD IMAGES
# --------------------------------------------------

print("\nLoading and preprocessing images...")

images = []
labels = []
valid_paths = []
failed_images = []

for row in df.itertuples(index=False):
    image_path = DATA_DIR / row.path

    try:
        with Image.open(image_path) as image:
            processed = preprocess_cnn_image(image)

        images.append(processed)
        labels.append(row.latex)
        valid_paths.append(str(image_path))

    except Exception as exc:
        failed_images.append((str(image_path), str(exc)))

if failed_images:
    print(f"\nSkipped {len(failed_images)} unreadable images.")
    for path, error in failed_images[:5]:
        print(f"  {path}: {error}")

X = np.asarray(images, dtype=np.float32)
labels = np.asarray(labels)

if len(X) == 0:
    raise ValueError("No images could be loaded.")

print("Image tensor shape:", X.shape)

# Encode string labels into integer class IDs.
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
class_names = encoder.classes_.tolist()

if len(class_names) < 3:
    raise ValueError("Fewer than three classes remain after loading images.")


# --------------------------------------------------
# TRAIN / VALIDATION / TEST SPLIT
# 70% train, 15% validation, 15% test
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print("\nDataset split:")
print("  Training:", len(X_train))
print("  Validation:", len(X_val))
print("  Test:", len(X_test))


# --------------------------------------------------
# CLASS WEIGHTS FOR IMBALANCED DATA
# --------------------------------------------------

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(np.unique(y_train), weights)
}


# --------------------------------------------------
# CNN ARCHITECTURE
# --------------------------------------------------

# CNN ARCHITECTURE

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08,
        fill_mode="constant",
        fill_value=1.0,
    ),
    tf.keras.layers.RandomRotation(
        factor=0.03,
        fill_mode="constant",
        fill_value=1.0,
    ),
    tf.keras.layers.RandomZoom(
        height_factor=(-0.08, 0.08),
        width_factor=(-0.08, 0.08),
        fill_mode="constant",
        fill_value=1.0,
    ),
])

model = tf.keras.Sequential([
    augmentation,
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

    tf.keras.layers.Conv2D(
        32, kernel_size=3, padding="same", activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(
        64, kernel_size=3, padding="same", activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(
        128, kernel_size=3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.40),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax",
    ),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1))
model.summary()

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
    ),
]

print("\nTraining CNN...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

# Load the best saved checkpoint, not necessarily the last epoch.
model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# FINAL TEST EVALUATION
# --------------------------------------------------

print("\nEvaluating on held-out test set...")

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1,
)

predictions = np.argmax(probabilities, axis=1)

accuracy = float(accuracy_score(y_test, predictions))
macro_f1 = float(f1_score(y_test, predictions, average="macro"))

top3_accuracy = float(
    top_k_accuracy_score(
        y_test,
        probabilities,
        k=3,
        labels=np.arange(len(class_names)),
    )
)

print(f"\nTest accuracy: {accuracy:.4f} ({accuracy:.2%})")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Top-3 accuracy: {top3_accuracy:.4f} ({top3_accuracy:.2%})")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(class_names)),
)

# --------------------------------------------------
# SAVE CLASS LABELS AND EVALUATION RESULTS
# --------------------------------------------------

with open(CLASSES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2, ensure_ascii=False)

report = {
    "model": "CNN",
    "image_size": IMAGE_SIZE,
    "class_count": len(class_names),
    "class_names": class_names,
    "dataset_images": int(len(X)),
    "train_images": int(len(X_train)),
    "validation_images": int(len(X_val)),
    "test_images": int(len(X_test)),
    "test_accuracy": accuracy,
    "macro_f1": macro_f1,
    "top3_accuracy": top3_accuracy,
    "classification_report": classification_report(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    ),
    "confusion_matrix": matrix.tolist(),
}

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2, ensure_ascii=False)

print("\nCNN saved to:", MODEL_PATH)
print("Class labels saved to:", CLASSES_PATH)
print("Evaluation report saved to:", REPORT_PATH)
print("\nTrained classes:", class_names)
print("\nTraining complete.")
