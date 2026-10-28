# Handwritten Mathematical Symbol Classifier

A Python project that recognises handwritten mathematical symbols using machine learning and deep learning. It includes an interactive Streamlit application where users can upload an image, draw a symbol, recognise mathematical expressions, and evaluate the models.

## Features

- **Image Classification:** Upload an image of a handwritten mathematical symbol and predict its class.
- **Draw a Symbol:** Draw a symbol on a canvas and classify it.
- **Expression Recognition:** Draw a mathematical expression and identify its symbols.
- **Calculator:** Evaluate supported mathematical expressions.
- **Model Evaluation:** View model performance and evaluation results.
- **Two Models:** Uses a Support Vector Machine (SVM) and a Convolutional Neural Network (CNN).

## Technologies Used

- Python
- TensorFlow and Keras
- Scikit-learn
- Streamlit
- NumPy
- Pillow

## Dataset

The project uses the HASYv2 dataset, which contains handwritten mathematical symbols. The data is used to train and evaluate the classification models.

## CNN Model Performance

The CNN was trained and evaluated using a train-validation-test split.

| Metric | Result |
|---|---:|
| Test Accuracy | 99.14% |
| Macro F1 Score | 98.28% |
| Top-3 Accuracy | 99.69% |

These results describe the CNN's performance on the held-out test set.

## Running the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/nadiacec054/ncaproj.git
   cd ncaproj
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies using the project's dependency file.

4. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL displayed in the terminal.

## Project Structure

- `app.py` — Streamlit application and user interface
- `train.py` — Model training
- `predict.py` — Symbol prediction
- `evaluate.py` — Model evaluation
- `models/` — Saved models and model metadata
- `evaluation_results/` — Evaluation outputs

## Project Goal

The goal of this project is to explore how machine learning and deep learning can be used to recognise handwritten mathematical symbols and provide a simple interactive interface for testing the models.
