# ncaproj
math symbol classifier 


built a Streamlit app where you can upload an image, draw a symbol, or draw a mathematical expression. There is also a calculator to evaluate typed expressions.



Tech Stack

Python
Scikit-learn
NumPy
Pillow
Streamlit

How I Built It

Dataset: Used the HASYv2 dataset, which contains images of handwritten mathematical symbols.

Training the model: Used Scikit-learn to train a model on the dataset to recognise different digits and mathematical symbols.

Image preprocessing: Created a separate preprocessing function to convert images to grayscale, crop the symbol, resize it, and prepare the pixel values for the model.

Symbol prediction: Used the trained model to predict the symbol in an uploaded image or a drawing on the app.

Calculator: Created a separate calculator module to evaluate mathematical expressions using Python's AST module.

Streamlit app: Connected the different modules into an app with options to upload images, draw symbols, draw expressions, and type expressions.


Install the dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

Limitations
Single-symbol recognition works better than full expression recognition. Handwritten expressions can sometimes be split or recognised incorrectly.
