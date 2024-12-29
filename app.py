import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import pickle
from utils import preprocess_image

# Load the trained model
with open("best_lgbm_model.pkl", "rb") as f:
    model = pickle.load(f)


# Streamlit app
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a digit (0-9), and the model will predict the digit.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Processing...")

    # Preprocess the image
    input_df = preprocess_image(image)

    # Predict the digit
    prediction = model.predict(input_df)
    st.write(f"**Predicted Digit:** {int(prediction[0])}")