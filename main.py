from fastapi import FastAPI, File, UploadFile
import pickle
import pandas as pd
from PIL import Image
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Load the saved model
with open('best_lgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocessing function
def preprocess_image(image):
    """
    Preprocess an image to be compatible with the model input format.

    Parameters:
        image: PIL Image object.
    
    Returns:
        pd.DataFrame: A DataFrame containing the flattened and normalized pixel values.
    """
    # Convert to grayscale
    image = image.convert('L')

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to a numpy array and normalize pixel values
    pixel_array = np.array(image) / 255.0

    # Flatten the array and create a DataFrame
    input_df = pd.DataFrame([pixel_array.flatten()], columns=[f'pixel{i}' for i in range(784)])
    
    return input_df

@app.get("/")
def read_root():
    return {"message": "Welcome to the MNIST Digit Prediction API"}



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the digit from an uploaded image file.

    Parameters:
        file: UploadFile object containing the image.
    
    Returns:
        JSON response with the predicted digit.
    """
    try:
        # Open the uploaded image
        image = Image.open(file.file)

        # Preprocess the image
        input_df = preprocess_image(image)

        # Predict using the loaded model
        prediction = model.predict(input_df)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        return {"error": str(e)}