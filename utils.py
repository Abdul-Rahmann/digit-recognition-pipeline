import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import janitor
import pandas as pd
import pandera as pa
import numpy as np
from PIL import Image

def show_digit(df, index):
    labels = df.iloc[:, 0]
    pixels = df.iloc[:, 1:]

    digit_label = labels.iloc[index]
    digit_pixels = pixels.iloc[index].values  # Convert row to array
    digit_pixels = digit_pixels.reshape(28, 28)  # Reshape to 28x28

    plt.imshow(digit_pixels, cmap='gray')
    plt.title(f"Digit: {digit_label}")
    plt.axis('off') 
    plt.show()

def read_data(path):
    return pd.read_csv(path).clean_names()

def validate_data(df):
    schema_columns = {
        'label': pa.Column(int, checks=pa.Check.in_range(0, 9))  # Label column
    }

    for i in range(784):
        schema_columns[f"pixel{i}"] = pa.Column(int, checks=pa.Check.in_range(0, 255))

    schema = pa.DataFrameSchema(schema_columns)

    try:
        validated_data = schema.validate(df)
        print("Data validation passed!")
        return validated_data  
    except pa.errors.SchemaError as e:
        print(f"Data validation failed:\n{e}")
        return None 

def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


import matplotlib.pyplot as plt
import janitor
import pandas as pd
import pandera as pa

def show_digit(df, index):
    labels = df.iloc[:, 0]
    pixels = df.iloc[:, 1:]

    digit_label = labels.iloc[index]
    digit_pixels = pixels.iloc[index].values  # Convert row to array
    digit_pixels = digit_pixels.reshape(28, 28)  # Reshape to 28x28

    plt.imshow(digit_pixels, cmap='gray')
    plt.title(f"Digit: {digit_label}")
    plt.axis('off')  # Turn off the axis for better visualization
    plt.show()

def read_data(path):
    return pd.read_csv(path).clean_names()

def validate_data(df):
    schema_columns = {
    'label': pa.Column(int, checks=pa.Check.in_range(0,9))
}

    for i in range(784):
        schema_columns[f"pixel{i}"] = pa.Column(int, checks=pa.Check.in_range(0, 255))

    schema = pa.DataFrameSchema(schema_columns)

    try:
        validated_data = schema.validate(df)
        print("Data validation passed")
    except pa.errors.SchemaError as e:
        print(f"Data Validation failed: \n{e}")

    return validate_data

def preprocess_image(image):
    """
    Preprocesses an image to be compatible with the model input format.

    Parameters:
        image (PIL.Image.Image or str): PIL Image object or path to the image file.
    
    Returns:
        pd.DataFrame: DataFrame with flattened, normalized pixel values.
    """
    # Load the image if a file path is provided
    if isinstance(image, str):
        image = Image.open(image)

    # Convert to grayscale
    image = image.convert('L')

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to a numpy array
    pixel_array = np.array(image)

    # Normalize pixel values to the range [0, 1]
    normalized_pixels = pixel_array / 255.0

    # Flatten the 2D array into a 1D array
    flattened_pixels = normalized_pixels.flatten()

    # Convert to DataFrame to match model input
    input_df = pd.DataFrame([flattened_pixels], columns=[f'pixel{i}' for i in range(784)])
    return input_df


