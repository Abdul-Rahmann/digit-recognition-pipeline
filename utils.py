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


