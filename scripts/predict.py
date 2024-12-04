# Import Libraries
import pandas as pd
import joblib
import os
from pathlib import Path

# File Paths
BASE_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_PATH / "models" / "linear_regression_model.pkl"
NEW_DATA_PATH = BASE_PATH / "data" / "new_data.csv"
PREDICTIONS_PATH = BASE_PATH / "data" / "predictions.csv"

# Function to Validate File Existence
def validate_file(file_path):
    """
    Check if a file exists and is non-empty.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Error: The file {file_path} does not exist.")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Error: The file {file_path} is empty. Please add valid data.")

# Function to Validate Required Columns
def validate_columns(dataframe, required_columns):
    """
    Validate that the required columns are present in the DataFrame.
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Error: Missing columns in data: {missing_columns}")

# Function to Prepare Features
def prepare_features(data):
    """
    Prepare features for prediction, including lagged close price and 7-day moving average.
    """
    data["Close_Lag1"] = data["Close Price"].shift(1)  # Lagged feature
    window = min(len(data), 7)  # Use smaller window if dataset is small
    data["7-day MA"] = data["Close Price"].rolling(window=window).mean()  # Adjusted moving average

    # Handle missing values
    data["Close_Lag1"] = data["Close_Lag1"].bfill()
    data["7-day MA"] = data["7-day MA"].bfill()

    # Check for remaining NaN values
    if data[["Close_Lag1", "7-day MA"]].isnull().any().any():
        print("Warning: Remaining NaN values detected after processing.")
        print(data[["Close_Lag1", "7-day MA"]].isnull().sum())

    # Drop rows with NaN values (final check)
    data.dropna(subset=["Close_Lag1", "7-day MA"], inplace=True)

    return data[["Close_Lag1", "7-day MA"]]

# Main Function
def main():
    try:
        # Step 1: Validate File Existence
        print(f"Validating file: {NEW_DATA_PATH}")
        validate_file(NEW_DATA_PATH)

        # Step 2: Load Trained Model
        print("Loading trained model...")
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")

        # Step 3: Load and Validate New Data
        print("Loading new data...")
        new_data = pd.read_csv(NEW_DATA_PATH)
        required_columns = ["time", "High Price", "Low Price", "Open Price", "Volume From", "Volume To", "Close Price"]
        validate_columns(new_data, required_columns)
        print("New data loaded and validated.")

        # Step 4: Prepare Features
        print("Preparing features for prediction...")
        X_new = prepare_features(new_data)

        # Step 5: Make Predictions
        if len(X_new) == 0:
            raise ValueError("Error: No valid data available for prediction after feature preparation.")
        print("Making predictions...")
        new_data["Predicted_Close"] = model.predict(X_new)

        # Step 6: Save Predictions
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_data.to_csv(PREDICTIONS_PATH, index=False)
        print(f"Predictions saved to {PREDICTIONS_PATH}")

    except Exception as e:
        print(f"Error: {e}")

# Run the Script
if __name__ == "__main__":
    main()
