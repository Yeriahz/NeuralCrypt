# Import Libraries
import pandas as pd
import joblib
import os
from pathlib import Path
from sqlalchemy import create_engine

# File Paths
BASE_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_PATH / "models" / "regression_model.pkl"
NEW_DATA_PATH = BASE_PATH / "data" / "new_data.csv"
PREDICTIONS_PATH = BASE_PATH / "data" / "predictions.csv"

# MySQL Configuration
MYSQL_CONFIG = {
    'host': '147.135.37.208',
    'port': 3307,
    'user': 'yeriahz_dev',
    'password': 'Wo58vIka16ka',
    'database': 'neural_crypt'
}

# Create MySQL Engine
engine = create_engine(f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")

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
    Prepare features for prediction, including lagged close price, 7-day moving average, volatility, VWAP, and percentage change.
    """
    data["Close_Lag1"] = data["Close Price"].shift(1)  # Lagged feature
    window = min(len(data), 7)  # Use smaller window if dataset is small
    data["7-day MA"] = data["Close Price"].rolling(window=window).mean()  # Adjusted moving average
    data["Volatility"] = (data["High Price"] - data["Low Price"]) / data["Low Price"] * 100
    data["VWAP"] = data["Volume To"] / (data["High Price"] + data["Low Price"] + data["Close Price"])
    data["Pct_Change"] = data["Close Price"].pct_change() * 100

    # Handle missing values
    feature_cols = ["Close_Lag1", "7-day MA", "Volatility", "VWAP", "Pct_Change"]
    for col in feature_cols:
        data[col] = data[col].fillna(data[col].mean())

    # Drop rows with remaining NaN values (final check)
    data.dropna(subset=feature_cols, inplace=True)

    return data[feature_cols]

# Function to Save Predictions to MySQL
def save_to_mysql(dataframe, table_name):
    try:
        dataframe.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        print(f"Predictions saved to MySQL table '{table_name}'.")
    except Exception as e:
        print(f"Error saving predictions to MySQL: {e}")

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

        # Step 6: Save Predictions to CSV
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_data.to_csv(PREDICTIONS_PATH, index=False)
        print(f"Predictions saved to {PREDICTIONS_PATH}")

        # Step 7: Save Predictions to MySQL
        save_to_mysql(new_data, "predictions")

    except Exception as e:
        print(f"Error: {e}")

# Run the Script
if __name__ == "__main__":
    main()
