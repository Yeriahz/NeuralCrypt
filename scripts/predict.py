# Import Libraries
import pandas as pd
import joblib
import os
from pathlib import Path
from sqlalchemy import create_engine

# File Paths
BASE_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_PATH / "models" / "multi_horizon_model.pkl"
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
    if not file_path.exists():
        raise FileNotFoundError(f"Error: The file {file_path} does not exist.")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Error: The file {file_path} is empty.")

# Function to Validate Required Columns
def validate_columns(dataframe, required_columns):
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Error: Missing columns in data: {missing_columns}")

# Function to Prepare Features
def prepare_features(data):
    """
    Create the same features as in predictive_model.py:
      - Close_Lag1
      - 7-day MA
      - Volatility
      - VWAP
      - Pct_Change
    """
    try:
        # 1) Previous day's Close
        data["Close_Lag1"] = data["Close Price"].shift(1)

        # 2) 7-day MA
        data["7-day MA"] = data["Close Price"].rolling(window=7, min_periods=1).mean()

        # 3) Volatility
        data["Volatility"] = (
            (data["High Price"] - data["Low Price"]) 
            / data["Low Price"] * 100
        )

        # 4) VWAP
        #    Volume To / (High Price + Low Price + Close Price)
        data["VWAP"] = data.apply(
            lambda row: row["Volume To"] / (row["High Price"] + row["Low Price"] + row["Close Price"])
            if (row["High Price"] + row["Low Price"] + row["Close Price"]) != 0 else 0,
            axis=1
        )

        # 5) Percentage Change (Close)
        data["Pct_Change"] = data["Close Price"].pct_change().fillna(0) * 100

        # Fill any NaN
        feature_cols = ["Close_Lag1", "7-day MA", "Volatility", "VWAP", "Pct_Change"]
        for col in feature_cols:
            data[col] = data[col].fillna(data[col].mean())

        # Return only the feature columns
        return data[feature_cols]

    except Exception as e:
        raise ValueError(f"Error in feature preparation: {e}")

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
        print(f"Validating file: {NEW_DATA_PATH}")
        validate_file(NEW_DATA_PATH)

        print("Loading trained model...")
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")

        print("Loading new data...")
        new_data = pd.read_csv(NEW_DATA_PATH)

        # Ensure these columns exist in new_data
        required_columns = ["time", "High Price", "Low Price", "Open Price", "Volume To", "Close Price"]
        validate_columns(new_data, required_columns)
        print("New data loaded and validated.")

        print("Preparing features for prediction...")
        X_new = prepare_features(new_data)

        if X_new.empty:
            raise ValueError("Error: No valid data available for prediction after feature preparation.")

        # Make multi-output predictions
        # (Our model now outputs multiple columns, e.g., Close_1, Close_3, Close_7, Close_14)
        print("Making predictions...")
        preds = model.predict(X_new)

        # If your model was trained with horizons = [1, 3, 7, 14], you'll have 4 columns of predictions:
        horizons = [1, 3, 7, 14]
        for i, h in enumerate(horizons):
            col_name = f"Pred_Close_{h}"
            new_data[col_name] = preds[:, i]

        # Save CSV locally
        print(f"Saving predictions to {PREDICTIONS_PATH}")
        new_data.to_csv(PREDICTIONS_PATH, index=False)
        print(f"Predictions saved to {PREDICTIONS_PATH}")

        # Optionally save to MySQL
        save_to_mysql(new_data, "predictions")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()