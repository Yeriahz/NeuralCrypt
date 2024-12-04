# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

# File Paths
BASE_PATH = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_PATH / "data" / "btc_historical_daily.csv"
ENHANCED_DATA_PATH = BASE_PATH / "data" / "enhanced_daily_data.csv"
MODEL_PATH = BASE_PATH / "models" / "linear_regression_model.pkl"

# Function to Load and Preprocess Data
def load_and_preprocess_data():
    print("Loading raw data...")
    data = pd.read_csv(RAW_DATA_PATH)
    data["time"] = pd.to_datetime(data["time"])
    
    print(f"Dataset after loading: {len(data)} rows")
    print(data.head())

    print("Creating features...")
    data["Close_Lag1"] = data["Close Price"].shift(1)  # Lagged feature
    data["7-day MA"] = data["Close Price"].rolling(window=7).mean()  # 7-day Moving Average

    print("\nHandling missing values...")
    data["Close_Lag1"] = data["Close_Lag1"].bfill()  # Backfill lagged feature
    data["7-day MA"] = data["7-day MA"].bfill()  # Backfill rolling average

    print("\nInspecting for NaN values...")
    print(data[["Close_Lag1", "7-day MA"]].isnull().sum())

    print(f"Rows before dropna: {len(data)}")
    data.dropna(subset=["Close_Lag1", "7-day MA"], inplace=True)
    print(f"Rows after dropna: {len(data)}")

    if len(data) == 0:
        raise ValueError("No data available for training. Ensure the dataset is sufficient and feature engineering is correct.")

    # Save enhanced data for reference
    data.to_csv(ENHANCED_DATA_PATH, index=False)
    print(f"Enhanced data saved to {ENHANCED_DATA_PATH}")

    return data

# Function to Train and Evaluate the Model
def train_and_evaluate_model(data):
    print("Preparing data for modeling...")
    X = data[["Close_Lag1", "7-day MA"]]
    y = data["Close Price"]

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Insufficient data for training. Check the feature engineering and dataset size.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

    print("Training linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")

    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model Performance:\n - MAE: {mae:.2f}\n - RMSE: {rmse:.2f}")

    return model

# Function to Save the Model
def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")

# Main Function
def main():
    try:
        data = load_and_preprocess_data()
        model = train_and_evaluate_model(data)
        save_model(model)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
