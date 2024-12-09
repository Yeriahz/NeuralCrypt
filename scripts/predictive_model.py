# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from sqlalchemy import create_engine

# File Paths
BASE_PATH = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_PATH / "data" / "cleaned" / "btc_historical_daily.csv"
ENHANCED_DATA_PATH = BASE_PATH / "data" / "enhanced_daily_data.csv"
MODEL_PATH = BASE_PATH / "models" / "regression_model.pkl"
PLOT_PATH = BASE_PATH / "plots" / "residual_plot.png"

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

# Function to Load and Validate Data
def load_and_validate_data():
    print("Loading raw data...")
    data = pd.read_csv(RAW_DATA_PATH)
    data["time"] = pd.to_datetime(data["time"])

    required_columns = ["time", "Close Price", "High Price", "Low Price", "Volume To"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing critical columns: {missing_columns}")

    print("Validating data...")
    numerical_cols = ["Close Price", "High Price", "Low Price", "Volume To"]

    # Convert numerical columns to numeric data types
    for col in numerical_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Handle missing values in numerical columns
    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            print(f"Filling missing values in numerical column '{col}' with column mean.")
            data.loc[:, col] = data[col].fillna(data[col].mean())

    # Drop rows where essential columns are still missing
    data.dropna(subset=numerical_cols, inplace=True)

    if data.empty:
        raise ValueError("No valid data available after validation. Please check the dataset for issues.")

    print(f"Dataset loaded successfully: {len(data)} rows")
    return data

# Function for Feature Engineering
def feature_engineering(data):
    print("Creating features...")
    data = data.copy()  # Avoid chained assignment warnings
    data["Close_Lag1"] = data["Close Price"].shift(1)
    data["7-day MA"] = data["Close Price"].rolling(window=7).mean()
    data["Volatility"] = (data["High Price"] - data["Low Price"]) / data["Low Price"] * 100
    data["VWAP"] = data["Volume To"] / (data["High Price"] + data["Low Price"] + data["Close Price"])
    data["Pct_Change"] = data["Close Price"].pct_change() * 100

    # Handle missing values in features
    print("Handling missing values in features...")
    feature_cols = ["Close_Lag1", "7-day MA", "Volatility", "VWAP", "Pct_Change"]
    for col in feature_cols:
        data.loc[:, col] = data[col].fillna(data[col].mean())

    # Drop any remaining rows with missing values
    data.dropna(subset=feature_cols, inplace=True)

    if data.empty:
        raise ValueError("All rows dropped during feature engineering. Check for excessive missing values or outliers.")

    # Save enhanced data to CSV and MySQL
    data.to_csv(ENHANCED_DATA_PATH, index=False)
    print(f"Enhanced data saved to {ENHANCED_DATA_PATH}")

    try:
        data.to_sql(name='enhanced_daily_data', con=engine, if_exists='replace', index=False)
        print("Enhanced data saved to MySQL table 'enhanced_daily_data'.")
    except Exception as e:
        print(f"Error saving enhanced data to MySQL: {e}")

    return data

# Function to Train and Evaluate the Model
def train_and_evaluate_model(data):
    print("Preparing data for modeling...")
    X = data[["Close_Lag1", "7-day MA", "Volatility", "VWAP", "Pct_Change"]]
    y = data["Close Price"]

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Insufficient data for training.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

    # Pipeline with Standard Scaler and Regression Model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Model training complete.")

    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance:\n - MAE: {mae:.2f}\n - RMSE: {rmse:.2f}\n - R² Score: {r2:.2f}")

    # Save Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Prices")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH)
    print(f"Residual plot saved to {PLOT_PATH}")

    return pipeline

# Function to Save the Model
def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")

# Main Function
def main():
    try:
        data = load_and_validate_data()
        data = feature_engineering(data)
        model = train_and_evaluate_model(data)
        save_model(model)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
