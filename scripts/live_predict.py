# Import Libraries
import requests
import pandas as pd
import joblib
import time
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sqlalchemy import create_engine
from tabulate import tabulate  # For better table formatting in CMD

# File Paths
BASE_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_PATH / "models" / "multi_horizon_model.pkl"
LOG_PATH = BASE_PATH / "logs" / "live_predictions.csv"

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

# Cryptocompare API Key and Base URL
API_KEY = '9bf2ff68d6680c3f789283da46195442cef9ee8f601182cfc731f248ab6616e9'
BASE_URL_OHLC = "https://min-api.cryptocompare.com/data/v2/histominute"

# Function to Fetch OHLC Data
def fetch_ohlc_data(crypto="BTC", currency="USD", limit=60):
    params = {
        "fsym": crypto,
        "tsym": currency,
        "limit": limit,
        "api_key": API_KEY
    }
    try:
        response = requests.get(BASE_URL_OHLC, params=params, timeout=10)
        response.raise_for_status()
        data = response.json().get("Data", {}).get("Data", [])
        return data
    except requests.RequestException as e:
        print(f"Error fetching OHLC data: {e}")
        return None

# Function to Calculate RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean().bfill()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean().bfill()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill()

# Function to Prepare Features for Prediction
def prepare_features(historical_data):
    df = pd.DataFrame(historical_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['Close'] = df['close']
    df['High'] = df['high']
    df['Low'] = df['low']
    df['Volume'] = df['volumeto']

    # Calculate new features
    df['Close_Lag1'] = df['Close'].shift(1).bfill()
    df['SMA_5'] = df['Close'].rolling(window=5).mean().bfill()
    df['EMA_5'] = df['Close'].ewm(span=5).mean().bfill()
    df['Volatility'] = ((df['High'] - df['Low']) / df['Low']) * 100
    df['Pct_Change'] = df['Close'].pct_change().bfill()
    df['RSI'] = calculate_rsi(df['Close'])

    # Ensure the latest data point is returned with the expected features
    try:
        latest_data = df.iloc[-1]
        features = latest_data[['Close_Lag1', 'SMA_5', 'EMA_5', 'Volatility', 'Pct_Change', 'RSI']].to_frame().T
    except KeyError as e:
        raise ValueError(f"Missing required feature during prediction: {e}")

    return features

# Function to Save Predictions to Log and MySQL
def save_predictions(timestamp, close_price, predictions, signals):
    horizons = ["5min", "15min", "30min", "45min", "1h"]
    log_data = {
        "timestamp": timestamp,
        "real_price": close_price,
        **{f"predicted_{horizon}": pred for horizon, pred in zip(horizons, predictions)},
        **{f"signal_{horizon}": sig for horizon, sig in zip(horizons, signals)}
    }

    # Save to MySQL
    df = pd.DataFrame([log_data])
    try:
        df.to_sql(name="multi_horizon_predictions", con=engine, if_exists="append", index=False)
        print("Predictions and signals saved to MySQL.")
    except Exception as e:
        print(f"Error saving predictions to MySQL: {e}")

    # Save to CSV Log
    log_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a") as log_file:
        if not log_exists:
            headers = ",".join(log_data.keys())
            log_file.write(headers + "\n")
        log_file.write(",".join(map(str, log_data.values())) + "\n")

# Main Function
def main():
    print("Loading trained model...")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first.")
        return
    print("Model loaded successfully.\n")

    while True:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        print(f"\nFetching OHLC data at {now}...")

        try:
            historical_data = fetch_ohlc_data("BTC", "USD", limit=60)
            if historical_data:
                close_price = historical_data[-1]["close"]
                features = prepare_features(historical_data)
                predictions = model.predict(features)[0]

                # Generate Buy/Sell/Hold signals
                signals = [
                    "Buy" if pred > close_price * 1.01 else
                    "Sell" if pred < close_price * 0.99 else
                    "Hold"
                    for pred in predictions
                ]

                # Display Predictions
                horizons = ["5 min", "15 min", "30 min", "45 min", "1 hour"]
                prediction_table = [
                    ["Current Price", f"${close_price:.2f}"],
                    *[[h, f"${p:.2f}", s] for h, p, s in zip(horizons, predictions, signals)]
                ]

                print(tabulate(prediction_table, headers=["Horizon", "Predicted Price", "Signal"], tablefmt="grid"))

                # Save to Log and MySQL
                save_predictions(now, close_price, predictions, signals)

            else:
                print("No data fetched. Skipping this iteration.")

        except Exception as e:
            print(f"Error during prediction: {e}")

        time.sleep(300)  # Fetch data every 5 minutes

if __name__ == "__main__":
    main()