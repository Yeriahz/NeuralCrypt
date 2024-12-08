# Import Libraries
import requests
import pandas as pd
import joblib
import time
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# File Paths
BASE_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_PATH / "models" / "regression_model.pkl"
LOG_PATH = BASE_PATH / "logs" / "live_predictions.csv"

# Cryptocompare API Key and Base URL
API_KEY = '9bf2ff68d6680c3f789283da46195442cef9ee8f601182cfc731f248ab6616e9'
BASE_URL_OHLC = "https://min-api.cryptocompare.com/data/v2/histohour"  # Fetch hourly data

# Function to Fetch OHLC Data
def fetch_ohlc_data(crypto="BTC", currency="USD", limit=7):
    params = {
        "fsym": crypto,
        "tsym": currency,
        "limit": limit,
        "api_key": API_KEY
    }
    try:
        response = requests.get(BASE_URL_OHLC, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()["Data"]["Data"]
            return data  # Return all fetched OHLC data
        else:
            print(f"Error fetching OHLC data: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None

# Function to Calculate Volatility
def calculate_volatility(high, low):
    return (high - low) / low * 100 if low > 0 else 0  # Avoid division by zero

# Function to Prepare Features for Prediction
def prepare_features(historical_data):
    if len(historical_data) < 2:
        raise ValueError("Insufficient historical data for feature calculation.")

    close_prices = [day["close"] for day in historical_data]
    high_prices = [day["high"] for day in historical_data]
    low_prices = [day["low"] for day in historical_data]
    volumes = [day["volumeto"] for day in historical_data]

    # Calculate Features
    close_lag1 = close_prices[-2]
    seven_day_ma = np.mean(close_prices)
    volatility = calculate_volatility(max(high_prices), min(low_prices))
    vwap = volumes[-1] / (max(high_prices) + min(low_prices) + close_prices[-1])
    pct_change = (close_prices[-1] - close_prices[-2]) / close_prices[-2] * 100

    # Create Feature DataFrame
    features = pd.DataFrame([{
        "Close_Lag1": close_lag1,
        "7-day MA": seven_day_ma,
        "Volatility": volatility,
        "VWAP": vwap,
        "Pct_Change": pct_change
    }])

    return features

# Function to Save to Log
def save_to_log(timestamp, close_price, prediction, signal):
    log_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a") as log_file:
        if not log_exists:
            log_file.write("timestamp,real_price,predicted_price,signal\n")
        log_file.write(f"{timestamp},{close_price},{prediction},{signal}\n")

# Main Function
def main():
    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.\n")

    os.makedirs(LOG_PATH.parent, exist_ok=True)
    fetch_interval = 300  # Fetch data every 5 minutes

    while True:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        print(f"Fetching OHLC data at {now}...")

        try:
            historical_data = fetch_ohlc_data("BTC", "USD", limit=7)
            if historical_data:
                latest_data = historical_data[-1]
                close_price = latest_data.get("close", 0)

                features = prepare_features(historical_data)
                prediction = model.predict(features)[0]

                # Determine Trading Signal
                if prediction > close_price * 1.02:
                    signal = "Buy"
                elif prediction < close_price * 0.98:
                    signal = "Sell"
                else:
                    signal = "Hold"

                # Display Predictions in CMD
                print("=" * 50)
                print(f"Real-Time BTC Close Price at {now}: ${close_price:.2f}")
                print(f"Predicted Close Price: ${prediction:.2f}")
                print(f"Trading Signal: {signal}")
                print("=" * 50)

                # Save to Log
                save_to_log(now, close_price, prediction, signal)

            else:
                print("No data fetched. Skipping this iteration.")

        except Exception as e:
            print(f"Error during prediction: {e}")

        time.sleep(fetch_interval)

if __name__ == "__main__":
    main()
