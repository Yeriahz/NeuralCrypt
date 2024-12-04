import requests
import pandas as pd
import joblib
import time
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np  # For standard deviation and confidence intervals

# File Paths
BASE_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_PATH / "models" / "linear_regression_model.pkl"
LOG_PATH = BASE_PATH / "logs" / "live_predictions.csv"

# Cryptocompare API Key and Base URL
API_KEY = '9bf2ff68d6680c3f789283da46195442cef9ee8f601182cfc731f248ab6616e9'
BASE_URL_OHLC = "https://min-api.cryptocompare.com/data/v2/histohour"  # Fetch hourly data

# Function to Fetch OHLC Data
def fetch_ohlc_data(crypto="BTC", currency="USD", limit=7):
    """
    Fetch historical OHLC data for the specified cryptocurrency.
    """
    params = {
        "fsym": crypto,
        "tsym": currency,
        "limit": limit,
        "api_key": API_KEY
    }
    response = requests.get(BASE_URL_OHLC, params=params)
    if response.status_code == 200:
        try:
            data = response.json()["Data"]["Data"]
            return data  # Return all fetched OHLC data
        except (KeyError, IndexError):
            print("Error: Unexpected API response format.")
            return None
    else:
        print(f"Error fetching OHLC data: {response.status_code}")
        return None

# Function to Calculate Volatility
def calculate_volatility(high, low):
    """
    Calculate market volatility as a percentage.
    """
    return (high - low) / low * 100 if low > 0 else 0  # Avoid division by zero

# Function to Get the Next Hourly Close Time
def get_next_hourly_close():
    """
    Get the next hourly close time in UTC.
    """
    now = datetime.now(timezone.utc)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return next_hour

# Function to Automatically Prepare Features for Prediction
def prepare_features(historical_data, current_close):
    """
    Calculate lagged close price and 7-day moving average based on historical data.
    """
    if historical_data and len(historical_data) >= 2:
        close_prices = [day["close"] for day in historical_data]
        close_lag1 = close_prices[-2]
        seven_day_ma = sum(close_prices) / len(close_prices)
    else:
        print("Warning: Insufficient historical data for features. Using current close as fallback.")
        close_lag1 = current_close
        seven_day_ma = current_close

    return pd.DataFrame({
        "Close_Lag1": [close_lag1],
        "7-day MA": [seven_day_ma]
    })

# Function to Calculate Confidence Intervals
def calculate_confidence_interval(prediction, std_dev, confidence_level=1.96):
    """
    Calculate confidence intervals based on standard deviation and confidence level.
    """
    lower_bound = prediction - (confidence_level * std_dev)
    upper_bound = prediction + (confidence_level * std_dev)
    return round(lower_bound, 2), round(upper_bound, 2)

# Main Function
def main():
    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.\n")
    
    # Simulated standard deviation (can be replaced with real analysis)
    std_dev = 100  # Example standard deviation in USD
    
    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    
    fetch_interval = 300  # Default fetch interval (every 5 minutes)
    while True:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        print(f"Fetching OHLC data at {now}...")
        historical_data = fetch_ohlc_data("BTC", "USD", limit=7)  # Fetch historical data for features
        
        if historical_data:
            # Get the latest OHLC data
            latest_data = historical_data[-1]
            high = latest_data.get("high", 0)
            low = latest_data.get("low", 0)
            close_price = latest_data.get("close", 0)
            
            # Validate OHLC values
            if high > 0 and low > 0 and close_price > 0:
                volatility = calculate_volatility(high, low)
                print(f"Market Volatility: {volatility:.2f}%")
                
                # Adjust Fetch Interval Based on Volatility
                if volatility > 10:
                    fetch_interval = 60  # Fetch every 1 minute
                elif volatility > 5:
                    fetch_interval = 120  # Fetch every 2 minutes
                elif volatility > 2:
                    fetch_interval = 300  # Fetch every 5 minutes
                else:
                    fetch_interval = 600  # Fetch every 10 minutes
            else:
                print("Error: Invalid 'high', 'low', or 'close' values in OHLC data.")
                volatility = 0
            
            print(f"Adjusted fetch interval: {fetch_interval} seconds\n")
            
            # Automatically Prepare Features
            features = prepare_features(historical_data, current_close=close_price)
            
            # Predict
            prediction = model.predict(features)[0]
            lower_bound, upper_bound = calculate_confidence_interval(prediction, std_dev)
            next_close_time = get_next_hourly_close().replace(microsecond=0)

            # Generate Trading Signal
            if prediction > close_price * 1.02:
                signal = "Buy"
            elif prediction < close_price * 0.98:
                signal = "Sell"
            else:
                signal = "Hold"

            print(f"Real-Time BTC Close Price at {now}: ${close_price:.2f}")
            print(f"Predicted Close Price for {next_close_time}: ${prediction:.2f}")
            print(f"Confidence Interval: [${lower_bound}, ${upper_bound}]")
            print(f"Trading Signal: {signal}\n")

            # Save to Log
            if not LOG_PATH.exists():
                with open(LOG_PATH, "w") as log_file:
                    log_file.write("timestamp,real_price,predicted_price,confidence_interval,signal,volatility\n")
            
            with open(LOG_PATH, "a") as log_file:
                log_file.write(f"{now},{close_price},{prediction},[{lower_bound},{upper_bound}],{signal},{volatility:.2f}\n")
        
        time.sleep(fetch_interval)

if __name__ == "__main__":
    main()
