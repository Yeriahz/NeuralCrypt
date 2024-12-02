import requests
import pandas as pd
import joblib
import time
import os
from datetime import datetime, timedelta, timezone

# File Paths
MODEL_PATH = r"C:\Users\jeria\NeuralCrypt Labs\models\linear_regression_model.pkl"
LOG_PATH = r"C:\Users\jeria\NeuralCrypt Labs\logs\live_predictions.csv"

# Cryptocompare API Key and Base URL
API_KEY = '9bf2ff68d6680c3f789283da46195442cef9ee8f601182cfc731f248ab6616e9'
BASE_URL_OHLC = "https://min-api.cryptocompare.com/data/v2/histohour"  # Fetch hourly data

# Function to Fetch OHLC Data
def fetch_ohlc_data(crypto="BTC", currency="USD", limit=1):
    params = {
        "fsym": crypto,
        "tsym": currency,
        "limit": limit,
        "api_key": API_KEY
    }
    response = requests.get(BASE_URL_OHLC, params=params)
    if response.status_code == 200:
        data = response.json()["Data"]["Data"]
        return data[-1]  # Return the most recent OHLC data
    else:
        print(f"Error fetching OHLC data: {response.status_code}")
        return None

# Function to Calculate Volatility
def calculate_volatility(high, low):
    return (high - low) / low * 100  # Volatility as a percentage

# Function to Get the Next Hourly Close Time
def get_next_hourly_close():
    now = datetime.now(timezone.utc)  # Get current UTC time
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return next_hour

# Main Function
def main():
    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.\n")
    
    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    
    fetch_interval = 300  # Default fetch interval (every 5 minutes)
    while True:
        now = datetime.now(timezone.utc).replace(microsecond=0)  # Current UTC time without milliseconds
        print(f"Fetching OHLC data at {now}...")
        ohlc_data = fetch_ohlc_data("BTC", "USD")
        
        if ohlc_data:
            # Debugging: Print OHLC data
            print(f"OHLC Data: {ohlc_data}")
            
            # Extract OHLC values
            high = ohlc_data.get("high", 0)
            low = ohlc_data.get("low", 0)
            close_price = ohlc_data.get("close", 0)
            
            # Validate OHLC values
            if high > 0 and low > 0 and close_price > 0:
                volatility = calculate_volatility(high, low)
                print(f"Market Volatility: {volatility:.2f}%")
                
                # Adjust Fetch Interval Based on Volatility
                if volatility > 5:  # High volatility
                    fetch_interval = 120  # Fetch every 2 minutes
                elif volatility > 2:  # Moderate volatility
                    fetch_interval = 300  # Fetch every 5 minutes
                else:  # Low volatility
                    fetch_interval = 600  # Fetch every 10 minutes
            else:
                print("Error: Invalid 'high', 'low', or 'close' values in OHLC data.")
                volatility = 0
            
            print(f"Adjusted fetch interval: {fetch_interval} seconds\n")
            
            # Prepare Features for Prediction
            close_lag1 = close_price * 0.99  # Example proxy for lagged feature
            ma_7_day = close_price * 1.01  # Example proxy for moving average
            
            features = pd.DataFrame({
                "Close_Lag1": [close_lag1],
                "7-day MA": [ma_7_day]
            })
            
            # Predict
            prediction = model.predict(features)[0]
            next_close_time = get_next_hourly_close().replace(microsecond=0)  # Remove milliseconds

            # Generate Trading Signal
            if prediction > close_price * 1.02:  # Predicted price 2% higher
                signal = "Buy"
            elif prediction < close_price * 0.98:  # Predicted price 2% lower
                signal = "Sell"
            else:
                signal = "Hold"

            print(f"Real-Time BTC Close Price at {now}: ${close_price:.2f}")
            print(f"Predicted Close Price for {next_close_time}: ${prediction:.2f}")
            print(f"Trading Signal: {signal}\n")
            
            # Save to Log
            with open(LOG_PATH, "a") as log_file:
                log_file.write(f"{now},{close_price},{prediction},{signal},{volatility}\n")
        
        # Wait before fetching data again
        time.sleep(fetch_interval)

# Run the Script
if __name__ == "__main__":
    main()
