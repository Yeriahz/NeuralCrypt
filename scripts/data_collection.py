# Import Libraries
import os
import requests
import pandas as pd
from pathlib import Path

# Replace 'your_api_key_here' with your actual API key from Cryptocompare
API_KEY = '9bf2ff68d6680c3f789283da46195442cef9ee8f601182cfc731f248ab6616e9'
BASE_URL = 'https://min-api.cryptocompare.com/data/v2/'
DATA_FOLDER = Path(__file__).resolve().parents[1] / "data"

print("API key and base URL set up.")

# Function to fetch data from any endpoint
def fetch_data(endpoint, params):
    url = f"{BASE_URL}{endpoint}"
    params["api_key"] = API_KEY
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data from {endpoint}: {response.status_code}")
        return None

# Historical Daily Data
def get_historical_daily(crypto="BTC", currency="USD", limit=100):
    params = {"fsym": crypto, "tsym": currency, "limit": limit}
    data = fetch_data("histoday", params)
    if data:
        df = pd.DataFrame(data["Data"]["Data"])
        df.rename(columns={
            "time": "time",
            "high": "High Price",
            "low": "Low Price",
            "open": "open",
            "volumefrom": "volumefrom",
            "volumeto": "volumeto",
            "close": "Close Price"
        }, inplace=True)
        df["time"] = pd.to_datetime(df["time"], unit="s")  # Convert timestamps to readable dates
        return df
    return None

# Fetch Latest Daily Data for Prediction
def get_latest_data(crypto="BTC", currency="USD", limit=5):
    params = {"fsym": crypto, "tsym": currency, "limit": limit}
    data = fetch_data("histoday", params)
    if data:
        df = pd.DataFrame(data["Data"]["Data"])
        df.rename(columns={
            "time": "time",
            "high": "High Price",
            "low": "Low Price",
            "open": "open",
            "volumefrom": "volumefrom",
            "volumeto": "volumeto",
            "close": "Close Price"
        }, inplace=True)
        df["time"] = pd.to_datetime(df["time"], unit="s")  # Convert timestamps to readable dates
        return df
    return None

# Main Function
def main():
    # Ensure the data folder exists
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    print("Fetching historical daily data...")
    daily_data = get_historical_daily("BTC", "USD", 100)  # Fetch 100 days of data
    if daily_data is not None:
        daily_data.to_csv(f"{DATA_FOLDER}\\btc_historical_daily.csv", index=False)
        print("Saved daily data to btc_historical_daily.csv")

    print("Fetching latest daily data for predictions...")
    latest_data = get_latest_data("BTC", "USD", 5)  # Fetch last 5 days for predictions
    if latest_data is not None:
        latest_data.to_csv(f"{DATA_FOLDER}\\new_data.csv", index=False)
        print("Saved latest data to new_data.csv")

# Run the script
if __name__ == "__main__":
    main()
