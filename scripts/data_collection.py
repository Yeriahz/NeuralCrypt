# Import Libraries
import os
import requests
import pandas as pd
from pathlib import Path

# Constants and Configurations
API_KEY = '9bf2ff68d6680c3f789283da46195442cef9ee8f601182cfc731f248ab6616e9'
BASE_URL = 'https://min-api.cryptocompare.com/data/v2/'
DATA_FOLDER = Path(__file__).resolve().parents[1] / "data"

print("API key and base URL set up.")

# Function to Fetch Data from Cryptocompare
def fetch_data(endpoint, params):
    """
    Fetch data from a specified Cryptocompare API endpoint.
    """
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
    """
    Fetch historical daily data for a given cryptocurrency.
    """
    params = {"fsym": crypto, "tsym": currency, "limit": limit}
    data = fetch_data("histoday", params)
    if data:
        df = pd.DataFrame(data["Data"]["Data"])
        df.rename(columns={
            "time": "time",
            "high": "High Price",
            "low": "Low Price",
            "open": "Open Price",
            "volumefrom": "Volume From",
            "volumeto": "Volume To",
            "close": "Close Price"
        }, inplace=True)
        df["time"] = pd.to_datetime(df["time"], unit="s")  # Convert timestamps to readable dates
        return df
    else:
        print("Failed to fetch historical daily data.")
        return None

# Fetch Latest Data for Predictions
def get_latest_data(crypto="BTC", currency="USD", limit=5):
    """
    Fetch the latest data for predictions.
    """
    params = {"fsym": crypto, "tsym": currency, "limit": limit}
    data = fetch_data("histoday", params)
    if data:
        df = pd.DataFrame(data["Data"]["Data"])
        df.rename(columns={
            "time": "time",
            "high": "High Price",
            "low": "Low Price",
            "open": "Open Price",
            "volumefrom": "Volume From",
            "volumeto": "Volume To",
            "close": "Close Price"
        }, inplace=True)
        df["time"] = pd.to_datetime(df["time"], unit="s")  # Convert timestamps to readable dates
        return df
    else:
        print("Failed to fetch latest data.")
        return None

# Save Data to CSV
def save_to_csv(dataframe, filename):
    """
    Save a DataFrame to a CSV file in the data folder.
    """
    file_path = DATA_FOLDER / filename
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)  # Ensure data folder exists
    dataframe.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

# Main Function
def main():
    print("Fetching historical daily data...")
    daily_data = get_historical_daily("BTC", "USD", 100)
    if daily_data is not None:
        save_to_csv(daily_data, "btc_historical_daily.csv")

    print("Fetching latest daily data for predictions...")
    latest_data = get_latest_data("BTC", "USD", 5)
    if latest_data is not None:
        save_to_csv(latest_data, "new_data.csv")

# Run the Script
if __name__ == "__main__":
    main()
