# Import Libraries
import os
import requests
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

# Constants and Configurations
CRYPTOCOMPARE_API_KEY = '9bf2ff68d6680c3f789283da46195442cef9ee8f601182cfc731f248ab6616e9'
COINGECKO_API_KEY = 'CG-2rFHTphfVY7RPmZopFvdTpQH'
CRYPTOCOMPARE_BASE_URL = 'https://min-api.cryptocompare.com/data/v2/'
COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3/'
DATA_FOLDER = Path(__file__).resolve().parents[1] / "data"

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

print("API keys, base URLs, and MySQL configurations set up.")

# Function to Fetch Data from an API
def fetch_data(api, endpoint, params=None):
    base_url = CRYPTOCOMPARE_BASE_URL if api == "cryptocompare" else COINGECKO_BASE_URL
    params = params or {}
    if api == "cryptocompare":
        params["api_key"] = CRYPTOCOMPARE_API_KEY

    url = f"{base_url}{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data from {api}: {e}")
        return None

# Fetch Historical Daily Data from CryptoCompare
def get_historical_daily(crypto="BTC", currency="USD", limit=100):
    params = {"fsym": crypto, "tsym": currency, "limit": limit}
    data = fetch_data("cryptocompare", "histoday", params)
    if data and "Data" in data.get("Data", {}):
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
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
    else:
        print("Failed to fetch historical daily data.")
        return None

# Fetch Latest Data for Predictions from CoinGecko
def get_latest_data(crypto_id="bitcoin", currency="usd", days="7"):
    endpoint = f"coins/{crypto_id.lower()}/market_chart"
    params = {"vs_currency": currency, "days": days}
    data = fetch_data("coingecko", endpoint, params)
    if data and "prices" in data:
        prices = data.get("prices", [])
        df = pd.DataFrame(prices, columns=["time", "Close Price"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["High Price"] = df["Close Price"] * 1.02
        df["Low Price"] = df["Close Price"] * 0.98
        df["Open Price"] = df["Close Price"]
        df["Volume From"] = 0
        df["Volume To"] = 0
        return df
    else:
        print("Failed to fetch latest data.")
        return None

# Save Data to CSV
def save_to_csv(dataframe, filename):
    file_path = DATA_FOLDER / filename
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

# Save Data to MySQL
def save_to_mysql(dataframe, table_name):
    try:
        dataframe.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        print(f"Data saved to MySQL table '{table_name}'.")
    except Exception as e:
        print(f"Error saving to MySQL: {e}")

# Main Function
def main():
    print("Fetching historical daily data from CryptoCompare...")
    daily_data = get_historical_daily("BTC", "USD", 100)
    if daily_data is not None and not daily_data.empty:
        save_to_csv(daily_data, "btc_historical_daily.csv")
        save_to_mysql(daily_data, "btc_historical_daily")

    print("Fetching latest hourly data for predictions from CoinGecko...")
    latest_data = get_latest_data("bitcoin", "usd", "7")
    if latest_data is not None and not latest_data.empty:
        save_to_csv(latest_data, "new_data.csv")
        save_to_mysql(latest_data, "new_data")

if __name__ == "__main__":
    main()
