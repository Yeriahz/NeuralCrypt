# live_predict.py

# Import Libraries
import requests
import pandas as pd
import joblib
import time
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sqlalchemy import create_engine, exc, text  # <-- important: import text
from tabulate import tabulate
from colorama import init, Fore, Style

###################################
#        INITIAL SETUP
###################################
init(autoreset=True)  # colorama: reset automatically

BASE_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_PATH / "models" / "multi_horizon_model.pkl"
LOG_PATH = BASE_PATH / "logs" / "live_predictions.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure logs folder exists

# MySQL Configuration
MYSQL_CONFIG = {
    'host': '147.135.37.208',
    'port': 3307,
    'user': 'yeriahz_dev',
    'password': 'Wo58vIka16ka',
    'database': 'neural_crypt'
}

engine = create_engine(
    f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
    f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
)

# Cryptocompare API Key and Base URL
API_KEY = '9bf2ff68d6680c3f789283da46195442cef9ee8f601182cfc731f248ab6616e9'
BASE_URL_OHLC = "https://min-api.cryptocompare.com/data/v2/histominute"

# Fetch interval: 15 seconds
FETCH_INTERVAL = 15

###################################
#        HELPER FUNCTIONS
###################################
def banner_print(message, color=Fore.CYAN):
    """Print a visually distinct banner with optional color."""
    print("\n" + "=" * 70)
    print(color + Style.BRIGHT + message + Style.RESET_ALL)
    print("=" * 70 + "\n")

def fetch_ohlc_data(crypto="BTC", currency="USD", limit=60):
    """
    Fetch minute-by-minute OHLC data from CryptoCompare.
    limit=60 => last 60 minutes by default.
    """
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
        print(Fore.RED + f"Error fetching OHLC data: {e}" + Style.RESET_ALL)
        return None

def prepare_features(historical_data):
    """
    Creates the EXACT columns your trained model uses:
    - Close_Lag1
    - 7-day MA
    - Volatility
    - VWAP
    - Pct_Change

    Returns (features_for_latest_row, latest_close_price).
    """
    df = pd.DataFrame(historical_data)
    if df.empty:
        raise ValueError("No historical data to create features.")

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["Close Price"] = df["close"]
    df["High Price"] = df["high"]
    df["Low Price"] = df["low"]
    df["Volume To"] = df["volumeto"]

    # 1) Lag
    df["Close_Lag1"] = df["Close Price"].shift(1)

    # 2) 7-day MA
    df["7-day MA"] = df["Close Price"].rolling(window=7, min_periods=1).mean()

    # 3) Volatility
    df["Volatility"] = (
        (df["High Price"] - df["Low Price"])
        / df["Low Price"] * 100
    )

    # 4) VWAP
    df["VWAP"] = df.apply(
        lambda row: row["Volume To"] / (
            row["High Price"] + row["Low Price"] + row["Close Price"]
        ) if (row["High Price"] + row["Low Price"] + row["Close Price"]) != 0 else 0,
        axis=1
    )

    # 5) Pct_Change
    df["Pct_Change"] = df["Close Price"].pct_change().fillna(0) * 100

    # Grab the latest row
    latest = df.iloc[-1]
    feature_cols = ["Close_Lag1", "7-day MA", "Volatility", "VWAP", "Pct_Change"]

    features_row = latest[feature_cols].to_frame().T
    # Address downcasting warning
    features_row = features_row.infer_objects(copy=False).fillna(features_row.mean())

    return features_row, latest["Close Price"]

def generate_signals(predictions, current_price):
    signals = []
    for pred_val in predictions:
        if pred_val > current_price * 1.01:
            signals.append("Buy")
        elif pred_val < current_price * 0.99:
            signals.append("Sell")
        else:
            signals.append("Hold")
    return signals

def ensure_columns_exist():
    """
    ALTER TABLE to add missing columns if not already present.
    Specifically for predicted_45min, signal_45min, etc.
    Wrap it in `text(...)` so SQLAlchemy recognizes it as executable.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                ALTER TABLE `multi_horizon_predictions`
                ADD COLUMN `predicted_45min` DOUBLE,
                ADD COLUMN `signal_45min` VARCHAR(255)
            """))
        print(Fore.GREEN + "Successfully added 'predicted_45min' and 'signal_45min' columns to MySQL table." + Style.RESET_ALL)

    except exc.OperationalError as e:
        # If error is "duplicate column name", we can ignore
        if "duplicate column name" in str(e).lower():
            print(Fore.YELLOW + "Columns already exist. Skipping column add step." + Style.RESET_ALL)
        else:
            print(Fore.RED + f"Error adding columns to MySQL table: {e}" + Style.RESET_ALL)

def save_predictions_to_logs(timestamp, close_price, predictions, signals):
    """
    Save to 'multi_horizon_predictions' table and local CSV log.
    We assume 5 horizons => 5 columns: 5min, 15min, 30min, 45min, 1h.
    """
    horizon_labels = ["5min", "15min", "30min", "45min", "1h"]

    row_data = {
        "timestamp": timestamp,
        "real_price": close_price
    }
    for lbl, pred, sig in zip(horizon_labels, predictions, signals):
        row_data[f"predicted_{lbl}"] = pred
        row_data[f"signal_{lbl}"] = sig

    df_log = pd.DataFrame([row_data])

    # 1) Insert into MySQL
    try:
        df_log.to_sql(
            name="multi_horizon_predictions",
            con=engine,
            if_exists="append",
            index=False
        )
        print(Fore.GREEN + "Predictions saved to MySQL: 'multi_horizon_predictions'." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error saving predictions to MySQL: {e}" + Style.RESET_ALL)

    # 2) Append to CSV
    log_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        if not log_exists:
            header_line = ",".join(df_log.columns)
            f.write(header_line + "\n")
        data_line = ",".join(map(str, df_log.iloc[0].values))
        f.write(data_line + "\n")
    print(Fore.GREEN + f"Predictions appended to CSV: {LOG_PATH}" + Style.RESET_ALL)

###################################
#            MAIN LOOP
###################################
def main():
    banner_print("Starting Live Prediction Script", color=Fore.MAGENTA)

    # 1) Load the trained model
    if not MODEL_PATH.exists():
        print(Fore.RED + f"Error: Model file {MODEL_PATH} not found. Please train the model first." + Style.RESET_ALL)
        return
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(Fore.RED + f"Error loading model: {e}" + Style.RESET_ALL)
        return
    print(Fore.CYAN + "Model loaded successfully.\n" + Style.RESET_ALL)

    # 2) Ensure MySQL table has the correct columns
    ensure_columns_exist()

    # 3) Start live loop
    print(Fore.YELLOW + f"Entering live prediction loop. Data will be fetched every {FETCH_INTERVAL} seconds. "
          "Press Ctrl+C to stop.\n" + Style.RESET_ALL)

    try:
        while True:
            now_utc = datetime.now(timezone.utc).replace(microsecond=0)
            banner_print(f"Fetching live OHLC data at {now_utc} UTC", Fore.BLUE)

            data = fetch_ohlc_data("BTC", "USD", limit=60)
            if not data:
                print(Fore.RED + "No data returned from API. Retrying in 15 seconds..." + Style.RESET_ALL)
                time.sleep(15)
                continue

            # Prepare features
            try:
                features, current_price = prepare_features(data)
            except Exception as e:
                print(Fore.RED + f"Error preparing features: {e}" + Style.RESET_ALL)
                time.sleep(15)
                continue

            # Predict
            try:
                pred_array = model.predict(features)
            except Exception as e:
                print(Fore.RED + f"Error during prediction: {e}" + Style.RESET_ALL)
                time.sleep(15)
                continue

            # Single vs. multi-output
            if pred_array.ndim == 1:
                predictions = [pred_array[0]]
            else:
                predictions = pred_array[0]

            # Generate signals
            signals = generate_signals(predictions, current_price)

            # Print table
            horizon_labels = ["5 min", "15 min", "30 min", "45 min", "1 hr"]
            table_data = [
                ["Current Price", f"${current_price:.2f}"]
            ]
            for lbl, pred, sig in zip(horizon_labels, predictions, signals):
                table_data.append([lbl, f"${pred:.2f}", sig])

            print(tabulate(table_data, headers=["Horizon", "Predicted Price", "Signal"], tablefmt="fancy_grid"))

            # Save predictions
            save_predictions_to_logs(now_utc, current_price, predictions, signals)

            # Sleep 15 seconds
            print(Fore.CYAN + f"\nSleeping for {FETCH_INTERVAL} seconds before next live update..." + Style.RESET_ALL)
            time.sleep(FETCH_INTERVAL)

    except KeyboardInterrupt:
        banner_print("Exiting Live Prediction Script (KeyboardInterrupt)", color=Fore.MAGENTA)
    except Exception as e:
        print(Fore.RED + f"Unexpected error in main loop: {e}" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
