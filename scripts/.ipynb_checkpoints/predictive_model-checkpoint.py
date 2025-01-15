# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = BASE_PATH / "data" / "cleaned" / "btc_historical_daily.csv"
MODEL_PATH = BASE_PATH / "models" / "multi_horizon_model.pkl"

def load_and_validate_data():
    print("Loading cleaned data...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    if df.empty:
        raise ValueError("No data available for training. Please check the data quality checks.")
    return df

def prepare_data(df):
    print("Preparing features and targets...")

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

    feature_cols = ["Close_Lag1", "7-day MA", "Volatility", "VWAP", "Pct_Change"]
    X = df[feature_cols]

    # Create multiple horizon targets
    # Example: 5 min, 15 min, 30 min, 45 min, 1 hr => shift by 1,3,6,9,12 if each row = 5min
    # But if each row = daily, pick smaller horizons or fetch more data
    horizons = [1, 3, 6, 9]  # example
    y = pd.DataFrame()
    for h in horizons:
        y[f"Close_{h}"] = df["Close Price"].shift(-h)

    valid_rows = y.dropna().index
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    print(f"Number of valid rows after horizon shift: {len(valid_rows)}")
    if len(valid_rows) == 0:
        raise ValueError("No valid data available after horizon shift. Adjust horizons or get more data.")
    return X, y

def train_and_evaluate_model(X, y):
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            random_state=42
        )))
    ])

    print("Training the model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

    print(f"Model Performance on Test Set:")
    print(f" - MAE:  {mae:.2f}")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - R²:   {r2:.2f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def main():
    try:
        df = load_and_validate_data()
        X, y = prepare_data(df)
        train_and_evaluate_model(X, y)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()