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

# File Paths
BASE_PATH = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = BASE_PATH / "data" / "cleaned" / "btc_historical_daily.csv"
MODEL_PATH = BASE_PATH / "models" / "multi_horizon_model.pkl"

# Function to load and validate data
def load_and_validate_data():
    print("Loading cleaned data...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    if df.empty:
        raise ValueError("No data available for training. Please check the data quality checks.")
    return df

# Function to prepare data for modeling
def prepare_data(df):
    print("Preparing features and targets...")
    # Create lag features and handle missing values
    df["Close_Lag1"] = df["Close Price"].shift(1).bfill()
    df["SMA_5"] = df["Close Price"].rolling(window=5, min_periods=1).mean()
    df["EMA_5"] = df["Close Price"].ewm(span=5).mean()
    df["Volatility"] = ((df["High Price"] - df["Low Price"]) / df["Low Price"] * 100).fillna(0)
    df["Pct_Change"] = df["Close Price"].pct_change().bfill()

    # Define feature set (X)
    X = df[["Close_Lag1", "SMA_5", "EMA_5", "Volatility", "Pct_Change"]]

    # Create target columns for multiple horizons
    horizons = [1, 3, 6, 12, 2016, 8640]  # 5 min, 15 min, 30 min, 1 hr, 7 days, 1 month
    y = pd.DataFrame()
    for h in horizons:
        y[f"Close_{h}"] = df["Close Price"].shift(-h)

    # Drop rows with NaN values in targets
    valid_rows = y.dropna().index
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    print(f"Number of valid rows after horizon shift: {len(valid_rows)}")

    if len(valid_rows) == 0:
        raise ValueError("No valid data available after creating horizon targets. Consider reducing horizons or adding more data.")

    return X, y

# Function to train and evaluate the model
def train_and_evaluate_model(X, y):
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the pipeline with a scaler and a random forest regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)))
    ])

    print("Training the model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='uniform_average'))
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

    print(f"Model Performance:\n - MAE: {mae:.2f}\n - RMSE: {rmse:.2f}\n - R²: {r2:.2f}")

    # Save the trained model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Main function
def main():
    try:
        df = load_and_validate_data()
        X, y = prepare_data(df)
        train_and_evaluate_model(X, y)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
