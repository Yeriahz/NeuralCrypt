# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# File Paths
RAW_DATA_PATH = r"C:\Users\jeria\NeuralCrypt Labs\data\processed_daily_data.csv"
ENHANCED_DATA_PATH = r"C:\Users\jeria\NeuralCrypt Labs\data\enhanced_daily_data.csv"
MODEL_PATH = r"C:\Users\jeria\NeuralCrypt Labs\models\linear_regression_model.pkl"

# Step 1: Load and Process Data
print("Loading raw data...")
data = pd.read_csv(RAW_DATA_PATH)
data["time"] = pd.to_datetime(data["time"])

# Check initial dataset size
print(f"Dataset after loading: {len(data)} rows")
print(data.head())

# Step 2: Feature Engineering
print("Creating features...")
data["Close_Lag1"] = data["Close Price"].shift(1)  # Lagged feature
data["7-day MA"] = data["Close Price"].rolling(window=2).mean()  # Simplified 2-day MA

# Handle Missing Values
print("\nHandling missing values...")
data["Close_Lag1"] = data["Close_Lag1"].bfill()  # Backfill for lagged feature
data["7-day MA"] = data["7-day MA"].bfill()      # Backfill for rolling average

# Inspect for remaining NaN values
print("\nInspecting for NaN values...")
print(data[["Close_Lag1", "7-day MA"]].isnull().sum())

# Drop rows only if critical feature columns have NaN
print(f"Rows before dropna: {len(data)}")
data.dropna(subset=["Close_Lag1", "7-day MA"], inplace=True)
print(f"Rows after dropna: {len(data)}")

# Exit if dataset is still empty
if len(data) == 0:
    print("Error: No data available for training. Ensure the dataset is sufficient and feature engineering is correct.")
    exit()

# Save enhanced data
data.to_csv(ENHANCED_DATA_PATH, index=False)
print(f"Enhanced data saved to {ENHANCED_DATA_PATH}")

# Step 3: Prepare Data for Modeling
print("Preparing data for modeling...")
X = data[["Close_Lag1", "7-day MA"]]
y = data["Close Price"]

# Check dataset size before splitting
if len(X) == 0 or len(y) == 0:
    print("Error: Insufficient data for training. Check the feature engineering and dataset size.")
    exit()

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

# Step 4: Train the Model
print("Training linear regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")

# Step 5: Evaluate the Model
print("Evaluating the model...")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model Performance:\n - MAE: {mae:.2f}\n - RMSE: {rmse:.2f}")

# Step 6: Save the Model
joblib.dump(model, MODEL_PATH)
print(f"Trained model saved to {MODEL_PATH}")
