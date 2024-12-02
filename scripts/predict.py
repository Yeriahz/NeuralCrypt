# Import Libraries
import pandas as pd
import numpy as np
import joblib
import os

# File Paths
MODEL_PATH = r"C:\Users\jeria\NeuralCrypt Labs\models\linear_regression_model.pkl"
NEW_DATA_PATH = r"C:\Users\jeria\NeuralCrypt Labs\data\new_data.csv"
PREDICTIONS_PATH = r"C:\Users\jeria\NeuralCrypt Labs\data\predictions.csv"

# Step 1: Validate File Existence and Non-Empty
if os.path.exists(NEW_DATA_PATH):
    if os.stat(NEW_DATA_PATH).st_size == 0:
        print(f"Error: The file {NEW_DATA_PATH} is empty. Please add valid data.")
        exit()
else:
    print(f"Error: The file {NEW_DATA_PATH} does not exist.")
    exit()

# Step 2: Load Trained Model
print("Loading trained model...")
model = joblib.load(MODEL_PATH)

# Step 3: Load New Data
print("Loading new data...")
new_data = pd.read_csv(NEW_DATA_PATH)

# Validate required columns
required_columns = ["time", "High Price", "Low Price", "open", "volumefrom", "volumeto", "Close Price"]
missing_columns = [col for col in required_columns if col not in new_data.columns]
if missing_columns:
    print(f"Error: Missing columns in new_data.csv: {missing_columns}")
    exit()

# Step 4: Prepare Features for Prediction
print("Preparing features for prediction...")
new_data["Close_Lag1"] = new_data["Close Price"].shift(1)  # Lagged feature
new_data["7-day MA"] = new_data["Close Price"].rolling(window=2).mean()  # Simplified 2-day moving average

# Handle Missing Values
new_data["Close_Lag1"] = new_data["Close_Lag1"].bfill()
new_data["7-day MA"] = new_data["7-day MA"].bfill()

# Select the features needed for prediction
X_new = new_data[["Close_Lag1", "7-day MA"]]

# Step 5: Make Predictions
print("Making predictions...")
new_data["Predicted_Close"] = model.predict(X_new)

# Step 6: Save Predictions
new_data.to_csv(PREDICTIONS_PATH, index=False)
print(f"Predictions saved to {PREDICTIONS_PATH}")
