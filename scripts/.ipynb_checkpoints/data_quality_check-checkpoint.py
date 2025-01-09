# Import Libraries
import pandas as pd
import numpy as np
from pathlib import Path

# File Paths
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data"
CLEANED_DATA_PATH = BASE_PATH / "data" / "cleaned"

# Ensure the cleaned data folder exists
CLEANED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Function to handle missing values
def handle_missing_values(df):
    print("Handling missing values...")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df

# Function to handle outliers
def handle_outliers(df, threshold=3):
    print("Handling outliers...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    initial_size = len(df)
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < threshold]
    final_size = len(df)
    removed_rows = initial_size - final_size
    print(f"Removed {removed_rows} rows due to outliers.")
    return df

# Function to check and clean data
def check_and_clean_data(file_path):
    print(f"\nChecking data quality for: {file_path}")
    df = pd.read_csv(file_path)
    
    # Display initial dataset size
    print(f"Initial dataset size: {len(df)} rows")

    # Display missing values
    missing_values = df.isnull().sum()
    print(f"\nMissing values detected:\n{missing_values}")

    # Handle missing values
    df = handle_missing_values(df)

    # Handle outliers
    df = handle_outliers(df)

    # Drop conversionType if it's irrelevant
    if "conversionType" in df.columns:
        df.drop(columns=["conversionType"], inplace=True)
        print("Dropped 'conversionType' column as it's irrelevant.")

    # Display final dataset size
    print(f"Final dataset size: {len(df)} rows")

    # Save cleaned data
    cleaned_file_path = CLEANED_DATA_PATH / file_path.name
    df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to {cleaned_file_path}")

# Main function to run data checks
def main():
    files_to_check = [
        DATA_PATH / "btc_historical_daily.csv",
        DATA_PATH / "new_data.csv"
    ]
    
    for file in files_to_check:
        check_and_clean_data(file)

if __name__ == "__main__":
    main()
