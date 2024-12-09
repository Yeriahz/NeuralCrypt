import pandas as pd
import numpy as np
from pathlib import Path

# File Paths
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_FOLDER = BASE_PATH / "data"
CLEANED_DATA_FOLDER = BASE_PATH / "data" / "cleaned"

# Create cleaned data folder if it doesn't exist
CLEANED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

# Function to detect and handle missing values
def handle_missing_values(df):
    print("Handling missing values...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col == "conversionSymbol":
                print(f"Handling missing values in '{col}' by filling with 'Unknown'.")
                df[col].fillna("Unknown", inplace=True)
            elif df[col].dtype == "object":
                df[col].fillna("Unknown", inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    return df

# Function to detect and handle outliers using IQR
def handle_outliers(df, columns):
    print("Handling outliers...")
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Replace outliers with the median
        median = df[col].median()
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"Outliers detected in column '{col}': {len(outliers)}")
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = median
    return df

# Function to check data quality and save cleaned data
def check_data_quality(file_path):
    print(f"\nChecking data quality for: {file_path}")
    df = pd.read_csv(file_path)

    # Display missing values
    missing_values = df.isnull().sum()
    print(f"\nMissing values detected in {file_path}:\n{missing_values}")

    # Handle missing values
    df = handle_missing_values(df)

    # Detect and handle outliers in numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    df = handle_outliers(df, numerical_columns)

    # Save cleaned data
    cleaned_file_path = CLEANED_DATA_FOLDER / file_path.name
    df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to {cleaned_file_path}")

# Main function
def main():
    files_to_check = [
        DATA_FOLDER / "btc_historical_daily.csv",
        DATA_FOLDER / "new_data.csv"
    ]

    for file_path in files_to_check:
        if file_path.exists():
            check_data_quality(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
