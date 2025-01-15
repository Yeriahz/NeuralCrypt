# Import Libraries
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate
from colorama import init, Fore, Style

init(autoreset=True)

BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data"
CLEANED_DATA_PATH = BASE_PATH / "data" / "cleaned"
CLEANED_DATA_PATH.mkdir(parents=True, exist_ok=True)

def banner_print(message, color=Fore.CYAN):
    print("\n" + "=" * 60)
    print(color + Style.BRIGHT + message + Style.RESET_ALL)
    print("=" * 60 + "\n")

def handle_missing_values(df):
    banner_print("CHECKING FOR MISSING VALUES", Fore.MAGENTA)

    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    
    table_data = [[col, missing_counts[col]] for col in df.columns]
    print(tabulate(table_data, headers=["Column", "Missing Count"], tablefmt="github"))

    if total_missing == 0:
        print(Fore.GREEN + "Great news! No missing values detected." + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "Some missing values detected. Filling them now..." + Style.RESET_ALL)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(df[col].mean())
    return df

def handle_outliers(df, threshold=4.0):
    banner_print("OUTLIER DETECTION", Fore.MAGENTA)
    
    skip_cols = [
        "High Price", "Low Price", "Open Price", 
        "Close Price", "Volume From", "Volume To",
        "conversionSymbol"
    ]
    if "conversionSymbol" in df.columns:
        df.drop(columns=["conversionSymbol"], inplace=True)

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in skip_cols
    ]
    
    initial_size = len(df)
    for col in numeric_cols:
        col_mean = df[col].mean()
        col_std = df[col].std(ddof=0)
        if col_std == 0:
            continue
        z_scores = np.abs((df[col] - col_mean) / col_std)
        df = df[z_scores < threshold]
    
    final_size = len(df)
    removed_rows = initial_size - final_size
    outlier_info = [
        ["Initial Rows", initial_size],
        ["Threshold Used", threshold],
        ["Removed Rows", removed_rows],
        ["Final Rows", final_size]
    ]
    print(tabulate(outlier_info, headers=["Outlier Info", "Value"], tablefmt="github"))

    if removed_rows == 0:
        print(Fore.GREEN + "No outliers removed. Your data looks stable!" + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + f"Removed {removed_rows} potential outliers." + Style.RESET_ALL)
    return df

def check_and_clean_data(file_path):
    banner_print(f"DATA QUALITY CHECK FOR: {file_path.name}", Fore.CYAN)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(Fore.RED + f"Error reading {file_path}: {e}" + Style.RESET_ALL)
        return

    print(f"{Fore.CYAN}Initial dataset size:{Style.RESET_ALL} {len(df)} rows")
    df = handle_missing_values(df)
    df = handle_outliers(df)

    if "conversionType" in df.columns:
        df.drop(columns=["conversionType"], inplace=True)
        print(Fore.GREEN + "Dropped 'conversionType' column as it's irrelevant." + Style.RESET_ALL)

    print(f"{Fore.CYAN}Final dataset size:{Style.RESET_ALL} {len(df)} rows")

    cleaned_file_path = CLEANED_DATA_PATH / file_path.name
    df.to_csv(cleaned_file_path, index=False)
    print(Fore.BLUE + f"Cleaned data saved to {cleaned_file_path}\n" + Style.RESET_ALL)

def main():
    files_to_check = [
        DATA_PATH / "btc_historical_daily.csv",
        DATA_PATH / "new_data.csv"
    ]
    for file in files_to_check:
        check_and_clean_data(file)

if __name__ == "__main__":
    main()
