# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# File paths
DAILY_DATA_PATH = r"C:\Users\jeria\NeuralCrypt Labs\data\btc_historical_daily.csv"
HOURLY_DATA_PATH = r"C:\Users\jeria\NeuralCrypt Labs\data\btc_historical_hourly.csv"
MINUTE_DATA_PATH = r"C:\Users\jeria\NeuralCrypt Labs\data\btc_historical_minute.csv"
PROCESSED_DATA_PATH = r"C:\Users\jeria\NeuralCrypt Labs\data\processed_daily_data.csv"

# Load Data
daily_data = pd.read_csv(DAILY_DATA_PATH)
hourly_data = pd.read_csv(HOURLY_DATA_PATH)
minute_data = pd.read_csv(MINUTE_DATA_PATH)

# Preview Data
print("Daily Data:")
print(daily_data.head())
print("\nHourly Data:")
print(hourly_data.head())
print("\nMinute Data:")
print(minute_data.head())

# Data Cleaning
# Convert UNIX timestamps to readable datetime
daily_data["time"] = pd.to_datetime(daily_data["time"], unit="s")
hourly_data["time"] = pd.to_datetime(hourly_data["time"], unit="s")
minute_data["time"] = pd.to_datetime(minute_data["time"], unit="s")

# Rename columns for clarity
daily_data.rename(columns={"close": "Close Price", "high": "High Price", "low": "Low Price"}, inplace=True)

# Check for missing values
print("\nMissing values in Daily Data:\n", daily_data.isnull().sum())

# Exploratory Data Analysis
# Summary statistics
print("\nDaily Data Summary:\n", daily_data.describe())

# Correlation Matrix for Numeric Columns
numeric_data = daily_data.select_dtypes(include=["number"])
print("\nCorrelation Matrix:\n", numeric_data.corr())

# Identify Non-Numeric Columns (Optional)
non_numeric_cols = daily_data.select_dtypes(exclude=["number"]).columns
print("\nNon-numeric columns:", non_numeric_cols)

# Visualize trends in close prices
plt.figure(figsize=(12, 8))  # Larger figure size
plt.plot(daily_data["time"], daily_data["Close Price"], marker="o", label="Close Price", color="blue")
plt.title("BTC Daily Close Price", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Price (USD)", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()  # Adjust layout to avoid label cutoff
plt.show()

# Key Insights
# Calculate average daily close price
avg_price = daily_data["Close Price"].mean()
print(f"\nAverage BTC Daily Close Price: ${avg_price:.2f}")

# Find highest and lowest price days
max_price_row = daily_data.loc[daily_data["Close Price"].idxmax()]
min_price_row = daily_data.loc[daily_data["Close Price"].idxmin()]

print("\nHighest Price Day:", max_price_row)
print("Lowest Price Day:", min_price_row)

# Add moving average (7-day)
daily_data["7-day MA"] = daily_data["Close Price"].rolling(window=7).mean()

# Save cleaned and processed data
daily_data.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"\nProcessed daily data saved to {PROCESSED_DATA_PATH}")
