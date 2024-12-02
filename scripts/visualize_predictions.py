# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# File Path
PREDICTIONS_PATH = r"C:\Users\jeria\NeuralCrypt Labs\data\predictions.csv"

# Load Predictions
print("Loading predictions...")
data = pd.read_csv(PREDICTIONS_PATH)

# Plot Predictions vs. Actual
plt.figure(figsize=(10, 6))
plt.plot(data["time"], data["Close Price"], label="Actual Close Price", marker="o")
plt.plot(data["time"], data["Predicted_Close"], label="Predicted Close Price", linestyle="--", marker="x")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.title("Actual vs. Predicted Close Prices")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
