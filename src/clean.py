import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/data.csv")

print("🔹 Original shape:", df.shape)

# Remove rows with missing values
df = df.dropna()

# Create a new feature: pickup hour
df["pickup_hour"] = pd.to_datetime(df["pickup"]).dt.hour

# Select useful columns
df = df[["pickup_hour", "passengers", "fare"]]

print("🔹 Cleaned shape:", df.shape)

# Save cleaned data
df.to_csv("data/processed/clean_data.csv", index=False)

print("✅ Cleaned data saved to data/processed/clean_data.csv")

