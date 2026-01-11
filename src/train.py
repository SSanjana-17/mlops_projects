import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load cleaned data
df = pd.read_csv("data/processed/clean_data.csv")

X = df[["pickup_hour", "passengers"]]
y = df["fare"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

# 🔴 Start MLflow tracking
mlflow.start_run()

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Log metrics & model
mlflow.log_metric("mse", mse)
mlflow.sklearn.log_model(model, "model")

# Save locally also
joblib.dump(model, "model/fare_model.pkl")

# End tracking
mlflow.end_run()

print("📉 MSE:", mse)
print("✅ Model tracked with MLflow")

