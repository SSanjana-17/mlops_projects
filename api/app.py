from fastapi import FastAPI
import joblib

# Create API app
app = FastAPI()

# Load trained model
model = joblib.load("model/fare_model.pkl")

@app.get("/")
def home():
    return {"message": "Taxi Fare Prediction API is running 🚕"}

@app.get("/predict")
def predict(pickup_hour: int, passengers: int):
    prediction = model.predict([[pickup_hour, passengers]])
    return {
        "pickup_hour": pickup_hour,
        "passengers": passengers,
        "predicted_fare": float(prediction[0])
    }

