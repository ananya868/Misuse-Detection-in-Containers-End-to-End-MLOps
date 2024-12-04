# Import necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

# Create an instance of FastAPI
app = FastAPI()

# Load the trained model (Assuming it's a pickle file)
model = joblib.load("tuning/artifacts/knn_v4.pkl")  # Replace with the path to your trained model

# Define the input format using Pydantic models
class PredictionRequest(BaseModel):
    features: List[float]


# Define the prediction endpoint
@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Convert the list of features to a NumPy array
    features = np.array(request.features).reshape(1, 9)
    prediction = model.predict(features)  # Make a prediction using the trained model
    return {"prediction": prediction[0]}  # Return the prediction

# Root endpoint for testing
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Misuse Detection API!"}



"""Get predictions from the API"""

# import requests 
# url = "http://127.0.0.1:8000/predict/"
# data = {
#     "features": [1.1, 2.1, 1.4, 2.2, 9.4, 1.7, 6.4, 7.0, 2.0]
# }
# response = requests.post(url, json=data)
# print(response.json()) 