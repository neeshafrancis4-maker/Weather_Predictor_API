from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from sklearn import tree

app = FastAPI(title="Weather Predictor API", description="A small ML model to predict RAIN")

# 1. Training Data
data = {
    'temperature': [22.5, 38.0, 19.0, 31.5, 42.0, 24.0, 16.5, 29.0, 33.5, 20.0],
    'humidity':    [85.0, 35.0, 92.0, 70.0, 25.0, 88.0, 96.0, 65.0, 55.0, 90.0],
    'wind_speed':  [12.0, 18.5,  6.0, 10.0, 22.0,  8.5,  4.0, 14.0, 11.0,  7.0],
    'rain':        [   1,    0,    1,    1,    0,    1,    1,    0,    0,    1]
}

d = pd.DataFrame(data)

# Features aur Target ko theek tareeqay se select karna
d_att = d[['temperature', 'humidity', 'wind_speed']]
d_pass = d['rain']

# Model Train karna
t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
t.fit(d_att, d_pass)

# Input Validation Schema
class WeatherInput(BaseModel):
    temperature: float = Field(..., ge=20, le=50, description="temperature between 20-50")
    humidity: float = Field(..., ge=10, le=100, description="humidity between 10-100")
    wind_speed: float = Field(..., ge=0, le=40, description="wind_speed between 0-40")

# Home Route
@app.get("/")
def home():
    return {
       "Welcome to Weather Prediction API!",
        "Go to /docs to test it."
    }

# Prediction Route
@app.post("/prediction")
def weather_prediction(input_data: WeatherInput):
    user_input = [[input_data.temperature, input_data.humidity, input_data.wind_speed]]
    prediction = t.predict(user_input)

    if prediction[0] == 1:
        return {
            "status": "success",
            "prediction": 1,
            "result": "Rain Expected!"
        }
    else:
        return {
            "status": "success",
            "prediction": 0,
            "result": "Clear Weather!"
        }