from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="DreamHouse AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# МАГІЯ ДЛЯ VERCEL: Визначаємо точний шлях до папки models
# __file__ - це шлях до цього файлу (main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Крокуємо на одну папку назад і заходимо в models
MODELS_DIR = os.path.join(BASE_DIR, '../models')

model_path = os.path.join(MODELS_DIR, 'house_pricer_xgb.pkl')
features_path = os.path.join(MODELS_DIR, 'model_features.pkl')

model = joblib.load(model_path)
model_features = joblib.load(features_path)

class HouseRequest(BaseModel):
    area_sqm: float
    rooms: int
    floors: int
    distance_city_km: float
    has_pool: int
    has_landscape: int
    material: str 

@app.get("/")
def read_root():
    return {"message": "Привіт! API працює. Перейди на /docs для тестування."}

@app.post("/predict_price")
def predict_price(house: HouseRequest):
    input_data = house.model_dump()
    df = pd.DataFrame([input_data])
    df_encoded = pd.get_dummies(df, columns=['material'], dtype=int)
    
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_final = df_encoded[model_features]
    predicted_price = model.predict(df_final)[0]
    
    return {
        "status": "success",
        "predicted_price_usd": round(float(predicted_price), 0),
        "message": f"Орієнтовна вартість будинку: ${predicted_price:,.0f}"
    }