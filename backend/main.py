from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

# Шляхи до папок
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../models')
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')

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

# 1. Віддаємо користувачам наш красивий веб-інтерфейс на головній сторінці!
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

# 2. А це наш ШІ-мозок, який рахує ціни
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