from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import xgboost as xgb
import os

app = FastAPI(title="DreamHouse AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../models')
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')

# 1. Змінили розширення на .json
model_path = os.path.join(MODELS_DIR, 'house_pricer_xgb.json')
features_path = os.path.join(MODELS_DIR, 'model_features.pkl')

# 2. Завантажуємо фічі
model_features = joblib.load(features_path)

# 3. Правильно завантажуємо XGBoost модель (Native Booster)
model = xgb.Booster()
model.load_model(model_path)

class HouseRequest(BaseModel):
    area_sqm: float
    rooms: int
    floors: int
    distance_city_km: float
    has_pool: int
    has_landscape: int
    material: str 

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

@app.post("/predict_price")
def predict_price(house: HouseRequest):
    input_data = house.model_dump()
    df = pd.DataFrame([input_data])
    df_encoded = pd.get_dummies(df, columns=['material'], dtype=int)
    
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_final = df_encoded[model_features]
    
    # 🚀 Використовуємо DMatrix для нативного Booster
    dmatrix = xgb.DMatrix(df_final)
    predicted_price = model.predict(dmatrix)[0]
    
    # 🚀 Safety Net: додаємо "запобіжник" для продакшну (ціна не може бути менше 10k)
    final_price = max(10000, float(predicted_price))
    
    return {
        "status": "success",
        "predicted_price_usd": round(final_price, 0),
        "message": f"Орієнтовна вартість будинку: ${final_price:,.0f}"
    }