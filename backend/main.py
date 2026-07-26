from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="DreamHouse AI API")

model = joblib.load('../models/house_pricer_xgb.pkl')
model_features = joblib.load('../models/model_features.pkl')

class HouseRequest(BaseModel):
    area_sqm: float
    rooms: int
    floors: int
    distance_city_km: float
    has_pool: int
    has_landscape: int
    material: str  # 'Wood', 'Brick', 'Aerated_block'

@app.post("/predict_price")
def predict_price(house: HouseRequest):
    # JSON -> dict
    input_data = house.model_dump()
    
    df = pd.DataFrame([input_data])
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['material'], dtype=int)
    
    for col in model_features:
        if col is not None and col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_final = df_encoded[model_features]

    predicted_price = model.predict(df_final)[0]
    
    return {
        "status": "success",
        "predicted_price_usd": round(float(predicted_price), 0),
        "message": f"Price: ${predicted_price:,.0f}"
    }
