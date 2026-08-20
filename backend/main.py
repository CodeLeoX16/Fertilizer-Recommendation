import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Fertilizer Recommendation API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML artifacts
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("soil_encoder.pkl", "rb") as f:
    soil_encoder = pickle.load(f)

with open("crop_encoder.pkl", "rb") as f:
    crop_encoder = pickle.load(f)

with open("fertilizer_encoder.pkl", "rb") as f:
    fert_encoder = pickle.load(f)


class PredictionInput(BaseModel):
    temperature: float = Field(..., example=25.0)
    moisture: float = Field(..., example=0.5)
    rainfall: float = Field(..., example=200.0)
    ph: float = Field(..., example=6.5)
    nitrogen: float = Field(..., example=50.0)
    phosphorous: float = Field(..., example=50.0)
    potassium: float = Field(..., example=50.0)
    carbon: float = Field(..., example=1.0)
    soil: str = Field(..., example="Loamy Soil")
    crop: str = Field(..., example="rice")


@app.get("/api/options")
def get_options():
    """Return available categorical dropdown options."""
    return {
        "soils": soil_encoder.classes_.tolist(),
        "crops": crop_encoder.classes_.tolist()
    }


@app.post("/api/predict")
def predict_fertilizer(data: PredictionInput):
    try:
        # Validate categorical inputs
        if data.soil not in soil_encoder.classes_:
            raise HTTPException(status_code=400, detail=f"Invalid soil type: {data.soil}")
        if data.crop not in crop_encoder.classes_:
            raise HTTPException(status_code=400, detail=f"Invalid crop type: {data.crop}")

        soil_encoded = soil_encoder.transform([data.soil])[0]
        crop_encoded = crop_encoder.transform([data.crop])[0]

        df = pd.DataFrame(
            [[
                data.temperature,
                data.moisture,
                data.rainfall,
                data.ph,
                data.nitrogen,
                data.phosphorous,
                data.potassium,
                data.carbon,
                soil_encoded,
                crop_encoded
            ]],
            columns=[
                "Temperature",
                "Moisture",
                "Rainfall",
                "PH",
                "Nitrogen",
                "Phosphorous",
                "Potassium",
                "Carbon",
                "Soil",
                "Crop"
            ]
        )

        pred = model.predict(df)
        fertilizer_name = fert_encoder.inverse_transform(pred)[0]

        return {
            "status": "success",
            "recommended_fertilizer": fertilizer_name
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))