
import joblib
import io
import pandas as pd
import numpy as np
from io import StringIO
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.features.build_features import house_price_cleaner, frontage_cleaning
from src.config import EXPERIMENTS_DIR

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model on startup, release resources on shutdown."""
    global model
    model = joblib.load(EXPERIMENTS_DIR / "best_model.pkl")
    print("Model loaded successfully.")
    yield
    # Cleanup on shutdown (if needed)

app = FastAPI(
    title="House Price Prediction API",
    description="Predict house sale prices using a trained ElasticNet regression pipeline.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running. Visit /docs for Swagger UI."}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/predictions")
def get_predictions():
    predictions_path = EXPERIMENTS_DIR / "prices_predictions_new_data.csv"

    if not predictions_path.exists():
        raise HTTPException(status_code=404, detail="No predictions found. Run predict first")

    df = pd.read_csv(predictions_path)

    return df.to_dict(orient="records")

@app.post("/predict")
async def predict_csv(file: UploadFile = File(...)):

    # Read uploaded CSV
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
        
        df = house_price_cleaner(df)
        df = frontage_cleaning(df)
        
        pred = model.predict(df)

        price = np.expm1(pred)
        
        df["PredictedPrice"] = price

        return df.to_dict(orient="records")
    
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail="Invalid CSV file")
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
