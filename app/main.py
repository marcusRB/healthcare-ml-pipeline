import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.model import DiabetesPredictor
from app.sample_data.generate_sample_data import generate_sample_data
from datetime import datetime

predictor = DiabetesPredictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs at startup and shutdown instead of on_event."""
    # Startup code
    try:
        df = generate_sample_data(500)
        predictor.train(df)
        print("✅ Model trained successfully with sample data on startup.")
    except Exception as e:
        print(f"❌ Error during startup: {e}")
    yield
    # Shutdown code (if any)
    print("🛑 FastAPI shutting down...")

app = FastAPI(title="Healthcare Prediction API (Local Demo)", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Healthcare Prediction API (Local Demo)", "status": "active"}

@app.post("/predict")
async def predict_local(records: int = 50):
    df = generate_sample_data(records)
    result_df = predictor.predict(df)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "records_processed": len(result_df),
        "positive_predictions": int(result_df['prediction'].sum()),
        "positive_rate": float(result_df['prediction'].mean()),
        "average_probability": float(result_df['probability'].mean())
    }
    return {
        "status": "success",
        "message": f"Processed {len(result_df)} records",
        "summary": summary,
        "sample_predictions": result_df.head(5).to_dict(orient="records")
    }

@app.get("/model/health")
def model_health():
    return {
        "model_loaded": predictor.model is not None,
        "features": predictor.features,
        "target": predictor.target
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
