from fastapi import FastAPI, HTTPException
from app.s3_client import S3Client
from app.model import DiabetesPredictor
import pandas as pd
from datetime import datetime
import os

app = FastAPI(title="Healthcare Prediction API")
s3 = S3Client()
predictor = DiabetesPredictor()

# Initialize with sample data (in production, you'd load a pre-trained model)
@app.on_event("startup")
async def startup_event():
    try:
        # For demo purposes, we'll generate sample data and train a model
        from app.sample_data.generate_sample_data import generate_sample_data
        df = generate_sample_data(500)
        predictor.train(df)
        print("Model trained on startup with sample data")
    except Exception as e:
        print(f"Error during startup: {e}")

@app.get("/")
def read_root():
    return {"message": "Healthcare Prediction API", "status": "active"}

@app.post("/predict")
async def predict_from_s3(bucket: str, input_key: str, output_key: str = None):
    try:
        # Read data from S3
        df = s3.read_csv(bucket, input_key)
        print(f"Read {len(df)} records from s3://{bucket}/{input_key}")
        
        # Make predictions
        result_df = predictor.predict(df)
        
        # Prepare output key with timestamp if not provided
        if output_key is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_key = f"predictions/predictions_{timestamp}.csv"
        
        # Write results to S3
        s3.write_csv(result_df, bucket, output_key)
        
        # Generate summary statistics
        summary = {
            "timestamp": datetime.now().isoformat(),
            "input_file": f"s3://{bucket}/{input_key}",
            "output_file": f"s3://{bucket}/{output_key}",
            "records_processed": len(result_df),
            "positive_predictions": int(result_df['prediction'].sum()),
            "positive_rate": float(result_df['prediction'].mean()),
            "average_probability": float(result_df['probability'].mean())
        }
        
        # Save summary as JSON
        summary_key = output_key.replace('.csv', '_summary.json')
        s3.write_json(summary, bucket, summary_key)
        
        return {
            "status": "success", 
            "message": f"Processed {len(result_df)} records",
            "predictions_saved_to": f"s3://{bucket}/{output_key}",
            "summary_saved_to": f"s3://{bucket}/{summary_key}",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/health")
def model_health():
    return {
        "model_loaded": predictor.model is not None,
        "features": predictor.features,
        "target": predictor.target
    }

@app.get("/s3/files")
def list_files(bucket: str, prefix: str = ""):
    try:
        files = s3.list_files(bucket, prefix)
        return {"bucket": bucket, "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))