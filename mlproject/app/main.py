from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import sys
import os

# Add src to path to import predict module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from app.schemas import Product, PricePredictionResponse, FraudDetectionResponse
from src.predict import predict_price, predict_fraud

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Pricing & Fraud Detection API",
    description="API for predicting product prices and detecting fraudulent activities.",
    version="1.0.0"
)

# Mount frontend directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('frontend/index.html')

@app.post("/predict/price", response_model=PricePredictionResponse)
async def get_price_prediction(product: Product):
    """
    Predicts the optimal price for a product based on its features.
    """
    try:
        input_data = product.dict()
        logger.info(f"Received price prediction request: {input_data}")
        
        prediction = predict_price(input_data)
        
        if prediction is None:
            logger.error("Prediction returned None. Model might not be loaded.")
            raise HTTPException(status_code=500, detail="Model could not make a prediction.")
            
        logger.info(f"Price prediction successful: ${prediction:.2f}")
        return PricePredictionResponse(predicted_price=prediction)
        
    except Exception as e:
        logger.exception(f"Exception in price prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/fraud", response_model=FraudDetectionResponse)
async def get_fraud_detection(product: Product):
    """
    Detects if a product listing or transaction is potentially fraudulent.
    """
    try:
        input_data = product.dict()
        logger.info(f"Received fraud detection request: {input_data}")

        # ✅ FIX: Ensure price exists (fraud model expects it)
        if "price" not in input_data or input_data["price"] is None:
            input_data["price"] = 0.0
            logger.warning("Price missing in fraud request. Defaulting price to 0.0")

        is_fraud = predict_fraud(input_data)

        if is_fraud is None:
            logger.error("Fraud detection returned None.")
            raise HTTPException(status_code=500, detail="Model could not make a detection.")

        logger.info(f"Fraud detection successful. Is Fraud: {is_fraud}")
        return FraudDetectionResponse(is_fraud=is_fraud)

    except Exception as e:
        logger.exception(f"Exception in fraud detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# To run the app: uvicorn app.main:app --reload --port 8000
if __name__ == '__main__':
    # This block is for development and might not be used when deploying with Gunicorn/Uvicorn CLI
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
