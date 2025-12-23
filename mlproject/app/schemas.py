from pydantic import BaseModel

class Product(BaseModel):
    brand: str
    category: str
    material: str
    rating: float
    transactions: int

class PricePredictionResponse(BaseModel):
    predicted_price: float

class FraudDetectionResponse(BaseModel):
    is_fraud: bool
