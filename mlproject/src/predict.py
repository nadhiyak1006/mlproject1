import pandas as pd
import joblib
from src.utils import load_model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load models and preprocessors
price_model = load_model('models/price_model.pkl')
price_preprocessor = load_model('models/price_preprocessor.pkl')

fraud_model = load_model('models/fraud_model.pkl')
fraud_preprocessor = load_model('models/fraud_preprocessor.pkl')

def predict_price(input_data):
    """Predicts the price for a given input."""
    if price_model is None or price_preprocessor is None:
        logging.error("Price prediction model or preprocessor not loaded.")
        return None

    try:
        # Convert input to dataframe
        df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        processed_data = price_preprocessor.transform(df)
        
        # Predict
        prediction = price_model.predict(processed_data)
        return prediction[0]
    except Exception as e:
        logging.error(f"Error during price prediction: {e}")
        return None

def predict_fraud(input_data):
    """Detects if a transaction is fraudulent."""
    if fraud_model is None or fraud_preprocessor is None:
        logging.error("Fraud detection model or preprocessor not loaded.")
        return None

    try:
        df = pd.DataFrame([input_data])

        # ⚠️ ENSURE SAME COLUMNS AS TRAINING
        processed_data = fraud_preprocessor.transform(df)

        prediction = fraud_model.predict(processed_data)

        return bool(prediction[0] == -1)

    except Exception as e:
        logging.error(f"Error during fraud detection: {e}")
        return None

        
        # Preprocess the input data
        processed_data = fraud_preprocessor.transform(df)
        
        # Predict (-1 for fraud, 1 for normal)
        prediction = fraud_model.predict(processed_data)
        
        # Return True for fraud, False for normal
        return prediction[0] == -1
    except Exception as e:
        logging.error(f"Error during fraud detection: {e}")
        return None

if __name__ == '__main__':
    # Example Usage
    sample_product = {
        'brand': 'Nike',
        'category': 'Shoes',
        'material': 'Leather',
        'rating': 4.5,
        'transactions': 1000,
        # 'price' is not needed for prediction input
    }
    
    # We need to add all columns expected by the preprocessor
    # Let's create a full sample, some with default values
    full_sample = {
        'brand': 'Nike',
        'category': 'Shoes',
        'material': 'Leather',
        'rating': 4.5,
        'transactions': 1000,
        'price': 0 # Dummy value, will be dropped before prediction
    }


    # For fraud detection, the input format is similar
    predicted_price = predict_price(full_sample)
    if predicted_price is not None:
        logging.info(f"Predicted Price: ${predicted_price:.2f}")

    is_fraud = predict_fraud(full_sample)
    if is_fraud is not None:
        logging.info(f"Is Fraudulent: {is_fraud}")
