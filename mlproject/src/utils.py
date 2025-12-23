import pandas as pd
import joblib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        return None

def save_model(model, file_path):
    """Saves a model to a .pkl file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        logging.info(f"Model saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving model to {file_path}: {e}")

def load_model(file_path):
    """Loads a model from a .pkl file."""
    try:
        model = joblib.load(file_path)
        logging.info(f"Model loaded successfully from {file_path}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading model from {file_path}: {e}")
        return None
