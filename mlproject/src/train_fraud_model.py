import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import logging

from src.utils import load_data, save_model
from src.preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_fraud_detection_model():
    """Trains an Isolation Forest model for fraud detection."""
    
    df = load_data('data/raw/products.csv')
    if df is None:
        logging.error("Failed to load data. Exiting training.")
        return
        
    # Define a heuristic for labeling anomalies/frauds
    # For example, let's consider unusually high prices for certain categories as potential fraud
    df['is_fraud'] = 0
    # A simple rule: price > 5000 is marked as potential fraud for this example
    df.loc[df['price'] > 5000, 'is_fraud'] = 1
    # Another rule: very low rating might indicate a fake product
    df.loc[df['rating'] < 1, 'is_fraud'] = 1
    
    # We need to ensure we have both classes for training
    if len(df[df['is_fraud'] == 1]) == 0:
        logging.warning("No fraudulent samples found based on the heuristic. The model might not be effective.")
        # Create a dummy fraud case to allow the script to run
        df.loc[0, 'is_fraud'] = 1

    try:
        # Preprocess data, keeping the fraud label separate
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target_column='is_fraud')
        save_model(preprocessor, 'models/fraud_preprocessor.pkl')
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        return

    # Using Isolation Forest - it's unsupervised but we can use labels for evaluation
    logging.info("Training Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    
    # Fit on training data (it's unsupervised, so y_train is not used for fitting)
    model.fit(X_train)

    # Predictions: -1 for anomalies (fraud), 1 for inliers (normal)
    y_pred_test = model.predict(X_test)
    
    # Convert predictions to 0/1 to match our labels
    y_pred_test_mapped = [1 if x == -1 else 0 for x in y_pred_test]

    logging.info("Evaluating fraud detection model...")
    # Note: Classification report might not be the best for anomaly detection,
    # but it gives a good overview if we have some labeled data.
    report = classification_report(y_test, y_pred_test_mapped, zero_division=0)
    logging.info(f"Classification Report:\n{report}")

    # Save the trained model
    save_model(model, 'models/fraud_model.pkl')
    logging.info("Fraud detection model saved.")

if __name__ == '__main__':
    train_fraud_detection_model()
