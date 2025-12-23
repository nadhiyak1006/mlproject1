import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.utils import load_data, save_model
from src.preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_price_prediction_model():
    """Trains and evaluates regression models, then saves the best one."""

    # Load data
    df = load_data('data/raw/products.csv')
    if df is None:
        logging.error("Failed to load data. Exiting training.")
        return

    # Preprocess
    try:
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
            df, target_column='price'
        )
        save_model(preprocessor, 'models/price_preprocessor.pkl')
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    best_model = None
    best_r2 = float("-inf")   # ✅ FIX HERE

    for name, model in models.items():
        logging.info(f"Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            logging.info(f"New best model selected: {name}")

    if best_model is not None:
        save_model(best_model, 'models/price_model.pkl')
        logging.info(
            f"Best model saved: {type(best_model).__name__} with R2={best_r2:.2f}"
        )
    else:
        logging.warning("No best model found.")

if __name__ == "__main__":
    train_price_prediction_model()
