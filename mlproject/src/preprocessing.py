import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import load_data
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(df):
    """Cleans the dataframe by removing duplicates and handling missing values."""
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    logging.info("Data cleaned: duplicates dropped and missing values handled.")
    return df

def get_preprocessor(numeric_features, categorical_features):
    """Creates a preprocessor pipeline for numeric and categorical features."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def preprocess_data(df, target_column=None, test_size=0.2, random_state=42):
    """
    Cleans, preprocesses, and splits the data.
    If target_column is specified, it separates features and target.
    """
    df = clean_data(df.copy())

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = get_preprocessor(numeric_features, categorical_features)

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Fit preprocessor on training data and transform both training and testing data
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        logging.info("Data preprocessed and split into training and testing sets.")
        return X_train_processed, X_test_processed, y_train, y_test, preprocessor
    else:
        # If no target column, just preprocess the whole dataset
        X_processed = preprocessor.fit_transform(X)
        logging.info("Data preprocessed.")
        return X_processed, preprocessor

if __name__ == '__main__':
    # Example usage:
    DATA_PATH = 'data/raw/products.csv'
    df = load_data(DATA_PATH)
    
    if df is not None:
        # For a regression task (predicting price)
        X_train_proc, X_test_proc, y_train, y_test, preprocessor = preprocess_data(df, target_column='price')
        
        # You can save the preprocessor for later use in inference
        from src.utils import save_model
        save_model(preprocessor, 'models/preprocessor.pkl')
