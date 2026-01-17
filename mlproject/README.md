# Smart Pricing & Fraud Detection System

This project is a production-ready Machine Learning application that provides a REST API for:
1.  **Predicting optimal product prices** using a regression model.
2.  **Detecting suspicious or fraudulent pricing patterns** using an anomaly detection model.

The system is built with Python, scikit-learn, and FastAPI, and includes a responsive frontend for interaction.

## Features

-   **Price Prediction**: Employs a `RandomForestRegressor` to estimate product prices.
-   **Fraud Detection**: Uses an `IsolationForest` model to identify anomalous listings.
-   **RESTful API**: Built with FastAPI for high performance and automatic documentation.
-   **Responsive Frontend**: A simple, user-friendly UI built with HTML, CSS, and JavaScript.
-   **Modular Codebase**: Follows a clean, production-grade project structure.
-   **Robust Logging**: Logs important events and errors to `logs/app.log`.
-   **Reproducible Environment**: All dependencies are listed in `requirements.txt`.

## Project Structure

```
mlproject/
├── data/
│   ├── raw/products.csv
│   └── processed/
├── notebooks/eda.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_price_model.py
│   ├── train_fraud_model.py
│   ├── predict.py
│   └── utils.py
├── models/
│   ├── price_model.pkl
│   └── fraud_model.pkl
├── app/
│   ├── main.py
│   └── schemas.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── logs/app.log
├── requirements.txt
└── README.md
```

## Setup and Installation

### Prerequisites

-   Python 3.9+
-   `pip` package manager

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mlproject
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## How to Run the Project

### 1. Train the Models

Before running the application, you need to train the machine learning models. The training scripts will process the raw data and save the trained models and preprocessors to the `models/` directory.

Run the training scripts from the root `mlproject` directory:

```bash
python src/train_price_model.py
python src/train_fraud_model.py
```

After running these scripts, you should see the following files in the `models/` directory:
-   `price_model.pkl`
-   `price_preprocessor.pkl`
-   `fraud_model.pkl`
-   `fraud_preprocessor.pkl`

### 2. Start the FastAPI Server

Once the models are trained, you can start the API server using Uvicorn.

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will be running at `http://localhost:8000`.

### 3. Access the Application

-   **Frontend UI**: Open your web browser and navigate to `http://localhost:8000`.
-   **API Documentation**: To see the auto-generated API documentation (Swagger UI), go to `http://localhost:8000/docs`.

## API Endpoints

The API provides two main endpoints for predictions:

### POST `/predict/price`

Predicts the price of a product.
-   **Request Body**:
    ```json
    {
      "brand": "string",
      "category": "string",
      "material": "string",
      "rating": float,
      "transactions": int
    }
    ```
-   **Response**:
    ```json
    {
      "predicted_price": float
    }
    ```

### POST `/predict/fraud`

Detects if a product listing is potentially fraudulent.
-   **Request Body**:
    ```json
    {
      "brand": "string",
      "category": "string",
      "material": "string",
      "rating": float,
      "transactions": int
    }
    ```
-   **Response**:
    ```json
    {
      "is_fraud": boolean
    }
    ```
---

