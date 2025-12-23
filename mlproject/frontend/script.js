document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('product-form');
    const predictPriceBtn = document.getElementById('predict-price-btn');
    const detectFraudBtn = document.getElementById('detect-fraud-btn');
    const resultsContainer = document.getElementById('results-container');
    const priceResult = document.getElementById('price-result');
    const fraudResult = document.getElementById('fraud-result');
    const errorContainer = document.getElementById('error-container');

    // Handle form submission for price prediction
    predictPriceBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        if (!form.checkValidity()) {
            displayError("Please fill out all fields correctly.");
            return;
        }
        await handleApiCall('/predict/price');
    });

    // Handle button click for fraud detection
    detectFraudBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        if (!form.checkValidity()) {
            displayError("Please fill out all fields correctly.");
            return;
        }
        await handleApiCall('/predict/fraud');
    });
    
    // Unified function to handle API calls
    async function handleApiCall(endpoint) {
        clearResults();
        
        const formData = new FormData(form);
        const productData = Object.fromEntries(formData.entries());
        
        // Ensure numeric types are correct
        productData.rating = parseFloat(productData.rating);
        productData.transactions = parseInt(productData.transactions, 10);

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(productData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An unknown error occurred.');
            }

            const result = await response.json();
            displayResult(endpoint, result);

        } catch (error) {
            displayError(error.message);
        }
    }

    function displayResult(endpoint, result) {
        resultsContainer.style.display = 'block';
        if (endpoint.includes('price')) {
            priceResult.innerHTML = `Predicted Price: <strong>$${result.predicted_price.toFixed(2)}</strong>`;
            priceResult.className = 'result-item price-normal';
        } else if (endpoint.includes('fraud')) {
            if (result.is_fraud) {
                fraudResult.innerHTML = `Fraud Status: <strong>Potential Fraud Detected</strong>`;
                fraudResult.className = 'result-item fraud-detected';
            } else {
                fraudResult.innerHTML = `Fraud Status: <strong>No Fraud Detected</strong>`;
                fraudResult.className = 'result-item fraud-not-detected';
            }
        }
    }

    function displayError(message) {
        errorContainer.textContent = `Error: ${message}`;
        errorContainer.style.display = 'block';
    }

    function clearResults() {
        priceResult.innerHTML = '';
        fraudResult.innerHTML = '';
        errorContainer.style.display = 'none';
        resultsContainer.style.display = 'none';
    }
});
