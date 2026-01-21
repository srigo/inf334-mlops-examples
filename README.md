# Credit Risk Analysis MLOps Project

This project demonstrates a complete MLOps pipeline for a Credit Risk Classification application. It uses **FastAPI** for serving predictions, **MLflow** for experiment tracking and model registry, and **Docker** for containerization.

## Project Structure

```
inf334-mlops/
├── docker-compose.yml       # Defines services (mlflow-server, app-service)
├── mlruns/                  # Shared volume for MLflow artifacts (models, metrics)
├── cred_analysis_app/       # Main application source code
│   ├── Dockerfile           # Container definition for the app
│   ├── requirements.txt     # Python dependencies
│   ├── data/                # Dataset (german_credit_data.csv) and production logs
│   └── src/
│       ├── train.py         # Training script (loads data, trains model, logs to MLflow)
│       └── api.py           # FastAPI app (loads model from MLflow, serves predictions)

```

## How to Run

### Prerequisites
- Docker and Docker Compose installed on your machine.

### Execution Instructions

1.  **Start the Services**
    Build and start the MLflow server and the application container.
    ```bash
    docker compose up --build -d
    ```
    - **MLflow UI**: http://localhost:5001
    - **API Endpoint**: http://localhost:8001

2.  **Train the Model**
    > [!IMPORTANT]
    > **You must run the training script inside the container.**
    > This ensures the model is registered with the correct internal network paths (`http://mlflow-server:5000`) so the API can find it.

    ```bash
    docker compose run --rm app-service python src/train.py
    ```
    *Wait for the script to finish. You should see "Model logged to MLflow."*

3.  **Restart the Application**
    The API loads the model only on startup. Since a new model was just trained, restart the service to pick it up.
    ```bash
    docker compose restart app-service
    ```

4.  **Test the API**
    Send a sample request to the prediction endpoint.

    ```bash
    curl -X POST "http://localhost:8001/predict" \
         -H "Content-Type: application/json" \
         -d '{
               "checking_status": "A11",
               "duration": 6,
               "credit_history": "A34",
               "purpose": "A43",
               "credit_amount": 1169,
               "savings_status": "A65",
               "employment": "A75",
               "installment_commitment": 4,
               "personal_status": "A93",
               "other_parties": "A101",
               "residence_since": 4,
               "property_magnitude": "A121",
               "age": 67,
               "other_payment_plans": "A143",
               "housing": "A152",
               "existing_credits": 2,
               "job": "A173",
               "num_dependents": 1,
               "own_telephone": "A192",
               "foreign_worker": "A201"
             }'
    ```

    **Expected Output:**
    ```json
    {"risk_prediction": 1, "class": "Good Risk"}
    ```

### Local Development (Optional)
If you wish to run locally (without Docker) using `uv`:

1.  `cd cred_analysis_app`
2.  `uv venv` and `source .venv/bin/activate`
3.  `uv pip install -r requirements.txt`
4.  `export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"`
5.  `python src/train.py`
6.  `uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload`
