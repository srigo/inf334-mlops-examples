import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import fetch_openml

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "credit-risk-classification"
DATA_PATH = "data/german_credit_data.csv"

def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
    else:
        print("Downloading data from OpenML (credit-g)...")
        # version 1 is usually the standard 'German Credit'
        data = fetch_openml(name='credit-g', version=1, as_frame=True)
        df = data.frame
        # Save for future use
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return df

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading data...")
    df = load_data()
    
    # Target column in OpenML credit-g is 'class' ('good', 'bad')
    target_col = 'class'
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Map target to 0/1 if necessary (RandomForest handles strings but metrics might need int)
    # usually 'good' -> 0, 'bad' -> 1 (Risk Class) or vice versa.
    # Let's map 'bad' -> 1 (Risk), 'good' -> 0 (Safe)
    y = y.map({'good': 0, 'bad': 1})

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # Define Model Pipeline
    # Using a simple RandomForest
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Start MLflow Run
    with mlflow.start_run() as run:
        print(f"Starting run: {run.info.run_id}")
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Log Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log Params
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Log Model
        # Using sklearn autolog logic or manual log_model
        signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name="CreditRiskModel"
        )
        
        print("Model logged to MLflow.")

if __name__ == "__main__":
    main()
