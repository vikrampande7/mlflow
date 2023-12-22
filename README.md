## MLFlow

Mlflow server tracking
- Command to run in Terminal - mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 5000
- Only artifacts are stored locally, metrics, parameters are logged in database and shown in MLflow UI
- Command to be added in script - mlflow.set_tracking_uri(uri="http://127.0.0.1:5000") 
