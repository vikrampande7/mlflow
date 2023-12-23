## MLFlow

Mlflow server tracking
- Command to run in Terminal - mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 5000
- Only artifacts are stored locally, metrics, parameters are logged in database and shown in MLflow UI
- Command to be added in script - mlflow.set_tracking_uri(uri="http://127.0.0.1:5000") 

MLFlow Signature
- Signatures can be added manually by defining input schema, output schema and input examples or other things if required.
- Signatures can also be added with infer_signature function by providing data examples (test data) and predicted values. Input examples are data columns and data values (test data preferred) in this case. 
- When save_model is used instead of log_model, artifacts are stored locally and not in mlflow server.