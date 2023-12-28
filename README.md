## MLFlow

Mlflow server tracking
- Command to run in Terminal - mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 5000
- Only artifacts are stored locally, metrics, parameters are logged in database and shown in MLflow UI
- Command to be added in script - mlflow.set_tracking_uri(uri="http://127.0.0.1:5000") 

MLFlow Signature
- Signatures can be added manually by defining input schema, output schema and input examples or other things if required.
- Signatures can also be added with infer_signature function by providing data examples (test data) and predicted values. Input examples are data columns and data values (test data preferred) in this case. 
- When save_model is used instead of log_model, artifacts are stored locally and not in mlflow server.

MLFlow Evaluation
- mlflow.evaluate() function is used, it takes several parameters as arguments.
- It creates Explainer Graphs, scatter plots, box plots etc.
- With the help of evaluate, we can compare different runs and expriments.
- It can also work with custom artifacts and metrics.

MLFlow Registry
- Register models through UI / API/ MLflow Client.
- We can register model using UI only after model logging and experimentation is done 
- We can register model using functions such as log_model() and register_model() 
- log_model() : passing registered_model_name parameter (model gets registered while logging)
- register_model(): model gets registered after logging process 
- We can load the registered model and make predictions. 

MLFlow Project
- Can be used to run the Machine Learning code in different systems without worrying about creating environments and dependencies.
- MLproject file contains: name, environment and entrypoints
- The environments are system environment, virtual environment, conda environment and docker environment.
- Virtual environment: we can specify python virtual environment in MLproject file as follows - "python_env: files/config?python_env.yaml"
- Conda environment: we can specify a conda environment config file in MLproject file - first run "conda env export --name env_name > conda.yaml" in terminal. Then add the path of this file in MLproject file - conda_env: conda.yaml
- Docker Environment: We can set up the docker environment as required.
- Running MLproject file - we can run this file using CLI or mlflow API
  - CLI Command: It mlflow run [OPTIONS] URI: mlflow run --entry-point ElasticNet -P alpha=0.5 -P l1_ratio=0.5 --experiment-name "experiment_mlproject" .
  - ![mlflow_mlproject_cli_command_terminal.png](outputs%2Fmlflow_mlproject_cli_command_terminal.png)
  - ![mlflow_mlproject_cli_ui_output.png](outputs%2Fmlflow_mlproject_cli_ui_output.png)
  - API/Script: Create a new script or use existing if any. Use mlflow.projects.run() function - we can define all the parameters in it same as CLI command.

MLFlow Client
- Can be used as lower level API that directly translates MLflow REST API Calls
- We can create client and with client we can create an experiment, get an experiment, rename, delete, restore experiments.
- search_experiments is useful with Mlflow client.
- We can log the information - params, metrics with client. We can also rename, get and set various runs with client.
- We can get the metrics, history of previous run with get run method.