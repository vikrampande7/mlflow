## MLFlow

![mlflow-logo](/outputs/mlflow_logo.png){: .align-center}

##### Steps to Run
1. Clone the git repository
2. Install all the required dependencies (refer conda.yaml) 
3. Activate the environment and run the code: run following command in terminal
4. Check the experiment and run status in mlflow UI. Run the follwing command in terminal.
```
git clone https://github.com/vikrampande7/mlflow.git
cd mlflow
python run.py
mlflow ui
```


##### Following are the modules for the project:
- ```data```: data for classification and regression with train-test split.
- ```mlflow-artifacts```: experiment of classification and saved artifact externally.
- ```mlruns```: contains the experiments and runs.
- ```outputs```: contains the images of experiments and UI.
- ```red-wine-data```: logged data for regression.
- ```signatureModel```: experimentation with Model Signature.
- ```classification.py```: contains the ML classification code.
- ```cli_commands```: Contains mlflow CLI commands.
- ```client_tracking.py```: Contains the script of tracking with Mlflow Client.
- ```conda.yaml```: Conda environment configuration file.
- ```mlflow.db```: sqlite db
- ```mlflow_client.py```: Experimentation with mlflow client.
- ```MLproject```: MLproject configuration file.
- ```mlserver_commands.txt```
- ```registerExtModel.py```: Experiment to log the external model 
- ```regression.py```: Contains the ML Regression code.
- ```regression_auto.py```: Contains the ML regression code with autologging function.
- ```regression_simplified.py```: Simplified version of ```regression.py```
- ```run.py```: script to run the code with entry_point

**Mlflow server tracking**
- Command to run in Terminal - mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 5000
- Only artifacts are stored locally, metrics, parameters are logged in database and shown in MLflow UI
- Command to be added in script - mlflow.set_tracking_uri(uri="http://127.0.0.1:5000") 

**MLFlow Signature**
- Signatures can be added manually by defining input schema, output schema and input examples or other things if required.
- Signatures can also be added with infer_signature function by providing data examples (test data) and predicted values. Input examples are data columns and data values (test data preferred) in this case. 
- When save_model is used instead of log_model, artifacts are stored locally and not in mlflow server.

**MLFlow Evaluation**
- mlflow.evaluate() function is used, it takes several parameters as arguments.
- It creates Explainer Graphs, scatter plots, box plots etc.
- With the help of evaluate, we can compare different runs and expriments.
- It can also work with custom artifacts and metrics.

**MLFlow Registry**
- Register models through UI / API/ MLflow Client.
- We can register model using UI only after model logging and experimentation is done 
- We can register model using functions such as log_model() and register_model() 
- log_model() : passing registered_model_name parameter (model gets registered while logging)
- register_model(): model gets registered after logging process 
- We can load the registered model and make predictions. 

**MLFlow Project**
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

**MLFlow Client**
- Can be used as lower level API that directly translates MLflow REST API Calls
- We can create client and with client we can create an experiment, get an experiment, rename, delete, restore experiments.
- search_experiments is useful with Mlflow client.
- We can log the information - params, metrics with client. We can also rename, get and set various runs with client.
- We can get the metrics, history of previous run with get run method.
- We can also delete the run by providing run id, restore run and search runs same as experiments.
- We can implement model versioning and management with MLFlow client. 

**MLFlow CLI Commands**:
- mlflow doctor: examines the system and shows metadata
- mlflow doctor mask-envs: to hide sensitive information
- mlflow artifact commands - download, list, log-artifact, log-artifacts
- Database management command - mlflow db - upgrade the database schema to the latest supported version.
- Experiments: Create, rename, delete experiments, etc.


**MLflow with AWS**

Sample ML model training, tracking and inference with AWS using AWS CodeCommit, SageMaker, EC2, S3.
1. Create AWS account, new user with and with limited but required access and permissions.
2. Create a new AWS CodeCommit Repo.
3. Create S3 bucket instance to store artifacts.
4. Create mlflow tracking server on AWS EC2.
- Create EC2 instance. Select Ubuntu as OS (or any OS of your choice)
- Run a set of commands to run mlflow on EC2.
- Set the tracking URI on AWS.
5. Clone the CodeCommit repo
- Generate the credentials for CodeCommit (HTTPS)
- Clone the CodeCommit repo to local.
6. Create your required Machine Learning codes say for data preprocessing, training, evaluation, etc.
- Log input, parameters, metrics, models, etc.
- Create MLproject file and set all the variables.
- Create a conda.yaml file.
- Create run.py file to run MLproject file
- Run the code locally to check if everything runs perfectly.
7. Push everything to CodeCommit repo.
8. AWS SageMaker Setup
- Add our Git repo to SageMaker (SageMaker-Notebook-Git Repo)
- Create a notebook instance and set instance type.
- Assign role, permissions. The role should be SageMaker execution role.
- Allow full access of S3 as we store artifacts in S3.
9. Run the codes on Notebook instance in AWS SageMaker.
10. Check the results, compare models and evaluate using MLflow UI set with AWS.
- Register the best model.
11. Deploy the model with build command in terminal.
- ```mlflow sagemaker build-and-push-container```
- Push the imag to ECR, check the docker image in ECR.
12. Deploy and create an endpoint.
- Either using CLI or creating a new script (API)
- Provide endpoint name, URI and necessary configurations.
- run the script.
- Check the output on SageMaker (SageMaker - Inference - endpoints)
13. Use the deployed model for inference. 