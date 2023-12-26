import mlflow
from mlflow import MlflowClient

client = MlflowClient()

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create a new experiment: 900681660077618817
# experiment_id = client.create_experiment(
#     name="experiment_client",
#     tags={"version": "1.0", "priority": "high"}
# )

# Get the existing experiment
experiment_id = "900681660077618817"
experiment = client.get_experiment(experiment_id=experiment_id)
# experiment = client.get_experiment_by_name(name="experiment_client") # Get experiment by name
# client.rename_experiment("experiment_client", "experiment_client_") # Rename experiment
# client.delete_experiment(experiment_id) # Delete experiment
# client.restore_experiment(experiment_id=experiment_id) # Restore deleted experiment

print(f"ExperimentID: {experiment.experiment_id}")
print(f"Experiment Name: {experiment.name}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Tags: {experiment.tags}")
print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
print(f"Creation Time: {experiment.creation_time}")