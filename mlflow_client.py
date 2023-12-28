import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

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


# Search Experiments function
experiments = client.search_experiments(
    view_type=ViewType.ALL,
    order_by=["experiment_id ASC"]
)
for exp in experiments:
    print(f"Experiment Name: {exp.name} AND Experiment ID: {exp.experiment_id}")


'''Create a Run'''
run_1 = client.create_run(
    experiment_id=experiment_id,
    tags={"runVersion": "1.0", "runNumber": "1"},
    run_name="run_1 from client"
)
run_2 = client.create_run(
    experiment_id=experiment_id,
    tags={"runVersion": "2.0", "runNumber": "2"},
    run_name="run_2 from client"
)
print(f"Run Name: {run_1.info.run_name}")
print(f"Run Tags: {run_1.data.tags}")
print(f"Run ID: {run_1.info.run_id}")
print(f"Run Status: {run_1.info.status}")

'''Get Run'''
# run_1 = client.get_run(run_1.info.run_id)