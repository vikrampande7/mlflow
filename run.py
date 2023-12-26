import mlflow
parameters = {
    "alpha": 0.3,
    "l1_ratio": 0.3
}
experiment_name = "experimet_mlproject"
entry_point = "ElasticNet"
mlflow.projects.run(
    uri=".",
    parameters=parameters,
    entry_point=entry_point,
    experiment_name=experiment_name
)