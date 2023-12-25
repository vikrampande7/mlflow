import pickle
import mlflow
import mlflow,sklearn

# filename = "elastic_net.pkl"
filename = "path_to_external_model"
loaded_model = pickle.load(open(filename, "rb"))

mlflow.set_tracking_uri(uri="")
exp = mlflow.set_experiment(experiment_name="exp_register_other_model")

mlflow.start_run()
mlflow.sklearn.log_model(
    loaded_model,
    'outside_model',
    serialization_format='cloudpickle',
    registered_model_name="elasticNet-outside_model"
)
mlflow.end_run()
