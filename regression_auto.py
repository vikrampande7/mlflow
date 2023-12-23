# Import Libraries
import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get Arguments from Command Line
parser = argparse.ArgumentParser()  # ArgumentParser object
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()
print(f"Arguments: {args}")


# Evaluation
def get_metrics(true, predicted):
    rmse = np.sqrt(mean_squared_error(y_true=true, y_pred=predicted))
    mae = mean_absolute_error(y_true=true, y_pred=predicted)
    r2 = r2_score(y_true=true, y_pred=predicted)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Read Data
    try:
        data = pd.read_csv('red-wine-quality.csv')
        print(f"Data shape: {data.shape}")
    except Exception as e:
        logger.exception(
            "Data file not found, check the path again. Error: %s", e
        )

    # Train Test Split
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)
    print(f"Train data shape: {train_data.shape} & Test data shape: {test_data.shape}")

    train_X = train_data.drop(['quality'], axis=1)
    test_X = test_data.drop(['quality'], axis=1)
    train_Y = train_data['quality']
    test_Y = test_data['quality']

    # Params
    alpha = args.alpha
    l1_ratio = args.l1_ratio

    exp = mlflow.set_experiment(experiment_name="experiment_signature_inferSignature")

    print(f"Name: {exp.name}")
    print("Experiment ID: {}".format(exp.experiment_id))
    print(f"Artifact Location: {exp.artifact_location}")
    print(f"Tags: {exp.tags}")
    print(f"Lifecycle Stage: {exp.lifecycle_stage}")
    print(f"Creation Timestamp: {exp.creation_time}")

    with mlflow.start_run(experiment_id=exp.experiment_id):
        # # AutoLog with MLFlow
        # mlflow.autolog(
        #     log_input_examples=True
        # )

        # # Library Specified AutoLog - sklearn
        # mlflow.sklearn.autolog(
        #     log_input_examples=True
        # )

        # Library Specified AutoLog - sklearn - Signature
        mlflow.sklearn.autolog(
            log_input_examples=False,
            log_model_signatures=False,
            log_models=False
        )

        # Create Model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_X, train_Y)

        preds = model.predict(test_X)

        (rmse, mae, r2) = get_metrics(test_Y, preds)

        print(f"Model Results with Alpha {alpha} & l1_ratio {l1_ratio}")
        print(f" RMSE: {rmse}\n MAE: {mae},\n R2_Score: {r2}")
        #
        # ##### For Model Signature ####
        # input_data = [
        #     {"name": "fixed acidity", "type": "double"},
        #     {"name": "volatile acidity", "type": "double"},
        #     {"name": "citric acid", "type": "double"},
        #     {"name": "residual sugar", "type": "double"},
        #     {"name": "chlorides", "type": "double"},
        #     {"name": "free sulfur dioxide", "type": "double"},
        #     {"name": "total sulfur dioxide", "type": "double"},
        #     {"name": "density", "type": "double"},
        #     {"name": "pH", "type": "double"},
        #     {"name": "sulphates", "type": "double"},
        #     {"name": "alcohol", "type": "double"},
        #     {"name": "quality", "type": "double"}
        # ]
        #
        # output_data = [{'type': 'long'}]
        #
        # input_example = {
        #     "fixed acidity": np.array([7.2, 7.5, 7.0, 6.8, 6.9]),
        #     "volatile acidity": np.array([0.35, 0.3, 0.28, 0.38, 0.25]),
        #     "citric acid": np.array([0.45, 0.5, 0.55, 0.4, 0.42]),
        #     "residual sugar": np.array([8.5, 9.0, 8.2, 7.8, 8.1]),
        #     "chlorides": np.array([0.045, 0.04, 0.035, 0.05, 0.042]),
        #     "free sulfur dioxide": np.array([30, 35, 40, 28, 32]),
        #     "total sulfur dioxide": np.array([120, 125, 130, 115, 110]),
        #     "density": np.array([0.997, 0.996, 0.995, 0.998, 0.994]),
        #     "pH": np.array([3.2, 3.1, 3.0, 3.3, 3.2]),
        #     "sulphates": np.array([0.65, 0.7, 0.68, 0.72, 0.62]),
        #     "alcohol": np.array([9.2, 9.5, 9.0, 9.8, 9.4]),
        #     "quality": np.array([6, 7, 6, 8, 7])
        # }
        #
        # input_schema = Schema([ColSpec(col["type"], col["name"]) for col in input_data])
        # output_schema = Schema([ColSpec(col["type"]) for col in output_data])
        #
        # signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # infer signature function
        signature = infer_signature(test_X, preds)
        input_example = {
            'columns': np.array(test_X.columns),
            'data': np.array(test_X.values)
        }

        mlflow.log_artifact("red-wine-quality.csv")
        # mlflow.sklearn.log_model(model, 'signatureModel', signature=signature, input_example=input_example)
        mlflow.sklearn.save_model(model, 'signatureModel', signature=signature, input_example=input_example)