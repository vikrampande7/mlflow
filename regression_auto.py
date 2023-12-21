# Import Libraries
import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
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

    exp = mlflow.set_experiment(experiment_name="experiment_auto_sklearn")

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

        # Library Specified AutoLog - sklearn
        mlflow.sklearn.autolog(
            log_input_examples=True
        )

        # Create Model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_X, train_Y)

        preds = model.predict(test_X)

        (rmse, mae, r2) = get_metrics(test_Y, preds)

        print(f"Model Results with Alpha {alpha} & l1_ratio {l1_ratio}")
        print(f" RMSE: {rmse}\n MAE: {mae},\n R2_Score: {r2}")

        mlflow.log_artifact("red-wine-quality.csv")
