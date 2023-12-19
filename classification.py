import warnings
import argparse
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--criterion", type=str, required=False, default="gini")
parser.add_argument("--max_depth", type=int, required=False, default=2)
parser.add_argument("--min_samples_split", type=int, required=False, default=2)
parser.add_argument("--min_samples_leaf", type=float, required=False, default=2)
args = parser.parse_args()
print(f"Arguments: {args}")


def get_metrics(true, predicted):
    accuracy = accuracy_score(true, predicted)
    f1 = f1_score(true, predicted, average="weighted")
    recall = recall_score(true, predicted, average="weighted")
    precision = precision_score(true, predicted, average="weighted")
    return accuracy, f1, precision, recall


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(101)

    try:
        iris_data = load_iris()
        print(f"Loaded IRIS")
    except Exception as e:
        logger.exception(
            "IRIS dataset not found, check import statement %s", e
        )

    X = iris_data.data
    y = iris_data.target

    train_X, test_X, train_Y, test_Y = train_test_split(
        X, y, test_size=0.3
    )

    enc = LabelEncoder()
    train_Y = enc.fit_transform(train_Y)

    criterion = args.criterion
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split
    min_samples_leaf = args.min_samples_leaf

    exp = mlflow.set_experiment(experiment_name="Experiment_3")

    with mlflow.start_run(experiment_id=exp.experiment_id):
        model = DecisionTreeClassifier(criterion=criterion,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       random_state=42)
        model.fit(train_X, train_Y)

        preds = model.predict(test_X)

        (acc, f1, precision, recall) = get_metrics(test_Y, preds)

        print(f"\nAccuracy: {acc} \nF1 {f1} \nPrecision: {precision} \nRecall: {recall}")

        params = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }

        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "ClassificationModel")

    run = mlflow.last_active_run()
    print(f"Last Active Run Name: {run.info.run_name}")
    print(f"Last Active Run ID: {run.info.run_id}")



