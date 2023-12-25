# Import Libraries
import os.path
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
from mlflow.models import make_metric
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get Arguments from Command Line
parser = argparse.ArgumentParser()  # ArgumentParser object
parser.add_argument("--alpha", type=float, required=False, default=0.4)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
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

    exp = mlflow.set_experiment(experiment_name="experiment_modelEval")

    with mlflow.start_run(experiment_id=exp.experiment_id):
        # Create Model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_X, train_Y)

        preds = model.predict(test_X)

        (rmse, mae, r2) = get_metrics(test_Y, preds)

        print(f"Model Results with Alpha {alpha} & l1_ratio {l1_ratio}")
        print(f" RMSE: {rmse}\n MAE: {mae},\n R2_Score: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        model_info = mlflow.sklearn.log_model(model, "regModel")

        # # Add a custom metrics for evaluation
        # def squared_diff_plus_half(eval_df, _builtin_metrics):
        #     return np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 0.5) ** 2)
        #
        # def targetsum_divided_by_two(_eval_df, builtin_metrics):
        #     return builtin_metrics["sum_on_target"] / 2
        #
        # squared_diff_plus_half_metric = make_metric(
        #     eval_fn=squared_diff_plus_half,
        #     greater_is_better=False,
        #     name="squared_diff_plus_half"
        # )
        #
        # targetsum_divided_by_two_metric = make_metric(
        #     eval_fn=targetsum_divided_by_two,
        #     greater_is_better=True,
        #     name=targetsum_divided_by_two
        # )
        
        # Add custom Artifact 
        # def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
        #     plt.scatter(eval_df["prediction"], eval_df["target"])
        #     plt.xlabel("Target")
        #     plt.ylabel("Predictions")
        #     plt.title("Scatter Plot of Targets vs Predicted")
        #     plot_path = os.path.join(artifacts_dir, "scatter_plot.png")
        #     plt.savefig(plot_path)
        #     return {"Example_scatter_plot_artifact": plot_path}


        # Model Evaluation -- evaluators=SHAP
        mlflow.evaluate(
            model_info.model_uri,
            test_data,
            targets="quality",
            model_type="regressor",
            evaluators=["default"]
            # custom_metrics=[targetsum_divided_by_two_metric, squared_diff_plus_half_metric]
            #custom_artifacts=[prediction_target_scatter()]
        )
