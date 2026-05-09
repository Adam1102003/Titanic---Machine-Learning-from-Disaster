import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from src.evaluate import evaluate
from src.preprocess import get_X_y
from src.tuning import tune_and_save

load_dotenv()


def setup_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI not set in .env file")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("titanic-training-pipeline")
    print(f"[✓] MLflow tracking URI: {tracking_uri}")


def main() -> None:
    setup_mlflow()

    cfg_data = OmegaConf.load("configs/data/titanic.yaml")
    cfg_training = OmegaConf.load("configs/training/default.yaml")

    train_df = pd.read_csv("data/processed/train_featured.csv")
    X, y = get_X_y(train_df)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg_data.test_size,
        random_state=cfg_data.random_state,
    )

    os.makedirs(cfg_training.models_output_dir, exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    scores = {}

    for model_name in cfg_training.models:
        model_cfg = OmegaConf.load(f"configs/model/{model_name}.yaml")
        output_path = os.path.join(
            cfg_training.models_output_dir, f"{model_name}.pkl"
        )

        with mlflow.start_run(run_name=model_name):

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", cfg_data.test_size)
            mlflow.log_param("random_state", cfg_data.random_state)
            mlflow.log_param("cv_folds", cfg_training.cv_folds)
            mlflow.log_params(
                {f"grid_{k}": str(v) for k, v in dict(model_cfg).items()}
            )

            # Train
            best_pipeline = tune_and_save(
                X_train,
                y_train,
                model_name=model_name,
                model_cfg=model_cfg,
                output_path=output_path,
                cv=cfg_training.cv_folds,
            )

            # Evaluate
            acc = evaluate(best_pipeline, X_val, y_val, model_name)
            scores[model_name] = round(acc, 4)

            # Log metrics
            mlflow.log_metric("accuracy", acc)

            # Log model to MLflow registry
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                artifact_path="model",
            )

            # Log the .pkl as artifact
            mlflow.log_artifact(output_path)

            print(f"[✓] MLflow run logged for '{model_name}'")

    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\n[✓] All done. Scores: {scores}")


if __name__ == "__main__":
    main()