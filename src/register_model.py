import os

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

load_dotenv()


def promote_to_production(model_name: str) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    # Find best run by accuracy
    experiment = client.get_experiment_by_name("titanic-training-pipeline")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model_name = '{model_name}'",
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if not runs:
        print(f"No runs found for: {model_name}")
        return

    best_run = runs[0]
    print(f"[✓] Best run accuracy: {best_run.data.metrics['accuracy']:.4f}")

    registered_name = f"titanic-{model_name}"

    try:
        client.get_registered_model(registered_name)
    except RestException as exc:
        error_text = getattr(exc, 'message', str(exc))
        if 'unsupported endpoint' in error_text.lower():
            print(
                "[!] Model registry is not supported by the configured MLflow tracking server."
            )
            print(
                "    This backend cannot create or promote registered models."
            )
            print(
                "    Use local model artifacts or a registry-enabled MLflow server instead."
            )
            return
        print(f"[i] Registered model '{registered_name}' not found. Creating it now...")
        try:
            client.create_registered_model(registered_name)
        except RestException as exc2:
            print(f"[!] Could not create registered model: {exc2}")
            return
    except Exception:
        print(f"[i] Registered model '{registered_name}' not found. Creating it now...")
        try:
            client.create_registered_model(registered_name)
        except RestException as exc2:
            print(f"[!] Could not create registered model: {exc2}")
            return

    try:
        versions = client.get_latest_versions(registered_name)
    except RestException as exc:
        print(f"[!] Failed to get model versions for '{registered_name}': {exc}")
        return
    except Exception as exc:
        print(f"[!] Failed to get model versions for '{registered_name}': {exc}")
        return

    if not versions:
        print(f"No registered versions found for '{registered_name}'.")
        return

    latest_version = versions[-1].version

    client.transition_model_version_stage(
        name=registered_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"[✓] '{registered_name}' v{latest_version} → Production")


if __name__ == "__main__":
    for model in ["random_forest", "logistic_regression", "gradient_boosting"]:
        promote_to_production(model)