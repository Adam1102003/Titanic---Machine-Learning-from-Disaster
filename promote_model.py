import json
import os
import pickle

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

SCORES_FILE = "metrics/scores.json"
MODELS_DIR  = "models"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()


def get_best_model_from_scores() -> tuple[str, float]:
    """Read metrics/scores.json and return the best model name and accuracy."""
    with open(SCORES_FILE) as f:
        scores = json.load(f)

    print("\n📊 Model Scores:")
    for name, acc in scores.items():
        print(f"   {name:<30} accuracy: {acc:.4f}")

    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    return best_model, best_score


def get_run_id_for_model(model_name: str) -> str | None:
    """Find the MLflow run ID for a given model name by searching experiments."""
    experiment = client.get_experiment_by_name("titanic-training-pipeline")
    if not experiment:
        print("[!] Experiment 'titanic-training-pipeline' not found on DagsHub.")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model_name = '{model_name}'",
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if runs:
        return runs[0].info.run_id
    return None


def promote_best_model() -> None:
    """Find best model, log it as Production in MLflow tags, save best model path."""

    best_model_name, best_score = get_best_model_from_scores()

    print(f"\n🏆 Best Model : {best_model_name}")
    print(f"   Accuracy   : {best_score:.4f}")

    # Find the MLflow run for the best model
    run_id = get_run_id_for_model(best_model_name)

    if run_id:
        # Tag the best run as Production using MLflow run tags
        client.set_tag(run_id, "stage", "Production")
        client.set_tag(run_id, "promoted_model", best_model_name)
        client.set_tag(run_id, "accuracy", str(best_score))
        print(f"\n✅ Tagged run {run_id} as Production on DagsHub")
    else:
        print(f"\n[!] Could not find MLflow run for '{best_model_name}' — tagging skipped.")

    # Save best model info locally for predict.py to use
    best_model_path = os.path.join(MODELS_DIR, f"{best_model_name}.pkl")

    promotion_info = {
        "best_model":      best_model_name,
        "best_model_path": best_model_path,
        "accuracy":        best_score,
        "run_id":          run_id,
        "stage":           "Production",
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/best_model.json", "w") as f:
        json.dump(promotion_info, f, indent=2)

    print(f"\nPromotion info saved to metrics/best_model.json")
    print(f"   Model file : {best_model_path}")
    print(f"   Run ID     : {run_id}")


if __name__ == "__main__":
    promote_best_model()