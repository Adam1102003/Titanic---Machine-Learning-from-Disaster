import json
import os

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv

from src.preprocess import get_X_y  # reuse your existing preprocessor

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

BEST_MODEL_FILE    = "metrics/best_model.json"
PROCESSED_DATA     = "data/processed/train_featured.csv"


def load_production_model():
    """Load the best model based on promotion info from metrics/best_model.json."""
    if not os.path.exists(BEST_MODEL_FILE):
        raise FileNotFoundError(
            "metrics/best_model.json not found. Run promote_model.py first."
        )

    with open(BEST_MODEL_FILE) as f:
        info = json.load(f)

    model_path = info["best_model_path"]
    model_name = info["best_model"]
    accuracy   = info["accuracy"]
    run_id     = info["run_id"]

    print(f"✅ Loading Production model : {model_name}")
    print(f"   Accuracy                 : {accuracy:.4f}")
    print(f"   Run ID                   : {run_id}")
    print(f"   Path                     : {model_path}")

    model = joblib.load(model_path)

    return model, model_name


def predict(input_data: pd.DataFrame) -> pd.DataFrame:
    """Load Production model and return predictions."""
    model, model_name = load_production_model()

    predictions   = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]

    results = input_data.copy()
    results["prediction"]           = predictions
    results["survival_probability"] = probabilities.round(4)
    results["survived_label"]       = results["prediction"].map(
        {1: "✅ Survived", 0: "❌ Did not survive"}
    )

    print(f"\n📊 Predictions using [{model_name}]:\n")
    print(
        results[["prediction", "survival_probability", "survived_label"]]
        .to_string(index=False)
    )

    return results


if __name__ == "__main__":
    # Load real processed data — same features the model was trained on
    print(f"📂 Loading processed data from: {PROCESSED_DATA}")
    df = pd.read_csv(PROCESSED_DATA)

    # get_X_y drops the target column and returns only features
    X, y = get_X_y(df)

    # Take 5 sample rows for prediction
    sample = X.sample(5, random_state=42)

    print(f"\n🔍 Sample input features:\n")
    print(sample.to_string(index=False))
    print()

    results = predict(sample)

    # Show actual vs predicted
    actual = y.loc[sample.index].values
    results["actual"] = actual
    results["correct"] = results["prediction"] == results["actual"]

    print(f"\nAccuracy on sample: {results['correct'].mean():.0%}")