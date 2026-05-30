import json
import os

import duckdb
import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task

load_dotenv()

# ── Config ─────────────────────────────────────────────────
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB    = os.getenv("MOTHERDUCK_DB", "titanic_db")
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI")
BEST_MODEL_FILE  = "metrics/best_model.json"

MD_CONN_STR = f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}"


# ── Task 1: Extract ────────────────────────────────────────

@task(name="Extract from MotherDuck", retries=2, retry_delay_seconds=5)
def extract() -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(" Connecting to MotherDuck...")

    conn = duckdb.connect(MD_CONN_STR)
    conn.execute(f"USE {MOTHERDUCK_DB}")

    df = conn.execute("SELECT * FROM test_passengers").df()
    conn.close()

    logger.info(f"✅ Extracted {len(df)} rows from MotherDuck")
    return df


# ── Task 2: Transform ──────────────────────────────────────

@task(name="Transform Data")
def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("🔧 Transforming data...")

    # Drop target column if it exists
    if "Survived" in df.columns:
        df = df.drop(columns=["Survived"])

    # Drop PassengerId — not a feature
    passenger_ids = None
    if "PassengerId" in df.columns:
        passenger_ids = df["PassengerId"].copy()
        df = df.drop(columns=["PassengerId"])

    # Drop any unnamed index columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))

    logger.info(f"✅ Transform complete — shape: {df.shape}")
    return df, passenger_ids


# ── Task 3: Load Model ─────────────────────────────────────

@task(name="Load Model from DagsHub")
def load_model():
    logger = get_run_logger()

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Read best model info from local promotion file
    if not os.path.exists(BEST_MODEL_FILE):
        raise FileNotFoundError(
            "metrics/best_model.json not found — run promote_model.py first"
        )

    with open(BEST_MODEL_FILE) as f:
        info = json.load(f)

    model_name = info["best_model"]
    model_path = info["best_model_path"]
    run_id     = info["run_id"]

    logger.info(f"Loading model   : {model_name}")
    logger.info(f"   Run ID          : {run_id}")
    logger.info(f"   Accuracy        : {info['accuracy']}")

    # Try loading from DagsHub MLflow artifact first
    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("✅ Model loaded from DagsHub MLflow artifact")
    except Exception as e:
        logger.warning(f"⚠️ Could not load from MLflow — falling back to local .pkl: {e}")
        model = joblib.load(model_path)
        logger.info(f"✅ Model loaded from local file: {model_path}")

    return model, model_name


# ── Task 4: Predict ────────────────────────────────────────

@task(name="Run Predictions")
def predict(
    model,
    model_name: str,
    df: pd.DataFrame,
    passenger_ids: pd.Series,
) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Running predictions with [{model_name}]...")

    predictions   = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    results = pd.DataFrame({
        "PassengerId":          passenger_ids.values if passenger_ids is not None
                                else range(len(df)),
        "prediction":           predictions.astype(int),
        "survived_label":       ["Survived" if p == 1 else "Did not survive"
                                 for p in predictions],
        "survival_probability": probabilities.round(4),
        "model_used":           model_name,
    })

    survived     = int(results["prediction"].sum())
    not_survived = len(results) - survived

    logger.info(f"✅ Predictions complete — {len(results)} total")
    logger.info(f"   Survived     : {survived}")
    logger.info(f"   Not survived : {not_survived}")

    return results


# ── Task 5: Save to MotherDuck ─────────────────────────────

@task(name="Save Predictions to MotherDuck", retries=2, retry_delay_seconds=5)
def save_predictions(results: pd.DataFrame) -> None:
    logger = get_run_logger()
    logger.info("💾 Saving predictions to MotherDuck...")

    conn = duckdb.connect(MD_CONN_STR)
    conn.execute(f"USE {MOTHERDUCK_DB}")

    conn.execute("DROP TABLE IF EXISTS predictions")
    conn.execute("""
        CREATE TABLE predictions AS
        SELECT * FROM results
    """)

    count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    # Quick summary
    summary = conn.execute("""
        SELECT
            survived_label,
            COUNT(*)            AS count,
            ROUND(AVG(survival_probability), 4) AS avg_probability
        FROM predictions
        GROUP BY survived_label
        ORDER BY survived_label
    """).df()

    conn.close()

    logger.info(f"Saved {count} predictions to MotherDuck → {MOTHERDUCK_DB}.predictions")
    logger.info(f"\n{summary.to_string(index=False)}")


# ── Flow ───────────────────────────────────────────────────

@flow(
    name="Titanic Batch Prediction",
    description="Extract from MotherDuck → Transform → Load model from DagsHub → Predict → Save to MotherDuck",
)
def batch_prediction_flow():
    logger = get_run_logger()
    logger.info("Starting Titanic Batch Prediction Flow")

    # Step 1 — Extract
    raw_df = extract()

    # Step 2 — Transform
    df, passenger_ids = transform(raw_df)

    # Step 3 — Load model
    model, model_name = load_model()

    # Step 4 — Predict
    results = predict(model, model_name, df, passenger_ids)

    # Step 5 — Save
    save_predictions(results)

    logger.info(" Batch prediction flow complete!")


if __name__ == "__main__":
    batch_prediction_flow()