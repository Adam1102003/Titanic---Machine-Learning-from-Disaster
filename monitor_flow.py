import json
import os
from datetime import datetime

import duckdb
import joblib
import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from prefect import flow, get_run_logger, task

load_dotenv()

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB    = os.getenv("MOTHERDUCK_DB", "titanic_db")
BEST_MODEL_FILE  = "metrics/best_model.json"
MD_CONN_STR      = f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}"

CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Deck", "AgeBand", "FareBand"]
NUMERICAL_FEATURES   = [
    "Pclass", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "IsSmallFamily", "IsLargeFamily",
    "IsChild", "FarePerPerson", "Age*Class", "Fare*Class",
]
PREDICTION_COL = "prediction"
TARGET         = "Survived"


# ── Task 1: Extract ────────────────────────────────────────

@task(name="Extract Data from MotherDuck", retries=2, retry_delay_seconds=5)
def extract() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_run_logger()
    logger.info("🦆 Connecting to MotherDuck...")

    conn = duckdb.connect(MD_CONN_STR)
    conn.execute(f"USE {MOTHERDUCK_DB}")

    current_df   = conn.execute("SELECT * FROM test_passengers").df()
    reference_df = conn.execute("SELECT * FROM train_reference").df()
    conn.close()

    logger.info(f"✅ Current data   : {len(current_df)} rows")
    logger.info(f"✅ Reference data : {len(reference_df)} rows")
    return current_df, reference_df


# ── Task 2: Transform ──────────────────────────────────────

@task(name="Transform & Encode")
def transform(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    logger = get_run_logger()

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        for col in ["Name", "Cabin", "Ticket"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        cat_map = {
            "Sex":      {"male": 1, "female": 0},
            "Embarked": {"C": 0, "Q": 1, "S": 2},
            "AgeBand":  {"Child": 0, "Teen": 1, "YoungAdult": 2,
                         "Adult": 3, "Senior": 4},
            "FareBand": {"Low": 0, "Mid": 1, "High": 2, "VeryHigh": 3},
            "Title":    {"Mr": 1, "Miss": 2, "Mrs": 3,
                         "Master": 4, "Rare": 5},
            "Deck":     {d: i for i, d in enumerate(
                ["Unknown", "A", "B", "C", "D", "E", "F", "G", "T"]
            )},
        }
        for col, mapping in cat_map.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        df = df.fillna(df.median(numeric_only=True))
        return df

    passenger_ids = None
    if "PassengerId" in current_df.columns:
        passenger_ids = current_df["PassengerId"].copy()
        current_df    = current_df.drop(columns=["PassengerId"])

    if "PassengerId" in reference_df.columns:
        reference_df = reference_df.drop(columns=["PassengerId"])

    current_df   = encode(current_df)
    reference_df = encode(reference_df)

    logger.info(f"✅ Transform complete — current: {current_df.shape}")
    return current_df, reference_df, passenger_ids


# ── Task 3: Load Model ─────────────────────────────────────

@task(name="Load Production Model")
def load_model() -> tuple:
    logger = get_run_logger()

    with open(BEST_MODEL_FILE) as f:
        info = json.load(f)

    model = joblib.load(info["best_model_path"])
    logger.info(f"✅ Loaded : {info['best_model']} (accuracy: {info['accuracy']})")
    return model, info["best_model"], info["run_id"]


# ── Task 4: Predict ────────────────────────────────────────

@task(name="Run Predictions")
def predict(
    model,
    model_name: str,
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    passenger_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_run_logger()

    feature_cols = [
        c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        if c in current_df.columns
    ]

    # Predict on current (test) data
    cur_preds = model.predict(current_df[feature_cols])
    cur_probs = model.predict_proba(current_df[feature_cols])[:, 1]

    current_df = current_df.copy()
    current_df[PREDICTION_COL]         = cur_preds.astype(int)
    current_df["survival_probability"] = cur_probs.round(4)
    current_df["survived_label"]       = current_df[PREDICTION_COL].map(
        {1: "Survived", 0: "Did not survive"}
    )
    current_df["model_used"] = model_name

    if passenger_ids is not None:
        current_df.insert(0, "PassengerId", passenger_ids.values)

    # Predict on reference (train) data for Evidently comparison
    ref_cols  = [c for c in feature_cols if c in reference_df.columns]
    ref_preds = model.predict(reference_df[ref_cols])
    ref_probs = model.predict_proba(reference_df[ref_cols])[:, 1]

    reference_df = reference_df.copy()
    reference_df[PREDICTION_COL]         = ref_preds.astype(int)
    reference_df["survival_probability"] = ref_probs.round(4)

    survived     = int(current_df[PREDICTION_COL].sum())
    not_survived = len(current_df) - survived
    logger.info(f"✅ Survived: {survived} | Not survived: {not_survived}")

    return current_df, reference_df


# ── Task 5: Monitor with Evidently ─────────────────────────

@task(name="Run Evidently Monitoring")
def monitor(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("📊 Running Evidently monitoring...")

    drop_cols = [
        "PassengerId", "Name", "Cabin", "Ticket",
        "survived_label", "model_used",
    ]
    keep_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [PREDICTION_COL]

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        cols = [c for c in keep_cols if c in df.columns]
        return df[cols].fillna(0)

    ref_clean = prep(reference_df)
    cur_clean = prep(current_df)

    common    = [c for c in ref_clean.columns if c in cur_clean.columns]
    ref_clean = ref_clean[common]
    cur_clean = cur_clean[common]

    # ✅ new evidently 0.7.x API
    report = Report(metrics=[
    DataDriftPreset(),
    DataSummaryPreset(),
])
    my_eval = report.run(
        reference_data=ref_clean,
        current_data=cur_clean,
    )

    # Extract results
    timestamp = datetime.utcnow().isoformat()
    rows = []

    try:
        result_dict = my_eval.dict() if hasattr(my_eval, "dict") else {}
        for key, value in result_dict.items():
            if isinstance(value, (int, float, bool, str)):
                rows.append({
                    "run_id":       run_id,
                    "timestamp":    timestamp,
                    "metric_group": "evidently",
                    "metric_name":  key,
                    "metric_value": str(value),
                })
    except Exception as e:
        logger.warning(f"⚠️ Could not extract metrics dict: {e}")

    # Always save at minimum a row confirming the run happened
    if not rows:
        rows.append({
            "run_id":       run_id,
            "timestamp":    timestamp,
            "metric_group": "evidently",
            "metric_name":  "status",
            "metric_value": "completed",
        })

    results_df = pd.DataFrame(rows)

    # Save HTML report
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/evidently_{timestamp[:10]}.html"

    try:
        my_eval.save_html(report_path)
        logger.info(f"✅ Report saved → {report_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not save HTML: {e}")

    logger.info(f"✅ Metrics rows : {len(results_df)}")
    return results_df



# ── Task 6: Save to MotherDuck ─────────────────────────────

@task(name="Save Results to MotherDuck", retries=2, retry_delay_seconds=5)
def save_results(
    predictions_df: pd.DataFrame,
    monitoring_df: pd.DataFrame,
) -> None:
    logger = get_run_logger()

    conn = duckdb.connect(MD_CONN_STR)
    conn.execute(f"USE {MOTHERDUCK_DB}")

    # Predictions
    conn.execute("DROP TABLE IF EXISTS predictions")
    conn.execute("CREATE TABLE predictions AS SELECT * FROM predictions_df")
    pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    logger.info(f"✅ Saved {pred_count} predictions")

    # Monitoring
    conn.execute("DROP TABLE IF EXISTS monitoring_results")
    conn.execute("CREATE TABLE monitoring_results AS SELECT * FROM monitoring_df")
    mon_count = conn.execute("SELECT COUNT(*) FROM monitoring_results").fetchone()[0]
    logger.info(f"✅ Saved {mon_count} monitoring metric rows")

    # Summary
    summary = conn.execute("""
        SELECT survived_label,
               COUNT(*)                            AS total,
               ROUND(AVG(survival_probability), 4) AS avg_prob
        FROM predictions
        GROUP BY survived_label
    """).df()
    logger.info(f"\n{summary.to_string(index=False)}")

    conn.close()


# ── Flow ───────────────────────────────────────────────────

@flow(
    name="Titanic Monitoring Flow",
    description="Batch predict + Evidently monitoring → MotherDuck",
)
def monitoring_flow():
    logger = get_run_logger()
    logger.info("🚢 Starting Titanic Monitoring Flow")

    current_df, reference_df        = extract()
    current_df, reference_df, ids   = transform(current_df, reference_df)
    model, model_name, run_id       = load_model()
    predictions_df, ref_with_preds  = predict(model, model_name,
                                               current_df, reference_df, ids)
    monitoring_df                   = monitor(predictions_df,
                                              ref_with_preds, run_id)
    save_results(predictions_df, monitoring_df)

    logger.info("🎉 Monitoring flow complete!")


if __name__ == "__main__":
    monitoring_flow()