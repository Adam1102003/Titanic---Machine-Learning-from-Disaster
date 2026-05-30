# 🚢 Titanic — Machine Learning from Disaster

> Lab 0 + Lab 6 + Lab 7 | ITI MLOps Track
> A fully automated and configurable training pipeline for the Kaggle Titanic dataset using scikit-learn, Hydra, DVC, MLflow, DagsHub, FastAPI, Prefect, Evidently AI, MotherDuck, and Streamlit.

---

## 🤔 What is This Project?

This project builds a **Machine Learning Pipeline** to predict whether a passenger survived the Titanic shipwreck based on information like their age, gender, ticket class, and more.

It is based on the famous [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) and is built the **MLOps way**:

- Every step from loading data to saving the model is **automated**
- All settings are **configurable via Hydra** — no hardcoded values in code
- Data and models are **versioned with DVC** and stored on **DagsHub**
- Every training run is **tracked with MLflow** — parameters, metrics, and artifacts
- The **best model is automatically promoted** to Production
- The Production model is **served via a REST API** built with FastAPI
- The API accepts **single or batch requests** and returns predictions with probabilities
- A **Prefect batch job** extracts data from MotherDuck, runs predictions, and monitors drift
- Model and data health is **monitored with Evidently AI** on every batch run
- All results are **stored in MotherDuck** (serverless cloud DuckDB)
- A **Streamlit dashboard** visualizes predictions and drift metrics in real time
- Code is **clean and professionally formatted**
- Project is **versioned with Git** and pushed to GitHub

---

## 🧠 What is a Machine Learning Pipeline?

Think of a pipeline like an **assembly line in a factory**:

```
Raw Data → Clean → Engineer Features → Train → Track → Promote → Serve → Monitor
```

Instead of running each step manually, a pipeline connects all steps so you run one command and everything happens automatically in the right order.

---

## 🏗️ Full System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                        │
│  Raw CSV → Clean → Feature Eng → Train → MLflow → Promote   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                  ONLINE SERVING  (Lab 6)                     │
│          FastAPI REST API  →  /predict  →  JSON              │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│               BATCH MONITORING  (Lab 7)                      │
│  MotherDuck → Prefect → Evidently AI → MotherDuck            │
│                              ↓                               │
│                    Streamlit Dashboard                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Titanic---Machine-Learning-from-Disaster/
│
├── configs/                        ← All Hydra configuration files
│   ├── config.yaml                 ← Main entry point config
│   ├── data/
│   │   └── titanic.yaml            ← Data paths and split settings
│   ├── model/
│   │   ├── random_forest.yaml      ← Random Forest hyperparameters
│   │   ├── logistic_regression.yaml← Logistic Regression hyperparameters
│   │   └── gradient_boosting.yaml  ← Gradient Boosting hyperparameters
│   └── training/
│       └── default.yaml            ← Which models to run, CV folds, output dir
│
├── data/
│   └── raw/                        ← train.csv and test.csv (tracked by DVC)
│
├── models/                         ← Saved .pkl model files (tracked by DVC)
│
├── metrics/
│   ├── scores.json                 ← Accuracy for each trained model
│   └── best_model.json             ← Production model info (name, path, run ID)
│
├── reports/
│   └── evidently_YYYY-MM-DD.html   ← Auto-generated Evidently drift report
│
├── outputs/                        ← Auto-created by Hydra (logs per run)
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── pipeline.log
│           └── .hydra/
│               ├── config.yaml
│               └── overrides.yaml
│
├── src/
│   ├── __init__.py
│   ├── load_data.py                ← Step 1: Load CSV files
│   ├── cleaning.py                 ← Step 2: Fix missing/bad data
│   ├── feature_engineering.py      ← Step 3: Create new useful columns
│   ├── preprocess.py               ← Step 4: Convert data to numbers
│   ├── training.py                 ← Step 5: Build and save model
│   ├── tuning.py                   ← Step 5+: GridSearchCV hyperparameter tuning
│   ├── evaluate.py                 ← Step 6: Print accuracy and report
│   └── stages/
│       ├── load_data.py            ← DVC Stage 1
│       ├── clean.py                ← DVC Stage 2
│       ├── feature_engineering.py  ← DVC Stage 3
│       └── train.py                ← DVC Stage 4 — trains + logs to MLflow
│
├── main.py                         ← 🌐 FastAPI online serving app (Lab 6)
├── promote_model.py                ← 🏆 Finds best model → tags as Production
├── predict.py                      ← 🔮 Offline batch predictions
├── load_to_motherduck.py           ← 🦆 Loads test + train data to MotherDuck (Lab 7)
├── monitor_flow.py                 ← ⚙️  Prefect + Evidently monitoring job (Lab 7)
├── dashboard.py                    ← 📊 Streamlit dashboard (Lab 7)
├── dvc.yaml                        ← DVC pipeline definition (6 stages)
├── dvc.lock                        ← Auto-generated by DVC
├── .env                            ← All credentials (never committed)
├── .dvc/                           ← DVC internals and remote config
├── pipeline.py                     ← 🚀 Hydra pipeline entry point
├── pyproject.toml                  ← Project settings and dependencies
├── uv.lock                         ← Locked dependency versions
└── README.md
```

---

## ⚙️ Setup from Scratch

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### 1. Install uv

```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/Adam1102003/Titanic---Machine-Learning-from-Disaster.git
cd Titanic---Machine-Learning-from-Disaster
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Set DVC credentials

Get your token from [DagsHub](https://dagshub.com) → Settings → Tokens → New Token

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN
```

### 5. Create `.env` file

```bash
# MLflow / DagsHub
MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_USERNAME/YOUR_REPO.mlflow
MLFLOW_TRACKING_USERNAME=YOUR_DAGSHUB_USERNAME
MLFLOW_TRACKING_PASSWORD=YOUR_DAGSHUB_TOKEN

# MotherDuck (Lab 7)
MOTHERDUCK_TOKEN=your_motherduck_token
MOTHERDUCK_DB=titanic_db
```

> ⚠️ Never commit `.env` to Git. It is already listed in `.gitignore`.

### 6. Pull data and models from DagsHub

```bash
dvc pull
```

---

## 🚀 Run Everything — Full End-to-End

```bash
# Step 1 — Run full training pipeline
dvc repro

# Step 2 — Start online serving API (Lab 6)
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Step 3 — Load data into MotherDuck (Lab 7 — run once)
uv run python load_to_motherduck.py

# Step 4 — Run batch prediction + monitoring (Lab 7)
uv run python monitor_flow.py

# Step 5 — Launch Streamlit dashboard (Lab 7)
uv run streamlit run dashboard.py
```

### Run individual DVC stages

```bash
dvc repro train      # train only
dvc repro promote    # promote best model only
dvc repro predict    # offline batch predict only
```

---

## 🔁 DVC Pipeline

### Pipeline Stages

```
+------------+
| load_data  |   Loads train.csv and test.csv
+------------+
      │
   +-------+
   | clean |    Fixes missing values and bad data
   +-------+
      │
+---------------------+
| feature_engineering |  Creates 12 new features
+---------------------+
      │
    +-------+
    | train |    Trains 3 models, logs params/metrics/artifacts to MLflow
    +-------+
      │
  +---------+
  | promote |   Reads scores.json, tags best model as Production
  +---------+
      │
  +---------+
  | predict |   Loads Production model, runs offline batch predictions
  +---------+
```

### Key DVC Commands

| Command | What it does |
|---|---|
| `dvc repro` | Run full pipeline (skips unchanged stages) |
| `dvc push` | Upload data/models to DagsHub |
| `dvc pull` | Download data/models from DagsHub |
| `dvc dag` | Show pipeline graph |
| `dvc status` | Show what has changed locally |
| `dvc metrics show` | Show accuracy scores |

---

## 📊 MLflow Experiment Tracking

Every training run is automatically logged to DagsHub via MLflow.

### What gets logged per run

| Type | What |
|---|---|
| **Parameters** | `model_name`, `test_size`, `random_state`, `cv_folds`, all grid search values |
| **Metrics** | `accuracy` on the validation set |
| **Artifacts** | Trained model (`model/`) and `.pkl` file |
| **Tags** | `stage=Production` on the best model run |

View all runs on DagsHub → **Experiments** tab.

---

## 🏆 Model Promotion

After training, `promote_model.py` automatically:

1. Reads `metrics/scores.json` to find the best model by accuracy
2. Finds the corresponding MLflow run on DagsHub
3. Tags that run with `stage=Production`
4. Saves promotion info to `metrics/best_model.json`

```bash
python promote_model.py
```

Example output:
```
📊 Model Scores:
   random_forest          accuracy: 0.8268
   logistic_regression    accuracy: 0.8156
   gradient_boosting      accuracy: 0.8324

🏆 Best Model  : gradient_boosting
   Accuracy    : 0.8324

✅ Tagged run abc123 as Production on DagsHub
🚀 Promotion info saved to metrics/best_model.json
```

---

## 🌐 Online Serving — FastAPI (Lab 6)

The trained Production model is served as a REST API using FastAPI.

### Start the server

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Expected startup output:
```
[✓] Loaded Production model : random_forest
    Accuracy                 : 0.8268
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info and available routes |
| `GET` | `/health` | Check if server and model are ready |
| `POST` | `/predict` | Run predictions (single or batch) |
| `GET` | `/docs` | Auto-generated interactive Swagger UI |

### Input Schema

Each passenger in the request must include all engineered features:

| Field | Type | Description |
|---|---|---|
| `Pclass` | int | Ticket class (1, 2, 3) |
| `Sex` | int | 0=female, 1=male |
| `Age` | float | Passenger age |
| `SibSp` | int | Siblings/spouses aboard |
| `Parch` | int | Parents/children aboard |
| `Fare` | float | Ticket fare |
| `Embarked` | int | 0=C, 1=Q, 2=S |
| `Title` | int | Encoded title from name |
| `Deck` | int | Encoded deck from cabin |
| `FamilySize` | int | SibSp + Parch + 1 |
| `IsAlone` | int | 1 if travelling alone |
| `IsSmallFamily` | int | 1 if family size 2–4 |
| `IsLargeFamily` | int | 1 if family size 5+ |
| `IsChild` | int | 1 if age < 12 |
| `FarePerPerson` | float | Fare / FamilySize |
| `AgeBand` | int | Age bucket (encoded) |
| `FareBand` | int | Fare quartile (encoded) |
| `Age_Class` | float | Age × Pclass |
| `Fare_Class` | float | Fare × Pclass |

---

## 🧪 Testing with Bruno

Download Bruno from [usebruno.com](https://www.usebruno.com/).

### Test 1 — Health Check

```
GET http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Test 2 — Single Passenger

```
POST http://localhost:8000/predict
Content-Type: application/json
```

```json
{
  "passengers": [
    {
      "Pclass": 1, "Sex": 0, "Age": 29.0, "SibSp": 0,
      "Parch": 0, "Fare": 211.0, "Embarked": 0,
      "Title": 2, "Deck": 3, "FamilySize": 1,
      "IsAlone": 1, "IsSmallFamily": 0, "IsLargeFamily": 0,
      "IsChild": 0, "FarePerPerson": 211.0,
      "AgeBand": 2, "FareBand": 3,
      "Age_Class": 29.0, "Fare_Class": 211.0
    }
  ]
}
```

Response:
```json
{
  "model_used": "random_forest",
  "total": 1,
  "results": [
    {
      "index": 0,
      "prediction": 1,
      "survived_label": "✅ Survived",
      "survival_probability": 0.8958
    }
  ]
}
```

### Test 3 — Batch of Passengers

```
POST http://localhost:8000/predict
Content-Type: application/json
```

```json
{
  "passengers": [
    {
      "Pclass": 1, "Sex": 0, "Age": 29.0, "SibSp": 0,
      "Parch": 0, "Fare": 211.0, "Embarked": 0,
      "Title": 2, "Deck": 3, "FamilySize": 1,
      "IsAlone": 1, "IsSmallFamily": 0, "IsLargeFamily": 0,
      "IsChild": 0, "FarePerPerson": 211.0,
      "AgeBand": 2, "FareBand": 3,
      "Age_Class": 29.0, "Fare_Class": 211.0
    },
    {
      "Pclass": 3, "Sex": 1, "Age": 22.0, "SibSp": 1,
      "Parch": 0, "Fare": 7.25, "Embarked": 2,
      "Title": 1, "Deck": 0, "FamilySize": 2,
      "IsAlone": 0, "IsSmallFamily": 1, "IsLargeFamily": 0,
      "IsChild": 0, "FarePerPerson": 3.625,
      "AgeBand": 1, "FareBand": 0,
      "Age_Class": 66.0, "Fare_Class": 21.75
    },
    {
      "Pclass": 2, "Sex": 0, "Age": 8.0, "SibSp": 3,
      "Parch": 1, "Fare": 21.075, "Embarked": 2,
      "Title": 3, "Deck": 0, "FamilySize": 5,
      "IsAlone": 0, "IsSmallFamily": 0, "IsLargeFamily": 1,
      "IsChild": 1, "FarePerPerson": 4.215,
      "AgeBand": 0, "FareBand": 1,
      "Age_Class": 16.0, "Fare_Class": 42.15
    }
  ]
}
```

Response:
```json
{
  "model_used": "random_forest",
  "total": 3,
  "results": [
    {
      "index": 0, "prediction": 1,
      "survived_label": "✅ Survived",
      "survival_probability": 0.8958
    },
    {
      "index": 1, "prediction": 0,
      "survived_label": "❌ Did not survive",
      "survival_probability": 0.1236
    },
    {
      "index": 2, "prediction": 1,
      "survived_label": "✅ Survived",
      "survival_probability": 0.7412
    }
  ]
}
```

### Offline vs Online Prediction

| | `predict.py` | `main.py` (FastAPI) |
|---|---|---|
| Type | Offline batch script | Online REST API |
| Input | CSV file | JSON HTTP request |
| Runs | Once then exits | Runs continuously |
| Trigger | `dvc repro predict` | HTTP POST to `/predict` |
| Use case | Evaluation & testing | Production serving |
| Batch support | ✅ full CSV | ✅ multiple passengers per request |

---

## ⚙️ Batch Monitoring — Prefect + Evidently + MotherDuck (Lab 7)

### Overview

A scheduled Prefect batch job that:
1. Pulls test data from MotherDuck
2. Runs predictions using the Production model
3. Computes data drift and quality metrics using Evidently AI
4. Saves all results back to MotherDuck

### Step 1 — Load data into MotherDuck

Run once to upload test and train reference data:

```bash
uv run python load_to_motherduck.py
```

This creates two tables in MotherDuck:

| Table | Contents |
|---|---|
| `test_passengers` | Processed test data (model input) |
| `train_reference` | Training data used as Evidently reference baseline |

### Step 2 — Run the monitoring flow

```bash
uv run python monitor_flow.py
```

The Prefect flow runs 5 tasks in sequence:

```
Extract Task      ← pulls test_passengers + train_reference from MotherDuck
     │
Transform Task    ← encodes categoricals, drops non-feature columns
     │
Load Model Task   ← loads Production model from best_model.json
     │
Predict Task      ← runs predictions on test data
     │
Monitor Task      ← Evidently: data drift + quality metrics
     │
Save Task         ← writes predictions + monitoring_results to MotherDuck
```

Expected output:
```
✅ Extracted 418 rows from MotherDuck
✅ Transform complete — current: (418, 19)
✅ Loaded model : random_forest (accuracy: 0.8268)
✅ Survived: 152 | Not survived: 266
✅ Report saved → reports/evidently_2026-05-30.html
✅ Saved 418 predictions → MotherDuck
✅ Saved monitoring metric rows → MotherDuck
🎉 Monitoring flow complete!
```

### MotherDuck Tables After Running

| Table | Contents |
|---|---|
| `test_passengers` | Raw processed test data |
| `train_reference` | Training reference data for Evidently |
| `predictions` | Model output with probabilities and labels |
| `monitoring_results` | Evidently drift and quality metrics per run |

Query results directly in MotherDuck UI:

```sql
-- Prediction summary
SELECT survived_label, COUNT(*) AS total,
       ROUND(AVG(survival_probability), 4) AS avg_prob
FROM titanic_db.predictions
GROUP BY survived_label;

-- Latest drift metrics
SELECT metric_group, metric_name, metric_value
FROM titanic_db.monitoring_results
ORDER BY timestamp DESC LIMIT 20;
```

### Step 3 — Schedule the flow (optional)

```python
# schedule_flow.py
from monitor_flow import monitoring_flow

if __name__ == "__main__":
    monitoring_flow.serve(
        name="titanic-batch-daily",
        cron="0 9 * * *",  # every day at 9 AM
    )
```

```bash
uv run python schedule_flow.py
```

---

## 📊 Streamlit Dashboard (Lab 7)

```bash
uv run streamlit run dashboard.py
```

Opens at `http://localhost:8501`

| Tab | Content |
|---|---|
| 📊 **Predictions** | Total passengers, survival counts, avg probability, bar charts, histogram, model info |
| 🔍 **Data Drift** | Drift metrics table, quality metrics table, embedded Evidently HTML report |
| 📋 **Raw Data** | Filterable predictions table with label and probability filters |

---

## 🌊 Hydra Configuration

### Override Config from Terminal

```bash
# Change test size
uv run python pipeline.py data.test_size=0.3

# Run only one model
uv run python pipeline.py "training.models=[gradient_boosting]"

# Combine multiple overrides
uv run python pipeline.py data.test_size=0.25 training.cv_folds=3 "training.models=[gradient_boosting]"
```

---

## 📊 Models and Results

| Model | What it does | Validation Accuracy |
|---|---|---|
| Logistic Regression | Draws a line to separate survivors from non-survivors | ~82% |
| Random Forest | Builds many decision trees and votes on the result | ~83–85% |
| Gradient Boosting | Builds trees one at a time, each correcting the last | ~83–87% |

---

## 🔧 Feature Engineering

| Feature | Description |
|---|---|
| `Title` | Extracted from passenger name (Mr, Mrs, Miss, Rare) |
| `Deck` | First letter of Cabin number |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | 1 if travelling alone |
| `IsSmallFamily` | 1 if family size between 2–4 |
| `IsLargeFamily` | 1 if family size 5+ |
| `IsChild` | 1 if age < 12 |
| `FarePerPerson` | `Fare / FamilySize` |
| `AgeBand` | Age bucketed into Child, Teen, YoungAdult, Adult, Senior |
| `FareBand` | Fare bucketed into quartiles |
| `Age*Class` | Interaction: Age × Pclass |
| `Fare*Class` | Interaction: Fare × Pclass |

---

## 🧹 Code Quality

Always run in this order before committing:

```bash
uv run isort src/ pipeline.py main.py monitor_flow.py load_to_motherduck.py dashboard.py
uv run black src/ pipeline.py main.py monitor_flow.py load_to_motherduck.py dashboard.py
uv run ruff check src/ pipeline.py main.py monitor_flow.py load_to_motherduck.py dashboard.py --fix
git add .
git commit -m "your message"
git push origin main
dvc push
```

---

## 📦 Dependencies

```toml
[project]
dependencies = [
    "scikit-learn",
    "pandas",
    "numpy",
    "joblib",
    "hydra-core",
    "omegaconf",
    "dvc",
    "dvclive",
    "dvc[http]",
    "mlflow",
    "python-dotenv",
    "fastapi",
    "uvicorn",
    "pydantic",
    "prefect",
    "evidently",
    "duckdb",
    "streamlit",
]

[dependency-groups]
dev = [
    "ruff",
    "black",
    "isort",
    "kaggle",
]
```

---

## 🔗 References

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [ITI MLOps Reference Repo](https://github.com/Ezzaldin97/ITI-MLOps/tree/not-configured-pipeline)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DagsHub](https://dagshub.com)
- [Prefect Documentation](https://docs.prefect.io/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [MotherDuck Documentation](https://motherduck.com/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Bruno](https://www.usebruno.com/)
- [uv Documentation](https://docs.astral.sh/uv/)
