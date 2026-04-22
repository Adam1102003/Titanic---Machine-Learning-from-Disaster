# 🚢 Titanic — Machine Learning from Disaster

> Lab 0 | ITI MLOps Track  
> A fully automated training pipeline for the Kaggle Titanic dataset using scikit-learn, with feature engineering, hyperparameter tuning, and clean code formatting.

---

## 📁 Project Structure

```
Titanic---Machine-Learning-from-Disaster/
├── data/
│   └── raw/                    # train.csv, test.csv (not tracked by git)
├── models/                     # saved .pkl model files (not tracked by git)
├── src/
│   ├── __init__.py
│   ├── load_data.py            # data loading
│   ├── cleaning.py             # missing value handling
│   ├── feature_engineering.py  # feature creation
│   ├── preprocess.py           # sklearn ColumnTransformer pipeline
│   ├── training.py             # model building and saving
│   ├── tuning.py               # GridSearchCV hyperparameter tuning
│   └── evaluate.py             # metrics and classification report
├── pipeline.py                 # main entry point
├── pyproject.toml              # project config + tool settings
├── uv.lock                     # locked dependencies
└── README.md
```

---

## ⚙️ Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### 1. Install uv

```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Download the Titanic dataset

Get it from [Kaggle](https://www.kaggle.com/competitions/titanic) and place the files:

```
data/raw/train.csv
data/raw/test.csv
```

Or use the Kaggle CLI:

```bash
uv run kaggle competitions download -c titanic
unzip titanic.zip -d data/raw/
```

---

## 🚀 Run the Pipeline

```bash
uv run python pipeline.py
```

The pipeline will:

1. Load `train.csv`
2. Clean the data (handle missing values, drop low-signal columns)
3. Engineer new features
4. Split into train/validation sets
5. Tune hyperparameters using `GridSearchCV` with 5-fold cross-validation
6. Evaluate each model and print accuracy + classification report
7. Save the best model for each algorithm to `models/`

---

## 🧠 Models

| Model | Notes |
|---|---|
| `RandomForestClassifier` | Ensemble of decision trees |
| `LogisticRegression` | Linear baseline model |
| `GradientBoostingClassifier` | Boosted trees, usually best performer |

All models are wrapped in a full **scikit-learn Pipeline** that includes preprocessing, so the saved `.pkl` file handles raw input end-to-end.

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
| `Age*Class` | Interaction term: Age × Pclass |
| `Fare*Class` | Interaction term: Fare × Pclass |

---

## 📊 Results

| Model | Validation Accuracy |
|---|---|
| Logistic Regression | ~82% |
| Random Forest (tuned) | ~85% |
| Gradient Boosting (tuned) | ~86–87% |

---

## 🧹 Code Quality

This project uses the following tools for formatting and linting:

| Tool | Purpose |
|---|---|
| [black](https://black.readthedocs.io/) | Code formatting |
| [isort](https://pycqa.github.io/isort/) | Import sorting |
| [ruff](https://docs.astral.sh/ruff/) | Fast linting and auto-fix |

Run all formatters:

```bash
uv run isort src/ pipeline.py
uv run black src/ pipeline.py
uv run ruff check src/ pipeline.py --fix
```

---

## 📦 Dependencies

Managed via `uv` and defined in `pyproject.toml`:

```toml
[project]
dependencies = [
    "scikit-learn",
    "pandas",
    "numpy",
    "joblib",
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
- [scikit-learn Docs](https://scikit-learn.org/)
- [uv Docs](https://docs.astral.sh/uv/)
- [ruff Docs](https://docs.astral.sh/ruff/)
