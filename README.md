# 🚢 Titanic — Machine Learning from Disaster

> Lab 0 | ITI MLOps Track  
> A fully automated and configurable training pipeline for the Kaggle Titanic dataset using scikit-learn and Hydra.

---

## 🤔 What is This Project?

This project builds a **Machine Learning Pipeline** to predict whether a passenger survived the Titanic shipwreck based on information like their age, gender, ticket class, and more.

It is based on the famous [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) and is built the **MLOps way**:

- Every step from loading data to saving the model is **automated**
- All settings are **configurable via Hydra** — no hardcoded values in code
- Every run is **logged and reproducible**
- Code is **clean and professionally formatted**
- Project is **versioned with Git** and pushed to GitHub

---

## 🧠 What is a Machine Learning Pipeline?

Think of a pipeline like an **assembly line in a factory**:

```
Raw Data → Clean → Engineer Features → Preprocess → Tune → Train → Save Model
```

Instead of running each step manually, a pipeline connects all steps so you run one command and everything happens automatically in the right order.

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
│   └── raw/                        ← train.csv and test.csv (not tracked by git)
│
├── models/                         ← Saved .pkl model files (not tracked by git)
│
├── outputs/                        ← Auto-created by Hydra (logs per run)
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── pipeline.log        ← Full console output for that run
│           └── .hydra/
│               ├── config.yaml     ← Exact config used
│               └── overrides.yaml  ← Any overrides passed from terminal
│
├── src/
│   ├── __init__.py
│   ├── load_data.py                ← Step 1: Load CSV files
│   ├── cleaning.py                 ← Step 2: Fix missing/bad data
│   ├── feature_engineering.py      ← Step 3: Create new useful columns
│   ├── preprocess.py               ← Step 4: Convert data to numbers
│   ├── training.py                 ← Step 5: Build and save model
│   ├── tuning.py                   ← Step 5+: GridSearchCV hyperparameter tuning
│   └── evaluate.py                 ← Step 6: Print accuracy and report
│
├── pipeline.py                     ← 🚀 MAIN FILE — runs all steps in order
├── pyproject.toml                  ← Project settings and dependencies
├── uv.lock                         ← Locked dependency versions
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

Get it from [Kaggle](https://www.kaggle.com/competitions/titanic) and place:

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

---

## 🌊 Hydra Configuration

All pipeline settings live in the `configs/` folder. You never need to touch the Python code to change behavior — just edit the config files or pass overrides from the terminal.

---

### `configs/config.yaml` — Main Config

This is the entry point. It links all sub-configs together.

```yaml
defaults:
  - data: titanic
  - training: default
  - _self_

name: titanic-pipeline

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  job:
    chdir: false
```

**To change:** You can swap which data or training config is loaded by changing `titanic` or `default` to another config file name you create.

---

### `configs/data/titanic.yaml` — Data Settings

Controls where data is loaded from and how it is split.

```yaml
train_path: data/raw/train.csv
test_path: data/raw/test.csv
test_size: 0.2
random_state: 42
```

| Setting | What it does | How to change |
|---|---|---|
| `train_path` | Path to training CSV | Point to a different dataset |
| `test_path` | Path to test CSV | Point to a different dataset |
| `test_size` | Fraction used for validation | `0.1` = 10%, `0.3` = 30% |
| `random_state` | Seed for reproducibility | Any integer |

**Example — use 30% for validation:**
```yaml
test_size: 0.3
```

---

### `configs/training/default.yaml` — Training Settings

Controls which models to train, how many CV folds to use, and where to save models.

```yaml
models:
  - random_forest
  - logistic_regression
  - gradient_boosting

cv_folds: 5
models_output_dir: models/
```

| Setting | What it does | How to change |
|---|---|---|
| `models` | List of models to train | Add or remove model names |
| `cv_folds` | Number of cross-validation folds | Higher = more reliable but slower |
| `models_output_dir` | Where to save `.pkl` files | Any folder path |

**Example — train only one model:**
```yaml
models:
  - gradient_boosting
```

**Example — use 10-fold CV:**
```yaml
cv_folds: 10
```

---

### `configs/model/random_forest.yaml` — Random Forest Hyperparameters

Controls the values GridSearchCV tries when tuning the Random Forest.

```yaml
n_estimators:
  - 100
  - 200
  - 300
max_depth:
  - 4
  - 6
  - 8
min_samples_split:
  - 2
  - 5
  - 10
min_samples_leaf:
  - 1
  - 2
  - 4
```

| Parameter | What it does | Tip |
|---|---|---|
| `n_estimators` | Number of trees in the forest | More trees = better but slower |
| `max_depth` | How deep each tree can grow | Deeper = more complex, risk of overfitting |
| `min_samples_split` | Min samples needed to split a node | Higher = simpler tree |
| `min_samples_leaf` | Min samples required at a leaf node | Higher = smoother model |

**Example — faster tuning with fewer options:**
```yaml
n_estimators:
  - 100
  - 200
max_depth:
  - 6
  - 8
```

---

### `configs/model/logistic_regression.yaml` — Logistic Regression Hyperparameters

```yaml
C:
  - 0.01
  - 0.1
  - 1
  - 10
  - 100
solver:
  - lbfgs
  - liblinear
penalty:
  - l2
```

| Parameter | What it does | Tip |
|---|---|---|
| `C` | Regularization strength (lower = stronger regularization) | Try values between 0.01 and 100 |
| `solver` | Algorithm used to optimize | `lbfgs` is good for most cases |
| `penalty` | Type of regularization | `l2` is the standard choice |

---

### `configs/model/gradient_boosting.yaml` — Gradient Boosting Hyperparameters

```yaml
n_estimators:
  - 100
  - 200
learning_rate:
  - 0.05
  - 0.1
  - 0.2
max_depth:
  - 3
  - 4
  - 5
subsample:
  - 0.8
  - 1.0
```

| Parameter | What it does | Tip |
|---|---|---|
| `n_estimators` | Number of boosting stages | More = better but slower |
| `learning_rate` | How much each tree contributes | Lower rate needs more estimators |
| `max_depth` | Depth of each tree | Keep low (3–5) for boosting |
| `subsample` | Fraction of samples per tree | Less than 1.0 reduces overfitting |

---

## ⚡ Override Config from Terminal

You can change any config value directly from the terminal without editing any file:

```bash
# Change test size
uv run python pipeline.py data.test_size=0.3

# Change random state
uv run python pipeline.py data.random_state=0

# Change number of CV folds
uv run python pipeline.py training.cv_folds=10

# Run only one model
uv run python pipeline.py "training.models=[gradient_boosting]"

# Run two models only
uv run python pipeline.py "training.models=[random_forest,gradient_boosting]"

# Combine multiple overrides
uv run python pipeline.py data.test_size=0.25 training.cv_folds=3 "training.models=[gradient_boosting]"
```

---

## 📊 Models and Results

| Model | What it does | Validation Accuracy |
|---|---|---|
| Logistic Regression | Draws a line to separate survivors from non-survivors | ~82% |
| Random Forest | Builds many decision trees and votes on the result | ~85% |
| Gradient Boosting | Builds trees one at a time, each correcting the last | ~86–87% |

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

| Tool | Purpose | Command |
|---|---|---|
| [isort](https://pycqa.github.io/isort/) | Sort imports | `uv run isort src/ pipeline.py` |
| [black](https://black.readthedocs.io/) | Format code | `uv run black src/ pipeline.py` |
| [ruff](https://docs.astral.sh/ruff/) | Lint and fix | `uv run ruff check src/ pipeline.py --fix` |

Run all three before every commit:

```bash
uv run isort src/ pipeline.py
uv run black src/ pipeline.py
uv run ruff check src/ pipeline.py --fix
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
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [black Documentation](https://black.readthedocs.io/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [ruff Documentation](https://docs.astral.sh/ruff/)
