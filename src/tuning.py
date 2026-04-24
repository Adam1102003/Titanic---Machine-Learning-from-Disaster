import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.preprocess import get_preprocessor

MODELS = {
    "random_forest": RandomForestClassifier(random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


def build_param_grid(model_name: str, model_cfg: DictConfig) -> dict:
    """Build param grid dynamically from config file."""
    return {f"classifier__{key}": list(values) for key, values in model_cfg.items()}


def tune_and_save(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    model_cfg: DictConfig,
    output_path: str,
    cv: int = 5,
) -> Pipeline:
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}"
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("classifier", MODELS[model_name]),
        ]
    )

    param_grid = build_param_grid(model_name, model_cfg)

    print(f"\n[~] Tuning '{model_name}' with {cv}-fold CV...")
    print(f"    Param grid: {param_grid}")

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f"[✓] Best params : {search.best_params_}")
    print(f"[✓] Best CV score: {search.best_score_:.4f}")

    joblib.dump(search.best_estimator_, output_path)
    print(f"[✓] Saved to {output_path}")

    return search.best_estimator_
