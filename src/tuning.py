import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.preprocess import get_preprocessor


PARAM_GRIDS = {
    "random_forest": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [4, 6, 8, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    },
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__penalty": ["l2"],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 4, 5],
        "classifier__subsample": [0.8, 1.0],
    },
}

MODELS = {
    "random_forest": RandomForestClassifier(random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


def tune_and_save(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    output_path: str,
    cv: int = 5,
) -> Pipeline:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}")

    pipeline = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", MODELS[model_name]),
    ])

    print(f"\n[~] Tuning '{model_name}' with {cv}-fold CV...")

    search = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRIDS[model_name],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f"Best params : {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")

    joblib.dump(search.best_estimator_, output_path)
    print(f"Saved to {output_path}")

    return search.best_estimator_