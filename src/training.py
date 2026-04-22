import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocess import get_preprocessor


MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42) ,
}


def build_model_pipeline(model_name: str, X_train=None) -> Pipeline:
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}"
        )

    return Pipeline(steps=[
        ("preprocessor", get_preprocessor(df=X_train)),
        ("classifier", MODELS[model_name]),
    ])


def train_and_save(X_train, y_train, model_name: str, output_path: str) -> Pipeline:
    pipeline = build_model_pipeline(model_name, X_train=X_train)
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, output_path)
    print(f"Model '{model_name}' saved to {output_path}")
    return pipeline
