from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def evaluate(pipeline: Pipeline, X_val, y_val, model_name: str) -> float:
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n[{model_name}] Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred))
    return acc          # ← now returns accuracy for metrics tracking