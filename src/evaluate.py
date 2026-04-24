from sklearn.metrics import accuracy_score, classification_report


def evaluate(pipeline, X_test, y_test, model_name: str) -> None:
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[{model_name}] Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
