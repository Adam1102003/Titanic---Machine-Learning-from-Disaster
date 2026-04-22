from sklearn.model_selection import train_test_split

from src.cleaning import clean_data
from src.evaluate import evaluate
from src.feature_engineering import engineer_features
from src.load_data import load_data
from src.preprocess import get_X_y
from src.tuning import tune_and_save


def run_pipeline() -> None:
    print("=" * 45)
    print("  Titanic Training Pipeline (Enhanced)")
    print("=" * 45)

    # 1. Load
    print("\n[1/5] Loading data...")
    train_df, _ = load_data("data/raw/train.csv", "data/raw/test.csv")

    # 2. Clean
    print("[2/5] Cleaning data...")
    train_df = clean_data(train_df)

    # 3. Feature Engineering
    print("[3/5] Engineering features...")
    train_df = engineer_features(train_df)

    # 4. Split
    print("[4/5] Splitting into train/validation...")
    X, y = get_X_y(train_df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Tune + Evaluate all models
    print("[5/5] Tuning and evaluating models...")
    for model_name in ["random_forest", "logistic_regression", "gradient_boosting"]:
        best_pipeline = tune_and_save(
            X_train,
            y_train,
            model_name=model_name,
            output_path=f"models/{model_name}.pkl",
        )
        evaluate(best_pipeline, X_val, y_val, model_name)

    print("\n[✓] Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()