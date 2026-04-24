import os

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from src.cleaning import clean_data
from src.evaluate import evaluate
from src.feature_engineering import engineer_features
from src.load_data import load_data
from src.preprocess import get_X_y
from src.tuning import tune_and_save


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_pipeline(cfg: DictConfig) -> None:
    print("=" * 45)
    print("  Titanic Training Pipeline (Hydra)")
    print("=" * 45)

    # 1. Load
    print("\n[1/5] Loading data...")
    train_df, _ = load_data(cfg.data.train_path, cfg.data.test_path)

    # 2. Clean
    print("[2/5] Cleaning data...")
    train_df = clean_data(train_df)

    # 3. Feature Engineering
    print("[3/5] Engineering features...")
    train_df = engineer_features(train_df)

    # 4. Split
    print("[4/5] Splitting data...")
    X, y = get_X_y(train_df)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
    )

    # 5. Tune + Evaluate each model
    print("[5/5] Tuning and evaluating models...")
    os.makedirs(cfg.training.models_output_dir, exist_ok=True)

    for model_name in cfg.training.models:

        # Load model config directly from yaml file
        model_cfg_path = os.path.join("configs", "model", f"{model_name}.yaml")
        model_cfg = OmegaConf.load(model_cfg_path)

        output_path = os.path.join(cfg.training.models_output_dir, f"{model_name}.pkl")

        best_pipeline = tune_and_save(
            X_train,
            y_train,
            model_name=model_name,
            model_cfg=model_cfg,  # ← passed directly from OmegaConf.load
            output_path=output_path,
            cv=cfg.training.cv_folds,
        )
        evaluate(best_pipeline, X_val, y_val, model_name)

    print("\n[✓] Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
