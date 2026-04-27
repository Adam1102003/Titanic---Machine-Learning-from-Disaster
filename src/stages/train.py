import json
import os

import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from src.evaluate import evaluate
from src.preprocess import get_X_y
from src.tuning import tune_and_save


def main() -> None:
    cfg_data = OmegaConf.load("configs/data/titanic.yaml")
    cfg_training = OmegaConf.load("configs/training/default.yaml")

    train_df = pd.read_csv("data/processed/train_featured.csv")

    X, y = get_X_y(train_df)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg_data.test_size,
        random_state=cfg_data.random_state,
    )

    os.makedirs(cfg_training.models_output_dir, exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    scores = {}

    for model_name in cfg_training.models:
        model_cfg = OmegaConf.load(f"configs/model/{model_name}.yaml")

        output_path = os.path.join(
            cfg_training.models_output_dir, f"{model_name}.pkl"
        )

        best_pipeline = tune_and_save(
            X_train,
            y_train,
            model_name=model_name,
            model_cfg=model_cfg,
            output_path=output_path,
            cv=cfg_training.cv_folds,
        )

        acc = evaluate(best_pipeline, X_val, y_val, model_name)
        scores[model_name] = round(acc, 4)

    # Save metrics to JSON for DVC to track
    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\n[✓] Metrics saved → metrics/scores.json")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()