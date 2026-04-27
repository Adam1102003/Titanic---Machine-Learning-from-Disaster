import os

import pandas as pd
from omegaconf import OmegaConf

from src.load_data import load_data


def main() -> None:
    cfg = OmegaConf.load("configs/data/titanic.yaml")

    os.makedirs("data/processed", exist_ok=True)

    train_df, _ = load_data(cfg.train_path, cfg.test_path)

    train_df.to_csv("data/processed/train_raw.csv", index=False)
    print(f"[✓] Loaded {len(train_df)} rows → data/processed/train_raw.csv")


if __name__ == "__main__":
    main()