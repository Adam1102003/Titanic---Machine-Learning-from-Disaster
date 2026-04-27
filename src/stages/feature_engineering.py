import pandas as pd

from src.feature_engineering import engineer_features


def main() -> None:
    train_df = pd.read_csv("data/processed/train_cleaned.csv")

    train_df = engineer_features(train_df)

    train_df.to_csv("data/processed/train_featured.csv", index=False)
    print(f"[✓] Features engineered → data/processed/train_featured.csv")


if __name__ == "__main__":
    main()