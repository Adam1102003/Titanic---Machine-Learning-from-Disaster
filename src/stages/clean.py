import pandas as pd

from src.cleaning import clean_data


def main() -> None:
    train_df = pd.read_csv("data/processed/train_raw.csv")

    train_df = clean_data(train_df)

    train_df.to_csv("data/processed/train_cleaned.csv", index=False)
    print(f"[✓] Cleaned data → data/processed/train_cleaned.csv")


if __name__ == "__main__":
    main()