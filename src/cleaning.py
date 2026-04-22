import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep Name and Cabin — needed for feature engineering
    # Only drop PassengerId and Ticket
    df.drop(columns=["PassengerId", "Ticket"], inplace=True, errors="ignore")

    # Fill Age with median grouped by Pclass and Sex
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # Fill Embarked with mode
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Fill Fare with median
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    return df