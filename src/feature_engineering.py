import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Family Features ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["IsSmallFamily"] = (df["FamilySize"].between(2, 4)).astype(int)
    df["IsLargeFamily"] = (df["FamilySize"] >= 5).astype(int)

    # --- Age Features ---
    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    )
    df["IsChild"] = (df["Age"] < 12).astype(int)

    # --- Fare Features ---
    df["FareBand"] = pd.qcut(
        df["Fare"],
        q=4,
        labels=["Low", "Mid", "High", "VeryHigh"],
        duplicates="drop",
    )
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # --- Title extracted from Name ---
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        df["Title"] = df["Title"].replace(
            ["Lady", "Countess", "Capt", "Col", "Don",
             "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
            "Rare",
        )
        df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # --- Deck from Cabin ---
    if "Cabin" in df.columns:
        df["Deck"] = df["Cabin"].str[0]
        df["Deck"].fillna("Unknown", inplace=True)
    else:
        df["Deck"] = "Unknown"

    # --- Interaction Features ---
    df["Age*Class"] = df["Age"] * df["Pclass"]
    df["Fare*Class"] = df["Fare"] * df["Pclass"]

    return df