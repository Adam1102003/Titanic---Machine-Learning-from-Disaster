import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "IsSmallFamily",
    "IsLargeFamily",
    "IsChild",
    "FarePerPerson",
    "Age*Class",
    "Fare*Class",
    "AgeBand",
    "FareBand",
    "Title",
    "Deck",
]
TARGET = "Survived"

numeric_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "Pclass",
    "FamilySize",
    "IsAlone",
    "IsSmallFamily",
    "IsLargeFamily",
    "IsChild",
    "FarePerPerson",
    "Age*Class",
    "Fare*Class",
]
categorical_features = ["Sex", "Embarked", "AgeBand", "FareBand", "Title", "Deck"]


def get_preprocessor(df: pd.DataFrame = None) -> ColumnTransformer:
    # Filter features based on what's available in the dataframe
    if df is not None:
        available_numeric = [f for f in numeric_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]
    else:
        available_numeric = numeric_features
        available_categorical = categorical_features

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, available_numeric),
            ("cat", categorical_transformer, available_categorical),
        ]
    )


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [f for f in FEATURES if f in df.columns]
    return df[available], df[TARGET]
