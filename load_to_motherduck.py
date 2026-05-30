import os

import duckdb
import pandas as pd
from dotenv import load_dotenv

from src.cleaning import clean_data
from src.feature_engineering import engineer_features

load_dotenv()

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB    = os.getenv("MOTHERDUCK_DB", "titanic_db")
TEST_PATH        = "data/raw/test.csv"
TRAIN_PATH       = "data/processed/train_featured.csv"


def get_connection():
    conn = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
    conn.execute(f"CREATE DATABASE IF NOT EXISTS {MOTHERDUCK_DB}")
    conn.execute(f"USE {MOTHERDUCK_DB}")
    return conn


def load_test_data(conn):
    print("\n📂 Loading raw test data...")
    df = pd.read_csv(TEST_PATH)
    print(f"   Raw shape : {df.shape}")

    # Keep PassengerId before cleaning drops it
    passenger_ids = df["PassengerId"].copy()

    print("🔧 Cleaning...")
    df = clean_data(df)

    print("🔧 Engineering features...")
    df = engineer_features(df)

    # Re-attach PassengerId
    df.insert(0, "PassengerId", passenger_ids.values)

    # Encode categoricals so DuckDB can store them
    cat_map = {
        "Sex":      {"male": 1, "female": 0},
        "Embarked": {"C": 0, "Q": 1, "S": 2},
        "AgeBand":  {"Child": 0, "Teen": 1, "YoungAdult": 2,
                     "Adult": 3, "Senior": 4},
        "FareBand": {"Low": 0, "Mid": 1, "High": 2, "VeryHigh": 3},
        "Title":    {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5},
        "Deck":     {d: i for i, d in enumerate(
            ["Unknown", "A", "B", "C", "D", "E", "F", "G", "T"]
        )},
    }
    for col, mapping in cat_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # Drop non-feature columns
    for col in ["Name", "Cabin", "Ticket"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df = df.fillna(df.median(numeric_only=True))

    print(f"   Final shape : {df.shape}")
    print(f"   Columns     : {list(df.columns)}")

    conn.execute("DROP TABLE IF EXISTS test_passengers")
    conn.execute("CREATE TABLE test_passengers AS SELECT * FROM df")
    count = conn.execute("SELECT COUNT(*) FROM test_passengers").fetchone()[0]
    print(f"✅ Loaded {count} rows → {MOTHERDUCK_DB}.test_passengers")


def load_train_reference(conn):
    print("\n📂 Loading train reference data...")
    df = pd.read_csv(TRAIN_PATH)
    print(f"   Raw shape : {df.shape}")

    # Encode categoricals
    cat_map = {
        "Sex":      {"male": 1, "female": 0},
        "Embarked": {"C": 0, "Q": 1, "S": 2},
        "AgeBand":  {"Child": 0, "Teen": 1, "YoungAdult": 2,
                     "Adult": 3, "Senior": 4},
        "FareBand": {"Low": 0, "Mid": 1, "High": 2, "VeryHigh": 3},
        "Title":    {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5},
        "Deck":     {d: i for i, d in enumerate(
            ["Unknown", "A", "B", "C", "D", "E", "F", "G", "T"]
        )},
    }
    for col, mapping in cat_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # Drop non-feature columns
    for col in ["Name", "Cabin", "Ticket", "PassengerId"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Drop unnamed index columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.fillna(df.median(numeric_only=True))

    print(f"   Final shape : {df.shape}")

    conn.execute("DROP TABLE IF EXISTS train_reference")
    conn.execute("CREATE TABLE train_reference AS SELECT * FROM df")
    count = conn.execute("SELECT COUNT(*) FROM train_reference").fetchone()[0]
    print(f"✅ Loaded {count} rows → {MOTHERDUCK_DB}.train_reference")


def verify(conn):
    print("\n🔍 Verifying tables in MotherDuck...")
    tables = conn.execute("SHOW TABLES").df()
    print(tables.to_string(index=False))


def main():
    print(f"🦆 Connecting to MotherDuck ({MOTHERDUCK_DB})...")
    conn = get_connection()

    load_test_data(conn)
    load_train_reference(conn)
    verify(conn)

    conn.close()
    print("\n🎉 All data loaded successfully!")


if __name__ == "__main__":
    main()