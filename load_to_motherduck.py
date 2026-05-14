import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

from src.cleaning import clean_data
from src.feature_engineering import engineer_features
from src.preprocess import get_X_y

load_dotenv()

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB    = os.getenv("MOTHERDUCK_DB", "titanic_db")
TEST_DATA_PATH   = "data/raw/test.csv"


def load_test_data():
    print("📂 Loading raw test data...")
    df = pd.read_csv(TEST_DATA_PATH)
    print(f"   Raw shape : {df.shape}")

    # Apply same steps as training pipeline
    print("🔧 Cleaning...")
    df = clean_data(df)

    print("🔧 Engineering features...")
    df = engineer_features(df)

    print(f"   Final shape : {df.shape}")
    print(f"   Columns     : {list(df.columns)}")

    # Connect to MotherDuck
    conn = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
    conn.execute(f"CREATE DATABASE IF NOT EXISTS {MOTHERDUCK_DB}")
    conn.execute(f"USE {MOTHERDUCK_DB}")

    conn.execute("DROP TABLE IF EXISTS test_passengers")
    conn.execute("CREATE TABLE test_passengers AS SELECT * FROM df")

    count = conn.execute("SELECT COUNT(*) FROM test_passengers").fetchone()[0]
    print(f"✅ Loaded {count} rows into MotherDuck → {MOTHERDUCK_DB}.test_passengers")
    conn.close()


if __name__ == "__main__":
    load_test_data()