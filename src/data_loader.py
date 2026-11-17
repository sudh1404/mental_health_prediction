from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "dreaddit"

TRAIN_FILE = DATA_DIR / "dreaddit-train.csv"
TEST_FILE = DATA_DIR / "dreaddit-test.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_dreaddit():
    """Load Dreaddit train and test datasets."""

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found:\n{TEST_FILE}"
        )

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Check required columns
    required_columns = {"text", "label"}

    for name, df in [("train", train_df), ("test", test_df)]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: {missing}"
            )

    return train_df, test_df


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    train_df, test_df = load_dreaddit()

    print("=" * 60)
    print("DREDDIT DATASET")
    print("=" * 60)

    print(f"\nTrain shape : {train_df.shape}")
    print(f"Test shape  : {test_df.shape}")

    print("\nColumns:")
    print(train_df.columns.tolist())

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts().sort_index())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts().sort_index())

    print("\nFirst post:")
    print(train_df["text"].iloc[0])

    print("\nFirst label:")
    print(train_df["label"].iloc[0])

    print("\nDataset loaded successfully!")