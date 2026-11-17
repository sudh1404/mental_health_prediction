from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "dreaddit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# DOWNLOAD DREADDIT
# ---------------------------------------------------------

def download_dreaddit():

    print("=" * 60)
    print("DOWNLOADING DREDDIT DATASET")
    print("=" * 60)

    print("\nLoading dataset from Hugging Face...")

    dataset = load_dataset("andreagasparini/dreaddit")

    print("\nDataset loaded successfully!")

    print("\nAvailable splits:")
    print(dataset)

    # -----------------------------------------------------
    # TRAIN DATA
    # -----------------------------------------------------

    train_df = dataset["train"].to_pandas()

    train_path = OUTPUT_DIR / "dreaddit-train.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    # -----------------------------------------------------
    # TEST DATA
    # -----------------------------------------------------

    test_df = dataset["test"].to_pandas()

    test_path = OUTPUT_DIR / "dreaddit-test.csv"

    test_df.to_csv(
        test_path,
        index=False
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"\nTraining samples : {len(train_df)}")
    print(f"Test samples     : {len(test_df)}")

    print("\nTraining columns:")
    print(train_df.columns.tolist())

    print("\nLabel distribution:")
    print(train_df["label"].value_counts().sort_index())

    print("\nFiles saved to:")

    print(train_path)
    print(test_path)


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    download_dreaddit()