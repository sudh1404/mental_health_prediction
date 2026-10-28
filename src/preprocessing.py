import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Clean Reddit text while preserving useful linguistic information.
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove Reddit user references
    text = re.sub(
        r"\bu/\w+\b",
        " ",
        text
    )

    # Remove subreddit references
    text = re.sub(
        r"\br/\w+\b",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Keep letters, numbers, spaces and apostrophes
    text = re.sub(
        r"[^a-z0-9'\s]",
        " ",
        text
    )

    # Remove repeated spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned text to the dataframe."""

    df = df.copy()

    df["cleaned_text"] = df["text"].apply(clean_text)

    # Remove empty entries
    df = df[
        df["cleaned_text"].str.len() > 0
    ].copy()

    return df


if __name__ == "__main__":

    examples = [
        "I can't sleep!!! I'm REALLY worried.",
        "Visit https://example.com",
        "I'm posting on r/anxiety"
    ]

    for example in examples:

        print("\nOriginal:")
        print(example)

        print("\nCleaned:")
        print(clean_text(example))