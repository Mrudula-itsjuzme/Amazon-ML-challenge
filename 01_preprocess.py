from pathlib import Path
import pandas as pd
import re
import unicodedata
from unidecode import unidecode


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = ROOT / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)


SOURCE_FILES = {
    "train": {
        "S1": TRAIN_DIR / "train_source1.tsv",
        "S2": TRAIN_DIR / "train_source2.tsv",
        "S3": TRAIN_DIR / "train_source3.tsv",
    },
    "test": {
        "S1": TEST_DIR / "test_source1.tsv",
        "S2": TEST_DIR / "test_source2.tsv",
        "S3": TEST_DIR / "test_source3.tsv",
    },
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize("NFKC", value)
    value = value.lower()
    value = unidecode(value)

    # Turn punctuation into spaces.
    value = re.sub(r"[^a-z0-9]+", " ", value)

    # Collapse whitespace.
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value):
    return normalize_text(value)


def normalize_address(value):
    return normalize_text(value)


def transliterate_name(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    value = unidecode(value)

    return normalize_text(value)


def extract_numbers(value):
    if not value:
        return []

    return re.findall(
        r"\d+(?:\.\d+)?",
        value,
    )


def preprocess_dataframe(df, source_name):
    required_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{source_name} missing columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    df = df[required_columns].copy()

    # Important: don't leave NaN values around.
    df["business_name"] = (
        df["business_name"]
        .fillna("")
        .astype(str)
    )

    df["business_address"] = (
        df["business_address"]
        .fillna("")
        .astype(str)
    )

    df["country"] = (
        df["country"]
        .fillna("")
        .astype(str)
    )

    # -----------------------------
    # Normalized representations
    # -----------------------------

    df["norm_name"] = (
        df["business_name"]
        .map(normalize_name)
    )

    df["translit_name"] = (
        df["business_name"]
        .map(transliterate_name)
    )

    df["norm_address"] = (
        df["business_address"]
        .map(normalize_address)
    )

    df["country_norm"] = (
        df["country"]
        .str.lower()
        .str.strip()
    )

    # -----------------------------
    # Missingness
    # -----------------------------

    df["name_missing"] = (
        df["norm_name"] == ""
    )

    df["address_missing"] = (
        df["norm_address"] == ""
    )

    # -----------------------------
    # Numeric tokens
    # -----------------------------

    df["name_numbers"] = (
        df["norm_name"]
        .map(extract_numbers)
        .map(lambda x: " ".join(sorted(set(x))))
    )

    df["address_numbers"] = (
        df["norm_address"]
        .map(extract_numbers)
        .map(lambda x: " ".join(sorted(set(x))))
    )

    df["source"] = source_name

    return df


def preprocess_split(split):
    print(f"\n{'=' * 60}")
    print(f"PREPROCESSING: {split.upper()}")
    print(f"{'=' * 60}")

    for source, path in SOURCE_FILES[split].items():

        if not path.exists():
            raise FileNotFoundError(
                f"Could not find: {path}"
            )

        print(f"\nLoading {source}:")
        print(path)

        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )

        print(
            f"Original rows: {len(df):,}"
        )

        df = preprocess_dataframe(
            df,
            source,
        )

        output = (
            OUTPUT_DIR
            / f"{split}_{source}_processed.tsv"
        )

        df.to_csv(
            output,
            sep="\t",
            index=False,
        )

        print(
            f"Saved: {output}"
        )


if __name__ == "__main__":

    preprocess_split("train")
    preprocess_split("test")