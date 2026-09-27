from pathlib import Path
import re
import unicodedata

import pandas as pd
from unidecode import unidecode


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = ROOT / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


# ============================================================
# LEGAL SUFFIX NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = {
    "incorporated": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "company": "co",
    "co": "co",
    "limited": "ltd",
    "ltd": "ltd",
    "private": "pvt",
    "pvt": "pvt",
    "priv": "pvt",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
}


LEGAL_SUFFIX_TOKENS = {
    "inc",
    "corp",
    "co",
    "ltd",
    "pvt",
    "llc",
    "llp",
    "plc",
}


# ============================================================
# ADDRESS ABBREVIATIONS
# ============================================================

ADDRESS_ABBREVIATIONS = {
    "road": "rd",
    "rd": "rd",
    "street": "st",
    "st": "st",
    "avenue": "ave",
    "ave": "ave",
    "drive": "dr",
    "dr": "dr",
    "boulevard": "blvd",
    "blvd": "blvd",
    "lane": "ln",
    "ln": "ln",
    "highway": "hwy",
    "hwy": "hwy",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "place": "pl",
    "pl": "pl",
    "square": "sq",
    "sq": "sq",
    "apartment": "apt",
    "apt": "apt",
    "suite": "ste",
    "ste": "ste",
    "floor": "fl",
    "fl": "fl",
}


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def normalize_unicode(value):
    """
    Unicode-aware normalization.

    Keeps Unicode characters intact.
    This is the main normalized representation.
    """
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))

    # Casefold is stronger and more Unicode-aware than lower().
    text = text.casefold()

    # Replace punctuation with spaces.
    # Keep Unicode letters/numbers.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def transliterate_text(value):
    """
    Create a Latin/transliterated representation.

    The original Unicode-normalized representation is preserved
    separately, so transliteration never destroys information.
    """
    normalized = normalize_unicode(value)

    if not normalized:
        return ""

    return unidecode(normalized)


# ============================================================
# BUSINESS NAME REPRESENTATIONS
# ============================================================

def normalize_legal_tokens(text):
    """
    Normalize common legal/business suffix variants.

    Example:

        abc private limited
        -> abc pvt ltd
    """
    if not text:
        return ""

    tokens = text.split()

    normalized_tokens = [
        LEGAL_SUFFIXES.get(token, token)
        for token in tokens
    ]

    return " ".join(normalized_tokens)


def remove_legal_suffixes(text):
    """
    Remove legal suffixes appearing at the END of a business name.

    Example:

        abc auto pvt ltd
        -> abc auto

    Only suffixes at the end are removed.
    """
    if not text:
        return ""

    tokens = normalize_legal_tokens(text).split()

    while tokens and tokens[-1] in LEGAL_SUFFIX_TOKENS:
        tokens.pop()

    return " ".join(tokens)


# ============================================================
# ADDRESS REPRESENTATIONS
# ============================================================

def normalize_address(text):
    """
    Normalize address while preserving meaningful tokens.
    Common address words are canonicalized.
    """
    normalized = normalize_unicode(text)

    if not normalized:
        return ""

    tokens = normalized.split()

    normalized_tokens = [
        ADDRESS_ABBREVIATIONS.get(token, token)
        for token in tokens
    ]

    return " ".join(normalized_tokens)


# ============================================================
# NUMERIC / POSTAL FEATURES
# ============================================================

def extract_numeric_tokens(text):
    """
    Extract numeric components.

    Examples:
        31
        1212
        522001
    """
    if not text:
        return ""

    numbers = re.findall(r"\b\d+\b", text)

    # Preserve order but remove duplicates.
    seen = set()
    unique_numbers = []

    for number in numbers:
        if number not in seen:
            seen.add(number)
            unique_numbers.append(number)

    return " ".join(unique_numbers)


def extract_postal_code(address):
    """
    Extract likely postal-code tokens.

    We don't assume the country is only US/India.
    The representation is deliberately conservative.
    """
    if not address:
        return ""

    matches = re.findall(r"\b\d{5,6}\b", address)

    if not matches:
        return ""

    return matches[-1]


# ============================================================
# DATAFRAME PREPROCESSING
# ============================================================

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

    # Keep only the challenge's actual source columns.
    result = df[required_columns].copy()

    # --------------------------------------------------------
    # Raw string cleanup
    # --------------------------------------------------------

    result["business_name"] = (
        result["business_name"]
        .fillna("")
        .astype(str)
    )

    result["business_address"] = (
        result["business_address"]
        .fillna("")
        .astype(str)
    )

    result["country"] = (
        result["country"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # NAME REPRESENTATIONS
    # --------------------------------------------------------

    result["name_norm"] = (
        result["business_name"]
        .map(normalize_unicode)
    )

    result["name_translit"] = (
        result["business_name"]
        .map(transliterate_text)
    )

    result["name_legal_norm"] = (
        result["name_norm"]
        .map(normalize_legal_tokens)
    )

    result["name_core"] = (
        result["name_norm"]
        .map(remove_legal_suffixes)
    )

    # --------------------------------------------------------
    # ADDRESS REPRESENTATIONS
    # --------------------------------------------------------

    result["address_norm"] = (
        result["business_address"]
        .map(normalize_address)
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    result["country_norm"] = (
        result["country"]
        .map(normalize_unicode)
    )

    # --------------------------------------------------------
    # MISSINGNESS
    # --------------------------------------------------------

    result["name_missing"] = (
        result["name_norm"] == ""
    )

    result["address_missing"] = (
        result["address_norm"] == ""
    )

    result["country_missing"] = (
        result["country_norm"] == ""
    )

    # --------------------------------------------------------
    # NUMERIC INFORMATION
    # --------------------------------------------------------

    result["name_numbers"] = (
        result["name_norm"]
        .map(extract_numeric_tokens)
    )

    result["address_numbers"] = (
        result["address_norm"]
        .map(extract_numeric_tokens)
    )

    result["postal_code"] = (
        result["address_norm"]
        .map(extract_postal_code)
    )

    # --------------------------------------------------------
    # LENGTH / TOKEN FEATURES
    # --------------------------------------------------------

    result["name_length"] = (
        result["name_norm"].str.len()
    )

    result["address_length"] = (
        result["address_norm"].str.len()
    )

    result["name_token_count"] = (
        result["name_norm"]
        .str.split()
        .str.len()
    )

    result["address_token_count"] = (
        result["address_norm"]
        .str.split()
        .str.len()
    )

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    result["source"] = source_name

    return result


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def preprocess_split(split):

    print("\n" + "=" * 60)
    print(f"PREPROCESSING: {split.upper()}")
    print("=" * 60)

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

        processed = preprocess_dataframe(
            df,
            source,
        )

        output = (
            OUTPUT_DIR
            / f"{split}_{source}_processed.tsv"
        )

        processed.to_csv(
            output,
            sep="\t",
            index=False,
        )

        print(
            f"Saved: {output}"
        )

        print(
            f"Columns: {len(processed.columns)}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    preprocess_split("train")
    preprocess_split("test")

    print("\nPreprocessing complete.")