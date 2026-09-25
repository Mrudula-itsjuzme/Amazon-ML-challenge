from pathlib import Path
import re
import unicodedata

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    "/Users/lakshmikotaru/amazon_ml_entity_resolution/"
    "student_resource"
)

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"

OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LEGAL / ADDRESS ABBREVIATIONS
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
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text: str) -> str:
    """
    Unicode normalization without destroying non-Latin scripts.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Normalize common whitespace characters.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def basic_normalize(text: str) -> str:
    """
    General normalization used for both names and addresses.

    Important:
    - preserves Unicode
    - lowercases
    - normalizes whitespace
    - separates punctuation where useful
    """
    if not isinstance(text, str):
        return ""

    text = normalize_unicode(text)

    # Case normalization.
    text = text.casefold()

    # Replace punctuation with spaces.
    # We intentionally DON'T delete Unicode letters/numbers.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str):
    if not text:
        return []

    return text.split()


# ============================================================
# BUSINESS NAME NORMALIZATION
# ============================================================

def normalize_business_name(text: str) -> str:
    return basic_normalize(text)


def normalize_legal_tokens(text: str) -> str:
    """
    Normalize common legal suffix variants.

    Example:
        "ABC PRIVATE LIMITED"
        -> "abc pvt ltd"
    """
    normalized = normalize_business_name(text)

    if not normalized:
        return ""

    tokens = normalized.split()

    normalized_tokens = [
        LEGAL_SUFFIXES.get(token, token)
        for token in tokens
    ]

    return " ".join(normalized_tokens)


def remove_legal_suffixes(text: str) -> str:
    """
    Create a 'core name'.

    Example:
        "Ruby Auto Pvt Ltd"
        -> "ruby auto"

    We only remove legal/business suffix tokens.
    """
    normalized = normalize_legal_tokens(text)

    if not normalized:
        return ""

    tokens = normalized.split()

    # Remove suffixes appearing at the end.
    while tokens and tokens[-1] in {
        "inc",
        "corp",
        "co",
        "ltd",
        "pvt",
        "llc",
        "llp",
        "plc",
    }:
        tokens.pop()

    return " ".join(tokens)


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_address(text: str) -> str:
    """
    Normalize address formatting and common abbreviations.

    Example:
        "1212 Old Rte 34, Sandwich, IL"
        ->
        "1212 old rte 34 sandwich il"
    """
    normalized = basic_normalize(text)

    if not normalized:
        return ""

    tokens = normalized.split()

    normalized_tokens = [
        ADDRESS_ABBREVIATIONS.get(token, token)
        for token in tokens
    ]

    return " ".join(normalized_tokens)


# ============================================================
# ADDRESS COMPONENT EXTRACTION
# ============================================================

POSTAL_PATTERN = re.compile(r"\b\d{5,6}\b")


def extract_postal_code(address: str) -> str:
    """
    Extract likely US/India-style numeric postal codes.

    We keep this deliberately conservative.
    """
    if not address:
        return ""

    matches = POSTAL_PATTERN.findall(address)

    if not matches:
        return ""

    # Usually the last postal-looking number is the most useful.
    return matches[-1]


def extract_numeric_tokens(address: str) -> str:
    """
    Extract numeric components such as:

        31
        1212
        522001
    """
    if not address:
        return ""

    numbers = re.findall(r"\b\d+\b", address)

    return " ".join(numbers)


# ============================================================
# RECORD PREPROCESSING
# ============================================================

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    # Keep the original columns.
    result = df.copy()

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------

    result["address_missing"] = (
        result["business_address"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    # --------------------------------------------------------
    # Business name
    # --------------------------------------------------------

    result["name_norm"] = (
        result["business_name"]
        .fillna("")
        .astype(str)
        .map(normalize_business_name)
    )

    result["name_legal_norm"] = (
        result["business_name"]
        .fillna("")
        .astype(str)
        .map(normalize_legal_tokens)
    )

    result["name_core"] = (
        result["business_name"]
        .fillna("")
        .astype(str)
        .map(remove_legal_suffixes)
    )

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------

    result["address_norm"] = (
        result["business_address"]
        .fillna("")
        .astype(str)
        .map(normalize_address)
    )

    result["postal_code"] = (
        result["address_norm"]
        .map(extract_postal_code)
    )

    result["numeric_tokens"] = (
        result["address_norm"]
        .map(extract_numeric_tokens)
    )

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    result["country_norm"] = (
        result["country"]
        .fillna("")
        .astype(str)
        .map(normalize_business_name)
    )

    # --------------------------------------------------------
    # Useful lengths
    # --------------------------------------------------------

    result["name_length"] = result["name_norm"].str.len()
    result["address_length"] = result["address_norm"].str.len()

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

    return result


# ============================================================
# FILE PROCESSING
# ============================================================

def process_file(input_path: Path, output_path: Path):

    print(f"\nProcessing: {input_path}")

    df = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"Rows: {len(df):,}")

    processed = preprocess_dataframe(df)

    processed.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    print(f"Saved: {output_path}")
    print(f"Columns: {len(processed.columns)}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    files = [
        ("train_source1.tsv", "train_source1.parquet"),
        ("train_source2.tsv", "train_source2.parquet"),
        ("train_source3.tsv", "train_source3.parquet"),
    ]

    for input_name, output_name in files:
        process_file(
            TRAIN_DIR / input_name,
            OUTPUT_DIR / output_name,
        )

    print("\nPreprocessing complete.")