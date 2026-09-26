from anyascii import anyascii
from pathlib import Path
import argparse
from collections import Counter
import re
import unicodedata

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

# Project layout:
#
# ML_challenge/
# ├── Amazon-ML-challenge/
# │   ├── code/business_entity_resolution/src/preprocess.py
# │   └── processed/
# └── student_resource/
#     └── dataset/
#
# preprocess.py
#   -> parents[0] = src
#   -> parents[1] = business_entity_resolution
#   -> parents[2] = code
#   -> parents[3] = Amazon-ML-challenge
#   -> parents[4] = ML_challenge

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = WORKSPACE_ROOT / "student_resource" / "dataset" / "train"
TEST_DIR = WORKSPACE_ROOT / "student_resource" / "dataset" / "test"

OUTPUT_DIR = REPO_ROOT / "processed"
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

    # Normalize Unicode compatibility forms.
    text = unicodedata.normalize("NFKC", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def basic_normalize(text: str) -> str:
    """
    General normalization used for names and addresses.

    Steps:
        1. Unicode normalization
        2. Case folding
        3. & -> and
        4. Punctuation normalization
        5. Whitespace normalization

    Important:
        Non-Latin Unicode characters are preserved.
    """

    if not isinstance(text, str):
        return ""

    text = normalize_unicode(text)

    # --------------------------------------------------------
    # Case normalization
    # --------------------------------------------------------

    text = text.casefold()

    # --------------------------------------------------------
    # Important challenge-specific normalization
    # --------------------------------------------------------
    #
    # Example:
    #     "A & B Motors"
    #         ->
    #     "a and b motors"
    #
    # The challenge explicitly mentions "&" vs "and"
    # as a business-name variation.
    #

    text = text.replace("&", " and ")

    # --------------------------------------------------------
    # Replace punctuation with spaces
    # --------------------------------------------------------

    # Keep Unicode letters/numbers.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    # --------------------------------------------------------
    # Collapse whitespace
    # --------------------------------------------------------

    text = re.sub(r"\s+", " ", text)

    return text.strip()
def transliterate_text(text):
    """
    Convert non-Latin Unicode text to an ASCII representation
    for secondary name matching/retrieval.

    The original Unicode text is preserved separately.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    return anyascii(text)

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
        ->
        "abc pvt ltd"
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
    Create a core business name by removing
    legal/business suffixes appearing at the end.

    Example:
        "Ruby Auto Pvt Ltd"
        ->
        "ruby auto"
    """

    normalized = normalize_legal_tokens(text)

    if not normalized:
        return ""

    tokens = normalized.split()

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
NUMBER_PATTERN = re.compile(r"\b\d+\b")


def extract_postal_code(address: str) -> str:
    """
    Extract likely numeric postal-code-like components.

    This is a weak feature, not a hard matching rule.
    """

    if not address:
        return ""

    matches = POSTAL_PATTERN.findall(address)

    if not matches:
        return ""

    return matches[-1]


def extract_numeric_tokens(address: str) -> str:
    """
    Extract numeric components from an address.

    Examples:
        31
        1212
        522001
    """

    if not address:
        return ""

    numbers = NUMBER_PATTERN.findall(address)

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

    result["name_translit"] = (
        result["business_name"]
        .fillna("")
        .astype(str)
        .map(transliterate_text)
        .map(basic_normalize)
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

    return result


# ============================================================
# FILE PROCESSING
# ============================================================

def process_file(
    input_path: Path,
    output_path: Path,
    force: bool = False,
):
    """
    Process one TSV file into Parquet.

    If the output already exists and force=False,
    preprocessing is skipped.
    """

    # --------------------------------------------------------
    # Skip existing output
    # --------------------------------------------------------

    if output_path.exists() and not force:
        print(f"\nSkipping: {input_path.name}")
        print(f"Already exists: {output_path}")

        return

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    print(f"\nProcessing: {input_path}")

    df = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed = preprocess_dataframe(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    processed.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    print(f"Saved: {output_path}")
    print(f"Columns: {len(processed.columns)}")


# ============================================================
# FILE LISTS
# ============================================================

TRAIN_FILES = [
    ("train_source1.tsv", "train_source1.parquet"),
    ("train_source2.tsv", "train_source2.parquet"),
    ("train_source3.tsv", "train_source3.parquet"),
]

TEST_FILES = [
    ("test_source1.tsv", "test_source1.parquet"),
    ("test_source2.tsv", "test_source2.parquet"),
    ("test_source3.tsv", "test_source3.parquet"),
]


# ============================================================
# PROCESS GROUP
# ============================================================

def process_group(
    files,
    input_dir: Path,
    group_name: str,
    force: bool = False,
):
    print(f"\n{'=' * 60}")
    print(f"{group_name}")
    print(f"{'=' * 60}")

    for input_name, output_name in files:

        process_file(
            input_dir / input_name,
            OUTPUT_DIR / output_name,
            force=force,
        )


# ============================================================
# SAMPLE TEST
# ============================================================
# ============================================================
# SCRIPT INSPECTION
# ============================================================

def detect_script(text: str) -> str:
    """
    Approximate script detection using Unicode character names.

    This is for dataset inspection only.
    It is NOT used as a matching feature.
    """

    if not text:
        return "EMPTY"

    script_counts = Counter()

    for char in text:

        if not char.isalpha():
            continue

        try:
            char_name = unicodedata.name(char)
        except ValueError:
            continue

        if "LATIN" in char_name:
            script = "LATIN"

        elif "DEVANAGARI" in char_name:
            script = "DEVANAGARI"

        elif "BENGALI" in char_name:
            script = "BENGALI"

        elif "GURMUKHI" in char_name:
            script = "GURMUKHI"

        elif "GUJARATI" in char_name:
            script = "GUJARATI"

        elif "ORIYA" in char_name or "ODIA" in char_name:
            script = "ODIA"

        elif "TAMIL" in char_name:
            script = "TAMIL"

        elif "TELUGU" in char_name:
            script = "TELUGU"

        elif "KANNADA" in char_name:
            script = "KANNADA"

        elif "MALAYALAM" in char_name:
            script = "MALAYALAM"

        elif "ARABIC" in char_name:
            script = "ARABIC"

        elif "CYRILLIC" in char_name:
            script = "CYRILLIC"

        elif "GREEK" in char_name:
            script = "GREEK"

        elif (
            "CJK" in char_name
            or "HIRAGANA" in char_name
            or "KATAKANA" in char_name
        ):
            script = "CJK_OR_JAPANESE"

        else:
            script = "OTHER"

        script_counts[script] += 1

    if not script_counts:
        return "NO_LETTERS"

    # Return the dominant script.
    return script_counts.most_common(1)[0][0]


def inspect_scripts():

    print("\n" + "=" * 60)
    print("MULTILINGUAL / SCRIPT INSPECTION")
    print("=" * 60)

    files = [
        ("S1", TRAIN_DIR / "train_source1.tsv"),
        ("S2", TRAIN_DIR / "train_source2.tsv"),
        ("S3", TRAIN_DIR / "train_source3.tsv"),
    ]

    for source_name, input_path in files:

        print(f"\n{'-' * 60}")
        print(f"{source_name}: {input_path.name}")
        print(f"{'-' * 60}")

        df = pd.read_csv(
            input_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            nrows=5000,
        )

        name_scripts = Counter(
            df["business_name"].map(detect_script)
        )

        address_scripts = Counter(
            df["business_address"].map(detect_script)
        )

        print("\nBusiness name scripts:")

        for script, count in name_scripts.most_common():
            percentage = count / len(df) * 100

            print(
                f"  {script:20s} "
                f"{count:5d} "
                f"({percentage:6.2f}%)"
            )

        print("\nAddress scripts:")

        for script, count in address_scripts.most_common():
            percentage = count / len(df) * 100

            print(
                f"  {script:20s} "
                f"{count:5d} "
                f"({percentage:6.2f}%)"
            )

        # ----------------------------------------------------
        # Show actual non-Latin examples
        # ----------------------------------------------------

        non_latin = df[
            df["business_name"].map(
                lambda x: detect_script(x) not in {
                    "LATIN",
                    "EMPTY",
                    "NO_LETTERS",
                }
            )
        ]

        print("\nNon-Latin business-name examples:")

        if len(non_latin) == 0:
            print("  None found in this 5,000-row sample.")

        else:
            for _, row in non_latin.head(10).iterrows():
                print(
                    f"  {row['business_name']}"
                )

    print("\nScript inspection complete.")
def run_sample(sample_size: int = 1000):
    """
    Quickly test the preprocessing logic on a small sample.

    This is useful when changing normalization rules.
    It avoids processing millions of rows.
    """

    print(f"\n{'=' * 60}")
    print(f"SAMPLE PREPROCESSING TEST ({sample_size:,} rows)")
    print(f"{'=' * 60}")

    input_path = TRAIN_DIR / "train_source1.tsv"

    df = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=sample_size,
    )

    processed = preprocess_dataframe(df)

    print("\nOriginal vs normalized names:\n")

    preview = processed[
        [
            "business_name",
            "name_norm",
            "name_legal_norm",
            "name_core",
            "name_translit",
        ]
    ].head(20)

    print(preview.to_string(index=False))

    print("\nOriginal vs normalized addresses:\n")

    preview_address = processed[
        [
            "business_address",
            "address_norm",
        ]
    ].head(20)

    print(preview_address.to_string(index=False))

    print("\nSample preprocessing test complete.")


# ============================================================
# COMMAND LINE
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge business entity preprocessing"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Process training data",
    )
    parser.add_argument(
        "--scripts",
        action="store_true",
        help="Inspect script distribution in 5,000 rows from each training source",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Process test data",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process both training and test data",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing Parquet files",
    )

    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Run preprocessing on only N training rows for a quick test",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    # --------------------------------------------------------
    # SCRIPT INSPECTION
    # --------------------------------------------------------

    if args.scripts:

        inspect_scripts()

    # --------------------------------------------------------
    # SAMPLE MODE
    # --------------------------------------------------------

    elif args.sample is not None:

        run_sample(args.sample)

    # --------------------------------------------------------
    # ALL MODE
    # --------------------------------------------------------

    elif args.all:

        process_group(
            TRAIN_FILES,
            TRAIN_DIR,
            "TRAINING DATA",
            force=args.force,
        )

        process_group(
            TEST_FILES,
            TEST_DIR,
            "TEST DATA",
            force=args.force,
        )

        print("\nPreprocessing complete.")

    # --------------------------------------------------------
    # TRAIN MODE
    # --------------------------------------------------------

    elif args.train:

        process_group(
            TRAIN_FILES,
            TRAIN_DIR,
            "TRAINING DATA",
            force=args.force,
        )

        print("\nTraining preprocessing complete.")

    # --------------------------------------------------------
    # TEST MODE
    # --------------------------------------------------------

    elif args.test:

        process_group(
            TEST_FILES,
            TEST_DIR,
            "TEST DATA",
            force=args.force,
        )

        print("\nTest preprocessing complete.")

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    else:

        print("\nNo processing mode selected.")

        print("\nUse one of:")
        print("  --scripts")
        print("  --sample 1000")
        print("  --train")
        print("  --test")
        print("  --all")

        print("\nAdd --force when you intentionally want")
        print("to regenerate existing Parquet files.")