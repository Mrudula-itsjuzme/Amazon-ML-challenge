from pathlib import Path
import ast
import json
import pickle
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "LightGBM is required for 03_filter_candidates.py.\n"
        "Install with: pip install lightgbm"
    ) from e


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT / "processed"
CANDIDATE_DIR = ROOT / "candidates"
FILTERED_DIR = ROOT / "filtered"
MODEL_DIR = ROOT / "models"

FILTERED_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# INPUTS
# ============================================================

S1_PATH = PROCESSED_DIR / "train_S1_processed.tsv"
S2_PATH = PROCESSED_DIR / "train_S2_processed.tsv"
S3_PATH = PROCESSED_DIR / "train_S3_processed.tsv"

GROUND_TRUTH = (
    ROOT
    / "student_resource"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

CANDIDATES = CANDIDATE_DIR / "train_candidate_pairs.tsv"


# ============================================================
# OUTPUTS
# ============================================================

FINAL_CANDIDATES = (
    FILTERED_DIR
    / "train_final_candidate_pairs.tsv"
)

SCORED_CANDIDATES = (
    FILTERED_DIR
    / "train_scored_candidate_pairs.tsv"
)

VALIDATION_DIAGNOSTICS = (
    FILTERED_DIR
    / "train_validation_diagnostics.tsv"
)

MODEL_PATH = MODEL_DIR / "entity_matcher.pkl"
MODEL_META_PATH = MODEL_DIR / "entity_matcher_meta.json"


# ============================================================
# DEVELOPMENT SPLIT
# ============================================================

TOTAL_DEV_SIZE = 5_000

TRAIN_SIZE = 3_000
VALID_SIZE = 1_000
TEST_SIZE = 1_000

RANDOM_SEED = 32


# ============================================================
# RUNTIME
# ============================================================

CHUNK_SIZE = 250_000

NEGATIVE_TO_POSITIVE_RATIO = 8


# ============================================================
# MODEL
# ============================================================

MODEL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",

    "n_estimators": 800,
    "learning_rate": 0.04,

    "num_leaves": 63,
    "max_depth": -1,

    "min_child_samples": 40,

    "subsample": 0.85,
    "colsample_bytree": 0.90,

    "reg_alpha": 0.2,
    "reg_lambda": 1.0,

    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# EXPECTED PROCESSED SCHEMA
# ============================================================
#
# These are the ACTUAL columns in your processed files.
#
# We deliberately validate them instead of using dict.get()
# so a future column mismatch causes an immediate error.
# ============================================================

REQUIRED_PROCESSED_COLUMNS = {
    "entity_id",
    "name_norm",
    "name_translit",
    "name_core",
    "address_norm",
    "country_norm",
    "postal_code",
    "name_missing",
    "address_missing",
    "country_missing",
    "name_numbers",
    "address_numbers",
}


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURE_COLUMNS = [
    # --------------------------------------------------------
    # Exact identity
    # --------------------------------------------------------
    "exact_name",
    "exact_core_name",
    "exact_translit",
    "exact_address",
    "exact_postal",
    "exact_country",

    # --------------------------------------------------------
    # Name similarities
    # --------------------------------------------------------
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_wratio",

    # --------------------------------------------------------
    # Core name
    # --------------------------------------------------------
    "core_name_ratio",
    "core_name_token_sort",
    "core_name_token_set",

    # --------------------------------------------------------
    # Transliteration
    # --------------------------------------------------------
    "translit_ratio",
    "translit_token_sort",
    "translit_token_set",

    # --------------------------------------------------------
    # Address similarities
    # --------------------------------------------------------
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "address_wratio",

    # --------------------------------------------------------
    # Name token overlap
    # --------------------------------------------------------
    "name_token_jaccard",
    "name_token_overlap",
    "name_token_containment",
    "name_shared_token_count",

    "name_first_token_match",
    "name_last_token_match",

    # --------------------------------------------------------
    # Address token overlap
    # --------------------------------------------------------
    "address_token_jaccard",
    "address_token_overlap",
    "address_token_containment",
    "address_shared_token_count",

    # --------------------------------------------------------
    # Numeric/address evidence
    # --------------------------------------------------------
    "number_jaccard",
    "number_overlap",
    "number_containment",
    "number_shared_count",
    "number_exact",

    # --------------------------------------------------------
    # Secondary fields
    # --------------------------------------------------------
    "same_postal",
    "postal_mismatch",
    "same_country_value",
    "country_mismatch",

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------
    "name_missing",
    "address_missing",
    "translit_missing",
    "postal_missing",
    "country_missing",

    # --------------------------------------------------------
    # Length / structure
    # --------------------------------------------------------
    "name_length_ratio",
    "address_length_ratio",
    "name_token_count_diff",
    "address_token_count_diff",

    # --------------------------------------------------------
    # Cross-field evidence
    # --------------------------------------------------------
    "name_address_mean",
    "name_address_min",
    "name_address_product",

    "strong_name_and_address",
    "strong_name_weak_address",
    "weak_name_strong_address",

    "name_and_number_agree",
    "name_agrees_address_number",
    "exact_name_or_translit",

    # --------------------------------------------------------
    # Retrieval provenance
    # --------------------------------------------------------
    "num_retrieval_channels",
    "strongest_route_priority",

    "route_exact_name",
    "route_exact_core",
    "route_exact_translit",
    "route_exact_address",
    "route_name_intersection",
    "route_address_intersection",
    "route_char_name",
    "route_char_address",
    "route_address_number_token",
    "route_address_number",
    "route_number_name",
    "route_rare_address",
    "route_postal_name",
    "route_rare_name",
    "route_prefix_name",
    "route_approx_address",
    "route_approx_name",
]


# ============================================================
# ROUTE PRIORITIES
# ============================================================

ROUTE_PRIORITIES = {
    "exact_name": 140,
    "exact_core": 135,
    "exact_translit": 130,
    "exact_address": 135,
    "name_intersection": 115,
    "address_intersection": 120,
    "char_name": 105,
    "char_address": 105,
    "address_number_token": 105,
    "address_number": 95,
    "number_name": 90,
    "rare_address": 88,
    "postal_name": 85,
    "rare_name": 75,
    "prefix_name": 55,
    "approx_address": 45,
    "approx_name": 40,
}


# ============================================================
# GROUND TRUTH
# ============================================================

def parse_ids(value):

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    if value.startswith("["):

        try:
            parsed = ast.literal_eval(value)

            return {
                str(x).strip()
                for x in parsed
                if str(x).strip()
            }

        except Exception:
            pass

    for separator in ["|", ",", ";"]:

        if separator in value:

            return {
                x.strip()
                for x in value.split(separator)
                if x.strip()
            }

    return {value}


def load_ground_truth():

    gt = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required - set(gt.columns)

    if missing:
        raise ValueError(
            f"Ground truth missing columns: {missing}"
        )

    truth = {}

    for row in gt.itertuples(index=False):

        truth[row.source1_entity_id] = parse_ids(
            row.matched_entity_ids
        )

    return truth


# ============================================================
# PROCESSED SCHEMA VALIDATION
# ============================================================

def validate_processed_schema(df, path):

    missing = (
        REQUIRED_PROCESSED_COLUMNS
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "\nProcessed file schema mismatch.\n"
            f"File: {path}\n"
            f"Missing required columns: {sorted(missing)}\n\n"
            "This is intentionally a hard failure. "
            "Do not silently continue with empty features."
        )


def validate_file_schema(path):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=2,
    )

    validate_processed_schema(
        df,
        path,
    )

    print(
        f"Schema OK: {path.name}"
    )


# ============================================================
# DETERMINISTIC S1 SPLIT
# ============================================================

def make_s1_split():

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    validate_processed_schema(
        s1,
        S1_PATH,
    )

    if TOTAL_DEV_SIZE > len(s1):

        raise ValueError(
            f"Requested {TOTAL_DEV_SIZE:,} S1 rows, "
            f"but only {len(s1):,} exist."
        )

    if (
        TRAIN_SIZE
        + VALID_SIZE
        + TEST_SIZE
        != TOTAL_DEV_SIZE
    ):

        raise ValueError(
            "TRAIN_SIZE + VALID_SIZE + TEST_SIZE "
            "must equal TOTAL_DEV_SIZE."
        )

    sampled = s1.sample(
        n=TOTAL_DEV_SIZE,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    train_ids = set(
        sampled.iloc[:TRAIN_SIZE]["entity_id"]
    )

    valid_ids = set(
        sampled.iloc[
            TRAIN_SIZE:
            TRAIN_SIZE + VALID_SIZE
        ]["entity_id"]
    )

    test_ids = set(
        sampled.iloc[
            TRAIN_SIZE + VALID_SIZE:
        ]["entity_id"]
    )

    print(
        "\nS1 development split:"
    )

    print(
        f"  Train:      {len(train_ids):,}"
    )

    print(
        f"  Validation: {len(valid_ids):,}"
    )

    print(
        f"  Test:       {len(test_ids):,}"
    )

    return train_ids, valid_ids, test_ids


# ============================================================
# STRING HELPERS
# ============================================================

def safe_string(value):

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    value = str(value).strip()

    if value.lower() in {
        "nan",
        "none",
        "null",
    }:
        return ""

    return value


def tokens(value):

    value = safe_string(value)

    if not value:
        return set()

    return {
        x
        for x in re.split(
            r"\s+",
            value,
        )
        if len(x) >= 2
    }


def parse_number_set(value):

    """
    Parse the already-computed *_numbers columns.

    Handles:
      - Python list/set representations
      - comma/pipe/semicolon separated strings
      - ordinary strings containing digits
    """

    value = safe_string(value)

    if not value:
        return set()

    if value.startswith(("[", "{", "(")):

        try:

            parsed = ast.literal_eval(value)

            if isinstance(
                parsed,
                (list, tuple, set),
            ):

                return {
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                }

        except Exception:
            pass

    pieces = re.split(
        r"[|,;]\s*",
        value,
    )

    result = set()

    for piece in pieces:

        piece = piece.strip()

        if not piece:
            continue

        found = re.findall(
            r"\d+",
            piece,
        )

        if found:
            result.update(found)
        else:
            result.add(piece)

    return result


def ratio(a, b):

    a = safe_string(a)
    b = safe_string(b)

    if not a or not b:
        return 0.0

    return fuzz.ratio(
        a,
        b,
    ) / 100.0


def token_sort(a, b):

    a = safe_string(a)
    b = safe_string(b)

    if not a or not b:
        return 0.0

    return (
        fuzz.token_sort_ratio(
            a,
            b,
        )
        / 100.0
    )


def token_set(a, b):

    a = safe_string(a)
    b = safe_string(b)

    if not a or not b:
        return 0.0

    return (
        fuzz.token_set_ratio(
            a,
            b,
        )
        / 100.0
    )


def wratio(a, b):

    a = safe_string(a)
    b = safe_string(b)

    if not a or not b:
        return 0.0

    return (
        fuzz.WRatio(
            a,
            b,
        )
        / 100.0
    )


def overlap_stats(a, b):

    inter = len(a & b)

    if not a and not b:
        return 0.0, 0.0, 0.0

    union = len(a | b)

    jaccard = (
        inter / union
        if union
        else 0.0
    )

    minimum = min(
        len(a),
        len(b),
    )

    overlap = (
        inter / minimum
        if minimum
        else 0.0
    )

    containment = (
        inter / max(
            len(a),
            len(b),
        )
        if max(
            len(a),
            len(b),
        )
        else 0.0
    )

    return (
        jaccard,
        overlap,
        containment,
    )


def length_ratio(a, b):

    a = safe_string(a)
    b = safe_string(b)

    if not a or not b:
        return 0.0

    return (
        min(
            len(a),
            len(b),
        )
        / max(
            len(a),
            len(b),
        )
    )


# ============================================================
# RETRIEVAL CHANNEL PARSING
# ============================================================

def parse_channels(value):

    if pd.isna(value):
        return set()

    value = safe_string(value)

    if not value:
        return set()

    if value.startswith(("{", "[")):

        try:

            parsed = ast.literal_eval(value)

            if isinstance(
                parsed,
                (set, list, tuple),
            ):

                return {
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                }

        except Exception:
            pass

    for separator in ["|", ",", ";"]:

        if separator in value:

            return {
                x.strip()
                for x in value.split(separator)
                if x.strip()
            }

    return {value}


def route_feature(
    channels,
    route,
):

    return int(
        route in channels
    )


def strongest_route(channels):

    if not channels:
        return 0

    return max(
        (
            ROUTE_PRIORITIES.get(
                channel,
                0,
            )
            for channel in channels
        ),
        default=0,
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def compute_features(
    s1,
    target,
    channels,
):

    # --------------------------------------------------------
    # ACTUAL PROCESSED COLUMNS
    # --------------------------------------------------------

    name1 = safe_string(
        s1["name_norm"]
    )

    name2 = safe_string(
        target["name_norm"]
    )

    core1 = safe_string(
        s1["name_core"]
    )

    core2 = safe_string(
        target["name_core"]
    )

    translit1 = safe_string(
        s1["name_translit"]
    )

    translit2 = safe_string(
        target["name_translit"]
    )

    address1 = safe_string(
        s1["address_norm"]
    )

    address2 = safe_string(
        target["address_norm"]
    )

    postal1 = safe_string(
        s1["postal_code"]
    )

    postal2 = safe_string(
        target["postal_code"]
    )

    country1 = safe_string(
        s1["country_norm"]
    )

    country2 = safe_string(
        target["country_norm"]
    )

    # --------------------------------------------------------
    # TOKENS
    # --------------------------------------------------------

    name_tokens_1 = tokens(name1)
    name_tokens_2 = tokens(name2)

    address_tokens_1 = tokens(address1)
    address_tokens_2 = tokens(address2)

    # --------------------------------------------------------
    # PRECOMPUTED NUMBER SETS
    # --------------------------------------------------------

    number_tokens_1 = parse_number_set(
        s1["address_numbers"]
    )

    number_tokens_2 = parse_number_set(
        target["address_numbers"]
    )

    (
        name_jaccard,
        name_overlap,
        name_containment,
    ) = overlap_stats(
        name_tokens_1,
        name_tokens_2,
    )

    (
        address_jaccard,
        address_overlap,
        address_containment,
    ) = overlap_stats(
        address_tokens_1,
        address_tokens_2,
    )

    (
        number_jaccard,
        number_overlap,
        number_containment,
    ) = overlap_stats(
        number_tokens_1,
        number_tokens_2,
    )

    # --------------------------------------------------------
    # EXACT FEATURES
    # --------------------------------------------------------

    exact_name = int(
        bool(name1)
        and name1 == name2
    )

    exact_core_name = int(
        bool(core1)
        and core1 == core2
    )

    exact_translit = int(
        bool(translit1)
        and translit1 == translit2
    )

    exact_address = int(
        bool(address1)
        and address1 == address2
    )

    same_postal = int(
        bool(postal1)
        and bool(postal2)
        and postal1 == postal2
    )

    postal_mismatch = int(
        bool(postal1)
        and bool(postal2)
        and postal1 != postal2
    )

    same_country = int(
        bool(country1)
        and bool(country2)
        and country1 == country2
    )

    country_mismatch = int(
        bool(country1)
        and bool(country2)
        and country1 != country2
    )

    # --------------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------------

    name_ratio_value = ratio(
        name1,
        name2,
    )

    name_token_sort_value = token_sort(
        name1,
        name2,
    )

    name_token_set_value = token_set(
        name1,
        name2,
    )

    name_wratio_value = wratio(
        name1,
        name2,
    )

    core_ratio_value = ratio(
        core1,
        core2,
    )

    core_token_sort_value = token_sort(
        core1,
        core2,
    )

    core_token_set_value = token_set(
        core1,
        core2,
    )

    translit_ratio_value = ratio(
        translit1,
        translit2,
    )

    translit_token_sort_value = token_sort(
        translit1,
        translit2,
    )

    translit_token_set_value = token_set(
        translit1,
        translit2,
    )

    address_ratio_value = ratio(
        address1,
        address2,
    )

    address_token_sort_value = token_sort(
        address1,
        address2,
    )

    address_token_set_value = token_set(
        address1,
        address2,
    )

    address_wratio_value = wratio(
        address1,
        address2,
    )

    # --------------------------------------------------------
    # STRUCTURAL TOKEN EVIDENCE
    # --------------------------------------------------------

    name_shared_token_count = len(
        name_tokens_1
        & name_tokens_2
    )

    address_shared_token_count = len(
        address_tokens_1
        & address_tokens_2
    )

    number_shared_count = len(
        number_tokens_1
        & number_tokens_2
    )

    name1_list = name1.split()
    name2_list = name2.split()

    name_first_token_match = int(
        bool(name1_list)
        and bool(name2_list)
        and name1_list[0]
        == name2_list[0]
    )

    name_last_token_match = int(
        bool(name1_list)
        and bool(name2_list)
        and name1_list[-1]
        == name2_list[-1]
    )

    number_exact = int(
        bool(number_tokens_1)
        and bool(number_tokens_2)
        and number_tokens_1
        == number_tokens_2
    )

    # --------------------------------------------------------
    # CROSS-FIELD EVIDENCE
    # --------------------------------------------------------

    name_address_mean = (
        name_ratio_value
        + address_ratio_value
    ) / 2.0

    name_address_min = min(
        name_ratio_value,
        address_ratio_value,
    )

    name_address_product = (
        name_ratio_value
        * address_ratio_value
    )

    strong_name_and_address = int(
        name_ratio_value >= 0.85
        and address_ratio_value >= 0.70
    )

    strong_name_weak_address = int(
        name_ratio_value >= 0.85
        and address_ratio_value < 0.40
    )

    weak_name_strong_address = int(
        name_ratio_value < 0.40
        and address_ratio_value >= 0.85
    )

    name_and_number_agree = int(
        number_shared_count > 0
        and name_ratio_value >= 0.70
    )

    name_agrees_address_number = int(
        name_ratio_value >= 0.70
        and number_jaccard > 0
    )

    exact_name_or_translit = int(
        exact_name
        or exact_translit
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        # ====================================================
        # EXACT
        # ====================================================

        "exact_name": exact_name,

        "exact_core_name": exact_core_name,

        "exact_translit": exact_translit,

        "exact_address": exact_address,

        "exact_postal": same_postal,

        "exact_country": same_country,

        # ====================================================
        # NAME
        # ====================================================

        "name_ratio": name_ratio_value,

        "name_token_sort": name_token_sort_value,

        "name_token_set": name_token_set_value,

        "name_wratio": name_wratio_value,

        # ====================================================
        # CORE NAME
        # ====================================================

        "core_name_ratio": core_ratio_value,

        "core_name_token_sort": (
            core_token_sort_value
        ),

        "core_name_token_set": (
            core_token_set_value
        ),

        # ====================================================
        # TRANSLITERATION
        # ====================================================

        "translit_ratio": (
            translit_ratio_value
        ),

        "translit_token_sort": (
            translit_token_sort_value
        ),

        "translit_token_set": (
            translit_token_set_value
        ),

        # ====================================================
        # ADDRESS
        # ====================================================

        "address_ratio": (
            address_ratio_value
        ),

        "address_token_sort": (
            address_token_sort_value
        ),

        "address_token_set": (
            address_token_set_value
        ),

        "address_wratio": (
            address_wratio_value
        ),

        # ====================================================
        # NAME TOKENS
        # ====================================================

        "name_token_jaccard": name_jaccard,

        "name_token_overlap": name_overlap,

        "name_token_containment": (
            name_containment
        ),

        "name_shared_token_count": (
            name_shared_token_count
        ),

        "name_first_token_match": (
            name_first_token_match
        ),

        "name_last_token_match": (
            name_last_token_match
        ),

        # ====================================================
        # ADDRESS TOKENS
        # ====================================================

        "address_token_jaccard": (
            address_jaccard
        ),

        "address_token_overlap": (
            address_overlap
        ),

        "address_token_containment": (
            address_containment
        ),

        "address_shared_token_count": (
            address_shared_token_count
        ),

        # ====================================================
        # NUMBERS
        # ====================================================

        "number_jaccard": number_jaccard,

        "number_overlap": number_overlap,

        "number_containment": (
            number_containment
        ),

        "number_shared_count": (
            number_shared_count
        ),

        "number_exact": number_exact,

        # ====================================================
        # SECONDARY
        # ====================================================

        "same_postal": same_postal,

        "postal_mismatch": postal_mismatch,

        "same_country_value": same_country,

        "country_mismatch": country_mismatch,

        # ====================================================
        # MISSINGNESS
        # ====================================================

        "name_missing": int(
            not name1
            or not name2
        ),

        "address_missing": int(
            not address1
            or not address2
        ),

        "translit_missing": int(
            not translit1
            or not translit2
        ),

        "postal_missing": int(
            not postal1
            or not postal2
        ),

        "country_missing": int(
            not country1
            or not country2
        ),

        # ====================================================
        # STRUCTURE
        # ====================================================

        "name_length_ratio": (
            length_ratio(
                name1,
                name2,
            )
        ),

        "address_length_ratio": (
            length_ratio(
                address1,
                address2,
            )
        ),

        "name_token_count_diff": abs(
            len(name_tokens_1)
            - len(name_tokens_2)
        ),

        "address_token_count_diff": abs(
            len(address_tokens_1)
            - len(address_tokens_2)
        ),

        # ====================================================
        # CROSS-FIELD
        # ====================================================

        "name_address_mean": (
            name_address_mean
        ),

        "name_address_min": (
            name_address_min
        ),

        "name_address_product": (
            name_address_product
        ),

        "strong_name_and_address": (
            strong_name_and_address
        ),

        "strong_name_weak_address": (
            strong_name_weak_address
        ),

        "weak_name_strong_address": (
            weak_name_strong_address
        ),

        "name_and_number_agree": (
            name_and_number_agree
        ),

        "name_agrees_address_number": (
            name_agrees_address_number
        ),

        "exact_name_or_translit": (
            exact_name_or_translit
        ),

        # ====================================================
        # RETRIEVAL PROVENANCE
        # ====================================================

        "num_retrieval_channels": len(
            channels
        ),

        "strongest_route_priority": (
            strongest_route(channels)
        ),

        "route_exact_name": route_feature(
            channels,
            "exact_name",
        ),

        "route_exact_core": route_feature(
            channels,
            "exact_core",
        ),

        "route_exact_translit": route_feature(
            channels,
            "exact_translit",
        ),

        "route_exact_address": route_feature(
            channels,
            "exact_address",
        ),

        "route_name_intersection": route_feature(
            channels,
            "name_intersection",
        ),

        "route_address_intersection": route_feature(
            channels,
            "address_intersection",
        ),

        "route_char_name": route_feature(
            channels,
            "char_name",
        ),

        "route_char_address": route_feature(
            channels,
            "char_address",
        ),

        "route_address_number_token": route_feature(
            channels,
            "address_number_token",
        ),

        "route_address_number": route_feature(
            channels,
            "address_number",
        ),

        "route_number_name": route_feature(
            channels,
            "number_name",
        ),

        "route_rare_address": route_feature(
            channels,
            "rare_address",
        ),

        "route_postal_name": route_feature(
            channels,
            "postal_name",
        ),

        "route_rare_name": route_feature(
            channels,
            "rare_name",
        ),

        "route_prefix_name": route_feature(
            channels,
            "prefix_name",
        ),

        "route_approx_address": route_feature(
            channels,
            "approx_address",
        ),

        "route_approx_name": route_feature(
            channels,
            "approx_name",
        ),
    }

    return result


# ============================================================
# ROUTE / FEATURE SANITY CHECK
# ============================================================

def validate_feature_sanity(
    feature_row,
    channels,
):

    failures = []

    if (
        "exact_name" in channels
        and feature_row["exact_name"] != 1
    ):
        failures.append(
            "route exact_name but exact_name feature != 1"
        )

    if (
        "exact_core" in channels
        and feature_row["exact_core_name"] != 1
    ):
        failures.append(
            "route exact_core but exact_core_name feature != 1"
        )

    if (
        "exact_translit" in channels
        and feature_row["exact_translit"] != 1
    ):
        failures.append(
            "route exact_translit but exact_translit feature != 1"
        )

    if (
        "exact_address" in channels
        and feature_row["exact_address"] != 1
    ):
        failures.append(
            "route exact_address but exact_address feature != 1"
        )

    if failures:

        raise RuntimeError(
            "\nFeature / candidate-route inconsistency detected.\n"
            + "\n".join(
                f"  - {x}"
                for x in failures
            )
            + "\n\n"
            "This means candidate generation and matcher "
            "normalization are not using the same fields."
        )


# ============================================================
# LOAD TARGET RECORDS
# ============================================================

def load_targets():

    print("\nLoading S2...")

    s2 = pd.read_csv(
        S2_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    validate_processed_schema(
        s2,
        S2_PATH,
    )

    print(
        f"S2 rows: {len(s2):,}"
    )

    print("\nLoading S3...")

    s3 = pd.read_csv(
        S3_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    validate_processed_schema(
        s3,
        S3_PATH,
    )

    print(
        f"S3 rows: {len(s3):,}"
    )

    targets = pd.concat(
        [
            s2,
            s3,
        ],
        ignore_index=True,
    )

    targets = targets.drop_duplicates(
        subset=["entity_id"],
        keep="first",
    )

    targets = targets.set_index(
        "entity_id"
    )

    print(
        f"Unique target entities: "
        f"{len(targets):,}"
    )

    return targets


# ============================================================
# READ CANDIDATES FOR A PARTICULAR S1 SET
# ============================================================

def load_candidate_pairs_for_ids(
    allowed_ids,
):

    pieces = []

    reader = pd.read_csv(
        CANDIDATES,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    total = 0

    for chunk in reader:

        chunk = chunk[
            chunk["s1_entity_id"].isin(
                allowed_ids
            )
        ]

        if len(chunk):

            pieces.append(chunk)

            total += len(chunk)

    if not pieces:

        return pd.DataFrame()

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    print(
        f"Loaded {total:,} candidate pairs."
    )

    return result


# ============================================================
# LOAD S1 LOOKUP
# ============================================================

def load_s1_lookup():

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    validate_processed_schema(
        s1,
        S1_PATH,
    )

    return {
        row.entity_id: row._asdict()
        for row in s1.itertuples(index=False)
    }


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

def build_features(
    candidates,
    s1_lookup,
    targets,
    truth,
    make_labels=True,
):

    rows = []
    labels = []

    total = len(candidates)

    sanity_checked = False

    for i, row in enumerate(
        candidates.itertuples(index=False),
        start=1,
    ):

        s1_id = row.s1_entity_id
        target_id = row.candidate_entity_id

        s1 = s1_lookup.get(
            s1_id
        )

        if s1 is None:

            raise RuntimeError(
                f"S1 entity missing from lookup: "
                f"{s1_id}"
            )

        if target_id not in targets.index:

            raise RuntimeError(
                f"Target entity missing from lookup: "
                f"{target_id}"
            )

        target = targets.loc[
            target_id
        ]

        if isinstance(
            target,
            pd.Series,
        ):

            target_dict = target.to_dict()

        else:

            target_dict = dict(target)

        channels = parse_channels(
            row.retrieval_channels
        )

        feature_row = compute_features(
            s1,
            target_dict,
            channels,
        )

        # ----------------------------------------------------
        # Critical sanity check.
        #
        # Run on first candidate and then periodically.
        # ----------------------------------------------------

        if (
            not sanity_checked
            or i % 100_000 == 0
        ):

            validate_feature_sanity(
                feature_row,
                channels,
            )

            sanity_checked = True

        rows.append(
            feature_row
        )

        if make_labels:

            labels.append(
                int(
                    target_id
                    in truth.get(
                        s1_id,
                        set(),
                    )
                )
            )

        if i % 100_000 == 0:

            print(
                f"Feature rows: "
                f"{i:,}/{total:,}"
            )

    X = pd.DataFrame(
        rows,
        columns=FEATURE_COLUMNS,
    )

    # --------------------------------------------------------
    # Global sanity check.
    #
    # These should NOT all be zero anymore.
    # --------------------------------------------------------

    similarity_columns = [
        "name_ratio",
        "name_token_set",
        "core_name_ratio",
        "translit_ratio",
        "address_ratio",
        "address_token_set",
        "name_token_jaccard",
        "address_token_jaccard",
        "number_jaccard",
    ]

    print(
        "\nFeature sanity summary:"
    )

    for column in similarity_columns:

        print(
            f"  {column:28s} "
            f"mean={X[column].mean():.6f} "
            f"max={X[column].max():.6f} "
            f"nonzero="
            f"{(X[column] > 0).mean():.2%}"
        )

    if make_labels:

        y = np.asarray(
            labels,
            dtype=np.int8,
        )

        return X, y

    return X


# ============================================================
# HARD-NEGATIVE / RANDOM-NEGATIVE SAMPLING
# ============================================================

def sample_training_data(
    X,
    y,
):

    positive_indices = np.flatnonzero(
        y == 1
    )

    negative_indices = np.flatnonzero(
        y == 0
    )

    print(
        "\nAvailable training examples:"
    )

    print(
        f"  Positive: {len(positive_indices):,}"
    )

    print(
        f"  Negative: {len(negative_indices):,}"
    )

    if not len(positive_indices):

        raise RuntimeError(
            "No positive training pairs found."
        )

    target_negative_count = min(
        len(negative_indices),
        len(positive_indices)
        * NEGATIVE_TO_POSITIVE_RATIO,
    )

    # --------------------------------------------------------
    # Hardness
    # --------------------------------------------------------

    hard_score_columns = [
        "name_ratio",
        "name_token_sort",
        "name_token_set",
        "name_wratio",

        "core_name_ratio",
        "core_name_token_sort",
        "core_name_token_set",

        "translit_ratio",
        "translit_token_sort",

        "address_ratio",
        "address_token_sort",
        "address_token_set",
        "address_wratio",

        "name_token_jaccard",
        "address_token_jaccard",

        "number_jaccard",
        "same_postal",
        "number_exact",
    ]

    available_columns = [
        c
        for c in hard_score_columns
        if c in X.columns
    ]

    negative_X = X.iloc[
        negative_indices
    ]

    hardness = negative_X[
        available_columns
    ].max(axis=1)

    # --------------------------------------------------------
    # 50% hard / 50% random
    # --------------------------------------------------------

    hard_count = min(
        target_negative_count // 2,
        len(negative_indices),
    )

    random_count = (
        target_negative_count
        - hard_count
    )

    hard_order = np.argsort(
        hardness.to_numpy()
    )[::-1]

    hard_indices = negative_indices[
        hard_order[:hard_count]
    ]

    hard_set = set(
        hard_indices.tolist()
    )

    remaining = np.asarray(
        [
            idx
            for idx in negative_indices
            if idx not in hard_set
        ],
        dtype=np.int64,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    if random_count > len(remaining):

        random_count = len(
            remaining
        )

    random_indices = rng.choice(
        remaining,
        size=random_count,
        replace=False,
    )

    selected_negatives = np.concatenate(
        [
            hard_indices,
            random_indices,
        ]
    )

    selected = np.concatenate(
        [
            positive_indices,
            selected_negatives,
        ]
    )

    rng.shuffle(
        selected
    )

    X_train = X.iloc[
        selected
    ].reset_index(drop=True)

    y_train = y[
        selected
    ]

    print(
        "\nActual classifier training set:"
    )

    print(
        f"  Rows: {len(X_train):,}"
    )

    print(
        f"  Positive: "
        f"{(y_train == 1).sum():,}"
    )

    print(
        f"  Negative: "
        f"{(y_train == 0).sum():,}"
    )

    print(
        f"  Hard negatives: "
        f"{len(hard_indices):,}"
    )

    print(
        f"  Random negatives: "
        f"{len(random_indices):,}"
    )

    return X_train, y_train


# ============================================================
# F0.5
# ============================================================

def f05_score(
    y_true,
    y_pred,
):

    tp = np.sum(
        (y_true == 1)
        & (y_pred == 1)
    )

    fp = np.sum(
        (y_true == 0)
        & (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1)
        & (y_pred == 0)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    if precision + recall == 0:

        return (
            0.0,
            precision,
            recall,
        )

    f05 = (
        1.25
        * precision
        * recall
        / (
            0.25 * precision
            + recall
        )
    )

    return (
        f05,
        precision,
        recall,
    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
):

    best = None

    thresholds = np.arange(
        0.10,
        0.996,
        0.01,
    )

    results = []

    print(
        "\nValidation threshold search:"
    )

    print(
        "-" * 70
    )

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>12} "
        f"{'F0.5':>12} "
        f"{'Predicted':>12}"
    )

    print(
        "-" * 70
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.int8)

        f05, precision, recall = (
            f05_score(
                y_true,
                predictions,
            )
        )

        predicted_count = int(
            predictions.sum()
        )

        result = (
            f05,
            precision,
            recall,
            threshold,
            predicted_count,
        )

        results.append(
            result
        )

        if (
            best is None
            or f05 > best[0]
        ):

            best = result

    for result in results:

        (
            f05,
            precision,
            recall,
            threshold,
            predicted,
        ) = result

        if (
            threshold <= 0.50
            or abs(
                threshold * 100
                - round(
                    threshold * 100 / 5
                ) * 5
            ) < 1e-9
        ):

            print(
                f"{threshold:10.2f} "
                f"{precision:12.6f} "
                f"{recall:12.6f} "
                f"{f05:12.6f} "
                f"{predicted:12,}"
            )

    (
        f05,
        precision,
        recall,
        threshold,
        predicted,
    ) = best

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST VALIDATION THRESHOLD"
    )

    print(
        "=" * 70
    )

    print(
        f"Threshold: {threshold:.3f}"
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall:    {recall:.6f}"
    )

    print(
        f"F0.5:      {f05:.6f}"
    )

    print(
        f"Predicted: {predicted:,}"
    )

    return float(
        threshold
    )


# ============================================================
# TRAIN
# ============================================================

def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
):

    print(
        "\nTraining LightGBM matcher..."
    )

    model = lgb.LGBMClassifier(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_valid,
                y_valid,
            )
        ],
        callbacks=[
            lgb.early_stopping(
                50,
                verbose=False,
            )
        ],
    )

    probabilities = (
        model.predict_proba(
            X_valid
        )[:, 1]
    )

    threshold = find_best_threshold(
        y_valid,
        probabilities,
    )

    return (
        model,
        threshold,
    )


# ============================================================
# MODEL FEATURE IMPORTANCE
# ============================================================

def print_feature_importance(
    model,
):

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        "\nTop model features:"
    )

    print(
        importance.head(25).to_string(
            index=False
        )
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    threshold,
):

    payload = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "random_seed": RANDOM_SEED,
    }

    with open(
        MODEL_PATH,
        "wb",
    ) as f:

        pickle.dump(
            payload,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    metadata = {
        "threshold": threshold,
        "feature_count": len(
            FEATURE_COLUMNS
        ),
        "train_size": TRAIN_SIZE,
        "validation_size": VALID_SIZE,
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "negative_positive_ratio": (
            NEGATIVE_TO_POSITIVE_RATIO
        ),
        "processed_name_column": "name_norm",
        "processed_core_name_column": "name_core",
        "processed_translit_column": "name_translit",
        "processed_address_column": "address_norm",
        "processed_postal_column": "postal_code",
        "processed_country_column": "country_norm",
    }

    with open(
        MODEL_META_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"\nSaved model:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print(
        f"Saved metadata:"
    )

    print(
        f"  {MODEL_META_PATH}"
    )


# ============================================================
# SAVE VALIDATION DIAGNOSTICS
# ============================================================

def save_validation_diagnostics(
    candidates,
    X,
    y,
    probabilities,
    threshold,
):

    diagnostics = candidates.reset_index(
        drop=True
    ).copy()

    diagnostics["true_label"] = y

    diagnostics[
        "match_probability"
    ] = probabilities

    diagnostics[
        "match_prediction"
    ] = (
        probabilities >= threshold
    ).astype(np.int8)

    diagnostics["error_type"] = np.select(
        [
            (y == 1)
            & (probabilities >= threshold),

            (y == 0)
            & (probabilities >= threshold),

            (y == 1)
            & (probabilities < threshold),

            (y == 0)
            & (probabilities < threshold),
        ],
        [
            "TP",
            "FP",
            "FN",
            "TN",
        ],
        default="UNKNOWN",
    )

    feature_frame = X.reset_index(
        drop=True
    ).copy()

    for column in feature_frame.columns:

        diagnostics[column] = (
            feature_frame[column].values
        )

    diagnostics["model_rank"] = (
        diagnostics
        .groupby(
            "s1_entity_id"
        )[
            "match_probability"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    diagnostics.to_csv(
        VALIDATION_DIAGNOSTICS,
        sep="\t",
        index=False,
    )

    print(
        "\nSaved validation diagnostics:"
    )

    print(
        f"  {VALIDATION_DIAGNOSTICS}"
    )

    print(
        f"  Rows: {len(diagnostics):,}"
    )

    return diagnostics


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "03 — PAIRWISE ENTITY MATCHER"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # SCHEMA CHECK
    # ========================================================

    print(
        "\nValidating processed schemas..."
    )

    validate_file_schema(
        S1_PATH
    )

    validate_file_schema(
        S2_PATH
    )

    validate_file_schema(
        S3_PATH
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    print(
        "\nLoading ground truth..."
    )

    truth = load_ground_truth()

    print(
        f"Ground-truth S1 entities: "
        f"{len(truth):,}"
    )

    train_ids, valid_ids, test_ids = (
        make_s1_split()
    )

    # ========================================================
    # CANDIDATE COVERAGE
    # ========================================================

    candidate_s1_ids = set()

    print(
        "\nChecking candidate coverage..."
    )

    reader = pd.read_csv(
        CANDIDATES,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "s1_entity_id"
        ],
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:

        candidate_s1_ids.update(
            chunk[
                "s1_entity_id"
            ].unique()
        )

    missing_train = (
        train_ids
        - candidate_s1_ids
    )

    missing_valid = (
        valid_ids
        - candidate_s1_ids
    )

    missing_test = (
        test_ids
        - candidate_s1_ids
    )

    print(
        f"Train S1 IDs missing from candidates: "
        f"{len(missing_train):,}"
    )

    print(
        f"Validation S1 IDs missing: "
        f"{len(missing_valid):,}"
    )

    print(
        f"Test S1 IDs missing: "
        f"{len(missing_test):,}"
    )

    # ========================================================
    # LOAD RECORDS
    # ========================================================

    s1_lookup = load_s1_lookup()

    targets = load_targets()

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BUILDING TRAINING FEATURES"
    )

    print(
        "=" * 70
    )

    train_candidates = (
        load_candidate_pairs_for_ids(
            train_ids
        )
    )

    X_train_all, y_train_all = (
        build_features(
            train_candidates,
            s1_lookup,
            targets,
            truth,
            make_labels=True,
        )
    )

    X_train, y_train = (
        sample_training_data(
            X_train_all,
            y_train_all,
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BUILDING VALIDATION FEATURES"
    )

    print(
        "=" * 70
    )

    valid_candidates = (
        load_candidate_pairs_for_ids(
            valid_ids
        )
    )

    X_valid, y_valid = (
        build_features(
            valid_candidates,
            s1_lookup,
            targets,
            truth,
            make_labels=True,
        )
    )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    model, threshold = train_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    print_feature_importance(
        model
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    save_model(
        model,
        threshold,
    )

    # ========================================================
    # SCORE VALIDATION
    # ========================================================

    print(
        "\nScoring validation candidates..."
    )

    valid_probabilities = (
        model.predict_proba(
            X_valid
        )[:, 1]
    )

    valid_predictions = (
        valid_probabilities
        >= threshold
    ).astype(np.int8)

    (
        f05,
        precision,
        recall,
    ) = f05_score(
        y_valid,
        valid_predictions,
    )

    print(
        "\nValidation matcher result"
    )

    print(
        "-" * 70
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall:    {recall:.6f}"
    )

    print(
        f"F0.5:      {f05:.6f}"
    )

    # ========================================================
    # SAVE VALIDATION DIAGNOSTICS
    # ========================================================

    save_validation_diagnostics(
        candidates=valid_candidates,
        X=X_valid,
        y=y_valid,
        probabilities=valid_probabilities,
        threshold=threshold,
    )

    # ========================================================
    # TEST
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SCORING HELD-OUT TEST"
    )

    print(
        "=" * 70
    )

    test_candidates = (
        load_candidate_pairs_for_ids(
            test_ids
        )
    )

    X_test, y_test = (
        build_features(
            test_candidates,
            s1_lookup,
            targets,
            truth,
            make_labels=True,
        )
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= threshold
    ).astype(np.int8)

    (
        test_f05,
        test_precision,
        test_recall,
    ) = f05_score(
        y_test,
        test_predictions,
    )

    print(
        "\nHeld-out TEST matcher result"
    )

    print(
        "-" * 70
    )

    print(
        f"Precision: {test_precision:.6f}"
    )

    print(
        f"Recall:    {test_recall:.6f}"
    )

    print(
        f"F0.5:      {test_f05:.6f}"
    )

    # ========================================================
    # WRITE TEST OUTPUT
    # ========================================================

    scored_test = (
        test_candidates.copy()
    )

    scored_test[
        "match_probability"
    ] = test_probabilities

    scored_test[
        "match_prediction"
    ] = test_predictions

    scored_test.to_csv(
        SCORED_CANDIDATES,
        sep="\t",
        index=False,
    )

    final_test = scored_test[
        scored_test[
            "match_prediction"
        ] == 1
    ].copy()

    final_test.to_csv(
        FINAL_CANDIDATES,
        sep="\t",
        index=False,
    )

    print(
        f"\nTest candidate pairs: "
        f"{len(test_candidates):,}"
    )

    print(
        f"Test predicted pairs: "
        f"{len(final_test):,}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "03 COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()