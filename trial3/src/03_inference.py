from pathlib import Path
import ast
import json
import pickle
import re
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "LightGBM is required.\n"
        "Install with: pip install lightgbm"
    ) from e


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "processed"
CANDIDATE_DIR = ROOT / "candidates"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

S1_PATH = PROCESSED_DIR / "test_S1_processed.tsv"
S2_PATH = PROCESSED_DIR / "test_S2_processed.tsv"
S3_PATH = PROCESSED_DIR / "test_S3_processed.tsv"

CANDIDATES = CANDIDATE_DIR / "test_candidate_pairs.tsv"

MODEL_PATH = MODEL_DIR / "entity_matcher.pkl"
MODEL_META_PATH = MODEL_DIR / "entity_matcher_meta.json"

MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
FINAL_CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"


# ============================================================
# RUNTIME
# ============================================================

# Candidate file is potentially ~380M rows.
# 50k keeps memory reasonable while still giving LightGBM
# reasonably sized prediction batches.
CANDIDATE_CHUNK_SIZE = 50_000

# Progress message frequency.
PROGRESS_EVERY = 500_000

# Frozen from development.
THRESHOLD = 0.970


# ============================================================
# EXPECTED PROCESSED SCHEMA
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
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "exact_name",
    "exact_core_name",
    "exact_translit",
    "exact_address",
    "exact_postal",
    "exact_country",

    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_wratio",

    "core_name_ratio",
    "core_name_token_sort",
    "core_name_token_set",

    "translit_ratio",
    "translit_token_sort",
    "translit_token_set",

    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "address_wratio",

    "name_token_jaccard",
    "name_token_overlap",
    "name_token_containment",
    "name_shared_token_count",
    "name_first_token_match",
    "name_last_token_match",

    "address_token_jaccard",
    "address_token_overlap",
    "address_token_containment",
    "address_shared_token_count",

    "number_jaccard",
    "number_overlap",
    "number_containment",
    "number_shared_count",
    "number_exact",

    "same_postal",
    "postal_mismatch",
    "same_country_value",
    "country_mismatch",

    "name_missing",
    "address_missing",
    "translit_missing",
    "postal_missing",
    "country_missing",

    "name_length_ratio",
    "address_length_ratio",
    "name_token_count_diff",
    "address_token_count_diff",

    "name_address_mean",
    "name_address_min",
    "name_address_product",

    "strong_name_and_address",
    "strong_name_weak_address",
    "weak_name_strong_address",

    "name_and_number_agree",
    "name_agrees_address_number",
    "exact_name_or_translit",

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
# SCHEMA
# ============================================================

def validate_processed_schema(df, path):

    missing = REQUIRED_PROCESSED_COLUMNS - set(df.columns)

    if missing:
        raise RuntimeError(
            "\nProcessed file schema mismatch.\n"
            f"File: {path}\n"
            f"Missing required columns: {sorted(missing)}"
        )


def validate_file_schema(path):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=2,
    )

    validate_processed_schema(df, path)

    print(f"Schema OK: {path.name}")


# ============================================================
# STRING / TOKEN HELPERS
# ============================================================

def safe_string(value):

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    value = str(value).strip()

    if value.lower() in {"nan", "none", "null"}:
        return ""

    return value


def tokens(value):

    value = safe_string(value)

    if not value:
        return set()

    return {
        x
        for x in re.split(r"\s+", value)
        if len(x) >= 2
    }


def parse_number_set(value):

    value = safe_string(value)

    if not value:
        return set()

    if value.startswith(("[", "{", "(")):

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple, set)):
                return {
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                }

        except Exception:
            pass

    pieces = re.split(r"[|,;]\s*", value)

    result = set()

    for piece in pieces:

        piece = piece.strip()

        if not piece:
            continue

        found = re.findall(r"\d+", piece)

        if found:
            result.update(found)
        else:
            result.add(piece)

    return result


# ============================================================
# PREPARE ONE PROCESSED DATAFRAME
# ============================================================

def prepare_dataframe(df):

    """
    Precompute everything that depends on ONE record.

    This is the major runtime optimization.

    The old implementation repeatedly performed:
        tokens(...)
        parse_number_set(...)
        split(...)
        safe_string(...)

    for every candidate pair.

    Here they are done once per record.
    """

    # Ensure required columns exist.
    validate_processed_schema(df, "in-memory dataframe")

    # Normalize primary string columns once.
    string_columns = [
        "name_norm",
        "name_core",
        "name_translit",
        "address_norm",
        "country_norm",
        "postal_code",
    ]

    for col in string_columns:
        df[col] = df[col].map(safe_string)

    # Precompute token sets.
    df["_name_tokens"] = df["name_norm"].map(tokens)
    df["_address_tokens"] = df["address_norm"].map(tokens)

    # Precompute numeric sets.
    df["_number_tokens"] = df["address_numbers"].map(
        parse_number_set
    )

    # Precompute lists.
    df["_name_list"] = df["name_norm"].str.split()

    # First / last tokens.
    df["_first_name_token"] = df["_name_list"].map(
        lambda x: x[0] if x else ""
    )

    df["_last_name_token"] = df["_name_list"].map(
        lambda x: x[-1] if x else ""
    )

    # Missingness.
    df["_name_missing"] = (
        df["name_norm"].eq("")
    )

    df["_address_missing"] = (
        df["address_norm"].eq("")
    )

    df["_translit_missing"] = (
        df["name_translit"].eq("")
    )

    df["_postal_missing"] = (
        df["postal_code"].eq("")
    )

    df["_country_missing"] = (
        df["country_norm"].eq("")
    )

    # Lengths.
    df["_name_length"] = df["name_norm"].str.len()
    df["_address_length"] = df["address_norm"].str.len()

    return df


# ============================================================
# ROUTE PARSING
# ============================================================

def parse_channels(value):

    if value is None:
        return set()

    value = safe_string(value)

    if not value:
        return set()

    if value.startswith(("{", "[")):

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (set, list, tuple)):
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


# ============================================================
# FAST CHANNEL DECODING
# ============================================================

# We encode channels into the exact route features needed by
# the model. This avoids repeatedly parsing strings where
# possible.

ROUTE_NAMES = [
    "exact_name",
    "exact_core",
    "exact_translit",
    "exact_address",
    "name_intersection",
    "address_intersection",
    "char_name",
    "char_address",
    "address_number_token",
    "address_number",
    "number_name",
    "rare_address",
    "postal_name",
    "rare_name",
    "prefix_name",
    "approx_address",
    "approx_name",
]

ROUTE_INDEX = {
    name: i
    for i, name in enumerate(ROUTE_NAMES)
}


def decode_channels(value):

    channels = parse_channels(value)

    flags = np.zeros(
        len(ROUTE_NAMES),
        dtype=np.int8,
    )

    strongest = 0

    for channel in channels:

        idx = ROUTE_INDEX.get(channel)

        if idx is not None:
            flags[idx] = 1

        priority = ROUTE_PRIORITIES.get(
            channel,
            0,
        )

        if priority > strongest:
            strongest = priority

    return (
        channels,
        flags,
        strongest,
    )


# ============================================================
# OVERLAP
# ============================================================

def overlap_stats(a, b):

    inter = len(a & b)

    if not a and not b:
        return 0.0, 0.0, 0.0, 0

    union = len(a | b)

    minimum = min(
        len(a),
        len(b),
    )

    maximum = max(
        len(a),
        len(b),
    )

    jaccard = (
        inter / union
        if union
        else 0.0
    )

    overlap = (
        inter / minimum
        if minimum
        else 0.0
    )

    containment = (
        inter / maximum
        if maximum
        else 0.0
    )

    return (
        jaccard,
        overlap,
        containment,
        inter,
    )


# ============================================================
# LENGTH RATIO
# ============================================================

def fast_length_ratio(len_a, len_b):

    if not len_a or not len_b:
        return 0.0

    return min(
        len_a,
        len_b,
    ) / max(
        len_a,
        len_b,
    )


# ============================================================
# FEATURE ROW
# ============================================================

def compute_feature_row(
    s1,
    target,
    channels,
    route_flags,
    strongest_priority,
):

    name1 = s1["name_norm"]
    name2 = target["name_norm"]

    core1 = s1["name_core"]
    core2 = target["name_core"]

    translit1 = s1["name_translit"]
    translit2 = target["name_translit"]

    address1 = s1["address_norm"]
    address2 = target["address_norm"]

    postal1 = s1["postal_code"]
    postal2 = target["postal_code"]

    country1 = s1["country_norm"]
    country2 = target["country_norm"]

    # --------------------------------------------------------
    # PRECOMPUTED SETS
    # --------------------------------------------------------

    name_tokens_1 = s1["_name_tokens"]
    name_tokens_2 = target["_name_tokens"]

    address_tokens_1 = s1["_address_tokens"]
    address_tokens_2 = target["_address_tokens"]

    number_tokens_1 = s1["_number_tokens"]
    number_tokens_2 = target["_number_tokens"]

    # --------------------------------------------------------
    # OVERLAPS
    # --------------------------------------------------------

    (
        name_jaccard,
        name_overlap,
        name_containment,
        name_shared,
    ) = overlap_stats(
        name_tokens_1,
        name_tokens_2,
    )

    (
        address_jaccard,
        address_overlap,
        address_containment,
        address_shared,
    ) = overlap_stats(
        address_tokens_1,
        address_tokens_2,
    )

    (
        number_jaccard,
        number_overlap,
        number_containment,
        number_shared,
    ) = overlap_stats(
        number_tokens_1,
        number_tokens_2,
    )

    # --------------------------------------------------------
    # EXACT
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
    # FUZZY
    # --------------------------------------------------------

    if name1 and name2:

        name_ratio_value = (
            fuzz.ratio(
                name1,
                name2,
            ) / 100.0
        )

        name_token_sort_value = (
            fuzz.token_sort_ratio(
                name1,
                name2,
            ) / 100.0
        )

        name_token_set_value = (
            fuzz.token_set_ratio(
                name1,
                name2,
            ) / 100.0
        )

        name_wratio_value = (
            fuzz.WRatio(
                name1,
                name2,
            ) / 100.0
        )

    else:

        name_ratio_value = 0.0
        name_token_sort_value = 0.0
        name_token_set_value = 0.0
        name_wratio_value = 0.0

    if core1 and core2:

        core_ratio_value = (
            fuzz.ratio(
                core1,
                core2,
            ) / 100.0
        )

        core_token_sort_value = (
            fuzz.token_sort_ratio(
                core1,
                core2,
            ) / 100.0
        )

        core_token_set_value = (
            fuzz.token_set_ratio(
                core1,
                core2,
            ) / 100.0
        )

    else:

        core_ratio_value = 0.0
        core_token_sort_value = 0.0
        core_token_set_value = 0.0

    if translit1 and translit2:

        translit_ratio_value = (
            fuzz.ratio(
                translit1,
                translit2,
            ) / 100.0
        )

        translit_token_sort_value = (
            fuzz.token_sort_ratio(
                translit1,
                translit2,
            ) / 100.0
        )

        translit_token_set_value = (
            fuzz.token_set_ratio(
                translit1,
                translit2,
            ) / 100.0
        )

    else:

        translit_ratio_value = 0.0
        translit_token_sort_value = 0.0
        translit_token_set_value = 0.0

    if address1 and address2:

        address_ratio_value = (
            fuzz.ratio(
                address1,
                address2,
            ) / 100.0
        )

        address_token_sort_value = (
            fuzz.token_sort_ratio(
                address1,
                address2,
            ) / 100.0
        )

        address_token_set_value = (
            fuzz.token_set_ratio(
                address1,
                address2,
            ) / 100.0
        )

        address_wratio_value = (
            fuzz.WRatio(
                address1,
                address2,
            ) / 100.0
        )

    else:

        address_ratio_value = 0.0
        address_token_sort_value = 0.0
        address_token_set_value = 0.0
        address_wratio_value = 0.0

    # --------------------------------------------------------
    # FIRST / LAST
    # --------------------------------------------------------

    first_match = int(
        bool(s1["_first_name_token"])
        and bool(target["_first_name_token"])
        and s1["_first_name_token"]
        == target["_first_name_token"]
    )

    last_match = int(
        bool(s1["_last_name_token"])
        and bool(target["_last_name_token"])
        and s1["_last_name_token"]
        == target["_last_name_token"]
    )

    number_exact = int(
        bool(number_tokens_1)
        and bool(number_tokens_2)
        and number_tokens_1 == number_tokens_2
    )

    # --------------------------------------------------------
    # CROSS-FIELD
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
        number_shared > 0
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
    # RETURN IN EXACT MODEL ORDER
    # --------------------------------------------------------

    return [
        exact_name,
        exact_core_name,
        exact_translit,
        exact_address,
        same_postal,
        same_country,

        name_ratio_value,
        name_token_sort_value,
        name_token_set_value,
        name_wratio_value,

        core_ratio_value,
        core_token_sort_value,
        core_token_set_value,

        translit_ratio_value,
        translit_token_sort_value,
        translit_token_set_value,

        address_ratio_value,
        address_token_sort_value,
        address_token_set_value,
        address_wratio_value,

        name_jaccard,
        name_overlap,
        name_containment,
        name_shared,

        first_match,
        last_match,

        address_jaccard,
        address_overlap,
        address_containment,
        address_shared,

        number_jaccard,
        number_overlap,
        number_containment,
        number_shared,
        number_exact,

        same_postal,
        postal_mismatch,
        same_country,
        country_mismatch,

        int(
            not name1
            or not name2
        ),

        int(
            not address1
            or not address2
        ),

        int(
            not translit1
            or not translit2
        ),

        int(
            not postal1
            or not postal2
        ),

        int(
            not country1
            or not country2
        ),

        fast_length_ratio(
            s1["_name_length"],
            target["_name_length"],
        ),

        fast_length_ratio(
            s1["_address_length"],
            target["_address_length"],
        ),

        abs(
            len(name_tokens_1)
            - len(name_tokens_2)
        ),

        abs(
            len(address_tokens_1)
            - len(address_tokens_2)
        ),

        name_address_mean,
        name_address_min,
        name_address_product,

        strong_name_and_address,
        strong_name_weak_address,
        weak_name_strong_address,

        name_and_number_agree,
        name_agrees_address_number,
        exact_name_or_translit,

        len(channels),
        strongest_priority,

        int(route_flags[ROUTE_INDEX["exact_name"]]),
        int(route_flags[ROUTE_INDEX["exact_core"]]),
        int(route_flags[ROUTE_INDEX["exact_translit"]]),
        int(route_flags[ROUTE_INDEX["exact_address"]]),
        int(route_flags[ROUTE_INDEX["name_intersection"]]),
        int(route_flags[ROUTE_INDEX["address_intersection"]]),
        int(route_flags[ROUTE_INDEX["char_name"]]),
        int(route_flags[ROUTE_INDEX["char_address"]]),
        int(route_flags[ROUTE_INDEX["address_number_token"]]),
        int(route_flags[ROUTE_INDEX["address_number"]]),
        int(route_flags[ROUTE_INDEX["number_name"]]),
        int(route_flags[ROUTE_INDEX["rare_address"]]),
        int(route_flags[ROUTE_INDEX["postal_name"]]),
        int(route_flags[ROUTE_INDEX["rare_name"]]),
        int(route_flags[ROUTE_INDEX["prefix_name"]]),
        int(route_flags[ROUTE_INDEX["approx_address"]]),
        int(route_flags[ROUTE_INDEX["approx_name"]]),
    ]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading saved model...")

    with open(
        MODEL_PATH,
        "rb",
    ) as f:

        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Saved model payload is not a dictionary."
        )

    model = payload.get("model")

    if model is None:
        raise RuntimeError(
            "Saved model payload does not contain 'model'."
        )

    saved_features = payload.get(
        "feature_columns"
    )

    if saved_features != FEATURE_COLUMNS:

        raise RuntimeError(
            "\nFEATURE COLUMN MISMATCH.\n"
            "The inference feature order does not exactly "
            "match the saved model."
        )

    saved_threshold = float(
        payload.get(
            "threshold",
            THRESHOLD,
        )
    )

    if abs(
        saved_threshold - THRESHOLD
    ) > 1e-12:

        raise RuntimeError(
            f"Saved threshold is {saved_threshold}, "
            f"but production threshold is {THRESHOLD}."
        )

    print(
        f"Model:     {MODEL_PATH}"
    )

    print(
        f"Features:  {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Threshold: {saved_threshold:.3f}"
    )

    return model


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

def load_processed_data():

    print("\nLoading S1...")

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

    print(
        f"S1 rows: {len(s1):,}"
    )

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

    print("\nPrecomputing S1 features...")

    s1 = prepare_dataframe(s1)

    print("Precomputing S2 features...")

    s2 = prepare_dataframe(s2)

    print("Precomputing S3 features...")

    s3 = prepare_dataframe(s3)

    return (
        s1,
        s2,
        s3,
    )


# ============================================================
# BUILD TARGET LOOKUP
# ============================================================

def build_target_lookup(
    s2,
    s3,
):

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
        "entity_id",
        drop=False,
    )

    print(
        f"\nUnique target entities: "
        f"{len(targets):,}"
    )

    return targets


# ============================================================
# BUILD S1 LOOKUP
# ============================================================

def build_s1_lookup(s1):

    return s1.set_index(
        "entity_id",
        drop=False,
    )


# ============================================================
# BUILD FEATURE MATRIX FOR CHUNK
# ============================================================

def build_feature_matrix(
    candidates,
    s1_lookup,
    target_lookup,
):

    n = len(candidates)

    X = np.empty(
        (
            n,
            len(FEATURE_COLUMNS),
        ),
        dtype=np.float32,
    )

    s1_ids = candidates[
        "s1_entity_id"
    ].to_numpy()

    target_ids = candidates[
        "candidate_entity_id"
    ].to_numpy()

    retrieval = candidates[
        "retrieval_channels"
    ].to_numpy()

    # --------------------------------------------------------
    # Local row caches.
    #
    # A candidate chunk often contains the same S1 many times.
    # Cache prepared row references and decoded channels.
    # --------------------------------------------------------

    s1_cache = {}
    target_cache = {}
    channel_cache = {}

    for i in range(n):

        s1_id = s1_ids[i]
        target_id = target_ids[i]

        # -----------------------------------------------
        # S1
        # -----------------------------------------------

        s1 = s1_cache.get(s1_id)

        if s1 is None:

            try:
                s1 = s1_lookup.loc[s1_id]
            except KeyError:
                raise RuntimeError(
                    f"S1 entity missing: {s1_id}"
                )

            s1_cache[s1_id] = s1

        # -----------------------------------------------
        # Target
        # -----------------------------------------------

        target = target_cache.get(target_id)

        if target is None:

            try:
                target = target_lookup.loc[
                    target_id
                ]
            except KeyError:
                raise RuntimeError(
                    f"Target entity missing: {target_id}"
                )

            target_cache[target_id] = target

        # -----------------------------------------------
        # Channels
        # -----------------------------------------------

        channel_value = retrieval[i]

        decoded = channel_cache.get(
            channel_value
        )

        if decoded is None:

            decoded = decode_channels(
                channel_value
            )

            channel_cache[channel_value] = decoded

        (
            channels,
            route_flags,
            strongest_priority,
        ) = decoded

        # -----------------------------------------------
        # Features
        # -----------------------------------------------

        X[i, :] = compute_feature_row(
            s1,
            target,
            channels,
            route_flags,
            strongest_priority,
        )

    return X


# ============================================================
# WRITE ONE S1 OUTPUT ROW
# ============================================================

def write_output_row(
    matching_handle,
    candidate_handle,
    s1_id,
    candidate_ids,
    matched_ids,
):

    candidate_string = ",".join(
        candidate_ids
    )

    matched_string = ",".join(
        matched_ids
    )

    candidate_handle.write(
        f"{s1_id}\t{candidate_string}\n"
    )

    matching_handle.write(
        f"{s1_id}\t{matched_string}\n"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("03 — TEST INFERENCE")
    print("=" * 70)

    start_time = time.time()

    # ========================================================
    # SCHEMA
    # ========================================================

    print("\nValidating processed schemas...")

    validate_file_schema(S1_PATH)
    validate_file_schema(S2_PATH)
    validate_file_schema(S3_PATH)

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # LOAD PROCESSED RECORDS
    # ========================================================

    s1, s2, s3 = load_processed_data()

    s1_lookup = build_s1_lookup(s1)

    target_lookup = build_target_lookup(
        s2,
        s3,
    )

    # Release standalone DataFrames.
    #
    # s1_lookup and target_lookup retain the actual data.
    del s2
    del s3

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print("\nPreparing output files...")

    with open(
        MATCHING_RESULTS,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as matching_handle, open(
        FINAL_CANDIDATES,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as candidate_handle:

        matching_handle.write(
            "source1_entity_id\tmatched_entity_ids\n"
        )

        candidate_handle.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        # ====================================================
        # CANDIDATE STREAM
        # ====================================================

        print(
            "\nStreaming candidate pairs..."
        )

        reader = pd.read_csv(
            CANDIDATES,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CANDIDATE_CHUNK_SIZE,
        )

        total_candidates = 0
        total_predicted = 0
        total_chunks = 0

        current_s1_id = None
        current_candidate_ids = []
        current_matched_ids = []

        seen_s1 = set()

        last_progress = time.time()

        for chunk in reader:

            total_chunks += 1

            # ------------------------------------------------
            # Basic validation.
            # ------------------------------------------------

            required_candidate_columns = {
                "s1_entity_id",
                "candidate_entity_id",
                "retrieval_channels",
            }

            missing = (
                required_candidate_columns
                - set(chunk.columns)
            )

            if missing:
                raise RuntimeError(
                    "Candidate file missing columns: "
                    f"{sorted(missing)}"
                )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # 02 writes candidates grouped by S1.
            # We rely on that to stream output.
            # ------------------------------------------------

            s1_values = chunk[
                "s1_entity_id"
            ].to_numpy()

            target_values = chunk[
                "candidate_entity_id"
            ].to_numpy()

            # ------------------------------------------------
            # Build features.
            # ------------------------------------------------

            feature_start = time.time()

            X = build_feature_matrix(
                chunk,
                s1_lookup,
                target_lookup,
            )

            feature_time = time.time() - feature_start

            # ------------------------------------------------
            # Predict.
            # ------------------------------------------------

            probabilities = model.predict_proba(
                X
            )[:, 1]

            predictions = (
                probabilities >= THRESHOLD
            )

            total_candidates += len(chunk)

            total_predicted += int(
                predictions.sum()
            )

            # ------------------------------------------------
            # Consume rows in original order.
            # ------------------------------------------------

            for i in range(len(chunk)):

                s1_id = str(
                    s1_values[i]
                )

                target_id = str(
                    target_values[i]
                )

                # --------------------------------------------
                # New S1 entity.
                # --------------------------------------------

                if (
                    current_s1_id is None
                ):

                    current_s1_id = s1_id

                elif s1_id != current_s1_id:

                    # Flush previous entity.
                    write_output_row(
                        matching_handle,
                        candidate_handle,
                        current_s1_id,
                        current_candidate_ids,
                        current_matched_ids,
                    )

                    seen_s1.add(
                        current_s1_id
                    )

                    current_s1_id = s1_id

                    current_candidate_ids = []
                    current_matched_ids = []

                # --------------------------------------------
                # Candidate always belongs in candidate output.
                # --------------------------------------------

                current_candidate_ids.append(
                    target_id
                )

                # --------------------------------------------
                # Only predicted matches belong in matching output.
                # --------------------------------------------

                if predictions[i]:

                    current_matched_ids.append(
                        target_id
                    )

            # ------------------------------------------------
            # Progress.
            # ------------------------------------------------

            if (
                total_candidates
                % PROGRESS_EVERY
                < len(chunk)
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                rate = (
                    total_candidates
                    / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"Processed "
                    f"{total_candidates:,} "
                    f"candidate pairs | "
                    f"Predicted "
                    f"{total_predicted:,} | "
                    f"Rate "
                    f"{rate:,.0f}/sec | "
                    f"Feature batch "
                    f"{feature_time:.1f}s"
                )

        # ====================================================
        # FLUSH FINAL S1
        # ====================================================

        if current_s1_id is not None:

            write_output_row(
                matching_handle,
                candidate_handle,
                current_s1_id,
                current_candidate_ids,
                current_matched_ids,
            )

            seen_s1.add(
                current_s1_id
            )

    # ========================================================
    # ZERO-CANDIDATE S1 ENTITIES
    # ========================================================
    #
    # 02 processes S1 sequentially, but some S1s can have zero
    # candidates. They still need one row in both outputs.
    #
    # We append those missing S1 IDs WITHOUT rereading the huge
    # candidate/output files.
    # ========================================================

    all_s1_ids = s1_lookup.index

    missing_s1 = [
        str(x)
        for x in all_s1_ids
        if str(x) not in seen_s1
    ]

    if missing_s1:

        print(
            f"\nS1 entities with zero candidates: "
            f"{len(missing_s1):,}"
        )

        with open(
            MATCHING_RESULTS,
            "a",
            encoding="utf-8",
            buffering=1024 * 1024,
        ) as matching_handle, open(
            FINAL_CANDIDATES,
            "a",
            encoding="utf-8",
            buffering=1024 * 1024,
        ) as candidate_handle:

            for s1_id in missing_s1:

                candidate_handle.write(
                    f"{s1_id}\t\n"
                )

                matching_handle.write(
                    f"{s1_id}\t\n"
                )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    elapsed = time.time() - start_time

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

    print(
        f"Candidate pairs processed: "
        f"{total_candidates:,}"
    )

    print(
        f"Predicted matches: "
        f"{total_predicted:,}"
    )

    print(
        f"S1 entities: "
        f"{len(s1_lookup):,}"
    )

    print(
        f"Zero-candidate S1: "
        f"{len(missing_s1):,}"
    )

    print(
        f"Threshold: "
        f"{THRESHOLD:.3f}"
    )

    print(
        f"Runtime: "
        f"{elapsed / 3600:.2f} hours"
    )

    print(
        f"\nMatching output:"
    )

    print(
        f"  {MATCHING_RESULTS}"
    )

    print(
        f"\nCandidate output:"
    )

    print(
        f"  {FINAL_CANDIDATES}"
    )


if __name__ == "__main__":
    main()