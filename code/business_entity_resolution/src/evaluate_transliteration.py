from pathlib import Path
import pandas as pd
import re
import unicodedata
from anyascii import anyascii
from rapidfuzz.fuzz import ratio


# ============================================================
# PATHS
# ============================================================

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = WORKSPACE_ROOT / "student_resource" / "dataset" / "train"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Transliterate
    text = anyascii(text)

    # Lowercase
    text = text.casefold()

    # & -> and
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD DATA
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    TRAIN_DIR / "train_ground_truth.tsv",
    sep="\t"
)

s1 = pd.read_csv(
    TRAIN_DIR / "train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "business_name"]
)

s2 = pd.read_csv(
    TRAIN_DIR / "train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name"]
)

s3 = pd.read_csv(
    TRAIN_DIR / "train_source3.tsv",
    sep="\t",
    usecols=["entity_id", "business_name"]
)


# ============================================================
# FIND MULTILINGUAL MATCHES
# ============================================================

def is_non_latin(text):
    if pd.isna(text):
        return False

    for ch in str(text):
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if "LATIN" not in name:
                return True

    return False


print("Finding non-Latin ground-truth records...")

s2_nonlatin = s2[s2["business_name"].apply(is_non_latin)].copy()
s3_nonlatin = s3[s3["business_name"].apply(is_non_latin)].copy()

print(f"S2 non-Latin records: {len(s2_nonlatin):,}")
print(f"S3 non-Latin records: {len(s3_nonlatin):,}")


# ============================================================
# BUILD LOOKUPS
# ============================================================

s1_names = dict(
    zip(
        s1["entity_id"],
        s1["business_name"]
    )
)

s2_names = dict(
    zip(
        s2_nonlatin["entity_id"],
        s2_nonlatin["business_name"]
    )
)

s3_names = dict(
    zip(
        s3_nonlatin["entity_id"],
        s3_nonlatin["business_name"]
    )
)


# ============================================================
# EVALUATE GROUND-TRUTH PAIRS
# ============================================================

# ============================================================
# BUILD NON-LATIN ID SET
# ============================================================

nonlatin_s2_ids = set(s2_names.keys())
nonlatin_s3_ids = set(s3_names.keys())


# ============================================================
# EVALUATE
# ============================================================

results = []

print("\nEvaluating multilingual ground-truth pairs...")

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id

    if s1_id not in s1_names:
        continue

    s1_name = s1_names[s1_id]
    s1_norm = normalize(s1_name)

    # matched_entity_ids may be empty, NaN, or a stringified list
    matched_ids = row.matched_entity_ids

    if pd.isna(matched_ids) or matched_ids is None:
        matched_ids = []
    elif isinstance(matched_ids, str):
        cleaned = matched_ids.strip()
        if not cleaned:
            matched_ids = []
        else:
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]

            matched_ids = [
                x.strip().strip("'").strip('"')
                for x in cleaned.split(",")
                if str(x).strip()
            ]
    else:
        matched_ids = [str(matched_ids).strip()]

    for matched_id in matched_ids:

        matched_id = str(matched_id).strip()

        if matched_id in nonlatin_s2_ids:
            source = "S2"
            target_name = s2_names[matched_id]

        elif matched_id in nonlatin_s3_ids:
            source = "S3"
            target_name = s3_names[matched_id]

        else:
            continue

        target_norm = normalize(target_name)

        similarity = ratio(
            s1_norm,
            target_norm
        )

        results.append({
            "source1_id": s1_id,
            "matched_id": matched_id,
            "source": source,
            "s1_name": s1_name,
            "matched_name": target_name,
            "s1_normalized": s1_norm,
            "transliterated_normalized": target_norm,
            "similarity": similarity
        })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("TRANSLITERATION EVALUATION")
print("=" * 70)

print(f"Multilingual ground-truth pairs evaluated: {len(results_df):,}")

if len(results_df) == 0:
    print("No multilingual ground-truth pairs were found.")
    raise SystemExit


# Exact matches
exact_rate = (
    results_df["s1_normalized"]
    == results_df["transliterated_normalized"]
).mean()

# Similarity thresholds
for threshold in [90, 80, 70, 60]:

    rate = (
        results_df["similarity"] >= threshold
    ).mean()

    print(
        f"Similarity >= {threshold}: "
        f"{rate:.2%}"
    )

print(f"Exact normalized match: {exact_rate:.2%}")


# ============================================================
# SOURCE BREAKDOWN
# ============================================================

print("\nSource breakdown:")

for source in ["S2", "S3"]:

    subset = results_df[
        results_df["source"] == source
    ]

    if len(subset) == 0:
        continue

    exact = (
        subset["s1_normalized"]
        == subset["transliterated_normalized"]
    ).mean()

    high = (
        subset["similarity"] >= 80
    ).mean()

    print(
        f"{source}: "
        f"{len(subset):,} pairs | "
        f"exact={exact:.2%} | "
        f"similarity>=80={high:.2%}"
    )


# ============================================================
# EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLES")
print("=" * 70)

display_columns = [
    "s1_name",
    "matched_name",
    "transliterated_normalized",
    "similarity"
]

print(
    results_df[
        display_columns
    ]
    .sort_values("similarity", ascending=False)
    .head(20)
    .to_string(index=False)
)


print("\nLOW-SIMILARITY EXAMPLES")
print("=" * 70)

print(
    results_df[
        display_columns
    ]
    .sort_values("similarity")
    .head(20)
    .to_string(index=False)
)