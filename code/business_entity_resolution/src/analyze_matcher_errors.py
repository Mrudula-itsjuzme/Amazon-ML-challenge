from pathlib import Path
import ast
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SRC_DIR.parents[3]
REPO_ROOT = SRC_DIR.parents[2]
PROCESSED_DIR = REPO_ROOT / "processed"
TRAIN_DIR = REPO_ROOT / "student_resource" / "dataset" / "train"
if not (TRAIN_DIR / "train_ground_truth.tsv").is_file():
    TRAIN_DIR = WORKSPACE_ROOT / "student_resource" / "dataset" / "train"

CANDIDATE_FILE = PROCESSED_DIR / "dev_candidates_5000_translit.parquet"
FEATURE_FILE = PROCESSED_DIR / "dev_pair_features_5000_translit.parquet"
MODEL_FILE = PROCESSED_DIR / "xgb_matcher_v2.json"

OUTPUT_FILE = PROCESSED_DIR / "matcher_error_analysis_5000.csv"

THRESHOLD = 0.95


# ============================================================
# HELPERS
# ============================================================

def parse_id_list(value):
    """
    Parse matched_entity_ids from the ground-truth TSV.

    Handles:
      ['123', '456']
      ["123", "456"]
      123,456
      empty / NaN
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]

        if isinstance(parsed, (str, int, float)):
            return [str(parsed).strip()]

    except Exception:
        pass

    value = (
        value
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def load_ground_truth():
    print("Loading ground truth...")

    gt_file = TRAIN_DIR / "train_ground_truth.tsv"

    gt = pd.read_csv(
        gt_file,
        sep="\t",
        dtype=str
    )

    truth = {}

    for _, row in gt.iterrows():
        s1 = str(row["source1_entity_id"])

        truth[s1] = set(
            parse_id_list(row["matched_entity_ids"])
        )

    return truth


def identify_source(entity_id):
    """
    Candidate IDs are expected to encode source information
    through the prefixes used by the challenge data.
    """
    entity_id = str(entity_id)

    if entity_id.startswith("S2-"):
        return "S2"

    if entity_id.startswith("S3-"):
        return "S3"

    return "UNKNOWN"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MATCHER ERROR ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    print("\nLoading feature dataset...")

    df = pd.read_parquet(FEATURE_FILE)

    print(f"Feature rows : {len(df):,}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading XGBoost model...")

    from xgboost import XGBClassifier

    model = XGBClassifier()

    model.load_model(MODEL_FILE)

    # --------------------------------------------------------
    # Determine feature columns
    # --------------------------------------------------------

    non_feature_columns = {
        "source1_entity_id",
        "candidate_entity_id",
        "label",
    }

    feature_columns = [
        c for c in df.columns
        if c not in non_feature_columns
    ]

    print(f"Feature count : {len(feature_columns)}")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    X = df[feature_columns]

    probabilities = model.predict_proba(X)[:, 1]

    df["probability"] = probabilities
    df["prediction"] = (
        df["probability"] >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    truth = load_ground_truth()

    # --------------------------------------------------------
    # Classify every candidate pair
    # --------------------------------------------------------

    categories = []

    for _, row in df.iterrows():

        s1 = str(row["source1_entity_id"])
        candidate = str(row["candidate_entity_id"])

        true_set = truth.get(s1, set())

        actual = candidate in true_set
        predicted = bool(row["prediction"])

        if predicted and actual:
            category = "TP"

        elif predicted and not actual:
            category = "FP"

        elif not predicted and actual:
            category = "FN"

        else:
            category = "TN"

        categories.append(category)

    df["error_type"] = categories

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("OVERALL ERROR BREAKDOWN")
    print("=" * 80)

    counts = df["error_type"].value_counts()

    for label in ["TP", "FP", "FN", "TN"]:
        print(
            f"{label:>3} : "
            f"{counts.get(label, 0):,}"
        )

    # --------------------------------------------------------
    # Candidate-side source distribution
    # --------------------------------------------------------

    df["candidate_source"] = df["candidate_entity_id"].apply(
        identify_source
    )

    print("\n" + "=" * 80)
    print("ERRORS BY SOURCE")
    print("=" * 80)

    source_table = pd.crosstab(
        df["candidate_source"],
        df["error_type"]
    )

    print(source_table)

    # --------------------------------------------------------
    # Error rates by probability bucket
    # --------------------------------------------------------

    df["prob_bucket"] = pd.cut(
        df["probability"],
        bins=[
            -np.inf,
            0.50,
            0.70,
            0.80,
            0.90,
            0.92,
            0.94,
            0.95,
            0.96,
            0.97,
            0.98,
            0.99,
            np.inf,
        ],
        right=False
    )

    print("\n" + "=" * 80)
    print("ERROR DISTRIBUTION BY MODEL CONFIDENCE")
    print("=" * 80)

    probability_table = (
        df.groupby("prob_bucket", observed=False)
        .agg(
            pairs=("probability", "size"),
            TP=("error_type", lambda x: (x == "TP").sum()),
            FP=("error_type", lambda x: (x == "FP").sum()),
            FN=("error_type", lambda x: (x == "FN").sum()),
        )
        .reset_index()
    )

    print(probability_table.to_string(index=False))

    # --------------------------------------------------------
    # Feature statistics for TP / FP / FN
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FEATURE MEANS BY ERROR TYPE")
    print("=" * 80)

    important_features = [
        "name_exact",
        "name_core_exact",
        "name_translit_exact",
        "name_similarity",
        "name_jaccard",
        "name_token_overlap",
        "name_length_ratio",
        "address_exact",
        "address_similarity",
        "address_jaccard",
        "address_token_overlap",
        "address_length_ratio",
        "country_match",
        "postal_match",
        "numeric_overlap",
        "has_numeric_overlap",
        "name_address_similarity",
        "both_exact",
    ]

    available_features = [
        x for x in important_features
        if x in df.columns
    ]

    feature_summary = (
        df[df["error_type"].isin(["TP", "FP", "FN"])]
        .groupby("error_type")[available_features]
        .mean()
        .T
    )

    print(feature_summary.to_string())

    # --------------------------------------------------------
    # High-confidence false positives
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("HIGH-CONFIDENCE FALSE POSITIVES")
    print("=" * 80)

    fp = (
        df[df["error_type"] == "FP"]
        .sort_values("probability", ascending=False)
        .head(30)
    )

    fp_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "probability",
    ] + available_features

    print(
        fp[fp_columns]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # High-confidence false negatives
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("HIGH-CONFIDENCE FALSE NEGATIVES")
    print("=" * 80)

    fn = (
        df[df["error_type"] == "FN"]
        .sort_values("probability", ascending=False)
        .head(30)
    )

    print(
        fn[fp_columns]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save detailed analysis
    # --------------------------------------------------------

    save_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "probability",
        "prediction",
        "label",
        "error_type",
        "candidate_source",
    ] + available_features

    df[save_columns].to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        f"Error analysis -> {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()