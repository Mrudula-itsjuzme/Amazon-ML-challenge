from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

FILTERED_DIR = ROOT / "filtered"
MODEL_DIR = ROOT / "models"

DIAGNOSTICS_PATH = (
    FILTERED_DIR
    / "train_validation_diagnostics.tsv"
)

MODEL_META_PATH = (
    MODEL_DIR
    / "entity_matcher_meta.json"
)

OUTPUT_DIR = FILTERED_DIR / "diagnostics"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

THRESHOLDS = [
    0.50,
    0.70,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.97,
    0.99,
]

TOP_KS = [
    1,
    3,
    5,
    10,
    20,
    50,
    100,
    200,
]


# ============================================================
# HELPERS
# ============================================================

def parse_channels(value):

    if pd.isna(value):
        return set()

    value = str(value).strip()

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

    for separator in [
        "|",
        ",",
        ";",
    ]:

        if separator in value:

            return {
                x.strip()
                for x in value.split(separator)
                if x.strip()
            }

    return {value}


def f05(
    precision,
    recall,
):

    if precision + recall == 0:
        return 0.0

    return (
        1.25
        * precision
        * recall
        / (
            0.25 * precision
            + recall
        )
    )


def evaluate_threshold(
    df,
    threshold,
):

    predicted = (
        df["match_probability"]
        >= threshold
    )

    truth = (
        df["true_label"]
        == 1
    )

    tp = int(
        (predicted & truth).sum()
    )

    fp = int(
        (predicted & ~truth).sum()
    )

    fn = int(
        (~predicted & truth).sum()
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

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f05": f05(
            precision,
            recall,
        ),
        "predicted": int(
            predicted.sum()
        ),
    }


# ============================================================
# 1. THRESHOLD LANDSCAPE
# ============================================================

def threshold_analysis(df):

    rows = [
        evaluate_threshold(
            df,
            threshold,
        )
        for threshold in THRESHOLDS
    ]

    result = pd.DataFrame(rows)

    path = (
        OUTPUT_DIR
        / "threshold_analysis.tsv"
    )

    result.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "1. THRESHOLD LANDSCAPE"
    )

    print(
        "=" * 80
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        f"\nSaved: {path}"
    )

    return result


# ============================================================
# 2. SCORE DISTRIBUTIONS
# ============================================================

def score_distribution(df):

    groups = []

    for label in [
        "TP",
        "FP",
        "FN",
        "TN",
    ]:

        subset = df[
            df["error_type"] == label
        ]

        if not len(subset):
            continue

        p = subset[
            "match_probability"
        ]

        groups.append(
            {
                "group": label,
                "count": len(subset),
                "mean": p.mean(),
                "median": p.median(),
                "p90": p.quantile(0.90),
                "p95": p.quantile(0.95),
                "p99": p.quantile(0.99),
                "min": p.min(),
                "max": p.max(),
            }
        )

    result = pd.DataFrame(groups)

    path = (
        OUTPUT_DIR
        / "score_distribution.tsv"
    )

    result.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "2. SCORE DISTRIBUTION BY ERROR TYPE"
    )

    print(
        "=" * 80
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        f"\nSaved: {path}"
    )

    return result


# ============================================================
# 3. HIGH-CONFIDENCE FALSE NEGATIVES
# ============================================================

def false_negative_analysis(df):

    fn = df[
        df["error_type"] == "FN"
    ].copy()

    fn = fn.sort_values(
        "match_probability",
        ascending=True,
    )

    path = (
        OUTPUT_DIR
        / "false_negatives.tsv"
    )

    fn.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "3. FALSE NEGATIVES"
    )

    print(
        "=" * 80
    )

    print(
        f"Total false negatives: {len(fn):,}"
    )

    for threshold in [
        0.95,
        0.90,
        0.80,
        0.70,
        0.50,
    ]:

        count = int(
            (
                fn["match_probability"]
                < threshold
            ).sum()
        )

        print(
            f"True matches scoring < "
            f"{threshold:.2f}: {count:,}"
        )

    print(
        "\nLowest-scoring true matches:"
    )

    display_columns = [
        "s1_entity_id",
        "candidate_entity_id",
        "match_probability",
        "model_rank",
        "retrieval_channels",
        "name_ratio",
        "name_token_set",
        "core_name_ratio",
        "translit_ratio",
        "address_ratio",
        "address_token_set",
        "name_token_jaccard",
        "address_token_jaccard",
        "number_jaccard",
        "number_exact",
        "same_postal",
        "strongest_route_priority",
    ]

    display_columns = [
        c
        for c in display_columns
        if c in fn.columns
    ]

    print(
        fn[
            display_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {path}"
    )

    return fn


# ============================================================
# 4. HIGH-CONFIDENCE FALSE POSITIVES
# ============================================================

def false_positive_analysis(df):

    fp = df[
        df["error_type"] == "FP"
    ].copy()

    fp = fp.sort_values(
        "match_probability",
        ascending=False,
    )

    path = (
        OUTPUT_DIR
        / "false_positives.tsv"
    )

    fp.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "4. FALSE POSITIVES"
    )

    print(
        "=" * 80
    )

    print(
        f"Total false positives: {len(fp):,}"
    )

    for threshold in [
        0.99,
        0.97,
        0.95,
        0.90,
        0.80,
    ]:

        count = int(
            (
                fp["match_probability"]
                >= threshold
            ).sum()
        )

        print(
            f"False positives >= "
            f"{threshold:.2f}: {count:,}"
        )

    print(
        "\nHighest-scoring false positives:"
    )

    display_columns = [
        "s1_entity_id",
        "candidate_entity_id",
        "match_probability",
        "model_rank",
        "retrieval_channels",
        "name_ratio",
        "name_token_set",
        "core_name_ratio",
        "translit_ratio",
        "address_ratio",
        "address_token_set",
        "name_token_jaccard",
        "address_token_jaccard",
        "number_jaccard",
        "number_exact",
        "same_postal",
        "strongest_route_priority",
    ]

    display_columns = [
        c
        for c in display_columns
        if c in fp.columns
    ]

    print(
        fp[
            display_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {path}"
    )

    return fp


# ============================================================
# 5. TRUE-MATCH RANK ANALYSIS
# ============================================================

def rank_analysis(df):

    positives = df[
        df["true_label"] == 1
    ].copy()

    rows = []

    for k in TOP_KS:

        found = int(
            (
                positives["model_rank"]
                <= k
            ).sum()
        )

        total = len(positives)

        recall = (
            found / total
            if total
            else 0.0
        )

        rows.append(
            {
                "top_k": k,
                "true_pairs_found": found,
                "total_true_pairs": total,
                "pair_recall": recall,
            }
        )

    result = pd.DataFrame(rows)

    path = (
        OUTPUT_DIR
        / "true_match_rank_analysis.tsv"
    )

    result.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "5. TRUE MATCH RANK ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        f"\nSaved: {path}"
    )

    return result


# ============================================================
# 6. ENTITY-LEVEL ANALYSIS
# ============================================================

def entity_analysis(df):

    grouped = (
        df
        .groupby("s1_entity_id")
    )

    rows = []

    for s1_id, group in grouped:

        truth_count = int(
            group["true_label"].sum()
        )

        predicted_count = int(
            group[
                "match_prediction"
            ].sum()
        )

        max_score = float(
            group[
                "match_probability"
            ].max()
        )

        true_scores = group.loc[
            group["true_label"] == 1,
            "match_probability",
        ]

        best_true_score = (
            float(true_scores.max())
            if len(true_scores)
            else np.nan
        )

        best_true_rank = (
            int(
                group.loc[
                    group["true_label"] == 1,
                    "model_rank",
                ].min()
            )
            if len(true_scores)
            else np.nan
        )

        rows.append(
            {
                "s1_entity_id": s1_id,
                "candidate_count": len(group),
                "truth_count": truth_count,
                "predicted_count": predicted_count,
                "max_score": max_score,
                "best_true_score": best_true_score,
                "best_true_rank": best_true_rank,
            }
        )

    result = pd.DataFrame(rows)

    path = (
        OUTPUT_DIR
        / "entity_analysis.tsv"
    )

    result.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "6. ENTITY-LEVEL ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"Entities: {len(result):,}"
    )

    print(
        f"No predicted matches: "
        f"{(result.predicted_count == 0).sum():,}"
    )

    print(
        f"No true matches: "
        f"{(result.truth_count == 0).sum():,}"
    )

    print(
        "\nCandidate-count distribution:"
    )

    print(
        result[
            "candidate_count"
        ].describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print(
        f"\nSaved: {path}"
    )

    return result


# ============================================================
# 7. ROUTE ANALYSIS
# ============================================================

def route_analysis(df):

    route_names = set()

    for value in df[
        "retrieval_channels"
    ]:

        route_names.update(
            parse_channels(value)
        )

    rows = []

    for route in sorted(
        route_names
    ):

        mask = df[
            "retrieval_channels"
        ].apply(
            lambda x: route
            in parse_channels(x)
        )

        subset = df[mask]

        if not len(subset):
            continue

        positives = int(
            subset["true_label"].sum()
        )

        predicted = int(
            subset["match_prediction"].sum()
        )

        true_positive = int(
            (
                (subset["true_label"] == 1)
                &
                (subset["match_prediction"] == 1)
            ).sum()
        )

        false_positive = int(
            (
                (subset["true_label"] == 0)
                &
                (subset["match_prediction"] == 1)
            ).sum()
        )

        precision = (
            true_positive
            / predicted
            if predicted
            else 0.0
        )

        recall_contribution = (
            true_positive
            / int(
                (df["true_label"] == 1)
                .sum()
            )
            if int(
                (df["true_label"] == 1)
                .sum()
            )
            else 0.0
        )

        rows.append(
            {
                "route": route,
                "candidate_pairs": len(subset),
                "true_pairs": positives,
                "predicted_pairs": predicted,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "route_precision": precision,
                "true_pair_recall_contribution": recall_contribution,
                "mean_probability": subset[
                    "match_probability"
                ].mean(),
            }
        )

    result = pd.DataFrame(rows)

    if len(result):

        result = result.sort_values(
            "true_positive",
            ascending=False,
        )

    path = (
        OUTPUT_DIR
        / "route_analysis.tsv"
    )

    result.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "7. RETRIEVAL ROUTE ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        f"\nSaved: {path}"
    )

    return result


# ============================================================
# 8. FEATURE SEPARATION
# ============================================================

def feature_separation(df):

    numeric_columns = [
        c
        for c in df.columns
        if c not in {
            "true_label",
            "match_prediction",
            "match_probability",
            "model_rank",
        }
        and pd.api.types.is_numeric_dtype(
            df[c]
        )
    ]

    rows = []

    for column in numeric_columns:

        positive = df.loc[
            df["true_label"] == 1,
            column,
        ]

        negative = df.loc[
            df["true_label"] == 0,
            column,
        ]

        if not len(
            positive
        ) or not len(
            negative
        ):
            continue

        pos_mean = positive.mean()
        neg_mean = negative.mean()

        rows.append(
            {
                "feature": column,
                "positive_mean": pos_mean,
                "negative_mean": neg_mean,
                "mean_gap": (
                    pos_mean - neg_mean
                ),
                "positive_median": positive.median(),
                "negative_median": negative.median(),
            }
        )

    result = pd.DataFrame(rows)

    if len(result):

        result[
            "abs_mean_gap"
        ] = result[
            "mean_gap"
        ].abs()

        result = result.sort_values(
            "abs_mean_gap",
            ascending=False,
        )

    path = (
        OUTPUT_DIR
        / "feature_separation.tsv"
    )

    result.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "8. FEATURE SEPARATION"
    )

    print(
        "=" * 80
    )

    print(
        result.head(40).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        f"\nSaved: {path}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "03 — MATCHER DIAGNOSTICS"
    )

    print(
        "=" * 80
    )

    if not DIAGNOSTICS_PATH.exists():

        raise FileNotFoundError(
            f"\nMissing:\n"
            f"{DIAGNOSTICS_PATH}\n\n"
            f"Run 03_filter_candidates.py first."
        )

    print(
        "\nLoading validation diagnostics..."
    )

    df = pd.read_csv(
        DIAGNOSTICS_PATH,
        sep="\t",
        dtype={
            "s1_entity_id": str,
            "candidate_entity_id": str,
        },
        keep_default_na=False,
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"S1 entities: "
        f"{df['s1_entity_id'].nunique():,}"
    )

    print(
        f"True pairs: "
        f"{int(df['true_label'].sum()):,}"
    )

    print(
        f"Predicted pairs: "
        f"{int(df['match_prediction'].sum()):,}"
    )

    # --------------------------------------------------------
    # Run diagnostics
    # --------------------------------------------------------

    threshold_analysis(df)

    score_distribution(df)

    false_negative_analysis(df)

    false_positive_analysis(df)

    rank_analysis(df)

    entity_analysis(df)

    route_analysis(df)

    feature_separation(df)

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DIAGNOSTICS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nAll diagnostic files are in:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()