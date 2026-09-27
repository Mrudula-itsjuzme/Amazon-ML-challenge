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
    0.96,
    0.97,
    0.98,
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


def safe_div(
    numerator,
    denominator,
):

    return (
        numerator / denominator
        if denominator
        else 0.0
    )


# ============================================================
# 1. PAIRWISE THRESHOLD LANDSCAPE
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
        "1. PAIRWISE THRESHOLD LANDSCAPE"
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

    best = result.loc[
        result["f05"].idxmax()
    ]

    print(
        "\nBest pairwise F0.5:"
    )

    print(
        f"  Threshold : {best['threshold']:.3f}"
    )

    print(
        f"  Precision : {best['precision']:.6f}"
    )

    print(
        f"  Recall    : {best['recall']:.6f}"
    )

    print(
        f"  F0.5      : {best['f05']:.6f}"
    )

    print(
        f"\nSaved: {path}"
    )

    return result


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

    precision = safe_div(
        tp,
        tp + fp,
    )

    recall = safe_div(
        tp,
        tp + fn,
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
# 2. ACTUAL CHALLENGE METRIC
# ============================================================

def challenge_metric_analysis(df):

    """
    The challenge uses macro F0.5 per S1 entity.

    For each S1:
        precision = TP / predicted
        recall    = TP / truth
        F0.5      = per-entity F0.5

    Singleton:
        truth=0, predicted=0 -> 1.0
        truth=0, predicted>0 -> 0.0

    This is NOT the same as global pairwise F0.5.
    """

    grouped = []

    for s1_id, group in df.groupby(
        "s1_entity_id",
        sort=False,
    ):

        truth = int(
            group["true_label"].sum()
        )

        predicted = int(
            group["match_prediction"].sum()
        )

        tp = int(
            (
                (group["true_label"] == 1)
                &
                (group["match_prediction"] == 1)
            ).sum()
        )

        fp = predicted - tp

        fn = truth - tp

        if truth == 0:

            precision = (
                1.0
                if predicted == 0
                else 0.0
            )

            recall = 1.0 if predicted == 0 else 0.0

            entity_f05 = (
                1.0
                if predicted == 0
                else 0.0
            )

            category = (
                "correct_singleton"
                if predicted == 0
                else "false_match_on_singleton"
            )

        else:

            precision = safe_div(
                tp,
                predicted,
            )

            recall = safe_div(
                tp,
                truth,
            )

            entity_f05 = f05(
                precision,
                recall,
            )

            if tp == truth and predicted == truth:
                category = "perfect"

            elif tp == 0:
                category = "zero_tp"

            elif fp > 0 and fn > 0:
                category = "both_fp_fn"

            elif fp > 0:
                category = "false_positive_only"

            elif fn > 0:
                category = "false_negative_only"

            else:
                category = "other"

        grouped.append(
            {
                "s1_entity_id": s1_id,
                "truth_count": truth,
                "predicted_count": predicted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "entity_f05": entity_f05,
                "category": category,
            }
        )

    result = pd.DataFrame(grouped)

    macro_f05 = result[
        "entity_f05"
    ].mean()

    singleton_mask = (
        result["truth_count"] == 0
    )

    singleton_count = int(
        singleton_mask.sum()
    )

    correct_singletons = int(
        (
            singleton_mask
            &
            (result["predicted_count"] == 0)
        ).sum()
    )

    false_singletons = int(
        (
            singleton_mask
            &
            (result["predicted_count"] > 0)
        ).sum()
    )

    path = (
        OUTPUT_DIR
        / "challenge_entity_metrics.tsv"
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
        "2. ACTUAL CHALLENGE METRIC — MACRO ENTITY F0.5"
    )

    print(
        "=" * 80
    )

    print(
        f"Entities                  : {len(result):,}"
    )

    print(
        f"Macro entity F0.5         : {macro_f05:.6f}"
    )

    print(
        f"Singleton entities        : {singleton_count:,}"
    )

    print(
        f"Correct singletons        : {correct_singletons:,}"
    )

    print(
        f"False matches on singleton: {false_singletons:,}"
    )

    print(
        "\nEntity outcome distribution:"
    )

    print(
        result["category"]
        .value_counts()
        .to_string()
    )

    print(
        "\nEntity F0.5 distribution:"
    )

    print(
        result[
            "entity_f05"
        ].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
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
# 3. SCORE DISTRIBUTIONS
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
        "3. SCORE DISTRIBUTION BY ERROR TYPE"
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
# 4. FALSE NEGATIVES
# ============================================================

def false_negative_analysis(df):

    fn = df[
        df["error_type"] == "FN"
    ].copy()

    fn = fn.sort_values(
        "match_probability",
        ascending=False,
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
        "4. FALSE NEGATIVES"
    )

    print(
        "=" * 80
    )

    print(
        f"Total false negatives: {len(fn):,}"
    )

    for threshold in [
        0.99,
        0.97,
        0.96,
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
        "\nHighest-scoring false negatives:"
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
# 5. FALSE POSITIVES
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
        "5. HIGH-CONFIDENCE FALSE POSITIVES"
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
        0.96,
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
# 6. TRUE MATCH RANK ANALYSIS
# ============================================================

def rank_analysis(df):

    positives = df[
        df["true_label"] == 1
    ].copy()

    rows = []

    total = len(positives)

    for k in TOP_KS:

        found = int(
            (
                positives["model_rank"]
                <= k
            ).sum()
        )

        recall = safe_div(
            found,
            total,
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
        "6. TRUE MATCH RANK ANALYSIS"
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
# 7. ENTITY-LEVEL ANALYSIS
# ============================================================

def entity_analysis(df):

    grouped = (
        df
        .groupby(
            "s1_entity_id",
            sort=False,
        )
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

        tp = int(
            (
                (group["true_label"] == 1)
                &
                (group["match_prediction"] == 1)
            ).sum()
        )

        fp = int(
            (
                (group["true_label"] == 0)
                &
                (group["match_prediction"] == 1)
            ).sum()
        )

        fn = int(
            (
                (group["true_label"] == 1)
                &
                (group["match_prediction"] == 0)
            ).sum()
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
                "tp": tp,
                "fp": fp,
                "fn": fn,
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
        "7. ENTITY-LEVEL ANALYSIS"
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
        f"True entities with no predicted matches: "
        f"{((result.truth_count > 0) & (result.predicted_count == 0)).sum():,}"
    )

    print(
        f"Entities with both FP and FN: "
        f"{((result.fp > 0) & (result.fn > 0)).sum():,}"
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
# 8. CANDIDATE COVERAGE / BLOCKER CEILING
# ============================================================

def candidate_coverage_analysis(df):

    """
    Determines whether every true pair survived 02.

    If a true pair is absent from this diagnostics file,
    03 can never recover it.

    Therefore:

        blocker recall = true pairs present / all true pairs

    IMPORTANT:
    The diagnostics file only contains candidate pairs.
    If 03's diagnostics were built from the candidate file,
    missing ground-truth pairs are invisible here.

    We therefore use the model metadata if available.
    """

    total_truth_pairs = None

    meta = {}

    if MODEL_META_PATH.exists():

        try:
            with open(
                MODEL_META_PATH,
                "r",
                encoding="utf-8",
            ) as f:
                meta = json.load(f)

        except Exception:
            meta = {}

    # Try common metadata field names.
    for key in [
        "validation_true_pairs",
        "true_pairs",
        "ground_truth_pairs",
        "candidate_true_pairs",
    ]:

        if key in meta:

            try:
                total_truth_pairs = int(
                    meta[key]
                )
                break

            except Exception:
                pass

    present_truth_pairs = int(
        df["true_label"].sum()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "8. BLOCKER / CANDIDATE COVERAGE"
    )

    print(
        "=" * 80
    )

    print(
        f"True pairs present in candidate set: "
        f"{present_truth_pairs:,}"
    )

    if total_truth_pairs is not None:

        blocker_recall = safe_div(
            present_truth_pairs,
            total_truth_pairs,
        )

        missed = (
            total_truth_pairs
            - present_truth_pairs
        )

        print(
            f"Total ground-truth true pairs: "
            f"{total_truth_pairs:,}"
        )

        print(
            f"Candidate recall: "
            f"{blocker_recall:.6f}"
        )

        print(
            f"True pairs missing before 03: "
            f"{missed:,}"
        )

    else:

        print(
            "Total ground-truth pair count was "
            "not found in model metadata."
        )

        print(
            "Therefore blocker recall cannot be "
            "computed from this file alone."
        )

        print(
            "The current candidate set contains "
            f"{present_truth_pairs:,} true pairs."
        )

    print(
        "\nCandidate count per S1:"
    )

    counts = (
        df.groupby(
            "s1_entity_id"
        )
        .size()
    )

    print(
        counts.describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )


# ============================================================
# 9. RETRIEVAL ROUTE ANALYSIS
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

    total_true_pairs = int(
        df["true_label"].sum()
    )

    for route in sorted(
        route_names
    ):

        mask = df[
            "retrieval_channels"
        ].apply(
            lambda x:
            route
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

        precision = safe_div(
            true_positive,
            predicted,
        )

        recall_contribution = safe_div(
            true_positive,
            total_true_pairs,
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
                "true_pair_recall_contribution":
                    recall_contribution,
                "mean_probability":
                    subset[
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
        "9. RETRIEVAL ROUTE ANALYSIS"
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
# 10. FEATURE SEPARATION
# ============================================================

def feature_separation(df):

    exclude = {
        "true_label",
        "match_prediction",
        "match_probability",
        "model_rank",
    }

    numeric_columns = [
        c
        for c in df.columns
        if c not in exclude
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
                "mean_gap":
                    pos_mean - neg_mean,
                "positive_median":
                    positive.median(),
                "negative_median":
                    negative.median(),
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
        "10. FEATURE SEPARATION"
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
# 11. ERROR CONCENTRATION
# ============================================================

def error_concentration(df):

    """
    Finds entities where errors are concentrated.

    This matters because macro F0.5 is entity-level.
    """

    entity = (
        df.groupby(
            "s1_entity_id"
        )
        .agg(
            candidates=(
                "candidate_entity_id",
                "count",
            ),
            truth=(
                "true_label",
                "sum",
            ),
            predicted=(
                "match_prediction",
                "sum",
            ),
        )
        .reset_index()
    )

    entity["tp"] = (
        df.assign(
            tp=(
                (df["true_label"] == 1)
                &
                (df["match_prediction"] == 1)
            ).astype(int)
        )
        .groupby(
            "s1_entity_id"
        )["tp"]
        .sum()
        .values
    )

    entity["fp"] = (
        entity["predicted"]
        - entity["tp"]
    )

    entity["fn"] = (
        entity["truth"]
        - entity["tp"]
    )

    entity["error_count"] = (
        entity["fp"]
        + entity["fn"]
    )

    worst = entity.sort_values(
        [
            "error_count",
            "candidates",
        ],
        ascending=False,
    )

    path = (
        OUTPUT_DIR
        / "error_concentration.tsv"
    )

    worst.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "11. ERROR CONCENTRATION"
    )

    print(
        "=" * 80
    )

    print(
        "Worst entities by total FP + FN:"
    )

    print(
        worst.head(30).to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {path}"
    )

    return worst


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "03 — MATCHER + CHALLENGE DIAGNOSTICS"
    )

    print(
        "=" * 80
    )

    if not DIAGNOSTICS_PATH.exists():

        raise FileNotFoundError(
            f"\nMissing:\n"
            f"{DIAGNOSTICS_PATH}\n\n"
            f"Run 03 matcher first."
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

    required = {
        "s1_entity_id",
        "candidate_entity_id",
        "true_label",
        "match_prediction",
        "match_probability",
        "model_rank",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                sorted(missing)
            )
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

    tp = int(((df["true_label"] == 1) & (df["match_prediction"] == 1)).sum())
    predicted_positive = int(df["match_prediction"].sum())

    precision = safe_div(tp, predicted_positive)

    print(f"Current pairwise precision: {precision:.6f}")

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    threshold_analysis(df)

    challenge_metric_analysis(df)

    score_distribution(df)

    false_negative_analysis(df)

    false_positive_analysis(df)

    rank_analysis(df)

    entity_analysis(df)

    candidate_coverage_analysis(df)

    route_analysis(df)

    feature_separation(df)

    error_concentration(df)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    entity_metrics = challenge_metric_analysis(df)

    macro_f05 = entity_metrics[
        "entity_f05"
    ].mean()

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL DIAGNOSTIC SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Pairwise predicted pairs : "
        f"{int(df['match_prediction'].sum()):,}"
    )

    print(
        f"Pairwise true pairs      : "
        f"{int(df['true_label'].sum()):,}"
    )

    print(
        f"Macro entity F0.5        : "
        f"{macro_f05:.6f}"
    )

    print(
        f"Singleton entities       : "
        f"{int((entity_metrics['truth_count'] == 0).sum()):,}"
    )

    correct_singletons = int(
        (
            (entity_metrics["truth_count"] == 0)
            & (entity_metrics["predicted_count"] == 0)
        ).sum()
    )

    print(f"Correct singletons       : {correct_singletons:,}")

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
        "\nAll diagnostic files:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()