from pathlib import Path
import pandas as pd
import numpy as np
import ast


ROOT = Path(__file__).resolve().parent

GROUND_TRUTH = (
    ROOT
    / "student_resource"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

CANDIDATES = (
    ROOT
    / "candidates"
    / "train_candidate_pairs.tsv"
)

FINAL_CANDIDATES = (
    ROOT
    / "filtered"
    / "train_final_candidate_pairs.tsv"
)


def parse_ids(value):

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    # Python list representation.
    if value.startswith("["):

        try:

            parsed = ast.literal_eval(
                value
            )

            return {
                str(x).strip()
                for x in parsed
                if str(x).strip()
            }

        except Exception:
            pass

    # Common separators.
    for separator in [
        "|",
        ",",
        ";",
    ]:

        if separator in value:

            return {
                x.strip()
                for x in value.split(
                    separator
                )
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

    print(
        "Ground-truth columns:",
        list(gt.columns),
    )

    if (
        "entity_id" not in gt.columns
        or "matched_entity_ids"
        not in gt.columns
    ):
        raise ValueError(
            "Expected columns "
            "'entity_id' and "
            "'matched_entity_ids'"
        )

    truth = {}

    for _, row in gt.iterrows():

        truth[
            row["entity_id"]
        ] = parse_ids(
            row["matched_entity_ids"]
        )

    return truth


def load_predictions(path):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    grouped = (
        df
        .groupby("s1_entity_id")
        ["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )

    return grouped


def evaluate(
    truth,
    predictions,
    label,
):

    print(
        f"\n{'=' * 60}"
    )

    print(label)

    print(
        f"{'=' * 60}"
    )

    tp = 0
    fp = 0
    fn = 0

    entity_f05 = []

    singleton_total = 0
    singleton_correct = 0

    for s1_id, true_ids in truth.items():

        predicted_ids = predictions.get(
            s1_id,
            set(),
        )

        correct = (
            true_ids
            & predicted_ids
        )

        false_positive = (
            predicted_ids
            - true_ids
        )

        false_negative = (
            true_ids
            - predicted_ids
        )

        tp += len(correct)
        fp += len(false_positive)
        fn += len(false_negative)

        # -----------------------------
        # Entity-level precision
        # -----------------------------

        if predicted_ids:

            precision = (
                len(correct)
                / len(predicted_ids)
            )

        else:

            precision = (
                1.0
                if not true_ids
                else 0.0
            )

        # -----------------------------
        # Entity-level recall
        # -----------------------------

        if true_ids:

            recall = (
                len(correct)
                / len(true_ids)
            )

        else:

            recall = (
                1.0
                if not predicted_ids
                else 0.0
            )

        # -----------------------------
        # F0.5
        # -----------------------------

        if (
            precision + recall
            > 0
        ):

            f05 = (
                1.25
                * precision
                * recall
                / (
                    0.25 * precision
                    + recall
                )
            )

        else:

            f05 = 0.0

        entity_f05.append(f05)

        # -----------------------------
        # Singleton/no-match
        # -----------------------------

        if not true_ids:

            singleton_total += 1

            if not predicted_ids:
                singleton_correct += 1

    # -----------------------------
    # Pair-level metrics
    # -----------------------------

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0
    )

    f05 = (
        1.25
        * precision
        * recall
        / (
            0.25 * precision
            + recall
        )
        if precision + recall
        else 0
    )

    # -----------------------------
    # Print
    # -----------------------------

    print(f"TP: {tp:,}")
    print(f"FP: {fp:,}")
    print(f"FN: {fn:,}")

    print(
        f"Pair precision: "
        f"{precision:.4f}"
    )

    print(
        f"Pair recall: "
        f"{recall:.4f}"
    )

    print(
        f"Pair F0.5: "
        f"{f05:.4f}"
    )

    print(
        f"Macro entity F0.5: "
        f"{np.mean(entity_f05):.4f}"
    )

    if singleton_total:

        print(
            f"Singleton/no-match accuracy: "
            f"{singleton_correct / singleton_total:.4f}"
        )

    candidate_counts = [
        len(
            predictions.get(
                s1_id,
                set(),
            )
        )
        for s1_id in truth
    ]

    print(
        f"Mean candidates/entity: "
        f"{np.mean(candidate_counts):.2f}"
    )

    print(
        f"Median candidates/entity: "
        f"{np.median(candidate_counts):.2f}"
    )

    print(
        f"Max candidates/entity: "
        f"{np.max(candidate_counts):,}"
    )


def main():

    truth = load_ground_truth()

    raw = load_predictions(
        CANDIDATES
    )

    final = load_predictions(
        FINAL_CANDIDATES
    )

    evaluate(
        truth,
        raw,
        "BLOCKER CANDIDATES",
    )

    evaluate(
        truth,
        final,
        "FILTERED CANDIDATES",
    )


if __name__ == "__main__":
    main()