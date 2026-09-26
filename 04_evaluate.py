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

CHUNK_SIZE = 500_000


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

    print(
        "Ground-truth columns:",
        list(gt.columns),
    )

    if (
        "source1_entity_id" not in gt.columns
        or "matched_entity_ids" not in gt.columns
    ):
        raise ValueError(
            "Expected columns "
            "'source1_entity_id' and "
            "'matched_entity_ids'"
        )

    truth = {}

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id

        matched_ids = parse_ids(
            row.matched_entity_ids
        )

        truth[s1_id] = matched_ids

    return truth


# ============================================================
# BUILD VECTORISED GROUND-TRUTH PAIRS
# ============================================================

def build_truth_pairs(truth):

    rows = []

    for s1_id, true_ids in truth.items():

        for target_id in true_ids:

            rows.append(
                (
                    s1_id,
                    target_id,
                )
            )

    truth_pairs = pd.DataFrame(
        rows,
        columns=[
            "s1_entity_id",
            "candidate_entity_id",
        ],
    )

    return truth_pairs


# ============================================================
# EVALUATION
# ============================================================

def evaluate_file(
    truth,
    truth_pairs,
    path,
    label,
):

    print(
        f"\n{'=' * 70}"
    )

    print(label)

    print(
        f"{'=' * 70}"
    )

    if not path.exists():

        print(
            f"File not found: {path}"
        )

        return

    total_true_pairs = len(truth_pairs)

    total_entities = len(truth)

    no_match_entities = sum(
        1
        for ids in truth.values()
        if not ids
    )

    total_predictions = 0
    tp = 0

    # Per-S1 statistics.
    predicted_counts = {}
    correct_counts = {}

    # We only need this for reporting missed true pairs.
    recovered_truth = {}

    chunk_number = 0

    print(
        f"Input: {path}"
    )

    print(
        f"Chunk size: {CHUNK_SIZE:,}"
    )

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    # --------------------------------------------------------
    # Evaluate each chunk using pandas operations rather than
    # Python row-by-row loops.
    # --------------------------------------------------------

    for chunk in reader:

        chunk_number += 1

        n = len(chunk)

        total_predictions += n

        # ----------------------------------------------------
        # Predicted candidate count per S1.
        # ----------------------------------------------------

        pred_counts = (
            chunk["s1_entity_id"]
            .value_counts()
        )

        for s1_id, count in pred_counts.items():

            predicted_counts[s1_id] = (
                predicted_counts.get(
                    s1_id,
                    0,
                )
                + int(count)
            )

        # ----------------------------------------------------
        # Find true pairs using a vectorised merge.
        # ----------------------------------------------------

        matched = chunk.merge(
            truth_pairs,
            on=[
                "s1_entity_id",
                "candidate_entity_id",
            ],
            how="inner",
        )

        chunk_tp = len(matched)

        tp += chunk_tp

        # ----------------------------------------------------
        # Correct predictions per S1.
        # ----------------------------------------------------

        if chunk_tp:

            correct = (
                matched["s1_entity_id"]
                .value_counts()
            )

            for s1_id, count in correct.items():

                correct_counts[s1_id] = (
                    correct_counts.get(
                        s1_id,
                        0,
                    )
                    + int(count)
                )

            # ------------------------------------------------
            # Track recovered truth IDs.
            # ------------------------------------------------

            for row in matched.itertuples(
                index=False
            ):

                recovered_truth.setdefault(
                    row.s1_entity_id,
                    set(),
                ).add(
                    row.candidate_entity_id
                )

        print(
            f"Processed chunk "
            f"{chunk_number:,} | "
            f"Predictions: "
            f"{total_predictions:,} | "
            f"TP: {tp:,}"
        )

    # ========================================================
    # PAIR METRICS
    # ========================================================

    fp = (
        total_predictions
        - tp
    )

    fn = (
        total_true_pairs
        - tp
    )

    precision = (
        tp / total_predictions
        if total_predictions
        else 0.0
    )

    recall = (
        tp / total_true_pairs
        if total_true_pairs
        else 1.0
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
        else 0.0
    )

    # For both files this is the fraction of all known true
    # pairs contained in this file.
    candidate_recall = recall

    # ========================================================
    # MACRO ENTITY F0.5
    # ========================================================

    entity_f05 = []

    for s1_id, true_ids in truth.items():

        predicted_count = predicted_counts.get(
            s1_id,
            0,
        )

        correct_count = correct_counts.get(
            s1_id,
            0,
        )

        # Precision.
        if predicted_count:

            entity_precision = (
                correct_count
                / predicted_count
            )

        else:

            entity_precision = (
                1.0
                if not true_ids
                else 0.0
            )

        # Recall.
        if true_ids:

            entity_recall = (
                correct_count
                / len(true_ids)
            )

        else:

            entity_recall = (
                1.0
                if predicted_count == 0
                else 0.0
            )

        # F0.5.
        if (
            entity_precision
            + entity_recall
            > 0
        ):

            entity_f05_value = (
                1.25
                * entity_precision
                * entity_recall
                / (
                    0.25
                    * entity_precision
                    + entity_recall
                )
            )

        else:

            entity_f05_value = 0.0

        entity_f05.append(
            entity_f05_value
        )

    # ========================================================
    # NO-MATCH ACCURACY
    # ========================================================

    singleton_correct = 0

    for s1_id, true_ids in truth.items():

        if not true_ids:

            if (
                predicted_counts.get(
                    s1_id,
                    0,
                )
                == 0
            ):

                singleton_correct += 1

    singleton_accuracy = (
        singleton_correct
        / no_match_entities
        if no_match_entities
        else 0.0
    )

    # ========================================================
    # CANDIDATE COUNT STATISTICS
    # ========================================================

    candidate_counts = np.array(
        [
            predicted_counts.get(
                s1_id,
                0,
            )
            for s1_id in truth
        ],
        dtype=np.int64,
    )

    # ========================================================
    # MISSED TRUE PAIRS
    # ========================================================

    missed_entities = 0
    missed_pairs = 0

    for s1_id, true_ids in truth.items():

        recovered = recovered_truth.get(
            s1_id,
            set(),
        )

        missing = (
            true_ids
            - recovered
        )

        if missing:

            missed_entities += 1
            missed_pairs += len(missing)

    # ========================================================
    # RESULTS
    # ========================================================

    print("\nResults")
    print("-" * 70)

    print(
        f"Ground-truth entities: "
        f"{total_entities:,}"
    )

    print(
        f"Ground-truth true pairs: "
        f"{total_true_pairs:,}"
    )

    print(
        f"Predicted pairs: "
        f"{total_predictions:,}"
    )

    print(
        f"TP: {tp:,}"
    )

    print(
        f"FP: {fp:,}"
    )

    print(
        f"FN: {fn:,}"
    )

    print(
        f"\nPair precision: "
        f"{precision:.6f}"
    )

    print(
        f"Pair recall: "
        f"{recall:.6f}"
    )

    print(
        f"Pair F0.5: "
        f"{f05:.6f}"
    )

    print(
        f"Candidate recall: "
        f"{candidate_recall:.6f}"
    )

    print(
        f"Macro entity F0.5: "
        f"{np.mean(entity_f05):.6f}"
    )

    if no_match_entities:

        print(
            f"No-match entities: "
            f"{no_match_entities:,}"
        )

        print(
            f"No-match accuracy: "
            f"{singleton_accuracy:.6f}"
        )

    print(
        f"\nMean candidates/entity: "
        f"{np.mean(candidate_counts):.2f}"
    )

    print(
        f"Median candidates/entity: "
        f"{np.median(candidate_counts):.2f}"
    )

    print(
        f"95th percentile: "
        f"{np.percentile(candidate_counts, 95):.2f}"
    )

    print(
        f"99th percentile: "
        f"{np.percentile(candidate_counts, 99):.2f}"
    )

    print(
        f"Max candidates/entity: "
        f"{np.max(candidate_counts):,}"
    )

    print(
        f"\nEntities with missed true matches: "
        f"{missed_entities:,}"
    )

    print(
        f"Missed true pairs: "
        f"{missed_pairs:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    truth = load_ground_truth()

    print(
        f"\nLoaded ground truth for "
        f"{len(truth):,} S1 entities."
    )

    truth_pairs = build_truth_pairs(
        truth
    )

    print(
        f"Total true pairs: "
        f"{len(truth_pairs):,}"
    )

    # ========================================================
    # IMPORTANT:
    #
    # Evaluate the 5.7M filtered file FIRST.
    #
    # This tells us immediately how much recall 03_filter
    # lost, without waiting for the 300M blocker evaluation.
    # ========================================================

    evaluate_file(
        truth,
        truth_pairs,
        FINAL_CANDIDATES,
        "FILTERED CANDIDATES — 03_filter_candidates.py",
    )

    # ========================================================
    # Then evaluate the 300M blocker file.
    # ========================================================

    evaluate_file(
        truth,
        truth_pairs,
        CANDIDATES,
        "BLOCKER CANDIDATES — 02_generate_candidates.py",
    )


if __name__ == "__main__":

    main()
