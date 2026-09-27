from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent

TEST_S1 = (
    ROOT
    / "processed"
    / "test_S1_processed.tsv"
)

FINAL_CANDIDATES = (
    ROOT
    / "filtered"
    / "test_final_candidate_pairs.tsv"
)

SUBMISSION_DIR = (
    ROOT
    / "submission"
)

SUBMISSION_DIR.mkdir(
    exist_ok=True
)

OUTPUT = (
    SUBMISSION_DIR
    / "matching_results.tsv"
)


def make_submission():

    s1 = pd.read_csv(
        TEST_S1,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    candidates = pd.read_csv(
        FINAL_CANDIDATES,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    grouped = (
        candidates
        .groupby("s1_entity_id")
        ["candidate_entity_id"]
        .apply(list)
        .to_dict()
    )

    rows = []

    for entity_id in s1["entity_id"]:

        matches = grouped.get(
            entity_id,
            [],
        )

        # Remove duplicates.
        matches = list(
            dict.fromkeys(matches)
        )

        rows.append(
            {
                "entity_id": entity_id,
                "matched_entity_ids":
                    "|".join(matches),
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT,
        sep="\t",
        index=False,
    )

    print(
        f"Saved: {OUTPUT}"
    )

    print(
        f"Rows: {len(result):,}"
    )

    print(
        "Empty predictions:",
        (
            result[
                "matched_entity_ids"
            ] == ""
        ).sum(),
    )


if __name__ == "__main__":
    make_submission()