from pathlib import Path
from collections import defaultdict
import argparse
import pandas as pd


ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT / "processed"
CANDIDATE_DIR = ROOT / "candidates"
CANDIDATE_DIR.mkdir(exist_ok=True)

MAX_POSTINGS = 500


def build_exact_index(df, column):
    index = defaultdict(list)

    for entity_id, value in zip(
        df["entity_id"],
        df[column],
    ):
        if value:
            index[value].append(entity_id)

    return index


def build_token_index(df, column):
    index = defaultdict(set)

    for entity_id, value in zip(
        df["entity_id"],
        df[column],
    ):
        for token in set(value.split()):

            if len(token) < 3:
                continue

            index[token].add(entity_id)

    return index


def build_number_index(df):
    index = defaultdict(set)

    for entity_id, value in zip(
        df["entity_id"],
        df["address_numbers"],
    ):
        for number in set(value.split()):

            if number:
                index[number].add(entity_id)

    return index


def add_candidates(
    candidate_set,
    s1_id,
    target_ids,
    channel,
):
    if not target_ids:
        return

    # Avoid explosive/common postings.
    if len(target_ids) > MAX_POSTINGS:
        return

    for target_id in target_ids:
        candidate_set[
            (s1_id, target_id)
        ].add(channel)


def generate_candidates(split):

    print(f"\n{'=' * 60}")
    print(f"CANDIDATE GENERATION: {split.upper()}")
    print(f"{'=' * 60}")

    s1 = pd.read_csv(
        PROCESSED_DIR
        / f"{split}_S1_processed.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s2 = pd.read_csv(
        PROCESSED_DIR
        / f"{split}_S2_processed.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s3 = pd.read_csv(
        PROCESSED_DIR
        / f"{split}_S3_processed.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    targets = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    print(f"S1 rows: {len(s1):,}")
    print(f"S2 rows: {len(s2):,}")
    print(f"S3 rows: {len(s3):,}")

    print("\nBuilding indexes...")

    name_index = build_exact_index(
        targets,
        "norm_name",
    )

    translit_index = build_exact_index(
        targets,
        "translit_name",
    )

    address_index = build_exact_index(
        targets,
        "norm_address",
    )

    address_token_index = build_token_index(
        targets,
        "norm_address",
    )

    number_index = build_number_index(
        targets
    )

    candidate_set = defaultdict(set)

    print("Generating candidates...")

    for i, row in s1.iterrows():

        s1_id = row["entity_id"]

        # -----------------------------
        # 1. Exact normalized name
        # -----------------------------

        add_candidates(
            candidate_set,
            s1_id,
            name_index.get(
                row["norm_name"],
                [],
            ),
            "exact_name",
        )

        # -----------------------------
        # 2. Transliteration
        # -----------------------------

        add_candidates(
            candidate_set,
            s1_id,
            translit_index.get(
                row["translit_name"],
                [],
            ),
            "translit_name",
        )

        # -----------------------------
        # 3. Exact address
        # -----------------------------

        add_candidates(
            candidate_set,
            s1_id,
            address_index.get(
                row["norm_address"],
                [],
            ),
            "exact_address",
        )

        # -----------------------------
        # 4. Address token
        # -----------------------------

        token_candidates = set()

        for token in set(
            row["norm_address"].split()
        ):

            if len(token) < 3:
                continue

            ids = address_token_index.get(
                token,
                set(),
            )

            if 0 < len(ids) <= MAX_POSTINGS:
                token_candidates.update(ids)

        add_candidates(
            candidate_set,
            s1_id,
            token_candidates,
            "address_token",
        )

        # -----------------------------
        # 5. Address number
        # -----------------------------

        number_candidates = set()

        for number in set(
            row["address_numbers"].split()
        ):

            if not number:
                continue

            ids = number_index.get(
                number,
                set(),
            )

            if 0 < len(ids) <= MAX_POSTINGS:
                number_candidates.update(ids)

        add_candidates(
            candidate_set,
            s1_id,
            number_candidates,
            "address_number",
        )

        if (i + 1) % 100_000 == 0:
            print(
                f"Processed "
                f"{i + 1:,}/{len(s1):,} | "
                f"Candidates: "
                f"{len(candidate_set):,}"
            )

    # -----------------------------
    # Save
    # -----------------------------

    rows = []

    for (s1_id, target_id), channels in candidate_set.items():

        rows.append(
            {
                "s1_entity_id": s1_id,
                "candidate_entity_id": target_id,
                "retrieval_channels": ",".join(
                    sorted(channels)
                ),
            }
        )

    result = pd.DataFrame(rows)

    output = (
        CANDIDATE_DIR
        / f"{split}_candidate_pairs.tsv"
    )

    result.to_csv(
        output,
        sep="\t",
        index=False,
    )

    print("\nDone.")
    print(
        f"Candidate pairs: "
        f"{len(result):,}"
    )
    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "test"],
        required=True,
    )

    args = parser.parse_args()

    generate_candidates(
        args.split
    )