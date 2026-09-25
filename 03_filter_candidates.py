from pathlib import Path
import argparse
import pandas as pd

from rapidfuzz.fuzz import ratio


ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT / "processed"
CANDIDATE_DIR = ROOT / "candidates"
FILTERED_DIR = ROOT / "filtered"

FILTERED_DIR.mkdir(exist_ok=True)


def similarity(a, b):

    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


def token_jaccard(a, b):

    a = set(a.split())
    b = set(b.split())

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def numeric_jaccard(a, b):

    a = set(a.split())
    b = set(b.split())

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def compute_features(a, b):

    return {
        "name_seq": similarity(
            a["norm_name"],
            b["norm_name"],
        ),

        "translit_seq": similarity(
            a["translit_name"],
            b["translit_name"],
        ),

        "addr_seq": similarity(
            a["norm_address"],
            b["norm_address"],
        ),

        "name_jaccard": token_jaccard(
            a["norm_name"],
            b["norm_name"],
        ),

        "addr_jaccard": token_jaccard(
            a["norm_address"],
            b["norm_address"],
        ),

        "addr_num_jaccard": numeric_jaccard(
            a["address_numbers"],
            b["address_numbers"],
        ),

        "same_country": int(
            bool(a["country_norm"])
            and bool(b["country_norm"])
            and a["country_norm"]
            == b["country_norm"]
        ),

        "address_missing": int(
            not a["norm_address"]
            or not b["norm_address"]
        ),
    }


def keep_candidate(features):

    name = features["name_seq"]
    translit = features["translit_seq"]
    addr = features["addr_seq"]
    addr_j = features["addr_jaccard"]
    num = features["addr_num_jaccard"]
    country = features["same_country"]
    missing = features["address_missing"]

    # Strong name + strong address evidence.
    if name >= 0.90:

        if num >= 0.50:
            return True

        if addr >= 0.70:
            return True

        if addr_j >= 0.50:
            return True

        # Strong name with missing address and
        # matching country.
        if missing and country:
            return True

    # Strong transliterated name.
    if translit >= 0.88:

        if num >= 0.50:
            return True

        if addr >= 0.72:
            return True

        if addr_j >= 0.50:
            return True

    # Strong address + reasonable name.
    if addr >= 0.88 and name >= 0.55:
        return True

    # Strong numeric address evidence.
    if num >= 0.75 and name >= 0.55:
        return True

    return False


def filter_candidates(split):

    print(f"\n{'=' * 60}")
    print(f"FILTERING: {split.upper()}")
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

    candidates = pd.read_csv(
        CANDIDATE_DIR
        / f"{split}_candidate_pairs.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    targets = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    s1_lookup = (
        s1
        .set_index("entity_id")
        .to_dict("index")
    )

    target_lookup = (
        targets
        .set_index("entity_id")
        .to_dict("index")
    )

    print(
        f"Input candidates: "
        f"{len(candidates):,}"
    )

    output_rows = []

    for i, row in candidates.iterrows():

        s1_id = row[
            "s1_entity_id"
        ]

        target_id = row[
            "candidate_entity_id"
        ]

        a = s1_lookup.get(s1_id)
        b = target_lookup.get(target_id)

        if a is None or b is None:
            continue

        features = compute_features(
            a,
            b,
        )

        if keep_candidate(features):

            output_rows.append(
                {
                    "s1_entity_id": s1_id,
                    "candidate_entity_id": target_id,
                    "retrieval_channels":
                        row[
                            "retrieval_channels"
                        ],
                    **features,
                }
            )

        if (i + 1) % 100_000 == 0:

            print(
                f"Processed "
                f"{i + 1:,}/"
                f"{len(candidates):,}"
            )

    result = pd.DataFrame(
        output_rows
    )

    output = (
        FILTERED_DIR
        / f"{split}_final_candidate_pairs.tsv"
    )

    result.to_csv(
        output,
        sep="\t",
        index=False,
    )

    print("\nDone.")

    print(
        f"Final candidates: "
        f"{len(result):,}"
    )

    if len(candidates):
        print(
            f"Retention: "
            f"{len(result) / len(candidates):.2%}"
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

    filter_candidates(
        args.split
    )