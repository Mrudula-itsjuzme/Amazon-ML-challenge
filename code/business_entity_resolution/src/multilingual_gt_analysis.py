from pathlib import Path
from collections import Counter
import re
import unicodedata

import pandas as pd


# ============================================================
# PATHS
# ============================================================

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = (
    WORKSPACE_ROOT
    / "student_resource"
    / "dataset"
    / "train"
)


# ============================================================
# SCRIPT DETECTION
# ============================================================

def detect_script(text: str) -> str:
    """
    Detect the dominant writing script in a string.

    This is analysis only.
    It is NOT used as a matching rule.
    """

    if not isinstance(text, str) or not text.strip():
        return "EMPTY"

    script_counts = Counter()

    for char in text:

        if not char.isalpha():
            continue

        try:
            char_name = unicodedata.name(char)
        except ValueError:
            continue

        if "LATIN" in char_name:
            script = "LATIN"

        elif "DEVANAGARI" in char_name:
            script = "DEVANAGARI"

        elif "BENGALI" in char_name:
            script = "BENGALI"

        elif "GURMUKHI" in char_name:
            script = "GURMUKHI"

        elif "GUJARATI" in char_name:
            script = "GUJARATI"

        elif "ORIYA" in char_name or "ODIA" in char_name:
            script = "ODIA"

        elif "TAMIL" in char_name:
            script = "TAMIL"

        elif "TELUGU" in char_name:
            script = "TELUGU"

        elif "KANNADA" in char_name:
            script = "KANNADA"

        elif "MALAYALAM" in char_name:
            script = "MALAYALAM"

        elif "ARABIC" in char_name:
            script = "ARABIC"

        elif "CYRILLIC" in char_name:
            script = "CYRILLIC"

        elif "GREEK" in char_name:
            script = "GREEK"

        elif (
            "CJK" in char_name
            or "HIRAGANA" in char_name
            or "KATAKANA" in char_name
        ):
            script = "CJK_OR_JAPANESE"

        else:
            script = "OTHER"

        script_counts[script] += 1

    if not script_counts:
        return "NO_LETTERS"

    return script_counts.most_common(1)[0][0]


# ============================================================
# GROUND TRUTH PARSING
# ============================================================

def load_ground_truth():

    gt_path = TRAIN_DIR / "train_ground_truth.tsv"

    print("\nLoading ground truth:")
    print(gt_path)

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    print(f"Ground-truth rows: {len(gt):,}")

    return gt


def collect_matched_ids(gt):

    matched_s2_ids = set()
    matched_s3_ids = set()

    # Keep the S1 -> matched IDs relationship for later.
    relationships = []

    print("\nParsing ground-truth matches...")

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id
        matched_ids_string = row.matched_entity_ids

        if not matched_ids_string:
            continue

        matched_ids = matched_ids_string.split(",")

        for matched_id in matched_ids:

            matched_id = matched_id.strip()

            if not matched_id:
                continue

            if matched_id.startswith("S2-"):
                matched_s2_ids.add(matched_id)

            elif matched_id.startswith("S3-"):
                matched_s3_ids.add(matched_id)

        relationships.append(
            (
                s1_id,
                matched_ids,
            )
        )

    print(f"Unique matched S2 IDs: {len(matched_s2_ids):,}")
    print(f"Unique matched S3 IDs: {len(matched_s3_ids):,}")

    return (
        matched_s2_ids,
        matched_s3_ids,
        relationships,
    )


# ============================================================
# FIND NON-LATIN MATCHED RECORDS
# ============================================================

def scan_source_for_nonlatin_matches(
    source_path,
    matched_ids,
    source_name,
    chunk_size=250_000,
):

    print("\n" + "=" * 60)
    print(f"SCANNING {source_name}")
    print("=" * 60)

    print(f"File: {source_path}")
    print(f"Matched IDs to inspect: {len(matched_ids):,}")

    found_records = {}

    if not matched_ids:
        return found_records

    chunks = pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
        ],
        chunksize=chunk_size,
    )

    rows_scanned = 0

    for chunk_number, chunk in enumerate(chunks, start=1):

        rows_scanned += len(chunk)

        # Only keep records that actually occur in GT.
        matched = chunk[
            chunk["entity_id"].isin(matched_ids)
        ]

        if matched.empty:
            continue

        for row in matched.itertuples(index=False):

            script = detect_script(row.business_name)

            if script not in {
                "LATIN",
                "EMPTY",
                "NO_LETTERS",
            }:

                found_records[row.entity_id] = {
                    "business_name": row.business_name,
                    "script": script,
                }

        if chunk_number % 5 == 0:
            print(
                f"  Scanned {rows_scanned:,} rows..."
            )

    print(
        f"\nNon-Latin matched {source_name} records: "
        f"{len(found_records):,}"
    )

    return found_records


# ============================================================
# BUILD NON-LATIN MATCH RELATIONSHIPS
# ============================================================

def build_multilingual_relationships(
    relationships,
    nonlatin_s2,
    nonlatin_s3,
):

    results = []

    nonlatin_ids = (
        set(nonlatin_s2.keys())
        | set(nonlatin_s3.keys())
    )

    print("\nBuilding S1 ↔ non-Latin relationships...")

    for s1_id, matched_ids in relationships:

        for matched_id in matched_ids:

            if matched_id not in nonlatin_ids:
                continue

            if matched_id.startswith("S2-"):
                record = nonlatin_s2[matched_id]
                source = "S2"

            else:
                record = nonlatin_s3[matched_id]
                source = "S3"

            results.append(
                {
                    "s1_id": s1_id,
                    "matched_id": matched_id,
                    "source": source,
                    "script": record["script"],
                    "matched_name": record["business_name"],
                }
            )

    return results


# ============================================================
# LOAD S1 NAMES FOR RELEVANT RECORDS
# ============================================================

def load_s1_names(
    s1_ids,
    chunk_size=250_000,
):

    print("\n" + "=" * 60)
    print("LOADING RELEVANT S1 NAMES")
    print("=" * 60)

    s1_path = TRAIN_DIR / "train_source1.tsv"

    results = {}

    chunks = pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
        ],
        chunksize=chunk_size,
    )

    rows_scanned = 0

    for chunk in chunks:

        rows_scanned += len(chunk)

        matched = chunk[
            chunk["entity_id"].isin(s1_ids)
        ]

        for row in matched.itertuples(index=False):

            results[row.entity_id] = row.business_name

        if len(results) >= len(s1_ids):
            break

    print(
        f"Found {len(results):,} / "
        f"{len(s1_ids):,} S1 records."
    )

    return results


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 60)
    print("MULTILINGUAL GROUND-TRUTH MATCH ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load GT
    # --------------------------------------------------------

    gt = load_ground_truth()

    # --------------------------------------------------------
    # 2. Collect actual matched S2/S3 IDs
    # --------------------------------------------------------

    (
        matched_s2_ids,
        matched_s3_ids,
        relationships,
    ) = collect_matched_ids(gt)

    # --------------------------------------------------------
    # 3. Scan S2
    # --------------------------------------------------------

    nonlatin_s2 = scan_source_for_nonlatin_matches(
        TRAIN_DIR / "train_source2.tsv",
        matched_s2_ids,
        "S2",
    )

    # --------------------------------------------------------
    # 4. Scan S3
    # --------------------------------------------------------

    nonlatin_s3 = scan_source_for_nonlatin_matches(
        TRAIN_DIR / "train_source3.tsv",
        matched_s3_ids,
        "S3",
    )

    # --------------------------------------------------------
    # 5. Build multilingual relationships
    # --------------------------------------------------------

    multilingual_pairs = build_multilingual_relationships(
        relationships,
        nonlatin_s2,
        nonlatin_s3,
    )

    print(
        f"\nTotal S1 ↔ non-Latin matched pairs: "
        f"{len(multilingual_pairs):,}"
    )

    # --------------------------------------------------------
    # 6. Load corresponding S1 names
    # --------------------------------------------------------

    s1_ids = {
        pair["s1_id"]
        for pair in multilingual_pairs
    }

    s1_names = load_s1_names(s1_ids)

    # --------------------------------------------------------
    # 7. Script distribution
    # --------------------------------------------------------

    script_counts = Counter(
        pair["script"]
        for pair in multilingual_pairs
    )

    print("\n" + "=" * 60)
    print("NON-LATIN MATCH DISTRIBUTION")
    print("=" * 60)

    for script, count in script_counts.most_common():

        percentage = (
            count / len(multilingual_pairs) * 100
            if multilingual_pairs
            else 0
        )

        print(
            f"{script:20s} "
            f"{count:8,d} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # 8. S2 vs S3
    # --------------------------------------------------------

    source_counts = Counter(
        pair["source"]
        for pair in multilingual_pairs
    )

    print("\nSource distribution:")

    for source, count in source_counts.items():

        print(
            f"  {source}: {count:,}"
        )

    # --------------------------------------------------------
    # 9. Print examples
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ACTUAL MULTILINGUAL GROUND-TRUTH EXAMPLES")
    print("=" * 60)

    if not multilingual_pairs:

        print(
            "\nNo non-Latin matched records were found."
        )

    else:

        for pair in multilingual_pairs[:30]:

            s1_name = s1_names.get(
                pair["s1_id"],
                "[S1 name not found]",
            )

            print("\n----------------------------------------")

            print(
                f"S1 [{pair['s1_id']}]: "
                f"{s1_name}"
            )

            print(
                f"{pair['source']} "
                f"[{pair['matched_id']}] "
                f"({pair['script']}): "
                f"{pair['matched_name']}"
            )

    # --------------------------------------------------------
    # 10. Final interpretation numbers
    # --------------------------------------------------------

    total_gt_pairs = sum(
        len(ids)
        for _, ids in relationships
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Total ground-truth S1 → S2/S3 pairs: "
        f"{total_gt_pairs:,}"
    )

    print(
        f"Ground-truth pairs with non-Latin S2/S3 name: "
        f"{len(multilingual_pairs):,}"
    )

    if total_gt_pairs > 0:

        percentage = (
            len(multilingual_pairs)
            / total_gt_pairs
            * 100
        )

        print(
            f"Percentage of GT pairs involving "
            f"non-Latin names: {percentage:.2f}%"
        )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()