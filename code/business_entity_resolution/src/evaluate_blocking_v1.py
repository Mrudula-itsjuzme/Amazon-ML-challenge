from pathlib import Path
import ast
import pandas as pd


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "processed"
TRAIN_DIR = REPO_ROOT.parent / "student_resource" / "dataset" / "train"


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

print("=" * 70)
print("CANDIDATE GENERATOR V1")
print("Exact name_norm blocking")
print("=" * 70)

print("\nLoading processed sources...")

s1 = pd.read_parquet(
    PROCESSED_DIR / "train_source1.parquet",
    columns=["entity_id", "name_norm"],
)

s2 = pd.read_parquet(
    PROCESSED_DIR / "train_source2.parquet",
    columns=["entity_id", "name_norm"],
)

s3 = pd.read_parquet(
    PROCESSED_DIR / "train_source3.parquet",
    columns=["entity_id", "name_norm"],
)

print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")


# ============================================================
# BUILD EXACT NAME INDEX
# ============================================================

print("\nBuilding exact name index...")

s2_index = (
    s2.groupby("name_norm")["entity_id"]
    .agg(list)
    .to_dict()
)

s3_index = (
    s3.groupby("name_norm")["entity_id"]
    .agg(list)
    .to_dict()
)

print(f"S2 name keys: {len(s2_index):,}")
print(f"S3 name keys: {len(s3_index):,}")


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("\nGenerating V1 candidates...")

candidate_counts = []
total_candidates = 0
matched_s1_with_candidates = 0

for row in s1.itertuples(index=False):

    name = row.name_norm

    s2_candidates = s2_index.get(name, [])
    s3_candidates = s3_index.get(name, [])

    count = len(s2_candidates) + len(s3_candidates)

    if count > 0:
        matched_s1_with_candidates += 1

    total_candidates += count

    candidate_counts.append(count)


# ============================================================
# CANDIDATE STATISTICS
# ============================================================

candidate_counts = pd.Series(candidate_counts)

print("\n" + "=" * 70)
print("V1 CANDIDATE STATISTICS")
print("=" * 70)

print(f"S1 records:                 {len(s1):,}")
print(f"S1 with >=1 candidate:      {matched_s1_with_candidates:,}")
print(f"S1 with 0 candidates:       {(candidate_counts == 0).sum():,}")
print(f"Total candidate pairs:      {total_candidates:,}")

print(
    f"Average candidates / S1:    "
    f"{candidate_counts.mean():.2f}"
)

print(
    f"Median candidates / S1:     "
    f"{candidate_counts.median():.0f}"
)

print(
    f"Maximum candidates / S1:    "
    f"{candidate_counts.max():,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    TRAIN_DIR / "train_ground_truth.tsv",
    sep="\t",
    dtype=str,
)

print(f"Ground-truth S1 rows: {len(gt):,}")


# ============================================================
# BUILD CANDIDATE SETS FOR RECALL CHECK
# ============================================================

print("\nEvaluating candidate recall...")

# For memory efficiency, create sets only for the names
# that occur in S1.

candidate_sets = {}

for row in s1.itertuples(index=False):

    name = row.name_norm

    candidates = set(
        s2_index.get(name, [])
        + s3_index.get(name, [])
    )

    candidate_sets[row.entity_id] = candidates


# ============================================================
# RECALL EVALUATION
# ============================================================

total_gt_pairs = 0
recovered_gt_pairs = 0

s1_with_gt = 0
s1_with_all_gt_recovered = 0

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id
    raw_matched = row.matched_entity_ids

    if pd.isna(raw_matched) or raw_matched is None:
        matched_ids = set()
    else:
        raw_matched = str(raw_matched).strip()

        if not raw_matched:
            matched_ids = set()
        elif raw_matched.startswith("[") and raw_matched.endswith("]"):
            try:
                parsed = ast.literal_eval(raw_matched)
                matched_ids = set(parsed)
            except (ValueError, SyntaxError):
                matched_ids = set()
        else:
            matched_ids = {
                x.strip()
                for x in raw_matched.split(",")
                if str(x).strip()
            }

    if not matched_ids:
        continue

    s1_with_gt += 1
    total_gt_pairs += len(matched_ids)

    candidates = candidate_sets.get(
        s1_id,
        set()
    )

    recovered = matched_ids.intersection(
        candidates
    )

    recovered_gt_pairs += len(recovered)

    if recovered == matched_ids:
        s1_with_all_gt_recovered += 1


# ============================================================
# FINAL RECALL
# ============================================================

pair_recall = (
    recovered_gt_pairs / total_gt_pairs
    if total_gt_pairs
    else 0
)

s1_recall = (
    s1_with_all_gt_recovered / s1_with_gt
    if s1_with_gt
    else 0
)


print("\n" + "=" * 70)
print("V1 RECALL RESULT")
print("=" * 70)

print(f"Ground-truth matched pairs:       {total_gt_pairs:,}")
print(f"Recovered ground-truth pairs:     {recovered_gt_pairs:,}")
print(f"Missed ground-truth pairs:        {total_gt_pairs - recovered_gt_pairs:,}")

print(
    f"\nPAIR-LEVEL CANDIDATE RECALL:      "
    f"{pair_recall:.4%}"
)

print(
    f"S1 with GT matches:               "
    f"{s1_with_gt:,}"
)

print(
    f"S1 with ALL GT matches recovered: "
    f"{s1_with_all_gt_recovered:,}"
)

print(
    f"S1-LEVEL FULL RECALL:             "
    f"{s1_recall:.4%}"
)

print("\nV1 evaluation complete.")