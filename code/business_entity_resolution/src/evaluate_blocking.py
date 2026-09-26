from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "processed"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("BLOCKING BASELINE INSPECTION")
print("=" * 70)

s1 = pd.read_parquet(
    PROCESSED_DIR / "train_source1.parquet",
    columns=[
        "entity_id",
        "name_norm",
        "name_core",
        "name_translit",
        "country_norm",
    ],
)

s2 = pd.read_parquet(
    PROCESSED_DIR / "train_source2.parquet",
    columns=[
        "entity_id",
        "name_norm",
        "name_core",
        "name_translit",
        "country_norm",
    ],
)

s3 = pd.read_parquet(
    PROCESSED_DIR / "train_source3.parquet",
    columns=[
        "entity_id",
        "name_norm",
        "name_core",
        "name_translit",
        "country_norm",
    ],
)


# ============================================================
# BASIC STATISTICS
# ============================================================

for source_name, df in [
    ("S1", s1),
    ("S2", s2),
    ("S3", s3),
]:

    print(f"\n{source_name}")
    print("-" * 70)

    print(f"Rows: {len(df):,}")

    print(
        f"Unique name_norm: "
        f"{df['name_norm'].nunique():,}"
    )

    print(
        f"Unique name_core: "
        f"{df['name_core'].nunique():,}"
    )

    print(
        f"Unique name_translit: "
        f"{df['name_translit'].nunique():,}"
    )

    print(
        f"Unique country_norm: "
        f"{df['country_norm'].nunique():,}"
    )

    print(
        f"Empty name_norm: "
        f"{(df['name_norm'] == '').sum():,}"
    )

    print(
        f"Empty name_core: "
        f"{(df['name_core'] == '').sum():,}"
    )

    print(
        f"Empty name_translit: "
        f"{(df['name_translit'] == '').sum():,}"
    )


print("\nInspection complete.")