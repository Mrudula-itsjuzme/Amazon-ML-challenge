from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "processed"

files = [
    PROCESSED_DIR / "train_source1.parquet",
    PROCESSED_DIR / "train_source2.parquet",
    PROCESSED_DIR / "train_source3.parquet",
]

print("=" * 70)
print("PROCESSED DATA VERIFICATION")
print("=" * 70)

for path in files:

    df = pd.read_parquet(path)

    print(f"\n{path.name}")
    print("-" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"name_translit exists: {'name_translit' in df.columns}")

    if "name_translit" in df.columns:

        missing = df["name_translit"].isna().sum()

        print(f"name_translit missing: {missing:,}")

        # Records where transliteration actually changed the name
        changed = df[
            df["name_norm"] != df["name_translit"]
        ]

        print(
            f"Names changed by transliteration: "
            f"{len(changed):,}"
        )

        if len(changed) > 0:

            print("\nExamples:")

            print(
                changed[
                    [
                        "business_name",
                        "name_norm",
                        "name_translit",
                    ]
                ]
                .head(10)
                .to_string(index=False)
            )

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)