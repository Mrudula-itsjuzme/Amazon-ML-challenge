"""Build exact, label-free token document frequencies over the full S2/S3 pool."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("entity_pipeline", Path(__file__).with_name("01_pipeline.py"))
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "student_resource/dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research_runs/global_target_token_df")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Global DF output already exists; preserve the frozen corpus statistics")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name_df, address_df = Counter(), Counter()
    population = 0
    for source_number in (2, 3):
        path = args.data_dir / "train" / f"train_source{source_number}.tsv"
        for chunk in pd.read_csv(path, sep="\t", dtype=str,
                                 keep_default_na=False, chunksize=50000):
            for name, address in zip(chunk.business_name, chunk.business_address):
                name_df.update(set(pipeline.norm(name).split()))
                address_df.update(set(pipeline.norm(address).split()))
            population += len(chunk)
            if population % 1000000 < 50000:
                print(f"Global DF targets: {population:,}", flush=True)
    if population != 10_320_219:
        raise ValueError(f"Unexpected target pool: {population}")
    pd.DataFrame(name_df.items(), columns=["token", "df"]).to_parquet(
        args.output_dir / "name_df.parquet", index=False)
    pd.DataFrame(address_df.items(), columns=["token", "df"]).to_parquet(
        args.output_dir / "address_df.parquet", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "target_pool_size": population, "labels_used": False,
        "name_vocabulary": len(name_df), "address_vocabulary": len(address_df),
    }, indent=2))
    print(f"Full target DF complete: {population:,} targets", flush=True)


if __name__ == "__main__":
    main()
