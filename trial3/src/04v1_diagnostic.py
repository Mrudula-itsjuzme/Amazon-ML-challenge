from pathlib import Path
import pandas as pd
import numpy as np
import ast
import re
import unicodedata


ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT / "processed"

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

S1_FILE = PROCESSED_DIR / "train_S1_processed.tsv"
S2_FILE = PROCESSED_DIR / "train_S2_processed.tsv"
S3_FILE = PROCESSED_DIR / "train_S3_processed.tsv"

CHUNK_SIZE = 500_000

S1_SAMPLE_SIZE = 5_000
RANDOM_SEED = 32


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

    truth = {}

    for row in gt.itertuples(index=False):

        truth[row.source1_entity_id] = parse_ids(
            row.matched_entity_ids
        )

    return truth


# ============================================================
# GET EXACT SAME 5K S1 SAMPLE
# ============================================================

def get_selected_s1():

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s1 = s1.sample(
        n=S1_SAMPLE_SIZE,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return s1


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value)

    if not value:
        return ""

    value = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()

    value = re.sub(
        r"[^\w]+",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def tokenize(value):

    value = normalize_text(value)

    if not value:
        return set()

    return {
        x
        for x in value.split()
        if len(x) >= 2
    }


def extract_numbers(value):

    if value is None:
        return set()

    return set(
        re.findall(
            r"\d+",
            str(value),
        )
    )


def jaccard(a, b):

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


# ============================================================
# EXTRACT ONLY REQUIRED PROCESSED COLUMNS
# ============================================================

def find_column(columns, *names):

    for name in names:

        if name in columns:
            return name

    return None


def prepare_row(row):

    columns = row.index

    name_col = find_column(
        columns,
        "name_norm",
        "norm_name",
    )

    core_col = find_column(
        columns,
        "name_core",
    )

    translit_col = find_column(
        columns,
        "name_translit",
        "translit_name",
    )

    address_col = find_column(
        columns,
        "address_norm",
        "norm_address",
    )

    country_col = find_column(
        columns,
        "country_norm",
    )

    postal_col = find_column(
        columns,
        "postal_code",
    )

    name = (
        str(row[name_col])
        if name_col
        else ""
    )

    core = (
        str(row[core_col])
        if core_col
        else ""
    )

    translit = (
        str(row[translit_col])
        if translit_col
        else ""
    )

    address = (
        str(row[address_col])
        if address_col
        else ""
    )

    country = (
        str(row[country_col])
        if country_col
        else ""
    )

    postal = (
        str(row[postal_col])
        if postal_col
        else ""
    )

    return {
        "name": name,
        "core": core,
        "translit": translit,
        "address": address,
        "country": country,
        "postal": postal,

        "name_tokens": tokenize(name),
        "address_tokens": tokenize(address),

        "name_numbers": extract_numbers(name),
        "address_numbers": extract_numbers(address),
    }


# ============================================================
# LOAD ONLY REQUIRED S1 RECORDS
# ============================================================

def build_s1_records(s1):

    records = {}

    for _, row in s1.iterrows():

        records[
            row["entity_id"]
        ] = prepare_row(row)

    return records


# ============================================================
# STREAM S2/S3 AND KEEP ONLY MISSED TARGET IDS
# ============================================================

def load_required_targets(target_ids):

    targets = {}

    for path, source_name in [
        (S2_FILE, "S2"),
        (S3_FILE, "S3"),
    ]:

        print(
            f"\nScanning {source_name} "
            f"for {len(target_ids):,} required target IDs..."
        )

        found = 0
        chunks = 0

        # Read only the columns we actually need.
        sample_columns = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            nrows=0,
        ).columns.tolist()

        wanted = [
            "entity_id",
        ]

        for column in [
            "name_norm",
            "name_core",
            "name_translit",
            "address_norm",
            "country_norm",
            "postal_code",
            "norm_name",
            "translit_name",
            "norm_address",
        ]:

            if column in sample_columns:
                wanted.append(column)

        wanted = list(dict.fromkeys(wanted))

        reader = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=wanted,
            chunksize=CHUNK_SIZE,
        )

        for chunk in reader:

            chunks += 1

            mask = chunk["entity_id"].isin(
                target_ids
            )

            matched = chunk.loc[mask]

            if len(matched):

                for _, row in matched.iterrows():

                    entity_id = row["entity_id"]

                    if entity_id not in targets:

                        targets[entity_id] = (
                            prepare_row(row)
                        )

                        found += 1

            if chunks % 5 == 0:

                print(
                    f"  scanned chunk {chunks:,} | "
                    f"found {found:,}/{len(target_ids):,}"
                )

            if found >= len(target_ids):

                print(
                    f"  All required target records found "
                    f"after {chunks:,} chunks."
                )

                break

        print(
            f"Finished {source_name}: "
            f"found {found:,} required records."
        )

    return targets


# ============================================================
# ANALYZE ONE MISSED PAIR
# ============================================================

def analyze_pair(s1, target):

    name_exact = (
        bool(s1["name"])
        and s1["name"] == target["name"]
    )

    core_exact = (
        bool(s1["core"])
        and s1["core"] == target["core"]
    )

    translit_exact = (
        bool(s1["translit"])
        and s1["translit"] == target["translit"]
    )

    address_exact = (
        bool(s1["address"])
        and s1["address"] == target["address"]
    )

    name_overlap = (
        s1["name_tokens"]
        & target["name_tokens"]
    )

    address_overlap = (
        s1["address_tokens"]
        & target["address_tokens"]
    )

    name_j = jaccard(
        s1["name_tokens"],
        target["name_tokens"],
    )

    address_j = jaccard(
        s1["address_tokens"],
        target["address_tokens"],
    )

    name_numbers = (
        s1["name_numbers"]
        & target["name_numbers"]
    )

    address_numbers = (
        s1["address_numbers"]
        & target["address_numbers"]
    )

    same_country = (
        bool(s1["country"])
        and bool(target["country"])
        and s1["country"] == target["country"]
    )

    same_postal = (
        bool(s1["postal"])
        and bool(target["postal"])
        and s1["postal"] == target["postal"]
    )

    name_prefix = (
        len(s1["name"]) >= 4
        and len(target["name"]) >= 4
        and s1["name"][:4]
        == target["name"][:4]
    )

    translit_prefix = (
        len(s1["translit"]) >= 4
        and len(target["translit"]) >= 4
        and s1["translit"][:4]
        == target["translit"][:4]
    )

    return {

        "exact_name": name_exact,

        "exact_core": core_exact,

        "exact_translit": translit_exact,

        "exact_address": address_exact,

        "name_token_overlap": bool(
            name_overlap
        ),

        "name_jaccard_ge_0.5": (
            name_j >= 0.5
        ),

        "name_jaccard_ge_0.75": (
            name_j >= 0.75
        ),

        "address_token_overlap": bool(
            address_overlap
        ),

        "address_jaccard_ge_0.3": (
            address_j >= 0.3
        ),

        "address_jaccard_ge_0.5": (
            address_j >= 0.5
        ),

        "address_jaccard_ge_0.75": (
            address_j >= 0.75
        ),

        "name_number_overlap": bool(
            name_numbers
        ),

        "address_number_overlap": bool(
            address_numbers
        ),

        "same_postal": same_postal,

        "same_country": same_country,

        "name_prefix_4": name_prefix,

        "translit_prefix_4": translit_prefix,

        "name_jaccard": name_j,

        "address_jaccard": address_j,

        "name_token_count_s1": len(
            s1["name_tokens"]
        ),

        "name_token_count_target": len(
            target["name_tokens"]
        ),

        "address_token_count_s1": len(
            s1["address_tokens"]
        ),

        "address_token_count_target": len(
            target["address_tokens"]
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRIAL 2 — BLOCKING DIAGNOSTIC v1")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Get EXACT SAME 5K S1 SAMPLE.
    # --------------------------------------------------------

    print("\nSelecting controlled S1 subset...")

    s1 = get_selected_s1()

    selected_s1_ids = set(
        s1["entity_id"]
    )

    print(
        f"S1 sample: "
        f"{len(s1):,}"
    )

    # --------------------------------------------------------
    # 2. Load ground truth.
    # --------------------------------------------------------

    print("\nLoading ground truth...")

    truth_all = load_ground_truth()

    truth = {
        s1_id: truth_all.get(
            s1_id,
            set(),
        )
        for s1_id in selected_s1_ids
    }

    total_true_pairs = sum(
        len(ids)
        for ids in truth.values()
    )

    print(
        f"True pairs: "
        f"{total_true_pairs:,}"
    )

    # --------------------------------------------------------
    # 3. Load ONLY candidate IDs.
    # --------------------------------------------------------

    print("\nLoading candidate pairs...")

    candidates = pd.read_csv(
        CANDIDATES,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "s1_entity_id",
            "candidate_entity_id",
        ],
    )

    candidate_pairs = set(
        zip(
            candidates["s1_entity_id"],
            candidates["candidate_entity_id"],
        )
    )

    print(
        f"Candidate pairs: "
        f"{len(candidate_pairs):,}"
    )

    del candidates

    # --------------------------------------------------------
    # 4. Identify MISSED true pairs.
    # --------------------------------------------------------

    missed_pairs = []

    for s1_id, target_ids in truth.items():

        for target_id in target_ids:

            if (
                s1_id,
                target_id,
            ) not in candidate_pairs:

                missed_pairs.append(
                    (
                        s1_id,
                        target_id,
                    )
                )

    del candidate_pairs

    print(
        f"\nMissed true pairs: "
        f"{len(missed_pairs):,}"
    )

    if not missed_pairs:

        print(
            "\nBlocking recall = 100%."
        )

        return

    # --------------------------------------------------------
    # 5. We ONLY need the target IDs from missed pairs.
    # --------------------------------------------------------

    missed_target_ids = {
        target_id
        for _, target_id in missed_pairs
    }

    missed_s1_ids = {
        s1_id
        for s1_id, _ in missed_pairs
    }

    print(
        f"Unique missed target IDs: "
        f"{len(missed_target_ids):,}"
    )

    print(
        f"Unique affected S1 IDs: "
        f"{len(missed_s1_ids):,}"
    )

    # --------------------------------------------------------
    # 6. Build records ONLY for those S1s.
    # --------------------------------------------------------

    s1_records = build_s1_records(
        s1[
            s1["entity_id"].isin(
                missed_s1_ids
            )
        ]
    )

    # --------------------------------------------------------
    # 7. STREAM S2/S3.
    #
    # We do NOT load 10.3 million records into memory.
    # We scan them in chunks and retain only target IDs
    # belonging to the 5,586 missed truth pairs.
    # --------------------------------------------------------

    target_records = load_required_targets(
        missed_target_ids
    )

    print(
        f"\nRequired target records loaded: "
        f"{len(target_records):,}"
    )

    # --------------------------------------------------------
    # 8. Analyze missed pairs.
    # --------------------------------------------------------

    rows = []

    print(
        "\nAnalyzing missed pairs..."
    )

    for i, (
        s1_id,
        target_id,
    ) in enumerate(
        missed_pairs,
        start=1,
    ):

        s1_record = s1_records.get(
            s1_id
        )

        target_record = target_records.get(
            target_id
        )

        if (
            s1_record is None
            or target_record is None
        ):

            continue

        signals = analyze_pair(
            s1_record,
            target_record,
        )

        rows.append(
            {
                "s1_entity_id": s1_id,
                "target_entity_id": target_id,
                **signals,
            }
        )

        if i % 1000 == 0:

            print(
                f"Analyzed "
                f"{i:,}/"
                f"{len(missed_pairs):,}"
            )

    df = pd.DataFrame(rows)

    # ========================================================
    # SIGNAL COVERAGE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MISSED TRUE-PAIR SIGNAL COVERAGE")
    print("=" * 70)

    print(
        f"\nAnalyzed missed pairs: "
        f"{len(df):,}"
    )

    signals = [
        "exact_name",
        "exact_core",
        "exact_translit",
        "exact_address",
        "name_token_overlap",
        "name_jaccard_ge_0.5",
        "name_jaccard_ge_0.75",
        "address_token_overlap",
        "address_jaccard_ge_0.3",
        "address_jaccard_ge_0.5",
        "address_jaccard_ge_0.75",
        "name_number_overlap",
        "address_number_overlap",
        "same_postal",
        "same_country",
        "name_prefix_4",
        "translit_prefix_4",
    ]

    print(
        f"\n{'Signal':<35}"
        f"{'Count':>10}"
        f"{'Coverage':>12}"
    )

    print("-" * 60)

    for signal in signals:

        count = int(
            df[signal].sum()
        )

        coverage = (
            count / len(df)
            if len(df)
            else 0
        )

        print(
            f"{signal:<35}"
            f"{count:>10,}"
            f"{coverage:>11.2%}"
        )

    # ========================================================
    # POTENTIAL RESCUE ROUTES
    # ========================================================

    print("\n")
    print("=" * 70)
    print("POTENTIAL RESCUE ROUTES")
    print("=" * 70)

    rescue = {

        "Exact name/core/translit/address":
            (
                df["exact_name"]
                | df["exact_core"]
                | df["exact_translit"]
                | df["exact_address"]
            ),

        "Name token overlap":
            df["name_token_overlap"],

        "Name Jaccard >= 0.50":
            df["name_jaccard_ge_0.5"],

        "Name Jaccard >= 0.75":
            df["name_jaccard_ge_0.75"],

        "Address token overlap":
            df["address_token_overlap"],

        "Address Jaccard >= 0.30":
            df["address_jaccard_ge_0.3"],

        "Address Jaccard >= 0.50":
            df["address_jaccard_ge_0.5"],

        "Number overlap":
            (
                df["name_number_overlap"]
                | df["address_number_overlap"]
            ),

        "Postal match":
            df["same_postal"],

        "Name prefix":
            (
                df["name_prefix_4"]
                | df["translit_prefix_4"]
            ),

        "Country + name token":
            (
                df["same_country"]
                & df["name_token_overlap"]
            ),

        "Country + address token":
            (
                df["same_country"]
                & df["address_token_overlap"]
            ),
    }

    print(
        f"\n{'Route':<40}"
        f"{'Recoverable':>15}"
        f"{'Coverage':>12}"
    )

    print("-" * 70)

    for route, mask in rescue.items():

        count = int(mask.sum())

        coverage = (
            count / len(df)
            if len(df)
            else 0
        )

        print(
            f"{route:<40}"
            f"{count:>15,}"
            f"{coverage:>11.2%}"
        )

    # ========================================================
    # EXAMPLES OF MISSED PAIRS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EXAMPLES OF MISSED PAIRS")
    print("=" * 70)

    # Reload original processed values for readable examples.
    # Only the affected S1 IDs and target IDs are used.

    example_count = min(
        20,
        len(df),
    )

    for _, row in df.head(
        example_count
    ).iterrows():

        s1_id = row["s1_entity_id"]
        target_id = row["target_entity_id"]

        s1_record = s1_records[s1_id]
        target_record = target_records[target_id]

        print("\n" + "-" * 70)

        print(
            f"S1:     {s1_id}"
        )

        print(
            f"Target: {target_id}"
        )

        print(
            f"S1 name:     {s1_record['name']}"
        )

        print(
            f"Target name: {target_record['name']}"
        )

        print(
            f"S1 address:     {s1_record['address']}"
        )

        print(
            f"Target address: {target_record['address']}"
        )

        print(
            f"S1 country:     {s1_record['country']}"
        )

        print(
            f"Target country: {target_record['country']}"
        )

        print(
            f"Name Jaccard: "
            f"{row['name_jaccard']:.3f}"
        )

        print(
            f"Address Jaccard: "
            f"{row['address_jaccard']:.3f}"
        )

        print(
            f"Name numbers: "
            f"{s1_record['name_numbers']}"
            f" vs "
            f"{target_record['name_numbers']}"
        )

        print(
            f"Address numbers: "
            f"{s1_record['address_numbers']}"
            f" vs "
            f"{target_record['address_numbers']}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        ROOT
        / "diagnostics"
        / "trial2_v1_missed_pairs.tsv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nSaved detailed diagnostic:"
    )

    print(output_path)


if __name__ == "__main__":
    main()

