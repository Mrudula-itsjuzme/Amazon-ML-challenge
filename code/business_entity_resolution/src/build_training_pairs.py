"""
Build training candidate pairs for the Amazon ML Business Entity Resolution task.

Development pipeline:

    S1 sample
       |
       +-- deterministic blockers
       |
       +-- name_core rare-token blocker
       |
       +-- name_translit exact blocker
       |
       +-- name_translit rare-token blocker
       |
       v
    candidate union
       |
       v
    ground-truth labels
       |
       v
    dev_candidates_5000_translit.parquet

The script deliberately keeps candidate generation generous.
The candidate set is the ceiling for downstream matching recall.
"""

from pathlib import Path
import argparse
import time

import duckdb


# ============================================================
# PATHS
# ============================================================

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = REPO_ROOT / "student_resource" / "dataset" / "train"
if not (TRAIN_DIR / "train_ground_truth.tsv").is_file():
    TRAIN_DIR = WORKSPACE_ROOT / "student_resource" / "dataset" / "train"
PROCESSED_DIR = REPO_ROOT / "processed"

S1_PATH = PROCESSED_DIR / "train_source1.parquet"
S2_PATH = PROCESSED_DIR / "train_source2.parquet"
S3_PATH = PROCESSED_DIR / "train_source3.parquet"

GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

DUCKDB_TMP = Path("/tmp/duckdb_tmp")
DUCKDB_TMP.mkdir(parents=True, exist_ok=True)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Build development candidate pairs"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="Number of S1 records to use"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed"
    )

    parser.add_argument(
        "--max-token-doc-freq",
        type=int,
        default=500,
        help="Maximum candidate-side document frequency for a token"
    )

    parser.add_argument(
        "--rare-tokens-per-entity",
        type=int,
        default=2,
        help="Number of rarest tokens used per entity"
    )

    parser.add_argument(
        "--max-block-size",
        type=int,
        default=300,
        help="Maximum candidate records allowed in an exact blocking bucket"
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output parquet path"
    )

    return parser.parse_args()


# ============================================================
# DUCKDB
# ============================================================

def connect_duckdb():

    con = duckdb.connect()

    con.execute(
        f"SET temp_directory = '{DUCKDB_TMP}'"
    )

    con.execute("SET threads = 4")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET max_temp_directory_size = '100GB'")

    return con


# ============================================================
# LOAD DATA
# ============================================================

def load_data(con, sample, seed):

    print("\nLoading processed parquet files...")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW s1_full AS

        SELECT
            entity_id,
            name_norm,
            name_core,
            name_translit,
            address_norm,
            postal_code,
            numeric_tokens,
            country_norm

        FROM read_parquet('{S1_PATH}')
        """
    )

    # Explicit reservoir + seed so repeated development runs use
    # the same S1 sample.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE s1 AS

        SELECT *

        FROM s1_full

        USING SAMPLE {sample} ROWS
        (reservoir, {seed})
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW candidate_source AS

        SELECT
            entity_id,
            name_norm,
            name_core,
            name_translit,
            address_norm,
            postal_code,
            numeric_tokens,
            country_norm

        FROM read_parquet('{S2_PATH}')

        UNION ALL

        SELECT
            entity_id,
            name_norm,
            name_core,
            name_translit,
            address_norm,
            postal_code,
            numeric_tokens,
            country_norm

        FROM read_parquet('{S3_PATH}')
        """
    )

    s1_count = con.execute(
        "SELECT COUNT(*) FROM s1"
    ).fetchone()[0]

    candidate_count = con.execute(
        "SELECT COUNT(*) FROM candidate_source"
    ).fetchone()[0]

    print(f"S1 development sample : {s1_count:,}")
    print(f"Candidate pool         : {candidate_count:,}")

    return s1_count, candidate_count


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(con):

    print("\nLoading ground truth...")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW gt AS

        SELECT *

        FROM read_csv(
            '{GT_PATH}',
            delim='\\t',
            header=true,
            quote='"',
            escape='"'
        )
        """
    )

    # The challenge GT contains matched_entity_ids as a
    # Python-list-like string, e.g.
    #
    # ['S2-123', 'S3-456']
    #
    # Remove brackets/quotes first, then split on commas.

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE truth_pairs AS

        SELECT

            CAST(source1_entity_id AS VARCHAR)
                AS source1_entity_id,

            TRIM(matched_id)
                AS candidate_entity_id

        FROM gt

        CROSS JOIN UNNEST(

            STRING_SPLIT(

                REPLACE(
                    REPLACE(
                        REPLACE(
                            COALESCE(
                                matched_entity_ids,
                                ''
                            ),
                            '[',
                            ''
                        ),
                        ']',
                        ''
                    ),
                    '''',
                    ''
                ),

                ','
            )

        ) AS t(matched_id)

        WHERE TRIM(matched_id) <> ''

          AND CAST(source1_entity_id AS VARCHAR)
              IN (
                  SELECT CAST(entity_id AS VARCHAR)
                  FROM s1
              )
        """
    )

    truth_count = con.execute(
        "SELECT COUNT(*) FROM truth_pairs"
    ).fetchone()[0]

    print(
        f"Ground-truth pairs in sample : {truth_count:,}"
    )

    return truth_count


# ============================================================
# EXACT BLOCKER HELPER
# ============================================================

def build_purged_key_blocker(
    con,
    block_name,
    s1_expression,
    candidate_expression,
    max_block_size
):

    print(f"\nBuilding blocker: {block_name}")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW
        s1_keyed_{block_name} AS

        SELECT

            CAST(entity_id AS VARCHAR)
                AS source1_entity_id,

            ({s1_expression})
                AS block_key

        FROM s1

        WHERE ({s1_expression}) IS NOT NULL

          AND ({s1_expression}) <> ''
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW
        candidate_keyed_{block_name} AS

        SELECT

            CAST(entity_id AS VARCHAR)
                AS candidate_entity_id,

            ({candidate_expression})
                AS block_key

        FROM candidate_source

        WHERE ({candidate_expression}) IS NOT NULL

          AND ({candidate_expression}) <> ''
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE
        good_keys_{block_name} AS

        SELECT block_key

        FROM (

            SELECT
                block_key,
                COUNT(*) AS freq

            FROM candidate_keyed_{block_name}

            GROUP BY block_key
        )

        WHERE freq <= {max_block_size}
        """
    )

    dropped = con.execute(
        f"""
        SELECT COUNT(*)

        FROM (

            SELECT
                block_key,
                COUNT(*) AS freq

            FROM candidate_keyed_{block_name}

            GROUP BY block_key

        )

        WHERE freq > {max_block_size}
        """
    ).fetchone()[0]

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE
        pairs_{block_name} AS

        SELECT DISTINCT

            s.source1_entity_id,
            c.candidate_entity_id

        FROM s1_keyed_{block_name} s

        INNER JOIN good_keys_{block_name} g

            ON g.block_key = s.block_key

        INNER JOIN candidate_keyed_{block_name} c

            ON c.block_key = s.block_key
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM pairs_{block_name}
        """
    ).fetchone()[0]

    print(
        f"  pairs={count:,}"
        f" | oversized keys purged={dropped:,}"
    )

    return count


# ============================================================
# CHANNEL 1
# DETERMINISTIC
# ============================================================

def build_deterministic_pairs(
    con,
    max_block_size
):

    print("\n" + "=" * 70)
    print("CHANNEL 1: DETERMINISTIC")
    print("=" * 70)

    blockers = {

        "exact_name_country": (
            "name_norm || '||' || country_norm",
            "name_norm || '||' || country_norm"
        ),

        "exact_core_name_country": (
            "name_core || '||' || country_norm",
            "name_core || '||' || country_norm"
        ),

        "exact_address": (
            "address_norm",
            "address_norm"
        ),

        "postal_code": (
            "postal_code",
            "postal_code"
        ),

        "house_number_country": (
            "split_part(numeric_tokens, ' ', 1)"
            " || '||' || country_norm",

            "split_part(numeric_tokens, ' ', 1)"
            " || '||' || country_norm"
        ),
    }

    tables = []

    for name, (
        s1_expression,
        candidate_expression
    ) in blockers.items():

        build_purged_key_blocker(
            con,
            name,
            s1_expression,
            candidate_expression,
            max_block_size
        )

        tables.append(f"pairs_{name}")

    union_sql = "\nUNION\n".join(
        f"""
        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM {table}
        """
        for table in tables
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE det_pairs AS

        {union_sql}
        """
    )

    count = con.execute(
        "SELECT COUNT(*) FROM det_pairs"
    ).fetchone()[0]

    print(
        f"\nDeterministic union : {count:,}"
    )

    return count


# ============================================================
# CHANNEL 2
# NAME_CORE TOKEN BLOCKING
# ============================================================

def build_token_pairs(
    con,
    max_token_doc_freq,
    rare_tokens_per_entity
):

    print("\n" + "=" * 70)
    print("CHANNEL 2: NAME_CORE TOKEN OVERLAP")
    print("=" * 70)

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW s1_tokens_all AS

        SELECT

            CAST(entity_id AS VARCHAR)
                AS source1_entity_id,

            country_norm,

            UNNEST(
                string_split(name_core, ' ')
            ) AS token

        FROM s1

        WHERE name_core <> ''
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW candidate_tokens_all AS

        SELECT

            CAST(entity_id AS VARCHAR)
                AS candidate_entity_id,

            country_norm,

            UNNEST(
                string_split(name_core, ' ')
            ) AS token

        FROM candidate_source

        WHERE name_core <> ''
        """
    )

    print(
        "Computing candidate-side token frequencies..."
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE token_doc_freq AS

        SELECT

            token,

            COUNT(
                DISTINCT candidate_entity_id
            ) AS doc_freq

        FROM candidate_tokens_all

        WHERE LENGTH(token) >= 2

        GROUP BY token
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s1_rare_tokens AS

        SELECT

            source1_entity_id,
            country_norm,
            token

        FROM (

            SELECT

                s.source1_entity_id,
                s.country_norm,
                s.token,

                ROW_NUMBER() OVER (

                    PARTITION BY
                        s.source1_entity_id

                    ORDER BY
                        t.doc_freq ASC,
                        s.token
                ) AS rn

            FROM s1_tokens_all s

            INNER JOIN token_doc_freq t

                ON t.token = s.token

            WHERE t.doc_freq <=
                  {max_token_doc_freq}
        )

        WHERE rn <= {rare_tokens_per_entity}
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE
        candidate_rare_tokens AS

        SELECT

            candidate_entity_id,
            country_norm,
            token

        FROM (

            SELECT

                c.candidate_entity_id,
                c.country_norm,
                c.token,

                ROW_NUMBER() OVER (

                    PARTITION BY
                        c.candidate_entity_id

                    ORDER BY
                        t.doc_freq ASC,
                        c.token
                ) AS rn

            FROM candidate_tokens_all c

            INNER JOIN token_doc_freq t

                ON t.token = c.token

            WHERE t.doc_freq <=
                  {max_token_doc_freq}
        )

        WHERE rn <= {rare_tokens_per_entity}
        """
    )

    s1_keys = con.execute(
        "SELECT COUNT(*) FROM s1_rare_tokens"
    ).fetchone()[0]

    candidate_keys = con.execute(
        "SELECT COUNT(*) FROM candidate_rare_tokens"
    ).fetchone()[0]

    print(
        f"S1 rare token keys        : {s1_keys:,}"
    )

    print(
        f"Candidate rare token keys : {candidate_keys:,}"
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE token_pairs AS

        SELECT DISTINCT

            s.source1_entity_id,
            c.candidate_entity_id

        FROM s1_rare_tokens s

        INNER JOIN candidate_rare_tokens c

            ON c.token = s.token

           AND c.country_norm = s.country_norm
        """
    )

    count = con.execute(
        "SELECT COUNT(*) FROM token_pairs"
    ).fetchone()[0]

    print(
        f"Token-overlap pairs : {count:,}"
    )

    return count


# ============================================================
# CHANNEL 3A
# EXACT TRANSLITERATION
# ============================================================

def build_translit_exact_pairs(
    con,
    max_block_size
):

    print("\n" + "=" * 70)
    print("CHANNEL 3A: EXACT TRANSLITERATED NAME")
    print("=" * 70)

    build_purged_key_blocker(
        con,

        "exact_translit_name_country",

        "name_translit || '||' || country_norm",

        "name_translit || '||' || country_norm",

        max_block_size
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE
        translit_exact_pairs AS

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM pairs_exact_translit_name_country
        """
    )

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM translit_exact_pairs
        """
    ).fetchone()[0]

    print(
        f"Exact transliteration pairs : {count:,}"
    )

    return count


# ============================================================
# CHANNEL 3B
# TRANSLITERATED TOKEN BLOCKING
# ============================================================

def build_translit_token_pairs(
    con,
    max_token_doc_freq,
    rare_tokens_per_entity
):

    print("\n" + "=" * 70)
    print("CHANNEL 3B: TRANSLITERATED TOKEN OVERLAP")
    print("=" * 70)

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW
        s1_translit_tokens_all AS

        SELECT

            CAST(entity_id AS VARCHAR)
                AS source1_entity_id,

            country_norm,

            UNNEST(
                string_split(name_translit, ' ')
            ) AS token

        FROM s1

        WHERE name_translit <> ''
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW
        candidate_translit_tokens_all AS

        SELECT

            CAST(entity_id AS VARCHAR)
                AS candidate_entity_id,

            country_norm,

            UNNEST(
                string_split(name_translit, ' ')
            ) AS token

        FROM candidate_source

        WHERE name_translit <> ''
        """
    )

    print(
        "Computing transliteration token frequencies..."
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE
        translit_token_doc_freq AS

        SELECT

            token,

            COUNT(
                DISTINCT candidate_entity_id
            ) AS doc_freq

        FROM candidate_translit_tokens_all

        WHERE LENGTH(token) >= 2

        GROUP BY token
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE
        s1_rare_translit_tokens AS

        SELECT

            source1_entity_id,
            country_norm,
            token

        FROM (

            SELECT

                s.source1_entity_id,
                s.country_norm,
                s.token,

                ROW_NUMBER() OVER (

                    PARTITION BY
                        s.source1_entity_id

                    ORDER BY
                        t.doc_freq ASC,
                        s.token
                ) AS rn

            FROM s1_translit_tokens_all s

            INNER JOIN translit_token_doc_freq t

                ON t.token = s.token

            WHERE t.doc_freq <=
                  {max_token_doc_freq}
        )

        WHERE rn <= {rare_tokens_per_entity}
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE
        candidate_rare_translit_tokens AS

        SELECT

            candidate_entity_id,
            country_norm,
            token

        FROM (

            SELECT

                c.candidate_entity_id,
                c.country_norm,
                c.token,

                ROW_NUMBER() OVER (

                    PARTITION BY
                        c.candidate_entity_id

                    ORDER BY
                        t.doc_freq ASC,
                        c.token
                ) AS rn

            FROM candidate_translit_tokens_all c

            INNER JOIN translit_token_doc_freq t

                ON t.token = c.token

            WHERE t.doc_freq <=
                  {max_token_doc_freq}
        )

        WHERE rn <= {rare_tokens_per_entity}
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE
        translit_token_pairs AS

        SELECT DISTINCT

            s.source1_entity_id,
            c.candidate_entity_id

        FROM s1_rare_translit_tokens s

        INNER JOIN candidate_rare_translit_tokens c

            ON c.token = s.token

           AND c.country_norm = s.country_norm
        """
    )

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM translit_token_pairs
        """
    ).fetchone()[0]

    print(
        f"Transliterated-token pairs : {count:,}"
    )

    return count


# ============================================================
# COMBINE
# ============================================================

def build_candidate_unions(con):

    # --------------------------------------------------------
    # Original baseline
    # --------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE
        baseline_candidates AS

        SELECT DISTINCT

            source1_entity_id,
            candidate_entity_id

        FROM (

            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM det_pairs

            UNION ALL

            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM token_pairs
        )
        """
    )

    baseline_count = con.execute(
        """
        SELECT COUNT(*)
        FROM baseline_candidates
        """
    ).fetchone()[0]

    # --------------------------------------------------------
    # New transliteration channel
    # --------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE
        translit_candidates AS

        SELECT DISTINCT

            source1_entity_id,
            candidate_entity_id

        FROM (

            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM translit_exact_pairs

            UNION ALL

            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM translit_token_pairs
        )
        """
    )

    translit_count = con.execute(
        """
        SELECT COUNT(*)
        FROM translit_candidates
        """
    ).fetchone()[0]

    # --------------------------------------------------------
    # Final union
    # --------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE
        dev_candidates AS

        SELECT DISTINCT

            source1_entity_id,
            candidate_entity_id

        FROM (

            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM baseline_candidates

            UNION ALL

            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM translit_candidates
        )
        """
    )

    final_count = con.execute(
        """
        SELECT COUNT(*)
        FROM dev_candidates
        """
    ).fetchone()[0]

    print("\n" + "=" * 70)
    print("CANDIDATE UNION")
    print("=" * 70)

    print(
        f"Baseline D+Token      : {baseline_count:,}"
    )

    print(
        f"Translit channel      : {translit_count:,}"
    )

    print(
        f"Final D+Token+Translit: {final_count:,}"
    )

    return baseline_count, translit_count, final_count


# ============================================================
# CANDIDATE RECALL
# ============================================================

def evaluate_candidates(
    con,
    total_s1,
    truth_count
):

    print("\n" + "=" * 70)
    print("CANDIDATE RECALL")
    print("=" * 70)

    results = {}

    for name, table in [
        ("Baseline D+Token", "baseline_candidates"),
        ("Final +Translit", "dev_candidates"),
    ]:

        recovered = con.execute(
            f"""
            SELECT COUNT(*)

            FROM truth_pairs t

            INNER JOIN {table} c

                ON c.source1_entity_id =
                   t.source1_entity_id

               AND c.candidate_entity_id =
                   t.candidate_entity_id
            """
        ).fetchone()[0]

        coverage = con.execute(
            f"""
            SELECT COUNT(DISTINCT source1_entity_id)

            FROM {table}
            """
        ).fetchone()[0]

        pair_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {table}
            """
        ).fetchone()[0]

        incremental = None

        if table == "dev_candidates":

            incremental = con.execute(
                """
                SELECT COUNT(*)

                FROM truth_pairs t

                INNER JOIN dev_candidates d

                    ON d.source1_entity_id =
                       t.source1_entity_id

                   AND d.candidate_entity_id =
                       t.candidate_entity_id

                LEFT JOIN baseline_candidates b

                    ON b.source1_entity_id =
                       t.source1_entity_id

                   AND b.candidate_entity_id =
                       t.candidate_entity_id

                WHERE b.source1_entity_id IS NULL
                """
            ).fetchone()[0]

        recall = (
            recovered / truth_count
            if truth_count > 0
            else 0.0
        )

        s1_coverage = (
            coverage / total_s1
            if total_s1 > 0
            else 0.0
        )

        avg_candidates = (
            pair_count / total_s1
            if total_s1 > 0
            else 0.0
        )

        results[name] = {
            "pairs": pair_count,
            "recovered": recovered,
            "recall": recall,
            "coverage": s1_coverage,
            "avg": avg_candidates,
            "incremental": incremental,
        }

        print(
            f"\n{name}"
        )

        print(
            f"  Candidate pairs : {pair_count:,}"
        )

        print(
            f"  GT recovered    : {recovered:,}/{truth_count:,}"
        )

        print(
            f"  Pair recall     : {recall:.2%}"
        )

        print(
            f"  S1 coverage     : {s1_coverage:.2%}"
        )

        print(
            f"  Avg candidates  : {avg_candidates:.2f}"
        )

        if incremental is not None:

            print(
                f"  NEW GT pairs from translit : "
                f"{incremental:,}"
            )

    return results


# ============================================================
# LABEL CANDIDATES
# ============================================================

def label_candidates(con):

    print("\nLabeling candidate pairs...")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE
        labeled_candidates AS

        SELECT

            c.source1_entity_id,
            c.candidate_entity_id,

            CASE
                WHEN t.candidate_entity_id IS NOT NULL
                THEN 1
                ELSE 0
            END AS label

        FROM dev_candidates c

        LEFT JOIN truth_pairs t

            ON t.source1_entity_id =
               c.source1_entity_id

           AND t.candidate_entity_id =
               c.candidate_entity_id
        """
    )

    total = con.execute(
        """
        SELECT COUNT(*)
        FROM labeled_candidates
        """
    ).fetchone()[0]

    positives = con.execute(
        """
        SELECT COUNT(*)
        FROM labeled_candidates
        WHERE label = 1
        """
    ).fetchone()[0]

    negatives = total - positives

    positive_rate = (
        positives / total
        if total > 0
        else 0.0
    )

    print(
        f"Total candidate pairs : {total:,}"
    )

    print(
        f"Positive pairs        : {positives:,}"
    )

    print(
        f"Negative pairs        : {negatives:,}"
    )

    print(
        f"Positive rate         : {positive_rate:.4%}"
    )

    return total, positives, negatives


# ============================================================
# SAVE
# ============================================================

def save_candidates(con, output_path):

    print("\nSaving candidate dataset...")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    con.execute(
        f"""
        COPY (

            SELECT

                source1_entity_id,
                candidate_entity_id,
                label

            FROM labeled_candidates

        )

        TO '{output_path}'

        (FORMAT PARQUET)
        """
    )

    print(
        f"Saved -> {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    args = parse_args()

    if args.out:

        output_path = Path(args.out)

    else:

        output_path = (
            PROCESSED_DIR /
            f"dev_candidates_{args.sample}_translit.parquet"
        )

    con = connect_duckdb()

    total_s1, total_candidates = load_data(
        con,
        args.sample,
        args.seed
    )

    truth_count = load_ground_truth(con)

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    build_deterministic_pairs(
        con,
        args.max_block_size
    )

    build_token_pairs(
        con,
        args.max_token_doc_freq,
        args.rare_tokens_per_entity
    )

    # --------------------------------------------------------
    # NEW TRANSLITERATION CHANNEL
    # --------------------------------------------------------

    build_translit_exact_pairs(
        con,
        args.max_block_size
    )

    build_translit_token_pairs(
        con,
        args.max_token_doc_freq,
        args.rare_tokens_per_entity
    )

    # --------------------------------------------------------
    # UNION
    # --------------------------------------------------------

    build_candidate_unions(con)

    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    evaluate_candidates(
        con,
        total_s1,
        truth_count
    )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    label_candidates(con)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_candidates(
        con,
        output_path
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Runtime : {elapsed:.1f} seconds"
    )

    con.close()


if __name__ == "__main__":
    main()