
from pathlib import Path
import argparse
import time
import numpy as np
import duckdb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PATHS (same convention as preprocess.py)
# ============================================================

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = REPO_ROOT / "student_resource" / "dataset" / "train"
PROCESSED_DIR = REPO_ROOT / "processed"

DUCKDB_TMP = Path("/tmp/duckdb_tmp")
DUCKDB_TMP.mkdir(exist_ok=True, parents=True)

S1_PATH = PROCESSED_DIR / "train_source1.parquet"
S2_PATH = PROCESSED_DIR / "train_source2.parquet"
S3_PATH = PROCESSED_DIR / "train_source3.parquet"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
if not GT_PATH.is_file():
    GT_PATH = WORKSPACE_ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"


# ============================================================
# CLI
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-channel candidate generation")

    parser.add_argument(
        "--channels", type=str, default="deterministic,token,ngram",
        help="Comma-separated subset of: deterministic,token,ngram",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Limit to N randomly-sampled S1 entities, for quick testing",
    )
    parser.add_argument(
        "--max-token-doc-freq", type=int, default=500,
        help="Secondary safety net: tokens appearing in more than this many "
             "candidate records are dropped entirely, on top of the "
             "rarest-token restriction below.",
    )
    parser.add_argument(
        "--rare-tokens-per-entity", type=int, default=2,
        help="PRIMARY safety net for the token_overlap channel: each record "
             "contributes only its N rarest tokens (by candidate-side "
             "document frequency) as blocking keys, instead of every token "
             "it has. Frequency-capping alone (--max-token-doc-freq) bounds "
             "individual keys but not the join itself - fanout is "
             "s1_side_freq x candidate_side_freq summed across every "
             "surviving token, which can still be hundreds of millions of "
             "rows at your scale. Restricting to each record's rarest N "
             "tokens bounds the join to a small multiple of record count "
             "instead, regardless of corpus size.",
    )
    parser.add_argument(
        "--max-block-size", type=int, default=300,
        help="Block purging cap for the deterministic channel: any exact-key "
             "value (name, address, postal code, house number...) shared by "
             "more than this many candidate-side records is dropped from "
             "blocking. This is what your postal_code / numeric_tokens blow-up "
             "was missing - a few oversized blocks (a handful of postal codes "
             "shared by thousands of records) otherwise dominate the whole "
             "join. Lower this if a run still explodes; raise it if recall "
             "looks capped by dropped blocks.",
    )
    parser.add_argument(
        "--ngram-topk", type=int, default=15,
        help="Max nearest neighbours per S1 entity in the ngram channel",
    )
    parser.add_argument(
        "--ngram-min-similarity", type=float, default=0.35,
        help="Minimum cosine similarity to keep an ngram-channel candidate",
    )
    parser.add_argument(
        "--ngram-batch-size", type=int, default=4000,
        help="S1 rows processed per batch within each country partition "
             "(controls peak memory during the ngram channel)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Where to save the combined candidate-pairs parquet "
             "(default: PROCESSED_DIR/candidate_pairs_v2_debug.parquet)",
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="Materialize the final combined UNION table. This can exhaust temp disk on full runs, "
             "so it is disabled by default. Use it only when you explicitly need the merged output.",
    )

    return parser.parse_args()


# ============================================================
# SETUP
# ============================================================

def connect_duckdb():
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{DUCKDB_TMP}'")
    con.execute("SET threads = 4")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET max_temp_directory_size = '100GB'")
    return con


def load_views(con, sample):
    print("\nLoading processed data...")

    con.execute(f"""
        CREATE OR REPLACE VIEW s1_full AS
        SELECT entity_id, name_norm, name_core, name_translit,
               address_norm, postal_code, numeric_tokens, country_norm
        FROM read_parquet('{S1_PATH}')
    """)

    if sample:
        con.execute(f"""
            CREATE OR REPLACE TABLE s1 AS
            SELECT * FROM s1_full USING SAMPLE {sample} ROWS
        """)
        print(f"Sampling {sample:,} S1 entities for this run.")
    else:
        con.execute("CREATE OR REPLACE VIEW s1 AS SELECT * FROM s1_full")

    con.execute(f"""
        CREATE OR REPLACE VIEW candidate_source AS
        SELECT entity_id, name_norm, name_core, name_translit,
               address_norm, postal_code, numeric_tokens, country_norm
        FROM read_parquet('{S2_PATH}')
        UNION ALL
        SELECT entity_id, name_norm, name_core, name_translit,
               address_norm, postal_code, numeric_tokens, country_norm
        FROM read_parquet('{S3_PATH}')
    """)

    total_s1 = con.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
    total_candidates = con.execute("SELECT COUNT(*) FROM candidate_source").fetchone()[0]

    print(f"S1 (this run): {total_s1:,}")
    print(f"Candidate pool (S2+S3): {total_candidates:,}")

    return total_s1, total_candidates


def load_ground_truth(con):
    print("\nLoading ground truth (restricted to this run's S1 sample)...")

    con.execute(f"""
        CREATE OR REPLACE VIEW gt AS
        SELECT * FROM read_csv('{GT_PATH}', delim='\\t', header=true,
                                quote='"', escape='"')
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE truth_pairs AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
            TRIM(matched_id) AS candidate_entity_id
        FROM gt
        CROSS JOIN UNNEST(
            STRING_SPLIT(
                REPLACE(
                    REPLACE(
                        REPLACE(
                            COALESCE(matched_entity_ids, ''),
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
        AND CAST(source1_entity_id AS VARCHAR) IN (SELECT entity_id FROM s1)
    """)

    truth_count = con.execute("SELECT COUNT(*) FROM truth_pairs").fetchone()[0]
    print(f"Ground-truth pairs in scope: {truth_count:,}")

    return truth_count


# ============================================================
# CHANNEL 1: DETERMINISTIC KEYS
# ============================================================

def build_purged_key_blocker(con, sub_name, s1_key_expr, cand_key_expr, max_block_size):
    """
    Build (source1_entity_id, candidate_entity_id) pairs for one exact-key
    blocker, with block purging: any key value whose candidate-side frequency
    exceeds max_block_size is dropped before the join ever happens.

    This is the fix for the postal_code / numeric_tokens blow-up: a handful of
    oversized blocks (a postal code shared by thousands of records, in dense
    urban zips this is realistic, not just a synthetic-data artifact) can
    dominate total candidate volume even though every OTHER key is fine.
    Purging them costs a small amount of recall on the (rare) true pairs that
    only share an oversized key - almost always an acceptable trade since
    those keys were mostly generating false candidates anyway.
    """

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s1_keyed_{sub_name} AS
        SELECT entity_id AS s1_entity_id, ({s1_key_expr}) AS block_key
        FROM s1
        WHERE ({s1_key_expr}) <> ''
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW cand_keyed_{sub_name} AS
        SELECT entity_id AS candidate_entity_id, ({cand_key_expr}) AS block_key
        FROM candidate_source
        WHERE ({cand_key_expr}) <> ''
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE good_keys_{sub_name} AS
        SELECT block_key
        FROM (
            SELECT block_key, COUNT(*) AS freq
            FROM cand_keyed_{sub_name}
            GROUP BY block_key
        )
        WHERE freq <= {max_block_size}
    """)

    dropped = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT block_key, COUNT(*) AS freq
            FROM cand_keyed_{sub_name}
            GROUP BY block_key
        ) WHERE freq > {max_block_size}
    """).fetchone()[0]

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pairs_{sub_name} AS
        SELECT DISTINCT s1t.s1_entity_id AS source1_entity_id,
               ct.candidate_entity_id AS candidate_entity_id
        FROM s1_keyed_{sub_name} s1t
        INNER JOIN good_keys_{sub_name} gk ON gk.block_key = s1t.block_key
        INNER JOIN cand_keyed_{sub_name} ct ON ct.block_key = s1t.block_key
    """)

    count = con.execute(f"SELECT COUNT(*) FROM pairs_{sub_name}").fetchone()[0]
    print(f"  [{sub_name:22}] pairs={count:>12,}  oversized keys purged={dropped:,}")


def build_deterministic_pairs(con, max_block_size):
    print("\n" + "=" * 70)
    print("CHANNEL: deterministic (exact keys, with block purging)")
    print("=" * 70)

    sub_blockers = {
        "exact_name_country": (
            "name_norm || '||' || country_norm",
            "name_norm || '||' || country_norm",
        ),
        "exact_core_name_country": (
            "name_core || '||' || country_norm",
            "name_core || '||' || country_norm",
        ),
        "exact_address": ("address_norm", "address_norm"),
        "postal_code": ("postal_code", "postal_code"),
        # First numeric token in the address (usually the house/building
        # number) combined with country. Kept alongside postal_code rather
        # than replacing it - they catch different failure modes (missing
        # PIN vs. missing/garbled house number) - but both go through the
        # same purging, since both can produce oversized blocks.
        "house_number_country": (
            "split_part(numeric_tokens, ' ', 1) || '||' || country_norm",
            "split_part(numeric_tokens, ' ', 1) || '||' || country_norm",
        ),
    }

    for sub_name, (s1_expr, cand_expr) in sub_blockers.items():
        build_purged_key_blocker(con, sub_name, s1_expr, cand_expr, max_block_size)

    union_sql = " UNION ".join(
        f"SELECT * FROM pairs_{sub_name}" for sub_name in sub_blockers
    )
    con.execute(f"CREATE OR REPLACE TEMP TABLE det_pairs AS {union_sql}")

    count = con.execute("SELECT COUNT(*) FROM det_pairs").fetchone()[0]
    print(f"Combined deterministic pairs (after purging + union): {count:,}")


# ============================================================
# CHANNEL 2: TOKEN-OVERLAP INVERTED INDEX
# ============================================================

def build_token_overlap_pairs(con, max_token_doc_freq, rare_tokens_per_entity):
    print("\n" + "=" * 70)
    print("CHANNEL: token_overlap (inverted index on RAREST name_core tokens)")
    print("=" * 70)

    con.execute("""
        CREATE OR REPLACE TEMP VIEW s1_tokens_all AS
        SELECT entity_id AS s1_entity_id, country_norm,
               UNNEST(string_split(name_core, ' ')) AS token
        FROM s1
        WHERE name_core <> ''
    """)

    con.execute("""
        CREATE OR REPLACE TEMP VIEW candidate_tokens_all AS
        SELECT entity_id AS candidate_entity_id, country_norm,
               UNNEST(string_split(name_core, ' ')) AS token
        FROM candidate_source
        WHERE name_core <> ''
    """)

    print("Computing token document frequency (candidate side)...")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE token_doc_freq AS
        SELECT token, COUNT(DISTINCT candidate_entity_id) AS doc_freq
        FROM candidate_tokens_all
        WHERE LENGTH(token) >= 2
        GROUP BY token
    """)

    # ------------------------------------------------------------
    # PRIMARY safety net: each record keeps only its N rarest tokens.
    # This is what actually bounds the join - a frequency cap alone bounds
    # individual keys, not total fanout (freq_s1 x freq_candidate summed
    # across every surviving token can still be enormous). Restricting each
    # record to a small, fixed number of blocking keys bounds the join to
    # roughly (rare_tokens_per_entity x record_count) regardless of corpus
    # size, which is what actually prevents the OOM you hit.
    # ------------------------------------------------------------

    print(f"Selecting each record's {rare_tokens_per_entity} rarest token(s)...")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_rare_tokens AS
        SELECT s1_entity_id, country_norm, token
        FROM (
            SELECT
                s1t.s1_entity_id, s1t.country_norm, s1t.token,
                ROW_NUMBER() OVER (
                    PARTITION BY s1t.s1_entity_id
                    ORDER BY COALESCE(tdf.doc_freq, 0) ASC
                ) AS rn
            FROM s1_tokens_all s1t
            INNER JOIN token_doc_freq tdf ON tdf.token = s1t.token
            WHERE tdf.doc_freq <= {max_token_doc_freq}
        )
        WHERE rn <= {rare_tokens_per_entity}
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE candidate_rare_tokens AS
        SELECT candidate_entity_id, country_norm, token
        FROM (
            SELECT
                ct.candidate_entity_id, ct.country_norm, ct.token,
                ROW_NUMBER() OVER (
                    PARTITION BY ct.candidate_entity_id
                    ORDER BY tdf.doc_freq ASC
                ) AS rn
            FROM candidate_tokens_all ct
            INNER JOIN token_doc_freq tdf ON tdf.token = ct.token
            WHERE tdf.doc_freq <= {max_token_doc_freq}
        )
        WHERE rn <= {rare_tokens_per_entity}
    """)

    n_s1_keys = con.execute("SELECT COUNT(*) FROM s1_rare_tokens").fetchone()[0]
    n_cand_keys = con.execute("SELECT COUNT(*) FROM candidate_rare_tokens").fetchone()[0]
    print(f"S1 blocking keys (after restriction): {n_s1_keys:,}")
    print(f"Candidate blocking keys (after restriction): {n_cand_keys:,}")

    print("Joining on shared rare tokens...")

    # Since both sides already only carry their rarest tokens, requiring
    # just ONE shared token is reasonable evidence (unlike the old version,
    # which needed >=2 because it was joining on every token including
    # fairly common ones).
    con.execute("""
        CREATE OR REPLACE TEMP TABLE token_pairs AS
        SELECT DISTINCT
            s1t.s1_entity_id AS source1_entity_id,
            ct.candidate_entity_id AS candidate_entity_id
        FROM s1_rare_tokens s1t
        INNER JOIN candidate_rare_tokens ct
            ON ct.token = s1t.token
           AND ct.country_norm = s1t.country_norm
    """)

    count = con.execute("SELECT COUNT(*) FROM token_pairs").fetchone()[0]
    print(f"Pairs produced: {count:,}")


# ============================================================
# CHANNEL 3: CHARACTER N-GRAM TF-IDF TOP-K (country-partitioned)
# ============================================================

def build_ngram_pairs(con, topk, min_similarity, batch_size):
    print("\n" + "=" * 70)
    print("CHANNEL: ngram_topk (fast bounded char n-gram retrieval)")
    print("=" * 70)

    countries = [
        row[0] for row in con.execute(
            "SELECT DISTINCT country_norm FROM s1 WHERE country_norm <> ''"
        ).fetchall()
    ]

    print(f"Country partitions to process: {len(countries)}")

    all_pairs = []

    for country in countries:
        country_start = time.time()

        s1_df = con.execute("""
            SELECT
                entity_id,
                CASE
                    WHEN name_core <> '' THEN name_core
                    ELSE name_norm
                END AS text
            FROM s1
            WHERE country_norm = ?
        """, [country]).fetchdf()

        cand_df = con.execute("""
            SELECT
                entity_id,
                CASE
                    WHEN name_core <> '' THEN name_core
                    ELSE name_norm
                END AS text
            FROM candidate_source
            WHERE country_norm = ?
        """, [country]).fetchdf()

        s1_df = s1_df[
            s1_df["text"].str.len() > 0
        ].reset_index(drop=True)

        cand_df = cand_df[
            cand_df["text"].str.len() > 0
        ].reset_index(drop=True)

        if len(s1_df) == 0 or len(cand_df) == 0:
            continue

        print(
            f"  [{country}] "
            f"S1={len(s1_df):,} "
            f"candidates={len(cand_df):,}"
        )

        # --------------------------------------------------------
        # Cheap 3-character prefix blocking
        # --------------------------------------------------------

        s1_df["prefix"] = (
            s1_df["text"]
            .str.replace(r"[^a-z0-9]", "", regex=True)
            .str[:3]
        )

        cand_df["prefix"] = (
            cand_df["text"]
            .str.replace(r"[^a-z0-9]", "", regex=True)
            .str[:3]
        )

        s1_df = s1_df[
            s1_df["prefix"] != ""
        ]

        cand_df = cand_df[
            cand_df["prefix"] != ""
        ]

        # Candidate prefix frequencies.
        prefix_counts = cand_df["prefix"].value_counts()

        # Keep only reasonably sized blocks.
        max_prefix_size = 2000

        valid_prefixes = set(
            prefix_counts[
                prefix_counts <= max_prefix_size
            ].index
        )

        s1_df = s1_df[
            s1_df["prefix"].isin(valid_prefixes)
        ].reset_index(drop=True)

        cand_df = cand_df[
            cand_df["prefix"].isin(valid_prefixes)
        ].reset_index(drop=True)

        print(
            f"  [{country}] "
            f"bounded candidates={len(cand_df):,}"
        )

        # --------------------------------------------------------
        # Build lookup of candidate rows by prefix.
        # This avoids repeatedly filtering the entire dataframe.
        # --------------------------------------------------------

        candidate_blocks = {
            prefix: block.reset_index(drop=True)
            for prefix, block
            in cand_df.groupby("prefix", sort=False)
        }

        s1_blocks = {
            prefix: block.reset_index(drop=True)
            for prefix, block
            in s1_df.groupby("prefix", sort=False)
        }

        # --------------------------------------------------------
        # Process each prefix independently.
        # --------------------------------------------------------

        processed_blocks = 0

        for prefix, s1_block in s1_blocks.items():

            cand_block = candidate_blocks.get(prefix)

            if cand_block is None or len(cand_block) == 0:
                continue

            # ----------------------------------------------------
            # Fit TF-IDF ONLY on this small block.
            # ----------------------------------------------------

            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                dtype=np.float32,
            )

            combined_text = pd.concat(
                [
                    s1_block["text"],
                    cand_block["text"],
                ],
                ignore_index=True,
            )

            vectorizer.fit(combined_text)

            s1_matrix = vectorizer.transform(
                s1_block["text"]
            )

            cand_matrix = vectorizer.transform(
                cand_block["text"]
            )

            # ----------------------------------------------------
            # Nearest-neighbor search inside this prefix only.
            # ----------------------------------------------------

            n_neighbors = min(
                topk,
                len(cand_block)
            )

            nn = NearestNeighbors(
                metric="cosine",
                algorithm="brute",
                n_jobs=-1,
            )

            nn.fit(cand_matrix)

            for start in range(
                0,
                len(s1_block),
                batch_size,
            ):

                end = min(
                    start + batch_size,
                    len(s1_block),
                )

                batch = s1_matrix[start:end]

                distances, indices = nn.kneighbors(
                    batch,
                    n_neighbors=n_neighbors,
                )

                similarities = 1 - distances

                for row_i in range(
                    batch.shape[0]
                ):

                    s1_entity = (
                        s1_block["entity_id"]
                        .iloc[start + row_i]
                    )

                    for col_j in range(
                        n_neighbors
                    ):

                        sim = similarities[
                            row_i,
                            col_j
                        ]

                        if sim >= min_similarity:

                            cand_entity = (
                                cand_block["entity_id"]
                                .iloc[
                                    indices[
                                        row_i,
                                        col_j
                                    ]
                                ]
                            )

                            all_pairs.append(
                                (
                                    s1_entity,
                                    cand_entity,
                                )
                            )

            processed_blocks += 1

        elapsed = time.time() - country_start

        print(
            f"  [{country}] "
            f"processed blocks={processed_blocks:,} "
            f"-> {len(all_pairs):,} cumulative pairs "
            f"({elapsed:.1f}s)"
        )

    # ------------------------------------------------------------
    # Materialize result.
    # ------------------------------------------------------------

    ngram_pairs_df = pd.DataFrame(
        all_pairs,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
    ).drop_duplicates()

    con.register(
        "ngram_pairs_df",
        ngram_pairs_df,
    )

    con.execute("""
        CREATE OR REPLACE TEMP TABLE ngram_pairs AS
        SELECT *
        FROM ngram_pairs_df
    """)

    print(
        f"Pairs produced: "
        f"{len(ngram_pairs_df):,}"
    )
    


# ============================================================
# EVALUATION
# ============================================================

def evaluate_channel(con, table_name, total_s1, truth_count):
    candidate_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    covered_s1 = con.execute(f"""
        SELECT COUNT(DISTINCT source1_entity_id) FROM {table_name}
    """).fetchone()[0]

    retrieved = con.execute(f"""
        SELECT COUNT(*)
        FROM truth_pairs t
        INNER JOIN {table_name} p
            ON p.source1_entity_id = t.source1_entity_id
           AND p.candidate_entity_id = t.candidate_entity_id
    """).fetchone()[0]

    max_per_s1 = con.execute(f"""
        SELECT COALESCE(MAX(cnt), 0) FROM (
            SELECT COUNT(*) AS cnt FROM {table_name} GROUP BY source1_entity_id
        )
    """).fetchone()[0]

    recall = retrieved / truth_count if truth_count else 0.0
    coverage = covered_s1 / total_s1 if total_s1 else 0.0
    avg_per_s1 = candidate_count / total_s1 if total_s1 else 0.0

    return {
        "candidate_pairs": candidate_count,
        "s1_coverage": coverage,
        "recall": recall,
        "avg_candidates": avg_per_s1,
        "max_candidates": max_per_s1,
    }


def print_row(name, r):
    print(f"{name:22}{r['recall']:>9.2%}{r['s1_coverage']:>14.2%}"
          f"{r['avg_candidates']:>12.2f}{r['max_candidates']:>12,}"
          f"{r['candidate_pairs']:>16,}")


def print_header():
    print(f"\n{'Channel':22}{'Recall':>10}{'S1 Coverage':>14}{'Avg Cand.':>12}"
          f"{'Max Cand.':>12}{'Total Cand.':>16}")
    print("-" * 95)


def print_summary(results, total_s1, truth_count):
    print("\n\n" + "=" * 95)
    print("CANDIDATE GENERATION V2 - SUMMARY")
    print("=" * 95)

    print_header()
    for name, r in results.items():
        print_row(name, r)

    print("-" * 95)
    print(f"S1 entities in this run: {total_s1:,}")
    print(f"Ground-truth pairs in scope: {truth_count:,}")
    print("=" * 95)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_arguments()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]

    out_path = Path(args.out) if args.out else PROCESSED_DIR / "candidate_pairs_v2_debug.parquet"

    con = connect_duckdb()
    total_s1, total_candidates = load_views(con, args.sample)
    truth_count = load_ground_truth(con)

    name_map = {"det_pairs": "deterministic", "token_pairs": "token_overlap",
                "ngram_pairs": "ngram_topk"}

    built_tables = []
    results = {}

    # ------------------------------------------------------------
    # Build + evaluate each channel IMMEDIATELY after it's built, and print
    # its row right away. This way, if a later channel crashes (OOM, bad
    # SQL, whatever), you still keep every result printed before that point
    # instead of losing everything the way the all-at-the-end version did.
    # ------------------------------------------------------------

    print_header()

    if "deterministic" in channels:
        build_deterministic_pairs(con, args.max_block_size)
        built_tables.append("det_pairs")
        results["deterministic"] = evaluate_channel(con, "det_pairs", total_s1, truth_count)
        print_row("deterministic", results["deterministic"])

    if "token" in channels:
        build_token_overlap_pairs(con, args.max_token_doc_freq, args.rare_tokens_per_entity)
        built_tables.append("token_pairs")
        results["token_overlap"] = evaluate_channel(con, "token_pairs", total_s1, truth_count)
        print_row("token_overlap", results["token_overlap"])

    if "ngram" in channels:
        build_ngram_pairs(con, args.ngram_topk, args.ngram_min_similarity,
                           args.ngram_batch_size)
        built_tables.append("ngram_pairs")
        results["ngram_topk"] = evaluate_channel(con, "ngram_pairs", total_s1, truth_count)
        print_row("ngram_topk", results["ngram_topk"])

    if not built_tables:
        raise SystemExit("No channels selected - check --channels")

    if not args.combine:
        print("\nSkipping combined union materialization to avoid temp-disk blowup.")
        print("Per-channel results above are the reliable output for this full run.")
        return

    # ------------------------------------------------------------
    # Combined union evaluation
    #
    # Evaluate the union directly against ground truth.
    # Do NOT materialize the complete union into a temporary table.
    # This saves substantial disk I/O and memory.
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMBINED UNION EVALUATION")
    print("=" * 70)

    if len(built_tables) == 1:

        combined_sql = f"""
            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM {built_tables[0]}
        """

    else:

        combined_sql = """
            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id
            FROM (
        """ + "\n        UNION ALL\n        ".join(
            f"""
            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM {table}
            """
            for table in built_tables
        ) + """
            )
        """

        # ------------------------------------------------------------
        # Evaluate recall directly against the union.
        # ------------------------------------------------------------

    combined_result = con.execute(f"""
        WITH combined AS (
            {combined_sql}
        ),

        matched AS (
            SELECT
                COUNT(*) AS recovered_pairs
            FROM combined c
            INNER JOIN truth_pairs t
                ON c.source1_entity_id = t.source1_entity_id
                AND c.candidate_entity_id = t.candidate_entity_id
        ),

        total_candidates AS (
            SELECT COUNT(*) AS candidate_pairs
            FROM combined
        ),

        covered_s1 AS (
            SELECT COUNT(DISTINCT source1_entity_id) AS covered_s1
            FROM combined
        )

        SELECT
            matched.recovered_pairs,
            total_candidates.candidate_pairs,
            covered_s1.covered_s1
        FROM matched
        CROSS JOIN total_candidates
        CROSS JOIN covered_s1
    """).fetchone()

    recovered_pairs = combined_result[0]
    candidate_pairs = combined_result[1]
    covered_s1 = combined_result[2]

    pair_recall = (
        recovered_pairs / truth_count
        if truth_count > 0
        else 0.0
    )

    s1_coverage = (
        covered_s1 / total_s1
        if total_s1 > 0
        else 0.0
    )

    avg_candidates = (
        candidate_pairs / total_s1
        if total_s1 > 0
        else 0.0
    )

        # Maximum candidate count for any S1.
    max_candidates = con.execute(f"""
        SELECT COALESCE(MAX(candidate_count), 0)
        FROM (
            SELECT
                source1_entity_id,
                COUNT(*) AS candidate_count
            FROM (
                {combined_sql}
            )
            GROUP BY source1_entity_id
        )
    """).fetchone()[0]

    print(
        f"Combined candidate pairs : {candidate_pairs:,}"
    )
    print(
        f"Recovered GT pairs       : {recovered_pairs:,}"
    )
    print(
        f"Ground-truth pairs       : {truth_count:,}"
    )
    print(
        f"PAIR-LEVEL RECALL        : {pair_recall:.2%}"
    )
    print(
        f"S1 COVERAGE              : {s1_coverage:.2%}"
    )
    print(
        f"AVG CANDIDATES/S1        : {avg_candidates:.2f}"
    )
    print(
        f"MAX CANDIDATES/S1        : {max_candidates:,}"
    )

        # ------------------------------------------------------------
        # Do NOT save the union at this stage.
        # We are only measuring candidate-generation recall.
        # ------------------------------------------------------------

    print(
            "\nCombined union evaluated without materializing "
            "the full candidate table."
        )

    con.close()
if __name__ == "__main__":
    main()