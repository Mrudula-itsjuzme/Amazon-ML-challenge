"""Unicode-aware, bounded candidate generation. Ground truth is never read here."""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import math
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize as sparse_normalize
from unidecode import unidecode


ROOT = Path(__file__).resolve().parents[2]
LEGAL = {"incorporated": "inc", "corporation": "corp", "company": "co",
         "limited": "ltd", "private": "pvt", "priv": "pvt", "pvt": "pvt"}
SUFFIXES = {"inc", "corp", "co", "ltd", "pvt", "llc", "llp", "plc"}
ROUTES = ("native_char", "native_word", "address_char", "numeric", "rare_token",
          "translit_char", "core_char", "reverse")
TEXT_ROUTES = (("native_char", "name_native", "char_wb", (3, 5)),
               ("native_word", "name_native", "word", (1, 2)),
               ("address_char", "address_native", "char_wb", (3, 5)))


def norm(value):
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    # Combining marks and Indic joiners are part of the script, not punctuation.
    return " ".join("".join(c if c.isalnum() or unicodedata.category(c).startswith("M")
                            or c in "\u200c\u200d" else " " for c in value).split())


def script(value):
    if not isinstance(value, str):
        return "unknown"
    counts = Counter()
    for char in value:
        if char.isalpha():
            counts[unicodedata.name(char, "UNKNOWN").split()[0]] += 1
    return counts.most_common(1)[0][0].lower() if counts else "unknown"


def prep(frame):
    frame = frame.copy()
    frame["name_native"] = frame.business_name.map(norm)
    frame["name_translit"] = frame.business_name.map(lambda s: norm(unidecode(s)) if isinstance(s, str) else "")
    frame["name_legal"] = frame.name_native.map(lambda s: " ".join(LEGAL.get(t, t) for t in s.split()))
    frame["name_core"] = frame.name_legal.map(lambda s: " ".join(t for t in s.split() if t not in SUFFIXES))
    frame["name_tokens"] = frame.name_native.map(lambda s: len(s.split()))
    frame["name_script"] = frame.business_name.map(script)
    frame["name_nonlatin"] = frame.name_script.map(lambda s: int(s not in ("latin", "unknown")))
    frame["address_native"] = frame.business_address.map(norm)
    frame["address_numbers"] = frame.address_native.map(lambda s: tuple(sorted(set(re.findall(r"\d+", s)))))
    frame["address_postal"] = frame.address_numbers.map(lambda a: tuple(n for n in a if 4 <= len(n) <= 6))
    frame["address_tokens"] = frame.address_native.map(lambda s: len(s.split()))
    frame["address_missing"] = (frame.address_native == "").astype("int8")
    frame["country_norm"] = frame.country.map(norm)
    return frame.drop(columns=["business_name", "business_address", "country"])


def read_source(path, limit=None):
    # A limit is useful for a smoke run; it is never selected using ground truth.
    return prep(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=limit))


def top_sparse(query, target, count):
    """Return nonzero top similarities per query without dense all-pairs arrays."""
    return top_sparse_scores((query @ target.T).tocsr(), count)


def top_sparse_scores(scores, count):
    for row in range(scores.shape[0]):
        lo, hi = scores.indptr[row:row + 2]
        columns = scores.indices[lo:hi]
        values = scores.data[lo:hi]
        if len(values) > count:
            keep = np.argpartition(values, -count)[-count:]
        else:
            keep = np.arange(len(values))
        keep = sorted(keep, key=lambda k: (-values[k], columns[k]))
        yield [(int(columns[k]), float(values[k])) for k in keep]


def vector_route(queries, targets, analyzer, ngrams, batch, top):
    if not any(targets):
        for _ in queries:
            yield []
        return
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams,
                                 dtype=np.float32, min_df=1)
    matrix = vectorizer.fit_transform(targets)
    for start in range(0, len(queries), batch):
        q = vectorizer.transform(queries[start:start + batch])
        yield from top_sparse(q, matrix, top)


def inverted_indices(target):
    number_index, token_index = defaultdict(list), defaultdict(list)
    for i, row in enumerate(target.itertuples(index=False)):
        for number in row.address_numbers:
            number_index[number].append(i)
        for token in set(row.name_native.split()) | set(row.address_native.split()):
            if len(token) >= 3:
                token_index[token].append(i)
    return number_index, token_index


def ranked_inverted(tokens, index, population, top, max_df_fraction=1.0):
    scores = defaultdict(float)
    for token in set(tokens):
        posting = index.get(token, ())
        if not posting or len(posting) / population > max_df_fraction:
            continue
        weight = math.log((population + 1) / (len(posting) + 1))
        for j in posting:
            scores[j] += weight
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]


def generate(source, target, batch=64, route_top=12, cap=60):
    """Return route evidence for the exact candidate pool scored by the matcher."""
    target_ids = target.entity_id.tolist()
    routes = {}
    for name, field, analyzer, ngrams in (
        ("native_char", "name_native", "char_wb", (3, 5)),
        ("native_word", "name_native", "word", (1, 2)),
        ("address_char", "address_native", "char_wb", (3, 5)),
    ):
        routes[name] = vector_route(source[field].tolist(), target[field].tolist(),
                                    analyzer, ngrams, batch, route_top)
    number_index, token_index = inverted_indices(target)
    records = []
    # Transliteration is queried only for non-Latin names or weak native evidence.
    translit_pending = []
    for i, row in enumerate(source.itertuples(index=False)):
        evidence = defaultdict(dict)
        for route in ("native_char", "native_word", "address_char"):
            for rank, (j, score) in enumerate(next(routes[route]), 1):
                evidence[j][route] = (rank, score)
        native_best = max((x[1] for e in evidence.values() for r, x in e.items()
                           if r == "native_char"), default=0)
        for route, tokens, index, fraction in (
            ("numeric", row.address_numbers, number_index, 1.0),
            ("rare_token", set(row.name_native.split()) | set(row.address_native.split()), token_index, 0.01),
        ):
            for rank, (j, score) in enumerate(ranked_inverted(tokens, index, len(target), route_top, fraction), 1):
                evidence[j][route] = (rank, score)
        if row.name_nonlatin or native_best < 0.35:
            translit_pending.append(i)
        records.append(evidence)
    if translit_pending:
        translit_rows = list(vector_route(
            source.iloc[translit_pending].name_translit.tolist(), target.name_translit.tolist(),
            "char_wb", (3, 5), batch, route_top))
        for i, matches in zip(translit_pending, translit_rows):
            for rank, (j, score) in enumerate(matches, 1):
                records[i][j]["translit_char"] = (rank, score)
    rows = []
    for i, evidence in enumerate(records):
        # Bound union on retrieval evidence only, with deterministic tie-breaking.
        ordered = sorted(evidence, key=lambda j: (
            -len(evidence[j]),
            -sum(1 / (rank + 1) for rank, _ in evidence[j].values()),
            target_ids[j]))[:cap]
        for j in ordered:
            item = {"source1_entity_id": source.entity_id.iat[i], "candidate_entity_id": target_ids[j],
                    "route_count": len(evidence[j])}
            for route in ROUTES:
                rank, score = evidence[j].get(route, (0, 0.0))
                item[f"{route}_rank"] = rank
                item[f"{route}_score"] = score
            rows.append(item)
    return pd.DataFrame(rows)


def sampled_source(path, count, chunk_size, offset=0):
    kept = pd.DataFrame()
    keep_count = count + offset
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             chunksize=chunk_size):
        chunk["_sample_hash"] = pd.util.hash_pandas_object(chunk.entity_id, index=False).to_numpy()
        kept = pd.concat((kept, chunk.nsmallest(keep_count, "_sample_hash")), ignore_index=True)
        kept = kept.nsmallest(keep_count, "_sample_hash")
    kept = kept.nsmallest(keep_count, "_sample_hash").iloc[offset:offset + count]
    return prep(kept.drop(columns="_sample_hash").reset_index(drop=True))


def target_chunks(folder, split, chunk_size):
    offset = 0
    for source_number in (2, 3):
        path = folder / f"{split}_source{source_number}.tsv"
        for raw in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                               chunksize=chunk_size):
            yield offset, raw
            offset += len(raw)


def hashed_vectorizers(features=2**20):
    return {name: HashingVectorizer(analyzer=analyzer, ngram_range=ngram,
                                    n_features=features, alternate_sign=False,
                                    norm=None, dtype=np.float32)
            for name, _, analyzer, ngram in TEXT_ROUTES}


def weighted_hash_matrix(vectorizer, texts, idf):
    x = vectorizer.transform(texts)
    x = x.multiply(idf).tocsr()
    return sparse_normalize(x, copy=False)


def token_matrix(token_lists, vocab, weights):
    indptr, indices, data = [0], [], []
    for values in token_lists:
        for token in set(values):
            j = vocab.get(token)
            if j is not None:
                indices.append(j)
                data.append(math.sqrt(weights[token]))
        indptr.append(len(indices))
    return csr_matrix((np.asarray(data, dtype=np.float32),
                       np.asarray(indices, dtype=np.int32),
                       np.asarray(indptr, dtype=np.int32)),
                      shape=(len(indptr) - 1, len(vocab)))


def reverse_vectorizers(features=2**18):
    return (
        HashingVectorizer(analyzer="word", ngram_range=(1, 2),
                          n_features=features, alternate_sign=False,
                          norm=None, dtype=np.float32),
        HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                          n_features=features, alternate_sign=False,
                          norm=None, dtype=np.float32),
        HashingVectorizer(analyzer="word", ngram_range=(1, 2),
                          n_features=features, alternate_sign=False,
                          norm=None, dtype=np.float32),
    )


def reverse_fields(raw):
    frame = retrieval_prep(raw)
    return (frame.name_native.tolist(),
            frame.name_core.str.replace(" ", "", regex=False).tolist(),
            frame.address_native.tolist())


def reverse_matrix(fields, vectorizers, idfs):
    blocks = []
    for values, vectorizer, idf, weight in zip(
            fields, vectorizers, idfs, (1.5, 1.5, 1.0)):
        block = vectorizer.transform(values).multiply(idf).tocsr()
        block.eliminate_zeros()
        block = sparse_normalize(block, copy=False)
        blocks.append(block * weight)
    return sparse_normalize(hstack(blocks, format="csr"), copy=False)


def prune_reverse_query(matrix, features, top_per_field=6):
    """Keep the strongest features in each view to bound reverse postings."""
    data, indices, indptr = [], [], [0]
    for row in range(matrix.shape[0]):
        lo, hi = matrix.indptr[row:row + 2]
        cols, values = matrix.indices[lo:hi], matrix.data[lo:hi]
        for block in range(3):
            positions = np.flatnonzero((cols >= block * features) &
                                       (cols < (block + 1) * features))
            if len(positions) > top_per_field:
                positions = positions[np.argpartition(values[positions],
                                                     -top_per_field)[-top_per_field:]]
            indices.extend(cols[positions])
            data.extend(values[positions])
        indptr.append(len(indices))
    trimmed = csr_matrix((np.asarray(data, dtype=np.float32),
                          np.asarray(indices, dtype=np.int32),
                          np.asarray(indptr, dtype=np.int32)), shape=matrix.shape)
    return sparse_normalize(trimmed, copy=False)


def reverse_retrieve(source, old_target, old_audit, folder, split,
                     source_chunk=50000, target_chunk=50000, reverse_top=10,
                     cap=100, hash_features=2**18, query_top_per_field=6,
                     profile_targets=None):
    """Rank each target against the complete S1 corpus, then invert top ranks."""
    source_path = folder / f"{split}_source1.tsv"
    vectorizers = reverse_vectorizers(hash_features)
    dfs = [np.zeros(hash_features, dtype=np.int32) for _ in vectorizers]
    source_population = 0
    print("Reverse pass 1: S1 document frequencies", flush=True)
    for raw in pd.read_csv(source_path, sep="\t", dtype=str,
                           keep_default_na=False, chunksize=source_chunk):
        for df, vec, values in zip(dfs, vectorizers, reverse_fields(raw)):
            df += np.asarray(vec.transform(values).sign().sum(axis=0)).ravel().astype(np.int32)
        source_population += len(raw)
        if source_population % 500000 < source_chunk:
            print(f"  S1 DF: {source_population:,}", flush=True)
    max_fractions = (.005, .001, .005)
    idfs = []
    for df, fraction in zip(dfs, max_fractions):
        idf = (np.log((source_population + 1) / (df + 1)) + 1).astype(np.float32)
        idf[(df == 0) | (df > source_population * fraction)] = 0
        idfs.append(idf)
    print("Reverse pass 2: build full S1 sparse index", flush=True)
    source_ids, blocks = [], []
    for raw in pd.read_csv(source_path, sep="\t", dtype=str,
                           keep_default_na=False, chunksize=source_chunk):
        source_ids.extend(raw.entity_id.tolist())
        blocks.append(reverse_matrix(reverse_fields(raw), vectorizers, idfs))
        if len(source_ids) % 500000 < source_chunk:
            print(f"  S1 index: {len(source_ids):,}", flush=True)
    full_source = vstack(blocks, format="csr")
    del blocks
    if len(source_ids) != source_population:
        raise ValueError("Full S1 index population changed between passes")
    wanted = set(source.entity_id)
    if not wanted.issubset(source_ids):
        raise ValueError("Study S1 rows are missing from the complete S1 index")
    cohort_rows = [j for j, sid in enumerate(source_ids) if sid in wanted]
    cohort_source = full_source[cohort_rows]
    source_index = {value: i for i, value in enumerate(source.entity_id)}
    evidence = [defaultdict(dict) for _ in range(len(source))]
    for row in old_audit.itertuples(index=False):
        i = source_index[row.source1_entity_id]
        for route in ROUTES:
            rank = getattr(row, f"{route}_rank", 0)
            if rank:
                evidence[i][row.candidate_entity_id][route] = (
                    rank, getattr(row, f"{route}_score"))
    print("Reverse pass 3: stream all targets against full S1 index", flush=True)
    extra_targets, target_population, fully_scored = [], 0, 0
    old_ids = set(old_target.entity_id)
    started = time.monotonic()
    for _, raw in target_chunks(folder, split, target_chunk):
        if profile_targets is not None:
            raw = raw.iloc[:max(0, profile_targets - target_population)]
            if raw.empty:
                break
        fields = reverse_fields(raw)
        query = prune_reverse_query(reverse_matrix(fields, vectorizers, idfs),
                                    hash_features, query_top_per_field)
        selected_local = set()
        for start in range(0, len(raw), 64):
            local_query = query[start:start + 64]
            cohort_hits = (local_query @ cohort_source.T).tocsr()
            active = np.flatnonzero(np.diff(cohort_hits.indptr))
            if not len(active):
                continue
            scores = (local_query[active] @ full_source.T).tocsr()
            fully_scored += len(active)
            for local, matches in enumerate(top_sparse_scores(scores, reverse_top)):
                raw_j = start + int(active[local])
                target_id = raw.entity_id.iat[raw_j]
                for rank, (source_j, score) in enumerate(matches, 1):
                    if score <= 0:
                        continue
                    sid = source_ids[source_j]
                    i = source_index.get(sid)
                    if i is not None:
                        evidence[i][target_id]["reverse"] = (rank, score)
                        if target_id not in old_ids:
                            selected_local.add(raw_j)
        if selected_local:
            extra_targets.append(prep(raw.iloc[sorted(selected_local)].copy()))
        target_population += len(raw)
        if target_population % 50000 < target_chunk or profile_targets is not None:
            print(f"  reverse targets: {target_population:,}; full-index scored: {fully_scored:,}; "
                  f"seconds: {time.monotonic() - started:.1f}", flush=True)
        if profile_targets is not None and target_population >= profile_targets:
            break
    target = pd.concat([old_target, *extra_targets], ignore_index=True).drop_duplicates("entity_id")
    def make_frame(limit):
        rows = []
        for i, items in enumerate(evidence):
            ordered = sorted(items, key=lambda tid: (
                -len(items[tid]),
                -sum(1 / (rank + 1) for rank, _ in items[tid].values()), tid))
            if limit is not None:
                ordered = ordered[:limit]
            for tid in ordered:
                row = {"source1_entity_id": source.entity_id.iat[i],
                       "candidate_entity_id": tid,
                       "route_count": len(items[tid])}
                for route in ROUTES:
                    rank, score = items[tid].get(route, (0, 0.0))
                    row[f"{route}_rank"], row[f"{route}_score"] = rank, score
                rows.append(row)
        return pd.DataFrame(rows)
    return target, make_frame(cap), make_frame(None), target_population, source_population


def reverse_ann_retrieve(source, old_target, old_audit, folder, split, cache_dir,
                         source_chunk=50000, target_chunk=20000, reverse_top=10,
                         cap=100, hash_features=2**18, components=128,
                         nlist=2048, nprobe=16, threads=8,
                         profile_targets=None, checkpoint_dir=None,
                         checkpoint_interval=100000):
    """Approximate full-S1 reverse search; labels never enter the index or search."""
    import sys
    local_deps = ROOT / "research_runs/python_deps"
    if local_deps.exists():
        sys.path.insert(0, str(local_deps))
    import faiss
    from sklearn.random_projection import SparseRandomProjection

    faiss.omp_set_num_threads(threads)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_path = folder / f"{split}_source1.tsv"
    vectorizers = reverse_vectorizers(hash_features)
    projector = SparseRandomProjection(n_components=components,
                                       random_state=2026, dense_output=True)
    projector.fit(csr_matrix((1, 3 * hash_features), dtype=np.float32))
    index_path = cache_dir / "reverse_ann.index"
    ids_path = cache_dir / "reverse_source_ids.parquet"
    idf_path = cache_dir / "reverse_idfs.npz"
    config_path = cache_dir / "reverse_ann_cache.json"
    source_stat = source_path.stat()
    expected = {"source_path": str(source_path.resolve()),
                "source_bytes": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "hash_features": hash_features, "components": components,
                "nlist": nlist}
    if all(path.exists() for path in (index_path, ids_path, idf_path, config_path)):
        recorded = json.loads(config_path.read_text())
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing reverse ANN cache has different source or settings")
        source_ids = pd.read_parquet(ids_path).entity_id.tolist()
        saved = np.load(idf_path)
        idfs = [saved[f"idf_{j}"] for j in range(3)]
        index = faiss.read_index(str(index_path))
        source_population = len(source_ids)
        print(f"Loaded cached ANN index: {source_population:,} S1", flush=True)
    else:
        dfs = [np.zeros(hash_features, dtype=np.int32) for _ in vectorizers]
        source_population = 0
        print("ANN pass 1: full-S1 document frequencies", flush=True)
        for raw in pd.read_csv(source_path, sep="\t", dtype=str,
                               keep_default_na=False, chunksize=source_chunk):
            for df, vec, values in zip(dfs, vectorizers, reverse_fields(raw)):
                df += np.asarray(vec.transform(values).sign().sum(axis=0)).ravel().astype(np.int32)
            source_population += len(raw)
            if source_population % 500000 < source_chunk:
                print(f"  S1 DF: {source_population:,}", flush=True)
        idfs = []
        for df, fraction in zip(dfs, (.005, .001, .005)):
            idf = (np.log((source_population + 1) / (df + 1)) + 1).astype(np.float32)
            idf[(df == 0) | (df > source_population * fraction)] = 0
            idfs.append(idf)
        np.savez_compressed(idf_path, **{f"idf_{j}": value for j, value in enumerate(idfs)})
        vectors_path = cache_dir / "reverse_source_vectors.f32"
        dense = np.memmap(vectors_path, mode="w+", dtype=np.float32,
                          shape=(source_population, components))
        source_ids, offset = [], 0
        print("ANN pass 2: project full S1 into cached dense vectors", flush=True)
        for raw in pd.read_csv(source_path, sep="\t", dtype=str,
                               keep_default_na=False, chunksize=source_chunk):
            sparse = reverse_matrix(reverse_fields(raw), vectorizers, idfs)
            block = np.asarray(projector.transform(sparse), dtype=np.float32)
            faiss.normalize_L2(block)
            dense[offset:offset + len(raw)] = block
            source_ids.extend(raw.entity_id.tolist())
            offset += len(raw)
            if offset % 500000 < source_chunk:
                print(f"  S1 vectors: {offset:,}", flush=True)
        if offset != source_population:
            raise ValueError("S1 population changed between ANN passes")
        dense.flush()
        pd.DataFrame({"entity_id": source_ids}).to_parquet(ids_path, index=False)
        actual_nlist = min(nlist, max(1, source_population // 40))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(components), components,
                                   actual_nlist, faiss.METRIC_INNER_PRODUCT)
        sample_count = min(source_population, max(100000, actual_nlist * 40))
        sample_rows = np.random.default_rng(2026).choice(source_population,
                                                         sample_count, replace=False)
        print(f"ANN pass 3: train IVF with {sample_count:,} S1 vectors", flush=True)
        index.train(np.asarray(dense[sample_rows], dtype=np.float32))
        for start in range(0, source_population, source_chunk):
            index.add(np.asarray(dense[start:start + source_chunk], dtype=np.float32))
            if (start + source_chunk) % 500000 < source_chunk:
                print(f"  ANN indexed: {min(start + source_chunk, source_population):,}", flush=True)
        faiss.write_index(index, str(index_path))
        config_path.write_text(json.dumps({**expected,
                                           "source_population": source_population,
                                           "actual_nlist": actual_nlist}, indent=2))
    index.nprobe = nprobe
    source_index = {value: i for i, value in enumerate(source.entity_id)}
    cohort_lookup = np.full(source_population, -1, dtype=np.int32)
    for j, sid in enumerate(source_ids):
        i = source_index.get(sid)
        if i is not None:
            cohort_lookup[j] = i
    if np.count_nonzero(cohort_lookup >= 0) != len(source):
        raise ValueError("Study S1 rows are missing from full ANN source index")
    evidence = [defaultdict(dict) for _ in range(len(source))]
    for row in old_audit.itertuples(index=False):
        i = source_index[row.source1_entity_id]
        for route in ROUTES:
            rank = getattr(row, f"{route}_rank", 0)
            if rank:
                evidence[i][row.candidate_entity_id][route] = (
                    rank, getattr(row, f"{route}_score"))
    old_ids = set(old_target.entity_id)
    extra_targets, target_population = [], 0
    scan_manifest = None
    pending_pairs, pending_targets = [], []
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        scan_manifest = checkpoint_dir / "scan_manifest.json"
        target_stats = [[p.name, p.stat().st_size, p.stat().st_mtime_ns]
                        for p in (folder / f"{split}_source2.tsv",
                                  folder / f"{split}_source3.tsv")]
        scan_config = {"source_ids": source.entity_id.tolist(),
                       "old_audit_rows": len(old_audit),
                       "old_target_rows": len(old_target),
                       "target_stats": target_stats,
                       "reverse_top": reverse_top, "hash_features": hash_features,
                       "components": components, "nlist": nlist,
                       "nprobe": nprobe, "target_chunk": target_chunk,
                       "checkpoint_interval": checkpoint_interval}
        if scan_manifest.exists():
            saved = json.loads(scan_manifest.read_text())
            if saved["config"] != scan_config:
                raise ValueError("ANN scan checkpoint settings or input files changed")
            target_population = saved["scanned_targets"]
            for shard in saved["shards"]:
                for row in pd.read_parquet(checkpoint_dir / shard["pairs"]).itertuples(index=False):
                    evidence[source_index[row.source1_entity_id]][row.candidate_entity_id]["reverse"] = (
                        row.reverse_rank, row.reverse_score)
                extra_targets.append(pd.read_parquet(checkpoint_dir / shard["targets"]))
            print(f"Resumed ANN scan at {target_population:,} targets", flush=True)
        else:
            saved = {"config": scan_config, "scanned_targets": 0, "shards": []}
        last_checkpoint = target_population

    def save_checkpoint():
        nonlocal last_checkpoint
        if scan_manifest is None or target_population == last_checkpoint:
            return
        stem = f"chunk_{last_checkpoint:09d}_{target_population:09d}"
        pair_file, target_file = stem + "_pairs.parquet", stem + "_targets.parquet"
        pd.DataFrame(pending_pairs, columns=["source1_entity_id", "candidate_entity_id",
                                                 "reverse_rank", "reverse_score"]).to_parquet(
            checkpoint_dir / pair_file, index=False)
        if pending_targets:
            pd.concat(pending_targets, ignore_index=True).to_parquet(
                checkpoint_dir / target_file, index=False)
        else:
            old_target.iloc[:0].to_parquet(checkpoint_dir / target_file, index=False)
        saved["shards"].append({"pairs": pair_file, "targets": target_file})
        saved["scanned_targets"] = target_population
        temporary = scan_manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved))
        temporary.replace(scan_manifest)
        pending_pairs.clear()
        pending_targets.clear()
        last_checkpoint = target_population
        print(f"  ANN checkpoint: {target_population:,} targets", flush=True)

    started = time.monotonic()
    print("ANN pass 4: stream all targets", flush=True)
    for offset, raw in target_chunks(folder, split, target_chunk):
        if offset + len(raw) <= target_population:
            continue
        if offset < target_population:
            raw = raw.iloc[target_population - offset:]
        if profile_targets is not None:
            raw = raw.iloc[:max(0, profile_targets - target_population)]
            if raw.empty:
                break
        sparse = reverse_matrix(reverse_fields(raw), vectorizers, idfs)
        query = np.asarray(projector.transform(sparse), dtype=np.float32)
        faiss.normalize_L2(query)
        scores, neighbors = index.search(query, reverse_top)
        valid = (neighbors >= 0) & (scores > 0)
        local_s1 = np.full(neighbors.shape, -1, dtype=np.int32)
        local_s1[valid] = cohort_lookup[neighbors[valid]]
        rows, ranks = np.nonzero(local_s1 >= 0)
        extra = set()
        for row, rank in zip(rows, ranks):
            target_id = raw.entity_id.iat[row]
            i = int(local_s1[row, rank])
            evidence[i][target_id]["reverse"] = (int(rank + 1), float(scores[row, rank]))
            if scan_manifest is not None:
                pending_pairs.append((source.entity_id.iat[i], target_id,
                                      int(rank + 1), float(scores[row, rank])))
            if target_id not in old_ids:
                extra.add(int(row))
        if extra:
            block = prep(raw.iloc[sorted(extra)].copy())
            extra_targets.append(block)
            if scan_manifest is not None:
                pending_targets.append(block)
        target_population += len(raw)
        if scan_manifest is not None and target_population - last_checkpoint >= checkpoint_interval:
            save_checkpoint()
        if target_population % 1000000 < target_chunk:
            elapsed = time.monotonic() - started
            print(f"  ANN targets: {target_population:,}; reverse cohort hits: "
                  f"{sum(len(items) for items in evidence):,}; seconds: {elapsed:.1f}", flush=True)
        if profile_targets is not None and target_population >= profile_targets:
            break
    save_checkpoint()
    target = pd.concat([old_target, *extra_targets], ignore_index=True).drop_duplicates("entity_id")
    def make_frame(limit):
        rows = []
        for i, items in enumerate(evidence):
            ordered = sorted(items, key=lambda tid: (
                -len(items[tid]),
                -sum(1 / (rank + 1) for rank, _ in items[tid].values()), tid))
            if limit is not None:
                ordered = ordered[:limit]
            for tid in ordered:
                row = {"source1_entity_id": source.entity_id.iat[i],
                       "candidate_entity_id": tid,
                       "route_count": len(items[tid])}
                for route in (*ROUTES, "reverse"):
                    rank, score = items[tid].get(route, (0, 0.0))
                    row[f"{route}_rank"], row[f"{route}_score"] = rank, score
                rows.append(row)
        return pd.DataFrame(rows)
    return target, make_frame(cap), make_frame(None), target_population, source_population


def retrieval_prep(raw, translit=False):
    frame = pd.DataFrame({"name_native": raw.business_name.map(norm),
                          "address_native": raw.business_address.map(norm)})
    frame["name_core"] = frame.name_native.map(
        lambda s: " ".join(LEGAL.get(t, t) for t in s.split()))
    frame["name_core"] = frame.name_core.map(
        lambda s: " ".join(t for t in s.split() if t not in SUFFIXES))
    frame["address_numbers"] = frame.address_native.map(
        lambda s: tuple(sorted(set(re.findall(r"\d+", s)))))
    if translit:
        frame["name_translit"] = raw.business_name.map(lambda s: norm(unidecode(s)))
    return frame


def bounded_map(executor, function, values, workers):
    pending, values = deque(), iter(values)
    for _ in range(2 * workers):
        try:
            pending.append(executor.submit(function, next(values)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(executor.submit(function, next(values)))
        except StopIteration:
            pass


_WORKER = {}


def init_worker(config):
    global _WORKER
    _WORKER = config


def first_pass_worker(item):
    _, raw = item
    frame = retrieval_prep(raw)
    vectorizers = hashed_vectorizers(_WORKER["hash_features"])
    dfs = {}
    for name, field, _, _ in TEXT_ROUTES:
        x = vectorizers[name].transform(frame[field].tolist())
        dfs[name] = np.asarray(x.sign().sum(axis=0)).ravel().astype(np.int32)
    numeric_df, token_df = Counter(), Counter()
    query_numbers, query_tokens = _WORKER["query_numbers"], _WORKER["query_tokens"]
    for row in frame.itertuples(index=False):
        numeric_df.update(set(row.address_numbers) & query_numbers)
        token_df.update((set(row.name_native.split()) | set(row.address_native.split())) & query_tokens)
    return len(frame), dfs, numeric_df, token_df


def scoring_worker(item):
    offset, raw = item
    frame = retrieval_prep(raw)
    config = _WORKER
    routes = {}
    for name, field, _, _ in TEXT_ROUTES:
        vectorizer = config["vectorizers"][name]
        matrix = weighted_hash_matrix(vectorizer, frame[field].tolist(), config["idfs"][name])
        routes[name] = matrix
    routes["numeric"] = token_matrix(frame.address_numbers,
                                      config["number_vocab"], config["num_weights"])
    routes["rare_token"] = token_matrix(
        (set(r.name_native.split()) | set(r.address_native.split()) for r in frame.itertuples(index=False)),
        config["rare_vocab"], config["rare_weights"])
    results = {}
    for route, matrix in routes.items():
        query = config["queries"][route]
        by_query = []
        for start in range(0, query.shape[0], config["batch"]):
            by_query.extend(top_sparse(query[start:start + config["batch"]], matrix,
                                       config["route_top"]))
        results[route] = by_query
    return offset, results


def translit_df_worker(item):
    _, raw = item
    vectorizer = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                   n_features=_WORKER["hash_features"],
                                   alternate_sign=False, norm=None, dtype=np.float32)
    x = vectorizer.transform(retrieval_prep(raw, translit=True).name_translit.tolist())
    return np.asarray(x.sign().sum(axis=0)).ravel().astype(np.int32)


def translit_scoring_worker(item):
    offset, raw = item
    config = _WORKER
    vectorizer = config["vectorizer"]
    matrix = weighted_hash_matrix(vectorizer,
                                   retrieval_prep(raw, translit=True).name_translit.tolist(),
                                   config["idf"])
    by_query = []
    for start in range(0, config["query"].shape[0], config["batch"]):
        by_query.extend(top_sparse(config["query"][start:start + config["batch"]],
                                   matrix, config["route_top"]))
    return offset, by_query


def augment_df_worker(item):
    _, raw = item
    frame = retrieval_prep(raw, translit=True)
    features = _WORKER["hash_features"]
    trans = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                              n_features=features, alternate_sign=False,
                              norm=None, dtype=np.float32)
    address = trans
    tx = trans.transform(frame.name_translit.tolist())
    ax = address.transform(frame.address_native.tolist())
    cx = trans.transform(frame.name_core.tolist())
    num_df = Counter()
    for nums in frame.address_numbers:
        num_df.update(set(nums) & _WORKER["query_numbers"])
    return (np.asarray(tx.sign().sum(axis=0)).ravel().astype(np.int32),
            np.asarray(ax.sign().sum(axis=0)).ravel().astype(np.int32),
            np.asarray(cx.sign().sum(axis=0)).ravel().astype(np.int32),
            num_df, len(frame))


def augment_score_worker(item):
    offset, raw = item
    config = _WORKER
    frame = retrieval_prep(raw, translit=True)
    trans = weighted_hash_matrix(config["vectorizer"], frame.name_translit.tolist(),
                                 config["trans_idf"])
    address = weighted_hash_matrix(config["vectorizer"], frame.address_native.tolist(),
                                   config["address_idf"])
    core = weighted_hash_matrix(config["vectorizer"], frame.name_core.tolist(),
                                config["core_idf"])
    numeric = token_matrix(frame.address_numbers, config["number_vocab"],
                           config["num_weights"])
    result = {"translit_char": [], "numeric": [], "core_char": []}
    for start in range(0, config["trans_q"].shape[0], config["batch"]):
        end = start + config["batch"]
        result["translit_char"].extend(top_sparse(
            config["trans_q"][start:end], trans, config["route_top"]))
        result["core_char"].extend(top_sparse(
            config["core_q"][start:end], core, config["route_top"]))
        num_scores = (config["num_q"][start:end] @ numeric.T).tocsr()
        addr_scores = (config["address_q"][start:end] @ address.T).tocsr()
        combined = num_scores + 4 * num_scores.sign().multiply(addr_scores)
        result["numeric"].extend(top_sparse_scores(combined.tocsr(), config["route_top"]))
    return offset, result


def augment_retrieval(source, old_target, old_audit, folder, split,
                      chunk_size=50000, batch=16, route_top=50, cap=60,
                      hash_features=2**20, workers=4):
    """Add unconditional transliteration and address-aware numeric ranking."""
    vectorizer = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                   n_features=hash_features, alternate_sign=False,
                                   norm=None, dtype=np.float32)
    numbers = set(n for values in source.address_numbers for n in values)
    trans_df = np.zeros(hash_features, dtype=np.int32)
    address_df = np.zeros(hash_features, dtype=np.int32)
    core_df = np.zeros(hash_features, dtype=np.int32)
    num_df, population = Counter(), 0
    print("Augment pass 1: transliteration/address IDF and numeric DF", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=({"hash_features": hash_features,
                                        "query_numbers": numbers},)) as executor:
        iterator = bounded_map(executor, augment_df_worker,
                               target_chunks(folder, split, chunk_size), workers)
        for k, (tdf, adf, cdf, ndf, count) in enumerate(iterator, 1):
            trans_df += tdf
            address_df += adf
            core_df += cdf
            num_df.update(ndf)
            population += count
            if k % 25 == 0:
                print(f"  augment pass 1: {population:,} targets", flush=True)
    trans_idf = (np.log((population + 1) / (trans_df + 1)) + 1).astype(np.float32)
    address_idf = (np.log((population + 1) / (address_df + 1)) + 1).astype(np.float32)
    core_idf = (np.log((population + 1) / (core_df + 1)) + 1).astype(np.float32)
    number_vocab = {n: i for i, n in enumerate(sorted(numbers))}
    num_weights = {n: math.log((population + 1) / (num_df[n] + 1)) for n in numbers}
    config = {"vectorizer": vectorizer, "trans_idf": trans_idf,
              "address_idf": address_idf, "core_idf": core_idf,
              "trans_q": weighted_hash_matrix(vectorizer, source.name_translit.tolist(), trans_idf),
              "address_q": weighted_hash_matrix(vectorizer, source.address_native.tolist(), address_idf),
              "core_q": weighted_hash_matrix(vectorizer, source.name_core.tolist(), core_idf),
              "num_q": token_matrix(source.address_numbers, number_vocab, num_weights),
              "number_vocab": number_vocab, "num_weights": num_weights,
              "batch": batch, "route_top": route_top}
    heaps = {r: [[] for _ in range(len(source))]
             for r in ("translit_char", "numeric", "core_char")}
    print("Augment pass 2: full-target route ranking", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(config,)) as executor:
        iterator = bounded_map(executor, augment_score_worker,
                               target_chunks(folder, split, chunk_size), workers)
        for k, (offset, results) in enumerate(iterator, 1):
            for route, by_query in results.items():
                for i, matches in enumerate(by_query):
                    heap = heaps[route][i]
                    for local_j, score in matches:
                        entry = (score, -(offset + local_j))
                        if len(heap) < route_top:
                            heapq.heappush(heap, entry)
                        elif entry > heap[0]:
                            heapq.heapreplace(heap, entry)
            if k % 25 == 0:
                print(f"  augment pass 2: {offset + chunk_size:,} targets scanned", flush=True)
    needed = {-negative_j for by_query in heaps.values() for heap in by_query
              for _, negative_j in heap}
    new_targets = []
    for offset, raw in target_chunks(folder, split, chunk_size):
        mask = np.fromiter((offset + j in needed for j in range(len(raw))), dtype=bool)
        if mask.any():
            subset = prep(raw.loc[mask].copy())
            subset["_global_index"] = offset + np.flatnonzero(mask)
            new_targets.append(subset)
    new_target = pd.concat(new_targets, ignore_index=True)
    index_to_id = dict(zip(new_target._global_index, new_target.entity_id))
    target = pd.concat((old_target, new_target.drop(columns="_global_index")),
                       ignore_index=True).drop_duplicates("entity_id")
    source_ids = source.entity_id.tolist()
    source_index = {value: i for i, value in enumerate(source_ids)}
    evidence = [defaultdict(dict) for _ in source_ids]
    for row in old_audit.itertuples(index=False):
        i = source_index[row.source1_entity_id]
        for route in ROUTES:
            if route in ("numeric", "translit_char"):
                continue
            rank = getattr(row, f"{route}_rank", 0)
            if rank:
                evidence[i][row.candidate_entity_id][route] = (
                    rank, getattr(row, f"{route}_score"))
    for route, by_query in heaps.items():
        for i, heap in enumerate(by_query):
            for rank, (score, negative_j) in enumerate(sorted(heap, reverse=True), 1):
                entity_id = index_to_id[-negative_j]
                evidence[i][entity_id][route] = (rank, score)
    def frame_for(cap_value):
        rows = []
        for i, items in enumerate(evidence):
            ordered = sorted(items, key=lambda entity_id: (
                -len(items[entity_id]),
                -sum(1 / (rank + 1) for rank, _ in items[entity_id].values()),
                entity_id))
            if cap_value is not None:
                ordered = ordered[:cap_value]
            for entity_id in ordered:
                item = {"source1_entity_id": source_ids[i], "candidate_entity_id": entity_id,
                        "route_count": len(items[entity_id])}
                for route in ROUTES:
                    rank, score = items[entity_id].get(route, (0, 0.0))
                    item[f"{route}_rank"], item[f"{route}_score"] = rank, score
                rows.append(item)
        return pd.DataFrame(rows)
    return target, frame_for(cap), frame_for(None), population


def stream_generate(source, folder, split, chunk_size=50000, batch=32,
                    route_top=20, cap=60, hash_features=2**20, audit_retrieval=False,
                    workers=4):
    """Full-target retrieval for a bounded S1 cohort; never reads labels."""
    vectorizers = hashed_vectorizers(hash_features)
    query_numbers = set(n for values in source.address_numbers for n in values)
    query_tokens = set(t for row in source.itertuples(index=False)
                       for t in set(row.name_native.split()) | set(row.address_native.split())
                       if len(t) >= 3)
    numeric_df, token_df = Counter(), Counter()
    dfs = {name: np.zeros(hash_features, dtype=np.int32) for name in vectorizers}
    population = 0
    print("Streaming pass 1: global document frequencies", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=({"hash_features": hash_features,
                                        "query_numbers": query_numbers,
                                        "query_tokens": query_tokens},)) as executor:
        iterator = bounded_map(executor, first_pass_worker,
                               target_chunks(folder, split, chunk_size), workers)
        for chunk_number, (chunk_count, chunk_dfs, chunk_nums, chunk_tokens) in enumerate(iterator, 1):
            population += chunk_count
            for name in vectorizers:
                dfs[name] += chunk_dfs[name]
            numeric_df.update(chunk_nums)
            token_df.update(chunk_tokens)
            if chunk_number % 25 == 0:
                print(f"  pass 1: {population:,} targets", flush=True)
    idfs = {name: (np.log((population + 1) / (df + 1)) + 1).astype(np.float32)
            for name, df in dfs.items()}
    number_vocab = {token: i for i, token in enumerate(sorted(query_numbers))}
    rare_tokens = {t for t in query_tokens if token_df[t] and token_df[t] / population <= .01}
    rare_vocab = {token: i for i, token in enumerate(sorted(rare_tokens))}
    num_weights = {t: math.log((population + 1) / (numeric_df[t] + 1)) for t in number_vocab}
    rare_weights = {t: math.log((population + 1) / (token_df[t] + 1)) for t in rare_vocab}
    queries = {name: weighted_hash_matrix(vectorizers[name], source[field].tolist(), idfs[name])
               for name, field, _, _ in TEXT_ROUTES}
    queries["numeric"] = token_matrix(source.address_numbers, number_vocab, num_weights)
    queries["rare_token"] = token_matrix(
        (set(r.name_native.split()) | set(r.address_native.split()) for r in source.itertuples(index=False)),
        rare_vocab, rare_weights)
    heaps = {route: [[] for _ in range(len(source))] for route in (*vectorizers, "numeric", "rare_token")}
    print("Streaming pass 2: global route top-k", flush=True)

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=({"vectorizers": vectorizers, "idfs": idfs,
                                        "queries": queries, "number_vocab": number_vocab,
                                        "rare_vocab": rare_vocab, "num_weights": num_weights,
                                        "rare_weights": rare_weights, "batch": batch,
                                        "route_top": route_top},)) as executor:
        iterator = bounded_map(executor, scoring_worker,
                               target_chunks(folder, split, chunk_size), workers)
        for chunk_number, (offset, results) in enumerate(iterator, 1):
            for route, by_query in results.items():
                for i, matches in enumerate(by_query):
                    heap = heaps[route][i]
                    for local_j, score in matches:
                        entry = (score, -(offset + local_j))
                        if len(heap) < route_top:
                            heapq.heappush(heap, entry)
                        elif entry > heap[0]:
                            heapq.heapreplace(heap, entry)
            if chunk_number % 25 == 0:
                print(f"  pass 2: {offset + chunk_size:,} targets scanned", flush=True)

    # Only records with weak native retrieval or a non-Latin S1 name use this route.
    weak = [i for i, row in enumerate(source.itertuples(index=False))
            if row.name_nonlatin or max((score for score, _ in heaps["native_char"][i]), default=0) < .35]
    if weak:
        print(f"Conditional transliteration: {len(weak)} of {len(source)} S1 queries", flush=True)
        trans_vectorizer = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                             n_features=hash_features, alternate_sign=False,
                                             norm=None, dtype=np.float32)
        trans_df = np.zeros(hash_features, dtype=np.int32)
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                 initargs=({"hash_features": hash_features},)) as executor:
            iterator = bounded_map(executor, translit_df_worker,
                                   target_chunks(folder, split, chunk_size), workers)
            for chunk_df in iterator:
                trans_df += chunk_df
        trans_idf = (np.log((population + 1) / (trans_df + 1)) + 1).astype(np.float32)
        trans_q = weighted_hash_matrix(trans_vectorizer,
                                       source.iloc[weak].name_translit.tolist(), trans_idf)
        heaps["translit_char"] = [[] for _ in range(len(source))]
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                 initargs=({"vectorizer": trans_vectorizer, "idf": trans_idf,
                                            "query": trans_q, "batch": batch,
                                            "route_top": route_top},)) as executor:
            iterator = bounded_map(executor, translit_scoring_worker,
                                   target_chunks(folder, split, chunk_size), workers)
            for offset, by_query in iterator:
                for local_i, matches in enumerate(by_query):
                    heap = heaps["translit_char"][weak[local_i]]
                    for local_j, score in matches:
                        entry = (score, -(offset + local_j))
                        if len(heap) < route_top:
                            heapq.heappush(heap, entry)
                        elif entry > heap[0]:
                            heapq.heapreplace(heap, entry)

    evidence = [defaultdict(dict) for _ in range(len(source))]
    for route, by_query in heaps.items():
        for i, heap in enumerate(by_query):
            for rank, (score, negative_j) in enumerate(sorted(heap, reverse=True), 1):
                evidence[i][-negative_j][route] = (rank, score)
    selected = []
    for items in evidence:
        ordered = sorted(items, key=lambda j: (-len(items[j]),
                         -sum(1 / (rank + 1) for rank, _ in items[j].values()), j))[:cap]
        selected.append(ordered)
    needed = ({j for row in evidence for j in row} if audit_retrieval
              else {j for row in selected for j in row})
    targets = []
    for offset, raw in target_chunks(folder, split, chunk_size):
        mask = np.fromiter((offset + j in needed for j in range(len(raw))), dtype=bool)
        if mask.any():
            subset = prep(raw.loc[mask].copy())
            subset["_global_index"] = offset + np.flatnonzero(mask)
            targets.append(subset)
    target = pd.concat(targets, ignore_index=True)
    index_to_id = dict(zip(target._global_index, target.entity_id))
    target = target.drop(columns="_global_index")
    def pair_rows(selections):
        rows = []
        for i, chosen in enumerate(selections):
            for j in chosen:
                row = {"source1_entity_id": source.entity_id.iat[i],
                       "candidate_entity_id": index_to_id[j], "route_count": len(evidence[i][j])}
                for route in ROUTES:
                    rank, score = evidence[i][j].get(route, (0, 0.0))
                    row[f"{route}_rank"], row[f"{route}_score"] = rank, score
                rows.append(row)
        return pd.DataFrame(rows)
    pairs = pair_rows(selected)
    audit = pair_rows([list(items) for items in evidence]) if audit_retrieval else None
    return target, pairs, population, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "student_resource/dataset")
    p.add_argument("--output-dir", type=Path, default=ROOT / "output")
    p.add_argument("--s1-limit", type=int)
    p.add_argument("--target-limit", type=int, help="Rows from each target file; smoke runs only")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--route-top", type=int, default=12)
    p.add_argument("--cap", type=int, default=60)
    p.add_argument("--stream-target", action="store_true", help="Scan the full target pool in chunks")
    p.add_argument("--sample-s1", type=int, help="Label-independent S1 sample for full-target research")
    p.add_argument("--sample-offset", type=int, default=0,
                   help="Skip this many lowest-hash S1 rows to form a disjoint cohort")
    p.add_argument("--target-chunk", type=int, default=50000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--audit-retrieval", action="store_true", help="Keep raw route union for ablations")
    p.add_argument("--augment-from", type=Path, help="Prior full-target route audit to augment")
    p.add_argument("--reverse-from", type=Path,
                   help="Add target-to-full-S1 retrieval to a prior full-target route audit")
    p.add_argument("--reverse-ann-from", type=Path,
                   help="Approximate target-to-full-S1 retrieval from a prior full-target route audit")
    p.add_argument("--ann-cache-dir", type=Path)
    p.add_argument("--ann-components", type=int, default=128)
    p.add_argument("--ann-nlist", type=int, default=2048)
    p.add_argument("--ann-nprobe", type=int, default=16)
    p.add_argument("--ann-threads", type=int, default=8)
    p.add_argument("--ann-profile-targets", type=int,
                   help="Bounded ANN speed profile; never a full-pool recall result")
    p.add_argument("--ann-checkpoint-dir", type=Path,
                   help="Persist target-scan chunks so an interrupted run can resume")
    p.add_argument("--ann-checkpoint-interval", type=int, default=100000)
    p.add_argument("--reverse-top", type=int, default=10)
    p.add_argument("--reverse-hash-features", type=int, default=2**18)
    p.add_argument("--reverse-query-top-per-field", type=int, default=6)
    p.add_argument("--reverse-source-chunk", type=int, default=50000)
    p.add_argument("--reverse-profile-targets", type=int,
                   help="Bounded runtime profile; output is not a full-pool retrieval result")
    args = p.parse_args()
    if args.sample_offset < 0:
        p.error("--sample-offset must be nonnegative")
    folder = args.data_dir / args.split
    source_population = None
    if args.reverse_from and args.reverse_ann_from:
        p.error("Choose one reverse retrieval mode")
    if args.reverse_ann_from:
        if args.augment_from or args.stream_target or args.sample_s1 or args.target_limit:
            p.error("--reverse-ann-from excludes other retrieval modes and target limits")
        prefix = args.reverse_ann_from / args.split
        s1 = pd.read_parquet(f"{prefix}_s1.parquet")
        target_path = Path(f"{prefix}_target.parquet")
        audit_path = Path(f"{prefix}_route_audit.parquet")
        if target_path.exists():
            old_target = pd.read_parquet(target_path)
        else:
            print("Reverse ANN base target artifact missing; running reverse-only retrieval profile.", flush=True)
            old_target = s1.iloc[0:0].copy()
        if audit_path.exists():
            old_audit = pd.read_parquet(audit_path)
        else:
            print("Reverse ANN base route audit missing; forward-route union is disabled for this run.", flush=True)
            old_audit = pd.DataFrame(columns=[
                "source1_entity_id", "candidate_entity_id",
                *[f"{route}_{suffix}" for route in ROUTES for suffix in ("rank", "score")]
            ])
        target, pairs, audit, population, source_population = reverse_ann_retrieve(
            s1, old_target, old_audit, folder, args.split,
            args.ann_cache_dir or args.output_dir / "ann_cache",
            source_chunk=args.reverse_source_chunk, target_chunk=args.target_chunk,
            reverse_top=args.reverse_top, cap=args.cap,
            hash_features=args.reverse_hash_features,
            components=args.ann_components, nlist=args.ann_nlist,
            nprobe=args.ann_nprobe, threads=args.ann_threads,
            profile_targets=args.ann_profile_targets,
            checkpoint_dir=args.ann_checkpoint_dir,
            checkpoint_interval=args.ann_checkpoint_interval)
    elif args.reverse_from:
        if args.augment_from or args.stream_target or args.sample_s1 or args.target_limit:
            p.error("--reverse-from excludes other retrieval modes and target limits")
        prefix = args.reverse_from / args.split
        s1 = pd.read_parquet(f"{prefix}_s1.parquet")
        old_target = pd.read_parquet(f"{prefix}_target.parquet")
        old_audit = pd.read_parquet(f"{prefix}_route_audit.parquet")
        target, pairs, audit, population, source_population = reverse_retrieve(
            s1, old_target, old_audit, folder, args.split,
            source_chunk=args.reverse_source_chunk, target_chunk=args.target_chunk,
            reverse_top=args.reverse_top, cap=args.cap,
            hash_features=args.reverse_hash_features,
            query_top_per_field=args.reverse_query_top_per_field,
            profile_targets=args.reverse_profile_targets)
    elif args.augment_from:
        if args.stream_target or args.sample_s1 or args.target_limit or args.sample_offset:
            p.error("--augment-from excludes --stream-target, --sample-s1 and --target-limit")
        prefix = args.augment_from / args.split
        s1 = pd.read_parquet(f"{prefix}_s1.parquet")
        old_target = pd.read_parquet(f"{prefix}_target.parquet")
        old_audit = pd.read_parquet(f"{prefix}_route_audit.parquet")
        target, pairs, audit, population = augment_retrieval(
            s1, old_target, old_audit, folder, args.split, args.target_chunk,
            args.batch, args.route_top, args.cap, workers=args.workers)
    elif args.stream_target:
        if not args.sample_s1 or args.target_limit:
            p.error("--stream-target requires --sample-s1 and excludes --target-limit")
        s1 = sampled_source(folder / f"{args.split}_source1.tsv", args.sample_s1,
                            args.target_chunk, args.sample_offset)
        target, pairs, population, audit = stream_generate(s1, folder, args.split, args.target_chunk,
                                                            args.batch, args.route_top, args.cap,
                                                            audit_retrieval=args.audit_retrieval,
                                                            workers=args.workers)
    else:
        s1 = read_source(folder / f"{args.split}_source1.tsv", args.s1_limit)
        targets = [read_source(folder / f"{args.split}_source{i}.tsv", args.target_limit) for i in (2, 3)]
        target = pd.concat(targets, ignore_index=True)
        pairs = generate(s1, target, args.batch, args.route_top, args.cap)
        population = len(target)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.split
    s1.to_parquet(f"{prefix}_s1.parquet", index=False)
    target.to_parquet(f"{prefix}_target.parquet", index=False)
    pairs.to_parquet(f"{prefix}_pairs.parquet", index=False)
    if (args.stream_target or args.augment_from or args.reverse_from or args.reverse_ann_from) and audit is not None:
        audit.to_parquet(f"{prefix}_route_audit.parquet", index=False)
        (args.output_dir / f"{args.split}_retrieval_run.json").write_text(json.dumps({
            "target_pool_size": population, "source_count": len(s1),
            "candidate_pair_count": len(pairs), "raw_pair_count": len(audit),
            "route_top": args.route_top, "cap": args.cap,
            "sample_offset": args.sample_offset if args.stream_target else None,
            "augmented_from": str(args.augment_from) if args.augment_from else None,
            "reverse_from": str(args.reverse_from) if args.reverse_from else None,
            "reverse_ann_from": str(args.reverse_ann_from) if args.reverse_ann_from else None,
            "reverse_top": args.reverse_top if args.reverse_from else None,
            "ann_profile_targets": args.ann_profile_targets if args.reverse_ann_from else None,
            "ann_components": args.ann_components if args.reverse_ann_from else None,
            "ann_nlist": args.ann_nlist if args.reverse_ann_from else None,
            "ann_nprobe": args.ann_nprobe if args.reverse_ann_from else None,
            "reverse_query_top_per_field": args.reverse_query_top_per_field if args.reverse_from else None,
            "reverse_profile_targets": args.reverse_profile_targets if args.reverse_from else None,
            "reverse_source_population": source_population,
            "labels_used_for_retrieval": False,
        }, indent=2))
    print(f"{args.split}: S1={len(s1)} target_pool={population} selected_target={len(target)} pairs={len(pairs)} "
          f"mean={len(pairs)/max(len(s1),1):.2f} p95={pairs.groupby('source1_entity_id').size().quantile(.95) if len(pairs) else 0:.1f}")


if __name__ == "__main__":
    main()
