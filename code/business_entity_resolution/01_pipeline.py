"""Unicode-aware, bounded candidate generation. Ground truth is never read here."""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize as sparse_normalize
from unidecode import unidecode


ROOT = Path(__file__).resolve().parents[2]
LEGAL = {"incorporated": "inc", "corporation": "corp", "company": "co",
         "limited": "ltd", "private": "pvt", "priv": "pvt", "pvt": "pvt"}
SUFFIXES = {"inc", "corp", "co", "ltd", "pvt", "llc", "llp", "plc"}
ROUTES = ("native_char", "native_word", "address_char", "numeric", "rare_token",
          "translit_char", "core_char")
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
    args = p.parse_args()
    if args.sample_offset < 0:
        p.error("--sample-offset must be nonnegative")
    folder = args.data_dir / args.split
    if args.augment_from:
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
    if (args.stream_target or args.augment_from) and audit is not None:
        audit.to_parquet(f"{prefix}_route_audit.parquet", index=False)
        (args.output_dir / f"{args.split}_retrieval_run.json").write_text(json.dumps({
            "target_pool_size": population, "source_count": len(s1),
            "candidate_pair_count": len(pairs), "raw_pair_count": len(audit),
            "route_top": args.route_top, "cap": args.cap,
            "sample_offset": args.sample_offset if args.stream_target else None,
            "augmented_from": str(args.augment_from) if args.augment_from else None,
            "labels_used_for_retrieval": False,
        }, indent=2))
    print(f"{args.split}: S1={len(s1)} target_pool={population} selected_target={len(target)} pairs={len(pairs)} "
          f"mean={len(pairs)/max(len(s1),1):.2f} p95={pairs.groupby('source1_entity_id').size().quantile(.95) if len(pairs) else 0:.1f}")


if __name__ == "__main__":
    main()
