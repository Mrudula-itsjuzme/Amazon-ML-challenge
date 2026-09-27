import re

with open("code/business_entity_resolution/01_pipeline.py", "r") as f:
    code = f.read()

new_funcs = """
def sum_idf_route(queries, targets, top, batch, max_df_fraction, min_len=0):
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    import scipy.sparse as sp
    
    def analyzer(tokens):
        if min_len > 0:
            return [t for t in tokens if len(t) >= min_len]
        return tokens
        
    vec = CountVectorizer(analyzer=analyzer, min_df=1, dtype=np.float32)
    T = vec.fit_transform(targets)
    T.data = np.ones_like(T.data)
    
    dfs = np.array(T.sum(axis=0)).flatten()
    N = T.shape[0]
    idfs = np.log((N + 1) / (dfs + 1))
    
    # apply max_df_fraction
    idfs[dfs / N > max_df_fraction] = 0.0
    
    W = sp.diags(idfs)
    T_scaled = T @ W
    
    if not any(targets):
        for _ in queries:
            yield []
        return
        
    for start in range(0, len(queries), batch):
        q = vec.transform(queries[start:start + batch])
        q.data = np.ones_like(q.data)
        yield from top_sparse_scores((q @ T_scaled.T).tocsr(), top)

"""

# Insert new_funcs before generate
code = code.replace("def generate(", new_funcs + "\ndef generate(")

# Modify generate to use sum_idf_route instead of inverted_indices
old_gen = """    number_index, token_index = inverted_indices(target)
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
        records.append(evidence)"""

new_gen = """    routes["numeric"] = sum_idf_route(source.address_numbers.tolist(), target.address_numbers.tolist(), route_top, batch, max_df_fraction=1.0, min_len=0)
    
    def get_rare_tokens(row):
        return list(set(row.name_native.split()) | set(row.address_native.split()))
    
    source_rare = [get_rare_tokens(r) for r in source.itertuples(index=False)]
    target_rare = [get_rare_tokens(r) for r in target.itertuples(index=False)]
    
    routes["rare_token"] = sum_idf_route(source_rare, target_rare, route_top, batch, max_df_fraction=0.01, min_len=3)
    
    records = []
    translit_pending = []
    for i, row in enumerate(source.itertuples(index=False)):
        evidence = defaultdict(dict)
        for route in ("native_char", "native_word", "address_char", "numeric", "rare_token"):
            for rank, (j, score) in enumerate(next(routes[route]), 1):
                evidence[j][route] = (rank, score)
        native_best = max((x[1] for e in evidence.values() for r, x in e.items()
                           if r == "native_char"), default=0)
        if row.name_nonlatin or native_best < 0.35:
            translit_pending.append(i)
        records.append(evidence)"""

code = code.replace(old_gen, new_gen)

with open("code/business_entity_resolution/01_pipeline_fast.py", "w") as f:
    f.write(code)

