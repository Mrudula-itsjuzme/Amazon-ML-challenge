import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy.sparse as sp
import os
import re

def normalize(text):
    if not isinstance(text, str): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

def analyzer(text):
    if not isinstance(text, str) or text == 'nan': return []
    norm = normalize(text)
    return [t for t in norm.split() if len(t) >= 3]

def run_tfidf_baseline():
    print("Loading test data...", flush=True)
    s1_test = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep='\t', usecols=['entity_id', 'business_name'], dtype=str, keep_default_na=False)
    s2_test = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep='\t', usecols=['entity_id', 'business_name'], dtype=str, keep_default_na=False)
    s3_test = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep='\t', usecols=['entity_id', 'business_name'], dtype=str, keep_default_na=False)
    target_test = pd.concat([s2_test, s3_test], ignore_index=True)
    
    print("Vectorizing...", flush=True)
    vec = TfidfVectorizer(analyzer=analyzer, min_df=1, dtype=np.float32)
    T = vec.fit_transform([str(x) for x in target_test["business_name"]])
    
    # Cap max_df manually by setting idf of frequent words to 0
    dfs = np.array(T.astype(bool).sum(axis=0)).flatten()
    T.data[np.repeat(dfs > 5000, T.getnnz(axis=0))] = 0.0
    T.eliminate_zeros()
    
    target_test_ids = target_test['entity_id'].values
    matches = {}
    candidates = {}
    
    s1_texts = [str(x) for x in s1_test["business_name"]]
    batch_size = 4096
    
    print("Querying and scoring...", flush=True)
    for start in range(0, len(s1_texts), batch_size):
        q = vec.transform(s1_texts[start:start + batch_size])
        scores_csr = (q @ T.T).tocsr()
        
        for row in range(scores_csr.shape[0]):
            s1_id = s1_test['entity_id'].iloc[start + row]
            lo, hi = scores_csr.indptr[row:row + 2]
            cols = scores_csr.indices[lo:hi]
            vals = scores_csr.data[lo:hi]
            
            if len(vals) > 0:
                # Keep ALL candidates with score > 0.70
                valid_idx = np.where(vals > 0.70)[0]
                if len(valid_idx) > 0:
                    best_cols = cols[valid_idx]
                    matched_ids = [target_test_ids[c] for c in best_cols]
                    matches[s1_id] = matched_ids
                    candidates[s1_id] = matched_ids
                else:
                    # No matches > 0.7, just use best candidate for candidate pairs (optional)
                    best_idx = np.argmax(vals)
                    candidates[s1_id] = [target_test_ids[cols[best_idx]]]
                    matches[s1_id] = []
            else:
                candidates[s1_id] = []
                matches[s1_id] = []

    os.makedirs("output", exist_ok=True)
    print("Writing TSVs...", flush=True)
    with open("output/matching_results.tsv", "w") as f1, open("output/candidate_pairs.tsv", "w") as f2:
        f1.write("source1_entity_id\tmatched_entity_ids\n")
        f2.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_test['entity_id']:
            m = matches.get(s1_id, [])
            c = candidates.get(s1_id, [])
            f1.write(f"{s1_id}\t{','.join(m)}\n")
            f2.write(f"{s1_id}\t{','.join(c)}\n")
            
    print("Done!")

if __name__ == '__main__':
    run_tfidf_baseline()
