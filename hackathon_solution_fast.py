import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import time
import os
import multiprocessing as mp
from functools import partial
import lightgbm as lgb
from feature_engineering import normalize, unidecode, FeatureExtractor
import math

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def top_sparse_scores(scores, count):
    for row in range(scores.shape[0]):
        lo, hi = scores.indptr[row:row + 2]
        columns = scores.indices[lo:hi]
        values = scores.data[lo:hi]
        if len(values) > count:
            idx = np.argpartition(values, -count)[-count:]
            yield zip(columns[idx], values[idx])
        else:
            idx = np.argsort(-values)
            yield zip(columns[idx], values[idx])

def generate_candidates_fast(s1_texts, target_texts, max_df_fraction=0.01, top=20, batch_size=4096):
    print("Vectorizing targets...", flush=True)
    def analyzer(text):
        if not isinstance(text, str) or text == 'nan': return []
        norm = normalize(text)
        return [t for t in norm.split() if len(t) >= 3]
        
    vec = CountVectorizer(analyzer=analyzer, min_df=1, dtype=np.float32)
    T = vec.fit_transform(target_texts)
    T.data = np.ones_like(T.data)
    
    dfs = np.array(T.sum(axis=0)).flatten()
    N = T.shape[0]
    idfs = np.log((N + 1) / (dfs + 1))
    idfs[dfs > 5000] = 0.0
    
    W = sp.diags(idfs)
    T_scaled = T @ W
    
    results = []
    print("Querying...", flush=True)
    for start in range(0, len(s1_texts), batch_size):
        q = vec.transform(s1_texts[start:start + batch_size])
        q.data = np.ones_like(q.data)
        scores_csr = (q @ T_scaled.T).tocsr()
        for c_list in top_sparse_scores(scores_csr, top):
            results.append([c[0] for c in c_list])
            
    return results

def extract_features_worker(chunk_pairs, s1_dict, target_dict, idf_dict, default_idf):
    extractor = FeatureExtractor(name_idf_dict=idf_dict, default_idf=default_idf)
    dataset = []
    for s1_id, c_id in chunk_pairs:
        r1 = s1_dict.get(s1_id)
        r2 = target_dict.get(c_id)
        if r1 is None or r2 is None: continue
        n1 = str(r1['business_name'])
        n2 = str(r2['business_name'])
        a1 = str(r1['business_address'])
        a2 = str(r2['business_address'])
        c1 = str(r1['country'])
        c2 = str(r2['country'])
        
        feats = extractor.extract_features(n1, n2, a1, a2, c1, c2)
        feats['source1_entity_id'] = s1_id
        feats['candidate_entity_id'] = c_id
        dataset.append(feats)
    return dataset

def compute_idf(target_df):
    doc_freq = Counter()
    total = len(target_df)
    for row in target_df['business_name']:
        if pd.isna(row): continue
        norm = normalize(row)
        doc_freq.update(set(norm.split()))
    idf_dict = {t: math.log(total / (f + 1)) for t, f in doc_freq.items()}
    return idf_dict, math.log(total / 1)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    # 1. LOAD TRAINING DATA
    print("Loading training data...", flush=True)
    s1_train = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep='\t', nrows=200000)
    s2_train = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep='\t', nrows=500000)
    s3_train = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep='\t', nrows=500000)
    target_train = pd.concat([s2_train, s3_train], ignore_index=True)
    gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep='\t')
    
    print("Generating train candidates...", flush=True)
    train_candidate_indices = generate_candidates_fast([str(x) if pd.notna(x) else "" for x in s1_train["business_name"]], [str(x) if pd.notna(x) else "" for x in target_train["business_name"]], top=10)
    
    target_train_ids = target_train['entity_id'].values
    train_candidates = {}
    for i, s1_id in enumerate(s1_train['entity_id']):
        train_candidates[s1_id] = [target_train_ids[idx] for idx in train_candidate_indices[i]]
        
    gt_dict = gt.dropna(subset=['matched_entity_ids']).set_index('source1_entity_id')['matched_entity_ids'].to_dict()
    
    pairs_to_evaluate = []
    target_train_dict = target_train.set_index('entity_id').to_dict('index')
    s1_train_dict = s1_train.set_index('entity_id').to_dict('index')
    
    for s1_id, candidates in train_candidates.items():
        gt_matches = set(gt_dict.get(s1_id, "").split(',')) if s1_id in gt_dict else set()
        for c_id in candidates:
            if c_id in target_train_dict:
                label = 1 if c_id in gt_matches else 0
                pairs_to_evaluate.append((s1_id, c_id, label))
        for c_id in gt_matches:
            if c_id not in candidates and c_id in target_train_dict:
                pairs_to_evaluate.append((s1_id, c_id, 1))
                
    idf_dict, default_idf = compute_idf(target_train)
    
    print(f"Extracting train features for {len(pairs_to_evaluate)} pairs...", flush=True)
    train_pairs_list = [(s1, c) for s1, c, _ in pairs_to_evaluate]
    labels = [l for _, _, l in pairs_to_evaluate]
    
    chunks = list(chunk_list(train_pairs_list, 10000))
    worker = partial(extract_features_worker, s1_dict=s1_train_dict, target_dict=target_train_dict, idf_dict=idf_dict, default_idf=default_idf)
    num_cpus = os.cpu_count() or 12
    with mp.Pool(num_cpus) as p:
        results = p.map(worker, chunks)
        
    train_feats = []
    for r in results: train_feats.extend(r)
    
    train_df = pd.DataFrame(train_feats)
    train_df['label'] = labels
    feature_cols = [c for c in train_df.columns if c not in ('source1_entity_id', 'candidate_entity_id', 'label')]
    
    print(f"Training LightGBM on {len(train_df)} pairs ({train_df['label'].sum()} positives)...", flush=True)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=num_cpus)
    clf.fit(train_df[feature_cols], train_df['label'])
    
    del s1_train, s2_train, s3_train, target_train, s1_train_dict, target_train_dict, train_df, train_feats
    
    # ---------------- TEST PHASE ----------------
    print("Loading test data...", flush=True)
    s1_test = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep='\t', dtype=str, keep_default_na=False)
    s2_test = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep='\t', dtype=str, keep_default_na=False)
    s3_test = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep='\t', dtype=str, keep_default_na=False)
    target_test = pd.concat([s2_test, s3_test], ignore_index=True)
    
    idf_dict, default_idf = compute_idf(target_test)
    target_test_dict = target_test.set_index('entity_id').to_dict('index')
    s1_test_dict = s1_test.set_index('entity_id').to_dict('index')
    
    print("Generating test candidates...", flush=True)
    test_candidate_indices = generate_candidates_fast([str(x) if pd.notna(x) else "" for x in s1_test["business_name"]], [str(x) if pd.notna(x) else "" for x in target_test["business_name"]], top=5)
    
    target_test_ids = target_test['entity_id'].values
    test_candidates = {}
    test_pairs_list = []
    for i, s1_id in enumerate(s1_test['entity_id']):
        cands = [target_test_ids[idx] for idx in test_candidate_indices[i]]
        test_candidates[s1_id] = cands
        for c_id in cands:
            test_pairs_list.append((s1_id, c_id))
            
    print(f"Extracting test features for {len(test_pairs_list)} pairs...", flush=True)
    chunks = list(chunk_list(test_pairs_list, 20000))
    worker = partial(extract_features_worker, s1_dict=s1_test_dict, target_dict=target_test_dict, idf_dict=idf_dict, default_idf=default_idf)
    with mp.Pool(num_cpus) as p:
        results = p.map(worker, chunks)
        
    test_feats = []
    for r in results: test_feats.extend(r)
    
    test_df = pd.DataFrame(test_feats)
    
    print("Predicting...", flush=True)
    probs = clf.predict_proba(test_df[feature_cols])[:, 1]
    test_df['score'] = probs
    
    threshold = 0.5
    matches = test_df[test_df['score'] >= threshold].groupby('source1_entity_id')['candidate_entity_id'].apply(list).to_dict()
    
    os.makedirs("output", exist_ok=True)
    print("Writing TSVs...", flush=True)
    with open("output/matching_results.tsv", "w") as f1, open("output/candidate_pairs.tsv", "w") as f2:
        f1.write("source1_entity_id\tmatched_entity_ids\n")
        f2.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_test['entity_id']:
            m = matches.get(s1_id, [])
            c = test_candidates.get(s1_id, [])
            f1.write(f"{s1_id}\t{','.join(m)}\n")
            f2.write(f"{s1_id}\t{','.join(c)}\n")
            
    print("Done!")
