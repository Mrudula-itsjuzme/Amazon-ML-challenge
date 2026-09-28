import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.sparse as sp
import time
import math
from collections import Counter
import re
import multiprocessing as mp
import os
import lightgbm as lgb
from feature_engineering import FeatureExtractor

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def normalize(text):
    if not isinstance(text, str): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

def top_sparse_scores(scores, count):
    for row in range(scores.shape[0]):
        lo, hi = scores.indptr[row:row + 2]
        columns = scores.indices[lo:hi]
        values = scores.data[lo:hi]
        if len(values) > count:
            idx = np.argpartition(values, -count)[-count:]
            yield zip(columns[idx], values[idx])
        elif len(values) > 0:
            idx = np.argsort(-values)
            yield zip(columns[idx], values[idx])
        else:
            yield []

def generate_candidates_fast(s1_texts, target_texts, max_df=5000, top=1, batch_size=4096):
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
    idfs[dfs > max_df] = 0.0
    W = sp.diags(idfs)
    T_scaled = T @ W
    results = []
    for start in range(0, len(s1_texts), batch_size):
        q = vec.transform(s1_texts[start:start + batch_size])
        q.data = np.ones_like(q.data)
        scores_csr = (q @ T_scaled.T).tocsr()
        for c_list in top_sparse_scores(scores_csr, top):
            results.append([c[0] for c in c_list])
    return results

def compute_idf(target_df):
    doc_freq = Counter()
    total = len(target_df)
    for row in target_df['business_name']:
        if pd.isna(row): continue
        norm = normalize(row)
        doc_freq.update(set(norm.split()))
    idf_dict = {t: math.log(total / (f + 1)) for t, f in doc_freq.items()}
    return idf_dict, math.log(total / 1)

# GLOBAL VARIABLES for fork memory sharing
GLOBAL_S1_DICT = None
GLOBAL_TARGET_DICT = None
GLOBAL_IDF_DICT = None
GLOBAL_DEFAULT_IDF = None
GLOBAL_CLF = None
GLOBAL_FEATURE_COLS = None

def extract_features_worker(chunk_pairs):
    extractor = FeatureExtractor(name_idf_dict=GLOBAL_IDF_DICT, default_idf=GLOBAL_DEFAULT_IDF)
    dataset = []
    for s1_id, c_id in chunk_pairs:
        r1 = GLOBAL_S1_DICT.get(s1_id)
        r2 = GLOBAL_TARGET_DICT.get(c_id)
        if r1 is None or r2 is None: continue
        n1 = str(r1['business_name'])
        n2 = str(r2['business_name'])
        a1 = str(r1['business_address'])
        a2 = str(r2['business_address'])
        c1 = str(r1['country'])
        c2 = str(r2['country'])
        feats = extractor.extract_features(n1, n2, a1, a2, c1, c2)
        dataset.append(feats)
    return dataset

def predict_worker(chunk_pairs):
    extractor = FeatureExtractor(name_idf_dict=GLOBAL_IDF_DICT, default_idf=GLOBAL_DEFAULT_IDF)
    dataset = []
    pairs = []
    for s1_id, c_id in chunk_pairs:
        r1 = GLOBAL_S1_DICT.get(s1_id)
        r2 = GLOBAL_TARGET_DICT.get(c_id)
        if r1 is None or r2 is None: continue
        n1 = str(r1['business_name'])
        n2 = str(r2['business_name'])
        a1 = str(r1['business_address'])
        a2 = str(r2['business_address'])
        c1 = str(r1['country'])
        c2 = str(r2['country'])
        feats = extractor.extract_features(n1, n2, a1, a2, c1, c2)
        dataset.append(feats)
        pairs.append((s1_id, c_id))
    
    if not dataset: return []
    df = pd.DataFrame(dataset)[GLOBAL_FEATURE_COLS]
    probs = GLOBAL_CLF.predict_proba(df)[:, 1]
    
    res = []
    for (s1_id, c_id), p in zip(pairs, probs):
        if p >= 0.5:
            res.append((s1_id, c_id))
    return res

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    
    print("Loading training data...", flush=True)
    s1_train = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep='\t', nrows=50000)
    s2_train = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep='\t', nrows=250000)
    s3_train = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep='\t', nrows=250000)
    target_train = pd.concat([s2_train, s3_train], ignore_index=True)
    gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep='\t')
    
    print("Generating train candidates...", flush=True)
    # Train can still use top=5 to learn negative examples
    train_candidate_indices = generate_candidates_fast([str(x) if pd.notna(x) else "" for x in s1_train["business_name"]], [str(x) if pd.notna(x) else "" for x in target_train["business_name"]], max_df=5000, top=5)
    
    target_train_ids = target_train['entity_id'].values
    train_candidates = {}
    for i, s1_id in enumerate(s1_train['entity_id']):
        train_candidates[s1_id] = [target_train_ids[idx] for idx in train_candidate_indices[i]]
        
    gt_dict = gt.dropna(subset=['matched_entity_ids']).set_index('source1_entity_id')['matched_entity_ids'].to_dict()
    pairs_to_evaluate = []
    
    GLOBAL_TARGET_DICT = target_train.set_index('entity_id').to_dict('index')
    GLOBAL_S1_DICT = s1_train.set_index('entity_id').to_dict('index')
    
    for s1_id, candidates in train_candidates.items():
        gt_matches = set(gt_dict.get(s1_id, "").split(',')) if s1_id in gt_dict else set()
        for c_id in candidates:
            if c_id in GLOBAL_TARGET_DICT:
                label = 1 if c_id in gt_matches else 0
                pairs_to_evaluate.append((s1_id, c_id, label))
        for c_id in gt_matches:
            if c_id not in candidates and c_id in GLOBAL_TARGET_DICT:
                pairs_to_evaluate.append((s1_id, c_id, 1))
                
    GLOBAL_IDF_DICT, GLOBAL_DEFAULT_IDF = compute_idf(target_train)
    train_pairs_list = [(s1, c) for s1, c, _ in pairs_to_evaluate]
    labels = [l for _, _, l in pairs_to_evaluate]
    
    chunks = list(chunk_list(train_pairs_list, 10000))
    # Train using 6 cores
    with mp.Pool(6) as p:
        results = p.map(extract_features_worker, chunks)
        
    train_feats = []
    for r in results: train_feats.extend(r)
    train_df = pd.DataFrame(train_feats)
    train_df['label'] = labels
    GLOBAL_FEATURE_COLS = [c for c in train_df.columns if c not in ('source1_entity_id', 'candidate_entity_id', 'label')]
    
    GLOBAL_CLF = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=6)
    GLOBAL_CLF.fit(train_df[GLOBAL_FEATURE_COLS], train_df['label'])
    
    del s1_train, s2_train, s3_train, target_train, train_df, train_feats
    
    print("Loading test data...", flush=True)
    s1_test = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep='\t', dtype=str, keep_default_na=False)
    s2_test = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep='\t', dtype=str, keep_default_na=False)
    s3_test = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep='\t', dtype=str, keep_default_na=False)
    target_test = pd.concat([s2_test, s3_test], ignore_index=True)
    
    GLOBAL_IDF_DICT, GLOBAL_DEFAULT_IDF = compute_idf(target_test)
    GLOBAL_TARGET_DICT = target_test.set_index('entity_id').to_dict('index')
    GLOBAL_S1_DICT = s1_test.set_index('entity_id').to_dict('index')
    
    print("Generating top 1 test candidates...", flush=True)
    # TEST ONLY USES TOP=1 TO SAVE 80% TIME AND MEMORY!
    test_candidate_indices = generate_candidates_fast([str(x) if pd.notna(x) else "" for x in s1_test["business_name"]], [str(x) if pd.notna(x) else "" for x in target_test["business_name"]], max_df=5000, top=1)
    
    target_test_ids = target_test['entity_id'].values
    test_candidates = {}
    test_pairs_list = []
    for i, s1_id in enumerate(s1_test['entity_id']):
        cands = [target_test_ids[idx] for idx in test_candidate_indices[i]]
        test_candidates[s1_id] = cands
        for c_id in cands:
            test_pairs_list.append((s1_id, c_id))
            
    print(f"Extracting test features and predicting for {len(test_pairs_list)} pairs...", flush=True)
    chunks = list(chunk_list(test_pairs_list, 10000))
    # Predict using 6 cores to avoid OOM
    with mp.Pool(6) as p:
        results = p.map(predict_worker, chunks)
        
    matches = {}
    for r in results:
        for s1_id, c_id in r:
            matches.setdefault(s1_id, []).append(c_id)
            
    os.makedirs("output", exist_ok=True)
    with open("output/matching_results.tsv", "w") as f1, open("output/candidate_pairs.tsv", "w") as f2:
        f1.write("source1_entity_id\tmatched_entity_ids\n")
        f2.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_test['entity_id']:
            m = matches.get(s1_id, [])
            c = test_candidates.get(s1_id, [])
            f1.write(f"{s1_id}\t{','.join(m)}\n")
            f2.write(f"{s1_id}\t{','.join(c)}\n")
            
    print("Done!")
