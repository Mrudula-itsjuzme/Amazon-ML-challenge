import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import re

print("Loading train...")
s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep='\t', usecols=['entity_id', 'business_name'], nrows=50000)
s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep='\t', usecols=['entity_id', 'business_name'], nrows=100000)
s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep='\t', usecols=['entity_id', 'business_name'], nrows=100000)
targets = pd.concat([s2, s3], ignore_index=True)
gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep='\t').dropna()
gt_dict = gt.set_index('source1_entity_id')['matched_entity_ids'].to_dict()

# Baseline 1: Exact Match
s1['clean'] = s1['business_name'].astype(str).str.lower().str.strip()
targets['clean'] = targets['business_name'].astype(str).str.lower().str.strip()
merged = s1.dropna(subset=['clean']).merge(targets.dropna(subset=['clean']), on='clean', suffixes=('_s1', '_t'))
exact_matches = merged.groupby('entity_id_s1')['entity_id_t'].apply(set).to_dict()

def calc_f05(preds, gts):
    tp = 0
    fp = 0
    fn = 0
    for s1_id, gt_match_str in gts.items():
        gt_set = set(gt_match_str.split(','))
        pred_set = preds.get(s1_id, set())
        for p in pred_set:
            if p in gt_set: tp += 1
            else: fp += 1
        for g in gt_set:
            if g not in pred_set: fn += 1
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
    return p, r, f05

print("Exact Match:", calc_f05(exact_matches, gt_dict))


print("Vectorizing...")
vec = TfidfVectorizer(analyzer=lambda x: [t for t in str(x).lower().strip().split() if len(t)>=3], min_df=1, dtype=np.float32)
T = vec.fit_transform(targets['business_name'])
dfs = np.array(T.astype(bool).sum(axis=0)).flatten()
T.data[np.repeat(dfs > 500, T.getnnz(axis=0))] = 0.0
T.eliminate_zeros()

s1_texts = s1['business_name'].tolist()
q = vec.transform(s1_texts)

print("Scoring...")
scores_csr = (q @ T.T).tocsr()
tfidf_matches = {}
for row in range(scores_csr.shape[0]):
    s1_id = s1['entity_id'].iloc[row]
    lo, hi = scores_csr.indptr[row:row + 2]
    cols = scores_csr.indices[lo:hi]
    vals = scores_csr.data[lo:hi]
    if len(vals) > 0:
        # Get all above 0.4
        valid = vals > 0.4
        if valid.any():
            matched_cols = cols[valid]
            tfidf_matches[s1_id] = set(targets['entity_id'].iloc[matched_cols])

print("TFIDF (thresh 0.4):", calc_f05(tfidf_matches, gt_dict))
