import pandas as pd
import numpy as np
from collections import defaultdict
import random
from feature_engineering import normalize, unidecode, FeatureExtractor
from collections import Counter
import math
import sys

print("Loading dataset...")
# Load a subset for testing the pipeline
s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep='\t', nrows=50000)
s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep='\t', nrows=150000)
gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep='\t', nrows=50000)

s1_dict = s1.set_index('entity_id').to_dict('index')
s2_dict = s2.set_index('entity_id').to_dict('index')
gt_dict = gt.dropna(subset=['matched_entity_ids']).set_index('source1_entity_id')['matched_entity_ids'].to_dict()

# 1. Calculate IDF weights
print("Computing IDF weights...")
def get_tokens(s):
    return set(s.split()) if s else set()

doc_freq = Counter()
total_docs = len(s1_dict) + len(s2_dict)
for row in s1_dict.values():
    doc_freq.update(get_tokens(normalize(row['business_name'])))
for row in s2_dict.values():
    doc_freq.update(get_tokens(normalize(row['business_name'])))

idf_dict = {token: math.log(total_docs / (freq + 1)) for token, freq in doc_freq.items()}
default_idf = math.log(total_docs / 1)

# 2. Build multi-route inverted indices for S2
print("Building multi-route inverted index...")
s2_raw_idx = defaultdict(list)
s2_trans_idx = defaultdict(list)

for s2_id, row in s2_dict.items():
    raw_norm = normalize(row['business_name'])
    trans_norm = normalize(unidecode(str(row['business_name'])).lower()) if pd.notna(row['business_name']) else ""
    
    for token in set(raw_norm.split()):
        if len(token) > 2: s2_raw_idx[token].append(s2_id)
        
    for token in set(trans_norm.split()):
        if len(token) > 2: s2_trans_idx[token].append(s2_id)

print("Generating candidates and extracting features...")
extractor = FeatureExtractor(name_idf_dict=idf_dict, default_idf=default_idf)
dataset = []

random.seed(42)

for s1_id, matched_str in gt_dict.items():
    if s1_id not in s1_dict: continue
    r1 = s1_dict[s1_id]
    
    n1_raw = str(r1['business_name'])
    n1_trans = unidecode(n1_raw).lower()
    
    n1_raw_norm = normalize(n1_raw)
    n1_trans_norm = normalize(n1_trans)
    
    a1 = str(r1['business_address'])
    
    gt_s2_matches = [m for m in matched_str.split(',') if m.startswith('S2-')]
    
    candidates = set()
    
    # Route 1: Raw Name overlap
    for token in set(n1_raw_norm.split()):
        if len(token) > 2 and token in s2_raw_idx:
            candidates.update(s2_raw_idx[token])
            
    # Route 2: Transliterated Name overlap
    for token in set(n1_trans_norm.split()):
        if len(token) > 2 and token in s2_trans_idx:
            candidates.update(s2_trans_idx[token])
            
    # Remove true matches from generic candidates, then downsample to max 5 hard negatives
    candidates.difference_update(gt_s2_matches)
    
    neg_samples = random.sample(list(candidates), min(5, len(candidates)))
    
    # Assemble final pairs (True matches + Hard negatives)
    pairs_to_evaluate = [(m, 1) for m in gt_s2_matches if m in s2_dict] + [(m, 0) for m in neg_samples]
    
    for s2_id, label in pairs_to_evaluate:
        r2 = s2_dict[s2_id]
        n2 = str(r2['business_name'])
        a2 = str(r2['business_address'])
        
        feats = extractor.extract_features(n1_raw, n2, a1, a2)
        feats['s1_id'] = s1_id
        feats['s2_id'] = s2_id
        feats['label'] = label
        
        dataset.append(feats)

df = pd.DataFrame(dataset)
print(f"\nGenerated {len(df)} candidate pairs ({df['label'].sum()} True Matches, {len(df) - df['label'].sum()} Hard Negatives)")
print("\nFeature Columns:")
print(df.columns.tolist())

print("\nSample Output:")
print(df.head(2).T)

df.to_csv("feature_dataset.csv", index=False)
print("Saved to feature_dataset.csv")
