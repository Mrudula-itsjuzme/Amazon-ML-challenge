import pandas as pd
import time
from collections import defaultdict
from feature_engineering import normalize

print("Loading...")
s1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep='\t', nrows=20000)
s2 = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep='\t', nrows=100000)

s2_raw_idx = defaultdict(list)
token_counts = defaultdict(int)
for row in s2['business_name']:
    raw_norm = normalize(row)
    for token in set(raw_norm.split()):
        if len(token) > 2: token_counts[token] += 1

valid_tokens = {t for t, c in token_counts.items() if c < len(s2) * 0.05} # max 5%

print("Building index...")
start = time.time()
for s2_id, row in zip(s2['entity_id'], s2['business_name']):
    raw_norm = normalize(row)
    for token in set(raw_norm.split()):
        if token in valid_tokens: s2_raw_idx[token].append(s2_id)
print(f"Index built in {time.time()-start:.2f}s")

print("Querying...")
start = time.time()
for n1_raw in s1['business_name']:
    n1_raw_norm = normalize(n1_raw)
    candidates = set()
    for token in set(n1_raw_norm.split()):
        if token in valid_tokens:
            candidates.update(s2_raw_idx[token])
print(f"Queried in {time.time()-start:.2f}s")
