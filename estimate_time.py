import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.sparse as sp
import time
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

print("Loading data...")
s2_test = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep='\t', usecols=['business_name'])
s3_test = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep='\t', usecols=['business_name'])
target_test = pd.concat([s2_test, s3_test], ignore_index=True)

s1_test = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep='\t', usecols=['business_name']).head(4096)

print("Vectorizing...")
vec = CountVectorizer(analyzer=analyzer, min_df=1, dtype=np.float32)
T = vec.fit_transform([str(x) if pd.notna(x) else "" for x in target_test["business_name"]])
T.data = np.ones_like(T.data)
dfs = np.array(T.sum(axis=0)).flatten()
N = T.shape[0]
idfs = np.log((N + 1) / (dfs + 1))
idfs[dfs / N > 0.01] = 0.0
W = sp.diags(idfs)
T_scaled = T @ W

print("Sparsity analysis:")
print("T_scaled nnz:", T_scaled.nnz)

q = vec.transform([str(x) if pd.notna(x) else "" for x in s1_test["business_name"]])
q.data = np.ones_like(q.data)

start_t = time.time()
scores_csr = (q @ T_scaled.T).tocsr()
print("Dot product took:", time.time() - start_t)

start_t = time.time()
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

results = list(top_sparse_scores(scores_csr, 5))
print("Argpartition took:", time.time() - start_t)

