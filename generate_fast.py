import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.sparse as sp
import time
import re
import multiprocessing as mp
import os

def normalize(text):
    if not isinstance(text, str): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

def analyzer(text):
    if not isinstance(text, str) or text == 'nan': return []
    norm = normalize(text)
    return [t for t in norm.split() if len(t) >= 3]

def top_sparse_scores(scores, count):
    for row in range(scores.shape[0]):
        lo, hi = scores.indptr[row:row + 2]
        columns = scores.indices[lo:hi]
        values = scores.data[lo:hi]
        if len(values) > count:
            idx = np.argpartition(values, -count)[-count:]
            yield [columns[i] for i in idx]
        else:
            idx = np.argsort(-values)
            yield [columns[i] for i in idx]

def process_chunk(start, end, s1_texts, vec, T_scaled, top):
    q = vec.transform(s1_texts[start:end])
    q.data = np.ones_like(q.data)
    scores_csr = (q @ T_scaled.T).tocsr()
    results = []
    for c_list in top_sparse_scores(scores_csr, top):
        results.append(c_list)
    return results

# We will modify hackathon_solution.py directly using sed to use this faster method!
