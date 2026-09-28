import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.sparse as sp

s2_test = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep='\t', usecols=['business_name'])
s3_test = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep='\t', usecols=['business_name'])
target_test = pd.concat([s2_test, s3_test], ignore_index=True)

def normalize(text):
    if not isinstance(text, str): return ""
    import re
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

def analyzer(text):
    if not isinstance(text, str) or text == 'nan': return []
    norm = normalize(text)
    return [t for t in norm.split() if len(t) >= 3]

vec = CountVectorizer(analyzer=analyzer, min_df=1, dtype=np.float32)
T = vec.fit_transform([str(x) if pd.notna(x) else "" for x in target_test["business_name"]])
T.data = np.ones_like(T.data)
dfs = np.array(T.sum(axis=0)).flatten()
idfs = np.log((T.shape[0] + 1) / (dfs + 1))
idfs[dfs > 500] = 0.0
W = sp.diags(idfs)
T_scaled = T @ W
print("max_df=500 nnz:", T_scaled.nnz)
