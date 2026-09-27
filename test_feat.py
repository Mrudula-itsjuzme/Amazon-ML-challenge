import pandas as pd
import time
from feature_engineering import FeatureExtractor
import random

s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep='\t', nrows=10000)
s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep='\t', nrows=10000)

extractor = FeatureExtractor()
start = time.time()
for i in range(10000):
    extractor.extract_features(
        s1.iloc[i]['business_name'], s2.iloc[i]['business_name'],
        s1.iloc[i]['business_address'], s2.iloc[i]['business_address']
    )
print(f"10000 pairs in {time.time()-start:.2f}s")
