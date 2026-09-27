import pandas as pd
import time
start = time.time()
print("Loading S2...")
s2 = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
print(f"Loaded S2 in {time.time()-start:.2f}s, shape {s2.shape}, memory: {s2.memory_usage(deep=True).sum()/1e9:.2f}GB")
