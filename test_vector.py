import sys
sys.path.append("code/business_entity_resolution")
import pandas as pd
import time
from importlib import import_module

m = import_module("01_pipeline")

print("Loading test data...")
s1 = m.read_source("student_resource/dataset/test/test_source1.tsv", 20000)
targets = [m.read_source(f"student_resource/dataset/test/test_source{i}.tsv", 100000) for i in (2, 3)]
target = pd.concat(targets, ignore_index=True)

print("Running vector_route...")
start = time.time()
gen = m.vector_route(s1["name_native"].tolist(), target["name_native"].tolist(), "char_wb", (3, 5), 1000, 12)
results = list(gen)
print(f"vector_route finished in {time.time()-start:.2f}s")
