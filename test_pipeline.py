import sys
sys.path.append("code/business_entity_resolution")
import pandas as pd
from importlib import import_module
import time

m = import_module("01_pipeline")

print("Loading test data...")
s1 = m.read_source("student_resource/dataset/test/test_source1.tsv", 20000)
targets = [m.read_source(f"student_resource/dataset/test/test_source{i}.tsv", 50000) for i in (2, 3)]
target = pd.concat(targets, ignore_index=True)

print("Running pipeline...")
start = time.time()
pairs = m.generate(s1, target, batch=1000, route_top=12, cap=60)
print(f"Generated pairs in {time.time()-start:.2f}s")
