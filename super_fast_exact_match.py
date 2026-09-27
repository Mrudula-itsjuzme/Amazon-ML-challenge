import pandas as pd
import os

print("Loading test data...")
s1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep='\t', usecols=['entity_id', 'business_name'], dtype=str)
s2 = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep='\t', usecols=['entity_id', 'business_name'], dtype=str)
s3 = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep='\t', usecols=['entity_id', 'business_name'], dtype=str)
targets = pd.concat([s2, s3], ignore_index=True).dropna(subset=['business_name'])
s1 = s1.dropna(subset=['business_name'])

s1['business_name_clean'] = s1['business_name'].str.lower().str.strip()
targets['business_name_clean'] = targets['business_name'].str.lower().str.strip()

print("Merging...")
# Find exact matches
merged = s1.merge(targets, on='business_name_clean', suffixes=('_s1', '_target'))

# Group by source1
matches = merged.groupby('entity_id_s1')['entity_id_target'].apply(list).to_dict()

os.makedirs("output", exist_ok=True)
s1_all = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep='\t', usecols=['entity_id'], dtype=str)

print("Writing TSVs...")
with open("output/matching_results.tsv", "w") as f1, open("output/candidate_pairs.tsv", "w") as f2:
    f1.write("source1_entity_id\tmatched_entity_ids\n")
    f2.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in s1_all['entity_id']:
        m = matches.get(s1_id, [])
        m = list(set(m))[:5] # top 5
        f1.write(f"{s1_id}\t{','.join(m)}\n")
        f2.write(f"{s1_id}\t{','.join(m)}\n")
        
print("Done!")
