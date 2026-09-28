import pandas as pd
import os

print("Loading test data...")
s1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep='\t', dtype=str).fillna("")
s2 = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep='\t', dtype=str).fillna("")
s3 = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep='\t', dtype=str).fillna("")
targets = pd.concat([s2, s3], ignore_index=True)

# Normalize
for df in [s1, targets]:
    df['name_clean'] = df['business_name'].astype(str).str.lower().str.strip()
    df['addr_clean'] = df['business_address'].astype(str).str.lower().str.strip()

print("Merging exact names...")
merged_name = s1[s1['name_clean'] != ""].merge(targets[targets['name_clean'] != ""], on='name_clean', suffixes=('_s1', '_target'))

print("Filtering for high precision...")
matches_dict = {}

# If Name matches, only keep if Address doesn't contradict
merged_name['addr_ok'] = (merged_name['addr_clean_s1'] == "") | (merged_name['addr_clean_target'] == "") | (merged_name['addr_clean_s1'] == merged_name['addr_clean_target'])

valid_names = merged_name[merged_name['addr_ok']]
for s1_id, t_id in zip(valid_names['entity_id_s1'], valid_names['entity_id_target']):
    matches_dict.setdefault(s1_id, set()).add(t_id)

print("Merging exact addresses...")
merged_addr = s1[s1['addr_clean'] != ""].merge(targets[targets['addr_clean'] != ""], on='addr_clean', suffixes=('_s1', '_target'))

# If Address matches, only keep if Name doesn't contradict (Name empty or equal)
merged_addr['name_ok'] = (merged_addr['name_clean_s1'] == "") | (merged_addr['name_clean_target'] == "") | (merged_addr['name_clean_s1'] == merged_addr['name_clean_target'])
valid_addrs = merged_addr[merged_addr['name_ok']]
for s1_id, t_id in zip(valid_addrs['entity_id_s1'], valid_addrs['entity_id_target']):
    matches_dict.setdefault(s1_id, set()).add(t_id)

os.makedirs("output", exist_ok=True)
s1_all = s1['entity_id']

print("Writing TSVs...")
with open("output/matching_results.tsv", "w") as f1, open("output/candidate_pairs.tsv", "w") as f2:
    f1.write("source1_entity_id\tmatched_entity_ids\n")
    f2.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in s1_all:
        m = list(matches_dict.get(s1_id, set()))[:100]
        f1.write(f"{s1_id}\t{','.join(m)}\n")
        f2.write(f"{s1_id}\t{','.join(m)}\n")

print("Done!")
