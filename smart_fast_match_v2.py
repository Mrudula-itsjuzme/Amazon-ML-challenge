import pandas as pd
import os

print("Loading data...", flush=True)
s1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep='\t', dtype=str).fillna("")
s2 = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep='\t', dtype=str).fillna("")
s3 = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep='\t', dtype=str).fillna("")
targets = pd.concat([s2, s3], ignore_index=True)

for df in [s1, targets]:
    df['name'] = df['business_name'].astype(str).str.lower().str.strip()
    df['addr'] = df['business_address'].astype(str).str.lower().str.strip()

print("Analyzing name frequencies...", flush=True)
name_counts = targets['name'].value_counts()
rare_names = set(name_counts[name_counts <= 3].index)

s1_rare = s1[s1['name'].isin(rare_names) & (s1['name'] != "")]
targets_rare = targets[targets['name'].isin(rare_names) & (targets['name'] != "")]

print("1. Matching rare names...", flush=True)
m1 = s1_rare.merge(targets_rare, on='name')
# Filter m1: if addresses are both present, they shouldn't strictly contradict
# Actually, for rare names, if the name is unique, we trust it anyway.

print("2. Matching exact addresses...", flush=True)
s1_addr = s1[s1['addr'].str.len() > 10]
targets_addr = targets[targets['addr'].str.len() > 10]
m2 = s1_addr.merge(targets_addr, on='addr')
# Filter m2: ensure names don't contradict
m2['name_ok'] = (m2['name_x'] == "") | (m2['name_y'] == "") | (m2['name_x'] == m2['name_y'])
m2 = m2[m2['name_ok']]

print("Concatenating matches...", flush=True)
matches = pd.concat([
    m1[['entity_id_x', 'entity_id_y']],
    m2[['entity_id_x', 'entity_id_y']]
]).drop_duplicates()

print("Formatting output...", flush=True)
matches_grouped = matches.groupby('entity_id_x')['entity_id_y'].apply(lambda x: ','.join(list(x)[:5])).reset_index()
matches_dict = dict(zip(matches_grouped['entity_id_x'], matches_grouped['entity_id_y']))

os.makedirs("output", exist_ok=True)
with open("output/matching_results_v4.tsv", "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1_id in s1['entity_id']:
        res = matches_dict.get(s1_id, "")
        f.write(f"{s1_id}\t{res}\n")

print("Done!", flush=True)
