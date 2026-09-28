import pandas as pd
import os

print("Loading data...", flush=True)
s1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep='\t', dtype=str).fillna("")
s2 = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep='\t', dtype=str).fillna("")
s3 = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep='\t', dtype=str).fillna("")
targets = pd.concat([s2, s3], ignore_index=True)

for df in [s1, targets]:
    df['name'] = df['business_name'].astype(str).str.lower().str.strip()
    df['city'] = df['city'].astype(str).str.lower().str.strip()
    df['addr'] = df['business_address'].astype(str).str.lower().str.strip()

print("Analyzing name frequencies...", flush=True)
name_counts = targets['name'].value_counts()
rare_names = set(name_counts[name_counts <= 5].index)

s1_rare = s1[s1['name'].isin(rare_names) & (s1['name'] != "")]
targets_rare = targets[targets['name'].isin(rare_names) & (targets['name'] != "")]

print("Matching rare names...", flush=True)
m1 = s1_rare.merge(targets_rare, on='name')

print("Matching frequent names (chains) by City...", flush=True)
s1_freq = s1[~s1['name'].isin(rare_names) & (s1['name'] != "")]
targets_freq = targets[~targets['name'].isin(rare_names) & (targets['name'] != "")]

s1_freq_city = s1_freq[s1_freq['city'] != ""]
targets_freq_city = targets_freq[targets_freq['city'] != ""]
m2 = s1_freq_city.merge(targets_freq_city, on=['name', 'city'])

print("Matching exact addresses...", flush=True)
s1_addr = s1[s1['addr'].str.len() > 10]
targets_addr = targets[targets['addr'].str.len() > 10]
m3 = s1_addr.merge(targets_addr, on='addr')

print("Concatenating matches...", flush=True)
matches = pd.concat([
    m1[['entity_id_x', 'entity_id_y']],
    m2[['entity_id_x', 'entity_id_y']],
    m3[['entity_id_x', 'entity_id_y']]
]).drop_duplicates()

print("Formatting output...", flush=True)
# Cap at 5 targets per S1 to maximize precision (prevents F-score drops)
matches_grouped = matches.groupby('entity_id_x')['entity_id_y'].apply(lambda x: ','.join(list(x)[:5])).reset_index()
matches_dict = dict(zip(matches_grouped['entity_id_x'], matches_grouped['entity_id_y']))

os.makedirs("output", exist_ok=True)
with open("output/matching_results_v3.tsv", "w") as f, open("output/candidate_pairs_v3.tsv", "w") as f2:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    f2.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in s1['entity_id']:
        res = matches_dict.get(s1_id, "")
        f.write(f"{s1_id}\t{res}\n")
        f2.write(f"{s1_id}\t{res}\n")

print("Done!", flush=True)
