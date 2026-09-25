import pandas as pd
import numpy as np
import re
import string
import unicodedata
import difflib

def normalize(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

def has_non_latin(text):
    if pd.isna(text): return False
    return bool(re.search(r'[^\x00-\x7F]', str(text)))

def seq_ratio(a, b):
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def jaccard(a, b):
    set_a, set_b = set(str(a).split()), set(str(b).split())
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

print("Loading data for deep dive...")
s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep='\t', nrows=200000)
s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep='\t', nrows=500000)
s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep='\t', nrows=500000)
gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep='\t', nrows=200000)

s1_dict = s1.set_index('entity_id').to_dict('index')
s2_dict = s2.set_index('entity_id').to_dict('index')
s3_dict = s3.set_index('entity_id').to_dict('index')
gt_dict = gt.dropna(subset=['matched_entity_ids']).set_index('source1_entity_id')['matched_entity_ids'].to_dict()

print("Building true matches dataset...")
matches = []
for s1_id, matched_str in gt_dict.items():
    if s1_id not in s1_dict: continue
    for match_id in matched_str.split(','):
        source_flag = "S2" if match_id.startswith("S2-") else "S3"
        target_dict = s2_dict if source_flag == "S2" else s3_dict
        
        if match_id in target_dict:
            r1 = s1_dict[s1_id]
            r2 = target_dict[match_id]
            matches.append({
                's1_id': s1_id,
                's2_id': match_id,
                'source': source_flag,
                'n1': str(r1['business_name']),
                'n2': str(r2['business_name']),
                'a1': str(r1['business_address']),
                'a2': str(r2['business_address']),
                'c1': str(r1['country'])
            })

df = pd.DataFrame(matches)
print(f"Extracted {len(df)} true matches.")

# Compute base features for the matches
df['norm_n1'] = df['n1'].apply(normalize)
df['norm_n2'] = df['n2'].apply(normalize)
df['norm_a1'] = df['a1'].apply(normalize)
df['norm_a2'] = df['a2'].apply(normalize)

df['name_seq'] = df.apply(lambda x: seq_ratio(x['norm_n1'], x['norm_n2']), axis=1)
df['addr_seq'] = df.apply(lambda x: seq_ratio(x['norm_a1'], x['norm_a2']), axis=1)
df['name_jaccard'] = df.apply(lambda x: jaccard(x['norm_n1'], x['norm_n2']), axis=1)

df['n1_nonlatin'] = df['n1'].apply(has_non_latin)
df['n2_nonlatin'] = df['n2'].apply(has_non_latin)
df['cross_script'] = df['n1_nonlatin'] != df['n2_nonlatin']

print("\n=== 1. CROSS-SCRIPT PREVALENCE ===")
cross_script_rate = df['cross_script'].mean()
print(f"Cross-script true matches: {df['cross_script'].sum()} ({cross_script_rate*100:.2f}%)")
if cross_script_rate > 0:
    print(f"Average name similarity for cross-script matches: {df[df['cross_script']]['name_seq'].mean():.3f}")
    print(f"Average name similarity for same-script matches: {df[~df['cross_script']]['name_seq'].mean():.3f}")

print("\n=== 2. S2 vs S3 DIFFICULTY ===")
print("Matches found in S2 vs S3:")
print(df['source'].value_counts())

print("\nMissing Addresses in True Matches by Source:")
df['addr_missing_target'] = df['a2'] == 'nan'
print(df.groupby('source')['addr_missing_target'].mean().round(4) * 100, "%")

print("\nAverage Similarities by Source:")
print(df.groupby('source')[['name_seq', 'addr_seq', 'name_jaccard']].mean().round(3))

print("\n=== 3. CORRUPTION TAXONOMY (True Matches) ===")
def get_corruption_type(row):
    if row['n1'] == row['n2'] and row['a1'] == row['a2']:
        return "1. Exact Match (Raw)"
    if row['norm_n1'] == row['norm_n2'] and row['norm_a1'] == row['norm_a2']:
        return "2. Exact Match (Normalized/Punctuation Only)"
    if row['norm_n1'] == row['norm_n2'] and row['norm_a1'] != row['norm_a2']:
        return "3. Exact Name, Diff Address"
    if row['norm_n1'] != row['norm_n2'] and row['norm_a1'] == row['norm_a2']:
        return "4. Diff Name, Exact Address"
    if row['name_jaccard'] == 1.0 and row['norm_n1'] != row['norm_n2']:
        return "5. Word Reorder / Exact Token Match"
    if row['cross_script']:
        return "6. Transliteration / Cross-Script"
    if row['a1'] == 'nan' or row['a2'] == 'nan':
        return "7. Missing Address"
    return "8. General Fuzzy / Other"

df['corruption_type'] = df.apply(get_corruption_type, axis=1)
print(df['corruption_type'].value_counts(normalize=True).round(4) * 100, "%")

print("\n=== 4. WORST TRUE MATCHES (Bottom 1%) ===")
df['combined_sim'] = df['name_seq'] + df['addr_seq']
worst = df.sort_values('combined_sim').head(15)
for _, row in worst.iterrows():
    print(f"\nID: {row['s1_id']} vs {row['s2_id']} (Combined Sim: {row['combined_sim']:.3f})")
    print(f"Name 1 : {row['n1']}")
    print(f"Name 2 : {row['n2']}")
    print(f"Addr 1 : {row['a1']}")
    print(f"Addr 2 : {row['a2']}")
