import pandas as pd
import numpy as np
import re
import string
import unicodedata
import difflib
from collections import defaultdict
import random
import sys

try:
    from unidecode import unidecode
except ImportError:
    print("Please install unidecode: pip install unidecode")
    sys.exit(1)

def normalize(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

def extract_numbers(text):
    if pd.isna(text): return set()
    return set(re.findall(r'\d+', str(text)))

def jaccard(set_a, set_b):
    if not set_a and not set_b: return 0.0 # Or np.nan? 0 is safer
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def has_non_latin(text):
    if pd.isna(text): return False
    return bool(re.search(r'[^\x00-\x7F]', str(text)))

def seq_ratio(a, b):
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

print("Loading dataset...")
s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep='\t', nrows=200000)
s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep='\t', nrows=500000)
gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep='\t', nrows=200000)

s1_dict = s1.set_index('entity_id').to_dict('index')
s2_dict = s2.set_index('entity_id').to_dict('index')
gt_dict = gt.dropna(subset=['matched_entity_ids']).set_index('source1_entity_id')['matched_entity_ids'].to_dict()

print("Building inverted index for hard negatives...")
s2_idx = defaultdict(list)
for s2_id, row in s2_dict.items():
    name = normalize(row['business_name'])
    for token in set(name.split()):
        if len(token) > 2: # Skip very short words
            s2_idx[token].append(s2_id)

print("Sampling pairs...")
true_matches = []
hard_negatives = []

random.seed(42)

for s1_id, matched_str in gt_dict.items():
    if s1_id not in s1_dict: continue
    r1 = s1_dict[s1_id]
    n1_norm = normalize(r1['business_name'])
    
    gt_s2_matches = [m for m in matched_str.split(',') if m.startswith('S2-')]
    
    # 1. True Matches
    for m in gt_s2_matches:
        if m in s2_dict:
            true_matches.append({
                's1_id': s1_id, 's2_id': m, 'label': 1,
                'n1': r1['business_name'], 'n2': s2_dict[m]['business_name'],
                'a1': r1['business_address'], 'a2': s2_dict[m]['business_address']
            })
            
    # 2. Hard Negatives
    # Find s2_ids that share a token but are not in gt
    candidates = set()
    for token in set(n1_norm.split()):
        if len(token) > 2 and token in s2_idx:
            candidates.update(s2_idx[token])
            
    # Remove true matches from candidates
    candidates.difference_update(gt_s2_matches)
    
    if candidates:
        # Pick up to 2 hard negatives per positive
        neg_samples = random.sample(list(candidates), min(2, len(candidates)))
        for neg_id in neg_samples:
            hard_negatives.append({
                's1_id': s1_id, 's2_id': neg_id, 'label': 0,
                'n1': r1['business_name'], 'n2': s2_dict[neg_id]['business_name'],
                'a1': r1['business_address'], 'a2': s2_dict[neg_id]['business_address']
            })

df = pd.DataFrame(true_matches + hard_negatives)
print(f"Dataset: {len(true_matches)} True Matches, {len(hard_negatives)} Hard Negatives")

print("Extracting features...")
df['norm_n1'] = df['n1'].apply(normalize)
df['norm_n2'] = df['n2'].apply(normalize)
df['norm_a1'] = df['a1'].apply(normalize)
df['norm_a2'] = df['a2'].apply(normalize)

df['name_seq'] = df.apply(lambda x: seq_ratio(x['norm_n1'], x['norm_n2']), axis=1)
df['addr_seq'] = df.apply(lambda x: seq_ratio(x['norm_a1'], x['norm_a2']), axis=1)

# 1. Numeric token overlap
df['num_n1'] = df['n1'].apply(extract_numbers)
df['num_n2'] = df['n2'].apply(extract_numbers)
df['num_a1'] = df['a1'].apply(extract_numbers)
df['num_a2'] = df['a2'].apply(extract_numbers)

df['num_overlap_name'] = df.apply(lambda x: jaccard(x['num_n1'], x['num_n2']), axis=1)
df['num_overlap_addr'] = df.apply(lambda x: jaccard(x['num_a1'], x['num_a2']), axis=1)

print("\n--- 1. NUMERIC TOKEN OVERLAP ---")
for lbl, name in [(1, "True Matches"), (0, "Hard Negatives")]:
    sub = df[df['label'] == lbl]
    print(f"\n{name}:")
    print("Name Numeric Jaccard (mean):", sub['num_overlap_name'].mean())
    print("Addr Numeric Jaccard (mean):", sub['num_overlap_addr'].mean())
    print("Addr Numeric Jaccard (90th percentile):", np.percentile(sub['num_overlap_addr'], 90))

# 2. Script-aware features
df['non_latin_1'] = df['n1'].apply(has_non_latin)
df['non_latin_2'] = df['n2'].apply(has_non_latin)
df['cross_script'] = df['non_latin_1'] != df['non_latin_2']
df['same_script'] = ~df['cross_script']

print("\n--- 2. SCRIPT-AWARE FEATURES ---")
print("True Matches:")
print(f"Cross-script: {df[(df['label'] == 1) & df['cross_script']].shape[0]} / {df[df['label'] == 1].shape[0]}")
print("Hard Negatives:")
print(f"Cross-script: {df[(df['label'] == 0) & df['cross_script']].shape[0]} / {df[df['label'] == 0].shape[0]}")

# 3. Transliteration feasibility
print("\n--- 3. TRANSLITERATION FEASIBILITY (True Matches Only) ---")
cross_df = df[(df['label'] == 1) & df['cross_script']].copy()
if len(cross_df) > 0:
    cross_df['trans_n1'] = cross_df['n1'].apply(lambda x: unidecode(str(x)).lower() if pd.notna(x) else "")
    cross_df['trans_n2'] = cross_df['n2'].apply(lambda x: unidecode(str(x)).lower() if pd.notna(x) else "")
    cross_df['trans_name_seq'] = cross_df.apply(lambda x: seq_ratio(normalize(x['trans_n1']), normalize(x['trans_n2'])), axis=1)
    
    print(f"Original Name Seq Ratio (Cross-script): {cross_df['name_seq'].mean():.4f}")
    print(f"Transliterated Name Seq Ratio: {cross_df['trans_name_seq'].mean():.4f}")
    improvements = (cross_df['trans_name_seq'] - cross_df['name_seq']) > 0.1
    print(f"% where transliteration improved similarity by >0.1: {improvements.mean()*100:.1f}%")

# 4. Name stability vs address stability
print("\n--- 4. NAME VS ADDRESS STABILITY (Percentiles) ---")
percentiles = [10, 25, 50, 75, 90]
print("True Matches:")
print("Name Seq:", np.percentile(df[df['label'] == 1]['name_seq'], percentiles))
print("Addr Seq:", np.percentile(df[df['label'] == 1]['addr_seq'], percentiles))

print("\nHard Negatives:")
print("Name Seq:", np.percentile(df[df['label'] == 0]['name_seq'], percentiles))
print("Addr Seq:", np.percentile(df[df['label'] == 0]['addr_seq'], percentiles))

# 5. Worst true matches vs Hardest negatives
print("\n--- 5. WORST TRUE MATCHES vs HARDEST NEGATIVES ---")
df['combined_sim'] = df['name_seq'] + df['addr_seq']

worst_matches = df[df['label'] == 1].sort_values('combined_sim', ascending=True).head(5)
hardest_negatives = df[df['label'] == 0].sort_values('combined_sim', ascending=False).head(5)

print("\nWORST TRUE MATCHES:")
for i, row in worst_matches.iterrows():
    print(f"Sim: {row['combined_sim']:.2f} | N1: {row['n1']} | N2: {row['n2']} | A1: {row['a1']} | A2: {row['a2']}")
    
print("\nHARDEST NEGATIVES:")
for i, row in hardest_negatives.iterrows():
    print(f"Sim: {row['combined_sim']:.2f} | N1: {row['n1']} | N2: {row['n2']} | A1: {row['a1']} | A2: {row['a2']}")

