import pandas as pd
import numpy as np
import re
import string
import unicodedata
from collections import Counter
import math

def normalize(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

def get_tokens(s):
    return set(s.split()) if s else set()

def extract_pincode(address):
    """Extracts 5 or 6 digit postal codes commonly found in US and India."""
    if pd.isna(address): return None
    matches = re.findall(r'\b\d{5,6}\b', str(address))
    return matches[-1] if matches else None

NAME_STOPWORDS = {"private", "limited", "pvt", "ltd", "llc", "inc", "corporation", "co", "and", "the", "of", "group"}
ADDR_STOPWORDS = {"road", "street", "floor", "no", "near", "opposite", "opp", "block", "phase", "sector"}

def remove_stopwords(tokens, stopwords):
    return {t for t in tokens if t not in stopwords}

def jaccard(set_a, set_b):
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

print("Loading data for advanced analysis...")
s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep='\t', nrows=10000)
s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep='\t', nrows=50000)
gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep='\t', nrows=10000)

s1_dict = s1.set_index('entity_id').to_dict('index')
s2_dict = s2.set_index('entity_id').to_dict('index')
gt_dict = gt.set_index('source1_entity_id')['matched_entity_ids'].to_dict()

# Calculate Inverse Document Frequency (IDF) for names
print("Computing IDF weights...")
doc_freq = Counter()
total_docs = len(s1_dict)
for row in s1_dict.values():
    tokens = get_tokens(normalize(row['business_name']))
    doc_freq.update(tokens)

idf = {token: math.log(total_docs / (freq + 1)) for token, freq in doc_freq.items()}
default_idf = math.log(total_docs / 1)

def tfidf_jaccard(set_a, set_b):
    if not set_a or not set_b: return 0.0
    intersection_weight = sum(idf.get(t, default_idf) for t in set_a & set_b)
    union_weight = sum(idf.get(t, default_idf) for t in set_a | set_b)
    return intersection_weight / union_weight if union_weight > 0 else 0.0

print("Generating pairs...")
pairs = []
name_index = {}
for s2_id, row in s2_dict.items():
    for t in get_tokens(normalize(row['business_name'])):
        if len(t) > 3: name_index.setdefault(t, []).append(s2_id)

# 1. Matches
matches_count = 0
for s1_id, row in s1_dict.items():
    if matches_count > 1000: break
    if s1_id not in gt_dict or pd.isna(gt_dict[s1_id]): continue
    for m in gt_dict[s1_id].split(','):
        if m in s2_dict:
            pairs.append((s1_id, m, 1, "match"))
            matches_count += 1
            break

# 2. Hard Negatives (Name)
hard_name = 0
for s1_id in list(s1_dict.keys())[:2000]:
    if hard_name > 1000: break
    found = False
    for t in get_tokens(normalize(s1_dict[s1_id]['business_name'])):
        if len(t) > 3 and t in name_index:
            for c in name_index[t]:
                if pd.isna(gt_dict.get(s1_id)) or c not in gt_dict[s1_id]:
                    pairs.append((s1_id, c, 0, "hard_name"))
                    hard_name += 1
                    found = True
                    break
        if found: break

# 3. Random Negatives
random_s1 = np.random.choice(list(s1_dict.keys()), 500)
random_s2 = np.random.choice(list(s2_dict.keys()), 500)
for a, b in zip(random_s1, random_s2):
    pairs.append((a, b, 0, "random"))

print(f"Generated {len(pairs)} pairs. Extracting advanced features...")
results = []
for s1_id, s2_id, label, ntype in pairs:
    r1, r2 = s1_dict[s1_id], s2_dict.get(s2_id)
    if not r2: continue
    
    n1, n2 = normalize(r1['business_name']), normalize(r2['business_name'])
    a1, a2 = str(r1['business_address']), str(r2['business_address'])
    
    t1, t2 = get_tokens(n1), get_tokens(n2)
    t1_nostop = remove_stopwords(t1, NAME_STOPWORDS)
    t2_nostop = remove_stopwords(t2, NAME_STOPWORDS)
    
    p1, p2 = extract_pincode(a1), extract_pincode(a2)
    pincode_match = 1 if p1 and p2 and p1 == p2 else (0 if p1 and p2 else np.nan)
    
    results.append({
        'label': label,
        'type': ntype,
        'pincode_match': pincode_match,
        'name_jaccard_raw': jaccard(t1, t2),
        'name_jaccard_nostop': jaccard(t1_nostop, t2_nostop),
        'name_jaccard_tfidf': tfidf_jaccard(t1, t2),
    })

df = pd.DataFrame(results)

print("\n=== ABLATION: RAW vs NO-STOPWORDS vs TF-IDF JACCARD ===")
cols = ['name_jaccard_raw', 'name_jaccard_nostop', 'name_jaccard_tfidf']
print(df.groupby(['label', 'type'])[cols].mean().round(3))

print("\n=== PINCODE EXTRACTION ANALYSIS ===")
pin_stats = df.groupby(['label', 'type'])['pincode_match'].agg(['mean', 'count'])
pin_stats.columns = ['match_rate_when_present', 'pairs_with_both_pincodes']
print(pin_stats.round(3))

print("\n=== FEATURE CORRELATION GAIN ===")
print(df[['label', 'name_jaccard_raw', 'name_jaccard_nostop', 'name_jaccard_tfidf']].corr()['label'].sort_values(ascending=False).round(3))
