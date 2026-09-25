import pandas as pd
import numpy as np
from collections import defaultdict
import re
from collections import Counter
from feature_engineering import normalize, unidecode, extract_numbers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import time

print("Loading dataset...")
# Load subset for evaluation
s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep='\t', nrows=10000)
s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep='\t', nrows=50000)
s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep='\t', nrows=50000)
gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep='\t', nrows=10000)

s1_dict = s1.set_index('entity_id').to_dict('index')
s2_dict = s2.set_index('entity_id').to_dict('index')
s3_dict = s3.set_index('entity_id').to_dict('index')
gt_dict = gt.dropna(subset=['matched_entity_ids']).set_index('source1_entity_id')['matched_entity_ids'].to_dict()

# Merge S2 and S3 into one target pool
target_dict = {**s2_dict, **s3_dict}
target_ids = list(target_dict.keys())

print(f"Target pool size: {len(target_ids)}")

def prep_corpus(doc_dict, ids, field, do_translit=False, n_grams=False):
    corpus = []
    for _id in ids:
        val = str(doc_dict[_id].get(field, ''))
        if val == 'nan': val = ""
        if do_translit:
            val = unidecode(val).lower()
        norm_val = normalize(val)
        if not n_grams:
            corpus.append(norm_val)
        else:
            corpus.append(norm_val.replace(" ", ""))
    return corpus

print("Building Route A: Raw Name char-TFIDF...")
target_names_raw = prep_corpus(target_dict, target_ids, 'business_name')
vectorizer_A = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
X_A = vectorizer_A.fit_transform(target_names_raw)
nn_A = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1).fit(X_A)

print("Building Route B: Translit Name char-TFIDF...")
target_names_trans = prep_corpus(target_dict, target_ids, 'business_name', do_translit=True)
vectorizer_B = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
X_B = vectorizer_B.fit_transform(target_names_trans)
nn_B = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1).fit(X_B)

print("Building Route C: Raw Name token inverted index...")
route_c_idx = defaultdict(list)
for _id, text in zip(target_ids, target_names_raw):
    for token in set(text.split()):
        if len(token) > 2:
            route_c_idx[token].append(_id)

# Cap list lengths to prevent explosion
for token in route_c_idx:
    if len(route_c_idx[token]) > 100:
        # Cap very generic tokens, or we could just prune them
        route_c_idx[token] = route_c_idx[token][:100]

print("Building Route D: Address char-TFIDF...")
target_addrs_raw = prep_corpus(target_dict, target_ids, 'business_address')
vectorizer_D = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
X_D = vectorizer_D.fit_transform(target_addrs_raw)
nn_D = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1).fit(X_D)

print("Building Route E: Numeric Address inverted index...")
route_e_idx = defaultdict(list)
for _id, text in zip(target_ids, target_addrs_raw):
    nums = extract_numbers(text)
    for num in set(nums):
        route_e_idx[num].append(_id)

for num in route_e_idx:
    if len(route_e_idx[num]) > 100:
        route_e_idx[num] = route_e_idx[num][:100]

print("Evaluating Recall...")

s1_eval_ids = [k for k in gt_dict.keys() if k in s1_dict] # evaluate on 1000 queries

s1_names_raw = prep_corpus(s1_dict, s1_eval_ids, 'business_name')
s1_names_trans = prep_corpus(s1_dict, s1_eval_ids, 'business_name', do_translit=True)
s1_addrs_raw = prep_corpus(s1_dict, s1_eval_ids, 'business_address')

Q_A = vectorizer_A.transform(s1_names_raw)
D_A, I_A = nn_A.kneighbors(Q_A)

Q_B = vectorizer_B.transform(s1_names_trans)
D_B, I_B = nn_B.kneighbors(Q_B)

Q_D = vectorizer_D.transform(s1_addrs_raw)
D_D, I_D = nn_D.kneighbors(Q_D)

def has_non_latin(text):
    return bool(re.search(r'[^\x00-\x7F]', str(text)))

metrics = {
    'total': 0, 'recovered': 0,
    'recovered_at_5': 0, 'recovered_at_10': 0, 'recovered_at_20': 0, 'recovered_at_50': 0,
    'cross_script_total': 0, 'cross_script_recovered': 0,
    'missing_addr_total': 0, 'missing_addr_recovered': 0,
    'route_contributions': defaultdict(int)
}

for idx, s1_id in enumerate(s1_eval_ids):
    gt_matches = set(m for m in gt_dict[s1_id].split(',') if m in target_dict)
    if not gt_matches: continue
    
    # Get candidates from A
    cand_A = [target_ids[i] for i in I_A[idx]]
    # Get candidates from B
    cand_B = [target_ids[i] for i in I_B[idx]]
    # Get candidates from D
    cand_D = [target_ids[i] for i in I_D[idx]]
    
    # Get candidates from C
    cand_C_counts = Counter()
    for token in set(s1_names_raw[idx].split()):
        if len(token) > 2 and token in route_c_idx:
            for c in route_c_idx[token]:
                cand_C_counts[c] += 1
    cand_C = [c for c, _ in cand_C_counts.most_common(50)]
    
    # Get candidates from E
    cand_E_counts = Counter()
    for num in set(extract_numbers(s1_addrs_raw[idx])):
        if num in route_e_idx:
            for c in route_e_idx[num]:
                cand_E_counts[c] += 1
    cand_E = [c for c, _ in cand_E_counts.most_common(50)]
    
    n1_raw = str(s1_dict[s1_id]['business_name'])
    a1_raw = str(s1_dict[s1_id]['business_address'])
    n1_nonlatin = has_non_latin(n1_raw)
    a1_missing = pd.isna(a1_raw) or a1_raw == 'nan'
    
    for m in gt_matches:
        metrics['total'] += 1
        n2_raw = str(target_dict[m]['business_name'])
        a2_raw = str(target_dict[m]['business_address'])
        n2_nonlatin = has_non_latin(n2_raw)
        cross_script = n1_nonlatin != n2_nonlatin
        a2_missing = pd.isna(a2_raw) or a2_raw == 'nan'
        missing_addr = a1_missing or a2_missing
        
        if cross_script: metrics['cross_script_total'] += 1
        if missing_addr: metrics['missing_addr_total'] += 1
        
        pos_A = cand_A.index(m) if m in cand_A else 999
        pos_B = cand_B.index(m) if m in cand_B else 999
        pos_C = cand_C.index(m) if m in cand_C else 999
        pos_D = cand_D.index(m) if m in cand_D else 999
        pos_E = cand_E.index(m) if m in cand_E else 999
        
        best_pos = min(pos_A, pos_B, pos_C, pos_D, pos_E)
        
        if best_pos < 999:
            metrics['recovered'] += 1
            if cross_script: metrics['cross_script_recovered'] += 1
            if missing_addr: metrics['missing_addr_recovered'] += 1
            
            if best_pos < 5: metrics['recovered_at_5'] += 1
            if best_pos < 10: metrics['recovered_at_10'] += 1
            if best_pos < 20: metrics['recovered_at_20'] += 1
            if best_pos < 50: metrics['recovered_at_50'] += 1
            
            # Determine which route found it (can be multiple, we log all that found it)
            if pos_A < 999: metrics['route_contributions']['Route A: Raw Name char-TFIDF'] += 1
            if pos_B < 999: metrics['route_contributions']['Route B: Translit Name char-TFIDF'] += 1
            if pos_C < 999: metrics['route_contributions']['Route C: Raw Name Tokens'] += 1
            if pos_D < 999: metrics['route_contributions']['Route D: Address char-TFIDF'] += 1
            if pos_E < 999: metrics['route_contributions']['Route E: Numeric Address'] += 1

print("\n=== RECALL EVALUATION RESULTS ===")
print(f"Total True Matches Evaluated: {metrics['total']}")
print(f"Overall Recall@50: {metrics['recovered'] / metrics['total'] * 100:.2f}%")
print(f"Recall@5: {metrics['recovered_at_5'] / metrics['total'] * 100:.2f}%")
print(f"Recall@10: {metrics['recovered_at_10'] / metrics['total'] * 100:.2f}%")
print(f"Recall@20: {metrics['recovered_at_20'] / metrics['total'] * 100:.2f}%")

if metrics['cross_script_total'] > 0:
    print(f"\nCross-script Recall: {metrics['cross_script_recovered'] / metrics['cross_script_total'] * 100:.2f}% ({metrics['cross_script_total']} total)")
if metrics['missing_addr_total'] > 0:
    print(f"Missing-addr Recall: {metrics['missing_addr_recovered'] / metrics['missing_addr_total'] * 100:.2f}% ({metrics['missing_addr_total']} total)")

print("\n=== ROUTE CONTRIBUTIONS (Hits) ===")
for k, v in sorted(metrics['route_contributions'].items(), key=lambda x: -x[1]):
    print(f"{k}: {v} / {metrics['recovered']} recovered matches")

