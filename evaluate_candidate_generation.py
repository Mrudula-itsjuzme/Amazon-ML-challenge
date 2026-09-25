import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
from feature_engineering import normalize, unidecode, extract_numbers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import time
import sys
import random

print("Loading Ground Truth and sampling S1 queries...")
gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep='\t')
gt = gt.dropna(subset=['matched_entity_ids'])

random.seed(42)
s1_eval_ids_full = list(gt['source1_entity_id'].unique())
s1_eval_ids_sampled = set(random.sample(s1_eval_ids_full, min(5000, len(s1_eval_ids_full))))

required_s2s3 = set()
gt_dict = gt.set_index('source1_entity_id')['matched_entity_ids'].to_dict()
for s1_id in s1_eval_ids_sampled:
    for m in gt_dict[s1_id].split(','):
        required_s2s3.add(m)

print(f"Sampled {len(s1_eval_ids_sampled)} S1 entities, requiring {len(required_s2s3)} S2/S3 matches.")

def load_and_filter(filepath, required_set, noise_count):
    df_iter = pd.read_csv(filepath, sep='\t', chunksize=100000)
    kept = []
    noise_added = 0
    for chunk in df_iter:
        req_df = chunk[chunk['entity_id'].isin(required_set)]
        kept.append(req_df)
        
        noise_df = chunk[~chunk['entity_id'].isin(required_set)]
        if noise_added < noise_count and len(noise_df):
            take = min(noise_count - noise_added, len(noise_df))
            kept.append(noise_df.sample(n=take, random_state=42 + noise_added))
            noise_added += take
            
    return pd.concat(kept).drop_duplicates(subset=['entity_id'])

print("Loading S1...")
s1 = load_and_filter("student_resource/dataset/train/train_source1.tsv", s1_eval_ids_sampled, 0)
print("Loading S2...")
s2 = load_and_filter("student_resource/dataset/train/train_source2.tsv", required_s2s3, 100000)
print("Loading S3...")
s3 = load_and_filter("student_resource/dataset/train/train_source3.tsv", required_s2s3, 100000)

s1_dict = s1.set_index('entity_id').to_dict('index')
s2_dict = s2.set_index('entity_id').to_dict('index')
s3_dict = s3.set_index('entity_id').to_dict('index')

# Merge S2 and S3 into one target pool
target_dict = {**s2_dict, **s3_dict}
target_ids = list(target_dict.keys())
TARGET_POOL_SIZE = len(target_ids)
print(f"Target pool size: {TARGET_POOL_SIZE}")

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

print("Building Routes...")
target_names_raw = prep_corpus(target_dict, target_ids, 'business_name')
target_names_trans = prep_corpus(target_dict, target_ids, 'business_name', do_translit=True)
target_addrs_raw = prep_corpus(target_dict, target_ids, 'business_address')

# Route A: Raw Name char-TFIDF
vectorizer_A = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
X_A = vectorizer_A.fit_transform(target_names_raw)
nn_A = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1).fit(X_A)

# Route B: Translit Name char-TFIDF
vectorizer_B = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
X_B = vectorizer_B.fit_transform(target_names_trans)
nn_B = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1).fit(X_B)

# Route C: Raw Name Word-TFIDF (New: replaces token inverted index to handle generic suffix issues)
vectorizer_C = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2)
X_C = vectorizer_C.fit_transform(target_names_raw)
nn_C = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1).fit(X_C)

# Route D: Address char-TFIDF
vectorizer_D = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
X_D = vectorizer_D.fit_transform(target_addrs_raw)
nn_D = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1).fit(X_D)

# Route E: Numeric Address inverted index
route_e_idx = defaultdict(list)
for _id, text in zip(target_ids, target_addrs_raw):
    nums = extract_numbers(text)
    for num in set(nums):
        route_e_idx[num].append(_id)

# Only cap extremely frequent numbers (e.g. 0 or 1 that might appear universally) to 500
for num in route_e_idx:
    if len(route_e_idx[num]) > 500:
        route_e_idx[num] = route_e_idx[num][:500]

print("Evaluating Recall...")
s1_eval_ids = list(s1_dict.keys())
s1_names_raw = prep_corpus(s1_dict, s1_eval_ids, 'business_name')
s1_names_trans = prep_corpus(s1_dict, s1_eval_ids, 'business_name', do_translit=True)
s1_addrs_raw = prep_corpus(s1_dict, s1_eval_ids, 'business_address')

Q_A = vectorizer_A.transform(s1_names_raw)
D_A, I_A = nn_A.kneighbors(Q_A)

Q_B = vectorizer_B.transform(s1_names_trans)
D_B, I_B = nn_B.kneighbors(Q_B)

Q_C = vectorizer_C.transform(s1_names_raw)
D_C, I_C = nn_C.kneighbors(Q_C)

Q_D = vectorizer_D.transform(s1_addrs_raw)
D_D, I_D = nn_D.kneighbors(Q_D)

def has_non_latin(text):
    return bool(re.search(r'[^\x00-\x7F]', str(text)))

metrics = {
    'total': 0, 'recovered': 0,
    'recovered_route_top5': 0, 'recovered_route_top10': 0, 'recovered_route_top20': 0, 'recovered_union': 0,
    'cross_script_total': 0, 'cross_script_recovered': 0,
    'missing_addr_total': 0, 'missing_addr_recovered': 0,
    'india_total': 0, 'india_recovered': 0,
    'us_total': 0, 'us_recovered': 0,
    'route_contributions': defaultdict(int),
    'route_uniques': defaultdict(int),
    'cross_script_route_uniques': defaultdict(int),
    'total_candidates_generated': 0
}

worst_matches = []

for idx, s1_id in enumerate(s1_eval_ids):
    gt_matches = set(m for m in gt_dict[s1_id].split(',') if m in target_dict)
    
    cand_A = [target_ids[i] for i in I_A[idx]]
    cand_B = [target_ids[i] for i in I_B[idx]]
    cand_C = [target_ids[i] for i in I_C[idx]]
    cand_D = [target_ids[i] for i in I_D[idx]]
    
    cand_E_counts = Counter()
    for num in set(extract_numbers(s1_addrs_raw[idx])):
        if num in route_e_idx:
            for c in route_e_idx[num]:
                cand_E_counts[c] += 1
    cand_E = [c for c, _ in cand_E_counts.most_common(50)]
    
    all_cands = set(cand_A + cand_B + cand_C + cand_D + cand_E)
    metrics['total_candidates_generated'] += len(all_cands)
    
    n1_raw = str(s1_dict[s1_id]['business_name'])
    a1_raw = str(s1_dict[s1_id]['business_address'])
    c1_raw = normalize(str(s1_dict[s1_id].get('country', '')))
    n1_nonlatin = has_non_latin(n1_raw)
    a1_missing = pd.isna(a1_raw) or a1_raw == 'nan' or len(a1_raw) < 2
    
    for m in gt_matches:
        metrics['total'] += 1
        n2_raw = str(target_dict[m]['business_name'])
        a2_raw = str(target_dict[m]['business_address'])
        c2_raw = normalize(str(target_dict[m].get('country', '')))
        
        n2_nonlatin = has_non_latin(n2_raw)
        cross_script = n1_nonlatin != n2_nonlatin
        a2_missing = pd.isna(a2_raw) or a2_raw == 'nan' or len(a2_raw) < 2
        missing_addr = a1_missing or a2_missing
        
        is_india = (c1_raw == 'india' or c2_raw == 'india')
        is_us = (c1_raw == 'us' or c2_raw == 'us' or c1_raw == 'usa' or c2_raw == 'usa')
        
        if cross_script: metrics['cross_script_total'] += 1
        if missing_addr: metrics['missing_addr_total'] += 1
        if is_india: metrics['india_total'] += 1
        if is_us: metrics['us_total'] += 1
        
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
            if is_india: metrics['india_recovered'] += 1
            if is_us: metrics['us_recovered'] += 1
            
            metrics['recovered_union'] += 1
            if best_pos < 5: metrics['recovered_route_top5'] += 1
            if best_pos < 10: metrics['recovered_route_top10'] += 1
            if best_pos < 20: metrics['recovered_route_top20'] += 1
            
            hits = []
            if pos_A < 999: hits.append('Route A (Raw Name Char-TFIDF)')
            if pos_B < 999: hits.append('Route B (Translit Char-TFIDF)')
            if pos_C < 999: hits.append('Route C (Raw Name Word-TFIDF)')
            if pos_D < 999: hits.append('Route D (Address Char-TFIDF)')
            if pos_E < 999: hits.append('Route E (Numeric Address)')
            
            for h in hits:
                metrics['route_contributions'][h] += 1
                
            if len(hits) == 1:
                metrics['route_uniques'][hits[0]] += 1
                if cross_script:
                    metrics['cross_script_route_uniques'][hits[0]] += 1
                
        else:
            worst_matches.append({'s1': n1_raw, 's2': n2_raw, 'a1': a1_raw, 'a2': a2_raw})

print("\n=== RECALL EVALUATION RESULTS ===")
print(f"Total True Matches Evaluated: {metrics['total']}")
print(f"Union Candidate Recall (top-50 per retrieval route before union): {metrics['recovered_union'] / metrics['total'] * 100:.2f}%")
print(f"Recovered within top-20 of at least one route: {metrics['recovered_route_top20'] / metrics['total'] * 100:.2f}%")
print(f"Recovered within top-10 of at least one route: {metrics['recovered_route_top10'] / metrics['total'] * 100:.2f}%")
print(f"Recovered within top-5 of at least one route: {metrics['recovered_route_top5'] / metrics['total'] * 100:.2f}%")
print("Note: these route-top-k figures are NOT Recall@k of a globally ranked union.")

if metrics['cross_script_total'] > 0:
    print(f"\nCross-script Recall: {metrics['cross_script_recovered'] / metrics['cross_script_total'] * 100:.2f}% ({metrics['cross_script_total']} total)")
if metrics['missing_addr_total'] > 0:
    print(f"Missing-addr Recall: {metrics['missing_addr_recovered'] / metrics['missing_addr_total'] * 100:.2f}% ({metrics['missing_addr_total']} total)")
if metrics['india_total'] > 0:
    print(f"India Recall: {metrics['india_recovered'] / metrics['india_total'] * 100:.2f}% ({metrics['india_total']} total)")
if metrics['us_total'] > 0:
    print(f"US Recall: {metrics['us_recovered'] / metrics['us_total'] * 100:.2f}% ({metrics['us_total']} total)")

print("\n=== ROUTE ATTRIBUTION ===")
print(f"{'Route':<35} | {'% Recovered':>15} | {'Unique Recoveries':>20}")
for k, v in sorted(metrics['route_contributions'].items(), key=lambda x: -x[1]):
    pct = v / metrics['recovered'] * 100
    unique = metrics['route_uniques'].get(k, 0)
    print(f"{k:<35} | {pct:>14.1f}% | {unique:>20}")

print("\n=== CROSS-SCRIPT UNIQUE RECOVERIES ===")
for k, v in metrics['cross_script_route_uniques'].items():
    print(f"{k}: {v} unique rescues for cross-script pairs")

avg_cands = metrics['total_candidates_generated'] / len(s1_eval_ids)
print(f"\n=== EFFICIENCY ===")
print(f"Average candidates per query: {avg_cands:.1f}")
search_space_elim = (1 - (avg_cands / TARGET_POOL_SIZE)) * 100
print(f"Candidate reduction ratio: {search_space_elim:.4f}% (eliminated {TARGET_POOL_SIZE - avg_cands:.1f} candidates/query)")

print("\n=== SAMPLE MISSED MATCHES (Worst Cases) ===")
for m in worst_matches[:10]:
    print(f"Missed:\n  S1 Name: {m['s1']} | S2 Name: {m['s2']}")
    print(f"  S1 Addr: {m['a1']} | S2 Addr: {m['a2']}\n")
