import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random
import math
from feature_engineering import normalize, unidecode, extract_numbers, FeatureExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

print("Loading dataset...")
gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep='\t')
gt = gt.dropna(subset=['matched_entity_ids'])

# Set seed for reproducibility
random.seed(42)
s1_eval_ids_full = list(gt['source1_entity_id'].unique())
# Sample 10000 queries for training
s1_eval_ids_sampled = set(random.sample(s1_eval_ids_full, min(10000, len(s1_eval_ids_full))))

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
        if noise_added < noise_count:
            take = min(noise_count - noise_added, len(noise_df))
            kept.append(noise_df.head(take))
            noise_added += take
            
    return pd.concat(kept).drop_duplicates(subset=['entity_id'])

print("Loading S1...")
s1 = load_and_filter("student_resource/dataset/train/train_source1.tsv", s1_eval_ids_sampled, 0)
print("Loading S2...")
s2 = load_and_filter("student_resource/dataset/train/train_source2.tsv", required_s2s3, 50000)
print("Loading S3...")
s3 = load_and_filter("student_resource/dataset/train/train_source3.tsv", required_s2s3, 50000)

s1_dict = s1.set_index('entity_id').to_dict('index')
s2_dict = s2.set_index('entity_id').to_dict('index')
s3_dict = s3.set_index('entity_id').to_dict('index')

target_dict = {**s2_dict, **s3_dict}
target_ids = list(target_dict.keys())
print(f"Target pool size: {len(target_ids)}")

print("Computing IDF weights for features...")
def get_tokens(s):
    return set(s.split()) if s else set()

doc_freq = Counter()
addr_freq = Counter()
total_docs = len(s1_dict) + len(target_dict)
for row in s1_dict.values():
    doc_freq.update(get_tokens(normalize(row.get('business_name', ''))))
    addr_freq.update(get_tokens(normalize(row.get('business_address', ''))))
for row in target_dict.values():
    doc_freq.update(get_tokens(normalize(row.get('business_name', ''))))
    addr_freq.update(get_tokens(normalize(row.get('business_address', ''))))

idf_dict = {token: math.log(total_docs / (freq + 1)) for token, freq in doc_freq.items()}
addr_idf = {token: math.log(total_docs / (freq + 1)) for token, freq in addr_freq.items()}
default_idf = math.log(total_docs / 1)

def prep_corpus(doc_dict, ids, field, do_translit=False):
    corpus = []
    for _id in ids:
        val = str(doc_dict[_id].get(field, ''))
        if val == 'nan': val = ""
        if do_translit: val = unidecode(val).lower()
        corpus.append(normalize(val))
    return corpus

print("Building Routes...")
target_names_raw = prep_corpus(target_dict, target_ids, 'business_name')
target_addrs_raw = prep_corpus(target_dict, target_ids, 'business_address')

vectorizer_A = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
X_A = vectorizer_A.fit_transform(target_names_raw)
nn_A = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1).fit(X_A)

vectorizer_C = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2)
X_C = vectorizer_C.fit_transform(target_names_raw)
nn_C = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1).fit(X_C)

vectorizer_D = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
X_D = vectorizer_D.fit_transform(target_addrs_raw)
nn_D = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1).fit(X_D)

route_e_idx = defaultdict(list)
for _id, text in zip(target_ids, target_addrs_raw):
    for num in set(extract_numbers(text)):
        route_e_idx[num].append(_id)

for num in route_e_idx:
    if len(route_e_idx[num]) > 1000:
        route_e_idx[num] = route_e_idx[num][:1000]

print("Generating Candidate Pairs & Features...")
extractor = FeatureExtractor(name_idf_dict=idf_dict, addr_idf_dict=addr_idf, default_idf=default_idf)
dataset = []

s1_eval_ids = list(s1_dict.keys())
s1_names_raw = prep_corpus(s1_dict, s1_eval_ids, 'business_name')
s1_addrs_raw = prep_corpus(s1_dict, s1_eval_ids, 'business_address')

Q_A = vectorizer_A.transform(s1_names_raw)
D_A, I_A = nn_A.kneighbors(Q_A)

Q_C = vectorizer_C.transform(s1_names_raw)
D_C, I_C = nn_C.kneighbors(Q_C)

Q_D = vectorizer_D.transform(s1_addrs_raw)
D_D, I_D = nn_D.kneighbors(Q_D)

for idx, s1_id in enumerate(s1_eval_ids):
    gt_matches = set(m for m in gt_dict[s1_id].split(',') if m in target_dict)
    
    cand_A = [target_ids[i] for i in I_A[idx]]
    cand_C = [target_ids[i] for i in I_C[idx]]
    cand_D = [target_ids[i] for i in I_D[idx]]
    
    cand_E_counts = Counter()
    for num in set(extract_numbers(s1_addrs_raw[idx])):
        if num in route_e_idx:
            for c in route_e_idx[num]:
                cand_E_counts[c] += 1
    cand_E = [c for c, _ in cand_E_counts.most_common(50)]
    
    # Exclude route B as decided
    all_cands = set(cand_A + cand_C + cand_D + cand_E)
    
    # Make sure we evaluate all GT matches if they happen to be in the pool
    # Wait, if they are missed by candidate generation, we should NOT artificially inject them.
    # The blocker must drop them if it missed them! But we can inject hard negatives.
    # To keep it realistic, we just evaluate the candidates generated by the blocker.
    # Note: If GT match is not in all_cands, it gets dropped, simulating real production recall.
    
    # Downsample negative candidates to keep dataset balanced-ish
    cands_list = list(all_cands)
    random.shuffle(cands_list)
    # limit to max 100 candidates per S1
    if len(cands_list) > 100:
        # Prioritize keeping gt_matches if they are in candidates
        found_gts = [c for c in cands_list if c in gt_matches]
        others = [c for c in cands_list if c not in gt_matches]
        cands_list = found_gts + others[:100 - len(found_gts)]
        
    for m in cands_list:
        label = 1 if m in gt_matches else 0
        r1 = s1_dict[s1_id]
        r2 = target_dict[m]
        
        feats = extractor.extract_features(
            r1.get('business_name'), r2.get('business_name'),
            r1.get('business_address'), r2.get('business_address'),
            r1.get('country'), r2.get('country')
        )
        
        feats['source1_entity_id'] = s1_id
        feats['candidate_entity_id'] = m
        feats['label'] = label
        
        dataset.append(feats)

df = pd.DataFrame(dataset)
print(f"Generated {len(df)} rows. True matches: {df['label'].sum()}")
df.to_parquet("candidate_features.parquet", index=False)
print("Saved to candidate_features.parquet")
