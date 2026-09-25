import pandas as pd
import numpy as np
import difflib
import re
import string
import unicodedata

def normalize(text):
    """Normalize text by lowercasing, unicode folding, and stripping punctuation."""
    if pd.isna(text): 
        return ""
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def seq_ratio(a, b):
    """SequenceMatcher ratio, a proxy for Levenshtein/Jaro similarity."""
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def get_tokens(s):
    return set(s.split()) if s else set()

def jaccard(a, b):
    """Token-level Jaccard similarity."""
    set_a, set_b = get_tokens(a), get_tokens(b)
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def char_ngrams(s, n=3):
    """Extract character n-grams."""
    s = s.replace(" ", "")
    if len(s) < n: return set([s]) if s else set()
    return set(s[i:i+n] for i in range(len(s)-n+1))

def ngram_jaccard(a, b, n=3):
    """Character n-gram Jaccard similarity."""
    set_a, set_b = char_ngrams(a, n), char_ngrams(b, n)
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def run_analysis(s1_path, s2_path, gt_path, sample_size=10000):
    """
    Runs the pairwise feature generation and analysis on a sampled subset of the data.
    Generates true matches, random negatives, and hard negatives (shared tokens).
    """
    print("Loading data...")
    s1 = pd.read_csv(s1_path, sep='\t', nrows=sample_size)
    s2 = pd.read_csv(s2_path, sep='\t', nrows=sample_size * 4)
    gt = pd.read_csv(gt_path, sep='\t', nrows=sample_size)

    s1_dict = s1.set_index('entity_id').to_dict('index')
    s2_dict = s2.set_index('entity_id').to_dict('index')
    gt_dict = gt.set_index('source1_entity_id')['matched_entity_ids'].to_dict()

    print("Building inverted index for hard negatives...")
    name_index = {}
    addr_index = {}
    for s2_id, row in s2_dict.items():
        for t in get_tokens(normalize(row['business_name'])):
            if len(t) > 3: name_index.setdefault(t, []).append(s2_id)
        for t in get_tokens(normalize(row['business_address'])):
            if len(t) > 4: addr_index.setdefault(t, []).append(s2_id)

    pairs = []
    print("Generating positive and negative pairs...")
    
    # 1. Matches
    matches_count = 0
    for s1_id, row in s1_dict.items():
        if matches_count > 2000: break
        if s1_id not in gt_dict or pd.isna(gt_dict[s1_id]): continue
        for m in gt_dict[s1_id].split(','):
            if m in s2_dict:
                pairs.append((s1_id, m, 1, "match"))
                matches_count += 1
                break

    # 2. Random negatives
    random_s1 = np.random.choice(list(s1_dict.keys()), 1000)
    random_s2 = np.random.choice(list(s2_dict.keys()), 1000)
    for a, b in zip(random_s1, random_s2):
        pairs.append((a, b, 0, "random"))

    # 3. Hard negatives (Name)
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

    print(f"Generated {len(pairs)} pairs. Extracting features...")
    results = []
    for s1_id, s2_id, label, ntype in pairs:
        r1, r2 = s1_dict[s1_id], s2_dict[s2_id]
        
        n1, n2 = str(r1['business_name']), str(r2['business_name'])
        a1, a2 = str(r1['business_address']), str(r2['business_address'])
        c1, c2 = str(r1['country']), str(r2['country'])
        
        norm_n1, norm_n2 = normalize(n1), normalize(n2)
        norm_a1, norm_a2 = normalize(a1), normalize(a2)
        
        results.append({
            'label': label,
            'type': ntype,
            'country': c1,
            'same_country': int(c1 == c2),
            'norm_name_exact': int(norm_n1 == norm_n2),
            'norm_addr_exact': int(norm_a1 == norm_a2),
            'name_seq_ratio': seq_ratio(norm_n1, norm_n2),
            'addr_seq_ratio': seq_ratio(norm_a1, norm_a2),
            'name_jaccard': jaccard(norm_n1, norm_n2),
            'addr_jaccard': jaccard(norm_a1, norm_a2),
            'name_3gram': ngram_jaccard(norm_n1, norm_n2, 3),
            'addr_3gram': ngram_jaccard(norm_a1, norm_a2, 3),
            'addr_missing_A': int(a1 == 'nan'),
            'addr_missing_B': int(a2 == 'nan')
        })

    df = pd.DataFrame(results)
    
    print("\n=== OVERALL CORRELATION ===")
    print(df.drop(['type', 'country'], axis=1).corr()['label'].sort_values(ascending=False).round(3))

    print("\n=== FEATURE MEANS BY PAIR TYPE ===")
    cols = ['name_seq_ratio', 'name_jaccard', 'name_3gram', 'addr_seq_ratio', 'addr_jaccard']
    print(df.groupby(['label', 'type'])[cols].mean().round(3))
    
if __name__ == '__main__':
    run_analysis(
        'student_resource/dataset/train/train_source1.tsv',
        'student_resource/dataset/train/train_source2.tsv',
        'student_resource/dataset/train/train_ground_truth.tsv'
    )
