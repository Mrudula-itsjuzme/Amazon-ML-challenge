import re
import string
import unicodedata
import difflib
from collections import Counter
import math

try:
    from unidecode import unidecode
except ImportError:
    # Fallback if unidecode isn't available
    def unidecode(text):
        return text

try:
    from thefuzz import fuzz
except ImportError:
    class fuzz:
        @staticmethod
        def token_set_ratio(a, b):
            return 0.0

def normalize(text):
    if not isinstance(text, str) or text == 'nan': return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

def get_ngrams(text, n=3):
    if not text: return set()
    # character n-grams
    text = text.replace(" ", "")
    if len(text) < n: return {text}
    return set(text[i:i+n] for i in range(len(text)-n+1))

def jaccard(set_a, set_b):
    if not set_a and not set_b: return 0.0
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def tfidf_jaccard(set_a, set_b, idf_dict, default_idf=1.0):
    if not set_a and not set_b: return 0.0
    if not set_a or not set_b: return 0.0
    
    intersection = set_a & set_b
    union = set_a | set_b
    
    intersection_weight = sum(idf_dict.get(token, default_idf) for token in intersection)
    union_weight = sum(idf_dict.get(token, default_idf) for token in union)
    
    if union_weight == 0: return 0.0
    return intersection_weight / union_weight

def seq_ratio(a, b):
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def extract_numbers(text):
    if not text: return []
    return re.findall(r'\d+', text)

def has_non_latin(text):
    if not isinstance(text, str) or text == 'nan': return False
    return bool(re.search(r'[^\x00-\x7F]', text))

class FeatureExtractor:
    def __init__(self, name_idf_dict=None, addr_idf_dict=None, default_idf=1.0):
        self.name_idf = name_idf_dict or {}
        self.addr_idf = addr_idf_dict or {}
        self.default_idf = default_idf
        
    def extract_features(self, n1, n2, a1, a2, c1=None, c2=None):
        # 1. Normalization & Transliteration
        norm_n1 = normalize(n1)
        norm_n2 = normalize(n2)
        trans_n1 = unidecode(n1).lower() if isinstance(n1, str) else ""
        trans_n2 = unidecode(n2).lower() if isinstance(n2, str) else ""
        trans_n1 = normalize(trans_n1)
        trans_n2 = normalize(trans_n2)
        
        norm_a1 = normalize(a1)
        norm_a2 = normalize(a2)
        
        # 2. Core String Features
        raw_name_seq = seq_ratio(norm_n1, norm_n2)
        translit_name_seq = seq_ratio(trans_n1, trans_n2)
        
        raw_name_3gram = jaccard(get_ngrams(norm_n1, 3), get_ngrams(norm_n2, 3))
        translit_name_3gram = jaccard(get_ngrams(trans_n1, 3), get_ngrams(trans_n2, 3))
        
        name_token_set_ratio = fuzz.token_set_ratio(norm_n1, norm_n2) / 100.0
        
        addr_seq = seq_ratio(norm_a1, norm_a2)
        addr_char_3gram = jaccard(get_ngrams(norm_a1, 3), get_ngrams(norm_a2, 3))
        
        # 3. Numeric Overlap Features
        num_a1 = extract_numbers(norm_a1)
        num_a2 = extract_numbers(norm_a2)
        num_a1_set = set(num_a1)
        num_a2_set = set(num_a2)
        
        addr_numeric_jaccard = jaccard(num_a1_set, num_a2_set)
        
        shared_nums = num_a1_set & num_a2_set
        addr_numeric_shared_count = len(shared_nums)
        addr_numeric_any_overlap = int(addr_numeric_shared_count > 0)
        addr_numeric_longest_match = max((len(n) for n in shared_nums), default=0)
        
        postal_a1 = {n for n in num_a1 if 4 <= len(n) <= 6}
        postal_a2 = {n for n in num_a2 if 4 <= len(n) <= 6}
        addr_numeric_5_6digit_overlap = jaccard(postal_a1, postal_a2)
        
        # 4. Token / TF-IDF Features
        tokens_n1 = set(norm_n1.split())
        tokens_n2 = set(norm_n2.split())
        name_jaccard = jaccard(tokens_n1, tokens_n2)
        name_tfidf_overlap = tfidf_jaccard(tokens_n1, tokens_n2, self.name_idf, self.default_idf)
        
        tokens_a1 = set(norm_a1.split())
        tokens_a2 = set(norm_a2.split())
        addr_jaccard = jaccard(tokens_a1, tokens_a2)
        addr_tfidf_overlap = tfidf_jaccard(tokens_a1, tokens_a2, self.addr_idf, self.default_idf)
        
        # 5. Contextual / Interaction Flags
        name_a_nonlatin = int(has_non_latin(n1))
        name_b_nonlatin = int(has_non_latin(n2))
        cross_script = int(name_a_nonlatin != name_b_nonlatin)
        
        addr_missing = int(not norm_a1 or not norm_a2 or str(a1) == 'nan' or str(a2) == 'nan')
        both_addr_present = int(not addr_missing)
        
        same_country = 0
        if c1 and c2 and str(c1) != 'nan' and str(c2) != 'nan':
            same_country = int(normalize(str(c1)) == normalize(str(c2)))
        
        return {
            'raw_name_seq': raw_name_seq,
            'translit_name_seq': translit_name_seq,
            'raw_name_3gram': raw_name_3gram,
            'translit_name_3gram': translit_name_3gram,
            'name_token_set_ratio': name_token_set_ratio,
            'addr_seq': addr_seq,
            'addr_char_3gram': addr_char_3gram,
            
            'addr_numeric_jaccard': addr_numeric_jaccard,
            'addr_numeric_shared_count': addr_numeric_shared_count,
            'addr_numeric_any_overlap': addr_numeric_any_overlap,
            'addr_numeric_longest_match': addr_numeric_longest_match,
            'addr_numeric_5_6digit_overlap': addr_numeric_5_6digit_overlap,
            
            'name_jaccard': name_jaccard,
            'name_tfidf_overlap': name_tfidf_overlap,
            'addr_jaccard': addr_jaccard,
            'addr_tfidf_overlap': addr_tfidf_overlap,
            
            'name_a_nonlatin': name_a_nonlatin,
            'name_b_nonlatin': name_b_nonlatin,
            'cross_script': cross_script,
            'addr_missing': addr_missing,
            'both_addr_present': both_addr_present,
            'same_country': same_country
        }

if __name__ == "__main__":
    extractor = FeatureExtractor()
    features = extractor.extract_features(
        "Tech Vision Constructions Private Limited", "टेक विजन कंस्ट्रक्शंस प्राइवेट लिमिटेड",
        "A-1102, Floor 11Th, Plot 8Pt, T-5 Dioro, Mumbai, 400001", "MUMBAI, A-1102, FLOOR 11TH, 400001",
        "India", "IN"
    )
    print("Features extracted successfully:", len(features))
