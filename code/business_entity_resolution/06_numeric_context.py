"""Label-free candidate numeric and source-context features for the K100 matcher."""
import re

import numpy as np
import pandas as pd


def numeric_context_features(pairs, base_features, source, target):
    sources = {r.entity_id: r for r in source.itertuples(index=False)}
    targets = {r.entity_id: r for r in target.itertuples(index=False)}
    unit_re = re.compile(r'\b(?:unit|suite|ste|apt|apartment|flat|room|shop|office)\s*(?:no\s*)?(\d+)')
    cache = {}

    def parts(row):
        if row.entity_id not in cache:
            numbers = re.findall(r'\d+', row.address_native)
            unit = unit_re.search(row.address_native)
            cache[row.entity_id] = (numbers[0] if numbers else '', unit.group(1) if unit else '', set(numbers))
        return cache[row.entity_id]

    rows = []
    for i, pair in enumerate(pairs.itertuples(index=False)):
        a, b = sources[pair.source1_entity_id], targets[pair.candidate_entity_id]
        premise_a, unit_a, numbers_a = parts(a)
        premise_b, unit_b, numbers_b = parts(b)
        both_premise = bool(premise_a and premise_b)
        both_unit = bool(unit_a and unit_b)
        rows.append({
            'premise_number_equal': int(both_premise and premise_a == premise_b),
            'premise_number_conflict': int(both_premise and premise_a != premise_b),
            'unit_number_equal': int(both_unit and unit_a == unit_b),
            'unit_number_conflict': int(both_unit and unit_a != unit_b),
            'number_containment': int(bool(numbers_a and numbers_b) and
                                      (numbers_a <= numbers_b or numbers_b <= numbers_a)),
            'number_relative_difference': (abs(int(premise_a) - int(premise_b)) /
                                           max(int(premise_a), int(premise_b), 1))
                                           if both_premise and len(premise_a) < 10 and len(premise_b) < 10 else 0.0,
            'source_name_tokens': float(a.name_tokens),
            'source_name_length': float(len(a.name_native)),
            'source_address_length': float(len(a.address_native)),
            'source_country_india': int(a.country_norm == 'india'),
            'target_country_missing': int(not b.country_norm),
            'native_when_address_missing': float(base_features.name_native_seq.iat[i]) * int(b.address_missing),
            'translit_when_cross_script': float(base_features.name_translit_seq.iat[i]) *
                                          int(a.name_script != b.name_script),
            'address_when_cross_script': float(base_features.address_seq.iat[i]) *
                                         int(a.name_script != b.name_script),
            'premise_equal_when_name_weak': int(both_premise and premise_a == premise_b) *
                                            int(float(base_features.name_native_seq.iat[i]) < .7),
        })
    return pd.DataFrame(rows, dtype=np.float32)
