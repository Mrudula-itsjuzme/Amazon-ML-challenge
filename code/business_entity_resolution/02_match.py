"""Pair features, grouped threshold tuning, LightGBM matching, and test output."""
import argparse
from collections import Counter
from functools import lru_cache
import math
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rapidfuzz.fuzz import ratio
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("native_char", "native_word", "address_char", "numeric", "rare_token",
          "translit_char", "core_char")


@lru_cache(maxsize=300000)
def tokens(s):
    return frozenset(s.split())


@lru_cache(maxsize=300000)
def ngrams(s, n=3):
    s = s.replace(" ", "")
    return frozenset(s[i:i+n] for i in range(max(1, len(s)-n+1))) if s else frozenset()


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def weighted_overlap(a, b, df, population):
    common = a & b
    if not common:
        return 0.0
    weight = lambda t: math.log((population + 1) / (df.get(t, 0) + 1))
    return sum(map(weight, common)) / max(sum(map(weight, a | b)), 1e-9)


def pair_features(a, b, name_df, addr_df, num_df, population):
    na, nb = a.name_native, b.name_native
    ta, tb = a.name_translit, b.name_translit
    aa, ab = a.address_native, b.address_native
    nt_a, nt_b = tokens(na), tokens(nb)
    at_a, at_b = tokens(aa), tokens(ab)
    nu_a, nu_b = set(a.address_numbers), set(b.address_numbers)
    native = ratio(na, nb) / 100 if na and nb else 0.0
    translit = ratio(ta, tb) / 100 if ta and tb else 0.0
    address = ratio(aa, ab) / 100 if aa and ab else 0.0
    missing = int(not aa or not ab)
    cross = int(a.name_script != b.name_script and "unknown" not in (a.name_script, b.name_script))
    country_missing = int(not a.country_norm or not b.country_norm)
    shared_numbers = nu_a & nu_b
    return {
        "name_native_seq": native,
        "name_native_3gram": jac(ngrams(na), ngrams(nb)),
        "name_native_token": jac(nt_a, nt_b),
        "name_native_idf": weighted_overlap(nt_a, nt_b, name_df, population),
        "name_translit_seq": translit,
        "name_translit_3gram": jac(ngrams(ta), ngrams(tb)),
        "name_legal_seq": ratio(a.name_legal, b.name_legal) / 100 if a.name_legal and b.name_legal else 0.0,
        "name_core_seq": ratio(a.name_core, b.name_core) / 100 if a.name_core and b.name_core else 0.0,
        "name_length_ratio": min(len(na), len(nb)) / max(len(na), len(nb), 1),
        "name_prefix": int(bool(na and nb) and na[:4] == nb[:4]),
        "name_suffix": int(bool(na and nb) and na[-4:] == nb[-4:]),
        "name_containment": int(bool(nt_a and nt_b) and (nt_a <= nt_b or nt_b <= nt_a)),
        "name_token_count_gap": abs(a.name_tokens - b.name_tokens),
        "address_seq": address,
        "address_3gram": jac(ngrams(aa), ngrams(ab)),
        "address_token": jac(at_a, at_b),
        "address_rare": weighted_overlap(at_a, at_b, addr_df, population),
        "number_overlap": jac(nu_a, nu_b),
        "number_weighted": weighted_overlap(nu_a, nu_b, num_df, population),
        "postal_overlap": jac(set(a.address_postal), set(b.address_postal)),
        "number_conflict": int(bool(nu_a and nu_b) and not shared_numbers),
        "number_shared": len(shared_numbers),
        "address_missing": missing,
        "source_address_missing": int(not aa),
        "target_address_missing": int(not ab),
        "name_core_token": jac(tokens(a.name_core), tokens(b.name_core)),
        "name_translit_token": jac(tokens(ta), tokens(tb)),
        "name_native_when_target_address_missing": native * int(not ab),
        "name_core_when_target_address_missing":
            (ratio(a.name_core, b.name_core) / 100 if a.name_core and b.name_core else 0.0) * int(not ab),
        "name_translit_when_target_address_missing": translit * int(not ab),
        "country_same": int(not country_missing and a.country_norm == b.country_norm),
        "country_missing": country_missing,
        "country_conflict": int(not country_missing and a.country_norm != b.country_norm),
        "cross_script": cross,
        "both_addresses": 1 - missing,
        "strong_name_address": native * address,
        "strong_name_missing": native * missing,
        "translit_cross": translit * cross,
        "name_number_conflict": native * int(bool(nu_a and nu_b) and not shared_numbers),
        "address_weak_name": address * (1 - native),
    }


def build_features(pairs, source, target):
    source_map = {r.entity_id: r for r in source.itertuples(index=False)}
    target_map = {r.entity_id: r for r in target.itertuples(index=False)}
    name_df, addr_df, num_df = Counter(), Counter(), Counter()
    for row in target.itertuples():
        name_df.update(tokens(row.name_native))
        addr_df.update(tokens(row.address_native))
        num_df.update(set(row.address_numbers))
    population = len(target)
    rows = []
    for p in pairs.itertuples(index=False):
        a = source_map[p.source1_entity_id]
        b = target_map[p.candidate_entity_id]
        features = pair_features(a, b, name_df, addr_df, num_df, population)
        features["route_count"] = p.route_count
        for route in ROUTES:
            rank = getattr(p, f"{route}_rank", 0)
            features[f"{route}_present"] = int(rank > 0)
            features[f"{route}_rank"] = rank
            features[f"{route}_score"] = getattr(p, f"{route}_score", 0.0)
        rows.append(features)
    frame = pd.DataFrame(rows, dtype=np.float32)
    groups = pairs.source1_entity_id.reset_index(drop=True)
    frame["candidate_count"] = groups.map(groups.value_counts()).to_numpy(dtype=np.float32)
    for column in ("name_native_seq", "name_translit_seq", "address_seq",
                   "name_core_seq", "native_char_score", "address_char_score",
                   "numeric_score", "rare_token_score", "translit_char_score"):
        values = frame[column]
        highest = values.groupby(groups).transform("max")
        frame[column + "_to_best"] = highest - values
        # The second-best score is shared by all candidates for this source.
        top_two = values.groupby(groups).transform(
            lambda item: item.nlargest(2).iloc[-1] if len(item) > 1 else 0.0)
        frame[column + "_best_margin"] = highest - top_two
    return frame


def global_target_idf_features(pairs, source, target, df_dir):
    """Exact label-free token DF features from the complete S2/S3 target pool."""
    df_dir = Path(df_dir)
    manifest = json.loads((df_dir / "manifest.json").read_text())
    population = manifest["target_pool_size"]
    if population != 10_320_219 or manifest["labels_used"] is not False:
        raise ValueError("Global token DF must cover the unlabeled full target pool")
    name_df = dict(pd.read_parquet(df_dir / "name_df.parquet").itertuples(index=False, name=None))
    address_df = dict(pd.read_parquet(df_dir / "address_df.parquet").itertuples(index=False, name=None))
    source_map = {row.entity_id: row for row in source.itertuples(index=False)}
    target_map = {row.entity_id: row for row in target.itertuples(index=False)}

    def idf(token, df):
        return math.log((population + 1) / (df.get(token, 0) + 1)) + 1

    def weighted_jaccard(left, right, df):
        union = left | right
        if not union:
            return 0.0
        return sum(idf(token, df) for token in left & right) / sum(
            idf(token, df) for token in union)

    def largest_idf(values, df):
        return max((idf(token, df) for token in values), default=0.0)

    rows = []
    for pair in pairs.itertuples(index=False):
        a = source_map[pair.source1_entity_id]
        b = target_map[pair.candidate_entity_id]
        an, bn = tokens(a.name_native), tokens(b.name_native)
        aa, ba = tokens(a.address_native), tokens(b.address_native)
        rows.append({
            "global_name_idf_jaccard": weighted_jaccard(an, bn, name_df),
            "global_address_idf_jaccard": weighted_jaccard(aa, ba, address_df),
            "shared_name_max_idf": largest_idf(an & bn, name_df),
            "shared_address_max_idf": largest_idf(aa & ba, address_df),
            "source_only_name_max_idf": largest_idf(an - bn, name_df),
            "target_only_name_max_idf": largest_idf(bn - an, name_df),
            "source_only_address_max_idf": largest_idf(aa - ba, address_df),
            "target_only_address_max_idf": largest_idf(ba - aa, address_df),
        })
    return pd.DataFrame(rows, dtype=np.float32)


def parse_truth(path, ids=None):
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if ids is not None:
        gt = gt[gt.source1_entity_id.isin(ids)]
    return {r.source1_entity_id: set(filter(None, r.matched_entity_ids.split(",")))
            for r in gt.itertuples(index=False)}


def entity_f05(actual, predicted):
    if not actual and not predicted:
        return 1.0
    tp = len(actual & predicted)
    if not tp:
        return 0.0
    precision, recall = tp / len(predicted), tp / len(actual)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def score_policy(pairs, probabilities, truth, ids, threshold):
    selected = pairs.loc[np.asarray(probabilities) >= threshold]
    predictions = selected.groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
    scores = [entity_f05(truth.get(i, set()), predictions.get(i, set())) for i in ids]
    return float(np.mean(scores)) if scores else float("nan")


def tune_threshold(pairs, probabilities, truth, ids):
    # A small, fixed grid limits threshold selection noise on grouped cohorts.
    thresholds = (.50, .70, .85, .90, .95, .975, .99, .995, .999)
    values = [(score_policy(pairs, probabilities, truth, ids, t), t) for t in thresholds]
    return max(values, key=lambda row: (row[0], row[1]))


def model(seed=42):
    return LGBMClassifier(n_estimators=350, learning_rate=.05, num_leaves=31,
                          max_depth=-1, colsample_bytree=.8, subsample=.8,
                          random_state=seed, n_jobs=8, verbosity=-1)


def load_inputs(folder, split):
    return tuple(pd.read_parquet(folder / f"{split}_{kind}.parquet") for kind in ("s1", "target", "pairs"))


def write_output(ids, pairs, probabilities, threshold, path, field):
    chosen = pairs.loc[np.asarray(probabilities) >= threshold]
    mapping = chosen.groupby("source1_entity_id").candidate_entity_id.agg(lambda s: ",".join(s)).to_dict()
    pd.DataFrame({"source1_entity_id": ids,
                  field: [mapping.get(i, "") for i in ids]}).to_csv(path, sep="\t", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["train", "test"], required=True)
    p.add_argument("--input-dir", type=Path, default=ROOT / "output")
    p.add_argument("--data-dir", type=Path, default=ROOT / "student_resource/dataset")
    p.add_argument("--model", type=Path, default=ROOT / "output/matcher.joblib")
    args = p.parse_args()
    source, target, pairs = load_inputs(args.input_dir, args.mode)
    X = build_features(pairs, source, target)
    if args.mode == "train":
        truth = parse_truth(args.data_dir / "train/train_ground_truth.tsv", set(source.entity_id))
        y = np.fromiter((p.candidate_entity_id in truth.get(p.source1_entity_id, set())
                         for p in pairs.itertuples(index=False)), dtype=np.int8)
        if len(np.unique(y)) != 2:
            raise ValueError("Training candidates need both positive and negative pairs")
        split = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
        fit, cal = next(split.split(X, y, pairs.source1_entity_id))
        calibration_model = model()
        calibration_model.fit(X.iloc[fit], y[fit])
        probabilities = calibration_model.predict_proba(X.iloc[cal])[:, 1]
        ids = set(pairs.source1_entity_id.iloc[cal])
        metric, threshold = tune_threshold(pairs.iloc[cal], probabilities, truth, ids)
        final = model()
        final.fit(X, y)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": final, "threshold": threshold, "features": list(X.columns)}, args.model)
        print(f"calibration: entities={len(ids)} macro_f05={metric:.6f} threshold={threshold:.3f}; model={args.model}")
    else:
        payload = joblib.load(args.model)
        if list(X.columns) != payload["features"]:
            raise ValueError("Feature schema differs from trained model")
        probabilities = payload["model"].predict_proba(X)[:, 1]
        args.input_dir.mkdir(parents=True, exist_ok=True)
        write_output(source.entity_id, pairs, probabilities, payload["threshold"],
                     args.input_dir / "matching_results.tsv", "matched_entity_ids")
        write_output(source.entity_id, pairs, np.ones(len(pairs)), .5,
                     args.input_dir / "candidate_pairs.tsv", "candidate_entity_ids")
        print(f"Wrote output for {len(source)} Source 1 records")


if __name__ == "__main__":
    main()
