import argparse
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

BETA = 0.5


def parse_ids(x):
    if pd.isna(x) or str(x).strip() == "":
        return set()
    return {v.strip() for v in str(x).split(",") if v.strip()}


def entity_f05(true_ids, pred_ids):
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids or not pred_ids:
        return 0.0

    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    beta2 = BETA ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f05(df, gt_map, threshold, s1_ids):
    chosen = df[df["prob"] >= threshold]
    pred_map = chosen.groupby("source1_entity_id")["candidate_entity_id"].agg(lambda x: set(x)).to_dict()

    scores = []
    for s1 in s1_ids:
        scores.append(entity_f05(gt_map.get(s1, set()), pred_map.get(s1, set())))

    return float(np.mean(scores)) if scores else np.nan


def tune_threshold(cal_df, gt_map, cal_s1_ids):
    thresholds = np.arange(0.05, 0.976, 0.025)
    rows = []

    for threshold in thresholds:
        rows.append({
            "threshold": threshold,
            "macro_f05": macro_f05(cal_df, gt_map, threshold, cal_s1_ids),
        })

    results = pd.DataFrame(rows)
    best = results.loc[results["macro_f05"].idxmax()]
    return float(best["threshold"]), results


def build_entity_metadata(df):
    """Build slice membership from TRUE candidate rows only.

    This avoids the old bug where an entity entered the cross-script or
    missing-address slice merely because one of its many negative candidates
    happened to have that property.
    """
    positive = df[df["label"] == 1].copy()
    meta = pd.DataFrame(index=pd.Index(df["source1_entity_id"].unique(), name="source1_entity_id"))

    if "cross_script" in positive.columns:
        x = positive.groupby("source1_entity_id")["cross_script"].max()
        meta["true_cross_script"] = x.reindex(meta.index, fill_value=0).astype(int)

    if "addr_missing" in positive.columns:
        x = positive.groupby("source1_entity_id")["addr_missing"].max()
        meta["true_missing_address"] = x.reindex(meta.index, fill_value=0).astype(int)
    elif "both_addr_present" in positive.columns:
        x = positive.groupby("source1_entity_id")["both_addr_present"].min()
        meta["true_missing_address"] = (1 - x.reindex(meta.index, fill_value=1)).astype(int)

    country_col = None
    for candidate in ["s1_country", "source1_country", "country"]:
        if candidate in df.columns:
            country_col = candidate
            break

    if country_col:
        country = df.groupby("source1_entity_id")[country_col].first()
        meta["country"] = country.reindex(meta.index)

    return meta


def evaluate_slices(df, gt_map, threshold, s1_ids, entity_meta):
    s1_ids = set(s1_ids)
    rows = []

    def add_slice(name, ids):
        ids = set(ids) & s1_ids
        if not ids:
            return
        rows.append({
            "slice": name,
            "entities": len(ids),
            "macro_f05": macro_f05(df, gt_map, threshold, ids),
        })

    add_slice("all", s1_ids)

    if "true_cross_script" in entity_meta.columns:
        cross = entity_meta.index[entity_meta["true_cross_script"] == 1]
        same = entity_meta.index[entity_meta["true_cross_script"] == 0]
        add_slice("cross_script_true_match", cross)
        add_slice("same_script_true_match", same)

    if "true_missing_address" in entity_meta.columns:
        missing = entity_meta.index[entity_meta["true_missing_address"] == 1]
        present = entity_meta.index[entity_meta["true_missing_address"] == 0]
        add_slice("missing_address_true_match", missing)
        add_slice("address_present_true_match", present)

    if "country" in entity_meta.columns:
        for country in sorted(entity_meta["country"].dropna().astype(str).unique()):
            ids = entity_meta.index[entity_meta["country"].astype(str) == country]
            add_slice(f"country_{country}", ids)

    return pd.DataFrame(rows)


def print_dataset_diagnostics(pairs):
    print("\n===== DATASET DIAGNOSTICS =====")
    print("Rows:", len(pairs))
    print("S1 entities:", pairs["source1_entity_id"].nunique())
    print("Positive pairs:", int((pairs["label"] == 1).sum()))
    print("Negative pairs:", int((pairs["label"] == 0).sum()))

    if "same_country" in pairs.columns:
        pos = pairs.loc[pairs["label"] == 1, "same_country"].mean()
        neg = pairs.loc[pairs["label"] == 0, "same_country"].mean()
        print(f"same_country rate, positives: {pos:.4f}")
        print(f"same_country rate, negatives: {neg:.4f}")

    positive = pairs[pairs["label"] == 1]
    if "cross_script" in positive.columns:
        print(f"True-pair cross-script rate: {positive['cross_script'].mean():.4f}")
    if "addr_missing" in positive.columns:
        print(f"True-pair missing-address rate: {positive['addr_missing'].mean():.4f}")


def main(args):
    print("Loading data...")

    if args.features.endswith(".parquet"):
        pairs = pd.read_parquet(args.features)
    else:
        pairs = pd.read_csv(args.features)

    pairs["source1_entity_id"] = pairs["source1_entity_id"].astype(str)
    pairs["candidate_entity_id"] = pairs["candidate_entity_id"].astype(str)

    gt = pd.read_csv(args.ground_truth, sep="\t")
    gt["source1_entity_id"] = gt["source1_entity_id"].astype(str)
    gt["true_ids"] = gt["matched_entity_ids"].apply(parse_ids)
    gt_map = dict(zip(gt["source1_entity_id"], gt["true_ids"]))

    print_dataset_diagnostics(pairs)
    entity_meta = build_entity_metadata(pairs)

    features = [
        "norm_name_exact",
        "norm_addr_exact",
        "raw_name_seq",
        "translit_name_seq",
        "raw_name_3gram",
        "translit_name_3gram",
        "name_token_set_ratio",
        "name_jaccard",
        "name_tfidf_overlap",
        "addr_seq",
        "addr_char_3gram",
        "addr_jaccard",
        "addr_tfidf_overlap",
        "addr_numeric_jaccard",
        "addr_numeric_shared_count",
        "addr_numeric_any_overlap",
        "addr_numeric_longest_match",
        "addr_numeric_5_6digit_overlap",
        "name_a_nonlatin",
        "name_b_nonlatin",
        "cross_script",
        "addr_missing",
        "both_addr_present",
        "same_country",
    ]
    features = [f for f in features if f in pairs.columns]

    print(f"\nUsing {len(features)} features:")
    print(features)

    X = pairs[features]
    y = pairs["label"]
    groups = pairs["source1_entity_id"]

    outer_cv = GroupKFold(n_splits=args.folds)
    fold_results = []
    slice_results = []

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        print(f"\n===== Fold {fold} =====")

        train_df = pairs.iloc[train_idx].copy()
        val_df = pairs.iloc[val_idx].copy()

        inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42 + fold)
        fit_idx, cal_idx = next(inner.split(train_df, train_df["label"], train_df["source1_entity_id"]))

        fit_df = train_df.iloc[fit_idx].copy()
        cal_df = train_df.iloc[cal_idx].copy()

        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42 + fold,
            n_jobs=-1,
        )

        model.fit(fit_df[features], fit_df["label"])

        cal_df["prob"] = model.predict_proba(cal_df[features])[:, 1]
        val_df["prob"] = model.predict_proba(val_df[features])[:, 1]

        cal_s1_ids = set(cal_df["source1_entity_id"])
        val_s1_ids = set(val_df["source1_entity_id"])

        threshold, threshold_table = tune_threshold(cal_df, gt_map, cal_s1_ids)
        val_score = macro_f05(val_df, gt_map, threshold, val_s1_ids)

        print(f"Threshold: {threshold:.3f}")
        print(f"Validation macro F0.5: {val_score:.6f}")

        fold_results.append({
            "fold": fold,
            "threshold": threshold,
            "macro_f05": val_score,
            "train_entities": train_df["source1_entity_id"].nunique(),
            "val_entities": len(val_s1_ids),
            "val_pairs": len(val_df),
        })

        fold_slices = evaluate_slices(val_df, gt_map, threshold, val_s1_ids, entity_meta)
        fold_slices["fold"] = fold
        slice_results.append(fold_slices)

        threshold_table.to_csv(f"thresholds_fold_{fold}.csv", index=False)

    fold_results = pd.DataFrame(fold_results)
    slice_results = pd.concat(slice_results, ignore_index=True)

    print("\n===== CV RESULTS =====")
    print(fold_results.to_string(index=False))
    print(f"\nMean macro F0.5: {fold_results['macro_f05'].mean():.6f}")
    print(f"Std macro F0.5: {fold_results['macro_f05'].std():.6f}")
    print(f"Mean threshold: {fold_results['threshold'].mean():.3f}")

    print("\n===== CORRECTED SLICE RESULTS =====")
    summary = slice_results.groupby("slice").agg(
        entities=("entities", "sum"),
        mean_f05=("macro_f05", "mean"),
        std_f05=("macro_f05", "std"),
    ).sort_values("mean_f05")
    print(summary)

    fold_results.to_csv("cv_results.csv", index=False)
    slice_results.to_csv("cv_slice_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    main(args)
