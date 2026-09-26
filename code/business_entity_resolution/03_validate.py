"""Leakage-free grouped validation and retrieval/failure diagnostics."""
import argparse
from collections import Counter, defaultdict
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from lightgbm import LGBMClassifier

LOCAL_DEPS = Path(__file__).resolve().parents[2] / "research_runs/python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("entity_match", Path(__file__).with_name("02_match.py"))
match = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(match)


def sealed_partitions(source):
    ids = source.entity_id.to_numpy().copy()
    np.random.default_rng(2026).shuffle(ids)
    first, second = int(.6 * len(ids)), int(.8 * len(ids))
    return set(ids[:first]), set(ids[first:second]), set(ids[second:])


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def ensure_route_columns(audit):
    audit = audit.copy()
    for route in match.ROUTES:
        for suffix, default in (("rank", 0), ("score", 0.0)):
            column = f"{route}_{suffix}"
            if column not in audit:
                audit[column] = default
    return audit


def retrieval_ablation(source, audit, truth, max_cap=300):
    """Select routes on a design split; inspect confirmation split once."""
    audit = ensure_route_columns(audit)
    routes = match.ROUTES
    design_ids, confirm_ids, _ = sealed_partitions(source)
    grouped = {i: frame for i, frame in audit.groupby("source1_entity_id", sort=False)}

    def score_many(keep, selected_ids, caps):
        labels = set(selected_ids)
        found = {cap: 0 for cap in caps}
        sizes = {cap: [] for cap in caps}
        total = 0
        for i in labels:
            frame = grouped.get(i)
            actual = truth.get(i, set())
            total += len(actual)
            if frame is None:
                for cap in caps:
                    sizes[cap].append(0)
                continue
            ranks = frame[[f"{r}_rank" for r in keep]].to_numpy()
            active = ranks > 0
            agreement = active.sum(axis=1)
            reciprocal = np.where(active, 1 / (ranks + 1), 0).sum(axis=1)
            indices = np.flatnonzero(agreement)
            indices = sorted(indices, key=lambda j: (-agreement[j], -reciprocal[j],
                                                      frame.candidate_entity_id.iat[j]))[:max(caps)]
            ordered = frame.candidate_entity_id.iloc[indices].tolist()
            for cap in caps:
                chosen = set(ordered[:cap])
                sizes[cap].append(len(chosen))
                found[cap] += len(actual & chosen)
        return {cap: {"recall": found[cap] / total if total else float("nan"),
                      "found": found[cap], "true": total,
                      "mean_candidates": float(np.mean(sizes[cap])),
                      "p95_candidates": float(np.quantile(sizes[cap], .95))}
                for cap in caps}

    caps = [c for c in (20, 40, 60, 80, 100, 120, 160, 200, 250, 300)
            if c <= max_cap]
    if not caps:
        raise ValueError("--cap must be at least 20 for retrieval ablation")
    full = score_many(routes, design_ids, caps)
    print("all_routes_design:", full)
    if max(result["recall"] for result in full.values()) < .99:
        print("retrieval_gate: FAIL; all-route design recall is below 99%")
        return {"status": "fail_design", "all_routes_design": full}
    options = []
    for size in range(1, len(routes) + 1):
        for keep in combinations(routes, size):
            for cap, result in score_many(keep, design_ids, caps).items():
                options.append((keep, cap, result))
    viable = [(keep, cap, result) for keep, cap, result in options if result["recall"] >= .99]
    if not viable:
        print("retrieval_gate: FAIL; no route subset reached 99% design recall")
        return {"status": "fail_design", "all_routes_design": full}
    selected, selected_cap, design = min(
        viable, key=lambda x: (len(x[0]), x[2]["mean_candidates"], x[1], x[0]))
    confirmation = score_many(selected, confirm_ids, [selected_cap])[selected_cap]
    print("selected_routes:", selected)
    print("selected_cap:", selected_cap)
    print("selected_design:", design)
    print("selected_confirmation:", confirmation)
    status = "pass" if confirmation["recall"] >= .99 else "fail_confirmation"
    print("retrieval_gate:", status)
    return {"status": status, "routes": selected, "cap": selected_cap,
            "design": design, "confirmation": confirmation,
            "all_routes_design": full}


def subset_pairs(audit, keep, cap):
    audit = ensure_route_columns(audit)
    rows = []
    for _, frame in audit.groupby("source1_entity_id", sort=False):
        frame = frame.copy()
        ranks = frame[[f"{r}_rank" for r in keep]].to_numpy()
        active = ranks > 0
        frame["_agreement"] = active.sum(axis=1)
        frame["_reciprocal"] = np.where(active, 1 / (ranks + 1), 0).sum(axis=1)
        frame = frame.loc[frame._agreement > 0].sort_values(
            ["_agreement", "_reciprocal", "candidate_entity_id"],
            ascending=[False, False, True]).head(cap)
        frame["route_count"] = frame._agreement
        for route in match.ROUTES:
            if route not in keep:
                frame[f"{route}_rank"] = 0
                frame[f"{route}_score"] = 0.0
        rows.append(frame.drop(columns=["_agreement", "_reciprocal"]))
    return pd.concat(rows, ignore_index=True)


def fit_with_oof_threshold(factory, X, y, pairs, groups, truth, rows, seed):
    oof = np.zeros(len(rows), dtype=np.float32)
    inner = GroupKFold(n_splits=3)
    for fold, (fit_pos, cal_pos) in enumerate(
            inner.split(rows, y[rows], groups[rows]), 1):
        fit, cal = rows[fit_pos], rows[cal_pos]
        clf = factory(seed + fold)
        clf.fit(X.iloc[fit], y[fit])
        oof[cal_pos] = clf.predict_proba(X.iloc[cal])[:, 1]
    oof_score, threshold = match.tune_threshold(pairs.iloc[rows], oof, truth,
                                                 set(groups[rows]))
    final = factory(seed)
    final.fit(X.iloc[rows], y[rows])
    return final, threshold, oof_score


def model_research(source, target, pairs, truth, selection=None, eval_target=None):
    """Compare on development folds or confirm a frozen choice once."""
    retrieval_design, retrieval_confirm, confirm_ids = sealed_partitions(source)
    design_ids = retrieval_design | retrieval_confirm
    X = match.build_features(pairs, source, target)
    if eval_target is None:
        eval_target = target
    # The hydrated target table contains only retrieved records. Its document
    # frequencies depend on the held-out S1 queries, so exclude those features.
    X = X.drop(columns=["name_native_idf", "address_rare", "number_weighted"])
    y = np.fromiter((p.candidate_entity_id in truth.get(p.source1_entity_id, set())
                     for p in pairs.itertuples(index=False)), dtype=np.int8)
    groups = pairs.source1_entity_id.to_numpy()
    variants = {"LightGBM": lambda seed: match.model(seed)}
    if CatBoostClassifier is not None:
        variants["CatBoost"] = lambda seed: CatBoostClassifier(
            iterations=350, depth=6, learning_rate=.05, l2_leaf_reg=5,
            loss_function="Logloss", random_seed=seed, thread_count=4,
            verbose=False, allow_writing_files=False)
    development = np.flatnonzero(np.isin(groups, list(design_ids)))
    if selection is None:
        outcomes = {}
        for name, factory in variants.items():
            fold_scores = []
            cv = GroupKFold(n_splits=3)
            for fold, (fit_cal_pos, val_pos) in enumerate(
                    cv.split(development, y[development], groups[development]), 1):
                fit_cal, val = development[fit_cal_pos], development[val_pos]
                if len(np.unique(y[fit_cal])) < 2:
                    raise ValueError("Too few positive pairs in development fold")
                clf, threshold, _ = fit_with_oof_threshold(
                    factory, X, y, pairs, groups, truth, fit_cal, 2026 + fold)
                val_prob = clf.predict_proba(X.iloc[val])[:, 1]
                metrics, slices, failures = evaluate(set(groups[val]), pairs.iloc[val], val_prob,
                                                      threshold, truth, source, eval_target,
                                                      full_target_pool=True)
                metrics["threshold"] = threshold
                for slice_name in ("cross_script", "missing_address", "short_name"):
                    metrics[slice_name + "_f05"] = slices.get(slice_name, (0, float("nan")))[1]
                    metrics[slice_name + "_entities"] = slices.get(slice_name, (0, float("nan")))[0]
                fold_scores.append(metrics)
                print(f"design {name} fold {fold}: {metrics}")
                print(f"design {name} fold {fold} failures: {dict(failures)}")
            numeric = pd.DataFrame(fold_scores).select_dtypes(include="number")
            outcomes[name] = {"mean": numeric.mean().to_dict(),
                              "std": numeric.std(ddof=1).fillna(0).to_dict(),
                              "folds": fold_scores}
        print("design_model_comparison:", outcomes)
        chosen = max(variants, key=lambda name: (outcomes[name]["mean"]["macro_f05"],
                                                outcomes[name]["mean"]["pair_precision"]))
        country_check = None
        countries = dict(zip(source.entity_id, source.country_norm))
        for country in ("us", "india"):
            held_ids = {i for i in design_ids if countries[i] == country}
            if len(held_ids) < 30 or len(design_ids - held_ids) < 30:
                continue
            fit_rows = np.flatnonzero(np.isin(groups, list(design_ids - held_ids)))
            val_rows = np.flatnonzero(np.isin(groups, list(held_ids)))
            if len(np.unique(y[fit_rows])) < 2 or not y[val_rows].any():
                continue
            clf, threshold, _ = fit_with_oof_threshold(
                variants[chosen], X, y, pairs, groups, truth, fit_rows, 27182)
            country_check, _, _ = evaluate(
                held_ids, pairs.iloc[val_rows], clf.predict_proba(X.iloc[val_rows])[:, 1],
                threshold, truth, source, eval_target, full_target_pool=True)
            print(f"development_country_holdout_{country}:", country_check)
            break
        return {"model": chosen, "development": outcomes,
                "development_country_holdout": country_check}
    chosen = selection["model"]
    if chosen not in variants:
        raise ValueError(f"Unknown frozen model: {chosen}")
    clf, threshold, oof_score = fit_with_oof_threshold(
        variants[chosen], X, y, pairs, groups, truth, development, 31415)
    confirm = np.flatnonzero(np.isin(groups, list(confirm_ids)))
    result, slices, failures = evaluate(confirm_ids, pairs.iloc[confirm],
                                        clf.predict_proba(X.iloc[confirm])[:, 1],
                                        threshold, truth, source, eval_target,
                                        full_target_pool=True)
    print("selected_model:", chosen, "threshold:", threshold,
          "development_oof_macro_f05:", oof_score)
    print("sealed_confirmation:", result)
    print("sealed_slices:", slices)
    print("sealed_failures:", dict(failures))
    importance = sorted(zip(X.columns, clf.feature_importances_),
                        key=lambda item: -item[1])[:25]
    print("feature_importance_top25:", importance)
    return {"model": chosen, "threshold": threshold, "development_oof_macro_f05": oof_score,
            "feature_importance_top25": importance,
            "sealed": result, "sealed_slices": slices, "sealed_failures": dict(failures)}


def evaluate(ids, pairs, probabilities, threshold, truth, source, target,
             full_target_pool=False):
    ids = list(ids)
    candidate_map = pairs.groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
    selected = pairs.loc[np.asarray(probabilities) >= threshold]
    predicted_map = selected.groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
    target_index = target.set_index("entity_id")
    source_index = source.set_index("entity_id")
    target_ids = set(target.entity_id)
    all_true_pairs = {(i, j) for i in ids for j in truth.get(i, set())}
    true_pairs = all_true_pairs if full_target_pool else {
        (i, j) for i in ids for j in truth.get(i, set()) if j in target_ids}
    retrieved = {(i, j) for i in ids for j in candidate_map.get(i, set()) if (i, j) in true_pairs}
    predicted = {(i, j) for i in ids for j in predicted_map.get(i, set())}
    tp = len(predicted & true_pairs)
    fp = len(predicted - true_pairs)
    fn = len(true_pairs - predicted)
    sizes = np.array([len(candidate_map.get(i, set())) for i in ids])
    singleton = [i for i in ids if not truth.get(i)]
    singleton_accuracy = np.mean([not predicted_map.get(i) for i in singleton]) if singleton else float("nan")
    metrics = {
        "entities": len(ids), "eligible_true_pairs": len(true_pairs),
        "target_pool_truth_coverage": len(true_pairs) / len(all_true_pairs) if all_true_pairs else float("nan"),
        "blocker_recall": len(retrieved) / len(true_pairs) if true_pairs else float("nan"),
        "macro_f05": np.mean([match.entity_f05(truth.get(i, set()), predicted_map.get(i, set())) for i in ids]),
        "pool_macro_f05": np.mean([match.entity_f05(truth.get(i, set()) if full_target_pool
                                                      else truth.get(i, set()) & target_ids,
                                                      predicted_map.get(i, set())) for i in ids]),
        "pair_precision": tp / (tp + fp) if tp + fp else float("nan"),
        "pair_recall": tp / len(true_pairs) if true_pairs else float("nan"),
        "singleton_accuracy": singleton_accuracy,
        "mean_candidates": sizes.mean(), "p95_candidates": np.quantile(sizes, .95),
        "p99_candidates": np.quantile(sizes, .99),
        "blocker_failures": len(true_pairs - retrieved),
        "matcher_false_negatives": len(retrieved - predicted),
        "matcher_false_positives": fp,
    }
    slices = {}
    def add(name, members):
        members = set(members)
        if members:
            slices[name] = (len(members), np.mean([
                match.entity_f05(truth.get(i, set()), predicted_map.get(i, set())) for i in members]))
    for name, predicate in (
        ("cross_script", lambda a, b: a.name_script != b.name_script and "unknown" not in (a.name_script, b.name_script)),
        ("same_script", lambda a, b: a.name_script == b.name_script),
        ("missing_address", lambda a, b: bool(a.address_missing or b.address_missing)),
        ("full_address", lambda a, b: not (a.address_missing or b.address_missing)),
        ("short_name", lambda a, b: a.name_tokens <= 2),
    ):
        members = set()
        for i, j in true_pairs:
            if j in target_index.index and predicate(source_index.loc[i], target_index.loc[j]):
                members.add(i)
        add(name, members)
    add("high_ambiguity", [i for i in ids if len(candidate_map.get(i, set())) >= 50])
    country_groups = source_index.loc[ids].groupby("country_norm")
    for country, frame in country_groups:
        if len(frame) >= 30:
            add("country_" + (country or "missing"), frame.index)
    failure = Counter()
    for kind, examples in (("blocker", true_pairs - retrieved),
                           ("matcher_fn", retrieved - predicted),
                           ("matcher_fp", predicted - true_pairs)):
        for i, j in examples:
            failure[kind] += 1
            if j not in target_index.index:
                failure[kind + ":target_not_hydrated"] += 1
                continue
            a, b = source_index.loc[i], target_index.loc[j]
            if a.name_tokens <= 2:
                failure[kind + ":short_name"] += 1
            if a.name_script != b.name_script and "unknown" not in (a.name_script, b.name_script):
                failure[kind + ":cross_script"] += 1
            if a.address_missing or b.address_missing:
                failure[kind + ":missing_address"] += 1
            elif match.ratio(a.address_native, b.address_native) < 45:
                failure[kind + ":low_address_similarity"] += 1
            if set(a.address_numbers) and set(b.address_numbers) and not set(a.address_numbers) & set(b.address_numbers):
                failure[kind + ":numeric_conflict"] += 1
            if len(candidate_map.get(i, set())) >= 50:
                failure[kind + ":high_ambiguity"] += 1
    return metrics, slices, failure


def hydrate_truth_for_diagnostics(target, truth, data_dir, cache_path):
    """Read true targets only for error slices; never add them to candidates/features."""
    if cache_path.exists():
        extra = pd.read_parquet(cache_path)
    else:
        needed = set().union(*truth.values()) - set(target.entity_id)
        frames = []
        for part in (2, 3):
            path = data_dir / f"train/train_source{part}.tsv"
            for chunk in pd.read_csv(path, sep="\t", dtype=str,
                                     keep_default_na=False, chunksize=100000):
                picked = chunk.loc[chunk.entity_id.isin(needed)]
                if len(picked):
                    frames.append(picked)
        found = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(found) != len(needed):
            raise ValueError(f"Could hydrate only {len(found)}/{len(needed)} missing true targets")
        spec = importlib.util.spec_from_file_location(
            "entity_pipeline", Path(__file__).with_name("01_pipeline.py"))
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
        extra = pipeline.prep(found)
        extra.to_parquet(cache_path, index=False)
    return pd.concat([target, extra], ignore_index=True)


def retrieval_k_report(audit, truth, ids, ks=(40, 60, 80, 100, 150)):
    """Report a fixed K grid with the frozen all-route agreement ranking."""
    audit = ensure_route_columns(audit)
    ids = set(ids)
    totals = {k: 0 for k in ks}
    n_true = sum(len(truth.get(i, set())) for i in ids)
    for sid, frame in audit.loc[audit.source1_entity_id.isin(ids)].groupby(
            "source1_entity_id", sort=False):
        ranks = frame[[f"{r}_rank" for r in match.ROUTES]].to_numpy()
        active = ranks > 0
        agreement = active.sum(axis=1)
        reciprocal = np.where(active, 1 / (ranks + 1), 0).sum(axis=1)
        order = sorted(range(len(frame)), key=lambda j: (
            -agreement[j], -reciprocal[j], frame.candidate_entity_id.iat[j]))
        targets = frame.candidate_entity_id.iloc[order].tolist()
        actual = truth.get(sid, set())
        for k in ks:
            totals[k] += len(actual.intersection(targets[:k]))
    return {k: {"found": totals[k], "true": n_true,
                "recall": totals[k] / n_true if n_true else float("nan")}
            for k in ks}


MISSING_ADDRESS_FEATURES = (
    "source_address_missing", "target_address_missing", "name_core_token",
    "name_translit_token", "name_native_when_target_address_missing",
    "name_core_when_target_address_missing",
    "name_translit_when_target_address_missing",
)


def precision_constrained_fit(X, y, pairs, truth, rows, seed):
    """Choose a threshold using grouped OOF predictions on training groups only."""
    groups = pairs.source1_entity_id.to_numpy()
    oof = np.zeros(len(rows), dtype=np.float32)
    for fold, (fit_pos, cal_pos) in enumerate(
            GroupKFold(n_splits=3).split(rows, y[rows], groups[rows]), 1):
        fit, cal = rows[fit_pos], rows[cal_pos]
        clf = match.model(seed + fold)
        clf.fit(X.iloc[fit], y[fit])
        oof[cal_pos] = clf.predict_proba(X.iloc[cal])[:, 1]
    options = []
    for threshold in (.5, .6, .7, .8, .9, .95, .975, .99, .995, .999):
        selected = oof >= threshold
        precision = float(y[rows][selected].mean()) if selected.any() else 0.0
        f05 = match.score_policy(pairs.iloc[rows], oof, truth,
                                 set(groups[rows]), threshold)
        options.append((threshold, precision, f05))
    eligible = [item for item in options if item[1] >= .97]
    chosen = max(eligible or options, key=lambda item: (
        item[2], item[1], item[0]) if eligible else (item[1], item[2], item[0]))
    clf = match.model(seed)
    clf.fit(X.iloc[rows], y[rows])
    return clf, chosen


def rerank_study(source, target, audit, truth, eval_target):
    """Fixed development-only comparison; the prior 200-S1 seal stays closed."""
    design_a, design_b, _ = sealed_partitions(source)
    development_ids = design_a | design_b
    audit = audit.loc[audit.source1_entity_id.isin(development_ids)].copy()
    conditions = (("baseline_k60", 60, False),
                  ("baseline_k150", 150, False),
                  ("missing_k100", 100, True),
                  ("missing_k150", 150, True))
    fold_ids = []
    ids = np.array(sorted(development_ids))
    for fit_pos, val_pos in GroupKFold(n_splits=3).split(ids, groups=ids):
        fold_ids.append((set(ids[fit_pos]), set(ids[val_pos])))
    outcomes = {}
    for name, cap, use_missing in conditions:
        pairs = subset_pairs(audit, match.ROUTES, cap)
        X = match.build_features(pairs, source, target).drop(
            columns=["name_native_idf", "address_rare", "number_weighted"])
        if not use_missing:
            X = X.drop(columns=list(MISSING_ADDRESS_FEATURES))
        y = np.fromiter((p.candidate_entity_id in truth.get(p.source1_entity_id, set())
                         for p in pairs.itertuples(index=False)), dtype=np.int8)
        groups = pairs.source1_entity_id.to_numpy()
        folds = []
        for fold, (fit_ids, val_ids) in enumerate(fold_ids, 1):
            fit = np.flatnonzero(np.isin(groups, list(fit_ids)))
            val = np.flatnonzero(np.isin(groups, list(val_ids)))
            clf, (threshold, inner_precision, inner_f05) = precision_constrained_fit(
                X, y, pairs, truth, fit, 51000 + fold)
            result, slices, failures = evaluate(
                val_ids, pairs.iloc[val], clf.predict_proba(X.iloc[val])[:, 1],
                threshold, truth, source, eval_target, full_target_pool=True)
            result.update({"threshold": threshold,
                           "inner_oof_precision": inner_precision,
                           "inner_oof_f05": inner_f05,
                           "missing_address_f05": slices.get("missing_address", (0, None))[1],
                           "missing_address_entities": slices.get("missing_address", (0, None))[0],
                           "cross_script_f05": slices.get("cross_script", (0, None))[1]})
            folds.append(result)
            print(f"rerank {name} fold {fold}:", result, flush=True)
            print(f"rerank {name} fold {fold} failures:", dict(failures), flush=True)
        frame = pd.DataFrame(folds).select_dtypes(include="number")
        outcomes[name] = {"mean": frame.mean().to_dict(),
                          "std": frame.std(ddof=1).fillna(0).to_dict(),
                          "folds": folds, "pairs": len(pairs),
                          "positive_pairs": int(y.sum())}
        print(f"rerank summary {name}:", outcomes[name]["mean"], flush=True)
    baseline = outcomes["baseline_k60"]["folds"]
    viable = []
    for name, _, _ in conditions[1:]:
        folds = outcomes[name]["folds"]
        if (all(f["macro_f05"] > b["macro_f05"] for f, b in zip(folds, baseline))
                and min(f["pair_precision"] for f in folds) >= .97):
            viable.append(name)
    selected = max(viable, key=lambda name: outcomes[name]["mean"]["macro_f05"]) if viable else None
    print("rerank_stable_precision_qualified_selection:", selected, flush=True)
    return {"status": "development_only", "selected": selected,
            "precision_floor": .97, "outcomes": outcomes}


def external_rerank_confirmation(train_dir, test_dir, data_dir):
    """One frozen K60 baseline vs K100 candidate on a disjoint full-pool cohort."""
    train_source, train_target, _ = match.load_inputs(train_dir, "train")
    test_source, test_target, _ = match.load_inputs(test_dir, "train")
    prior_a, prior_b, _ = sealed_partitions(train_source)
    development_ids = prior_a | prior_b
    if len(train_source) != 1000 or len(test_source) != 500:
        raise ValueError("Expected the frozen 1,000-S1 development and 500-S1 external cohorts")
    if set(train_source.entity_id) & set(test_source.entity_id):
        raise ValueError("External confirmation overlaps development")
    train_truth = match.parse_truth(data_dir / "train/train_ground_truth.tsv", development_ids)
    test_truth = match.parse_truth(data_dir / "train/train_ground_truth.tsv", set(test_source.entity_id))
    train_audit = pd.read_parquet(train_dir / "train_route_audit.parquet")
    train_audit = train_audit.loc[train_audit.source1_entity_id.isin(development_ids)]
    test_audit = pd.read_parquet(test_dir / "train_route_audit.parquet")
    test_eval_target = hydrate_truth_for_diagnostics(
        test_target, test_truth, data_dir,
        test_dir / "train_truth_targets_diagnostics.parquet")
    results = {}
    for name, cap, use_missing in (("baseline_k60", 60, False),
                                   ("missing_k100", 100, True)):
        train_pairs = subset_pairs(train_audit, match.ROUTES, cap)
        test_pairs = subset_pairs(test_audit, match.ROUTES, cap)
        train_X = match.build_features(train_pairs, train_source, train_target).drop(
            columns=["name_native_idf", "address_rare", "number_weighted"])
        test_X = match.build_features(test_pairs, test_source, test_target).drop(
            columns=["name_native_idf", "address_rare", "number_weighted"])
        if not use_missing:
            train_X = train_X.drop(columns=list(MISSING_ADDRESS_FEATURES))
            test_X = test_X.drop(columns=list(MISSING_ADDRESS_FEATURES))
        if list(train_X.columns) != list(test_X.columns):
            raise ValueError("Training and external feature schemas differ")
        y = np.fromiter((p.candidate_entity_id in train_truth.get(p.source1_entity_id, set())
                         for p in train_pairs.itertuples(index=False)), dtype=np.int8)
        clf, (threshold, inner_precision, inner_f05) = precision_constrained_fit(
            train_X, y, train_pairs, train_truth, np.arange(len(train_pairs)), 63000)
        metrics, slices, failures = evaluate(
            set(test_source.entity_id), test_pairs, clf.predict_proba(test_X)[:, 1],
            threshold, test_truth, test_source, test_eval_target, full_target_pool=True)
        results[name] = {"threshold": threshold, "development_oof_precision": inner_precision,
                         "development_oof_f05": inner_f05,
                         "training_pairs": len(train_pairs), "training_positives": int(y.sum()),
                         "external_pairs": len(test_pairs), "metrics": metrics,
                         "slices": slices, "failures": dict(failures)}
        print("external", name, results[name], flush=True)
    baseline = results["baseline_k60"]["metrics"]
    candidate = results["missing_k100"]["metrics"]
    success = candidate["macro_f05"] > baseline["macro_f05"] and candidate["pair_precision"] >= .97
    return {"status": "confirmed" if success else "not_confirmed",
            "target_pool_size": json.loads((test_dir / "train_retrieval_run.json").read_text())["target_pool_size"],
            "external_entities": len(test_source),
            "external_true_links": sum(map(len, test_truth.values())),
            "results": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=ROOT / "output")
    p.add_argument("--data-dir", type=Path, default=ROOT / "student_resource/dataset")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--retrieval-only", action="store_true", help="Use raw route audit for a frozen design/confirmation ablation")
    p.add_argument("--cap", type=int, default=300)
    p.add_argument("--model-design", action="store_true", help="Compare models on development folds only")
    p.add_argument("--model-confirm", action="store_true", help="Evaluate the frozen choice once on sealed S1 groups")
    p.add_argument("--fullpool-model-design", action="store_true",
                   help="Compare fixed matchers on full-target retrieved candidates")
    p.add_argument("--fullpool-model-confirm", action="store_true",
                   help="Evaluate the selected matcher once on locked S1 groups")
    p.add_argument("--rerank-study", action="store_true",
                   help="Compare fixed wider-union rerankers on development S1 only")
    p.add_argument("--rerank-external-confirm", action="store_true",
                   help="Once-only external check of frozen K60 and K100 policies")
    p.add_argument("--train-dir", type=Path,
                   default=ROOT / "research_runs/ber_fullpool_1000_offset300_aug1")
    args = p.parse_args()
    if sum((args.retrieval_only, args.model_design, args.model_confirm,
            args.fullpool_model_design, args.fullpool_model_confirm,
            args.rerank_study, args.rerank_external_confirm)) > 1:
        p.error("Choose one validation mode")
    if args.rerank_external_confirm:
        output = args.input_dir / "external_rerank_confirmation.json"
        if output.exists():
            p.error("External confirmation already exists; do not reopen it")
        if not (args.train_dir / "rerank_development_study.json").exists():
            p.error("Run development reranking study before external confirmation")
        run = json.loads((args.input_dir / "train_retrieval_run.json").read_text())
        if run["target_pool_size"] < 10_000_000 or run["labels_used_for_retrieval"] is not False:
            p.error("External retrieval provenance is invalid")
        result = external_rerank_confirmation(args.train_dir, args.input_dir, args.data_dir)
        output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
        return
    source, target, pairs = match.load_inputs(args.input_dir, "train")
    truth = match.parse_truth(args.data_dir / "train/train_ground_truth.tsv", set(source.entity_id))
    if args.rerank_study:
        study_path = args.input_dir / "rerank_development_study.json"
        if study_path.exists():
            p.error("This reranking comparison already ran; do not retune on the same folds")
        if len(source) != 1000 or not (args.input_dir / "fullpool_model_confirmation.json").exists():
            p.error("Use the completed 1,000-S1 full-pool cohort for development-only reranking")
        eval_target = hydrate_truth_for_diagnostics(
            target, truth, args.data_dir,
            args.input_dir / "train_truth_targets_diagnostics.parquet")
        decision = rerank_study(
            source, target, pd.read_parquet(args.input_dir / "train_route_audit.parquet"),
            truth, eval_target)
        study_path.write_text(json.dumps(json_safe(decision), indent=2, allow_nan=False))
        return
    if args.fullpool_model_design or args.fullpool_model_confirm:
        if len(source) < 500 or not (args.input_dir / "train_route_audit.parquet").exists():
            p.error("Full-pool matcher study requires >=500 S1 and a full-target route audit")
        run_path = args.input_dir / "train_retrieval_run.json"
        if not run_path.exists():
            p.error("Missing full-target retrieval provenance manifest")
        run = json.loads(run_path.read_text())
        if (run["target_pool_size"] < 10_000_000 or
                run["source_count"] != len(source) or
                run["candidate_pair_count"] != len(pairs) or
                run["labels_used_for_retrieval"] is not False):
            p.error("Retrieval provenance does not verify a label-free full target scan")
        design_path = args.input_dir / "fullpool_model_selection.json"
        confirm_path = args.input_dir / "fullpool_model_confirmation.json"
        if confirm_path.exists():
            p.error("Locked confirmation already exists; this cohort is closed")
        eval_target = hydrate_truth_for_diagnostics(
            target, truth, args.data_dir,
            args.input_dir / "train_truth_targets_diagnostics.parquet")
        design_a, design_b, locked = sealed_partitions(source)
        study_ids = locked if args.fullpool_model_confirm else design_a | design_b
        study_pairs = pairs.loc[pairs.source1_entity_id.isin(study_ids)]
        positive = sum(p.candidate_entity_id in truth.get(p.source1_entity_id, set())
                       for p in study_pairs.itertuples(index=False))
        profile = {"entities": len(study_ids),
                   "true_links": sum(len(truth.get(i, set())) for i in study_ids),
                   "candidate_pairs": len(study_pairs), "positive_pairs": positive,
                   "negative_pairs": len(study_pairs) - positive,
                   "candidate_recall_at_k": retrieval_k_report(
                       pd.read_parquet(args.input_dir / "train_route_audit.parquet"),
                       truth, study_ids)}
        print("fullpool_cohort_profile:", profile)
        if args.fullpool_model_design:
            if design_path.exists():
                p.error("Model selection already exists; do not repeat tuning on this cohort")
            selection = model_research(source, target, pairs, truth,
                                       eval_target=eval_target)
            selection["cohort_profile"] = profile
            design_path.write_text(json.dumps(json_safe(selection), indent=2,
                                              allow_nan=False))
        else:
            if not design_path.exists():
                p.error("Run --fullpool-model-design first")
            selection = json.loads(design_path.read_text())
            result = model_research(source, target, pairs, truth, selection,
                                    eval_target=eval_target)
            result["cohort_profile"] = profile
            confirm_path.write_text(json.dumps(json_safe(result), indent=2,
                                               allow_nan=False))
        return
    if args.retrieval_only:
        path = args.input_dir / "train_route_audit.parquet"
        if not path.exists():
            p.error(f"Missing route audit: {path}; rerun 01_pipeline.py with --audit-retrieval")
        decision = retrieval_ablation(source, pd.read_parquet(path), truth, args.cap)
        with (args.input_dir / "retrieval_selection.json").open("w") as handle:
            json.dump(json_safe(decision), handle, indent=2, allow_nan=False)
        return
    if args.model_design or args.model_confirm:
        decision_path = args.input_dir / "retrieval_selection.json"
        if not decision_path.exists():
            p.error("Run --retrieval-only first to freeze the route decision")
        decision = json.loads(decision_path.read_text())
        if decision["status"] != "pass":
            p.error("The frozen retrieval decision did not pass 99% confirmation recall")
        keep, cap = tuple(decision["routes"]), int(decision["cap"])
        audit = pd.read_parquet(args.input_dir / "train_route_audit.parquet")
        pairs = subset_pairs(audit, keep, cap)
        selection_path = args.input_dir / "model_selection.json"
        confirmation_path = args.input_dir / "model_confirmation.json"
        if args.model_design:
            if confirmation_path.exists():
                p.error("Sealed confirmation already exists; do not retune on this cohort")
            selection = model_research(source, target, pairs, truth)
            selection_path.write_text(json.dumps(json_safe(selection), indent=2, allow_nan=False))
        else:
            if not selection_path.exists():
                p.error("Run --model-design before opening the sealed confirmation")
            if confirmation_path.exists():
                p.error("Sealed confirmation already exists; do not reopen it")
            selection = json.loads(selection_path.read_text())
            confirmation = model_research(source, target, pairs, truth, selection)
            confirmation_path.write_text(json.dumps(json_safe(confirmation), indent=2, allow_nan=False))
        return
    X = match.build_features(pairs, source, target)
    full_target_pool = (args.input_dir / "train_route_audit.parquet").exists()
    y = np.fromiter((p.candidate_entity_id in truth.get(p.source1_entity_id, set())
                     for p in pairs.itertuples(index=False)), dtype=np.int8)
    if len(np.unique(y)) != 2:
        raise ValueError("Need positive and negative retrieved pairs for grouped CV")
    ids = source.entity_id.to_numpy()
    groups = pairs.source1_entity_id.to_numpy()
    all_rows, all_slices = [], defaultdict(list)
    for fold, (train_ids_idx, validation_ids_idx) in enumerate(GroupKFold(args.folds).split(ids, groups=ids), 1):
        train_ids, validation_ids = set(ids[train_ids_idx]), set(ids[validation_ids_idx])
        train_rows = np.flatnonzero(np.isin(groups, list(train_ids)))
        validation_rows = np.flatnonzero(np.isin(groups, list(validation_ids)))
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42 + fold)
        fit_pos, cal_pos = next(splitter.split(train_rows, y[train_rows], groups[train_rows]))
        fit_rows, cal_rows = train_rows[fit_pos], train_rows[cal_pos]
        clf = match.model(42 + fold)
        clf.fit(X.iloc[fit_rows], y[fit_rows])
        cal_prob = clf.predict_proba(X.iloc[cal_rows])[:, 1]
        cal_ids = set(groups[cal_rows])
        _, threshold = match.tune_threshold(pairs.iloc[cal_rows], cal_prob, truth, cal_ids)
        val_prob = clf.predict_proba(X.iloc[validation_rows])[:, 1]
        metrics, slices, failures = evaluate(validation_ids, pairs.iloc[validation_rows], val_prob,
                                              threshold, truth, source, target,
                                              full_target_pool=full_target_pool)
        metrics.update(fold=fold, threshold=threshold)
        all_rows.append(metrics)
        for name, (count, score) in slices.items():
            all_slices[name].append((count, score))
        print(f"fold {fold}: " + " ".join(f"{k}={v:.6g}" for k, v in metrics.items() if isinstance(v, (int, float, np.number))))
        print("failure categories:", dict(failures))
    print("mean metrics:", pd.DataFrame(all_rows).mean(numeric_only=True).to_dict())
    print("slices:", {k: {"entities": sum(c for c, _ in v), "weighted_macro_f05":
                           sum(c*s for c, s in v)/sum(c for c, _ in v)} for k, v in all_slices.items()})
    # Country holdout is separate from random entity CV and is run only when each side has positives.
    countries = source.groupby("country_norm").size().sort_values(ascending=False)
    for country, count in countries.items():
        if count < 100 or count > .8 * len(source):
            continue
        held = set(source.loc[source.country_norm == country, "entity_id"])
        train_rows = np.flatnonzero(~np.isin(groups, list(held)))
        val_rows = np.flatnonzero(np.isin(groups, list(held)))
        if len(np.unique(y[train_rows])) != 2 or not y[val_rows].any():
            continue
        inner = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=314)
        fit_pos, cal_pos = next(inner.split(train_rows, y[train_rows], groups[train_rows]))
        fit_rows, cal_rows = train_rows[fit_pos], train_rows[cal_pos]
        clf = match.model(314)
        clf.fit(X.iloc[fit_rows], y[fit_rows])
        _, threshold = match.tune_threshold(pairs.iloc[cal_rows], clf.predict_proba(X.iloc[cal_rows])[:, 1],
                                            truth, set(groups[cal_rows]))
        result, _, _ = evaluate(held, pairs.iloc[val_rows], clf.predict_proba(X.iloc[val_rows])[:, 1],
                                threshold, truth, source, target,
                                full_target_pool=full_target_pool)
        print(f"held-out country {country}: {result}")
        break


if __name__ == "__main__":
    main()
