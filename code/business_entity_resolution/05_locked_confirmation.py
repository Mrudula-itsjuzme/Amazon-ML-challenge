"""Once-only full-pool confirmation of a frozen IDF matcher and two controls."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("entity_validation", HERE / "03_validate.py")
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)
match = validation.match


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_retrieval(folder, source_count):
    run = json.loads((folder / "train_retrieval_run.json").read_text())
    if (run["target_pool_size"] != 10_320_219 or
            run["source_count"] != source_count or
            run["labels_used_for_retrieval"] is not False):
        raise ValueError(f"Invalid full-pool label-free retrieval: {folder}")
    return run


def model_features(pairs, source, target, df_dir=None, drop_missing=False):
    frame = match.build_features(pairs, source, target).drop(
        columns=["name_native_idf", "address_rare", "number_weighted"])
    if drop_missing:
        frame = frame.drop(columns=list(validation.MISSING_ADDRESS_FEATURES))
    if df_dir is not None:
        frame = pd.concat([frame, match.global_target_idf_features(
            pairs, source, target, df_dir)], axis=1)
    return frame


def fit_policy(X, y, pairs, truth, grid, precision_floor, seed):
    groups = pairs.source1_entity_id.to_numpy()
    rows = np.arange(len(pairs))
    oof = np.zeros(len(pairs), dtype=np.float32)
    for fold, (fit, cal) in enumerate(
            GroupKFold(n_splits=3).split(rows, y, groups), 1):
        clf = match.model(seed + fold)
        clf.fit(X.iloc[fit], y[fit])
        oof[cal] = clf.predict_proba(X.iloc[cal])[:, 1]
    options = []
    for threshold in grid:
        selected = oof >= threshold
        precision = float(y[selected].mean()) if selected.any() else 0.0
        macro = match.score_policy(pairs, oof, truth, set(groups), threshold)
        options.append((threshold, precision, macro))
    if precision_floor is None:
        chosen = max(options, key=lambda row: (row[2], row[0]))
    else:
        eligible = [row for row in options if row[1] >= precision_floor]
        chosen = max(eligible or options, key=lambda row: (
            row[2], row[1], row[0]) if eligible else (row[1], row[2], row[0]))
    final = match.model(seed)
    final.fit(X, y)
    return final, chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true",
                        help="Check retrieval and feature schemas without reading labels")
    args = parser.parse_args()
    config_path = args.frozen_config.resolve()
    config = json.loads(config_path.read_text())
    dev_dir = ROOT / config["development_dir"]
    confirm_dir = ROOT / config["confirmation_dir"]
    confirm_base = ROOT / config["confirmation_base_dir"]
    previous_external = ROOT / config["previous_external_dir"]
    df_dir = ROOT / config["global_df_dir"]
    data_dir = ROOT / config["data_dir"]
    output = confirm_dir / "frozen_idf_confirmation.json"
    started = confirm_dir / "frozen_idf_confirmation.started.json"
    if output.exists() or started.exists():
        parser.error("Confirmation is already opened; refusing to repeat it")
    if config["selected_policy"] != "global_idf_fine_k100":
        raise ValueError("Frozen selected policy changed")
    for filename, key in (("01_pipeline.py", "pipeline_sha256"),
                          ("02_match.py", "matcher_sha256"),
                          ("03_validate.py", "validator_sha256"),
                          ("05_locked_confirmation.py", "confirmation_script_sha256")):
        if sha256(HERE / filename) != config[key]:
            raise ValueError(f"Frozen code changed: {filename}")
    if sha256(df_dir / "name_df.parquet") != config["name_df_sha256"] or \
            sha256(df_dir / "address_df.parquet") != config["address_df_sha256"]:
        raise ValueError("Global target document frequencies changed")
    source, target, _ = match.load_inputs(dev_dir, "train")
    confirm_source, confirm_target, _ = match.load_inputs(confirm_dir, "train")
    previous_source = pd.read_parquet(previous_external / "train_s1.parquet")
    if len(source) != 1000 or len(confirm_source) != 500:
        raise ValueError("Unexpected development or confirmation cohort size")
    if (set(confirm_source.entity_id) & set(source.entity_id) or
            set(confirm_source.entity_id) & set(previous_source.entity_id)):
        raise ValueError("Confirmation S1 overlaps a previously opened cohort")
    ensure_retrieval(dev_dir, len(source))
    ensure_retrieval(confirm_dir, len(confirm_source))
    base_run = ensure_retrieval(confirm_base, len(confirm_source))
    if base_run["sample_offset"] != 1800:
        raise ValueError("Confirmation was not drawn from the frozen hash offset")
    old_routes = tuple(match.ROUTES)
    if old_routes != tuple(config["routes"]):
        raise ValueError("Active route set differs from frozen policy")
    dev_ids_a, dev_ids_b, old_locked = validation.sealed_partitions(source)
    dev_ids = dev_ids_a | dev_ids_b
    if len(dev_ids) != 800 or len(old_locked) != 200:
        raise ValueError("Development partition changed")
    dev_audit = pd.read_parquet(dev_dir / "train_route_audit.parquet")
    dev_audit = dev_audit[dev_audit.source1_entity_id.isin(dev_ids)]
    confirm_audit = pd.read_parquet(confirm_dir / "train_route_audit.parquet")
    pairs = {
        "earlier_k60": (validation.subset_pairs(dev_audit, old_routes, 60),
                         validation.subset_pairs(confirm_audit, old_routes, 60)),
        "retained_k100": (validation.subset_pairs(dev_audit, old_routes, 100),
                           validation.subset_pairs(confirm_audit, old_routes, 100)),
    }
    # Freeze candidate identities and schemas before opening confirmation labels.
    for name, (train_pairs, test_pairs) in pairs.items():
        if (len(train_pairs) != 800 * (60 if name == "earlier_k60" else 100) or
                len(test_pairs) != 500 * (60 if name == "earlier_k60" else 100)):
            raise ValueError(f"Unexpected candidate counts for {name}")
    if args.preflight:
        for name in ("earlier_k60", "retained_k100", "global_idf_fine_k100"):
            train_pairs, test_pairs = pairs["earlier_k60" if name == "earlier_k60"
                                            else "retained_k100"]
            selected = name == "global_idf_fine_k100"
            train_X = model_features(train_pairs, source, target,
                                     df_dir if selected else None)
            test_X = model_features(test_pairs, confirm_source, confirm_target,
                                    df_dir if selected else None)
            if list(train_X.columns) != list(test_X.columns):
                raise ValueError(f"Feature schema differs for {name}")
            print(f"Preflight {name}: {len(train_X)} train, {len(test_X)} confirmation, "
                  f"{len(train_X.columns)} features", flush=True)
        return
    confirm_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(started, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump({"config_sha256": sha256(config_path), "status": "opened_once"}, handle)

    dev_truth = match.parse_truth(data_dir / "train/train_ground_truth.tsv", dev_ids)
    confirm_ids = set(confirm_source.entity_id)
    confirm_truth = match.parse_truth(data_dir / "train/train_ground_truth.tsv", confirm_ids)
    # Truth hydration is diagnostic only, after retrieval and feature inputs froze.
    eval_target = validation.hydrate_truth_for_diagnostics(
        confirm_target, confirm_truth, data_dir,
        confirm_dir / "train_truth_targets_diagnostics.parquet")
    results = {}
    for name in ("earlier_k60", "retained_k100", "global_idf_fine_k100"):
        train_pairs, test_pairs = pairs["earlier_k60" if name == "earlier_k60"
                                        else "retained_k100"]
        selected = name == "global_idf_fine_k100"
        train_X = model_features(train_pairs, source, target,
                                 df_dir if selected else None)
        test_X = model_features(test_pairs, confirm_source, confirm_target,
                                df_dir if selected else None)
        if list(train_X.columns) != list(test_X.columns):
            raise ValueError(f"Feature schema differs for {name}")
        y = np.fromiter((pair.candidate_entity_id in dev_truth.get(
            pair.source1_entity_id, set()) for pair in train_pairs.itertuples(index=False)),
            dtype=np.int8)
        policy = config["policies"][name]
        clf, (threshold, inner_precision, inner_macro) = fit_policy(
            train_X, y, train_pairs, dev_truth, policy["threshold_grid"],
            policy["precision_floor"], policy["seed"])
        probs = clf.predict_proba(test_X)[:, 1]
        metrics, slices, failures = validation.evaluate(
            confirm_ids, test_pairs, probs, threshold, confirm_truth,
            confirm_source, eval_target, full_target_pool=True)
        importance = sorted(zip(train_X.columns, clf.feature_importances_),
                            key=lambda row: -row[1])[:25]
        results[name] = {
            "threshold": threshold,
            "development_oof_precision": inner_precision,
            "development_oof_macro_f05": inner_macro,
            "training_pairs": len(train_pairs), "training_positives": int(y.sum()),
            "confirmation_pairs": len(test_pairs),
            "metrics": metrics, "slices": slices,
            "failures": dict(failures), "feature_importance_top25": importance,
        }
        print(name, metrics, flush=True)
    profile = {
        "target_pool_size": 10_320_219,
        "confirmation_entities": len(confirm_ids),
        "confirmation_true_links": sum(map(len, confirm_truth.values())),
        "candidate_recall_at_k": validation.retrieval_k_report(
            confirm_audit, confirm_truth, confirm_ids),
    }
    report = {"status": "completed_once", "selected_policy": config["selected_policy"],
              "config_sha256": sha256(config_path), "profile": profile,
              "results": results}
    output.write_text(json.dumps(validation.json_safe(report), indent=2, allow_nan=False))
    print(f"Wrote locked confirmation: {output}", flush=True)


if __name__ == "__main__":
    main()
