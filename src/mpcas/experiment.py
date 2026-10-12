"""Nested discovery-set selection followed by one locked validation evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from .data import load_table, outcome_rows, task_features
from .selection import FeatureSelector, xgb_parameters


def _folds(y: np.ndarray, count: int, seed: int):
    if len(np.unique(y)) != 2 or np.bincount(y, minlength=2).min() < count:
        raise ValueError(f"Both outcome classes need at least {count} discovery samples")
    return StratifiedKFold(n_splits=count, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def _selector(config: dict) -> FeatureSelector:
    return FeatureSelector(config["random_seed"], config["missing_fraction_max"], config["variance_threshold"], config["boruta_max_iter"], config["n_jobs"])


def _fit_predict(x_fit: pd.DataFrame, y_fit: np.ndarray, x_test: pd.DataFrame, k: int, config: dict):
    selector = _selector(config).fit(x_fit, y_fit)
    actual_k = min(k, len(selector.ranking_))
    model = xgb.XGBClassifier(**xgb_parameters(config["random_seed"], config["n_jobs"]))
    model.fit(selector.transform(x_fit, actual_k), y_fit)
    score = model.predict_proba(selector.transform(x_test, actual_k))[:, 1]
    return score, selector, model, actual_k


def _select_k(x: pd.DataFrame, y: np.ndarray, counts: list[int], config: dict) -> tuple[int, list[dict]]:
    scores = {k: np.empty(len(y), dtype=float) for k in counts}
    for fit_idx, test_idx in _folds(y, config["inner_folds"], config["random_seed"]):
        selector = _selector(config).fit(x.iloc[fit_idx], y[fit_idx])
        for k in counts:
            actual_k = min(k, len(selector.ranking_))
            model = xgb.XGBClassifier(**xgb_parameters(config["random_seed"], config["n_jobs"]))
            model.fit(selector.transform(x.iloc[fit_idx], actual_k), y[fit_idx])
            scores[k][test_idx] = model.predict_proba(selector.transform(x.iloc[test_idx], actual_k))[:, 1]
    rows = [{"requested_features": k, "discovery_tuning_auc": float(roc_auc_score(y, scores[k]))} for k in counts]
    best = sorted(rows, key=lambda row: (-row["discovery_tuning_auc"], row["requested_features"]))[0]
    return int(best["requested_features"]), rows


def _threshold(y: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, scores)
    finite = np.isfinite(thresholds)
    idx = np.argmax(np.where(finite, tpr - fpr, -np.inf))
    return float(thresholds[idx])


def _auc_ci(y: np.ndarray, scores: np.ndarray, resamples: int, seed: int) -> list[float] | None:
    if resamples <= 0:
        return None
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(resamples):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) == 2:
            aucs.append(float(roc_auc_score(y[index], scores[index])))
    return [float(v) for v in np.quantile(aucs, [0.025, 0.975])] if aucs else None


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float, resamples: int = 0, seed: int = 0) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "n_positive": int(y.sum()),
        "auc": float(roc_auc_score(y, scores)),
        "auc_95pct_bootstrap_ci": _auc_ci(y, scores, resamples, seed),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "ppv": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if tn + fn else None,
        "accuracy": float(accuracy_score(y, pred)),
        "threshold": threshold,
    }


def run_one(frame: pd.DataFrame, outcome: str, tasks: list[str], counts: list[int], config: dict, output: Path) -> dict:
    subset, y = outcome_rows(frame, outcome)
    feature_names = task_features(subset, tasks)
    train_mask = subset.cohort.eq("discovery").to_numpy()
    validation_mask = ~train_mask
    x_train, x_val = subset.loc[train_mask, feature_names].reset_index(drop=True), subset.loc[validation_mask, feature_names].reset_index(drop=True)
    y_train, y_val = y[train_mask], y[validation_mask]
    if len(np.unique(y_val)) != 2:
        raise ValueError(f"{outcome}: validation cohort needs both classes")
    oof = np.empty(len(y_train), dtype=float)
    outer_choices = []
    for outer_fold, (fit_idx, test_idx) in enumerate(_folds(y_train, config["outer_folds"], config["random_seed"]), 1):
        selected_k, _ = _select_k(x_train.iloc[fit_idx], y_train[fit_idx], counts, config)
        oof[test_idx], _, _, _ = _fit_predict(x_train.iloc[fit_idx], y_train[fit_idx], x_train.iloc[test_idx], selected_k, config)
        outer_choices.append({"fold": outer_fold, "requested_features": selected_k})
    threshold = _threshold(y_train, oof)
    selected_k, tuning_rows = _select_k(x_train, y_train, counts, config)
    val_scores, selector, model, actual_k = _fit_predict(x_train, y_train, x_val, selected_k, config)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "outcome": outcome, "tasks": tasks, "requested_features": selected_k,
        "selected_features": selector.ranking_[:actual_k],
        "discovery_nested_oof": _metrics(y_train, oof, threshold, config.get("bootstrap_resamples", 0), config["random_seed"]),
        "validation_locked": _metrics(y_val, val_scores, threshold, config.get("bootstrap_resamples", 0), config["random_seed"]),
        "outer_fold_choices": outer_choices, "inner_cv_tuning": tuning_rows,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"feature": selector.ranking_, "mean_abs_shap_discovery": [selector.shap_values_[c] for c in selector.ranking_]}).to_csv(output / "discovery_feature_ranking.csv", index=False)
    pd.DataFrame({"subject": subset.loc[train_mask, "subject"].to_numpy(), "y_true": y_train, "score": oof}).to_csv(output / "discovery_oof_predictions.csv", index=False)
    pd.DataFrame({"subject": subset.loc[validation_mask, "subject"].to_numpy(), "y_true": y_val, "score": val_scores}).to_csv(output / "validation_predictions.csv", index=False)
    joblib.dump({"selector": selector, "model": model, "feature_count": actual_k, "threshold": threshold, "outcome": outcome}, output / "model.joblib")
    return metrics


def run_experiments(data: Path, config_path: Path, output: Path, profiles: list[str]) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    frame = load_table(data)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    unknown = set(profiles) - set(config["profiles"])
    if unknown:
        raise ValueError(f"Unknown profiles: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "profiles": profiles,
        "cohort_counts": frame.cohort.value_counts().to_dict(),
        "validation_policy": "Labels in validation are used only after all model choices and the threshold are locked",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in profiles:
        profile = config["profiles"][name]
        results = {}
        for outcome in profile["outcomes"]:
            results[outcome] = run_one(frame, outcome, profile["tasks"], profile["feature_counts"], config, output / name / outcome)
        if name == "diagnostic_aid":
            primary = ["NCI_vs_nonNCI", "MCI_vs_nonMCI", "DE_vs_nonDE"]
        else:
            primary = list(profile["outcomes"])
        summary = {
            "primary_outcomes": primary,
            "discovery_macro_auc": float(np.mean([results[k]["discovery_nested_oof"]["auc"] for k in primary])),
            "validation_macro_auc": float(np.mean([results[k]["validation_locked"]["auc"] for k in primary])),
            "outcomes": {k: {"discovery_auc": v["discovery_nested_oof"]["auc"], "validation_auc": v["validation_locked"]["auc"]} for k, v in results.items()},
        }
        (output / name / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
