"""Boruta and TreeSHAP fitted solely on the supplied training rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class FeatureSelector:
    def __init__(self, seed: int, missing_fraction_max: float, variance_threshold: float, boruta_max_iter: int, n_jobs: int):
        self.seed = seed
        self.missing_fraction_max = missing_fraction_max
        self.variance_threshold = variance_threshold
        self.boruta_max_iter = boruta_max_iter
        self.n_jobs = n_jobs

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "FeatureSelector":
        usable = x.columns[x.isna().mean().le(self.missing_fraction_max) & x.notna().any()]
        if len(usable) == 0:
            raise ValueError("All features exceed the missingness limit")
        self.input_columns_ = list(x.columns)
        self.kept_columns_ = list(usable)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        imputed = self.imputer_.fit_transform(x[self.kept_columns_])
        self.variance_ = VarianceThreshold(self.variance_threshold)
        filtered = self.variance_.fit_transform(imputed)
        self.variance_columns_ = list(np.asarray(self.kept_columns_)[self.variance_.get_support()])
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(filtered)
        support = np.zeros(len(self.variance_columns_), dtype=bool)
        tasks = sorted({column.split("__", 1)[0] for column in self.variance_columns_})
        for task in tasks:
            positions = [i for i, column in enumerate(self.variance_columns_) if column.startswith(f"{task}__")]
            forest = RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", n_jobs=self.n_jobs, random_state=self.seed)
            boruta = BorutaPy(forest, n_estimators="auto", max_iter=self.boruta_max_iter, random_state=self.seed, verbose=0)
            boruta.fit(scaled[:, positions], y)
            support[positions] = boruta.support_ | boruta.support_weak_
        if not support.any():
            raise ValueError("Boruta selected no features in this training fold")
        self.boruta_columns_ = list(np.asarray(self.variance_columns_)[support])
        selected = scaled[:, support]
        model = xgb.XGBClassifier(**xgb_parameters(self.seed, self.n_jobs))
        model.fit(selected, y)
        contrib = model.get_booster().predict(xgb.DMatrix(selected), pred_contribs=True)
        importance = np.abs(np.asarray(contrib)[:, :-1]).mean(axis=0)
        order = sorted(range(len(importance)), key=lambda i: (-importance[i], self.boruta_columns_[i]))
        self.ranking_ = [self.boruta_columns_[i] for i in order]
        self.shap_values_ = {self.boruta_columns_[i]: float(importance[i]) for i in order}
        return self

    def transform(self, x: pd.DataFrame, k: int) -> np.ndarray:
        missing = set(self.input_columns_) - set(x.columns)
        if missing:
            raise ValueError(f"Missing model input columns: {sorted(missing)[:5]}")
        imputed = self.imputer_.transform(x[self.kept_columns_])
        scaled = self.scaler_.transform(self.variance_.transform(imputed))
        positions = [self.variance_columns_.index(c) for c in self.ranking_[:k]]
        return scaled[:, positions].astype(np.float32)


def xgb_parameters(seed: int, n_jobs: int) -> dict:
    return {
        "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
        "n_estimators": 300, "learning_rate": 0.03, "max_depth": 3,
        "min_child_weight": 2.0, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 2.0, "max_bin": 128,
        "random_state": seed, "n_jobs": n_jobs, "verbosity": 0,
    }
