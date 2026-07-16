from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


RAW_SPIROMETRY_COLUMNS = ["fev1_ml", "fvc_ml"]

# For serving we default to the same behavior as training/evaluation.
DIAGNOSIS_DROP_RAW_SPIROMETRY = os.environ.get("COPD_DIAGNOSIS_DROP_RAW_SPIROMETRY", "1") == "1"
GOLD_DROP_RAW_SPIROMETRY = os.environ.get("COPD_GOLD_DROP_RAW_SPIROMETRY", "0") == "1"


@dataclass
class EnsembleBundle:
    feature_columns: list[str]
    fitted_base_models: dict[str, Any]
    fitted_meta_model: Any

    def _ensure_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.feature_columns, fill_value=0.0)

    def _base_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self._ensure_columns(X)
        return np.hstack([m.predict_proba(X) for m in self.fitted_base_models.values()])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta = self._base_proba(X)
        return self.fitted_meta_model.predict_proba(meta)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        meta = self._base_proba(X)
        return self.fitted_meta_model.predict(meta)


@dataclass
class DoubleTargetSystemBundle:
    diagnosis_ensemble: EnsembleBundle
    gold_ensemble: EnsembleBundle
    diagnosis_threshold: float
    diagnosis_label_encoder: LabelEncoder
    gold_label_encoder: LabelEncoder

    diagnosis_proba_column: str = "diagnosis_proba_copd"

    @property
    def copd_label_value(self) -> int:
        classes = list(self.diagnosis_label_encoder.classes_)
        if "copd" not in classes:
            return 1
        return int(classes.index("copd"))

    @property
    def no_copd_label_value(self) -> int:
        classes = list(self.diagnosis_label_encoder.classes_)
        if "no_copd" not in classes:
            return 0
        return int(classes.index("no_copd"))

    def _diagnosis_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_diag = (
            X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if DIAGNOSIS_DROP_RAW_SPIROMETRY
            else X
        )
        proba = self.diagnosis_ensemble.predict_proba(X_diag)
        idx = self.copd_label_value
        if idx < 0 or idx >= proba.shape[1]:
            idx = 1 if proba.shape[1] > 1 else 0
        return proba[:, idx]

    def predict_diagnosis_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_diag = (
            X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if DIAGNOSIS_DROP_RAW_SPIROMETRY
            else X
        )
        return self.diagnosis_ensemble.predict_proba(X_diag)

    def predict_diagnosis(self, X: pd.DataFrame) -> np.ndarray:
        p = self._diagnosis_proba(X)
        return np.where(p >= self.diagnosis_threshold, self.copd_label_value, self.no_copd_label_value)

    def predict_gold(self, X: pd.DataFrame) -> np.ndarray:
        gold_pred, _ = self._gold_pred_and_proba(X)
        return gold_pred

    def predict_gold_proba(self, X: pd.DataFrame) -> np.ndarray:
        _, proba = self._gold_pred_and_proba(X)
        return proba

    def _gold_pred_and_proba(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        diag_pred = self.predict_diagnosis(X)

        X_gold = X.copy()
        if GOLD_DROP_RAW_SPIROMETRY:
            X_gold = X_gold.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
        X_gold[self.diagnosis_proba_column] = self._diagnosis_proba(X)

        gold_pred = np.zeros(X.shape[0], dtype=int)
        proba = np.zeros((X.shape[0], 5), dtype=float)

        mask = np.asarray(diag_pred) == self.copd_label_value
        proba[~mask, 0] = 1.0

        if np.any(mask):
            sub_pred = self.gold_ensemble.predict(X_gold.loc[X_gold.index[mask]])
            gold_pred[mask] = np.asarray(sub_pred, dtype=int) + 1
            sub_proba = self.gold_ensemble.predict_proba(X_gold.loc[X_gold.index[mask]])
            proba[mask, 1:5] = sub_proba

        return gold_pred, proba


def load_ensemble_bundle(path: str) -> EnsembleBundle:
    bundle = joblib.load(os.path.join(path, "ensemble.joblib"))
    feature_columns = list(bundle.get("feature_columns") or [])
    fitted_base_models = bundle.get("fitted_base_models") or {}
    fitted_meta_model = bundle.get("fitted_meta_model")
    if fitted_meta_model is None or not fitted_base_models:
        raise ValueError(f"Invalid ensemble artifact at: {path}")
    return EnsembleBundle(
        feature_columns=feature_columns,
        fitted_base_models=fitted_base_models,
        fitted_meta_model=fitted_meta_model,
    )


def load_system_bundle(system_dir: str) -> DoubleTargetSystemBundle:
    """Load the saved system directory produced by the training DAG."""
    system_json_path = os.path.join(system_dir, "system.json")
    with open(system_json_path, "r", encoding="utf-8") as fh:
        sys_meta = json.load(fh)

    diagnosis_ensemble = load_ensemble_bundle(os.path.join(system_dir, "diagnosis_ensemble"))
    gold_ensemble = load_ensemble_bundle(os.path.join(system_dir, "gold_ensemble"))

    diag_le = LabelEncoder()
    gold_le = LabelEncoder()

    diag_meta = sys_meta.get("diagnosis_label_encoder") or {}
    gold_meta = sys_meta.get("gold_label_encoder") or {}

    diag_classes = diag_meta.get("classes")
    gold_classes = gold_meta.get("classes")

    if not diag_classes or not gold_classes:
        raise ValueError("Label encoders are missing from system.json; retrain with encoders persisted")

    diag_le.classes_ = np.array(diag_classes)
    gold_le.classes_ = np.array(gold_classes)

    threshold = float(sys_meta.get("diagnosis_threshold", 0.5))

    return DoubleTargetSystemBundle(
        diagnosis_ensemble=diagnosis_ensemble,
        gold_ensemble=gold_ensemble,
        diagnosis_threshold=threshold,
        diagnosis_label_encoder=diag_le,
        gold_label_encoder=gold_le,
        diagnosis_proba_column=str(sys_meta.get("diagnosis_proba_column", "diagnosis_proba_copd")),
    )
