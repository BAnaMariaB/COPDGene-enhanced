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


def _align_model_input(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return X
    return X.reindex(columns=list(feature_names), fill_value=0.0)


@dataclass
class EnsembleBundle:
    feature_columns: list[str]
    fitted_base_models: dict[str, Any]
    fitted_meta_model: Any

    def _ensure_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.feature_columns, fill_value=0.0)
    @staticmethod
    def _model_predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X))
            if scores.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1.0 - p1, p1])
            scores = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            denom = exp_scores.sum(axis=1, keepdims=True)
            denom[denom == 0.0] = 1.0
            return exp_scores / denom
        raise ValueError(f"Model {type(model).__name__} has neither predict_proba nor decision_function")

    def _base_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self._ensure_columns(X)
        return np.hstack([self._model_predict_proba(m, X) for m in self.fitted_base_models.values()])

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

    @property
    def copd_class_index(self) -> int:
        # For the stacked ensemble bundle, proba columns are in encoded-label order.
        return self.copd_label_value

    def _diagnosis_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_diag = (
            X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if DIAGNOSIS_DROP_RAW_SPIROMETRY
            else X
        )
        proba = self.diagnosis_ensemble.predict_proba(X_diag)
        idx = self.copd_class_index
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
            model_classes = getattr(self.gold_ensemble.fitted_meta_model, "classes_", None)
            sub_proba_aligned = np.zeros((sub_proba.shape[0], 4), dtype=float)
            if model_classes is None:
                width = min(sub_proba.shape[1], 4)
                sub_proba_aligned[:, :width] = sub_proba[:, :width]
            else:
                classes = [int(c) for c in np.asarray(model_classes).tolist()]
                for idx, cls in enumerate(classes):
                    if 0 <= cls < 4 and idx < sub_proba.shape[1]:
                        sub_proba_aligned[:, cls] = sub_proba[:, idx]
            proba[mask, 1:5] = sub_proba_aligned

        return gold_pred, proba


@dataclass
class BestOfBothSystemBundle:
    diagnosis_model: Any
    gold_model: Any
    diagnosis_threshold: float
    diagnosis_label_encoder: LabelEncoder
    gold_label_encoder: LabelEncoder

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

    @property
    def copd_class_index(self) -> int:
        if hasattr(self.diagnosis_model, "classes_"):
            classes = list(self.diagnosis_model.classes_)
            label = self.copd_label_value
            if label in classes:
                return int(classes.index(label))
        return 1

    def _diagnosis_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_diag = (
            X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if DIAGNOSIS_DROP_RAW_SPIROMETRY
            else X
        )
        X_diag = _align_model_input(self.diagnosis_model, X_diag)
        proba = self.diagnosis_model.predict_proba(X_diag)
        idx = self.copd_class_index
        if idx < 0 or idx >= proba.shape[1]:
            idx = 1 if proba.shape[1] > 1 else 0
        return proba[:, idx]

    def predict_diagnosis_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_diag = (
            X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if DIAGNOSIS_DROP_RAW_SPIROMETRY
            else X
        )
        X_diag = _align_model_input(self.diagnosis_model, X_diag)
        return self.diagnosis_model.predict_proba(X_diag)

    def predict_diagnosis(self, X: pd.DataFrame) -> np.ndarray:
        p = self._diagnosis_proba(X)
        return np.where(p >= self.diagnosis_threshold, self.copd_label_value, self.no_copd_label_value)

    def _gold_pred_and_proba(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        diag_pred = self.predict_diagnosis(X)
        X_gold = (
            X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if GOLD_DROP_RAW_SPIROMETRY
            else X
        )
        X_gold = _align_model_input(self.gold_model, X_gold)

        gold_pred = np.zeros(X.shape[0], dtype=int)
        proba = np.zeros((X.shape[0], 5), dtype=float)

        mask = np.asarray(diag_pred) == self.copd_label_value
        proba[~mask, 0] = 1.0

        if np.any(mask):
            sub_pred = self.gold_model.predict(X_gold.loc[X_gold.index[mask]])
            gold_pred[mask] = np.asarray(sub_pred, dtype=int) + 1
            sub_proba = self.gold_model.predict_proba(X_gold.loc[X_gold.index[mask]])
            proba[mask, 1:5] = sub_proba

        return gold_pred, proba

    def predict_gold(self, X: pd.DataFrame) -> np.ndarray:
        gold_pred, _ = self._gold_pred_and_proba(X)
        return gold_pred

    def predict_gold_proba(self, X: pd.DataFrame) -> np.ndarray:
        _, proba = self._gold_pred_and_proba(X)
        return proba


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


def load_system_bundle(system_dir: str) -> DoubleTargetSystemBundle | BestOfBothSystemBundle:
    """Load a saved system bundle directory produced by the training DAG."""
    system_json_path = os.path.join(system_dir, "system.json")
    with open(system_json_path, "r", encoding="utf-8") as fh:
        sys_meta = json.load(fh)

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

    if sys_meta.get("bundle_type") == "best_of_both_v1":
        diagnosis_model = joblib.load(os.path.join(system_dir, "diagnosis_model.joblib"))
        gold_model = joblib.load(os.path.join(system_dir, "gold_model.joblib"))
        return BestOfBothSystemBundle(
            diagnosis_model=diagnosis_model,
            gold_model=gold_model,
            diagnosis_threshold=threshold,
            diagnosis_label_encoder=diag_le,
            gold_label_encoder=gold_le,
        )

    diagnosis_ensemble = load_ensemble_bundle(os.path.join(system_dir, "diagnosis_ensemble"))
    gold_ensemble = load_ensemble_bundle(os.path.join(system_dir, "gold_ensemble"))

    return DoubleTargetSystemBundle(
        diagnosis_ensemble=diagnosis_ensemble,
        gold_ensemble=gold_ensemble,
        diagnosis_threshold=threshold,
        diagnosis_label_encoder=diag_le,
        gold_label_encoder=gold_le,
        diagnosis_proba_column=str(sys_meta.get("diagnosis_proba_column", "diagnosis_proba_copd")),
    )
