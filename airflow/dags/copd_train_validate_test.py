"""
COPD model training / validation / testing DAG.

SCOPE: train a single double-target ensemble system that predicts both COPD
diagnosis and GOLD stage from the NHANES-derived preprocessed dataset.

Targets:
  - copd_diagnosis : binary (0 = no COPD, 1 = COPD), based on FEV1/FVC < 0.70
  - gold_stage       : multi-class (0 = no COPD, 1 = GOLD 1, 2 = GOLD 2,
                       3 = GOLD 3, 4 = GOLD 4), based on FEV1 % predicted

The model is one `COPDDoubleTargetSystem` object that owns two
`COPDEnsembleClassifier` instances: one for the diagnosis task and one for the GOLD
stage task. The two classifiers are connected because the GOLD-stage ensemble
receives the predicted COPD-diagnosis probability as an additional input feature,
so inference always flows through the diagnosis ensemble first and then the GOLD
stage ensemble. Each base classifier is trained on the same data, and a
LogisticRegression meta-learner combines their predicted class probabilities. All
work is tracked with MLflow.

Modeling strategy:
  1. Load the NHANES-derived preprocessed dataset for the DAG run date.
  2. Split into train / validation / test (70 / 15 / 15), stratified on both targets.
  3. Encode both targets with `sklearn.preprocessing.LabelEncoder`.
  4. Instantiate one `COPDDoubleTargetSystem` with two ensemble classifiers.
  5. Fit the system: the diagnosis ensemble is trained first, then the GOLD-stage
     ensemble is trained using the original features plus the predicted diagnosis
     probability from the validation set.
  6. Evaluate the full system on the held-out test set for both targets using
     accuracy, precision, recall, f1_macro, and roc_auc_ovr.
  7. Persist two champion records (diagnosis + GOLD stage) ready for a PostgreSQL
     champion table later.

MLflow setup:
  - Tracking URI: `MLFLOW_TRACKING_URI` env var, defaulting to a local SQLite
    DB (`sqlite:///.../mlflow.db`).
  - Experiment name: `MLFLOW_EXPERIMENT_NAME` env var, defaulting to
    `copd_nhanes_double_target_classification`.
  - S3 artifact store: set the standard MLflow S3 env vars (no code changes
    required). See README.md for the exact variables.

Artifact layout (per run date):
  data/artifacts/<YYYY-MM-DD>/
  ├── models/
  │   └── copd_double_target_system/
  │       ├── system.joblib
  │       ├── diagnosis_ensemble/
  │       ├── gold_ensemble/
  │       └── label_encoders.json
  ├── splits/
  │   ├── X_train.csv, X_val.csv, X_test.csv
  │   ├── y_diagnosis_*.npy, y_gold_*.npy
  │   └── label_encoders.json
  ├── test_metrics.json
  └── champion_diagnosis.json
  └── champion_gold_stage.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from airflow.sdk import dag, get_current_context, task
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", os.path.expanduser("~/airflow"))
PREPROCESSED_ROOT = os.environ.get(
    "COPD_PREPROCESSED_ROOT", os.path.join(AIRFLOW_HOME, "data", "preprocessed")
)
ARTIFACT_ROOT = os.environ.get(
    "COPD_ARTIFACT_ROOT", os.path.join(AIRFLOW_HOME, "data", "artifacts")
)
# Target columns in the NHANES-derived preprocessed dataset.
DIAGNOSIS_TARGET_COLUMN = os.environ.get("COPD_DIAGNOSIS_TARGET_COLUMN", "copd_diagnosis")
GOLD_TARGET_COLUMN = os.environ.get("COPD_GOLD_TARGET_COLUMN", "gold_stage")
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "sqlite:///"
    + os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mlflow.db"
    ),
)
MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    os.environ.get("COPD_MLFLOW_EXPERIMENT_NAME", "copd_nhanes_double_target_classification"),
)

# Reproducible train/val/test split sizes.
TEST_SIZE = float(os.environ.get("COPD_TEST_SIZE", "0.15"))
VAL_SIZE = float(os.environ.get("COPD_VAL_SIZE", "0.15"))
RANDOM_STATE = int(os.environ.get("COPD_RANDOM_STATE", "42"))

# Stacking setting: train the meta-model on out-of-fold (OOF) base predictions.
# Set COPD_OOF_FOLDS>1 to enable.
OOF_FOLDS = int(os.environ.get("COPD_OOF_FOLDS", "1"))

# Sample-weight controls.
# - Diagnosis is extremely imbalanced (no COPD is the majority). Using fully-balanced
#   weights can cause over-prediction of COPD, which severely hurts overall GOLD metrics.
DIAGNOSIS_USE_SAMPLE_WEIGHTS = os.environ.get("COPD_DIAGNOSIS_USE_SAMPLE_WEIGHTS", "0") == "1"
GOLD_USE_SAMPLE_WEIGHTS = os.environ.get("COPD_GOLD_USE_SAMPLE_WEIGHTS", "1") == "1"

# Optional tuning of the diagnosis probability threshold to improve downstream GOLD-stage
# metrics (since GOLD_0 vs GOLD_1..4 depends heavily on diagnosis gating).
TUNE_DIAGNOSIS_THRESHOLD = os.environ.get("COPD_TUNE_DIAGNOSIS_THRESHOLD", "1") == "1"
DIAGNOSIS_THRESHOLD_GRID = [
    float(x)
    for x in os.environ.get(
        "COPD_DIAGNOSIS_THRESHOLD_GRID",
        "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
    ).split(",")
    if x.strip()
]
TUNE_DIAGNOSIS_THRESHOLD_FOR = os.environ.get(
    "COPD_TUNE_DIAGNOSIS_THRESHOLD_FOR", "gold_stage_f1_macro"
)

# Human-readable class labels for each target. The LabelEncoder determines the
# integer encoding from the observed values in the partition.
DIAGNOSIS_CLASS_LABELS = ["no_copd", "copd"]
GOLD_CLASS_LABELS = ["GOLD_0", "GOLD_1", "GOLD_2", "GOLD_3", "GOLD_4"]

# Leakage guard: these columns are used to DEFINE the targets. If kept as
# features, the task becomes close to deterministic and yields misleadingly high
# scores.
#
# We ALWAYS drop target-definition columns (ratio and % predicted).
#
# For diagnosis, raw spirometry measurements are also dropped by default to avoid
# a trivial reconstruction of the COPD criterion from FEV1/FVC.
# GOLD staging is allowed to use raw spirometry (it is clinically defined from
# spirometry), but still must not use the already-derived %pred/ratio columns.
TARGET_DEFINITION_COLUMNS = [
    "fev1_fvc_ratio",
    "fev1_pct_predicted",
]
RAW_SPIROMETRY_COLUMNS = [
    "fev1_ml",
    "fvc_ml",
]
DIAGNOSIS_DROP_RAW_SPIROMETRY = os.environ.get(
    "COPD_DIAGNOSIS_DROP_RAW_SPIROMETRY", "1"
) == "1"
GOLD_DROP_RAW_SPIROMETRY = os.environ.get(
    "COPD_GOLD_DROP_RAW_SPIROMETRY", "0"
) == "1"

# Default base classifiers for the single ensemble system. They are all trained on
# the same data and are always used together during inference.
DEFAULT_BASE_MODELS = {
    "catboost": CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        verbose=False,
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
    ),
    # NOTE: objective/eval_metric are intentionally NOT hardcoded. XGBClassifier
    # auto-infers binary:logistic for 2 classes and multi:softprob for >2 classes,
    # and sets num_class automatically. Hardcoding multi:softprob makes XGBoost
    # require an explicit num_class even for the binary diagnosis target.
    "xgboost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=2,
    ),
    "logistic_regression": LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE
    ),
}
DEFAULT_META_MODEL = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _partition_paths(ds: str) -> dict[str, str]:
    """Return deterministic input/output paths for a given partition date."""
    preprocessed_dir = os.path.join(PREPROCESSED_ROOT, ds)
    artifact_dir = os.path.join(ARTIFACT_ROOT, ds)
    return {
        "preprocessed_dir": preprocessed_dir,
        "preprocessed_csv": os.path.join(preprocessed_dir, "central_preprocessed_dataset.csv"),
        "artifact_dir": artifact_dir,
        "models_dir": os.path.join(artifact_dir, "models"),
        "system_dir": os.path.join(artifact_dir, "models", "copd_double_target_system"),
        "diagnosis_ensemble_dir": os.path.join(artifact_dir, "models", "copd_double_target_system", "diagnosis_ensemble"),
        "gold_ensemble_dir": os.path.join(artifact_dir, "models", "copd_double_target_system", "gold_ensemble"),
        "metrics_path": os.path.join(artifact_dir, "test_metrics.json"),
        "champion_diagnosis_path": os.path.join(artifact_dir, "champion_diagnosis.json"),
        "champion_gold_path": os.path.join(artifact_dir, "champion_gold_stage.json"),
    }


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list[int] | None = None,
) -> dict[str, float]:
    """Compute classification metrics used for model evaluation.

    `labels` can be provided to ensure macro metrics are computed over a fixed
    label set (e.g. GOLD_0..GOLD_4), even if some classes are missing from a
    particular split.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
    }
    if y_proba is not None and len(np.unique(y_true)) >= 2:
        try:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
            if np.isfinite(auc):
                metrics["roc_auc_ovr"] = auc
        except ValueError:
            pass
    return metrics


def _build_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, LabelEncoder, LabelEncoder]:
    """Extract the two target columns from the NHANES-derived preprocessed dataset.

    Returns:
        X: feature DataFrame with both targets removed
        y_diagnosis: encoded diagnosis labels
        y_gold: encoded GOLD stage labels
        diagnosis_encoder: fitted LabelEncoder for diagnosis
        gold_encoder: fitted LabelEncoder for GOLD stage
    """
    missing = [col for col in (DIAGNOSIS_TARGET_COLUMN, GOLD_TARGET_COLUMN) if col not in df.columns]
    if missing:
        raise ValueError(f"Target columns not found in preprocessed data: {missing}")

    df = df.copy()

    # Use string labels for the encoders to keep human-readable class names.
    y_diagnosis_labels = df[DIAGNOSIS_TARGET_COLUMN].apply(lambda v: DIAGNOSIS_CLASS_LABELS[int(v)] if int(v) < len(DIAGNOSIS_CLASS_LABELS) else str(v))
    y_gold_labels = df[GOLD_TARGET_COLUMN].apply(lambda v: GOLD_CLASS_LABELS[int(v)] if int(v) < len(GOLD_CLASS_LABELS) else str(v))

    X = df.drop(
        columns=[DIAGNOSIS_TARGET_COLUMN, GOLD_TARGET_COLUMN, *TARGET_DEFINITION_COLUMNS],
        errors="ignore",
    )

    diagnosis_encoder = LabelEncoder()
    diagnosis_encoder.fit(y_diagnosis_labels)
    y_diagnosis = pd.Series(diagnosis_encoder.transform(y_diagnosis_labels), index=df.index)

    gold_encoder = LabelEncoder()
    gold_encoder.fit(y_gold_labels)
    y_gold = pd.Series(gold_encoder.transform(y_gold_labels), index=df.index)

    return X, y_diagnosis, y_gold, diagnosis_encoder, gold_encoder


def _setup_mlflow(dag_run_id: str | None) -> tuple[str, str]:
    """Set MLflow tracking URI/experiment and return the experiment id."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id if experiment else "unknown"
    return experiment_id, dag_run_id or "unknown"


# ---------------------------------------------------------------------------
# Ensemble classifier: single system that owns and connects all base classifiers
# ---------------------------------------------------------------------------

class COPDEnsembleClassifier:
    """Single ensemble system that combines multiple base classifiers.

    All base classifiers are trained on the same (X_train, y_train). The
    meta-learner is then trained on the base classifiers' predicted class
    probabilities on a separate validation set, so inference always flows through
    every base classifier and the meta-learner.
    """

    def __init__(
        self,
        base_models: dict[str, Any] | None = None,
        meta_model: Any | None = None,
        random_state: int = RANDOM_STATE,
        use_sample_weights: bool = True,
    ) -> None:
        self.base_model_configs = base_models or DEFAULT_BASE_MODELS
        self.meta_model_config = meta_model or DEFAULT_META_MODEL
        self.random_state = random_state
        self.use_sample_weights = use_sample_weights

        self.fitted_base_models: dict[str, Any] = {}
        self.fitted_meta_model: Any | None = None
        self.label_encoder: LabelEncoder | None = None
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> "COPDEnsembleClassifier":
        """Train all base classifiers on the same data, then fit the meta-learner."""
        # Store the feature column order for inference reproducibility.
        self.feature_columns = list(X_train.columns)

        sw_train = None
        sw_val = None
        if self.use_sample_weights:
            # Use balanced sample weights to reduce the impact of strong class imbalance.
            sw_train = compute_sample_weight(class_weight="balanced", y=y_train)
            sw_val = compute_sample_weight(class_weight="balanced", y=y_val)

        # Default: meta-model is trained on base probabilities on a held-out validation set.
        # If OOF_FOLDS>1, train the meta-model on out-of-fold predictions computed on the
        # training set (stacking), and keep the external validation split for evaluation.
        use_oof = OOF_FOLDS > 1

        if use_oof:
            y_train_arr = np.asarray(y_train)
            classes, counts = np.unique(y_train_arr, return_counts=True)
            min_class = int(counts.min()) if len(counts) else 0
            n_splits = max(1, min(OOF_FOLDS, min_class))
            if n_splits < 2:
                use_oof = False
            else:
                self.oof_n_splits = n_splits
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        oof_blocks: list[np.ndarray] = []

        # Fit base models (and optionally compute OOF probabilities).
        for name, model in self.base_model_configs.items():
            if use_oof:
                # Determine the number of classes for a consistent proba matrix.
                n_classes = int(len(np.unique(y_train_arr)))
                oof_proba = np.zeros((X_train.shape[0], n_classes), dtype=float)
                for tr_idx, hold_idx in skf.split(X_train, y_train_arr):
                    fold_model = clone(model) if hasattr(model, "get_params") else model
                    X_tr = X_train.iloc[tr_idx]
                    y_tr = y_train_arr[tr_idx]
                    if sw_train is not None:
                        try:
                            fold_model.fit(X_tr, y_tr, sample_weight=sw_train[tr_idx])
                        except TypeError:
                            fold_model.fit(X_tr, y_tr)
                    else:
                        fold_model.fit(X_tr, y_tr)
                    oof_proba[hold_idx, :] = fold_model.predict_proba(X_train.iloc[hold_idx])
                oof_blocks.append(oof_proba)

            fitted = clone(model) if hasattr(model, "get_params") else model
            if sw_train is not None:
                try:
                    fitted.fit(X_train, y_train, sample_weight=sw_train)
                except TypeError:
                    fitted.fit(X_train, y_train)
            else:
                fitted.fit(X_train, y_train)
            self.fitted_base_models[name] = fitted

        # Fit the meta-learner.
        self.fitted_meta_model = (
            clone(self.meta_model_config)
            if hasattr(self.meta_model_config, "get_params")
            else self.meta_model_config
        )

        if use_oof:
            meta_features = np.hstack(oof_blocks)
            if sw_train is not None:
                try:
                    self.fitted_meta_model.fit(meta_features, y_train_arr, sample_weight=sw_train)
                except TypeError:
                    self.fitted_meta_model.fit(meta_features, y_train_arr)
            else:
                self.fitted_meta_model.fit(meta_features, y_train_arr)
        else:
            meta_features = self._base_proba(X_val)
            if sw_val is not None:
                try:
                    self.fitted_meta_model.fit(meta_features, y_val, sample_weight=sw_val)
                except TypeError:
                    self.fitted_meta_model.fit(meta_features, y_val)
            else:
                self.fitted_meta_model.fit(meta_features, y_val)

        self.is_fitted = True
        return self

    def _ensure_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reindex X to match the feature column order used during training."""
        if not hasattr(self, "feature_columns"):
            return X
        return X.reindex(columns=self.feature_columns, fill_value=0.0)

    def _base_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return a matrix where each block is one base classifier's class probabilities."""
        if not self.fitted_base_models:
            raise ValueError("Base classifiers have not been fitted yet")
        X = self._ensure_columns(X)
        return np.hstack([model.predict_proba(X) for model in self.fitted_base_models.values()])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Run inference through every base classifier, then the meta-learner."""
        if not self.is_fitted:
            raise ValueError("Ensemble has not been fitted")
        meta_features = self._base_proba(X)
        return self.fitted_meta_model.predict_proba(meta_features)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels (as encoded integers)."""
        if not self.is_fitted:
            raise ValueError("Ensemble has not been fitted")
        meta_features = self._base_proba(X)
        # Use the meta-model's class predictions directly (do NOT argmax), because
        # class labels are not guaranteed to be 0..K-1 in all training setups.
        return self.fitted_meta_model.predict(meta_features)

    def predict_labels(self, X: pd.DataFrame) -> np.ndarray:
        """Return human-readable class labels using the stored LabelEncoder."""
        encoded = self.predict(X)
        if self.label_encoder is None:
            return encoded
        return self.label_encoder.inverse_transform(encoded)

    def base_model_metrics(
        self, X: pd.DataFrame, y: np.ndarray, labels: list[int] | None = None
    ) -> dict[str, dict[str, float]]:
        """Evaluate each fitted base classifier individually."""
        X = self._ensure_columns(X)
        return {
            name: _classification_metrics(y, model.predict(X), model.predict_proba(X), labels=labels)
            for name, model in self.fitted_base_models.items()
        }

    def metrics(self, X: pd.DataFrame, y: np.ndarray, labels: list[int] | None = None) -> dict[str, float]:
        """Evaluate the full ensemble on (X, y)."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        return _classification_metrics(y, y_pred, y_proba, labels=labels)

    def save(self, path: str) -> None:
        """Persist the entire ensemble classifier system as one artifact."""
        _ensure_dir(path)
        bundle = {
            "base_model_configs": self.base_model_configs,
            "meta_model_config": self.meta_model_config,
            "random_state": self.random_state,
            "feature_columns": getattr(self, "feature_columns", None),
            "fitted_base_models": self.fitted_base_models,
            "fitted_meta_model": self.fitted_meta_model,
            "label_encoder": self.label_encoder,
            "is_fitted": self.is_fitted,
            "base_model_names": list(self.fitted_base_models.keys()),
        }
        joblib.dump(bundle, os.path.join(path, "ensemble.joblib"))
        with open(os.path.join(path, "base_models.json"), "w", encoding="utf-8") as fh:
            json.dump({"base_models": list(self.fitted_base_models.keys())}, fh, indent=2)
        if self.label_encoder is not None:
            le_dict = {
                "classes": self.label_encoder.classes_.tolist(),
                "encoded_labels": list(range(len(self.label_encoder.classes_))),
            }
            with open(os.path.join(path, "label_encoder.json"), "w", encoding="utf-8") as fh:
                json.dump(le_dict, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "COPDEnsembleClassifier":
        """Load a previously saved ensemble classifier system."""
        bundle = joblib.load(os.path.join(path, "ensemble.joblib"))
        instance = cls(
            base_models=bundle["base_model_configs"],
            meta_model=bundle["meta_model_config"],
            random_state=bundle["random_state"],
        )
        instance.feature_columns = bundle.get("feature_columns")
        instance.fitted_base_models = bundle["fitted_base_models"]
        instance.fitted_meta_model = bundle["fitted_meta_model"]
        instance.label_encoder = bundle.get("label_encoder")
        instance.is_fitted = bundle["is_fitted"]
        return instance


# ---------------------------------------------------------------------------
# Double-target system: connects diagnosis and GOLD stage ensembles
# ---------------------------------------------------------------------------

class COPDDoubleTargetSystem:
    """Single system that owns a COPD diagnosis ensemble and a GOLD-stage ensemble.

    The diagnosis ensemble is trained first. The GOLD-stage ensemble is then
    trained on the original features plus the predicted COPD-diagnosis probability,
    so the two models are connected and inference always flows through the
    diagnosis ensemble before the GOLD-stage ensemble.
    """

    def __init__(
        self,
        diagnosis_base_models: dict[str, Any] | None = None,
        gold_base_models: dict[str, Any] | None = None,
        meta_model: Any | None = None,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.diagnosis_ensemble = COPDEnsembleClassifier(
            base_models=diagnosis_base_models or DEFAULT_BASE_MODELS,
            meta_model=meta_model or DEFAULT_META_MODEL,
            random_state=random_state,
            use_sample_weights=DIAGNOSIS_USE_SAMPLE_WEIGHTS,
        )
        self.gold_ensemble = COPDEnsembleClassifier(
            base_models=gold_base_models or DEFAULT_BASE_MODELS,
            meta_model=meta_model or DEFAULT_META_MODEL,
            random_state=random_state,
            use_sample_weights=GOLD_USE_SAMPLE_WEIGHTS,
        )
        self.diagnosis_threshold: float = 0.5
        self.diagnosis_label_encoder: LabelEncoder | None = None
        self.gold_label_encoder: LabelEncoder | None = None
        self.diagnosis_proba_column = "diagnosis_proba_copd"
        # Index of the COPD class in diagnosis_ensemble.predict_proba output.
        # Determined from diagnosis_label_encoder when available.
        self.copd_class_index: int = 1
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_diagnosis_train: np.ndarray,
        y_gold_train: np.ndarray,
        X_val: pd.DataFrame,
        y_diagnosis_val: np.ndarray,
        y_gold_val: np.ndarray,
    ) -> "COPDDoubleTargetSystem":
        """Train the diagnosis ensemble first, then the GOLD-stage ensemble."""
        X_train_diag = X_train.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore") if DIAGNOSIS_DROP_RAW_SPIROMETRY else X_train
        X_val_diag = X_val.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore") if DIAGNOSIS_DROP_RAW_SPIROMETRY else X_val
        self.diagnosis_ensemble.fit(X_train_diag, y_diagnosis_train, X_val_diag, y_diagnosis_val)

        # Determine which probability column corresponds to COPD.
        self.copd_label_value = 1
        self.no_copd_label_value = 0
        if self.diagnosis_label_encoder is not None and hasattr(self.diagnosis_label_encoder, "classes_"):
            classes = list(self.diagnosis_label_encoder.classes_)
            if "copd" in classes:
                self.copd_class_index = int(classes.index("copd"))
                self.copd_label_value = int(self.diagnosis_label_encoder.transform(["copd"])[0])
            if "no_copd" in classes:
                self.no_copd_label_value = int(self.diagnosis_label_encoder.transform(["no_copd"])[0])

        # Augment the GOLD-stage feature space with the predicted COPD probability.
        X_train_gold = X_train.copy()
        X_val_gold = X_val.copy()
        if GOLD_DROP_RAW_SPIROMETRY:
            X_train_gold = X_train_gold.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            X_val_gold = X_val_gold.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
        X_train_gold[self.diagnosis_proba_column] = self._diagnosis_proba(X_train)
        X_val_gold[self.diagnosis_proba_column] = self._diagnosis_proba(X_val)

        # Train GOLD-stage model only on true COPD rows; GOLD_0 is a "no COPD" sentinel.
        train_mask = (np.asarray(y_diagnosis_train) == self.copd_label_value) & (np.asarray(y_gold_train) > 0)
        val_mask = (np.asarray(y_diagnosis_val) == self.copd_label_value) & (np.asarray(y_gold_val) > 0)

        y_gold_train_sub = np.asarray(y_gold_train)[train_mask] - 1
        y_gold_val_sub = np.asarray(y_gold_val)[val_mask] - 1

        self.gold_ensemble.fit(
            X_train_gold.loc[X_train_gold.index[train_mask]],
            y_gold_train_sub,
            X_val_gold.loc[X_val_gold.index[val_mask]],
            y_gold_val_sub,
        )

        if TUNE_DIAGNOSIS_THRESHOLD:
            self._tune_diagnosis_threshold(X_val, y_diagnosis_val, y_gold_val)

        self.is_fitted = True
        return self

    def _diagnosis_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the probability of COPD from the diagnosis ensemble."""
        X_diag = X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore") if DIAGNOSIS_DROP_RAW_SPIROMETRY else X
        proba = self.diagnosis_ensemble.predict_proba(X_diag)
        idx = self.copd_class_index
        if idx < 0 or idx >= proba.shape[1]:
            idx = 1 if proba.shape[1] > 1 else 0
        return proba[:, idx]

    def _predict_diagnosis_with_threshold(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        proba = self._diagnosis_proba(X)
        # In the LabelEncoder for diagnosis, COPD is identified by the string label "copd".
        # We treat `proba` as P(COPD) and apply a tunable threshold.
        return np.where(
            proba >= threshold,
            getattr(self, "copd_label_value", 1),
            getattr(self, "no_copd_label_value", 0),
        )

    def predict_diagnosis(self, X: pd.DataFrame) -> np.ndarray:
        """Return encoded diagnosis predictions."""
        # For binary diagnosis, use probability-thresholding (tunable) rather than
        # the meta-model's default decision threshold.
        return self._predict_diagnosis_with_threshold(X, getattr(self, "diagnosis_threshold", 0.5))

    def predict_gold(self, X: pd.DataFrame) -> np.ndarray:
        """Return encoded GOLD-stage predictions (GOLD_0..GOLD_4)."""
        gold_pred, _ = self._gold_pred_and_proba(X, getattr(self, "diagnosis_threshold", 0.5))
        return gold_pred

    def predict_diagnosis_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return full diagnosis class probabilities."""
        X_diag = X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore") if DIAGNOSIS_DROP_RAW_SPIROMETRY else X
        return self.diagnosis_ensemble.predict_proba(X_diag)

    def _gold_pred_and_proba(
        self, X: pd.DataFrame, diagnosis_threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        diag_pred = self._predict_diagnosis_with_threshold(X, diagnosis_threshold)
        X_gold = X.copy()
        if GOLD_DROP_RAW_SPIROMETRY:
            X_gold = X_gold.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
        X_gold[self.diagnosis_proba_column] = self._diagnosis_proba(X)

        gold_pred = np.zeros(X.shape[0], dtype=int)
        proba = np.zeros((X.shape[0], 5), dtype=float)
        mask = np.asarray(diag_pred) == getattr(self, "copd_label_value", 1)
        proba[~mask, 0] = 1.0
        if np.any(mask):
            sub_pred = self.gold_ensemble.predict(X_gold.loc[X_gold.index[mask]])
            gold_pred[mask] = np.asarray(sub_pred, dtype=int) + 1
            sub_proba = self.gold_ensemble.predict_proba(X_gold.loc[X_gold.index[mask]])
            proba[mask, 1:5] = sub_proba
        return gold_pred, proba

    def predict_gold_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return full 5-class GOLD-stage probabilities (GOLD_0..GOLD_4)."""
        _, proba = self._gold_pred_and_proba(X, getattr(self, "diagnosis_threshold", 0.5))
        return proba

    def metrics(
        self, X: pd.DataFrame, y_diagnosis: np.ndarray, y_gold: np.ndarray
    ) -> dict[str, Any]:
        """Evaluate both ensembles on (X, y_diagnosis, y_gold)."""
        y_diag_pred = self.predict_diagnosis(X)
        y_diag_proba = self.predict_diagnosis_proba(X)
        y_gold_pred, y_gold_proba = self._gold_pred_and_proba(
            X, getattr(self, "diagnosis_threshold", 0.5)
        )
        diag_labels = (
            list(range(len(self.diagnosis_label_encoder.classes_)))
            if self.diagnosis_label_encoder is not None and hasattr(self.diagnosis_label_encoder, "classes_")
            else None
        )
        gold_labels = (
            list(range(len(self.gold_label_encoder.classes_)))
            if self.gold_label_encoder is not None and hasattr(self.gold_label_encoder, "classes_")
            else list(range(5))
        )

        out = {
            "diagnosis": _classification_metrics(y_diagnosis, y_diag_pred, y_diag_proba, labels=diag_labels),
            "gold_stage": _classification_metrics(y_gold, y_gold_pred, y_gold_proba, labels=gold_labels),
            "diagnosis_base": self.diagnosis_ensemble.base_model_metrics(X, y_diagnosis, labels=diag_labels),
            "gold_stage_base": self._gold_base_metrics(X, y_diagnosis, y_gold),
        }
        out["diagnosis_threshold"] = getattr(self, "diagnosis_threshold", 0.5)
        return out

    def _tune_diagnosis_threshold(
        self, X_val: pd.DataFrame, y_diagnosis_val: np.ndarray, y_gold_val: np.ndarray
    ) -> None:
        """Tune diagnosis_threshold on validation to improve downstream metrics."""
        best_t = getattr(self, "diagnosis_threshold", 0.5)
        best_score = -1.0

        for t in DIAGNOSIS_THRESHOLD_GRID:
            # Diagnosis predictions under threshold.
            y_diag_pred = self._predict_diagnosis_with_threshold(X_val, t)
            y_diag_proba = self.predict_diagnosis_proba(X_val)
            diag_labels = (
                list(range(len(self.diagnosis_label_encoder.classes_)))
                if self.diagnosis_label_encoder is not None and hasattr(self.diagnosis_label_encoder, "classes_")
                else None
            )
            gold_labels = (
                list(range(len(self.gold_label_encoder.classes_)))
                if self.gold_label_encoder is not None and hasattr(self.gold_label_encoder, "classes_")
                else list(range(5))
            )

            diag_metrics = _classification_metrics(
                y_diagnosis_val, y_diag_pred, y_diag_proba, labels=diag_labels
            )

            # GOLD predictions/proba under threshold.
            y_gold_pred, y_gold_proba = self._gold_pred_and_proba(X_val, t)
            gold_metrics = _classification_metrics(y_gold_val, y_gold_pred, y_gold_proba, labels=gold_labels)

            if TUNE_DIAGNOSIS_THRESHOLD_FOR == "diagnosis_f1_macro":
                score = float(diag_metrics.get("f1_macro", 0.0))
            elif TUNE_DIAGNOSIS_THRESHOLD_FOR == "gold_stage_accuracy":
                score = float(gold_metrics.get("accuracy", 0.0))
            elif TUNE_DIAGNOSIS_THRESHOLD_FOR == "gold_stage_f1_weighted":
                score = float(gold_metrics.get("f1_weighted", 0.0))
            else:
                # Default: optimize overall GOLD-stage macro F1.
                score = float(gold_metrics.get("f1_macro", 0.0))

            if score > best_score:
                best_score = score
                best_t = t

        self.diagnosis_threshold = float(best_t)

    def _gold_base_metrics(
        self, X: pd.DataFrame, y_diagnosis: np.ndarray, y_gold: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """Evaluate GOLD base models on true COPD rows only (4-class, GOLD_1..GOLD_4)."""
        X_gold = X.copy()
        if GOLD_DROP_RAW_SPIROMETRY:
            X_gold = X_gold.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
        X_gold[self.diagnosis_proba_column] = self._diagnosis_proba(X)
        mask = (np.asarray(y_diagnosis) == getattr(self, "copd_label_value", 1)) & (np.asarray(y_gold) > 0)
        if not np.any(mask):
            return {}
        y_sub = np.asarray(y_gold)[mask] - 1
        return self.gold_ensemble.base_model_metrics(
            X_gold.loc[X_gold.index[mask]], y_sub, labels=list(range(4))
        )

    def save(self, path: str) -> None:
        """Persist the entire double-target system."""
        _ensure_dir(path)
        self.diagnosis_ensemble.save(os.path.join(path, "diagnosis_ensemble"))
        self.gold_ensemble.save(os.path.join(path, "gold_ensemble"))
        bundle = {
            "diagnosis_proba_column": self.diagnosis_proba_column,
            "diagnosis_threshold": getattr(self, "diagnosis_threshold", 0.5),
            "is_fitted": self.is_fitted,
            "random_state": self.diagnosis_ensemble.random_state,
        }
        if self.diagnosis_label_encoder is not None:
            bundle["diagnosis_label_encoder"] = {
                "classes": self.diagnosis_label_encoder.classes_.tolist(),
                "encoded_labels": list(range(len(self.diagnosis_label_encoder.classes_))),
            }
        if self.gold_label_encoder is not None:
            bundle["gold_label_encoder"] = {
                "classes": self.gold_label_encoder.classes_.tolist(),
                "encoded_labels": list(range(len(self.gold_label_encoder.classes_))),
            }
        with open(os.path.join(path, "system.json"), "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "COPDDoubleTargetSystem":
        """Load a previously saved double-target system."""
        instance = cls()
        instance.diagnosis_ensemble = COPDEnsembleClassifier.load(os.path.join(path, "diagnosis_ensemble"))
        instance.gold_ensemble = COPDEnsembleClassifier.load(os.path.join(path, "gold_ensemble"))
        with open(os.path.join(path, "system.json"), "r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        instance.diagnosis_proba_column = bundle["diagnosis_proba_column"]
        instance.diagnosis_threshold = float(bundle.get("diagnosis_threshold", 0.5))
        instance.is_fitted = bundle["is_fitted"]
        if "diagnosis_label_encoder" in bundle:
            le = LabelEncoder()
            le.classes_ = np.array(bundle["diagnosis_label_encoder"]["classes"])
            instance.diagnosis_label_encoder = le

            # Recompute COPD class index + encoded label values after reload.
            classes = list(le.classes_)
            if "copd" in classes:
                instance.copd_class_index = int(classes.index("copd"))
                instance.copd_label_value = int(le.transform(["copd"])[0])
            if "no_copd" in classes:
                instance.no_copd_label_value = int(le.transform(["no_copd"])[0])
        if "gold_label_encoder" in bundle:
            le = LabelEncoder()
            le.classes_ = np.array(bundle["gold_label_encoder"]["classes"])
            instance.gold_label_encoder = le
        return instance


# ---------------------------------------------------------------------------
# DAG tasks
# ---------------------------------------------------------------------------

@dag(
    dag_id="copd_train_validate_test",
    description="Train, validate, test, and champion-select a COPD diagnosis + GOLD stage double-target ensemble.",
    default_args={
        "owner": "ml",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["copd", "ml", "training", "validation", "testing", "catboost", "ensemble", "classification", "nhanes", "gold"],
)
def copd_train_validate_test():
    @task
    def start() -> None:
        """Explicit start marker for readability in the task graph."""

    @task
    def load_data() -> dict[str, Any]:
        """Load the NHANES-derived preprocessed dataset, build two targets, and split."""
        context = get_current_context()
        ds = context["ds"]
        dag_run_id = str(context.get("run_id", "unknown"))
        paths = _partition_paths(ds)
        _ensure_dir(paths["artifact_dir"])

        df = pd.read_csv(paths["preprocessed_csv"])
        X, y_diagnosis, y_gold, diagnosis_encoder, gold_encoder = _build_targets(df)
        y_diagnosis_np = y_diagnosis.to_numpy()
        y_gold_np = y_gold.to_numpy()

        # Stratify on the binary diagnosis target so the COPD-positive rate is
        # preserved across splits. GOLD stage is too imbalanced for a combined
        # stratification (GOLD 4 has only 2 records in the full cohort).
        X_train_val, X_test, y_diag_train_val, y_diag_test, y_gold_train_val, y_gold_test = train_test_split(
            X, y_diagnosis_np, y_gold_np,
            test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_diagnosis_np
        )
        val_relative_size = VAL_SIZE / (1.0 - TEST_SIZE)
        X_train, X_val, y_diag_train, y_diag_val, y_gold_train, y_gold_val = train_test_split(
            X_train_val, y_diag_train_val, y_gold_train_val,
            test_size=val_relative_size, random_state=RANDOM_STATE, stratify=y_diag_train_val
        )

        splits_dir = os.path.join(paths["artifact_dir"], "splits")
        _ensure_dir(splits_dir)
        split_paths = {
            "X_train": os.path.join(splits_dir, "X_train.csv"),
            "X_val": os.path.join(splits_dir, "X_val.csv"),
            "X_test": os.path.join(splits_dir, "X_test.csv"),
            "y_diagnosis_train": os.path.join(splits_dir, "y_diagnosis_train.npy"),
            "y_diagnosis_val": os.path.join(splits_dir, "y_diagnosis_val.npy"),
            "y_diagnosis_test": os.path.join(splits_dir, "y_diagnosis_test.npy"),
            "y_gold_train": os.path.join(splits_dir, "y_gold_train.npy"),
            "y_gold_val": os.path.join(splits_dir, "y_gold_val.npy"),
            "y_gold_test": os.path.join(splits_dir, "y_gold_test.npy"),
            "label_encoders": os.path.join(splits_dir, "label_encoders.json"),
        }
        X_train.to_csv(split_paths["X_train"], index=False)
        X_val.to_csv(split_paths["X_val"], index=False)
        X_test.to_csv(split_paths["X_test"], index=False)
        np.save(split_paths["y_diagnosis_train"], y_diag_train)
        np.save(split_paths["y_diagnosis_val"], y_diag_val)
        np.save(split_paths["y_diagnosis_test"], y_diag_test)
        np.save(split_paths["y_gold_train"], y_gold_train)
        np.save(split_paths["y_gold_val"], y_gold_val)
        np.save(split_paths["y_gold_test"], y_gold_test)
        with open(split_paths["label_encoders"], "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "diagnosis": {
                        "classes": diagnosis_encoder.classes_.tolist(),
                        "encoded_labels": list(range(len(diagnosis_encoder.classes_))),
                    },
                    "gold_stage": {
                        "classes": gold_encoder.classes_.tolist(),
                        "encoded_labels": list(range(len(gold_encoder.classes_))),
                    },
                },
                fh,
                indent=2,
            )

        experiment_id, _ = _setup_mlflow(dag_run_id)
        with mlflow.start_run(run_name=f"load_data_{ds}") as run:
            mlflow.set_tag("dag_id", "copd_train_validate_test")
            mlflow.set_tag("task_id", "load_data")
            mlflow.set_tag("partition_ds", ds)
            mlflow.log_params(
                {
                    "diagnosis_target_column": DIAGNOSIS_TARGET_COLUMN,
                    "gold_target_column": GOLD_TARGET_COLUMN,
                    "test_size": TEST_SIZE,
                    "val_size": VAL_SIZE,
                    "random_state": RANDOM_STATE,
                    "n_features": X.shape[1],
                    "diagnosis_classes": DIAGNOSIS_CLASS_LABELS,
                    "gold_classes": GOLD_CLASS_LABELS,
                    "oof_folds": OOF_FOLDS,
                    "diagnosis_drop_raw_spirometry": DIAGNOSIS_DROP_RAW_SPIROMETRY,
                    "gold_drop_raw_spirometry": GOLD_DROP_RAW_SPIROMETRY,
                    "dropped_target_definition_columns": TARGET_DEFINITION_COLUMNS,
                }
            )
            mlflow.log_metrics(
                {
                    "train_rows": X_train.shape[0],
                    "val_rows": X_val.shape[0],
                    "test_rows": X_test.shape[0],
                }
            )
            for idx, cls in enumerate(diagnosis_encoder.classes_):
                mlflow.log_metric(f"diagnosis_class_{cls}_count", int(np.sum(y_diagnosis_np == idx)))
            for idx, cls in enumerate(gold_encoder.classes_):
                mlflow.log_metric(f"gold_class_{cls}_count", int(np.sum(y_gold_np == idx)))
            run_id = run.info.run_id

        return {
            "partition_ds": ds,
            "dag_run_id": dag_run_id,
            "experiment_id": experiment_id,
            "load_data_run_id": run_id,
            "split_paths": split_paths,
            "n_features": X.shape[1],
            "n_rows": X.shape[0],
            "diagnosis_n_classes": len(diagnosis_encoder.classes_),
            "gold_n_classes": len(gold_encoder.classes_),
            "diagnosis_classes": diagnosis_encoder.classes_.tolist(),
            "gold_classes": gold_encoder.classes_.tolist(),
            "label_encoders_path": split_paths["label_encoders"],
        }

    @task
    def train_ensemble(data_info: dict[str, Any]) -> dict[str, Any]:
        """Train the double-target COPDDoubleTargetSystem and log it to MLflow."""
        ds = data_info["partition_ds"]
        dag_run_id = data_info["dag_run_id"]
        splits = data_info["split_paths"]
        paths = _partition_paths(ds)
        diagnosis_n_classes = data_info["diagnosis_n_classes"]
        gold_n_classes = data_info["gold_n_classes"]

        X_train = pd.read_csv(splits["X_train"])
        X_val = pd.read_csv(splits["X_val"])
        y_diag_train = np.load(splits["y_diagnosis_train"])
        y_diag_val = np.load(splits["y_diagnosis_val"])
        y_gold_train = np.load(splits["y_gold_train"])
        y_gold_val = np.load(splits["y_gold_val"])

        # Load the fitted label encoders so the system can decode predictions.
        with open(splits["label_encoders"], "r", encoding="utf-8") as fh:
            le_data = json.load(fh)
        diagnosis_encoder = LabelEncoder()
        diagnosis_encoder.classes_ = np.array(le_data["diagnosis"]["classes"])
        gold_encoder = LabelEncoder()
        gold_encoder.classes_ = np.array(le_data["gold_stage"]["classes"])

        experiment_id, _ = _setup_mlflow(dag_run_id)
        system = COPDDoubleTargetSystem()
        system.diagnosis_label_encoder = diagnosis_encoder
        system.gold_label_encoder = gold_encoder
        system.fit(
            X_train, y_diag_train, y_gold_train,
            X_val, y_diag_val, y_gold_val,
        )

        val_metrics = system.metrics(X_val, y_diag_val, y_gold_val)

        system.save(paths["system_dir"])

        with mlflow.start_run(run_name=f"copd_double_target_system_{ds}") as run:
            mlflow.set_tag("dag_id", "copd_train_validate_test")
            mlflow.set_tag("task_id", "train_ensemble")
            mlflow.set_tag("model_name", "copd_double_target_system")
            mlflow.set_tag("partition_ds", ds)
            mlflow.set_tag("problem_type", "classification")
            mlflow.log_params(
                {
                    "diagnosis_base_models": list(system.diagnosis_ensemble.base_model_configs.keys()),
                    "gold_base_models": list(system.gold_ensemble.base_model_configs.keys()),
                    "meta_model": type(system.diagnosis_ensemble.meta_model_config).__name__,
                    "random_state": system.diagnosis_ensemble.random_state,
                    "diagnosis_n_classes": diagnosis_n_classes,
                    "gold_n_classes": gold_n_classes,
                    "diagnosis_classes": data_info["diagnosis_classes"],
                    "gold_classes": data_info["gold_classes"],
                    "diagnosis_use_sample_weights": DIAGNOSIS_USE_SAMPLE_WEIGHTS,
                    "gold_use_sample_weights": GOLD_USE_SAMPLE_WEIGHTS,
                    "tune_diagnosis_threshold": TUNE_DIAGNOSIS_THRESHOLD,
                    "tune_diagnosis_threshold_for": TUNE_DIAGNOSIS_THRESHOLD_FOR,
                }
            )
            mlflow.log_metrics({f"val_diagnosis_{k}": v for k, v in val_metrics["diagnosis"].items()})
            mlflow.log_metrics({f"val_gold_stage_{k}": v for k, v in val_metrics["gold_stage"].items()})
            for name, metrics in val_metrics["diagnosis_base"].items():
                mlflow.log_metrics({f"val_diagnosis_base_{name}_{k}": v for k, v in metrics.items()})
            for name, metrics in val_metrics["gold_stage_base"].items():
                mlflow.log_metrics({f"val_gold_base_{name}_{k}": v for k, v in metrics.items()})
            mlflow.log_artifact(paths["system_dir"])

            mlflow.sklearn.log_model(
                system.diagnosis_ensemble.fitted_meta_model,
                artifact_path="diagnosis_meta_model",
                registered_model_name=None,
            )
            mlflow.sklearn.log_model(
                system.gold_ensemble.fitted_meta_model,
                artifact_path="gold_meta_model",
                registered_model_name=None,
            )

            run_id = run.info.run_id

        return {
            "model_name": "copd_double_target_system",
            "model_type": "COPDDoubleTargetSystem",
            "mlflow_run_id": run_id,
            "experiment_id": experiment_id,
            "artifact_uri": mlflow.get_run(run_id).info.artifact_uri,
            "local_model_path": paths["system_dir"],
            "diagnosis_local_model_path": paths["diagnosis_ensemble_dir"],
            "gold_local_model_path": paths["gold_ensemble_dir"],
            "val_metrics": val_metrics,
            "diagnosis_n_classes": diagnosis_n_classes,
            "gold_n_classes": gold_n_classes,
            "diagnosis_classes": data_info["diagnosis_classes"],
            "gold_classes": data_info["gold_classes"],
            "params": {
                "diagnosis_base_models": list(system.diagnosis_ensemble.base_model_configs.keys()),
                "gold_base_models": list(system.gold_ensemble.base_model_configs.keys()),
                "meta_model": type(system.diagnosis_ensemble.meta_model_config).__name__,
                "random_state": system.diagnosis_ensemble.random_state,
            },
        }

    @task
    def evaluate_ensemble(
        data_info: dict[str, Any],
        ensemble_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate the double-target system on the held-out test set."""
        ds = data_info["partition_ds"]
        dag_run_id = data_info["dag_run_id"]
        splits = data_info["split_paths"]
        paths = _partition_paths(ds)

        X_test = pd.read_csv(splits["X_test"])
        y_diag_test = np.load(splits["y_diagnosis_test"])
        y_gold_test = np.load(splits["y_gold_test"])

        system = COPDDoubleTargetSystem.load(ensemble_info["local_model_path"])
        test_metrics = system.metrics(X_test, y_diag_test, y_gold_test)

        # Log test metrics back into the system's MLflow run.
        mlflow.start_run(run_id=ensemble_info["mlflow_run_id"])
        mlflow.log_metrics({f"test_diagnosis_{k}": v for k, v in test_metrics["diagnosis"].items()})
        mlflow.log_metrics({f"test_gold_stage_{k}": v for k, v in test_metrics["gold_stage"].items()})
        for name, metrics in test_metrics["diagnosis_base"].items():
            mlflow.log_metrics({f"test_diagnosis_base_{name}_{k}": v for k, v in metrics.items()})
        for name, metrics in test_metrics["gold_stage_base"].items():
            mlflow.log_metrics({f"test_gold_base_{name}_{k}": v for k, v in metrics.items()})
        mlflow.end_run()

        comparison = {
            "partition_ds": ds,
            "dag_run_id": dag_run_id,
            "model_name": ensemble_info["model_name"],
            "mlflow_run_id": ensemble_info["mlflow_run_id"],
            "test_metrics": test_metrics,
            "diagnosis_n_classes": ensemble_info["diagnosis_n_classes"],
            "gold_n_classes": ensemble_info["gold_n_classes"],
            "diagnosis_classes": ensemble_info["diagnosis_classes"],
            "gold_classes": ensemble_info["gold_classes"],
        }
        _ensure_dir(paths["artifact_dir"])
        with open(paths["metrics_path"], "w", encoding="utf-8") as fh:
            json.dump(comparison, fh, indent=2)

        experiment_id, _ = _setup_mlflow(dag_run_id)
        with mlflow.start_run(run_name=f"evaluate_ensemble_{ds}") as run:
            mlflow.set_tag("dag_id", "copd_train_validate_test")
            mlflow.set_tag("task_id", "evaluate_ensemble")
            mlflow.set_tag("partition_ds", ds)
            mlflow.log_param("experiment_id", experiment_id)
            mlflow.log_artifact(paths["metrics_path"])
            evaluation_run_id = run.info.run_id

        return {
            "comparison": comparison,
            "evaluation_run_id": evaluation_run_id,
            "experiment_id": experiment_id,
        }

    def _build_champion_records(
        ensemble_info: dict[str, Any],
        eval_info: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Return two champion records: one for diagnosis and one for GOLD stage."""
        comparison = eval_info["comparison"]
        test_metrics = comparison["test_metrics"]
        base = {
            "model_name": ensemble_info["model_name"],
            "mlflow_run_id": ensemble_info["mlflow_run_id"],
            "experiment_id": ensemble_info["experiment_id"],
            "partition_ds": comparison["partition_ds"],
            "artifact_uri": ensemble_info["artifact_uri"],
            "local_model_path": ensemble_info["local_model_path"],
            "model_type": ensemble_info["model_type"],
            "params": ensemble_info["params"],
            "dag_run_id": comparison["dag_run_id"],
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        return {
            "diagnosis": {
                **base,
                "target": "copd_diagnosis",
                "metric_name": "diagnosis_f1_macro",
                "metric_value": test_metrics["diagnosis"]["f1_macro"],
                "sub_model_path": ensemble_info["diagnosis_local_model_path"],
                "n_classes": comparison["diagnosis_n_classes"],
                "classes": comparison["diagnosis_classes"],
            },
            "gold_stage": {
                **base,
                "target": "gold_stage",
                "metric_name": "gold_stage_f1_macro",
                "metric_value": test_metrics["gold_stage"]["f1_macro"],
                "sub_model_path": ensemble_info["gold_local_model_path"],
                "n_classes": comparison["gold_n_classes"],
                "classes": comparison["gold_classes"],
            },
        }

    def _register_champion_in_database(record: dict[str, Any]) -> None:
        """Placeholder for the future PostgreSQL champion-table insertion."""
        raise NotImplementedError(
            "PostgreSQL champion registration is intentionally not implemented. "
            "Use the champion JSON produced by select_champion and insert it into "
            "the champion_models table using your chosen PSQL connection."
        )

    @task
    def select_champion(
        ensemble_info: dict[str, Any],
        eval_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist two champion records: one for diagnosis and one for GOLD stage."""
        ds = eval_info["comparison"]["partition_ds"]
        dag_run_id = eval_info["comparison"]["dag_run_id"]
        paths = _partition_paths(ds)

        champion_records = _build_champion_records(ensemble_info, eval_info)

        _ensure_dir(paths["artifact_dir"])
        with open(paths["champion_diagnosis_path"], "w", encoding="utf-8") as fh:
            json.dump(champion_records["diagnosis"], fh, indent=2)
        with open(paths["champion_gold_path"], "w", encoding="utf-8") as fh:
            json.dump(champion_records["gold_stage"], fh, indent=2)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_id=champion_records["diagnosis"]["mlflow_run_id"]):
            mlflow.set_tag("champion", "true")
            mlflow.set_tag("partition_ds", ds)
            mlflow.set_tag("dag_run_id", dag_run_id)
            mlflow.log_metric("champion_diagnosis_f1_macro", champion_records["diagnosis"]["metric_value"])
            mlflow.log_metric("champion_gold_stage_f1_macro", champion_records["gold_stage"]["metric_value"])
            mlflow.log_artifact(paths["champion_diagnosis_path"])
            mlflow.log_artifact(paths["champion_gold_path"])

        dm = eval_info["comparison"]["test_metrics"]["diagnosis"]
        gm = eval_info["comparison"]["test_metrics"]["gold_stage"]
        print(
            f"[champion] {champion_records['diagnosis']['model_name']} "
            f"diagnosis(acc={dm.get('accuracy', 0.0):.3f},p={dm.get('precision_macro', 0.0):.3f},r={dm.get('recall_macro', 0.0):.3f},f1={dm.get('f1_macro', 0.0):.3f}) "
            f"gold(acc={gm.get('accuracy', 0.0):.3f},p={gm.get('precision_macro', 0.0):.3f},r={gm.get('recall_macro', 0.0):.3f},f1={gm.get('f1_macro', 0.0):.3f}) "
            f"run_id={champion_records['diagnosis']['mlflow_run_id']}"
        )
        return {
            "diagnosis": champion_records["diagnosis"],
            "gold_stage": champion_records["gold_stage"],
        }

    @task
    def complete() -> None:
        """Final marker task for downstream dependencies."""

    # -----------------------------------------------------------------------
    # Task wiring
    # -----------------------------------------------------------------------
    start_task = start()
    data_info = load_data()
    ensemble_info = train_ensemble(data_info)
    eval_info = evaluate_ensemble(data_info, ensemble_info)
    champion_records = select_champion(ensemble_info, eval_info)
    complete_task = complete()

    start_task >> data_info >> ensemble_info >> eval_info >> champion_records >> complete_task


copd_dag = copd_train_validate_test()
