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
  ├── plots/
  │   ├── ensemble_test_metrics.png
  │   ├── lightgbm_test_metrics.png
  │   └── best_of_both_test_metrics.png
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
import matplotlib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from airflow.sdk import dag, get_current_context, task
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Champion registry (PostgreSQL) for online serving.
CHAMPION_REGISTRY_DATABASE_URL = os.environ.get("CHAMPION_REGISTRY_DATABASE_URL", "").strip()
# This must match what the serving backend queries via CHAMPION_MODEL_NAME.
CHAMPION_REGISTRY_MODEL_NAME = os.environ.get("CHAMPION_MODEL_NAME", "copd_double_target_system").strip()

# Stable MLflow artifact paths for serving to download.
# Each candidate logs a model bundle under:
#   candidates/<candidate_name>/model_system/
# and preprocessing artifacts under:
#   candidates/<candidate_name>/preprocessing/
CANDIDATE_ARTIFACT_ROOT = "candidates"
MODEL_SYSTEM_ARTIFACT_DIR = "model_system"
PREPROCESSING_ARTIFACT_FILE = "preprocessing/preprocessing_artifacts.joblib"

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
METRIC_PLOT_ORDER = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc_ovr",
]

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

# Default base classifiers for the champion-era stacked system.
DEFAULT_BASE_MODELS: dict[str, Any] = {
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
    "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
}

LIGHTGBM_ONLY_BASE_MODEL = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=2,
    verbose=-1,
)

DEFAULT_META_MODEL = LogisticRegression(
    max_iter=2000, random_state=RANDOM_STATE
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _partition_paths(ds: str) -> dict[str, str]:
    """Return deterministic input/output paths for a given partition date."""
    preprocessed_dir = os.path.join(PREPROCESSED_ROOT, ds)
    artifact_dir = os.path.join(ARTIFACT_ROOT, ds)
    return {
        "preprocessed_dir": preprocessed_dir,
        "preprocessed_csv": os.path.join(preprocessed_dir, "central_preprocessed_dataset.csv"),
        "preprocessing_joblib": os.path.join(preprocessed_dir, "preprocessing_artifacts.joblib"),
        "preprocessing_summary": os.path.join(preprocessed_dir, "preprocessing_artifacts.json"),
        "preprocessing_manifest": os.path.join(preprocessed_dir, "preprocessing_manifest.json"),
        "artifact_dir": artifact_dir,
        "models_dir": os.path.join(artifact_dir, "models"),
        "system_dir": os.path.join(artifact_dir, "models", "copd_double_target_system"),
        "diagnosis_ensemble_dir": os.path.join(artifact_dir, "models", "copd_double_target_system", "diagnosis_ensemble"),
        "gold_ensemble_dir": os.path.join(artifact_dir, "models", "copd_double_target_system", "gold_ensemble"),
        "candidates_dir": os.path.join(artifact_dir, CANDIDATE_ARTIFACT_ROOT),
        "plots_dir": os.path.join(artifact_dir, "plots"),
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


def _probability_distribution_summary(values: np.ndarray) -> dict[str, Any]:
    """Summarize a probability vector to detect compressed meta outputs."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {"count": 0}

    bins = np.linspace(0.0, 1.0, 11)
    hist, edges = np.histogram(arr, bins=bins)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
        "histogram": [
            {
                "left": float(edges[idx]),
                "right": float(edges[idx + 1]),
                "count": int(hist[idx]),
            }
            for idx in range(len(hist))
        ],
    }


def _save_candidate_metrics_plot(
    candidate_name: str,
    test_metrics: dict[str, Any],
    output_path: str,
) -> None:
    """Render one matplotlib figure per candidate with both tasks side by side."""
    diagnosis_metrics = test_metrics.get("diagnosis", {})
    gold_metrics = test_metrics.get("gold_stage", {})
    metric_names = [
        metric
        for metric in METRIC_PLOT_ORDER
        if metric in diagnosis_metrics or metric in gold_metrics
    ]
    if not metric_names:
        return

    diagnosis_values = [float(diagnosis_metrics.get(metric, 0.0)) for metric in metric_names]
    gold_values = [float(gold_metrics.get(metric, 0.0)) for metric in metric_names]

    labels = [name.replace("_", "\n") for name in metric_names]
    positions = np.arange(len(metric_names))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    fig.patch.set_facecolor("#f4f7fb")

    plot_config = [
        (axes[0], "COPD Diagnosis", diagnosis_values, "#1d4ed8"),
        (axes[1], "GOLD Stage", gold_values, "#0f766e"),
    ]
    for axis, task_name, values, color in plot_config:
        bars = axis.bar(positions, values, color=color, alpha=0.9, width=0.65)
        axis.set_title(task_name, fontsize=12, fontweight="bold")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(value + 0.02, 1.03),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_ylabel("Score", fontsize=10)
    threshold = test_metrics.get("diagnosis_threshold")
    threshold_suffix = f" | diagnosis_threshold={threshold:.2f}" if isinstance(threshold, (int, float)) else ""
    fig.suptitle(
        f"{candidate_name} test-set metrics by task{threshold_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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

    def _prepare_base_model(self, name: str, model_config: Any, y_fit: np.ndarray) -> Any:
        return clone(model_config) if hasattr(model_config, "get_params") else model_config

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
                    fold_model = self._prepare_base_model(name, model, y_train_arr[tr_idx])
                    X_tr = X_train.iloc[tr_idx]
                    y_tr = y_train_arr[tr_idx]
                    if sw_train is not None:
                        try:
                            fold_model.fit(X_tr, y_tr, sample_weight=sw_train[tr_idx])
                        except TypeError:
                            fold_model.fit(X_tr, y_tr)
                    else:
                        fold_model.fit(X_tr, y_tr)
                    hold_proba = self._model_predict_proba(fold_model, X_train.iloc[hold_idx])
                    oof_proba[hold_idx, :] = self._align_proba_to_n_classes(
                        hold_proba, getattr(fold_model, "classes_", None), n_classes
                    )
                oof_blocks.append(oof_proba)

            fitted = self._prepare_base_model(name, model, y_train)
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

    @staticmethod
    def _model_predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities, deriving them from decision_function if needed."""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X))
            # Binary case: scores shape (n_samples,)
            if scores.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1.0 - p1, p1])
            # Multiclass: softmax over class scores.
            scores = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            denom = exp_scores.sum(axis=1, keepdims=True)
            denom[denom == 0.0] = 1.0
            return exp_scores / denom
        raise ValueError(f"Model {type(model).__name__} has neither predict_proba nor decision_function")

    @staticmethod
    def _align_proba_to_n_classes(
        proba: np.ndarray, model_classes: Any | None, n_classes: int
    ) -> np.ndarray:
        """Align model probabilities to [0..n_classes-1] columns."""
        if proba.shape[1] == n_classes:
            return proba
        aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
        if model_classes is None:
            width = min(proba.shape[1], n_classes)
            aligned[:, :width] = proba[:, :width]
            return aligned
        classes = [int(c) for c in np.asarray(model_classes).tolist()]
        for idx, cls in enumerate(classes):
            if 0 <= cls < n_classes and idx < proba.shape[1]:
                aligned[:, cls] = proba[:, idx]
        return aligned

    def _base_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return a matrix where each block is one base classifier's class probabilities."""
        if not self.fitted_base_models:
            raise ValueError("Base classifiers have not been fitted yet")
        X = self._ensure_columns(X)
        return np.hstack([self._model_predict_proba(model, X) for model in self.fitted_base_models.values()])

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
        out: dict[str, dict[str, float]] = {}
        for name, model in self.fitted_base_models.items():
            proba = self._model_predict_proba(model, X)
            out[name] = _classification_metrics(y, model.predict(X), proba, labels=labels)
        return out

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
            model_classes = getattr(self.gold_ensemble.fitted_meta_model, "classes_", None)
            sub_proba_aligned = COPDEnsembleClassifier._align_proba_to_n_classes(
                sub_proba, model_classes, 4
            )
            proba[mask, 1:5] = sub_proba_aligned
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
            "diagnosis_probability_distribution": _probability_distribution_summary(self._diagnosis_proba(X)),
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


class COPDBestOfBothSystem:
    """Cascade system: RandomForest for diagnosis, XGBoost for GOLD stage.

    Unlike `COPDDoubleTargetSystem`, the GOLD-stage model does NOT consume the
    diagnosis probability as a feature. The only coupling is gating: if the
    diagnosis prediction is "no COPD" then GOLD is forced to GOLD_0.
    """

    def __init__(
        self,
        diagnosis_model: Any | None = None,
        gold_model: Any | None = None,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.diagnosis_model_config = diagnosis_model or RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=2,
        )
        # GOLD stage always uses a 4-class model (GOLD_1..GOLD_4 encoded as 0..3).
        self.gold_model_config = gold_model or XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=2,
        )
        self.diagnosis_threshold: float = 0.5
        self.diagnosis_label_encoder: LabelEncoder | None = None
        self.gold_label_encoder: LabelEncoder | None = None

        self.diagnosis_model: Any | None = None
        self.gold_model: Any | None = None

        self.copd_label_value: int = 1
        self.no_copd_label_value: int = 0
        self.copd_class_index: int = 1
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_diagnosis_train: np.ndarray,
        y_gold_train: np.ndarray,
        X_val: pd.DataFrame,
        y_diagnosis_val: np.ndarray,
        y_gold_val: np.ndarray,
    ) -> "COPDBestOfBothSystem":
        X_train_diag = (
            X_train.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if DIAGNOSIS_DROP_RAW_SPIROMETRY
            else X_train
        )
        X_val_diag = (
            X_val.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if DIAGNOSIS_DROP_RAW_SPIROMETRY
            else X_val
        )

        if self.diagnosis_label_encoder is not None and hasattr(self.diagnosis_label_encoder, "classes_"):
            classes = list(self.diagnosis_label_encoder.classes_)
            if "copd" in classes:
                self.copd_label_value = int(self.diagnosis_label_encoder.transform(["copd"])[0])
            if "no_copd" in classes:
                self.no_copd_label_value = int(self.diagnosis_label_encoder.transform(["no_copd"])[0])

        sw_diag = None
        if DIAGNOSIS_USE_SAMPLE_WEIGHTS:
            sw_diag = compute_sample_weight(class_weight="balanced", y=y_diagnosis_train)

        self.diagnosis_model = (
            clone(self.diagnosis_model_config)
            if hasattr(self.diagnosis_model_config, "get_params")
            else self.diagnosis_model_config
        )
        if sw_diag is not None:
            try:
                self.diagnosis_model.fit(X_train_diag, y_diagnosis_train, sample_weight=sw_diag)
            except TypeError:
                self.diagnosis_model.fit(X_train_diag, y_diagnosis_train)
        else:
            self.diagnosis_model.fit(X_train_diag, y_diagnosis_train)

        # Determine which probability column corresponds to COPD.
        if hasattr(self.diagnosis_model, "classes_"):
            classes = list(self.diagnosis_model.classes_)
            if self.copd_label_value in classes:
                self.copd_class_index = int(classes.index(self.copd_label_value))

        # GOLD-stage model is trained only on true COPD rows; GOLD_0 is a sentinel.
        X_train_gold = (
            X_train.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if GOLD_DROP_RAW_SPIROMETRY
            else X_train
        )
        X_val_gold = (
            X_val.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if GOLD_DROP_RAW_SPIROMETRY
            else X_val
        )

        train_mask = (np.asarray(y_diagnosis_train) == self.copd_label_value) & (np.asarray(y_gold_train) > 0)
        val_mask = (np.asarray(y_diagnosis_val) == self.copd_label_value) & (np.asarray(y_gold_val) > 0)

        y_gold_train_sub = np.asarray(y_gold_train)[train_mask] - 1
        y_gold_val_sub = np.asarray(y_gold_val)[val_mask] - 1

        sw_gold = None
        if GOLD_USE_SAMPLE_WEIGHTS and y_gold_train_sub.size:
            sw_gold = compute_sample_weight(class_weight="balanced", y=y_gold_train_sub)

        self.gold_model = (
            clone(self.gold_model_config)
            if hasattr(self.gold_model_config, "get_params")
            else self.gold_model_config
        )
        if y_gold_train_sub.size:
            if sw_gold is not None:
                try:
                    self.gold_model.fit(
                        X_train_gold.loc[X_train_gold.index[train_mask]],
                        y_gold_train_sub,
                        sample_weight=sw_gold,
                    )
                except TypeError:
                    self.gold_model.fit(
                        X_train_gold.loc[X_train_gold.index[train_mask]],
                        y_gold_train_sub,
                    )
            else:
                self.gold_model.fit(
                    X_train_gold.loc[X_train_gold.index[train_mask]],
                    y_gold_train_sub,
                )

        if TUNE_DIAGNOSIS_THRESHOLD:
            self._tune_diagnosis_threshold(X_val, y_diagnosis_val, y_gold_val)

        self.is_fitted = True
        return self

    def _diagnosis_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_diag = (
            X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if DIAGNOSIS_DROP_RAW_SPIROMETRY
            else X
        )
        proba = self.diagnosis_model.predict_proba(X_diag)
        idx = self.copd_class_index
        if idx < 0 or idx >= proba.shape[1]:
            idx = 1 if proba.shape[1] > 1 else 0
        return proba[:, idx]

    def _predict_diagnosis_with_threshold(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        proba = self._diagnosis_proba(X)
        return np.where(proba >= threshold, self.copd_label_value, self.no_copd_label_value)

    def predict_diagnosis(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_diagnosis_with_threshold(X, getattr(self, "diagnosis_threshold", 0.5))

    def predict_diagnosis_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_diag = (
            X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if DIAGNOSIS_DROP_RAW_SPIROMETRY
            else X
        )
        return self.diagnosis_model.predict_proba(X_diag)

    def _gold_pred_and_proba(self, X: pd.DataFrame, diagnosis_threshold: float) -> tuple[np.ndarray, np.ndarray]:
        diag_pred = self._predict_diagnosis_with_threshold(X, diagnosis_threshold)
        X_gold = (
            X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
            if GOLD_DROP_RAW_SPIROMETRY
            else X
        )

        gold_pred = np.zeros(X.shape[0], dtype=int)
        proba = np.zeros((X.shape[0], 5), dtype=float)

        mask = np.asarray(diag_pred) == self.copd_label_value
        proba[~mask, 0] = 1.0
        if np.any(mask) and self.gold_model is not None:
            sub_proba = self.gold_model.predict_proba(X_gold.loc[X_gold.index[mask]])
            sub_pred = np.asarray(self.gold_model.predict(X_gold.loc[X_gold.index[mask]]), dtype=int)
            gold_pred[mask] = sub_pred + 1
            proba[mask, 1:5] = sub_proba
        return gold_pred, proba

    def predict_gold(self, X: pd.DataFrame) -> np.ndarray:
        gold_pred, _ = self._gold_pred_and_proba(X, getattr(self, "diagnosis_threshold", 0.5))
        return gold_pred

    def predict_gold_proba(self, X: pd.DataFrame) -> np.ndarray:
        _, proba = self._gold_pred_and_proba(X, getattr(self, "diagnosis_threshold", 0.5))
        return proba

    def metrics(self, X: pd.DataFrame, y_diagnosis: np.ndarray, y_gold: np.ndarray) -> dict[str, Any]:
        y_diag_pred = self.predict_diagnosis(X)
        y_diag_proba = self.predict_diagnosis_proba(X)
        y_gold_pred, y_gold_proba = self._gold_pred_and_proba(X, getattr(self, "diagnosis_threshold", 0.5))

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
            "diagnosis_base": {
                "random_forest": _classification_metrics(
                    y_diagnosis,
                    self.diagnosis_model.predict(
                        X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
                        if DIAGNOSIS_DROP_RAW_SPIROMETRY
                        else X
                    ),
                    y_diag_proba,
                    labels=diag_labels,
                )
            },
            "gold_stage_base": {},
        }
        if self.gold_model is not None:
            # Evaluate GOLD base model on true COPD rows only (4-class).
            mask = (np.asarray(y_diagnosis) == self.copd_label_value) & (np.asarray(y_gold) > 0)
            if np.any(mask):
                X_gold = (
                    X.drop(columns=RAW_SPIROMETRY_COLUMNS, errors="ignore")
                    if GOLD_DROP_RAW_SPIROMETRY
                    else X
                )
                y_sub = np.asarray(y_gold)[mask] - 1
                y_pred_sub = self.gold_model.predict(X_gold.loc[X_gold.index[mask]])
                y_proba_sub = self.gold_model.predict_proba(X_gold.loc[X_gold.index[mask]])
                out["gold_stage_base"] = {
                    "xgboost": _classification_metrics(y_sub, y_pred_sub, y_proba_sub, labels=list(range(4)))
                }

        out["diagnosis_threshold"] = getattr(self, "diagnosis_threshold", 0.5)
        return out

    def _tune_diagnosis_threshold(self, X_val: pd.DataFrame, y_diagnosis_val: np.ndarray, y_gold_val: np.ndarray) -> None:
        best_t = getattr(self, "diagnosis_threshold", 0.5)
        best_score = -1.0

        for t in DIAGNOSIS_THRESHOLD_GRID:
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
            y_gold_pred, y_gold_proba = self._gold_pred_and_proba(X_val, t)
            gold_metrics = _classification_metrics(y_gold_val, y_gold_pred, y_gold_proba, labels=gold_labels)

            if TUNE_DIAGNOSIS_THRESHOLD_FOR == "diagnosis_f1_macro":
                score = float(diag_metrics.get("f1_macro", 0.0))
            elif TUNE_DIAGNOSIS_THRESHOLD_FOR == "gold_stage_accuracy":
                score = float(gold_metrics.get("accuracy", 0.0))
            elif TUNE_DIAGNOSIS_THRESHOLD_FOR == "gold_stage_f1_weighted":
                score = float(gold_metrics.get("f1_weighted", 0.0))
            else:
                score = float(gold_metrics.get("f1_macro", 0.0))

            if score > best_score:
                best_score = score
                best_t = t

        self.diagnosis_threshold = float(best_t)

    def save(self, path: str) -> None:
        _ensure_dir(path)
        if self.diagnosis_model is None or self.gold_model is None:
            raise ValueError("System is not fitted")

        joblib.dump(self.diagnosis_model, os.path.join(path, "diagnosis_model.joblib"))
        joblib.dump(self.gold_model, os.path.join(path, "gold_model.joblib"))

        bundle = {
            "bundle_type": "best_of_both_v1",
            "diagnosis_threshold": getattr(self, "diagnosis_threshold", 0.5),
            "is_fitted": self.is_fitted,
            "diagnosis_drop_raw_spirometry": DIAGNOSIS_DROP_RAW_SPIROMETRY,
            "gold_drop_raw_spirometry": GOLD_DROP_RAW_SPIROMETRY,
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
    def load(cls, path: str) -> "COPDBestOfBothSystem":
        with open(os.path.join(path, "system.json"), "r", encoding="utf-8") as fh:
            bundle = json.load(fh)

        instance = cls()
        instance.diagnosis_model = joblib.load(os.path.join(path, "diagnosis_model.joblib"))
        instance.gold_model = joblib.load(os.path.join(path, "gold_model.joblib"))
        instance.diagnosis_threshold = float(bundle.get("diagnosis_threshold", 0.5))
        instance.is_fitted = bool(bundle.get("is_fitted", True))

        if "diagnosis_label_encoder" in bundle:
            le = LabelEncoder()
            le.classes_ = np.array(bundle["diagnosis_label_encoder"]["classes"])
            instance.diagnosis_label_encoder = le
            classes = list(le.classes_)
            if "copd" in classes:
                instance.copd_label_value = int(le.transform(["copd"])[0])
            if "no_copd" in classes:
                instance.no_copd_label_value = int(le.transform(["no_copd"])[0])

        if "gold_label_encoder" in bundle:
            le = LabelEncoder()
            le.classes_ = np.array(bundle["gold_label_encoder"]["classes"])
            instance.gold_label_encoder = le

        if hasattr(instance.diagnosis_model, "classes_"):
            classes = list(instance.diagnosis_model.classes_)
            if instance.copd_label_value in classes:
                instance.copd_class_index = int(classes.index(instance.copd_label_value))

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
        ensemble_base_models = {
            "catboost": DEFAULT_BASE_MODELS["catboost"],
            "xgboost": DEFAULT_BASE_MODELS["xgboost"],
            "logistic_regression": DEFAULT_BASE_MODELS["logistic_regression"],
        }
        system = COPDDoubleTargetSystem(
            diagnosis_base_models=ensemble_base_models,
            gold_base_models=ensemble_base_models,
        )
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
                    "oof_folds": OOF_FOLDS,
                }
            )
            mlflow.log_metrics({f"val_diagnosis_{k}": v for k, v in val_metrics["diagnosis"].items()})
            mlflow.log_metrics({f"val_gold_stage_{k}": v for k, v in val_metrics["gold_stage"].items()})
            for key in ("mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "max"):
                if key in val_metrics["diagnosis_probability_distribution"]:
                    mlflow.log_metric(
                        f"val_diagnosis_probability_{key}",
                        float(val_metrics["diagnosis_probability_distribution"][key]),
                    )
            for name, metrics in val_metrics["diagnosis_base"].items():
                mlflow.log_metrics({f"val_diagnosis_base_{name}_{k}": v for k, v in metrics.items()})
            for name, metrics in val_metrics["gold_stage_base"].items():
                mlflow.log_metrics({f"val_gold_base_{name}_{k}": v for k, v in metrics.items()})
            # Log the full system + preprocessing artifacts under stable, candidate-scoped
            # paths so the serving backend can download them by (run_id, artifact_path).
            candidate_prefix = f"{CANDIDATE_ARTIFACT_ROOT}/ensemble"
            mlflow.log_artifacts(
                paths["system_dir"],
                artifact_path=f"{candidate_prefix}/{MODEL_SYSTEM_ARTIFACT_DIR}",
            )
            if os.path.exists(paths["preprocessing_joblib"]):
                mlflow.log_artifact(
                    paths["preprocessing_joblib"],
                    artifact_path=f"{candidate_prefix}/preprocessing",
                )
            if os.path.exists(paths["preprocessing_summary"]):
                mlflow.log_artifact(
                    paths["preprocessing_summary"],
                    artifact_path=f"{candidate_prefix}/preprocessing",
                )
            if os.path.exists(paths["preprocessing_manifest"]):
                mlflow.log_artifact(
                    paths["preprocessing_manifest"],
                    artifact_path=f"{candidate_prefix}/preprocessing",
                )

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

        candidate_prefix = f"{CANDIDATE_ARTIFACT_ROOT}/ensemble"
        return {
            "candidate_name": "ensemble",
            "enabled": True,
            "model_name": "copd_double_target_system",
            "model_type": "COPDDoubleTargetSystem",
            "mlflow_run_id": run_id,
            "experiment_id": experiment_id,
            "artifact_uri": mlflow.get_run(run_id).info.artifact_uri,
            "local_model_path": paths["system_dir"],
            "local_preprocessing_path": paths["preprocessing_joblib"],
            "diagnosis_local_model_path": paths["diagnosis_ensemble_dir"],
            "gold_local_model_path": paths["gold_ensemble_dir"],
            "val_metrics": val_metrics,
            "diagnosis_n_classes": diagnosis_n_classes,
            "gold_n_classes": gold_n_classes,
            "diagnosis_classes": data_info["diagnosis_classes"],
            "gold_classes": data_info["gold_classes"],
            "model_artifact_path": f"{candidate_prefix}/{MODEL_SYSTEM_ARTIFACT_DIR}",
            "preprocessing_artifact_path": f"{candidate_prefix}/{PREPROCESSING_ARTIFACT_FILE}",
            "params": {
                "diagnosis_base_models": list(system.diagnosis_ensemble.base_model_configs.keys()),
                "gold_base_models": list(system.gold_ensemble.base_model_configs.keys()),
                "meta_model": type(system.diagnosis_ensemble.meta_model_config).__name__,
                "random_state": system.diagnosis_ensemble.random_state,
                "oof_folds": OOF_FOLDS,
            },
        }

    @task
    def train_lightgbm(data_info: dict[str, Any]) -> dict[str, Any]:
        """Train a LightGBM-only stacked system (variant 2)."""
        ds = data_info["partition_ds"]
        dag_run_id = data_info["dag_run_id"]
        splits = data_info["split_paths"]
        paths = _partition_paths(ds)

        X_train = pd.read_csv(splits["X_train"])
        X_val = pd.read_csv(splits["X_val"])
        y_diag_train = np.load(splits["y_diagnosis_train"])
        y_diag_val = np.load(splits["y_diagnosis_val"])
        y_gold_train = np.load(splits["y_gold_train"])
        y_gold_val = np.load(splits["y_gold_val"])

        with open(splits["label_encoders"], "r", encoding="utf-8") as fh:
            le_data = json.load(fh)
        diagnosis_encoder = LabelEncoder()
        diagnosis_encoder.classes_ = np.array(le_data["diagnosis"]["classes"])
        gold_encoder = LabelEncoder()
        gold_encoder.classes_ = np.array(le_data["gold_stage"]["classes"])

        experiment_id, _ = _setup_mlflow(dag_run_id)

        lgbm_only = {"lightgbm": LIGHTGBM_ONLY_BASE_MODEL}
        system = COPDDoubleTargetSystem(diagnosis_base_models=lgbm_only, gold_base_models=lgbm_only)
        system.diagnosis_label_encoder = diagnosis_encoder
        system.gold_label_encoder = gold_encoder
        system.fit(X_train, y_diag_train, y_gold_train, X_val, y_diag_val, y_gold_val)
        val_metrics = system.metrics(X_val, y_diag_val, y_gold_val)

        local_model_dir = os.path.join(paths["candidates_dir"], "lightgbm", MODEL_SYSTEM_ARTIFACT_DIR)
        system.save(local_model_dir)

        with mlflow.start_run(run_name=f"copd_lightgbm_system_{ds}") as run:
            mlflow.set_tag("dag_id", "copd_train_validate_test")
            mlflow.set_tag("task_id", "train_lightgbm")
            mlflow.set_tag("model_name", "copd_lightgbm_system")
            mlflow.set_tag("partition_ds", ds)
            mlflow.set_tag("problem_type", "classification")
            mlflow.log_params(
                {
                    "candidate": "lightgbm",
                    "diagnosis_base_models": list(system.diagnosis_ensemble.base_model_configs.keys()),
                    "gold_base_models": list(system.gold_ensemble.base_model_configs.keys()),
                    "meta_model": type(system.diagnosis_ensemble.meta_model_config).__name__,
                    "random_state": system.diagnosis_ensemble.random_state,
                    "diagnosis_use_sample_weights": DIAGNOSIS_USE_SAMPLE_WEIGHTS,
                    "gold_use_sample_weights": GOLD_USE_SAMPLE_WEIGHTS,
                    "tune_diagnosis_threshold": TUNE_DIAGNOSIS_THRESHOLD,
                    "tune_diagnosis_threshold_for": TUNE_DIAGNOSIS_THRESHOLD_FOR,
                }
            )
            mlflow.log_metrics({f"val_diagnosis_{k}": v for k, v in val_metrics["diagnosis"].items()})
            mlflow.log_metrics({f"val_gold_stage_{k}": v for k, v in val_metrics["gold_stage"].items()})

            candidate_prefix = f"{CANDIDATE_ARTIFACT_ROOT}/lightgbm"
            mlflow.log_artifacts(
                local_model_dir,
                artifact_path=f"{candidate_prefix}/{MODEL_SYSTEM_ARTIFACT_DIR}",
            )
            if os.path.exists(paths["preprocessing_joblib"]):
                mlflow.log_artifact(
                    paths["preprocessing_joblib"],
                    artifact_path=f"{candidate_prefix}/preprocessing",
                )
            if os.path.exists(paths["preprocessing_summary"]):
                mlflow.log_artifact(
                    paths["preprocessing_summary"],
                    artifact_path=f"{candidate_prefix}/preprocessing",
                )
            if os.path.exists(paths["preprocessing_manifest"]):
                mlflow.log_artifact(
                    paths["preprocessing_manifest"],
                    artifact_path=f"{candidate_prefix}/preprocessing",
                )

            run_id = run.info.run_id

        candidate_prefix = f"{CANDIDATE_ARTIFACT_ROOT}/lightgbm"
        return {
            "candidate_name": "lightgbm",
            "enabled": True,
            "model_name": "copd_lightgbm_system",
            "model_type": "COPDDoubleTargetSystem",
            "mlflow_run_id": run_id,
            "experiment_id": experiment_id,
            "artifact_uri": mlflow.get_run(run_id).info.artifact_uri,
            "local_model_path": local_model_dir,
            "local_preprocessing_path": paths["preprocessing_joblib"],
            "val_metrics": val_metrics,
            "model_artifact_path": f"{candidate_prefix}/{MODEL_SYSTEM_ARTIFACT_DIR}",
            "preprocessing_artifact_path": f"{candidate_prefix}/{PREPROCESSING_ARTIFACT_FILE}",
            "params": {
                "diagnosis_base_models": list(system.diagnosis_ensemble.base_model_configs.keys()),
                "gold_base_models": list(system.gold_ensemble.base_model_configs.keys()),
                "meta_model": type(system.diagnosis_ensemble.meta_model_config).__name__,
                "random_state": system.diagnosis_ensemble.random_state,
            },
        }

    @task
    def train_best_of_both(data_info: dict[str, Any]) -> dict[str, Any]:
        """Train RandomForest (diagnosis) + XGBoost (GOLD) cascade system (variant 3)."""
        ds = data_info["partition_ds"]
        dag_run_id = data_info["dag_run_id"]
        splits = data_info["split_paths"]
        paths = _partition_paths(ds)

        X_train = pd.read_csv(splits["X_train"])
        X_val = pd.read_csv(splits["X_val"])
        y_diag_train = np.load(splits["y_diagnosis_train"])
        y_diag_val = np.load(splits["y_diagnosis_val"])
        y_gold_train = np.load(splits["y_gold_train"])
        y_gold_val = np.load(splits["y_gold_val"])

        with open(splits["label_encoders"], "r", encoding="utf-8") as fh:
            le_data = json.load(fh)
        diagnosis_encoder = LabelEncoder()
        diagnosis_encoder.classes_ = np.array(le_data["diagnosis"]["classes"])
        gold_encoder = LabelEncoder()
        gold_encoder.classes_ = np.array(le_data["gold_stage"]["classes"])

        experiment_id, _ = _setup_mlflow(dag_run_id)

        system = COPDBestOfBothSystem()
        system.diagnosis_label_encoder = diagnosis_encoder
        system.gold_label_encoder = gold_encoder
        system.fit(X_train, y_diag_train, y_gold_train, X_val, y_diag_val, y_gold_val)
        val_metrics = system.metrics(X_val, y_diag_val, y_gold_val)

        local_model_dir = os.path.join(paths["candidates_dir"], "best_of_both", MODEL_SYSTEM_ARTIFACT_DIR)
        system.save(local_model_dir)

        with mlflow.start_run(run_name=f"copd_best_of_both_system_{ds}") as run:
            mlflow.set_tag("dag_id", "copd_train_validate_test")
            mlflow.set_tag("task_id", "train_best_of_both")
            mlflow.set_tag("model_name", "copd_best_of_both_system")
            mlflow.set_tag("partition_ds", ds)
            mlflow.set_tag("problem_type", "classification")
            mlflow.log_params(
                {
                    "candidate": "best_of_both",
                    "diagnosis_model": type(system.diagnosis_model).__name__,
                    "gold_model": type(system.gold_model).__name__,
                    "random_state": RANDOM_STATE,
                    "diagnosis_use_sample_weights": DIAGNOSIS_USE_SAMPLE_WEIGHTS,
                    "gold_use_sample_weights": GOLD_USE_SAMPLE_WEIGHTS,
                    "tune_diagnosis_threshold": TUNE_DIAGNOSIS_THRESHOLD,
                    "tune_diagnosis_threshold_for": TUNE_DIAGNOSIS_THRESHOLD_FOR,
                }
            )
            mlflow.log_metrics({f"val_diagnosis_{k}": v for k, v in val_metrics["diagnosis"].items()})
            mlflow.log_metrics({f"val_gold_stage_{k}": v for k, v in val_metrics["gold_stage"].items()})

            candidate_prefix = f"{CANDIDATE_ARTIFACT_ROOT}/best_of_both"
            mlflow.log_artifacts(
                local_model_dir,
                artifact_path=f"{candidate_prefix}/{MODEL_SYSTEM_ARTIFACT_DIR}",
            )
            if os.path.exists(paths["preprocessing_joblib"]):
                mlflow.log_artifact(
                    paths["preprocessing_joblib"],
                    artifact_path=f"{candidate_prefix}/preprocessing",
                )
            if os.path.exists(paths["preprocessing_summary"]):
                mlflow.log_artifact(
                    paths["preprocessing_summary"],
                    artifact_path=f"{candidate_prefix}/preprocessing",
                )
            if os.path.exists(paths["preprocessing_manifest"]):
                mlflow.log_artifact(
                    paths["preprocessing_manifest"],
                    artifact_path=f"{candidate_prefix}/preprocessing",
                )

            mlflow.sklearn.log_model(system.diagnosis_model, artifact_path="best_of_both_diagnosis_model")
            mlflow.xgboost.log_model(system.gold_model, artifact_path="best_of_both_gold_model")

            run_id = run.info.run_id

        candidate_prefix = f"{CANDIDATE_ARTIFACT_ROOT}/best_of_both"
        return {
            "candidate_name": "best_of_both",
            "enabled": True,
            "model_name": "copd_best_of_both_system",
            "model_type": "COPDBestOfBothSystem",
            "mlflow_run_id": run_id,
            "experiment_id": experiment_id,
            "artifact_uri": mlflow.get_run(run_id).info.artifact_uri,
            "local_model_path": local_model_dir,
            "local_preprocessing_path": paths["preprocessing_joblib"],
            "val_metrics": val_metrics,
            "model_artifact_path": f"{candidate_prefix}/{MODEL_SYSTEM_ARTIFACT_DIR}",
            "preprocessing_artifact_path": f"{candidate_prefix}/{PREPROCESSING_ARTIFACT_FILE}",
            "params": {
                "diagnosis_model": type(system.diagnosis_model).__name__,
                "gold_model": type(system.gold_model).__name__,
                "random_state": RANDOM_STATE,
            },
        }

    @task
    def evaluate_candidates(
        data_info: dict[str, Any],
        ensemble_info: dict[str, Any],
        lightgbm_info: dict[str, Any],
        best_of_both_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate all candidates on the held-out test set and pick a champion."""
        ds = data_info["partition_ds"]
        dag_run_id = data_info["dag_run_id"]
        splits = data_info["split_paths"]
        paths = _partition_paths(ds)

        _setup_mlflow(dag_run_id)

        X_test = pd.read_csv(splits["X_test"])
        y_diag_test = np.load(splits["y_diagnosis_test"])
        y_gold_test = np.load(splits["y_gold_test"])

        candidates: dict[str, dict[str, Any]] = {}

        def _load_system(info: dict[str, Any]) -> Any:
            if info["model_type"] == "COPDDoubleTargetSystem":
                return COPDDoubleTargetSystem.load(info["local_model_path"])
            if info["model_type"] == "COPDBestOfBothSystem":
                return COPDBestOfBothSystem.load(info["local_model_path"])
            raise ValueError(f"Unknown model_type: {info['model_type']}")

        for info in (ensemble_info, lightgbm_info, best_of_both_info):
            name = info["candidate_name"]
            if not info.get("enabled", True):
                candidates[name] = {
                    "candidate_name": name,
                    "disabled": True,
                    "reason": info.get("reason", "disabled"),
                }
                continue

            system = _load_system(info)
            test_metrics = system.metrics(X_test, y_diag_test, y_gold_test)
            system_score = float(
                (test_metrics["diagnosis"]["f1_macro"] + test_metrics["gold_stage"]["f1_macro"]) / 2.0
            )
            metrics_plot_path = os.path.join(paths["plots_dir"], f"{name}_test_metrics.png")
            _save_candidate_metrics_plot(name, test_metrics, metrics_plot_path)

            # Log test metrics back into each candidate's MLflow run.
            mlflow.start_run(run_id=info["mlflow_run_id"])
            mlflow.log_metrics({f"test_diagnosis_{k}": v for k, v in test_metrics["diagnosis"].items()})
            mlflow.log_metrics({f"test_gold_stage_{k}": v for k, v in test_metrics["gold_stage"].items()})
            mlflow.log_metric("test_system_score_avg_f1_macro", system_score)
            for base_name, metrics in test_metrics.get("diagnosis_base", {}).items():
                mlflow.log_metrics({f"test_diagnosis_base_{base_name}_{k}": v for k, v in metrics.items()})
            for base_name, metrics in test_metrics.get("gold_stage_base", {}).items():
                mlflow.log_metrics({f"test_gold_base_{base_name}_{k}": v for k, v in metrics.items()})
            if os.path.exists(metrics_plot_path):
                mlflow.log_artifact(
                    metrics_plot_path,
                    artifact_path=f"{CANDIDATE_ARTIFACT_ROOT}/{name}/evaluation",
                )
            mlflow.end_run()

            candidates[name] = {
                "candidate_name": name,
                "model_name": info["model_name"],
                "model_type": info["model_type"],
                "mlflow_run_id": info["mlflow_run_id"],
                "model_artifact_path": info.get("model_artifact_path"),
                "preprocessing_artifact_path": info.get("preprocessing_artifact_path"),
                "system_score": system_score,
                "test_metrics": test_metrics,
                "metrics_plot_path": metrics_plot_path,
            }

        enabled_candidates = [c for c in candidates.values() if not c.get("disabled")]
        if not enabled_candidates:
            raise RuntimeError("No enabled candidates were evaluated")
        champion_candidate = max(enabled_candidates, key=lambda x: float(x.get("system_score", -1.0)))[
            "candidate_name"
        ]

        comparison = {
            "partition_ds": ds,
            "dag_run_id": dag_run_id,
            "champion_candidate": champion_candidate,
            "candidates": candidates,
        }

        _ensure_dir(paths["artifact_dir"])
        with open(paths["metrics_path"], "w", encoding="utf-8") as fh:
            json.dump(comparison, fh, indent=2)

        experiment_id, _ = _setup_mlflow(dag_run_id)
        with mlflow.start_run(run_name=f"evaluate_candidates_{ds}") as run:
            mlflow.set_tag("dag_id", "copd_train_validate_test")
            mlflow.set_tag("task_id", "evaluate_candidates")
            mlflow.set_tag("partition_ds", ds)
            mlflow.log_param("experiment_id", experiment_id)
            mlflow.log_param("champion_candidate", champion_candidate)
            mlflow.log_artifact(paths["metrics_path"])
            for candidate in candidates.values():
                if candidate.get("disabled"):
                    continue
                metrics_plot_path = candidate.get("metrics_plot_path")
                if metrics_plot_path and os.path.exists(metrics_plot_path):
                    mlflow.log_artifact(metrics_plot_path, artifact_path="candidate_metric_plots")
            evaluation_run_id = run.info.run_id

        return {
            "partition_ds": ds,
            "dag_run_id": dag_run_id,
            "champion_candidate": champion_candidate,
            "candidates": candidates,
            "evaluation_run_id": evaluation_run_id,
            "experiment_id": experiment_id,
        }

    def _build_champion_records(
        champion_info: dict[str, Any],
        eval_info: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Return two champion records (diagnosis + GOLD) for the chosen candidate."""
        ds = eval_info["partition_ds"]
        dag_run_id = eval_info["dag_run_id"]
        candidate_name = eval_info["champion_candidate"]
        candidate = eval_info["candidates"][candidate_name]
        test_metrics = candidate["test_metrics"]

        base = {
            "candidate_name": candidate_name,
            "model_name": CHAMPION_REGISTRY_MODEL_NAME,
            "candidate_model_name": champion_info["model_name"],
            "mlflow_run_id": champion_info["mlflow_run_id"],
            "experiment_id": champion_info["experiment_id"],
            "partition_ds": ds,
            "artifact_uri": champion_info.get("artifact_uri"),
            "local_model_path": champion_info["local_model_path"],
            "model_type": champion_info["model_type"],
            "params": champion_info.get("params"),
            "dag_run_id": dag_run_id,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "model_artifact_path": champion_info.get("model_artifact_path"),
            "preprocessing_artifact_path": champion_info.get("preprocessing_artifact_path"),
        }

        return {
            "diagnosis": {
                **base,
                "target": "copd_diagnosis",
                "metric_name": "diagnosis_f1_macro",
                "metric_value": test_metrics["diagnosis"]["f1_macro"],
                "sub_model_path": champion_info.get("diagnosis_local_model_path")
                or champion_info.get("local_model_path"),
            },
            "gold_stage": {
                **base,
                "target": "gold_stage",
                "metric_name": "gold_stage_f1_macro",
                "metric_value": test_metrics["gold_stage"]["f1_macro"],
                "sub_model_path": champion_info.get("gold_local_model_path")
                or champion_info.get("local_model_path"),
            },
        }

    def _registry_model_artifact_path(champion_info: dict[str, Any]) -> str | None:
        local_path = champion_info.get("local_model_path")
        if local_path and os.path.exists(local_path):
            return str(local_path)
        return champion_info.get("model_artifact_path")

    def _registry_preprocessing_artifact_path(champion_info: dict[str, Any]) -> str | None:
        local_path = champion_info.get("local_preprocessing_path")
        if local_path and os.path.exists(local_path):
            return str(local_path)
        return champion_info.get("preprocessing_artifact_path")

    def _register_champion_in_database(record: dict[str, Any]) -> int | None:
        """Insert the champion record into the Postgres registry table (if configured)."""
        if not CHAMPION_REGISTRY_DATABASE_URL:
            print("[registry] CHAMPION_REGISTRY_DATABASE_URL not set; skipping champion registry insert")
            return None

        try:
            import psycopg2
            from psycopg2.extras import Json
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "psycopg2 is required to write champion registry rows; install psycopg2-binary"
            ) from e

        def _normalize(url: str) -> str:
            if url.startswith("postgresql+psycopg2://"):
                return "postgresql://" + url[len("postgresql+psycopg2://") :]
            return url

        ddl = """
        CREATE TABLE IF NOT EXISTS champion_models (
          id BIGSERIAL PRIMARY KEY,
          model_name TEXT NOT NULL,
          target TEXT NOT NULL,
          mlflow_tracking_uri TEXT NOT NULL,
          mlflow_experiment_name TEXT NOT NULL,
          mlflow_run_id TEXT NOT NULL,
          artifact_uri TEXT,
          model_artifact_path TEXT,
          preprocessing_artifact_path TEXT,
          metric_name TEXT,
          metric_value DOUBLE PRECISION,
          params_json JSONB,
          registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          is_active BOOLEAN NOT NULL DEFAULT TRUE
        );
        CREATE INDEX IF NOT EXISTS champion_models_active_idx
          ON champion_models(model_name, target)
          WHERE is_active;
        """

        url = _normalize(CHAMPION_REGISTRY_DATABASE_URL)
        conn = psycopg2.connect(url)
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(ddl)

            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE champion_models SET is_active=FALSE WHERE model_name=%s AND target=%s AND is_active=TRUE",
                    (record["model_name"], record["target"]),
                )
                cur.execute(
                    """
                    INSERT INTO champion_models (
                      model_name, target,
                      mlflow_tracking_uri, mlflow_experiment_name,
                      mlflow_run_id, artifact_uri,
                      model_artifact_path, preprocessing_artifact_path,
                      metric_name, metric_value,
                      params_json,
                      is_active
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE)
                    RETURNING id
                    """,
                    (
                        record["model_name"],
                        record["target"],
                        MLFLOW_TRACKING_URI,
                        MLFLOW_EXPERIMENT_NAME,
                        record["mlflow_run_id"],
                        record.get("artifact_uri"),
                        record.get("model_artifact_path"),
                        record.get("preprocessing_artifact_path"),
                        record.get("metric_name"),
                        float(record.get("metric_value")) if record.get("metric_value") is not None else None,
                        Json(record.get("params_json") or {}),
                    ),
                )
                new_id = int(cur.fetchone()[0])
            conn.commit()
            print(
                f"[registry] upserted champion model_name={record['model_name']} target={record['target']} id={new_id}"
            )
            return new_id
        finally:
            conn.close()

    @task
    def select_champion(
        ensemble_info: dict[str, Any],
        lightgbm_info: dict[str, Any],
        best_of_both_info: dict[str, Any],
        eval_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist champion records and update the serving registry for the chosen candidate."""
        ds = eval_info["partition_ds"]
        dag_run_id = eval_info["dag_run_id"]
        champion_candidate = eval_info["champion_candidate"]
        paths = _partition_paths(ds)

        candidate_infos = {
            "ensemble": ensemble_info,
            "lightgbm": lightgbm_info,
            "best_of_both": best_of_both_info,
        }
        champion_info = candidate_infos[champion_candidate]
        champion_test_metrics = eval_info["candidates"][champion_candidate]["test_metrics"]

        champion_records = _build_champion_records(champion_info, eval_info)

        _ensure_dir(paths["artifact_dir"])
        with open(paths["champion_diagnosis_path"], "w", encoding="utf-8") as fh:
            json.dump(champion_records["diagnosis"], fh, indent=2)
        with open(paths["champion_gold_path"], "w", encoding="utf-8") as fh:
            json.dump(champion_records["gold_stage"], fh, indent=2)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_id=champion_info["mlflow_run_id"]):
            mlflow.set_tag("champion", "true")
            mlflow.set_tag("champion_candidate", champion_candidate)
            mlflow.set_tag("partition_ds", ds)
            mlflow.set_tag("dag_run_id", dag_run_id)
            mlflow.log_metric("champion_diagnosis_f1_macro", champion_records["diagnosis"]["metric_value"])
            mlflow.log_metric("champion_gold_stage_f1_macro", champion_records["gold_stage"]["metric_value"])
            mlflow.log_metric(
                "champion_system_score_avg_f1_macro",
                float(eval_info["candidates"][champion_candidate]["system_score"]),
            )
            mlflow.log_artifact(paths["champion_diagnosis_path"])
            mlflow.log_artifact(paths["champion_gold_path"])

        # Champion registry insert (system + per-target records).
        system_score = float(eval_info["candidates"][champion_candidate]["system_score"])

        system_registry_record = {
            "model_name": CHAMPION_REGISTRY_MODEL_NAME,
            "target": "system",
            "mlflow_run_id": champion_info["mlflow_run_id"],
            "artifact_uri": champion_info.get("artifact_uri"),
            "model_artifact_path": _registry_model_artifact_path(champion_info),
            "preprocessing_artifact_path": _registry_preprocessing_artifact_path(champion_info),
            "metric_name": "system_score_avg_f1_macro",
            "metric_value": system_score,
            "params_json": {
                **(champion_info.get("params") or {}),
                "candidate": champion_candidate,
                "diagnosis_threshold": champion_test_metrics.get("diagnosis_threshold"),
                "metrics": {
                    "diagnosis": champion_test_metrics.get("diagnosis"),
                    "gold_stage": champion_test_metrics.get("gold_stage"),
                },
            },
        }

        diagnosis_registry_record = {
            "model_name": CHAMPION_REGISTRY_MODEL_NAME,
            "target": "copd_diagnosis",
            "mlflow_run_id": champion_info["mlflow_run_id"],
            "artifact_uri": champion_info.get("artifact_uri"),
            "model_artifact_path": _registry_model_artifact_path(champion_info),
            "preprocessing_artifact_path": _registry_preprocessing_artifact_path(champion_info),
            "metric_name": champion_records["diagnosis"]["metric_name"],
            "metric_value": champion_records["diagnosis"]["metric_value"],
            "params_json": {
                **(champion_info.get("params") or {}),
                "candidate": champion_candidate,
                "diagnosis_threshold": champion_test_metrics.get("diagnosis_threshold"),
            },
        }

        gold_registry_record = {
            "model_name": CHAMPION_REGISTRY_MODEL_NAME,
            "target": "gold_stage",
            "mlflow_run_id": champion_info["mlflow_run_id"],
            "artifact_uri": champion_info.get("artifact_uri"),
            "model_artifact_path": _registry_model_artifact_path(champion_info),
            "preprocessing_artifact_path": _registry_preprocessing_artifact_path(champion_info),
            "metric_name": champion_records["gold_stage"]["metric_name"],
            "metric_value": champion_records["gold_stage"]["metric_value"],
            "params_json": {
                **(champion_info.get("params") or {}),
                "candidate": champion_candidate,
                "diagnosis_threshold": champion_test_metrics.get("diagnosis_threshold"),
            },
        }

        _register_champion_in_database(system_registry_record)
        _register_champion_in_database(diagnosis_registry_record)
        _register_champion_in_database(gold_registry_record)

        dm = champion_test_metrics["diagnosis"]
        gm = champion_test_metrics["gold_stage"]
        print(
            f"[champion] candidate={champion_candidate} model={champion_info['model_name']} "
            f"diagnosis(acc={dm.get('accuracy', 0.0):.3f},p={dm.get('precision_macro', 0.0):.3f},r={dm.get('recall_macro', 0.0):.3f},f1={dm.get('f1_macro', 0.0):.3f}) "
            f"gold(acc={gm.get('accuracy', 0.0):.3f},p={gm.get('precision_macro', 0.0):.3f},r={gm.get('recall_macro', 0.0):.3f},f1={gm.get('f1_macro', 0.0):.3f}) "
            f"system_score={system_score:.3f} run_id={champion_info['mlflow_run_id']}"
        )
        return {
            "champion_candidate": champion_candidate,
            "diagnosis": champion_records["diagnosis"],
            "gold_stage": champion_records["gold_stage"],
        }

    @task
    def complete() -> None:
        """Final marker task for downstream dependencies."""
        try:
            import sys

            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if workspace_root not in sys.path:
                sys.path.insert(0, workspace_root)

            from airflow.sdk import get_current_context
            from serving.common.champion_registry import connect as db_connect
            from serving.common.champion_registry import upsert_pipeline_event

            if CHAMPION_REGISTRY_DATABASE_URL:
                context = get_current_context()
                with db_connect() as conn:
                    upsert_pipeline_event(
                        conn,
                        pipeline_name="copd_train_validate_test",
                        event_type="completion",
                        status="success",
                        model_name=CHAMPION_REGISTRY_MODEL_NAME,
                        target="system",
                        run_id=str(context.get("run_id")),
                        logical_date=str(context.get("ds")),
                        details_json={},
                    )
        except Exception as exc:  # noqa: BLE001
            print(f"[training_registry] skipped: {exc}")

    # -----------------------------------------------------------------------
    # Task wiring
    # -----------------------------------------------------------------------
    start_task = start()
    data_info = load_data()

    ensemble_info = train_ensemble(data_info)
    lightgbm_info = train_lightgbm(data_info)
    best_of_both_info = train_best_of_both(data_info)

    eval_info = evaluate_candidates(data_info, ensemble_info, lightgbm_info, best_of_both_info)
    champion_records = select_champion(ensemble_info, lightgbm_info, best_of_both_info, eval_info)
    complete_task = complete()

    start_task >> data_info
    data_info >> [ensemble_info, lightgbm_info, best_of_both_info]
    [ensemble_info, lightgbm_info, best_of_both_info] >> eval_info
    eval_info >> champion_records >> complete_task


copd_dag = copd_train_validate_test()
