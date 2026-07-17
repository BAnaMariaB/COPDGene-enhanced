"""Shared constants and helpers for standalone candidate-model scripts.

These are intentionally mirrored from airflow/dags/copd_train_validate_test.py
(target columns, class labels, split sizes, metrics function, MLflow tracking
URI/experiment) so results logged here land in the SAME MLflow experiment and
are directly comparable to the ensemble system already built there. This file
does not import that DAG module directly (it uses @dag/@task decorators and
requires a full Airflow environment) - it is a plain, standalone module so it
can run with just scikit-learn/mlflow/pandas installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent

PREPROCESSED_ROOT = Path(
    os.environ.get("COPD_PREPROCESSED_ROOT", str(REPO_ROOT / "data" / "preprocessed"))
)

# Target columns in the NHANES-derived preprocessed dataset. Kept identical to
# the DAG's env var names so overriding one overrides both pipelines.
DIAGNOSIS_TARGET_COLUMN = os.environ.get("COPD_DIAGNOSIS_TARGET_COLUMN", "copd_diagnosis")
GOLD_TARGET_COLUMN = os.environ.get("COPD_GOLD_TARGET_COLUMN", "gold_stage")

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "sqlite:///" + str(REPO_ROOT / "mlflow.db")
)
MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    os.environ.get("COPD_MLFLOW_EXPERIMENT_NAME", "copd_nhanes_double_target_classification"),
)

TEST_SIZE = float(os.environ.get("COPD_TEST_SIZE", "0.15"))
VAL_SIZE = float(os.environ.get("COPD_VAL_SIZE", "0.15"))
RANDOM_STATE = int(os.environ.get("COPD_RANDOM_STATE", "42"))

DIAGNOSIS_CLASS_LABELS = ["no_copd", "copd"]
GOLD_CLASS_LABELS = ["GOLD_0", "GOLD_1", "GOLD_2", "GOLD_3", "GOLD_4"]

# The "ready to use" preprocessed dataset still contains the raw spirometry
# values that both targets are directly derived from:
#   copd_diagnosis = fev1_fvc_ratio < 0.70
#   gold_stage     = banded from fev1_pct_predicted
# fev1_ml and fvc_ml together reconstruct fev1_fvc_ratio exactly
# (ratio = fev1_ml / fvc_ml), so keeping either the ratio or both raw values
# lets a model just recover the label instead of predicting it (this is what
# produced a perfect 1.0 test F1 on a first run - not a good model, a leak).
# Dropping all four also matches the actual clinical use case: a screening
# tool that already requires a full spirometry reading to make its
# prediction has no value, since the diagnosis can just be computed directly
# from the same reading with no model at all.
SPIROMETRY_LEAKAGE_COLUMNS = ["fev1_fvc_ratio", "fev1_pct_predicted", "fev1_ml", "fvc_ml"]


def partition_csv_path(ds: str) -> Path:
    return PREPROCESSED_ROOT / ds / "central_preprocessed_dataset.csv"


def load_preprocessed(ds: str) -> pd.DataFrame:
    path = partition_csv_path(ds)
    if not path.exists():
        raise FileNotFoundError(
            f"No preprocessed dataset at {path}. Run the copd_ingestion DAG for "
            f"partition {ds} first."
        )
    return pd.read_csv(path)


def build_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, LabelEncoder, LabelEncoder]:
    """Split the preprocessed dataset into features and the two encoded targets.

    Mirrors _build_targets() in copd_train_validate_test.py exactly, so the
    resulting labels line up with the ensemble system's labels.
    """
    missing = [col for col in (DIAGNOSIS_TARGET_COLUMN, GOLD_TARGET_COLUMN) if col not in df.columns]
    if missing:
        raise ValueError(f"Target columns not found in preprocessed data: {missing}")

    df = df.copy()

    y_diagnosis_labels = df[DIAGNOSIS_TARGET_COLUMN].apply(
        lambda v: DIAGNOSIS_CLASS_LABELS[int(v)] if int(v) < len(DIAGNOSIS_CLASS_LABELS) else str(v)
    )
    y_gold_labels = df[GOLD_TARGET_COLUMN].apply(
        lambda v: GOLD_CLASS_LABELS[int(v)] if int(v) < len(GOLD_CLASS_LABELS) else str(v)
    )

    X = df.drop(columns=[DIAGNOSIS_TARGET_COLUMN, GOLD_TARGET_COLUMN], errors="ignore")

    diagnosis_encoder = LabelEncoder()
    diagnosis_encoder.fit(y_diagnosis_labels)
    y_diagnosis = pd.Series(diagnosis_encoder.transform(y_diagnosis_labels), index=df.index)

    gold_encoder = LabelEncoder()
    gold_encoder.fit(y_gold_labels)
    y_gold = pd.Series(gold_encoder.transform(y_gold_labels), index=df.index)

    return X, y_diagnosis, y_gold, diagnosis_encoder, gold_encoder


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, float]:
    """Identical to _classification_metrics() in copd_train_validate_test.py."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) >= 2:
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["roc_auc_ovr"] = float("nan")
    return metrics


def build_gold_positive_labels(df: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.Series, LabelEncoder]:
    """Build a GOLD-stage label restricted to COPD-positive rows only.

    gold_stage == 0 only ever occurs for non-COPD rows in this dataset, so
    staging only makes sense conditional on a positive diagnosis (matching
    the clinical use case: you stage severity only after diagnosing COPD).

    Returns:
        gold_labels: pd.Series indexed the same as X. NA for non-COPD rows,
            a string label ("GOLD_1".."GOLD_4") for COPD-positive rows.
        gold_encoder: LabelEncoder fitted on ALL COPD-positive rows in the
            full dataset (not a single split), so it always knows every class
            that exists anywhere even if a given train/val/test split happens
            not to include one (GOLD_4 has only 2 records total).
    """
    gold_raw_int = df.loc[X.index, GOLD_TARGET_COLUMN].astype(int)
    is_positive = gold_raw_int > 0
    gold_labels = pd.Series(pd.NA, index=X.index, dtype=object)
    gold_labels[is_positive] = gold_raw_int[is_positive].map(lambda v: GOLD_CLASS_LABELS[v])

    gold_encoder = LabelEncoder()
    gold_encoder.fit(gold_labels[is_positive].to_numpy())
    return gold_labels, gold_encoder


def artifact_dir(ds: str, model_name: str) -> Path:
    root = Path(os.environ.get("COPD_ARTIFACT_ROOT", str(REPO_ROOT / "data" / "artifacts")))
    path = root / ds / "models" / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def drop_nan(metrics: dict) -> dict:
    """Filter out NaN metric values before logging to MLflow.

    Some splits do not contain every class (GOLD_4 has only 2 records in the
    whole cohort), so roc_auc_ovr can come back NaN - see
    classification_metrics. MLflow's SQLite backend does not handle logging
    NaN metric values reliably, so skip them instead of logging a placeholder.
    """
    return {k: v for k, v in metrics.items() if v == v}


THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)


def tune_threshold(model, X_val, y_val, positive_code: int, negative_code: int) -> tuple[float, float]:
    """Sweep thresholds on the positive-class probability, pick the one that
    maximizes val f1_macro instead of using the implicit 0.5 cutoff."""
    class_index = list(model.classes_).index(positive_code)
    proba_positive = model.predict_proba(X_val)[:, class_index]

    best_threshold, best_f1 = 0.5, -1.0
    for threshold in THRESHOLD_GRID:
        preds = np.where(proba_positive >= threshold, positive_code, negative_code)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(threshold)
    return best_threshold, best_f1


def predict_with_threshold(model, X_eval, positive_code: int, negative_code: int, threshold: float) -> np.ndarray:
    class_index = list(model.classes_).index(positive_code)
    proba_positive = model.predict_proba(X_eval)[:, class_index]
    return np.where(proba_positive >= threshold, positive_code, negative_code)
