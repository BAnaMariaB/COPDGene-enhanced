"""Combine the tuned Random Forest diagnosis model and the tuned XGBoost
GOLD-stage model into one two-stage prediction pipeline: diagnose first,
then stage severity only for predicted-positive cases.

This is NOT a stacking/blending ensemble like the COPDEnsembleClassifier in
airflow/dags/copd_train_validate_test.py - it is a routing pipeline that uses
each already-proven-better model for its own stage, motivated by the
head-to-head result from the two standalone scripts:
  - Random Forest wins diagnosis   (test F1 macro 0.684 vs XGBoost's 0.656)
  - XGBoost wins GOLD stage        (test F1 macro 0.408 vs Random Forest's 0.326)

Run with:
    python modeling/train_combined.py
    python modeling/train_combined.py --ds 2026-07-16
"""

from __future__ import annotations

import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    DIAGNOSIS_TARGET_COLUMN,
    GOLD_TARGET_COLUMN,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    SPIROMETRY_LEAKAGE_COLUMNS,
    TEST_SIZE,
    VAL_SIZE,
    artifact_dir,
    build_gold_positive_labels,
    build_targets,
    classification_metrics,
    drop_nan,
    load_preprocessed,
    predict_with_threshold,
    tune_threshold,
)
from train_random_forest import tune_random_forest
from train_xgboost import tune_xgboost


class COPDBestOfBothSystem:
    """Two-stage COPD prediction: Random Forest diagnosis, then XGBoost GOLD
    staging, only for rows the diagnosis stage predicts positive.

    Not an ensemble in the stacking/blending sense - each stage is a single
    already-trained model doing the part it is better at, chained together.
    """

    def __init__(
        self,
        diagnosis_model,
        gold_model,
        positive_code: int,
        negative_code: int,
        diagnosis_threshold: float,
    ) -> None:
        self.diagnosis_model = diagnosis_model
        self.gold_model = gold_model
        self.positive_code = positive_code
        self.negative_code = negative_code
        self.diagnosis_threshold = diagnosis_threshold

    def predict_diagnosis(self, X: pd.DataFrame) -> np.ndarray:
        return predict_with_threshold(
            self.diagnosis_model, X, self.positive_code, self.negative_code, self.diagnosis_threshold
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Full two-stage prediction. gold_stage is NaN for rows predicted
        COPD-negative, since staging is only meaningful for positive cases."""
        diagnosis_pred = self.predict_diagnosis(X)
        gold_pred = np.full(len(X), np.nan, dtype=object)
        positive_mask = diagnosis_pred == self.positive_code
        if positive_mask.any():
            gold_pred[positive_mask] = self.gold_model.predict(X[positive_mask])
        return pd.DataFrame({"diagnosis": diagnosis_pred, "gold_stage": gold_pred}, index=X.index)


def train_and_evaluate(ds: str) -> dict:
    df = load_preprocessed(ds)
    X, y_diagnosis, _old_gold, diagnosis_encoder, _old_gold_encoder = build_targets(df)

    dropped_leakage_cols = [c for c in SPIROMETRY_LEAKAGE_COLUMNS if c in X.columns]
    X = X.drop(columns=dropped_leakage_cols)

    gold_labels_raw, gold_encoder = build_gold_positive_labels(df, X)

    X_train_val, X_test, y_diag_train_val, y_diag_test = train_test_split(
        X, y_diagnosis, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_diagnosis
    )
    val_relative_size = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_diag_train, y_diag_val = train_test_split(
        X_train_val, y_diag_train_val, test_size=val_relative_size, random_state=RANDOM_STATE, stratify=y_diag_train_val,
    )

    def gold_positive_subset(X_split):
        labels = gold_labels_raw.loc[X_split.index].dropna()
        X_sub = X_split.loc[labels.index]
        y_sub = gold_encoder.transform(labels.to_numpy())
        return X_sub, y_sub

    X_gold_train, y_gold_train = gold_positive_subset(X_train)
    X_gold_val, y_gold_val = gold_positive_subset(X_val)
    X_gold_test, y_gold_test = gold_positive_subset(X_test)

    # Stage 1: Random Forest diagnosis (identical setup to train_random_forest.py).
    diagnosis_model, diagnosis_tuning = tune_random_forest(X_train, y_diag_train)
    positive_code = diagnosis_encoder.transform(["copd"])[0]
    negative_code = diagnosis_encoder.transform(["no_copd"])[0]
    diag_threshold, _ = tune_threshold(diagnosis_model, X_val, y_diag_val, positive_code, negative_code)

    # Stage 2: XGBoost GOLD stage, trained on truly-positive rows only
    # (identical setup to train_xgboost.py).
    gold_model, gold_tuning = tune_xgboost(X_gold_train, y_gold_train)

    system = COPDBestOfBothSystem(diagnosis_model, gold_model, positive_code, negative_code, diag_threshold)

    # --- Stage-level metrics, same numbers as the two standalone scripts ---
    y_val_pred_diag = system.predict_diagnosis(X_val)
    y_test_pred_diag = system.predict_diagnosis(X_test)
    val_diag_metrics = classification_metrics(y_diag_val, y_val_pred_diag, diagnosis_model.predict_proba(X_val))
    test_diag_metrics = classification_metrics(y_diag_test, y_test_pred_diag, diagnosis_model.predict_proba(X_test))

    val_gold_metrics = classification_metrics(
        y_gold_val, gold_model.predict(X_gold_val), gold_model.predict_proba(X_gold_val)
    )
    test_gold_metrics = classification_metrics(
        y_gold_test, gold_model.predict(X_gold_test), gold_model.predict_proba(X_gold_test)
    )

    # --- End-to-end pipeline metric: what actually happens on the test set ---
    # Uses the PREDICTED diagnosis (not the true label) to decide who gets
    # staged, then scores GOLD predictions only among rows that are truly
    # COPD-positive AND were caught by stage 1. Truly-positive rows the
    # diagnosis stage misses never reach staging at all - that is a real
    # limitation of a cascade, reported explicitly instead of hidden.
    true_positive_mask = y_diag_test.to_numpy() == positive_code
    caught_mask = true_positive_mask & (y_test_pred_diag == positive_code)
    missed_by_diagnosis = int(true_positive_mask.sum() - caught_mask.sum())

    end_to_end_metrics = {
        "n_true_positive": int(true_positive_mask.sum()),
        "n_missed_by_diagnosis_stage": missed_by_diagnosis,
    }
    if caught_mask.any():
        X_caught = X_test[caught_mask]
        gold_true_caught = gold_encoder.transform(gold_labels_raw.loc[X_caught.index].to_numpy())
        gold_pred_caught = gold_model.predict(X_caught)
        end_to_end_metrics.update(
            {
                f"gold_stage_{k}": v
                for k, v in classification_metrics(
                    gold_true_caught, gold_pred_caught, gold_model.predict_proba(X_caught)
                ).items()
            }
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"combined_best_of_both_{ds}") as run:
        mlflow.set_tag("model_name", "combined_best_of_both")
        mlflow.set_tag("dataset", "nhanes")
        mlflow.set_tag("partition_ds", ds)
        mlflow.set_tag("problem_type", "classification")
        mlflow.set_tag("gold_stage_scope", "copd_positive_only")
        mlflow.set_tag("pipeline_kind", "routed_two_stage")
        mlflow.set_tag("diagnosis_sub_model", "random_forest")
        mlflow.set_tag("gold_stage_sub_model", "xgboost")
        mlflow.log_params(
            {
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
                "diagnosis_target_column": DIAGNOSIS_TARGET_COLUMN,
                "gold_target_column": GOLD_TARGET_COLUMN,
                "diagnosis_classes": diagnosis_encoder.classes_.tolist(),
                "gold_classes": gold_encoder.classes_.tolist(),
                "dropped_leakage_columns": dropped_leakage_cols,
                "diagnosis_max_depth": diagnosis_tuning["max_depth"],
                "diagnosis_min_samples_leaf": diagnosis_tuning["min_samples_leaf"],
                "diagnosis_decision_threshold": diag_threshold,
                "gold_max_depth": gold_tuning["max_depth"],
                "gold_learning_rate": gold_tuning["learning_rate"],
            }
        )
        mlflow.log_metrics({f"val_diagnosis_{k}": v for k, v in drop_nan(val_diag_metrics).items()})
        mlflow.log_metrics({f"test_diagnosis_{k}": v for k, v in drop_nan(test_diag_metrics).items()})
        mlflow.log_metrics({f"val_gold_stage_{k}": v for k, v in drop_nan(val_gold_metrics).items()})
        mlflow.log_metrics({f"test_gold_stage_{k}": v for k, v in drop_nan(test_gold_metrics).items()})
        mlflow.log_metrics({f"pipeline_{k}": v for k, v in drop_nan(end_to_end_metrics).items()})

        mlflow.sklearn.log_model(diagnosis_model, artifact_path="diagnosis_model")
        mlflow.xgboost.log_model(gold_model, artifact_path="gold_stage_model")

        run_id = run.info.run_id

    out_dir = artifact_dir(ds, "combined_best_of_both")
    joblib.dump(
        {
            "diagnosis_model": diagnosis_model,
            "gold_model": gold_model,
            "diagnosis_threshold": diag_threshold,
            "positive_code": positive_code,
            "negative_code": negative_code,
            "diagnosis_classes": diagnosis_encoder.classes_.tolist(),
            "gold_classes": gold_encoder.classes_.tolist(),
        },
        out_dir / "combined_system.joblib",
    )

    summary = {
        "model_name": "combined_best_of_both",
        "dataset": "nhanes",
        "partition_ds": ds,
        "mlflow_run_id": run_id,
        "gold_stage_scope": "copd_positive_only",
        "diagnosis_sub_model": "random_forest",
        "gold_stage_sub_model": "xgboost",
        "val_diagnosis": val_diag_metrics,
        "test_diagnosis": test_diag_metrics,
        "val_gold_stage": val_gold_metrics,
        "test_gold_stage": test_gold_metrics,
        "end_to_end_pipeline": end_to_end_metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[combined] mlflow_run_id={run_id}")
    print(f"[combined] test_diagnosis_f1_macro={test_diag_metrics['f1_macro']:.4f} (random_forest)")
    print(f"[combined] test_gold_stage_f1_macro={test_gold_metrics['f1_macro']:.4f} (xgboost, standalone eval)")
    print(
        f"[combined] end-to-end: {end_to_end_metrics['n_true_positive']} true positives in test, "
        f"{end_to_end_metrics['n_missed_by_diagnosis_stage']} missed by the diagnosis stage"
    )
    if "gold_stage_f1_macro" in end_to_end_metrics:
        print(f"[combined] end-to-end gold_stage_f1_macro={end_to_end_metrics['gold_stage_f1_macro']:.4f} (on rows correctly caught by stage 1)")
    print(f"[combined] wrote summary -> {out_dir / 'metrics.json'}")
    print(f"[combined] wrote reusable model bundle -> {out_dir / 'combined_system.joblib'}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the combined Random Forest + XGBoost routing pipeline.")
    parser.add_argument("--ds", default="2026-07-16", help="Partition date (YYYY-MM-DD) to load.")
    args = parser.parse_args()
    train_and_evaluate(args.ds)


if __name__ == "__main__":
    main()
