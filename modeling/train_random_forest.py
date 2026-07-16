"""Train Random Forest candidate models on the NHANES-derived preprocessed dataset.

Trains two independent RandomForestClassifier models (one per target) rather
than the stacked/cascaded design used by the ensemble system in
airflow/dags/copd_train_validate_test.py - this is meant to be a plain, simple
candidate model for comparison, matching the "Random Forest" line item in the
project spec's candidate model list. It uses the same split, metrics, and
MLflow experiment as that ensemble so the two are directly comparable.

Improvements applied over the first baseline run:
  1. gold_stage is now conditional on a positive diagnosis: staged severity
     only makes sense once COPD is diagnosed, and training on the full
     dataset meant ~88% of rows were "no COPD" for that model too, on top of
     the diagnosis model's own imbalance.
  2. Tree depth / leaf size are chosen by cross-validated grid search instead
     of left unbounded, which was letting trees fit noise in the rare classes.
  3. The diagnosis decision threshold is tuned on the validation set instead
     of using the default 0.5 cutoff.
  4. Hyperparameter selection uses stratified k-fold CV within the training
     set. The final reported numbers still come from the same fixed
     train/val/test split as before, so results stay comparable to the
     ensemble system.

Run with:
    python modeling/train_random_forest.py
    python modeling/train_random_forest.py --ds 2026-07-16
"""

from __future__ import annotations

import argparse
import json

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

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
    load_preprocessed,
)

N_ESTIMATORS = 400
CLASS_WEIGHT = "balanced"
PARAM_GRID = {
    "max_depth": [4, 6, 8, 10, None],
    "min_samples_leaf": [1, 4],
}
MIN_CV_FOLDS = 5
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)


def drop_nan(metrics: dict) -> dict:
    # Some splits do not contain every class (GOLD_4 has only 2 records in the
    # whole cohort), so roc_auc_ovr can come back NaN - see
    # classification_metrics. MLflow's SQLite backend does not handle logging
    # NaN metric values reliably, so skip them instead of logging a
    # placeholder value.
    return {k: v for k, v in metrics.items() if v == v}


def tune_random_forest(X_train, y_train) -> tuple[RandomForestClassifier, dict]:
    """Grid-search max_depth/min_samples_leaf via stratified k-fold CV.

    Falls back to a fixed, moderate-depth model instead of crashing when a
    class has too few members for cross-validation (GOLD_4 territory).
    """
    class_counts = np.bincount(y_train)
    n_splits = min(MIN_CV_FOLDS, int(class_counts[class_counts > 0].min()))

    if n_splits < 2:
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=6,
            min_samples_leaf=1,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model, {"max_depth": 6, "min_samples_leaf": 1, "cv_used": False, "cv_folds": 0}

    base_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search = GridSearchCV(
        base_model,
        PARAM_GRID,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    tuning_info = {**search.best_params_, "cv_used": True, "cv_folds": n_splits}
    return search.best_estimator_, tuning_info


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


def train_and_evaluate(ds: str) -> dict:
    df = load_preprocessed(ds)
    X, y_diagnosis, _old_gold, diagnosis_encoder, _old_gold_encoder = build_targets(df)

    dropped_leakage_cols = [c for c in SPIROMETRY_LEAKAGE_COLUMNS if c in X.columns]
    X = X.drop(columns=dropped_leakage_cols)

    gold_labels_raw, gold_encoder = build_gold_positive_labels(df, X)

    # Split on diagnosis only, same as before. GOLD stage rows are derived
    # afterward as a positive-only subset of each split.
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

    # --- diagnosis model: CV-tuned depth, then a tuned decision threshold ---
    diagnosis_model, diagnosis_tuning = tune_random_forest(X_train, y_diag_train)

    positive_code = diagnosis_encoder.transform(["copd"])[0]
    negative_code = diagnosis_encoder.transform(["no_copd"])[0]
    diag_threshold, diag_threshold_val_f1 = tune_threshold(
        diagnosis_model, X_val, y_diag_val, positive_code, negative_code
    )

    y_val_pred_diag = predict_with_threshold(diagnosis_model, X_val, positive_code, negative_code, diag_threshold)
    y_test_pred_diag = predict_with_threshold(diagnosis_model, X_test, positive_code, negative_code, diag_threshold)
    val_diag_metrics = classification_metrics(y_diag_val, y_val_pred_diag, diagnosis_model.predict_proba(X_val))
    test_diag_metrics = classification_metrics(y_diag_test, y_test_pred_diag, diagnosis_model.predict_proba(X_test))

    # --- gold stage model: CV-tuned depth, positive-only, plain argmax ---
    gold_model, gold_tuning = tune_random_forest(X_gold_train, y_gold_train)

    def evaluate(model, X_eval, y_eval):
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)
        return classification_metrics(y_eval, y_pred, y_proba)

    val_gold_metrics = evaluate(gold_model, X_gold_val, y_gold_val)
    test_gold_metrics = evaluate(gold_model, X_gold_test, y_gold_test)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"random_forest_tuned_{ds}") as run:
        mlflow.set_tag("model_name", "random_forest")
        mlflow.set_tag("dataset", "nhanes")
        mlflow.set_tag("partition_ds", ds)
        mlflow.set_tag("problem_type", "classification")
        mlflow.set_tag("gold_stage_scope", "copd_positive_only")
        mlflow.set_tag("tuning", "cv_grid_search_and_threshold")
        mlflow.log_params(
            {
                "n_estimators": N_ESTIMATORS,
                "class_weight": CLASS_WEIGHT,
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
                "diagnosis_cv_folds": diagnosis_tuning["cv_folds"],
                "diagnosis_decision_threshold": diag_threshold,
                "gold_max_depth": gold_tuning["max_depth"],
                "gold_min_samples_leaf": gold_tuning["min_samples_leaf"],
                "gold_cv_folds": gold_tuning["cv_folds"],
            }
        )
        mlflow.log_metrics({f"val_diagnosis_{k}": v for k, v in drop_nan(val_diag_metrics).items()})
        mlflow.log_metrics({f"test_diagnosis_{k}": v for k, v in drop_nan(test_diag_metrics).items()})
        mlflow.log_metrics({f"val_gold_stage_{k}": v for k, v in drop_nan(val_gold_metrics).items()})
        mlflow.log_metrics({f"test_gold_stage_{k}": v for k, v in drop_nan(test_gold_metrics).items()})

        mlflow.sklearn.log_model(diagnosis_model, artifact_path="diagnosis_model")
        mlflow.sklearn.log_model(gold_model, artifact_path="gold_stage_model")

        run_id = run.info.run_id

    out_dir = artifact_dir(ds, "random_forest")
    summary = {
        "model_name": "random_forest",
        "dataset": "nhanes",
        "partition_ds": ds,
        "mlflow_run_id": run_id,
        "gold_stage_scope": "copd_positive_only",
        "diagnosis_tuning": diagnosis_tuning,
        "diagnosis_decision_threshold": diag_threshold,
        "gold_tuning": gold_tuning,
        "val_diagnosis": val_diag_metrics,
        "test_diagnosis": test_diag_metrics,
        "val_gold_stage": val_gold_metrics,
        "test_gold_stage": test_gold_metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[random_forest] mlflow_run_id={run_id}")
    print(f"[random_forest] diagnosis: max_depth={diagnosis_tuning['max_depth']} threshold={diag_threshold:.2f}")
    print(f"[random_forest] test_diagnosis_f1_macro={test_diag_metrics['f1_macro']:.4f}")
    print(f"[random_forest] gold_stage: max_depth={gold_tuning['max_depth']} (positive-only, {len(y_gold_train)} train rows)")
    print(f"[random_forest] test_gold_stage_f1_macro={test_gold_metrics['f1_macro']:.4f}")
    print(f"[random_forest] wrote summary -> {out_dir / 'metrics.json'}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest candidate models on NHANES data.")
    parser.add_argument("--ds", default="2026-07-16", help="Partition date (YYYY-MM-DD) to load.")
    args = parser.parse_args()
    train_and_evaluate(args.ds)


if __name__ == "__main__":
    main()
