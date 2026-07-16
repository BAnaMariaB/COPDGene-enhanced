"""Train XGBoost candidate models on the NHANES-derived preprocessed dataset.

Mirrors train_random_forest.py exactly in structure (same data, same leakage
columns dropped, same gold_stage-conditional-on-diagnosis restructuring, same
CV-tuned hyperparameters, same threshold tuning, same MLflow experiment) so
the two are directly comparable as separate candidate models. Only the
underlying estimator changes: XGBClassifier instead of RandomForestClassifier.

XGBoost has no class_weight="balanced" option like scikit-learn's ensembles,
so class imbalance is handled via per-sample weights (compute_sample_weight)
instead, which works the same way for both the binary and multiclass targets.

Run with:
    python modeling/train_xgboost.py
    python modeling/train_xgboost.py --ds 2026-07-16
"""

from __future__ import annotations

import argparse
import json

import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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

N_ESTIMATORS = 300
PARAM_GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.05, 0.1],
}
MIN_CV_FOLDS = 5


def tune_xgboost(X_train, y_train) -> tuple[XGBClassifier, dict]:
    """Grid-search max_depth/learning_rate via stratified k-fold CV.

    Falls back to a fixed set of hyperparameters instead of crashing when a
    class has too few members for cross-validation (GOLD_4 territory).
    """
    class_counts = np.bincount(y_train)
    n_splits = min(MIN_CV_FOLDS, int(class_counts[class_counts > 0].min()))
    sample_weight = compute_sample_weight("balanced", y_train)
    num_class = len(np.unique(y_train))
    objective = "binary:logistic" if num_class == 2 else "multi:softprob"
    eval_metric = "logloss" if num_class == 2 else "mlogloss"

    if n_splits < 2:
        model = XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=4,
            learning_rate=0.1,
            objective=objective,
            eval_metric=eval_metric,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model, {
            "max_depth": 4, "learning_rate": 0.1, "n_estimators": N_ESTIMATORS,
            "cv_used": False, "cv_folds": 0,
        }

    base_model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        objective=objective,
        eval_metric=eval_metric,
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
    search.fit(X_train, y_train, sample_weight=sample_weight)
    tuning_info = {**search.best_params_, "n_estimators": N_ESTIMATORS, "cv_used": True, "cv_folds": n_splits}
    return search.best_estimator_, tuning_info


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

    # --- diagnosis model: CV-tuned depth/learning_rate, then a tuned threshold ---
    diagnosis_model, diagnosis_tuning = tune_xgboost(X_train, y_diag_train)

    positive_code = diagnosis_encoder.transform(["copd"])[0]
    negative_code = diagnosis_encoder.transform(["no_copd"])[0]
    diag_threshold, diag_threshold_val_f1 = tune_threshold(
        diagnosis_model, X_val, y_diag_val, positive_code, negative_code
    )

    y_val_pred_diag = predict_with_threshold(diagnosis_model, X_val, positive_code, negative_code, diag_threshold)
    y_test_pred_diag = predict_with_threshold(diagnosis_model, X_test, positive_code, negative_code, diag_threshold)
    val_diag_metrics = classification_metrics(y_diag_val, y_val_pred_diag, diagnosis_model.predict_proba(X_val))
    test_diag_metrics = classification_metrics(y_diag_test, y_test_pred_diag, diagnosis_model.predict_proba(X_test))

    # --- gold stage model: CV-tuned depth/learning_rate, positive-only, plain argmax ---
    gold_model, gold_tuning = tune_xgboost(X_gold_train, y_gold_train)

    def evaluate(model, X_eval, y_eval):
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)
        return classification_metrics(y_eval, y_pred, y_proba)

    val_gold_metrics = evaluate(gold_model, X_gold_val, y_gold_val)
    test_gold_metrics = evaluate(gold_model, X_gold_test, y_gold_test)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"xgboost_tuned_{ds}") as run:
        mlflow.set_tag("model_name", "xgboost")
        mlflow.set_tag("dataset", "nhanes")
        mlflow.set_tag("partition_ds", ds)
        mlflow.set_tag("problem_type", "classification")
        mlflow.set_tag("gold_stage_scope", "copd_positive_only")
        mlflow.set_tag("tuning", "cv_grid_search_and_threshold")
        mlflow.log_params(
            {
                "n_estimators": N_ESTIMATORS,
                "class_imbalance_handling": "compute_sample_weight(balanced)",
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
                "diagnosis_target_column": DIAGNOSIS_TARGET_COLUMN,
                "gold_target_column": GOLD_TARGET_COLUMN,
                "diagnosis_classes": diagnosis_encoder.classes_.tolist(),
                "gold_classes": gold_encoder.classes_.tolist(),
                "dropped_leakage_columns": dropped_leakage_cols,
                "diagnosis_max_depth": diagnosis_tuning["max_depth"],
                "diagnosis_learning_rate": diagnosis_tuning["learning_rate"],
                "diagnosis_cv_folds": diagnosis_tuning["cv_folds"],
                "diagnosis_decision_threshold": diag_threshold,
                "gold_max_depth": gold_tuning["max_depth"],
                "gold_learning_rate": gold_tuning["learning_rate"],
                "gold_cv_folds": gold_tuning["cv_folds"],
            }
        )
        mlflow.log_metrics({f"val_diagnosis_{k}": v for k, v in drop_nan(val_diag_metrics).items()})
        mlflow.log_metrics({f"test_diagnosis_{k}": v for k, v in drop_nan(test_diag_metrics).items()})
        mlflow.log_metrics({f"val_gold_stage_{k}": v for k, v in drop_nan(val_gold_metrics).items()})
        mlflow.log_metrics({f"test_gold_stage_{k}": v for k, v in drop_nan(test_gold_metrics).items()})

        mlflow.xgboost.log_model(diagnosis_model, artifact_path="diagnosis_model")
        mlflow.xgboost.log_model(gold_model, artifact_path="gold_stage_model")

        run_id = run.info.run_id

    out_dir = artifact_dir(ds, "xgboost")
    summary = {
        "model_name": "xgboost",
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

    print(f"[xgboost] mlflow_run_id={run_id}")
    print(f"[xgboost] diagnosis: max_depth={diagnosis_tuning['max_depth']} lr={diagnosis_tuning['learning_rate']} threshold={diag_threshold:.2f}")
    print(f"[xgboost] test_diagnosis_f1_macro={test_diag_metrics['f1_macro']:.4f}")
    print(f"[xgboost] gold_stage: max_depth={gold_tuning['max_depth']} lr={gold_tuning['learning_rate']} (positive-only, {len(y_gold_train)} train rows)")
    print(f"[xgboost] test_gold_stage_f1_macro={test_gold_metrics['f1_macro']:.4f}")
    print(f"[xgboost] wrote summary -> {out_dir / 'metrics.json'}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost candidate models on NHANES data.")
    parser.add_argument("--ds", default="2026-07-16", help="Partition date (YYYY-MM-DD) to load.")
    args = parser.parse_args()
    train_and_evaluate(args.ds)


if __name__ == "__main__":
    main()
