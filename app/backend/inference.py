"""Champion model loading + the prediction cascade.

The repo has ONE decided target: `gold_copd` (binary COPD via the GOLD
criterion), from eda_feature_engineering/. Screening on its own is the whole app
today.

A second severity-staging stage is planned but deliberately unbuilt — see
feature_engineering.GOLD_STAGE_CASCADE_NOTE. The cascade below is wired for it
and degrades cleanly while it doesn't exist. When it lands it will be
CONDITIONAL: trained only on COPD-positive rows, so it never sees a negative
example. Feeding it a screen-negative subject would ask it to pick among four
severity grades for someone it believes has no disease. It would answer anyway,
with confidence. So:

    gold_copd  -> if positive -> gold_stage (when a champion exists)
               -> else        -> no staging, and say why

Artifacts needed to serve:

  1. gold_copd champion       — MLflow, located via artifact_uri in Postgres
  2. gold_stage champion      — optional; staging degrades gracefully without it
  3. serving_preprocessor.joblib — a ColumnTransformer fitted on
     features.EXPECTED_MODEL_COLUMNS and nothing else

`preprocessing_artifacts.joblib` from the ingestion DAG is NOT (3): that task
still leaves all four spirometry columns in as undifferentiated features with no
target handling. See validate_preprocessor().
"""

from __future__ import annotations

import os
import threading
from typing import Any

import joblib
import numpy as np
import pandas as pd

import config
from features import (
    EXPECTED_MODEL_COLUMNS,
    TARGET_COLUMN,
    TARGET_SIDE_COLUMNS,
    derive,
)

DIAGNOSIS_TARGET = config.DIAGNOSIS_TARGET
GOLD_TARGET = config.GOLD_STAGE_TARGET

# gold_copd is a boolean target, so a champion may register its classes as
# ["false","true"], ["no_copd","copd"], or ["0","1"]. Any of these count as
# positive; matching one hardcoded string would silently never stage.
POSITIVE_LABELS = {"copd", "true", "1", "gold_copd", "yes", "positive"}


class ModelNotReady(RuntimeError):
    """Raised when the app cannot serve predictions, with a fixable reason."""


class _Cache:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.run_ids: dict[str, str] = {}
        self.models: dict[str, Any] = {}
        self.preprocessor: Any = None


_cache = _Cache()


def reset_cache() -> None:
    with _cache.lock:
        _cache.run_ids = {}
        _cache.models = {}
        _cache.preprocessor = None


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def _find_serving_artifact() -> str | None:
    """Newest partition under PREPROCESSED_ROOT holding a serving preprocessor."""
    root = config.PREPROCESSED_ROOT
    if not os.path.isdir(root):
        return None
    for ds in sorted(os.listdir(root), reverse=True):
        candidate = os.path.join(root, ds, config.SERVING_ARTIFACT_NAME)
        if os.path.isfile(candidate):
            return candidate
    return None


def validate_preprocessor(pre: Any) -> None:
    """Fail loudly if the preprocessor doesn't match the serving contract."""
    try:
        fitted = list(pre.feature_names_in_)
    except AttributeError as exc:
        raise ModelNotReady(
            "Serving preprocessor has no feature_names_in_; it was not fitted on a "
            "named DataFrame. Refit it with a pandas DataFrame."
        ) from exc

    leaked = [c for c in fitted if c in TARGET_SIDE_COLUMNS]
    if leaked:
        raise ModelNotReady(
            f"Serving preprocessor was fitted on target-side columns {leaked}. "
            f"'{TARGET_COLUMN}' is defined as fev1_fvc_ratio < 0.70, and fev1 + fvc "
            "reconstruct that ratio exactly, so these are the target rather than "
            "features. This matches excluded_features in "
            "eda_feature_engineering/feature_engineering.py::TARGET_CANDIDATES. "
            "Refit on features.EXPECTED_MODEL_COLUMNS only."
        )

    if TARGET_COLUMN in fitted:
        raise ModelNotReady(
            f"Serving preprocessor was fitted on the label column '{TARGET_COLUMN}'."
        )

    unknown = [c for c in fitted if c not in EXPECTED_MODEL_COLUMNS]
    if unknown:
        raise ModelNotReady(
            f"Serving preprocessor expects columns the app does not supply: {unknown}. "
            f"The app supplies: {EXPECTED_MODEL_COLUMNS}. Either add them to "
            "FEATURE_SCHEMA in features.py, or refit without them. (Column names in "
            "features.py mirror eda_feature_engineering/feature_engineering.py; if "
            "that module's engineered columns changed, update FEATURE_SCHEMA.)"
        )

    missing = [c for c in EXPECTED_MODEL_COLUMNS if c not in fitted]
    if missing:
        raise ModelNotReady(
            f"Serving preprocessor was not fitted on {missing}, which the form "
            "collects. Refit, or remove them from FEATURE_SCHEMA."
        )


def load_preprocessor() -> Any:
    path = _find_serving_artifact()
    if path is None:
        raise ModelNotReady(
            f"No '{config.SERVING_ARTIFACT_NAME}' found under {config.PREPROCESSED_ROOT}. "
            "The preprocessing task must save a ColumnTransformer fitted on serving "
            "features only, alongside preprocessing_artifacts.joblib."
        )
    obj = joblib.load(path)
    pre = obj.get("preprocessor") if isinstance(obj, dict) else obj
    validate_preprocessor(pre)
    return pre


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_model(artifact_uri: str) -> Any:
    import mlflow  # lazy: keeps startup fast when MLflow is unreachable
    import mlflow.pyfunc

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    try:
        return mlflow.pyfunc.load_model(artifact_uri)
    except Exception as exc:
        raise ModelNotReady(
            f"Could not load artifact from '{artifact_uri}': {exc}"
        ) from exc


def ensure_loaded(champions: dict[str, dict[str, Any]]) -> None:
    """Load/refresh cached models whose champion run id has changed.

    The diagnosis model is mandatory. The gold_stage model is optional: screening
    is still useful on its own, and staging simply reports as unavailable.
    """
    if DIAGNOSIS_TARGET not in champions:
        raise ModelNotReady(
            f"No champion registered for '{DIAGNOSIS_TARGET}'. The modeling DAG has "
            "not written a row with is_champion = TRUE for that target."
        )

    with _cache.lock:
        if _cache.preprocessor is None:
            _cache.preprocessor = load_preprocessor()

        for target, champ in champions.items():
            run_id = champ["mlflow_run_id"]
            if _cache.run_ids.get(target) == run_id and target in _cache.models:
                continue
            _cache.models[target] = load_model(champ["artifact_uri"])
            _cache.run_ids[target] = run_id

        for stale in set(_cache.models) - set(champions):
            _cache.models.pop(stale, None)
            _cache.run_ids.pop(stale, None)


def status() -> dict[str, Any]:
    return {
        "models_loaded": sorted(_cache.models),
        "run_ids": dict(_cache.run_ids),
        "preprocessor_loaded": _cache.preprocessor is not None,
    }


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def _to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame([derive(r) for r in rows])
    for col in EXPECTED_MODEL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[EXPECTED_MODEL_COLUMNS]


def _run(target: str, X: Any, class_labels: list[str]) -> list[dict[str, Any]]:
    model = _cache.models[target]
    raw = np.asarray(model.predict(pd.DataFrame(X)))

    out: list[dict[str, Any]] = []
    if raw.ndim == 2 and raw.shape[1] == len(class_labels):
        for probs in raw:
            idx = int(np.argmax(probs))
            out.append({
                "prediction": class_labels[idx],
                "probabilities": {l: float(p) for l, p in zip(class_labels, probs)},
            })
    else:
        for label in raw.ravel():
            out.append({"prediction": str(label), "probabilities": None})
    return out


def predict(rows: list[dict[str, Any]],
            champions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the cascade. One result dict per input row."""
    if _cache.preprocessor is None or DIAGNOSIS_TARGET not in _cache.models:
        raise ModelNotReady("Models are not loaded.")

    X = _cache.preprocessor.transform(_to_frame(rows))

    diag_champ = champions[DIAGNOSIS_TARGET]
    diagnoses = _run(DIAGNOSIS_TARGET, X, diag_champ["class_labels"])

    results: list[dict[str, Any]] = []
    for i, diag in enumerate(diagnoses):
        results.append({
            "diagnosis": {
                **diag,
                "model_name": diag_champ["model_name"],
                "mlflow_run_id": diag_champ["mlflow_run_id"],
            },
            "gold_stage": None,
            "staging_skipped_reason": None,
        })

    positive_idx = [
        i for i, d in enumerate(diagnoses)
        if str(d["prediction"]).strip().lower() in POSITIVE_LABELS
    ]

    if not positive_idx:
        for r in results:
            r["staging_skipped_reason"] = "screen_negative"
        return results

    gold_champ = champions.get(GOLD_TARGET)
    if gold_champ is None or GOLD_TARGET not in _cache.models:
        for i in positive_idx:
            results[i]["staging_skipped_reason"] = "no_gold_stage_champion"
        for i in set(range(len(results))) - set(positive_idx):
            results[i]["staging_skipped_reason"] = "screen_negative"
        return results

    X_pos = X[positive_idx]
    stages = _run(GOLD_TARGET, X_pos, gold_champ["class_labels"])

    for i, stage in zip(positive_idx, stages):
        results[i]["gold_stage"] = {
            **stage,
            "model_name": gold_champ["model_name"],
            "mlflow_run_id": gold_champ["mlflow_run_id"],
        }
    for i in set(range(len(results))) - set(positive_idx):
        results[i]["staging_skipped_reason"] = "screen_negative"

    return results
