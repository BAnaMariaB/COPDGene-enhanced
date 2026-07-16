"""Feature engineering for the COPD dataset.

TARGET DECISION (per team discussion, 2026-07-16 evening): regression on
`fev1_phase2` hit an RMSE ceiling -- the only strongly correlated feature is
baseline `fev1` itself, and a single predicted number isn't useful in
production anyway. The team is switching to classification:

  - `gold_copd` (RECOMMENDED_TARGET) -- derived: `fev1_fvc_ratio < 0.70` (the
    GOLD diagnostic criterion for airflow obstruction). Binary
    classification: does this participant have COPD or not.
  - `fev1` / `fev1_phase2` -- kept as reference/rejected candidates, not for
    modeling. They're the evidence for *why* the team pivoted (worth keeping
    for the presentation), not something to train the real model on.

NOT built here yet: a second-stage "if COPD-positive, predict GOLD stage
severity (1-4)" cascade, which the team also discussed. Real GOLD staging
needs age/sex/height-adjusted % predicted FEV1 (reference equations), which is
what the NHANES data being pulled in separately is for. Building a competing
version of that here risks duplicating/conflicting with that work -- it's a
documented future extension (see GOLD_STAGE_CASCADE_NOTE below), not
implemented.

This module still builds a leakage-correct, target-ready dataset for all
three candidates (see TARGET_CANDIDATES), so the rejected ones stay available
for comparison even though gold_copd is the one to actually model on.
"""

from __future__ import annotations

import json

import pandas as pd

from paths import OUTPUT_DIR

CATEGORICAL_COLUMNS = ["gender", "race", "smoking_status"]

# `respiratory` is pipe-delimited multi-label free text, e.g.
# "hay fever|bronchitis attacks|pneumonia" -- NOT a single category. ~25.6% of
# rows have no value reported. NOTE: the shared ingestion DAG's `preprocessing`
# task (airflow/dags/copd_ingestion.py) drops this column entirely rather than
# parsing it -- our parsed flags are an intentional addition on top of that.
RESPIRATORY_COLUMN = "respiratory"

GOLD_THRESHOLD = 0.70

RECOMMENDED_TARGET = "gold_copd"

# Second-stage cascade the team also discussed (binary COPD -> if positive,
# predict GOLD stage severity 1-4). NOT implemented here: real GOLD staging
# needs age/sex/height-adjusted % predicted FEV1, which depends on the NHANES
# reference data someone else on the team is integrating separately. Building
# our own version now would risk a second divergent implementation of the
# same thing -- wire this in once that reference data/equation is shared,
# don't reinvent it here.
GOLD_STAGE_CASCADE_NOTE = (
    "Planned second-stage target, not yet built: for gold_copd-positive rows, "
    "classify GOLD stage severity (1-4). Needs % predicted FEV1 from "
    "age/sex/height-adjusted reference equations (NHANES-based work in "
    "progress elsewhere on the team) -- do not build a competing version "
    "of this independently."
)

# For each candidate target: which OTHER columns must be dropped from the
# feature set, and why. Columns not listed here are safe to keep as features
# for that target. `sid` is never a feature for any target -- it's kept in
# the output CSV for traceability only (see README.md).
TARGET_CANDIDATES = {
    "gold_copd": {
        "status": "recommended -- team decision, 2026-07-16",
        "kind": "classification",
        "description": f"GOLD criterion: fev1_fvc_ratio < {GOLD_THRESHOLD} (airflow obstruction).",
        "excluded_features": ["fev1_fvc_ratio", "fev1", "fev1_phase2"],
        "exclusion_reasoning": {
            "fev1_fvc_ratio": "this is the exact value the label is thresholded from -- direct leakage.",
            "fev1": (
                "fev1 + fvc together reconstruct fev1_fvc_ratio exactly, so the "
                "label is deterministically recoverable if both are kept -- "
                "drop fev1, keep fvc alone as the spirometry input."
            ),
            "fev1_phase2": "measured five years after baseline -- not available at prediction time.",
        },
    },
    "fev1_phase2": {
        "status": "rejected -- RMSE ceiling (only fev1 correlates), not production-useful as a single number",
        "kind": "regression",
        "description": "FEV1 measured five years after baseline -- predicts lung-function decline.",
        "excluded_features": [],
        "exclusion_reasoning": {
            "_none": (
                "fev1, fvc, fev1_fvc_ratio, and gold_copd are all baseline "
                "values known before the 5-year follow-up, so they're "
                "legitimate predictors here -- no leakage. (Kept for "
                "reference/comparison only -- not the modeling target.)"
            ),
        },
    },
    "fev1": {
        "status": "reference only -- not a target the team is pursuing",
        "kind": "regression",
        "description": "Baseline FEV1 (current lung function).",
        "excluded_features": ["fev1_fvc_ratio", "fev1_phase2", "gold_copd"],
        "exclusion_reasoning": {
            "fev1_fvc_ratio": "fev1 = fev1_fvc_ratio * fvc almost exactly -> direct leakage.",
            "fev1_phase2": (
                "measured five years after baseline -> not available at "
                "prediction time for a baseline target (it's a separate "
                "candidate target, not a feature)."
            ),
            "gold_copd": "derived from fev1_fvc_ratio, which already leaks fev1 -> same leakage, one step removed.",
        },
    },
}


def _split_conditions(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    # --- clinically meaningful derived features -----------------------------
    out["pulse_pressure"] = out["blood_pressure_systolic"] - out["blood_pressure_diastolic"]

    # Pack-years: standard smoking-exposure metric (packs/day * years smoked).
    # This cohort is smokers-only (COPDGene enrollment criteria), so
    # cigs_per_day_avg / duration_smoking have no missing values to special-case.
    out["pack_years"] = (out["cigs_per_day_avg"] / 20.0) * out["duration_smoking"]

    # Air-trapping proxy from imaging: how much less air comes out than went in.
    out["air_trapping_ratio"] = out["lung_volume_expiratory"] / out["lung_volume_inspiratory"]

    # Multi-label respiratory history -> one boolean flag per distinct
    # condition (NOT one flag per pipe-combo -- a naive value_counts().head(N)
    # would treat "hay fever|pneumonia" as its own category, which is wrong).
    out["respiratory_reported"] = out[RESPIRATORY_COLUMN].notna()
    condition_lists = out[RESPIRATORY_COLUMN].apply(_split_conditions)
    all_conditions = sorted({c for conds in condition_lists for c in conds})
    engineered_respiratory_cols = []
    for condition in all_conditions:
        safe_name = condition.lower().replace(" ", "_")
        col_name = f"respiratory_is_{safe_name}"
        out[col_name] = condition_lists.apply(lambda conds, c=condition: c in conds)
        engineered_respiratory_cols.append(col_name)

    # Derived classification target candidate: GOLD criterion. Preserve NaN
    # for rows with missing fev1_fvc_ratio instead of silently coercing them
    # to False (`NaN < 0.70` is False in pandas, which would be wrong here).
    ratio = out["fev1_fvc_ratio"]
    out["gold_copd"] = (ratio < GOLD_THRESHOLD).astype("boolean")
    out.loc[ratio.isna(), "gold_copd"] = pd.NA

    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype("category")

    manifest = {
        "recommended_target": RECOMMENDED_TARGET,
        "gold_stage_cascade_note": GOLD_STAGE_CASCADE_NOTE,
        "target_candidates": TARGET_CANDIDATES,
        "reference_columns_not_features": {
            "sid": "subject identifier -- kept in the CSV for traceability, never a feature for any target.",
        },
        "flagged_not_dropped": {
            "respiratory_is_copd": (
                "a COPD diagnosis flag is clinically correlated with FEV1/GOLD "
                "status by definition -- not exact leakage, but review with the "
                "team before using it as a feature for the gold_copd target."
            ),
        },
        "engineered_columns": [
            "pulse_pressure",
            "pack_years",
            "air_trapping_ratio",
            "respiratory_reported",
            "gold_copd",
        ]
        + engineered_respiratory_cols,
        "categorical_columns_cast": CATEGORICAL_COLUMNS,
        "respiratory_conditions_found": all_conditions,
        "gold_threshold": GOLD_THRESHOLD,
    }

    return out, manifest


def build_dataset_for_target(df: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, dict]:
    """Return a feature set that's leakage-correct for `target_name`.

    The target column itself stays in the output (as `y` for whoever trains
    on it); only the OTHER columns that would leak it are dropped.
    """
    spec = TARGET_CANDIDATES[target_name]
    drop_cols = [c for c in spec["excluded_features"] if c in df.columns]
    dataset = df.drop(columns=drop_cols, errors="ignore")
    info = {
        "target": target_name,
        "kind": spec["kind"],
        "description": spec["description"],
        "dropped_columns": drop_cols,
        "exclusion_reasoning": spec["exclusion_reasoning"],
    }
    return dataset, info


def write_manifest(manifest: dict, path=None) -> None:
    path = path or (OUTPUT_DIR / "feature_engineering_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"[feature_engineering] wrote manifest -> {path}")
