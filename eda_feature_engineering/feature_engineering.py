"""Feature engineering for the FEV1 prediction target.

Produces a feature-engineered dataframe (still unencoded/unscaled — see
README.md for why encoding is deliberately left as an open question) plus a
written manifest documenting every decision made along the way.
"""

from __future__ import annotations

import json

import pandas as pd

from paths import OUTPUT_DIR

TARGET = "fev1"

# See eda.py for the reasoning — these two let a model reconstruct FEV1
# almost exactly rather than predict it.
LEAKAGE_COLUMNS = ["fev1_fvc_ratio", "fev1_phase2"]

CATEGORICAL_COLUMNS = ["gender", "race", "smoking_status"]

# `respiratory` is pipe-delimited multi-label free text, e.g.
# "hay fever|bronchitis attacks|pneumonia" — NOT a single category. ~25.6% of
# rows have no value reported.
RESPIRATORY_COLUMN = "respiratory"


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
    # condition (NOT one flag per pipe-combo — a naive value_counts().head(N)
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

    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype("category")

    dropped_for_leakage = [c for c in LEAKAGE_COLUMNS if c in out.columns]

    manifest = {
        "target": TARGET,
        "dropped_for_leakage": dropped_for_leakage,
        "leakage_reasoning": {
            "fev1_fvc_ratio": "fev1 = fev1_fvc_ratio * fvc almost exactly -> direct leakage.",
            "fev1_phase2": "looks like a repeat FEV1 measurement for the same participant -> near-duplicate of the target.",
        },
        "flagged_not_dropped": {
            "fvc": (
                "kept as a feature (physiologically legitimate input), but combined "
                "with fev1_fvc_ratio it reconstructs fev1 exactly -- only use one of "
                "the two alongside fvc."
            ),
            "respiratory_is_copd": (
                "a COPD diagnosis flag is clinically correlated with FEV1 by definition "
                "(diagnostic criteria use an FEV1/FVC threshold) -- not exact leakage like "
                "the ratio, but review with the team before using it as a feature."
            ),
        },
        "engineered_columns": [
            "pulse_pressure",
            "pack_years",
            "air_trapping_ratio",
            "respiratory_reported",
        ]
        + engineered_respiratory_cols,
        "categorical_columns_cast": CATEGORICAL_COLUMNS,
        "respiratory_conditions_found": all_conditions,
    }

    return out, manifest


def build_model_ready(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    engineered, manifest = engineer_features(df)
    model_ready = engineered.drop(columns=manifest["dropped_for_leakage"], errors="ignore")
    return model_ready, manifest


def write_manifest(manifest: dict, path=None) -> None:
    path = path or (OUTPUT_DIR / "feature_engineering_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"[feature_engineering] wrote manifest -> {path}")
