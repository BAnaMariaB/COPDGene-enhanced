"""Runtime configuration for the prediction API.

Every value is env-overridable so the same image runs locally, in docker, and on
Beanstalk/GCP without code changes.
"""

from __future__ import annotations

import os

# --- Database (app-side registry, read-only for this service) ----------------
DATABASE_URL = os.environ.get(
    "COPD_DATABASE_URL",
    "postgresql+psycopg://copd:copd@localhost:5432/copd",
)

# --- MLflow ------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

# --- Preprocessing artifacts -------------------------------------------------
# Root written by the ingestion DAG (COPD_PREPROCESSED_ROOT over there).
PREPROCESSED_ROOT = os.environ.get(
    "COPD_PREPROCESSED_ROOT",
    os.path.expanduser("~/airflow/data/preprocessed"),
)

# The *serving* preprocessor: fitted on FEATURE_COLUMNS only (no spirometry).
# See app/README.md — the DAG's current preprocessing_artifacts.joblib is fitted
# on target columns too and therefore cannot be used to serve predictions.
SERVING_ARTIFACT_NAME = os.environ.get(
    "COPD_SERVING_ARTIFACT_NAME",
    "serving_preprocessor.joblib",
)

# --- Targets -----------------------------------------------------------------
# The repo currently has ONE decided target: `gold_copd` (binary COPD via the
# GOLD criterion), from eda_feature_engineering/feature_engineering.py.
#
# CHANGES_MADE.md describes a different pair — `copd_diagnosis` + `gold_stage` —
# from an NHANES rewrite that is not in this branch. `copd_diagnosis` and
# `gold_copd` are the same quantity (fev1_fvc_ratio < 0.70) under two names;
# eda_feature_engineering/README.md open question #1 predicted exactly this
# collision. These are env-overridable so whichever name the team settles on can
# be set without a code change.
DIAGNOSIS_TARGET = os.environ.get("COPD_DIAGNOSIS_TARGET", "gold_copd")

# Optional second stage. Not built anywhere in the repo yet; screening works
# without it.
GOLD_STAGE_TARGET = os.environ.get("COPD_GOLD_STAGE_TARGET", "gold_stage")

# --- CORS --------------------------------------------------------------------
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("COPD_CORS_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]
