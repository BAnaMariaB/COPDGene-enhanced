"""
COPD data ingestion DAG.

SCOPE: ingest the real NHANES 2011-2012 primary dataset, build a COPD cohort with
derived clinical features, and produce a preprocessed dataset for the training DAG.

Sources (all keyed by `SEQN`):
  - DEMO_G : demographics (age, sex, race/ethnicity)
  - SPX_G  : spirometry (FVC, FEV1, quality flags)
  - BMX_G  : body measures (height, weight, BMI)
  - SMQ_G  : smoking questionnaire (status, pack-years)

The NHANES files are downloaded once to a shared raw zone. A downstream task then
merges them, filters to adults with acceptable spirometry, derives clinically
meaningful features (FEV1/FVC ratio, FEV1 % predicted, GOLD stage, BMI category,
age group, pack-years), and writes the preprocessed dataset with two target
columns: `copd_diagnosis` and `gold_stage`.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from airflow.sdk import dag, task

# Allow the DAG to import the standalone NHANES cohort builder in scripts/.
_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root of the raw landing zone. Overridable via env var so the same DAG works
# in docker (default) and in any other Airflow environment.
AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", os.path.expanduser("~/airflow"))
RAW_ROOT = os.environ.get("COPD_RAW_ROOT", os.path.join(AIRFLOW_HOME, "data", "raw"))

# HTTP settings
REQUEST_TIMEOUT = 60          # seconds
REQUEST_MAX_RETRIES = 3       # handled by Airflow task retries too; this is per-call

# Preprocessing artifacts are stored separately from raw landing data.
ARTIFACT_ROOT = os.environ.get("COPD_ARTIFACT_ROOT", os.path.join(AIRFLOW_HOME, "data", "artifacts"))
PREPROCESSED_ROOT = os.environ.get(
    "COPD_PREPROCESSED_ROOT",
    os.path.join(AIRFLOW_HOME, "data", "preprocessed"),
)

# NHANES 2011-2012 primary source files. These are real CDC data files used to
# build a custom COPD cohort, replacing the previous public teaching files.
NHANES_SOURCES = {
    "DEMO_G": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.xpt",
        "ext": "xpt",
    },
    "SPX_G": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/SPX_G.xpt",
        "ext": "xpt",
    },
    "BMX_G": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BMX_G.xpt",
        "ext": "xpt",
    },
    "SMQ_G": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/SMQ_G.xpt",
        "ext": "xpt",
    },
}

# Legacy teaching sources (kept for reference, no longer used by default).
SOURCES = {
    "demographics": {
        "url": "https://raw.githubusercontent.com/khasenst/datasets_teaching/refs/heads/main/copd_data_demographics.csv",
        "ext": "csv",
    },
    "imaging": {
        "url": "https://raw.githubusercontent.com/khasenst/datasets_teaching/refs/heads/main/copd_data_imaging.json",
        "ext": "json",
    },
    "spirometry": {
        "url": "https://raw.githubusercontent.com/khasenst/datasets_teaching/refs/heads/main/copd_data_spirometry.html",
        "ext": "html",
    },
}


# ---------------------------------------------------------------------------
# Ingestion logic
# ---------------------------------------------------------------------------

@task
def ingest_source(source_name: str, url: str, ext: str, ds: str = None, run_id: str = None) -> str:
    """Download one raw source and land it, byte-for-byte, in the raw zone.

    Returns the path of the landed file (also pushed to XCom automatically).
    Pure ingestion: no decoding, parsing, or transformation is performed.
    """
    # Partition by the DAG run's logical date (data interval start).
    if ds is None:
        raise ValueError("Airflow did not inject ds into ingest_source")
    partition_dir = os.path.join(RAW_ROOT, source_name, ds)
    os.makedirs(partition_dir, exist_ok=True)

    dest_path = os.path.join(partition_dir, f"{source_name}.{ext}")
    meta_path = dest_path + ".meta.json"

    # Download the raw bytes.
    session = requests.Session()
    last_err = None
    for attempt in range(1, REQUEST_MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            break
        except requests.RequestException as err:  # network / HTTP error
            last_err = err
            if attempt == REQUEST_MAX_RETRIES:
                raise
    else:  # pragma: no cover - loop always breaks or raises
        raise last_err

    raw_bytes = resp.content  # exact bytes, undecoded

    # Write raw payload unchanged.
    with open(dest_path, "wb") as fh:
        fh.write(raw_bytes)

    # Build an ingestion metadata sidecar (provenance / lineage, not content changes).
    checksum = hashlib.sha256(raw_bytes).hexdigest()
    metadata = {
        "source_name": source_name,
        "source_url": url,
        "http_status": resp.status_code,
        "content_type": resp.headers.get("Content-Type"),
        "content_length_reported": resp.headers.get("Content-Length"),
        "bytes_written": len(raw_bytes),
        "sha256": checksum,
        "landed_path": dest_path,
        "partition_ds": ds,
        "dag_run_id": run_id,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "ingested_by_host": socket.gethostname(),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(
        f"[ingest] {source_name}: {len(raw_bytes)} bytes -> {dest_path} "
        f"(sha256={checksum[:12]}...)"
    )
    return dest_path


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "ingestion",
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="copd_ingestion",
    description="Ingest NHANES 2011-2012 sources and build a COPD cohort with derived clinical features.",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 * * * *",  # hourly, at the top of every hour
    catchup=False,
    max_active_runs=1,
    tags=["copd", "ingestion", "nhanes", "cohort", "raw"],
)
def copd_ingestion():
    @task
    def start() -> None:
        """Explicit start marker for readability in the task graph."""

    @task
    def nhanes_ingest() -> list[str]:
        """Ensure the four NHANES 2011-2012 XPT files are present in the raw zone."""
        from airflow.sdk import get_current_context

        context = get_current_context()
        ds = context["ds"]
        nhanes_raw_dir = Path(RAW_ROOT) / "nhanes" / ds
        nhanes_raw_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []
        for source_name, cfg in NHANES_SOURCES.items():
            dest_path = nhanes_raw_dir / f"{source_name}.{cfg['ext']}"
            if not dest_path.exists():
                # Re-use the same download logic as ingest_source.
                session = requests.Session()
                for attempt in range(1, REQUEST_MAX_RETRIES + 1):
                    try:
                        resp = session.get(cfg["url"], timeout=REQUEST_TIMEOUT)
                        resp.raise_for_status()
                        break
                    except requests.RequestException as err:
                        if attempt == REQUEST_MAX_RETRIES:
                            raise
                dest_path.write_bytes(resp.content)
                print(f"[nhanes] downloaded {source_name}: {len(resp.content)} bytes")
            else:
                print(f"[nhanes] using existing {source_name}")
            downloaded.append(str(dest_path))
        return downloaded

    @task
    def build_cohort(nhanes_paths: list[str]) -> str:
        """Build the NHANES COPD cohort and return the cohort CSV path."""
        import build_nhanes_cohort as cohort_builder

        cohort_builder.RAW_DIR = Path(nhanes_paths[0]).parent
        cohort_builder.COHORT_DIR = Path(PREPROCESSED_ROOT)
        cohort = cohort_builder.build_cohort()
        cohort_path = cohort_builder.COHORT_DIR / "nhanes_copd_cohort.csv"
        cohort_builder.COHORT_DIR.mkdir(parents=True, exist_ok=True)
        cohort.to_csv(cohort_path, index=False)
        print(f"[cohort] built {len(cohort)} rows -> {cohort_path}")
        return str(cohort_path)

    @task
    def preprocessing(cohort_path: str) -> str:
        """Impute missing values, encode categoricals, and scale numerics."""
        from airflow.sdk import get_current_context
        import joblib
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        context = get_current_context()
        ds = context["ds"]
        preprocessed_dir = os.path.join(PREPROCESSED_ROOT, ds)
        os.makedirs(preprocessed_dir, exist_ok=True)

        cohort = pd.read_csv(cohort_path)

        # Targets are kept as-is in the output for the training DAG to consume.
        target_cols = ["copd_diagnosis", "gold_stage"]
        for col in target_cols:
            if col not in cohort.columns:
                raise ValueError(f"Target column '{col}' not found in NHANES cohort")

        # SEQN is the unique identifier and must not leak into features.
        leakage_cols = ["SEQN"]
        feature_df = cohort.drop(columns=leakage_cols + target_cols, errors="ignore")

        categorical_cols = [
            col
            for col in feature_df.columns
            if feature_df[col].dtype.name in ("category", "object")
        ]
        numeric_cols = [
            col
            for col in feature_df.columns
            if col not in categorical_cols
            and pd.api.types.is_numeric_dtype(feature_df[col])
        ]

        if not categorical_cols and not numeric_cols:
            raise ValueError("no usable feature columns found in NHANES cohort")

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    categorical_cols,
                ),
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        X_array = preprocessor.fit_transform(feature_df)
        X_columns = preprocessor.get_feature_names_out()
        # XGBoost rejects feature names containing brackets or angle brackets.
        X_columns_clean = [
            col.replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_")
            for col in X_columns
        ]
        X_df = pd.DataFrame(X_array, columns=X_columns_clean, index=feature_df.index)

        # Add the raw targets back so the training DAG can split them.
        preprocessed_df = pd.concat([X_df, cohort[target_cols].reset_index(drop=True)], axis=1)

        artifacts = {
            "preprocessor": preprocessor,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "dropped_leakage_columns": leakage_cols,
            "target_columns": target_cols,
            "source_path": cohort_path,
            "partition_ds": ds,
        }

        artifact_path = os.path.join(preprocessed_dir, "preprocessing_artifacts.joblib")
        joblib.dump(artifacts, artifact_path)

        summary_path = os.path.join(preprocessed_dir, "preprocessing_artifacts.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "artifact_path": artifact_path,
                    "numeric_columns": numeric_cols,
                    "categorical_columns": categorical_cols,
                    "dropped_leakage_columns": leakage_cols,
                    "target_columns": target_cols,
                    "partition_ds": ds,
                    "n_rows": len(preprocessed_df),
                    "n_feature_columns": len(X_columns_clean),
                },
                fh,
                indent=2,
            )

        central_dataset_path = os.path.join(preprocessed_dir, "central_preprocessed_dataset.csv")
        preprocessed_df.to_csv(central_dataset_path, index=False)

        return artifact_path

    @task
    def ingestion_complete(preprocessing_artifacts_path: str) -> None:
        """Final marker task for downstream dependencies."""
        print(f"[preprocess] artifacts saved at {preprocessing_artifacts_path}")

    start_task = start()
    nhanes_paths = nhanes_ingest()
    cohort_path = build_cohort(nhanes_paths)
    preprocessing_task = preprocessing(cohort_path)
    complete_task = ingestion_complete(preprocessing_task)

    start_task >> nhanes_paths >> cohort_path >> preprocessing_task >> complete_task


copd_dag = copd_ingestion()
