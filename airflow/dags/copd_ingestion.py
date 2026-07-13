"""
COPD data ingestion DAG.

SCOPE: ingestion only. This DAG pulls three raw source files exactly as they are
published and lands them, unmodified, into a date-partitioned raw zone. It does
NOT parse, clean, validate schemas, or join the sources — that is downstream work
owned by other team members.

Sources (all keyed by `sid`):
  - demographics : CSV
  - imaging      : JSON
  - spirometry   : HTML (contains an HTML <table>)

Landing layout (one partition per DAG run date):
  data/raw/<source>/<YYYY-MM-DD>/<source>.<ext>
  data/raw/<source>/<YYYY-MM-DD>/<source>.<ext>.meta.json   (ingestion metadata sidecar)

The three source tasks run in parallel; a final `ingestion_complete` marker task
fans them back in so downstream DAGs can depend on a single point.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from datetime import datetime, timedelta, timezone

import requests
from airflow.sdk import dag, task

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

# Each source: logical name -> (url, file extension)
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
    description="Ingest raw COPD sources (CSV/JSON/HTML) into a date-partitioned raw zone. Ingestion only.",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 * * * *",  # hourly, at the top of every hour
    catchup=False,
    max_active_runs=1,
    tags=["copd", "ingestion", "raw"],
)
def copd_ingestion():
    @task
    def start() -> None:
        """Explicit start marker for readability in the task graph."""

    @task
    def preprocessing(demographics_path: str, imaging_path: str, spirometry_path: str) -> str:
        """Impute missing values and fit reusable preprocessing artifacts."""
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

        demographics = pd.read_csv(demographics_path)
        imaging = pd.read_json(imaging_path)
        spirometry = pd.read_html(spirometry_path)[0]

        merged = demographics.merge(imaging, on="sid", how="outer", suffixes=("_demographics", "_imaging"))
        merged = merged.merge(spirometry, on="sid", how="outer", suffixes=("", "_spirometry"))

        if "sid" not in merged.columns:
            raise ValueError("merged dataset does not contain required sid column")

        lower_columns = {col.lower(): col for col in merged.columns}
        categorical_cols = [
            lower_columns[name]
            for name in ("gender", "race", "smoking_status")
            if name in lower_columns
        ]
        leakage_cols = [col for col in ("sid",) if col in merged.columns]
        numeric_cols = [
            col
            for col in merged.columns
            if col not in categorical_cols + leakage_cols
            and pd.api.types.is_numeric_dtype(merged[col])
        ]

        if not categorical_cols and not numeric_cols:
            raise ValueError("no usable feature columns found after applying the ColumnTransformer rules")

        feature_df = merged.drop(columns=leakage_cols, errors="ignore")

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

        preprocessed_array = preprocessor.fit_transform(feature_df)
        preprocessed_columns = preprocessor.get_feature_names_out()
        preprocessed_df = pd.DataFrame(preprocessed_array, columns=preprocessed_columns, index=feature_df.index)

        artifacts = {
            "preprocessor": preprocessor,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "dropped_leakage_columns": leakage_cols,
            "source_paths": {
                "demographics": demographics_path,
                "imaging": imaging_path,
                "spirometry": spirometry_path,
            },
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
                    "partition_ds": ds,
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
    demographics_task = ingest_source(
        source_name="demographics",
        url=SOURCES["demographics"]["url"],
        ext=SOURCES["demographics"]["ext"],
    )
    imaging_task = ingest_source(
        source_name="imaging",
        url=SOURCES["imaging"]["url"],
        ext=SOURCES["imaging"]["ext"],
    )
    spirometry_task = ingest_source(
        source_name="spirometry",
        url=SOURCES["spirometry"]["url"],
        ext=SOURCES["spirometry"]["ext"],
    )
    preprocessing_task = preprocessing(demographics_task, imaging_task, spirometry_task)
    complete_task = ingestion_complete(preprocessing_task)

    start_task >> [demographics_task, imaging_task, spirometry_task] >> preprocessing_task >> complete_task


copd_dag = copd_ingestion()
