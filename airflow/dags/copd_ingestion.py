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
RAW_ROOT = os.environ.get("COPD_RAW_ROOT", "/opt/airflow/data/raw")

# HTTP settings
REQUEST_TIMEOUT = 60          # seconds
REQUEST_MAX_RETRIES = 3       # handled by Airflow task retries too; this is per-call

# Preprocessing artifacts are stored separately from raw landing data.
ARTIFACT_ROOT = os.environ.get("COPD_ARTIFACT_ROOT", "/opt/airflow/data/artifacts")
PREPROCESSED_ROOT = os.environ.get("COPD_PREPROCESSED_ROOT", "/opt/airflow/data/preprocessed")

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

def ingest_source(source_name: str, url: str, ext: str, **context) -> str:
    """Download one raw source and land it, byte-for-byte, in the raw zone.

    Returns the path of the landed file (also pushed to XCom automatically).
    Pure ingestion: no decoding, parsing, or transformation is performed.
    """
    # Partition by the DAG run's logical date (data interval start).
    ds = context["ds"]  # 'YYYY-MM-DD'
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
        "dag_run_id": context["run_id"],
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
    def ingest_demographics() -> str:
        return ingest_source(
            source_name="demographics",
            url=SOURCES["demographics"]["url"],
            ext=SOURCES["demographics"]["ext"],
        )

    @task
    def ingest_imaging() -> str:
        return ingest_source(
            source_name="imaging",
            url=SOURCES["imaging"]["url"],
            ext=SOURCES["imaging"]["ext"],
        )

    @task
    def ingest_spirometry() -> str:
        return ingest_source(
            source_name="spirometry",
            url=SOURCES["spirometry"]["url"],
            ext=SOURCES["spirometry"]["ext"],
        )

    @task
    def preprocessing(demographics_path: str, imaging_path: str, spirometry_path: str) -> str:
        """Impute missing values and fit reusable preprocessing artifacts."""
        from airflow.sdk import get_current_context
        import joblib
        import pandas as pd
        from sklearn.impute import SimpleImputer
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

        feature_df = merged.drop(columns=["sid"])
        numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [col for col in feature_df.columns if col not in numeric_cols]

        numeric_imputer = SimpleImputer(strategy="median")
        categorical_imputer = SimpleImputer(strategy="most_frequent")

        numeric_data = numeric_imputer.fit_transform(feature_df[numeric_cols]) if numeric_cols else None
        categorical_data = (
            categorical_imputer.fit_transform(feature_df[categorical_cols]) if categorical_cols else None
        )

        scaler = StandardScaler()
        if numeric_cols:
            scaler.fit(numeric_data)

        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if categorical_cols:
            one_hot_encoder.fit(categorical_data)

        artifacts = {
            "numeric_imputer": numeric_imputer,
            "categorical_imputer": categorical_imputer,
            "scaler": scaler,
            "one_hot_encoder": one_hot_encoder,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
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
                    "partition_ds": ds,
                },
                fh,
                indent=2,
            )

        central_dataset_path = os.path.join(preprocessed_dir, "central_preprocessed_dataset.csv")
        merged.to_csv(central_dataset_path, index=False)

        return artifact_path

    @task
    def ingestion_complete(preprocessing_artifacts_path: str) -> None:
        """Final marker task for downstream dependencies."""
        print(f"[preprocess] artifacts saved at {preprocessing_artifacts_path}")

    start_task = start()
    demographics_task = ingest_demographics()
    imaging_task = ingest_imaging()
    spirometry_task = ingest_spirometry()
    preprocessing_task = preprocessing(demographics_task, imaging_task, spirometry_task)
    complete_task = ingestion_complete(preprocessing_task)

    start_task >> [demographics_task, imaging_task, spirometry_task] >> preprocessing_task >> complete_task


dag = copd_ingestion()
