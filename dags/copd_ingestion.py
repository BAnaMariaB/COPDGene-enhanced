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
from datetime import datetime, timezone

import requests
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root of the raw landing zone. Overridable via env var so the same DAG works
# in docker (default) and in any other Airflow environment.
RAW_ROOT = os.environ.get("COPD_RAW_ROOT", "/opt/airflow/data/raw")

# HTTP settings
REQUEST_TIMEOUT = 60          # seconds
REQUEST_MAX_RETRIES = 3       # handled by Airflow task retries too; this is per-call

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
    "retry_delay": __import__("datetime").timedelta(minutes=2),
}

with DAG(
    dag_id="copd_ingestion",
    description="Ingest raw COPD sources (CSV/JSON/HTML) into a date-partitioned raw zone. Ingestion only.",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 * * * *",  # hourly, at the top of every hour
    catchup=False,
    max_active_runs=1,
    tags=["copd", "ingestion", "raw"],
) as dag:

    start = EmptyOperator(task_id="start")
    ingestion_complete = EmptyOperator(task_id="ingestion_complete")

    for name, cfg in SOURCES.items():
        task = PythonOperator(
            task_id=f"ingest_{name}",
            python_callable=ingest_source,
            op_kwargs={
                "source_name": name,
                "url": cfg["url"],
                "ext": cfg["ext"],
            },
        )
        # Each source task runs in parallel between start and the completion marker.
        start >> task >> ingestion_complete
