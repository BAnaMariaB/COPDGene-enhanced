"""
COPD data ingestion DAG.

SCOPE: ingestion only. This DAG pulls three raw source files exactly as they are
published and lands them, unmodified, into a date-partitioned raw zone. It does
NOT parse, clean, validate schemas, or join the sources — that is downstream work
owned by other team members.

Core sources (all keyed by `sid`, merged downstream):
  - demographics : CSV  (static file download)
  - imaging      : JSON (static file download)
  - spirometry   : HTML (static file download; contains an HTML <table>)

Context sources (population-level, NOT keyed by `sid`, landed raw only):
  - cdc_copd_prevalence : JSON (live CDC Socrata SODA API)
  - smoking_prevalence  : HTML (web scrape of a Wikipedia article)

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
# Some sites (e.g. Wikipedia) reject the default python-requests User-Agent with
# HTTP 403, so we send a descriptive one. Being a good scraping citizen too.
USER_AGENT = "COPDGene-ingestion/0.1 (student project; contact: team)"

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

# Additional "context" sources added to satisfy the project's multi-source,
# multi-method ingestion requirement (live API + web scrape, from other websites).
#
# IMPORTANT: these are POPULATION-LEVEL (national / state / country) and are NOT
# keyed by `sid`. They are landed raw here for provenance, but are deliberately
# NOT merged into the sid-level dataset by the preprocessing step — there is no
# join key. Downstream may use them for coarse context features only.
#
#   - cdc_copd_prevalence : live REST API pull (CDC Socrata SODA), JSON
#   - smoking_prevalence  : web scrape of an HTML page (tables parsed downstream)
CONTEXT_SOURCES = {
    "cdc_copd_prevalence": {
        # CDC U.S. Chronic Disease Indicators, filtered to COPD (topicid=COPD).
        # SODA API returns JSON; $limit is set high enough to return all COPD rows.
        "url": "https://data.cdc.gov/resource/hksd-2xuw.json?topicid=COPD&$limit=50000",
        "ext": "json",
        "kind": "api",
    },
    "smoking_prevalence": {
        # Wikipedia article containing tobacco-use prevalence tables.
        "url": "https://en.wikipedia.org/wiki/Prevalence_of_tobacco_use",
        "ext": "html",
        "kind": "web_scrape",
    },
}


# ---------------------------------------------------------------------------
# Ingestion logic
# ---------------------------------------------------------------------------

@task
def ingest_source(
    source_name: str,
    url: str,
    ext: str,
    source_kind: str = "download",
    ds: str = None,
    run_id: str = None,
) -> str:
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
    session.headers.update({"User-Agent": USER_AGENT})
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
        "ingestion_kind": source_kind,
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
        """Merge and transform the three core sid-keyed sources only.

        Context sources (cdc_copd_prevalence, smoking_prevalence) are deliberately
        excluded because they are population-level and have no `sid` join key. The
        output is a clean, model-ready matrix plus reusable sklearn artifacts.

        Steps:
        1. Load the three core sources and validate their schemas.
        2. Outer-merge on `sid`.
        3. Drop leakage / non-feature columns (`sid`, `visit_date`, `respiratory`).
        4. One-hot encode categoricals (`gender`, `race`, `smoking_status`) and
           median-impute + standard-scale numeric columns.
        5. Persist the central preprocessed dataset, the fitted preprocessor, and
           a JSON preprocessing manifest.
        """
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

        # Expected schemas for the three core sid-keyed sources. These are the only
        # columns allowed in the final merged dataset; anything else is treated as
        # schema drift and rejected. This guards against context-source columns or
        # future upstream changes leaking into the sid-level model matrix.
        EXPECTED_COLUMNS = {
            "demographics": {
                "sid",
                "visit_year",
                "visit_date",
                "visit_age",
                "gender",
                "race",
                "smoking_status",
                "height_cm",
                "weight_kg",
                "blood_pressure_systolic",
                "blood_pressure_diastolic",
                "heart_rate",
                "hours_on_oxygen",
                "bmi",
                "smoke_start_age",
                "cigs_per_day_avg",
                "duration_smoking",
                "respiratory",
            },
            "imaging": {
                "sid",
                "lung_volume_inspiratory",
                "emphysema_percentage",
                "lung_volume_expiratory",
                "gas_trapping_percentage",
                "mean_density_inspiratory",
                "mean_density_expiratory",
            },
            "spirometry": {
                "sid",
                "fev1_fvc_ratio",
                "fev1",
                "fvc",
                "fev1_phase2",
            },
        }

        def _validate_source(name: str, df: pd.DataFrame) -> None:
            expected = EXPECTED_COLUMNS[name]
            actual = set(df.columns)
            missing = expected - actual
            extra = actual - expected
            if missing:
                raise ValueError(
                    f"{name} is missing required columns: {sorted(missing)}"
                )
            if extra:
                raise ValueError(
                    f"{name} has unexpected columns that are not present in the "
                    f"expected schema for the other core sources: {sorted(extra)}"
                )

        demographics = pd.read_csv(demographics_path)
        imaging = pd.read_json(imaging_path)
        spirometry = pd.read_html(spirometry_path)[0]

        _validate_source("demographics", demographics)
        _validate_source("imaging", imaging)
        _validate_source("spirometry", spirometry)

        merged = demographics.merge(
            imaging, on="sid", how="outer", suffixes=("_demographics", "_imaging")
        )
        merged = merged.merge(
            spirometry, on="sid", how="outer", suffixes=("", "_spirometry")
        )

        if "sid" not in merged.columns:
            raise ValueError("merged dataset does not contain required sid column")

        # Columns that are not model features. `sid` is a leakage key; `visit_date`
        # is a non-numeric string; `respiratory` is a pipe-delimited free-text field
        # that would need a dedicated text-encoding step. None of these are present
        # in imaging or spirometry, so they are explicitly excluded from the model
        # matrix to keep the dataset aligned with the columns that all three sources
        # can contribute.
        DROP_REASONS = {
            "sid": "subject identifier / leakage key",
            "visit_date": "non-informative string column, not present in other sources",
            "respiratory": "pipe-delimited free-text, not present in other sources",
        }
        dropped_feature_cols = [col for col in DROP_REASONS if col in merged.columns]
        feature_df = merged.drop(columns=dropped_feature_cols, errors="ignore")

        lower_columns = {col.lower(): col for col in feature_df.columns}
        categorical_cols = [
            lower_columns[name]
            for name in ("gender", "race", "smoking_status")
            if name in lower_columns
        ]
        numeric_cols = [
            col
            for col in feature_df.columns
            if col not in categorical_cols
            and pd.api.types.is_numeric_dtype(feature_df[col])
        ]

        if not categorical_cols and not numeric_cols:
            raise ValueError("no usable feature columns found after applying the ColumnTransformer rules")

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

        artifact_path = os.path.join(preprocessed_dir, "preprocessing_artifacts.joblib")
        summary_path = os.path.join(preprocessed_dir, "preprocessing_artifacts.json")
        manifest_path = os.path.join(preprocessed_dir, "preprocessing_manifest.json")
        central_dataset_path = os.path.join(preprocessed_dir, "central_preprocessed_dataset.csv")

        artifacts = {
            "preprocessor": preprocessor,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "dropped_columns": {
                "leakage": ["sid"],
                "non_feature": ["visit_date", "respiratory"],
                "all": dropped_feature_cols,
                "reasons": DROP_REASONS,
            },
            "source_paths": {
                "demographics": demographics_path,
                "imaging": imaging_path,
                "spirometry": spirometry_path,
            },
            "partition_ds": ds,
            "central_dataset_path": central_dataset_path,
            "manifest_path": manifest_path,
            "summary_path": summary_path,
        }
        joblib.dump(artifacts, artifact_path)

        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "artifact_path": artifact_path,
                    "numeric_columns": numeric_cols,
                    "categorical_columns": categorical_cols,
                    "dropped_columns": artifacts["dropped_columns"],
                    "partition_ds": ds,
                },
                fh,
                indent=2,
            )

        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "partition_ds": ds,
                    "source_paths": artifacts["source_paths"],
                    "expected_columns": {k: sorted(v) for k, v in EXPECTED_COLUMNS.items()},
                    "actual_columns": {
                        "demographics": sorted(demographics.columns.tolist()),
                        "imaging": sorted(imaging.columns.tolist()),
                        "spirometry": sorted(spirometry.columns.tolist()),
                    },
                    "context_sources_excluded": [
                        "cdc_copd_prevalence",
                        "smoking_prevalence",
                    ],
                    "context_sources_exclusion_reason": (
                        "population-level sources with no sid join key; "
                        "they are landed raw but not merged into the sid-level dataset"
                    ),
                    "merged_shape": merged.shape,
                    "dropped_columns": artifacts["dropped_columns"],
                    "feature_columns": {
                        "categorical": categorical_cols,
                        "numeric": numeric_cols,
                    },
                    "preprocessed_shape": preprocessed_df.shape,
                    "preprocessed_columns": preprocessed_columns.tolist(),
                    "output_files": {
                        "central_dataset": central_dataset_path,
                        "artifacts": artifact_path,
                        "summary": summary_path,
                    },
                },
                fh,
                indent=2,
            )

        preprocessed_df.to_csv(central_dataset_path, index=False)

        print(
            f"[preprocess] merged={merged.shape} -> features={feature_df.shape} "
            f"-> preprocessed={preprocessed_df.shape} -> {central_dataset_path}"
        )
        return artifact_path

    @task
    def ingestion_complete(preprocessing_artifacts_path: str) -> None:
        """Final marker task for downstream dependencies."""
        try:
            from airflow.sdk import get_current_context
            from serving.common.champion_registry import connect as db_connect
            from serving.common.champion_registry import upsert_pipeline_event

            registry_url = os.environ.get("CHAMPION_REGISTRY_DATABASE_URL", "").strip()
            if registry_url:
                context = get_current_context()
                with db_connect() as conn:
                    upsert_pipeline_event(
                        conn,
                        pipeline_name="copd_ingestion",
                        event_type="completion",
                        status="success",
                        run_id=str(context.get("run_id")),
                        logical_date=str(context.get("ds")),
                        details_json={"preprocessing_artifacts_path": preprocessing_artifacts_path},
                    )
        except Exception as exc:  # noqa: BLE001
            print(f"[ingestion_registry] skipped: {exc}")
        print(f"[preprocess] artifacts saved at {preprocessing_artifacts_path}")

    start_task = start()
    demographics_task = ingest_source(
        source_name="demographics",
        url=SOURCES["demographics"]["url"],
        ext=SOURCES["demographics"]["ext"],
        source_kind="static_file",
    )
    imaging_task = ingest_source(
        source_name="imaging",
        url=SOURCES["imaging"]["url"],
        ext=SOURCES["imaging"]["ext"],
        source_kind="static_file",
    )
    spirometry_task = ingest_source(
        source_name="spirometry",
        url=SOURCES["spirometry"]["url"],
        ext=SOURCES["spirometry"]["ext"],
        source_kind="static_file",
    )

    # Context sources: landed raw only, one task each (explicit task_id per source).
    # They are NOT passed into preprocessing because they have no `sid` join key.
    context_tasks = [
        ingest_source.override(task_id=f"ingest_{name}")(
            source_name=name,
            url=cfg["url"],
            ext=cfg["ext"],
            source_kind=cfg["kind"],
        )
        for name, cfg in CONTEXT_SOURCES.items()
    ]

    preprocessing_task = preprocessing(demographics_task, imaging_task, spirometry_task)
    complete_task = ingestion_complete(preprocessing_task)

    # Core path: 3 sid-keyed sources -> merge/preprocess -> complete.
    start_task >> [demographics_task, imaging_task, spirometry_task] >> preprocessing_task >> complete_task
    # Context path: parallel raw landings that also gate completion (no merge).
    start_task >> context_tasks
    for context_task in context_tasks:
        context_task >> complete_task


copd_dag = copd_ingestion()
