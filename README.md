# COPD Ingestion Pipeline (Airflow)

Ingestion-only Airflow pipeline for the COPD project. It pulls raw source data
from several websites, using several methods (static download, live REST API, and
web scrape), and lands each payload **byte-for-byte unchanged** into a
date-partitioned raw zone. No parsing, cleaning, schema validation, or joining
happens here — that is downstream work owned by other team members.

## Scope

**Core sources** — keyed by `sid`, merged downstream:

| Source | Format | Method | URL |
|---|---|---|---|
| demographics | CSV | static file | `.../copd_data_demographics.csv` |
| imaging | JSON | static file | `.../copd_data_imaging.json` |
| spirometry | HTML | static file | `.../copd_data_spirometry.html` |

**Context sources** — population-level, **not** keyed by `sid`, landed raw only
(added to meet the "multiple websites / API / scraping" requirement):

| Source | Format | Method | URL |
|---|---|---|---|
| cdc_copd_prevalence | JSON | live API (CDC Socrata SODA) | `data.cdc.gov/resource/hksd-2xuw.json?topicid=COPD` |
| smoking_prevalence | HTML | web scrape (Wikipedia) | `en.wikipedia.org/wiki/Prevalence_of_tobacco_use` |

The three core sources share a `sid` key and are merged by the preprocessing step.
The two context sources have **no `sid` join key**, so they are landed raw for
provenance and gate completion, but are deliberately **not** merged into the
sid-level dataset. This pipeline does not touch any payload's contents — it only
downloads and lands them. One DAG, `copd_ingestion`, runs all downloads in
parallel. See [`docs/data_overview.md`](docs/data_overview.md) for how each
source is best worked into the modeling data.

## Landing layout

Each run writes to a partition named after the run's logical date:

```
data/raw/
├── demographics/<YYYY-MM-DD>/demographics.csv (+ .meta.json)
├── imaging/<YYYY-MM-DD>/imaging.json (+ .meta.json)
├── spirometry/<YYYY-MM-DD>/spirometry.html (+ .meta.json)
├── cdc_copd_prevalence/<YYYY-MM-DD>/cdc_copd_prevalence.json (+ .meta.json)
└── smoking_prevalence/<YYYY-MM-DD>/smoking_prevalence.html (+ .meta.json)
```

The `.meta.json` sidecar next to each file records ingestion provenance only
(source URL, ingestion kind — `static_file` / `api` / `web_scrape` — HTTP status,
byte count, SHA-256, timestamp, host, run id). It never modifies the payload.
Downstream tasks can read the raw file and ignore or use the sidecar as they wish.

## Run it locally (Docker)

Requirements: Docker + Docker Compose.

```bash
cd copd-ingestion

# 1. (Linux/macOS) make Airflow write files as your user:
echo "AIRFLOW_UID=$(id -u)" > .env
echo "_PIP_ADDITIONAL_REQUIREMENTS=requests>=2.31,<3" >> .env

# 2. start the stack (first boot runs db init + creates the admin user)
docker compose up -d

# 3. open the UI
#    http://localhost:8080   login: airflow / airflow
```

In the UI, un-pause **`copd_ingestion`** and press ▶ **Trigger DAG**. After it
runs, the raw files appear under `./data/raw/<source>/<date>/` on your machine.

Stop everything with `docker compose down` (add `-v` to also wipe the metadata DB).

## Run it locally with your existing `~/airflow` install

Use the helper script to keep the pipeline outputs inside the project
directory while reusing the Airflow home you already have:

```bash
cd copd-ingestion
bash scripts/run_airflow_local.sh
```

That setup keeps Airflow metadata under `~/airflow/` and writes pipeline output
directly into the repo:

- `./data/raw/...`
- `./data/preprocessed/...`
- `./data/artifacts/...`

If you need to reset a broken local Airflow state, run:

```bash
bash scripts/reset_airflow_local.sh
```

That clears the Airflow home in `~/airflow/` and removes generated output under
`./data/` so the next run starts clean.

## Run the DAG in an existing Airflow

If you already have Airflow, just drop `dags/copd_ingestion.py` into your
`dags/` folder and ensure `requests` is installed. Set where files land with:

```bash
export COPD_RAW_ROOT=/your/raw/zone
export COPD_PREPROCESSED_ROOT=/your/preprocessed/zone
export COPD_ARTIFACT_ROOT=/your/artifacts/zone
```

## Design notes

- **Schedule** — runs **hourly** (`0 * * * *`). It ships **paused** by default
  (`DAGS_ARE_PAUSED_AT_CREATION=true`); un-pause it once in the UI and it then
  triggers itself every hour. `catchup=False` means no backfill of missed hours.
- **One DAG, three parallel ingest tasks** fan out from a `start` marker and
  fan back into preprocessing, then into an `ingestion_complete` marker so
  downstream DAGs can depend on a single task.
- **Idempotent per run date** — re-running a date overwrites that partition, so
  retries and backfills are safe.
- **Retries** — 3 task retries (Airflow) plus 3 per-request HTTP retries.
- **Raw fidelity** — files are written from `response.content` (raw bytes),
  never decoded or reformatted.

## Training / validation / testing DAG

The second DAG, `copd_train_validate_test`, consumes the preprocessed dataset
produced by `copd_ingestion` (`data/preprocessed/<ds>/central_preprocessed_dataset.csv`)
and trains a single **multi-class classifier** to predict the **FEV1 phase 2
severity class** (`fev1_phase2`). The numeric target is binned into three balanced
text classes using the training-set tertiles, then encoded with
`sklearn.preprocessing.LabelEncoder`:

- `low` — bottom third of FEV1 phase 2 values
- `medium` — middle third of FEV1 phase 2 values
- `high` — top third of FEV1 phase 2 values

### Modeling pipeline

1. `load_data` — reads the preprocessed CSV, derives the three balanced classes from the tertiles of `fev1_phase2`, encodes them with `LabelEncoder`, and splits into train / validation / test (70/15/15) with stratification.
2. `train_ensemble` — trains the single `COPDEnsembleClassifier` system, which fits all base classifiers on the same training data and then trains a LogisticRegression meta-learner on their validation-set class probabilities.
3. `evaluate_ensemble` — computes accuracy, precision, recall, `f1_macro`, and `roc_auc_ovr` on the held-out test set for the ensemble system.
4. `select_champion` — writes the champion record JSON for the ensemble system using `f1_macro` as the champion metric.
5. `complete` — final marker.

### MLflow tracking

Set these environment variables to control tracking:

- `MLFLOW_TRACKING_URI` — defaults to a local SQLite database (`sqlite:///.../mlflow.db`).
- `MLFLOW_EXPERIMENT_NAME` — defaults to `copd_fev1_phase2_classification`.
- `COPD_PREPROCESSED_ROOT` — defaults to `~/airflow/data/preprocessed`.
- `COPD_ARTIFACT_ROOT` — defaults to `~/airflow/data/artifacts`.
- `COPD_TARGET_COLUMN` — defaults to `fev1_phase2`.

To use S3 for artifact storage later, set the standard MLflow S3 variables before
starting Airflow (no code changes are required):

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MLFLOW_ARTIFACT_ROOT=s3://your-bucket/mlflow-artifacts
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=...
```

### Champion table (PostgreSQL, future work)

The DAG does **not** connect to PostgreSQL yet. It writes a champion record to
`data/artifacts/<ds>/champion.json` with the exact schema a future
`champion_models` table expects:

```json
{
  "model_name": "copd_ensemble",
  "mlflow_run_id": "...",
  "experiment_id": "...",
  "metric_name": "f1_macro",
  "metric_value": 0.828,
  "partition_ds": "2026-07-16",
  "artifact_uri": "...",
  "model_type": "COPDEnsembleClassifier",
  "params": { ... },
  "dag_run_id": "...",
  "registered_at": "..."
}
```

A future task owner can read this JSON and insert it into the PSQL
`champion_models` table using an Airflow connection.

## Handoff to colleagues

Downstream (preparation/transformation) reads from `data/raw/<source>/<date>/`.
The pipeline guarantees the file is present and unchanged from source; it makes
no guarantees about the file's internal schema or quality — that is their layer.
For the preprocessing step that merges the three core `sid`-keyed sources and
produces the model-ready dataset, see [`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md).
