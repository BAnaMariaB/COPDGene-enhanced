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
and trains and compares three candidate systems for two tasks:

- `copd_diagnosis` binary prediction
- `gold_stage` multiclass prediction

### Candidate systems

1. `ensemble`
   - `COPDDoubleTargetSystem`
   - CatBoost + XGBoost + LogisticRegression base learners for both targets
   - LogisticRegression meta-learner with out-of-fold stacking and calibrated tree-model probabilities
2. `lightgbm`
   - `COPDDoubleTargetSystem`
   - LightGBM for both targets
3. `best_of_both`
   - RandomForest for diagnosis
   - XGBoost for GOLD stage

### Modeling pipeline

1. `load_data` loads the preprocessed CSV and splits into train / validation / test.
2. `train_candidate_systems` fits the three variants.
3. `evaluate_candidate_systems` scores each candidate on the held-out test set.
4. `select_champion` chooses the best system by average macro F1 across both tasks.
5. `register_champion` writes the active champion into PostgreSQL and MLflow metadata.
6. `complete` finishes the DAG.

### MLflow and registry

Set these environment variables to control tracking and artifact loading:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`
- `COPD_PREPROCESSED_ROOT`
- `COPD_ARTIFACT_ROOT`
- `CHAMPION_REGISTRY_DATABASE_URL`
- `CHAMPION_MODEL_NAME`
- `CHAMPION_MODEL_TARGET`

For S3-backed MLflow artifacts, configure the standard MLflow/AWS variables before
starting Airflow:

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=...
```

The DAG writes the active champion into the `champion_models` PostgreSQL table
with the model artifact path, preprocessing artifact path, metric metadata, and
active flag needed by the serving layer.

## Handoff to colleagues

Downstream processing reads the preprocessed files from
`data/preprocessed/<ds>/central_preprocessed_dataset.csv`. The serving stack
consumes:

- the champion row from PostgreSQL
- the model bundle from MLflow
- the preprocessing bundle from MLflow

For the current deployment flow, use `docker-compose.serving.yaml` locally and
Elastic Beanstalk for production.

See [`DEPLOYMENT_LAB.md`](DEPLOYMENT_LAB.md) for the exact step-by-step deployment lab.
