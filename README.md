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

## Handoff to colleagues

Downstream (preparation/transformation) reads from `data/raw/<source>/<date>/`.
The pipeline guarantees the file is present and unchanged from source; it makes
no guarantees about the file's internal schema or quality — that is their layer.
