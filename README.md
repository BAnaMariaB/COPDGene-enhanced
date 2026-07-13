# COPD Ingestion Pipeline (Airflow)

Ingestion-only Airflow pipeline for the COPD project. It pulls three raw source
files and lands them, **byte-for-byte unchanged**, into a date-partitioned raw
zone. No parsing, cleaning, schema validation, or joining happens here — that is
downstream work owned by other team members.

## Scope

| Source | Format | URL |
|---|---|---|
| demographics | CSV | `.../copd_data_demographics.csv` |
| imaging | JSON | `.../copd_data_imaging.json` |
| spirometry | HTML | `.../copd_data_spirometry.html` |

All three share a `sid` key, but this pipeline does **not** touch the contents —
it only downloads and lands them. One DAG, `copd_ingestion`, runs the three
downloads in parallel.

## Landing layout

Each run writes to a partition named after the run's logical date:

```
data/raw/
├── demographics/<YYYY-MM-DD>/demographics.csv
│                              demographics.csv.meta.json
├── imaging/<YYYY-MM-DD>/imaging.json
│                        imaging.json.meta.json
└── spirometry/<YYYY-MM-DD>/spirometry.html
                            spirometry.html.meta.json
```

The `.meta.json` sidecar next to each file records ingestion provenance only
(source URL, HTTP status, byte count, SHA-256, timestamp, host, run id). It never
modifies the payload. Downstream tasks can read the raw file and ignore or use
the sidecar as they wish.

## Run it locally (Docker)

Requirements: Docker + Docker Compose.

```bash
cd copd-ingestion

# 1. (Linux/macOS) make Airflow write files as your user:
echo "AIRFLOW_UID=$(id -u)" > .env
echo "_PIP_ADDITIONAL_REQUIREMENTS=requests==2.32.3" >> .env

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
