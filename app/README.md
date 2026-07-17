# Web Application Layer

React frontend + FastAPI backend + PostgreSQL model registry. Serves the champion
classifier for the team's decided target.

Built against `eda_feature_engineering/` — currently the only place in the repo
with a decided, leakage-correct target.

```
app/
├── backend/
│   ├── features.py   ← serving feature contract + leakage boundary. Start here.
│   ├── inference.py  model loading, cache, contract validation, the cascade
│   ├── main.py       endpoints
│   ├── db.py         model_registry reads (one champion per target)
│   └── config.py     env-overridable settings
├── frontend/         React (Vite). Form renders from GET /model/schema.
└── db/init.sql       model_registry table
```

## Run it

```bash
# 1. App stack (Postgres + MLflow + backend + frontend)
docker compose -f docker-compose.app.yaml up -d

# frontend  http://localhost:5173
# backend   http://localhost:8000/docs
# mlflow    http://localhost:5000
```

The app will start and the form will render, but `/predict` returns **503** until
a champion and a serving preprocessor exist — see "What's still needed" below.
That's intended: it fails loudly with a message naming what's missing.

Without Docker:

```bash
cd app/backend && pip install -r requirements.txt && uvicorn main:app --reload
cd app/frontend && npm install && npm run dev     # proxies /api -> :8000
```

Tests:

```bash
pip install -r app/backend/requirements.txt
pip install pytest httpx
pytest tests/test_web_app.py
```

The suite imports `eda_feature_engineering/feature_engineering.py` directly and
asserts this app's leakage boundary and derived-feature formulas match it. Change
that module without changing `features.py` and these fail on purpose.

## The target

```
gold_copd := fev1_fvc_ratio < 0.70        binary, ~38.9% positive
```

matching `feature_engineering.RECOMMENDED_TARGET`.

Severity staging (`gold_stage`) is wired in the backend but **not built anywhere
in the repo** — see `feature_engineering.GOLD_STAGE_CASCADE_NOTE`. Without a
registered `gold_stage` champion, screening works and the readout says staging
isn't available. When staging lands it will be conditional: trained on
COPD-positive rows only, so it must never be asked about a screen-negative
subject. `inference.predict()` already enforces that.

### Target naming is unsettled

`CHANGES_MADE.md` calls this same quantity `copd_diagnosis`;
`eda_feature_engineering` calls it `gold_copd`. Both are `fev1_fvc_ratio < 0.70`.
That module's open question #1 predicted this exact collision. The backend
defaults to `gold_copd` and is env-overridable:

```bash
COPD_DIAGNOSIS_TARGET=copd_diagnosis   # if the team settles on the other name
COPD_GOLD_STAGE_TARGET=gold_stage
```

## The leakage boundary

Taken verbatim from `feature_engineering.TARGET_CANDIDATES["gold_copd"]`:

| Excluded | Why |
|---|---|
| `fev1_fvc_ratio` | the exact value the label is thresholded from |
| `fev1` | `fev1` + `fvc` reconstruct the ratio exactly |
| `fev1_phase2` | measured 5 years after baseline; unavailable at prediction time |

`fvc` is kept — their documented decision, and algebraically sound: `fvc` alone
cannot recover the ratio.

`validate_preprocessor()` refuses to load a preprocessor fitted on any excluded
column and names the offenders.
`tests/test_web_app.py::test_leakage_boundary_matches_eda_module` pins agreement.

> **The ingestion DAG's `preprocessing` task cannot serve this.** It leaves all
> four spirometry columns in as undifferentiated features with no target
> handling — `eda_feature_engineering/README.md` open question #1 says so too. So
> `preprocessing_artifacts.joblib` is not the serving preprocessor.

## The serving feature set

Form fields (`FEATURE_SCHEMA` in `features.py`):

| Group | Fields |
|---|---|
| Demographics | `visit_age`, `gender`, `race`, `height_cm`, `weight_kg`, `visit_year` |
| Smoking | `smoking_status`, `smoke_start_age`, `cigs_per_day_avg`, `duration_smoking` |
| Vitals | `blood_pressure_systolic`, `blood_pressure_diastolic`, `heart_rate`, `hours_on_oxygen` |
| CT imaging | `emphysema_percentage`, `gas_trapping_percentage`, `lung_volume_inspiratory`, `lung_volume_expiratory`, `mean_density_inspiratory`, `mean_density_expiratory` |
| Spirometry | `fvc` |
| Respiratory history | `respiratory_conditions` (multi-select of 8) |

Derived server-side, mirroring `engineer_features()`:

| Derived | Formula |
|---|---|
| `bmi` | `weight_kg / (height_cm/100)²` |
| `pulse_pressure` | `systolic − diastolic` |
| `pack_years` | `(cigs_per_day_avg / 20) × duration_smoking` |
| `air_trapping_ratio` | `lung_volume_expiratory / lung_volume_inspiratory` |
| `respiratory_reported` | any condition selected |
| `respiratory_is_*` | one boolean per condition (8 of them) |

`bmi` is a source column, but it equals the derived value to within 0.005 across
all 2620 rows, so the form computes it rather than asking twice.
`respiratory_conditions` is collected but not passed to the model — it only feeds
the flags. Never features, for any target: `sid`, `visit_date`, `respiratory`.

### Values are coded, and the codes are undocumented

Verified against the live source files:

| Column | Values | Counts |
|---|---|---|
| `gender` | `1` / `2` | 1333 / 1287 |
| `race` | `1` / `2` | 1887 / 733 |
| `smoking_status` | `1` / `2` | 1411 / 1209 — current vs former; no never-smokers |

`docs/data_overview.md` says `smoking_status` is "Current vs Former" but not which
code is which, and says nothing about `gender` or `race`. The form's
`option_labels` use the standard COPDGene convention as a **provisional guess**.

**Confirm these against the data dictionary before the demo.** A flipped label
means the form collects the opposite of what the model was trained on, and
nothing errors — the app just returns quietly wrong answers.

## Two things the team should decide

**1. FVC in a screening form.** `fvc` is a legitimate feature for training — no
argument there. But the app asks a user for it, and FVC comes from a spirometry
test that also yields FEV1. Anyone who can fill in this form could compute
FEV1/FVC directly and wouldn't need the model. Either drop `fvc` from the feature
set (making it a genuine pre-spirometry screen, at some cost in accuracy), or
reframe the app as something other than screening. Right now the form asks for it
because the model is trained on it.

**2. `respiratory_is_copd` as a feature.** `feature_engineering`'s own manifest
flags it: *"review with the team before using it as a feature for the gold_copd
target."* It's a self-reported COPD diagnosis being used to predict COPD. The form
currently offers it, because the dataset has it. If the team keeps it, expect a
question about it in the presentation.

## What's still needed

`/predict` returns 503 until all three exist. None do yet.

### 1. A row in `model_registry`

Schema in `db/init.sql`. `is_champion` is scoped per target. Promotion is one
transaction:

```sql
BEGIN;
UPDATE model_registry SET is_champion = FALSE
  WHERE is_champion AND target_name = 'gold_copd';
UPDATE model_registry SET is_champion = TRUE
  WHERE mlflow_run_id = :run_id AND target_name = 'gold_copd';
COMMIT;
```

Nothing in the repo writes this table. `copd_train_validate_test.py`'s
`select_champion` task is still an empty `pass`. Whoever fills it owns this.

`class_labels` must be a JSON array whose **order matches the column order of
`predict_proba`** — e.g. `["no_copd", "copd"]`. The readout labels its bars from
this array; wrong order means the app confidently shows the wrong class.

### 2. An MLflow pyfunc model at `artifact_uri`

Ideally wrapped so `predict()` returns probabilities `(n_rows, n_classes)`. Hard
labels work, but then the readout has no distribution to show.

### 3. `serving_preprocessor.joblib`

In `data/preprocessed/<ds>/`, fitted on `features.EXPECTED_MODEL_COLUMNS` and
nothing else, on a named DataFrame (so `feature_names_in_` is populated).

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | DB status, which models are loaded |
| GET | `/model/champions` | metadata per target; `503` if no diagnosis champion |
| GET | `/model/schema` | form fields; **always 200**, works before any model exists |
| POST | `/predict` | `{"features": {...}}` → screen, then staging if positive |
| POST | `/predict/batch` | multipart CSV; per-row errors don't fail the run |

`/predict` response:

```json
{
  "diagnosis": {
    "prediction": "copd",
    "probabilities": {"no_copd": 0.2, "copd": 0.8},
    "model_name": "xgb_gold_copd",
    "mlflow_run_id": "..."
  },
  "gold_stage": null,
  "staging_skipped_reason": "no_gold_stage_champion"
}
```

Validation errors return `422` with `detail.field_errors` keyed by field name; the
form renders them inline. Spirometry leak columns are rejected as unknown fields.
Batch CSVs may carry `respiratory_conditions` in the source's own pipe-delimited
form (`asthma|pneumonia`).

## Notes for whoever dockerizes the rest

- `docker-compose.yaml` mounts `./dags:/opt/airflow/dags`, but the DAGs live in
  `./airflow/dags`. As written, Airflow boots with an empty DAG folder.
- `COPD_PREPROCESSED_ROOT` isn't set there, so preprocessing output lands under
  `AIRFLOW_HOME` inside the container rather than `./data`, where this app reads it.
- The app stack uses its own Postgres on host port **5433**; 5432 is the Airflow
  metadata DB.
