# COPD Data Preprocessing

This document describes the preprocessing pipeline that turns the three raw,
subject-level (`sid`-keyed) sources into a single model-ready dataset. It lives in
the `preprocessing` task of the `copd_ingestion` DAG in
`airflow/dags/copd_ingestion.py`.

## Scope and design choices

The latest ingestion update added two **context** sources that are
population-level and have no `sid` join key:

- `cdc_copd_prevalence` — live CDC Socrata SODA API
- `smoking_prevalence` — Wikipedia HTML scrape

These are **intentionally excluded** from the preprocessing merge. They are
landed raw under `data/raw/<source>/<date>/` for provenance, but they cannot be
row-joined to anonymized subjects. The preprocessing step only uses the three core
`sid`-keyed sources.

## Inputs

| Source | Format | Role | Expected columns |
|---|---|---|---|
| `demographics` | CSV | Core subject table | `sid`, `visit_year`, `visit_date`, `visit_age`, `gender`, `race`, `smoking_status`, `height_cm`, `weight_kg`, `blood_pressure_systolic`, `blood_pressure_diastolic`, `heart_rate`, `hours_on_oxygen`, `bmi`, `smoke_start_age`, `cigs_per_day_avg`, `duration_smoking`, `respiratory` |
| `imaging` | JSON | CT-derived features | `sid`, `lung_volume_inspiratory`, `emphysema_percentage`, `lung_volume_expiratory`, `gas_trapping_percentage`, `mean_density_inspiratory`, `mean_density_expiratory` |
| `spirometry` | HTML table | Lung-function target | `sid`, `fev1_fvc_ratio`, `fev1`, `fvc`, `fev1_phase2` |

If any of the three files deviates from its expected schema, the task fails with
a clear `ValueError`. This prevents the two context sources (or future upstream
changes) from silently leaking columns into the subject-level dataset.

## Outputs

All outputs are written under `data/preprocessed/<date>/`:

| File | Description |
|---|---|
| `central_preprocessed_dataset.csv` | The final model-ready matrix (one row per subject) |
| `preprocessing_artifacts.joblib` | Fitted `sklearn` `ColumnTransformer` and column lists for reuse on new data |
| `preprocessing_artifacts.json` | Human-readable summary of the artifacts |
| `preprocessing_manifest.json` | Full preprocessing provenance: source paths, schemas, shapes, feature groups, dropped columns, and context-source exclusion notes |

## Preprocessing steps

### 1. Load and validate schemas

Each source is read into a pandas DataFrame and checked against its expected
column set. The check fails if any required column is missing or if any
unexpected column is present.

### 2. Outer-merge on `sid`

The three validated sources are merged with outer joins on `sid`, so every
subject present in at least one source is kept. This is a **sid-only merge**:
context sources are never merged here because they have no `sid` key.

### 3. Drop non-feature columns

The following columns are removed before model training:

| Column | Reason |
|---|---|
| `sid` | Subject identifier; using it as a feature would be leakage. |
| `visit_date` | Non-informative string column; not present in `imaging` or `spirometry`. |
| `respiratory` | Pipe-delimited free-text field (e.g. `asthma\|bronchitis attacks`) that would require a separate text-encoding step; not present in the other sources. |

These columns are recorded in the manifest under `dropped_columns`.

### 4. Feature type split

Remaining columns are split into two groups:

- **Categorical**: `gender`, `race`, `smoking_status` (identified by name, case-insensitive).
- **Numeric**: all other numeric columns (e.g. `visit_age`, `bmi`, `emphysema_percentage`, `fev1`, `fev1_phase2`, ...).

### 5. Impute and transform

A `sklearn.compose.ColumnTransformer` applies separate pipelines to each group:

- **Categorical**: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- **Numeric**: `SimpleImputer(strategy="median")` → `StandardScaler()`.

The result is a dense NumPy array with one-hot encoded categorical columns and
standardized numeric columns. Column names are reconstructed from the fitted
preprocessor.

### 6. Persist artifacts

The fitted `ColumnTransformer` is saved with `joblib`, and the transformed matrix
is written to `central_preprocessed_dataset.csv`. JSON sidecars record the
column lists, dropped columns, and preprocessing provenance.

## Target variables

The preprocessed dataset includes both potential modeling targets from the
spirometry source:

- `fev1_phase2` — continuous FEV1 five years later (regression target).
- `fev1_fvc_ratio` — can be thresholded at `< 0.70` (GOLD criterion) to create a
  classification target.

## What is deliberately not in the dataset

- **Context source columns**: `cdc_copd_prevalence` and `smoking_prevalence` are
  not `sid`-keyed, so they cannot be merged row-wise. They remain in `data/raw/`
  only.
- **Dropped columns**: `sid`, `visit_date`, `respiratory` are excluded as
  described above.
- **Any future column not in the expected schemas**: schema validation rejects
  it.

## How to inspect a run

After the DAG runs, look at the manifest for the shape and column decisions:

```bash
jq . data/preprocessed/<YYYY-MM-DD>/preprocessing_manifest.json
```

Then open `data/preprocessed/<YYYY-MM-DD>/central_preprocessed_dataset.csv` to
see the final model matrix.
