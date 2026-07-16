# EDA & Feature Engineering (staging area)

**Status: DRAFT.** This folder is intentionally kept separate from
`airflow/dags/` and `data/` until the group confirms where it should live —
see "Open questions" below before wiring any of this into the real pipeline.

## What's here

- `fetch_raw_data.py` — pulls the same three raw sources as
  `airflow/dags/copd_ingestion.py` into `eda_feature_engineering/raw/`, independent of a
  running Airflow instance, so this can be iterated on without Docker.
- `load_and_merge.py` — merges demographics + imaging + spirometry on `sid`,
  same outer-join strategy as the ingestion DAG's `preprocessing` task.
- `eda.py` — missingness report, per-column distribution plots (histogram +
  KDE — **no boxplots**, by request), a correlation heatmap, and a printed
  flag on which columns leak the FEV1 target.
- `feature_engineering.py` — target = `fev1`. Drops `fev1_fvc_ratio` and
  `fev1_phase2` (see leakage note below), engineers `pulse_pressure`,
  `pack_years`, `air_trapping_ratio`, and one boolean flag per condition
  parsed out of the multi-label `respiratory` column. Writes a manifest
  documenting every decision.
- `synthetic_data.py` — fits a KDE per numeric column (and an empirical
  frequency table per categorical column) on the real data, then samples new
  rows from those fitted distributions. Also plots real-vs-synthetic
  overlays so the shapes can be checked before the rows get used anywhere.
- `run_pipeline.py` — runs all of the above end to end.

## Run it

```bash
pip install -r eda_feature_engineering/requirements.txt
python eda_feature_engineering/run_pipeline.py
```

Output lands in `eda_feature_engineering/output/`:

- `plots/distributions/*.png`, `plots/correlation_heatmap.png`,
  `plots/synthetic_vs_real/*.png`
- `feature_engineering_manifest.json`
- `centralized_dataset_real.csv` — real rows only, feature-engineered,
  leakage columns dropped.
- `centralized_dataset_with_synthetic.csv` — same, with synthetic rows
  appended and flagged via `is_synthetic`.

## Leakage note (read before modeling)

`fev1_fvc_ratio` and `fvc` together reconstruct `fev1` almost exactly
(`fev1 = fev1_fvc_ratio * fvc`), and `fev1_phase2` looks like a repeat FEV1
measurement for the same participant. `fev1_fvc_ratio` and `fev1_phase2` are
dropped by default; `fvc` is kept as a feature but flag it with the team if
`fev1_fvc_ratio` ever gets reintroduced alongside it. `respiratory_is_copd`
is also worth a second look — a COPD diagnosis is clinically defined partly
by an FEV1/FVC threshold, so it's correlated with the target by definition,
just not an exact algebraic leak like the ratio.

## Synthetic data — one real limitation

Synthetic rows are sampled **per column independently**. Real cross-column
relationships (age vs. FEV1, height vs. weight, etc.) are NOT preserved.
Treat synthetic rows as useful for padding volume or stress-testing code, not
as a substitute for real physiological correlation structure. They should
never land in a validation/test split used to score a real model.

## Open questions for the group (unresolved as of writing)

1. There's already a generic `preprocessing` task inside
   `airflow/dags/copd_ingestion.py` (impute + one-hot encode + scale) that
   also writes a `central_preprocessed_dataset.csv`. This folder's output
   overlaps with that. Does this logic get merged into that task, become its
   own DAG step, or replace it?
2. Should the Modeling DAG train on
   `centralized_dataset_real.csv` or `centralized_dataset_with_synthetic.csv`?
3. Where should encoding/scaling of the categorical columns happen — here, or
   in the existing `ColumnTransformer` in the ingestion DAG?

Until these are answered, nothing in this folder is imported by
`airflow/dags/*`.
