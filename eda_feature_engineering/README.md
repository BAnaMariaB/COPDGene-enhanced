# EDA & Feature Engineering (staging area)

**Status: DRAFT.** This folder is intentionally kept separate from
`airflow/dags/` and `data/` until the group confirms where it should live —
see "Open questions" below before wiring any of this into the real pipeline.

**The team has not finalized a prediction target yet.** Rather than block on
that, this pipeline produces a leakage-correct, ready-to-train CSV for each of
the three live candidates (see `DATA_PREPROCESSING.md` at the repo root and
"Candidate targets" below) — whichever one gets picked is already sitting in
`output/`.

## What's here

- `fetch_raw_data.py` — pulls the same three raw sources as
  `airflow/dags/copd_ingestion.py` into `eda_feature_engineering/raw/`,
  independent of a running Airflow instance, so this can be iterated on
  without Docker.
- `load_and_merge.py` — merges demographics + imaging + spirometry on `sid`,
  same outer-join strategy as the ingestion DAG's `preprocessing` task.
- `eda.py` — missingness report, per-column distribution plots (histogram +
  KDE — **no boxplots**, by request), a correlation heatmap, and stats on all
  three candidate targets (including GOLD-threshold class balance).
- `feature_engineering.py` — engineers `pulse_pressure`, `pack_years`,
  `air_trapping_ratio`, one boolean flag per condition parsed out of the
  multi-label `respiratory` column, and the derived `gold_copd` classification
  target. Defines `TARGET_CANDIDATES`: for each candidate target, exactly
  which other columns must be dropped from the feature set and why. Writes a
  manifest documenting every decision.
- `synthetic_data.py` — fits a KDE per numeric column (and an empirical
  frequency table per categorical column) on the real data, then samples new
  rows from those fitted distributions. Also plots real-vs-synthetic
  overlays so the shapes can be checked before the rows get used anywhere.
- `run_pipeline.py` — runs all of the above end to end.

## Candidate targets

| Target | Kind | Description |
|---|---|---|
| `fev1` | regression | Baseline FEV1 (current lung function). |
| `fev1_phase2` | regression | FEV1 measured **five years after baseline** (per the spirometry data dictionary — a longitudinal follow-up value, *not* a repeat of the baseline test). Predicts lung-function decline. |
| `gold_copd` | classification | Derived: `fev1_fvc_ratio < 0.70` — the GOLD diagnostic criterion for airflow obstruction. ~38.8% of rows fall below threshold. |

Each has a different correct set of columns to exclude from features — see
`feature_engineering.TARGET_CANDIDATES` for the exact list and reasoning per
target. The exclusion rules aren't symmetric (e.g. `fev1_phase2` must be
dropped as a feature when predicting `fev1`, but `fev1` is a perfectly valid
feature when predicting `fev1_phase2`), so use `build_dataset_for_target()`
rather than hand-picking columns to drop yourself.

## Run it

```bash
pip install -r eda_feature_engineering/requirements.txt
python eda_feature_engineering/run_pipeline.py
```

Output lands in `eda_feature_engineering/output/`:

- `plots/distributions/*.png`, `plots/correlation_heatmap.png`,
  `plots/synthetic_vs_real/*.png`
- `feature_engineering_manifest.json` — engineered columns + target candidate
  definitions
- `target_candidates_manifest.json` — per-target dropped columns + reasoning
- `centralized_dataset_full_real.csv` — every engineered column, nothing
  target-specific dropped (all three candidate target columns present).
- `centralized_dataset_full_with_synthetic.csv` — same, with synthetic rows
  appended and flagged via `is_synthetic`.
- `centralized_dataset_target_fev1.csv` — ready to train on, target = `fev1`.
- `centralized_dataset_target_fev1_phase2.csv` — ready to train on, target =
  `fev1_phase2`.
- `centralized_dataset_target_gold_copd.csv` — ready to train on, target =
  `gold_copd` (binary, `NaN` where `fev1_fvc_ratio` was missing).

**Once the team picks a target, use the matching `centralized_dataset_target_*.csv` file directly — don't hand-drop columns from the full dataset.**

## Leakage note (read before modeling)

- **Target = `fev1`**: drop `fev1_fvc_ratio` (reconstructs `fev1` via `fvc`
  almost exactly), `fev1_phase2` (not available at prediction time — it's a
  future value, not a duplicate), and `gold_copd` (derived from the same ratio
  that leaks `fev1`).
- **Target = `fev1_phase2`**: no drops needed. `fev1`, `fvc`,
  `fev1_fvc_ratio`, and `gold_copd` are all baseline values known before the
  5-year follow-up — legitimate predictors, not leakage.
- **Target = `gold_copd`**: drop `fev1_fvc_ratio` (it's literally the
  thresholded source of the label), `fev1` (kept alongside `fvc` it would let
  the ratio — and therefore the label — be reconstructed exactly), and
  `fev1_phase2` (future value). `fvc` alone is kept.

`respiratory_is_copd` is worth a second look regardless of target — a COPD
diagnosis is clinically defined partly by an FEV1/FVC threshold, so it's
correlated with `gold_copd` by definition, just not an exact algebraic leak.

`sid` is kept in every output CSV for traceability but is never a feature —
drop it before training regardless of target.

## Synthetic data — one real limitation

Synthetic rows are sampled **per column independently**. Real cross-column
relationships (age vs. FEV1, height vs. weight, etc.) are NOT preserved.
Treat synthetic rows as useful for padding volume or stress-testing code, not
as a substitute for real physiological correlation structure. They should
never land in a validation/test split used to score a real model. (The
target-ready CSVs are real-only; synthetic rows only exist in the `_full_`
files.)

## Open questions for the group (unresolved as of writing)

1. There's already a generic `preprocessing` task inside
   `airflow/dags/copd_ingestion.py` (impute + one-hot encode + scale, now with
   schema validation per the latest `dev` update) that also writes a
   `central_preprocessed_dataset.csv`. It currently leaves all four
   spirometry columns in as undifferentiated features with no target/leakage
   handling — that decision is deferred to whoever trains a model. Does this
   folder's logic get merged into that task, become its own DAG step, or stay
   separate?
2. Which of the three `centralized_dataset_target_*.csv` files does the
   Modeling DAG actually train on? (Depends on the team's target decision.)
3. Where should encoding/scaling of the categorical columns happen — here, or
   in the existing `ColumnTransformer` in the ingestion DAG?
4. The shared `preprocessing` task drops `respiratory` entirely; this folder
   parses it into per-condition flags instead. Worth reconciling if/when this
   gets merged in.

Until these are answered, nothing in this folder is imported by
`airflow/dags/*`.
