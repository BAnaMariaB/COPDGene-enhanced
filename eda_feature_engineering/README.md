# EDA & Feature Engineering (staging area)

**Status: DRAFT**, but the target is now decided. This folder is intentionally
kept separate from `airflow/dags/` and `data/` until the group confirms where
it should live — see "Open questions" below before wiring any of this into the
real pipeline.

**Target decision (team chat, 2026-07-16 evening):** regression on
`fev1_phase2` hit an RMSE ceiling — the only strongly correlated feature is
baseline `fev1` itself, and a single predicted number isn't clinically useful
in production anyway. The team is switching to classification. The
recommended target is **`gold_copd`** (binary COPD classification via the
GOLD criterion) — it's built and ready in
`centralized_dataset_target_gold_copd.csv`.

**Not built yet, on purpose:** the team also discussed a second-stage cascade
(if COPD-positive, classify GOLD stage severity 1-4). Real GOLD staging needs
age/sex/height-adjusted % predicted FEV1 (reference equations), which is what
the NHANES data being integrated separately (elsewhere on the team) is for.
Building a competing version of that here risks duplicating/conflicting with
that work, so it's deliberately left as a documented future extension —
see `feature_engineering.GOLD_STAGE_CASCADE_NOTE`.

**Also not touched:** `airflow/dags/copd_ingestion.py`. Whoever owns
preprocessing is actively rewriting that same `preprocessing` task in
parallel — editing it here too would just create a second collision on the
same file, on top of the two overlaps already flagged in "Open questions".

## What's here

- `fetch_raw_data.py` — pulls the same three raw sources as
  `airflow/dags/copd_ingestion.py` into `eda_feature_engineering/raw/`,
  independent of a running Airflow instance, so this can be iterated on
  without Docker.
- `load_and_merge.py` — merges demographics + imaging + spirometry on `sid`,
  same outer-join strategy as the ingestion DAG's `preprocessing` task.
- `eda.py` — missingness report, per-column distribution plots (histogram +
  KDE — **no boxplots**, by request), a correlation heatmap, and stats on all
  three candidate targets (including GOLD-threshold class balance: ~38.8%
  positive).
- `feature_engineering.py` — engineers `pulse_pressure`, `pack_years`,
  `air_trapping_ratio`, one boolean flag per condition parsed out of the
  multi-label `respiratory` column, and the derived `gold_copd` classification
  target. Defines `TARGET_CANDIDATES` (with a `status` per candidate —
  recommended / rejected / reference-only) and `RECOMMENDED_TARGET`. Writes a
  manifest documenting every decision.
- `synthetic_data.py` — fits a KDE per numeric column (and an empirical
  frequency table per categorical column) on the real data, then samples new
  rows from those fitted distributions. Also plots real-vs-synthetic
  overlays so the shapes can be checked before the rows get used anywhere.
- `run_pipeline.py` — runs all of the above end to end; prints the
  recommended target and its output path clearly at the end.

## Candidate targets

| Target | Kind | Status | Description |
|---|---|---|---|
| `gold_copd` | classification | **recommended** | Derived: `fev1_fvc_ratio < 0.70` — the GOLD diagnostic criterion for airflow obstruction. ~38.8% of rows fall below threshold. |
| `fev1_phase2` | regression | rejected | FEV1 measured **five years after baseline** (per the spirometry data dictionary — a longitudinal follow-up value, *not* a repeat of the baseline test). Hits an RMSE ceiling: only `fev1` correlates strongly, and a raw number isn't production-useful. Kept for reference — it's the evidence for the pivot. |
| `fev1` | regression | reference only | Baseline FEV1 (current lung function). Not a target the team is pursuing. |

Each target has a different correct set of columns to exclude from features —
see `feature_engineering.TARGET_CANDIDATES` for the exact list and reasoning
per target. The exclusion rules aren't symmetric (e.g. `fev1_phase2` must be
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
  definitions + `recommended_target`
- `target_candidates_manifest.json` — per-target dropped columns + reasoning
- `centralized_dataset_full_real.csv` — every engineered column, nothing
  target-specific dropped (all three candidate target columns present).
- `centralized_dataset_full_with_synthetic.csv` — same, with synthetic rows
  appended and flagged via `is_synthetic`.
- **`centralized_dataset_target_gold_copd.csv` — train on this one.** Binary
  target, `NaN` where `fev1_fvc_ratio` was missing (10 rows).
- `centralized_dataset_target_fev1_phase2.csv` / `..._target_fev1.csv` —
  rejected/reference candidates only, kept for the presentation narrative,
  not for the real model.

## Leakage note (read before modeling)

- **Target = `gold_copd` (the one to use)**: drop `fev1_fvc_ratio` (it's
  literally the thresholded source of the label), `fev1` (kept alongside
  `fvc` it would let the ratio — and therefore the label — be reconstructed
  exactly), and `fev1_phase2` (future value, not available at prediction
  time). `fvc` alone is kept as the spirometry input.
- **Target = `fev1_phase2`** (reference only): no drops needed — `fev1`,
  `fvc`, `fev1_fvc_ratio`, and `gold_copd` are all baseline values known
  before the 5-year follow-up.
- **Target = `fev1`** (reference only): drop `fev1_fvc_ratio`, `fev1_phase2`,
  and `gold_copd` — all leak `fev1` directly or one step removed.

`respiratory_is_copd` is worth a second look for the `gold_copd` target
specifically — a COPD diagnosis is clinically defined partly by an FEV1/FVC
threshold, so it's correlated with the label by definition, just not an exact
algebraic leak.

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
   handling. **Whoever owns that task is mid-rewrite of it right now**
   (integrating NHANES data), so this folder deliberately hasn't touched it —
   coordinate naming before merging (e.g. if their rewrite also derives a
   binary COPD flag, make sure it isn't a second column meaning the same
   thing as `gold_copd` under a different name).
2. Where should encoding/scaling of the categorical columns happen — here, or
   in the existing `ColumnTransformer` in the ingestion DAG?
3. The shared `preprocessing` task drops `respiratory` entirely; this folder
   parses it into per-condition flags instead. Worth reconciling if/when this
   gets merged in.
4. The GOLD-stage-severity cascade (see top of this file) needs the
   NHANES-based reference values from whoever is building that — not
   duplicated here.

Until these are answered, nothing in this folder is imported by
`airflow/dags/*`.
