# Data overview

Reference for every data source the ingestion pipeline pulls, how each is
ingested, and how it should be worked into our modeling dataset. For the wider
list of *candidate* sources we evaluated (including ones we did not wire up), see
[`data_sources.md`](data_sources.md).

The `copd_ingestion` DAG lands each source **byte-for-byte unchanged** under
`data/raw/<source>/<YYYY-MM-DD>/`, next to a `.meta.json` provenance sidecar. The
sidecar records `ingestion_kind` (`static_file`, `api`, or `web_scrape`), the
source URL, HTTP status, byte count, SHA-256, and timestamp. Ingestion does no
parsing, cleaning, or joining — everything below under "how to work it in" is
downstream (preparation / feature) work, not part of the ingestion job.

## Summary

| Source | Format | Method | Join key | Role |
|---|---|---|---|---|
| demographics | CSV | static download | `sid` | core feature/target table |
| imaging | JSON | static download | `sid` | core features |
| spirometry | HTML table | static download | `sid` | core features + target (`fev1_phase2`) |
| cdc_copd_prevalence | JSON | live API (CDC SODA) | none | context only |
| smoking_prevalence | HTML | web scrape (Wikipedia) | none | context only |

## Core sources (keyed by `sid`)

These three come from the teaching repo, share the `sid` subject key, and are the
real basis for the model. The `preprocessing` task already merges them on `sid`,
imputes missing values, one-hot encodes categoricals, scales numerics, and writes
`central_preprocessed_dataset.csv`.

### demographics (CSV)
Subject-level demographics and history: age, gender, race, smoking status,
BMI/height/weight, cigarettes per day, and respiratory history flags.
- Ingested by: `ingest_source("demographics", ...)`.
- How to work it in: base table of the merge. `gender`, `race`, `smoking_status`
  are categoricals (note `smoking_status` has only *Current* vs *Former* smoker —
  there are no non-smokers, so smoker status is a poor prediction target).

### imaging (JSON)
CT-derived lung imaging measures: emphysema %, gas trapping %, inspiratory and
expiratory lung volumes, mean lung density.
- Ingested by: `ingest_source("imaging", ...)`.
- How to work it in: `pd.read_json` then merge on `sid`. Strong numeric predictors
  of lung function (emphysema % and lung volume especially). Several have ~10
  missing values — imputation handled in preprocessing.

### spirometry (HTML table)
Lung-function measurements, including the modeling target. Contains `fev1`, `fvc`,
`fev1_fvc_ratio`, and `fev1_phase2` (FEV1 five years later).
- Ingested by: `ingest_source("spirometry", ...)`; the HTML holds one `<table>`.
- How to work it in: `pd.read_html(path)[0]` then merge on `sid`.
- **Target options:** predict `fev1_phase2` directly (regression), or derive a
  COPD-status label from `fev1_fvc_ratio < 0.70` (GOLD criterion) for
  classification. Confirm the exact target definition with the instructor.

## Context sources (not keyed by `sid`)

These were added to meet the "multiple websites / API / scraping" requirement.
They are **population-level** (national / state / country), so they **cannot be
row-joined** to our anonymized subjects — there is no shared key and no location
on our subjects. Treat them as optional context, or simply as evidence of
multi-source, multi-method ingestion. They are landed raw and gate DAG
completion, but are intentionally excluded from the `sid` merge.

### cdc_copd_prevalence (JSON, live API)
CDC U.S. Chronic Disease Indicators filtered to COPD (`topicid=COPD`), pulled live
from the Socrata SODA API. Fields include state, year, question, data value (e.g.
age-adjusted COPD prevalence %), and stratification (sex, race).
- Ingested by: `ingest_source("cdc_copd_prevalence", ...)`, `kind="api"`.
- How to work it in: mainly a reference/context layer. You *could* attach a
  national or per-stratum COPD prevalence as a constant context feature, but it
  adds little predictive signal (same value for every subject). Best use: talking
  point / descriptive context in the write-up, and proof of a real API source.

### smoking_prevalence (HTML, web scrape)
Wikipedia article with tobacco-use prevalence tables by country.
- Ingested by: `ingest_source("smoking_prevalence", ...)`, `kind="web_scrape"`.
- How to work it in: parse with `pd.read_html` downstream if wanted. Same caveat:
  country-level, not joinable to subjects. Primary value is satisfying the
  scraping requirement from a different website.

## Recommended enrichment (not ingested)

### GLI-2012 spirometry reference equations
This is the highest-value addition for modeling, but it is a **computation, not a
downloadable dataset**, so it lives downstream (preparation), not in the ingestion
DAG. Using each subject's own age, sex, height, and ethnicity, compute predicted
FEV1/FVC, then derive **FEV1 % predicted**, a **z-score**, and the **lower limit
of normal**. These are true per-subject features (joinable by construction) and
give a clean, standard way to define the COPD-classification target. Implement
from the published GLI-2012 LMS coefficients or a package (R `rspiro`, Python
ports). Reference calculator: https://pft-calculator.com/

### Optional, if time allows
- **NHANES** spirometry + demographics — real extra patient rows for augmentation;
  needs heavy schema harmonization and has no 5-year follow-up target.
- **Synthetic data (SDV / CTGAN)** — augment/rebalance the joined dataset; train
  only on our real rows, use for training only, never as test data.

See [`data_sources.md`](data_sources.md) for the fuller evaluation and effort
estimates for these.
