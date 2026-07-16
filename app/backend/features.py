"""Serving feature schema — the single source of truth for what the app accepts.

Rebuilt against `eda_feature_engineering/` (commit adding the EDA & feature
engineering staging area). That module is currently the ONLY place in the repo
with a decided, leakage-correct target, so the app conforms to it rather than to
CHANGES_MADE.md — the NHANES ingestion rewrite that document describes is not in
this branch (`airflow/dags/copd_ingestion.py` is still the three-source teaching
dataset, and `scripts/build_nhanes_cohort.py` does not exist).

TARGET
------
    gold_copd := fev1_fvc_ratio < 0.70          (binary; ~38.9% positive)

matching `feature_engineering.RECOMMENDED_TARGET`.

LEAKAGE BOUNDARY
----------------
Taken verbatim from `feature_engineering.TARGET_CANDIDATES["gold_copd"]
["excluded_features"]`:

    fev1_fvc_ratio  the exact value the label is thresholded from
    fev1            fev1 + fvc reconstruct the ratio exactly
    fev1_phase2     measured 5 years after baseline; unavailable at prediction time

Note `fvc` is deliberately NOT excluded — that is their documented decision, and
it is algebraically sound (fvc alone cannot recover the ratio). See the note on
FVC below for the separate, unresolved serving question it raises.

COLUMN VALUES
-------------
Verified against the live source files rather than inferred:

    gender          coded 1 / 2   (n=1333 / 1287)
    race            coded 1 / 2   (n=1887 / 733)
    smoking_status  coded 1 / 2   (n=1411 / 1209) — Current vs Former only;
                                  this cohort has no never-smokers
    visit_age       42.4 .. 81.0
    visit_year      2008 .. 2011
    respiratory     8 distinct conditions, pipe-delimited; 25.6% of rows null

The 1/2 codes are NOT defined in `docs/data_overview.md`, which documents only
that smoking_status is "Current vs Former" without saying which code is which.
`option_labels` below uses the standard COPDGene convention as a PROVISIONAL
guess. Confirm against the data dictionary before the demo — a flipped label
means the form silently collects the opposite of what the model was trained on,
and nothing will error.
"""

from __future__ import annotations

from typing import Any

# --- The target --------------------------------------------------------------

TARGET_COLUMN = "gold_copd"
GOLD_THRESHOLD = 0.70

# Excluded per feature_engineering.TARGET_CANDIDATES["gold_copd"].
TARGET_SIDE_COLUMNS = ["fev1_fvc_ratio", "fev1", "fev1_phase2"]

# Never features, for any target (per eda_feature_engineering/README.md and the
# ingestion DAG's own DROP_REASONS).
NON_FEATURE_COLUMNS = ["sid", "visit_date", "respiratory"]

CATEGORICAL_FEATURES = ["gender", "race", "smoking_status"]

# The 8 conditions parsed out of the multi-label `respiratory` column, verified
# against the source. feature_engineering.engineer_features() derives these
# from the data, so this list must be re-checked if the source changes.
RESPIRATORY_CONDITIONS = [
    "asthma",
    "bronchitis attacks",
    "chronic bronchitis",
    "copd",
    "emphysema",
    "hay fever",
    "pneumonia",
    "sleep apnea",
]

RESPIRATORY_FLAG_COLUMNS = [
    f"respiratory_is_{c.lower().replace(' ', '_')}" for c in RESPIRATORY_CONDITIONS
]

# ---------------------------------------------------------------------------
# Form fields — what a user actually types. Ordered; drives field order.
# Ranges come from the observed source data, widened slightly where a tight
# bound would reject a plausible new subject.
# ---------------------------------------------------------------------------

FEATURE_SCHEMA: list[dict[str, Any]] = [
    # --- Demographics --------------------------------------------------------
    {"name": "visit_age", "label": "Age", "group": "Demographics",
     "type": "number", "unit": "years", "min": 40, "max": 90, "required": True},
    {"name": "gender", "label": "Gender", "group": "Demographics",
     "type": "categorical", "options": ["1", "2"],
     "option_labels": {"1": "1 — male", "2": "2 — female"},
     "help": "Source-coded value. Mapping is provisional — confirm the data dictionary.",
     "required": True},
    {"name": "race", "label": "Race", "group": "Demographics",
     "type": "categorical", "options": ["1", "2"],
     "option_labels": {"1": "1 — non-Hispanic white", "2": "2 — African American"},
     "help": "Source-coded value. Mapping is provisional — confirm the data dictionary.",
     "required": True},
    {"name": "height_cm", "label": "Height", "group": "Demographics",
     "type": "number", "unit": "cm", "min": 130, "max": 210, "required": True},
    {"name": "weight_kg", "label": "Weight", "group": "Demographics",
     "type": "number", "unit": "kg", "min": 35, "max": 180, "required": True},
    {"name": "visit_year", "label": "Visit year", "group": "Demographics",
     "type": "number", "min": 2008, "max": 2011, "required": False,
     "help": "Source data spans 2008–2011 only."},

    # --- Smoking -------------------------------------------------------------
    {"name": "smoking_status", "label": "Smoking status", "group": "Smoking",
     "type": "categorical", "options": ["1", "2"],
     "option_labels": {"1": "1 — current", "2": "2 — former"},
     "help": "Cohort is smokers-only; there is no never-smoker code.",
     "required": True},
    {"name": "smoke_start_age", "label": "Age started smoking", "group": "Smoking",
     "type": "number", "unit": "years", "min": 5, "max": 55, "required": True},
    {"name": "cigs_per_day_avg", "label": "Cigarettes per day", "group": "Smoking",
     "type": "number", "unit": "avg", "min": 1, "max": 100, "required": True},
    {"name": "duration_smoking", "label": "Years smoked", "group": "Smoking",
     "type": "number", "unit": "years", "min": 1, "max": 70, "required": True,
     "help": "Not age minus start age — quitters smoked for less."},

    # --- Vitals --------------------------------------------------------------
    {"name": "blood_pressure_systolic", "label": "Systolic BP", "group": "Vitals",
     "type": "number", "unit": "mmHg", "min": 70, "max": 230, "required": True},
    {"name": "blood_pressure_diastolic", "label": "Diastolic BP", "group": "Vitals",
     "type": "number", "unit": "mmHg", "min": 35, "max": 140, "required": True},
    {"name": "heart_rate", "label": "Heart rate", "group": "Vitals",
     "type": "number", "unit": "bpm", "min": 35, "max": 150, "required": True},
    {"name": "hours_on_oxygen", "label": "Hours on oxygen", "group": "Vitals",
     "type": "number", "unit": "hrs/day", "min": 0, "max": 24, "required": True},

    # --- CT imaging ----------------------------------------------------------
    {"name": "emphysema_percentage", "label": "Emphysema", "group": "CT imaging",
     "type": "number", "unit": "%", "min": 0, "max": 100, "required": True},
    {"name": "gas_trapping_percentage", "label": "Gas trapping", "group": "CT imaging",
     "type": "number", "unit": "%", "min": 0, "max": 100, "required": True},
    {"name": "lung_volume_inspiratory", "label": "Lung volume (inspiratory)", "group": "CT imaging",
     "type": "number", "unit": "L", "min": 1, "max": 12, "required": True},
    {"name": "lung_volume_expiratory", "label": "Lung volume (expiratory)", "group": "CT imaging",
     "type": "number", "unit": "L", "min": 0.5, "max": 10, "required": True},
    {"name": "mean_density_inspiratory", "label": "Mean density (inspiratory)", "group": "CT imaging",
     "type": "number", "unit": "HU", "min": -1000, "max": -400, "required": True},
    {"name": "mean_density_expiratory", "label": "Mean density (expiratory)", "group": "CT imaging",
     "type": "number", "unit": "HU", "min": -1000, "max": -300, "required": True},

    # --- Spirometry ----------------------------------------------------------
    # See the FVC note in the module docstring / app/README.md. This is here
    # because the model is trained on it, not because it belongs in a screen.
    {"name": "fvc", "label": "FVC", "group": "Spirometry",
     "type": "number", "unit": "L", "min": 0.5, "max": 8, "required": True,
     "help": "Kept as a feature by the gold_copd target spec. Requires a spirometry test — see README."},

    # --- Respiratory history -------------------------------------------------
    {"name": "respiratory_conditions", "label": "Reported conditions",
     "group": "Respiratory history",
     "type": "multi_categorical", "options": RESPIRATORY_CONDITIONS, "required": False,
     "help": "Select all reported. Leave empty if no history was recorded (25.6% of the source)."},
]

FEATURE_NAMES = [f["name"] for f in FEATURE_SCHEMA]

# Collected but not passed to the model — they only feed derive().
FORM_ONLY_FIELDS = ["respiratory_conditions"]

# Computed server-side. `bmi` is a source column, but it equals
# weight/(height/100)^2 to within 0.005 across all 2620 rows, so the form
# derives it rather than asking twice. The rest mirror
# feature_engineering.engineer_features().
DERIVED_FEATURES = [
    "bmi",
    "pulse_pressure",
    "pack_years",
    "air_trapping_ratio",
    "respiratory_reported",
] + RESPIRATORY_FLAG_COLUMNS

# Exactly what the serving preprocessor must have been fitted on.
EXPECTED_MODEL_COLUMNS = [
    n for n in FEATURE_NAMES if n not in FORM_ONLY_FIELDS
] + DERIVED_FEATURES


# ---------------------------------------------------------------------------
# Derivations — must mirror feature_engineering.engineer_features() exactly
# ---------------------------------------------------------------------------

def derive(row: dict[str, Any]) -> dict[str, Any]:
    """Turn form values into the column set the preprocessor expects."""
    out = dict(row)

    def num(name):
        v = out.get(name)
        return float(v) if v not in (None, "") else None

    height, weight = num("height_cm"), num("weight_kg")
    out["bmi"] = round(weight / (height / 100.0) ** 2, 2) if height and weight else None

    sys_bp, dia_bp = num("blood_pressure_systolic"), num("blood_pressure_diastolic")
    out["pulse_pressure"] = sys_bp - dia_bp if sys_bp is not None and dia_bp is not None else None

    cigs, years = num("cigs_per_day_avg"), num("duration_smoking")
    out["pack_years"] = (cigs / 20.0) * years if cigs is not None and years is not None else None

    insp, exp = num("lung_volume_inspiratory"), num("lung_volume_expiratory")
    out["air_trapping_ratio"] = exp / insp if insp else None

    selected = {str(c).strip().lower() for c in (out.get("respiratory_conditions") or [])}
    # In the source, `respiratory` is non-null iff at least one condition is
    # listed, so "any selected" is exactly notna().
    out["respiratory_reported"] = len(selected) > 0
    for condition, col in zip(RESPIRATORY_CONDITIONS, RESPIRATORY_FLAG_COLUMNS):
        out[col] = condition in selected

    for field in FORM_ONLY_FIELDS:
        out.pop(field, None)
    return out


def group_order() -> list[str]:
    seen: list[str] = []
    for f in FEATURE_SCHEMA:
        if f["group"] not in seen:
            seen.append(f["group"])
    return seen
