"""Build the NHANES 2011-2012 COPD cohort with derived clinical features.

This script merges the NHANES 2011-2012 DEMO_G, SPX_G, BMX_G, and SMQ_G files,
filters to adults with acceptable spirometry, and derives:

  - Demographics: age, sex, race/ethnicity, height, weight, BMI
  - Spirometry: FEV1, FVC, FEV1/FVC ratio
  - FEV1 % predicted using NHANES III reference equations (Hankinson 1999)
  - COPD diagnosis (FEV1/FVC < 0.70)
  - GOLD stage (0-4) based on FEV1 % predicted
  - Smoking status and pack-years
  - BMI category and age group

Output is written to data/nhanes_cohort/nhanes_copd_cohort.csv and is intended to
be consumed by the copd_ingestion DAG as the primary preprocessed dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from spiref import nhanes3


RAW_DIR = Path("data/nhanes_raw")
COHORT_DIR = Path("data/nhanes_cohort")
COHORT_PATH = COHORT_DIR / "nhanes_copd_cohort.csv"
META_PATH = COHORT_DIR / "nhanes_copd_cohort.meta.json"


# NHANES 2011-2012 spirometry quality codes (from SPXNSTAT cross-tab):
# 1 = complete with acceptable data, 2 = complete but no acceptable data,
# 3 = incomplete, 4 = not done.
ACCEPTABLE_SPXNSTAT = 1.0


def load_nhanes() -> dict[str, pd.DataFrame]:
    """Load the four NHANES XPT files we need."""
    return {
        "demo": pd.read_sas(RAW_DIR / "DEMO_G.xpt"),
        "spx": pd.read_sas(RAW_DIR / "SPX_G.xpt"),
        "bmx": pd.read_sas(RAW_DIR / "BMX_G.xpt"),
        "smq": pd.read_sas(RAW_DIR / "SMQ_G.xpt"),
    }


def _decode_bytes(series: pd.Series) -> pd.Series:
    """Decode SAS byte-string columns to plain strings."""
    if series.dtype == object:
        return series.apply(
            lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
        )
    return series


def merge_core(demo: pd.DataFrame, spx: pd.DataFrame, bmx: pd.DataFrame) -> pd.DataFrame:
    """Merge DEMO, SPX, and BMX on SEQN and rename columns to friendly names."""
    df = (
        demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"]]
        .merge(
            spx[["SEQN", "SPXNSTAT", "SPXNFVC", "SPXNFEV1", "SPXNQFVC", "SPXNQFV1"]],
            on="SEQN",
            how="inner",
        )
        .merge(
            bmx[["SEQN", "BMXHT", "BMXWT", "BMXBMI"]],
            on="SEQN",
            how="inner",
        )
    )

    df = df.rename(
        columns={
            "RIDAGEYR": "age",
            "RIAGENDR": "sex",
            "RIDRETH3": "race_ethnicity",
            "SPXNFVC": "fvc_ml",
            "SPXNFEV1": "fev1_ml",
            "BMXHT": "height_cm",
            "BMXWT": "weight_kg",
            "BMXBMI": "bmi",
        }
    )
    df["fev1_fvc_ratio"] = df["fev1_ml"] / df["fvc_ml"]
    return df


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Keep adults with complete, acceptable spirometry and valid measures."""
    df = df[df["SPXNSTAT"] == ACCEPTABLE_SPXNSTAT].copy()
    df = df[df["age"] >= 18].copy()
    df = df[df["fvc_ml"] > 0].copy()
    df = df[df["fev1_ml"] > 0].copy()
    df = df[df["fev1_fvc_ratio"] > 0].copy()
    df = df[df["height_cm"] > 0].copy()
    df = df[df["bmi"] > 0].copy()

    # Quality flags A/B/C are acceptable; D is poor quality.
    for col in ("SPXNQFVC", "SPXNQFV1"):
        df[col] = _decode_bytes(df[col])
        df = df[df[col].isin(["A", "B", "C"])].copy()

    return df


def _spiref_race(race_ethnicity: int) -> tuple[str, float]:
    """Map NHANES RIDRETH3 to spiref race label and optional Asian correction factor.

    Returns (race_label, correction_factor). For Asian participants we apply a
    0.88 correction to the Caucasian reference predicted value, consistent with
    Hankinson et al. 2009 (MESA Lung) findings.
    """
    mapping = {
        1: ("MexAm", 1.0),  # Mexican American
        2: ("Cau", 1.0),    # Other Hispanic -> Caucasian reference
        3: ("Cau", 1.0),    # Non-Hispanic White
        4: ("AfrAm", 1.0),  # Non-Hispanic Black
        6: ("Cau", 0.88),   # Non-Hispanic Asian (apply correction to Caucasian ref)
        7: ("Cau", 1.0),    # Other/Multiracial
    }
    return mapping.get(race_ethnicity, ("Cau", 1.0))


def _spiref_sex(sex_code: int) -> str:
    return "male" if sex_code == 1 else "female"


def compute_fev1_percent_predicted(df: pd.DataFrame) -> pd.Series:
    """Compute FEV1 % predicted using NHANES III reference equations via spiref."""
    rvc = nhanes3.NHANESReferenceValueCalculator()
    preds = []
    for _, row in df.iterrows():
        race, correction = _spiref_race(int(row["race_ethnicity"]))
        pred = rvc.calculate_fev1(
            _spiref_sex(int(row["sex"])),
            float(row["height_cm"]),
            float(row["age"]),
            race=race,
        )
        pred *= correction
        preds.append(pred)
    preds = np.array(preds)
    observed_l = df["fev1_ml"].to_numpy() / 1000.0
    return pd.Series((observed_l / preds) * 100.0, index=df.index)


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add COPD diagnosis and GOLD stage columns."""
    df["copd_diagnosis"] = (df["fev1_fvc_ratio"] < 0.70).astype(int)

    def gold_stage(row: pd.Series) -> int:
        if row["fev1_fvc_ratio"] >= 0.70:
            return 0
        pct = row["fev1_pct_predicted"]
        if pct >= 80:
            return 1
        if pct >= 50:
            return 2
        if pct >= 30:
            return 3
        return 4

    df["gold_stage"] = df.apply(gold_stage, axis=1)
    return df


def compute_smoking_features(df: pd.DataFrame, smq: pd.DataFrame) -> pd.DataFrame:
    """Derive smoking status and pack-years from SMQ_G and merge onto the cohort."""
    smq = smq[["SEQN", "SMQ020", "SMQ040", "SMD030", "SMD641", "SMD650"]].copy()

    def smoking_status(row: pd.Series) -> str:
        if row["SMQ020"] != 1.0:
            return "never"
        if row["SMQ040"] in (1.0, 2.0):
            return "current"
        if row["SMQ040"] == 3.0:
            return "former"
        return "unknown"

    def pack_years(row: pd.Series, age: float) -> float:
        if row["SMQ020"] != 1.0:
            return 0.0

        age_started = row["SMD030"]
        if pd.isna(age_started) or age_started < 5 or age_started > 99 or age_started > age:
            age_started = 18.0  # fallback for missing/unreasonable age started

        years_smoked = max(0.0, age - age_started)

        if row["SMQ040"] in (1.0, 2.0):  # current
            cigs_per_day = row["SMD650"]
        elif row["SMQ040"] == 3.0:  # former
            cigs_per_day = row["SMD641"]
        else:
            cigs_per_day = np.nan

        if pd.isna(cigs_per_day) or cigs_per_day <= 0 or cigs_per_day > 99:
            cigs_per_day = 10.0  # conservative fallback for missing intensity

        return (cigs_per_day / 20.0) * years_smoked

    smq["smoking_status"] = smq.apply(smoking_status, axis=1)
    smq["pack_years"] = smq.apply(lambda row: pack_years(row, np.nan), axis=1)

    df = df.merge(
        smq[["SEQN", "smoking_status", "pack_years"]],
        on="SEQN",
        how="left",
    )
    df["smoking_status"] = df["smoking_status"].fillna("unknown")
    df["pack_years"] = df["pack_years"].fillna(0.0)
    return df


def compute_derived_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add BMI category and age group."""
    bins_bmi = [0, 18.5, 25.0, 30.0, np.inf]
    labels_bmi = ["underweight", "normal", "overweight", "obese"]
    df["bmi_category"] = pd.cut(df["bmi"], bins=bins_bmi, labels=labels_bmi, right=False)

    bins_age = [0, 40, 50, 60, 70, np.inf]
    labels_age = ["<40", "40-49", "50-59", "60-69", "70+"]
    df["age_group"] = pd.cut(df["age"], bins=bins_age, labels=labels_age, right=False)
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return the clean cohort with feature and target columns."""
    return df[
        [
            "SEQN",
            "age",
            "age_group",
            "sex",
            "race_ethnicity",
            "height_cm",
            "weight_kg",
            "bmi",
            "bmi_category",
            "fvc_ml",
            "fev1_ml",
            "fev1_fvc_ratio",
            "fev1_pct_predicted",
            "smoking_status",
            "pack_years",
            "copd_diagnosis",
            "gold_stage",
        ]
    ].copy()


def build_cohort() -> pd.DataFrame:
    """Run the full NHANES COPD cohort build."""
    data = load_nhanes()
    df = merge_core(data["demo"], data["spx"], data["bmx"])
    df = apply_quality_filters(df)
    df["fev1_pct_predicted"] = compute_fev1_percent_predicted(df)
    df = compute_targets(df)
    df = compute_smoking_features(df, data["smq"])
    df = compute_derived_categories(df)
    df = select_final_columns(df)
    return df


def main() -> None:
    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort()
    cohort.to_csv(COHORT_PATH, index=False)

    metadata = {
        "source": "NHANES 2011-2012 (DEMO_G, SPX_G, BMX_G, SMQ_G)",
        "n_rows": int(len(cohort)),
        "n_features": int(cohort.shape[1] - 2),  # excluding SEQN and targets
        "copd_prevalence": float(cohort["copd_diagnosis"].mean()),
        "gold_stage_distribution": cohort["gold_stage"].value_counts().sort_index().to_dict(),
        "smoking_status_distribution": cohort["smoking_status"].value_counts().to_dict(),
        "age_group_distribution": cohort["age_group"].value_counts().to_dict(),
        "bmi_category_distribution": cohort["bmi_category"].value_counts().to_dict(),
        "output_path": str(COHORT_PATH),
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Built NHANES COPD cohort: {len(cohort)} rows -> {COHORT_PATH}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
