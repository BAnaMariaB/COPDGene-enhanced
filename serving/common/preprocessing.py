from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd


_SANITIZE_RE = re.compile(r"[\[\]<>]")


def sanitize_feature_names(names: list[str]) -> list[str]:
    """Match the training pipeline's feature-name sanitization."""
    return [_SANITIZE_RE.sub("_", n) for n in names]


def derive_age_group(age: float | int | None) -> str | None:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return None
    a = float(age)
    if a < 40:
        return "<40"
    if a < 50:
        return "40-49"
    if a < 60:
        return "50-59"
    if a < 70:
        return "60-69"
    return "70+"


def derive_bmi_category(bmi: float | int | None) -> str | None:
    if bmi is None or (isinstance(bmi, float) and np.isnan(bmi)):
        return None
    b = float(bmi)
    if b < 18.5:
        return "underweight"
    if b < 25:
        return "normal"
    if b < 30:
        return "overweight"
    return "obese"


@dataclass(frozen=True)
class PreprocessingArtifacts:
    preprocessor: Any
    numeric_columns: list[str]
    categorical_columns: list[str]


def load_preprocessing_artifacts(joblib_path: str) -> PreprocessingArtifacts:
    """Load the preprocessing artifacts bundle saved during dataset preparation."""
    bundle = joblib.load(joblib_path)
    preprocessor = bundle.get("preprocessor")
    if preprocessor is None:
        raise ValueError(f"preprocessor not found in preprocessing artifacts: {joblib_path}")
    numeric_columns = list(bundle.get("numeric_columns", []))
    categorical_columns = list(bundle.get("categorical_columns", []))
    return PreprocessingArtifacts(
        preprocessor=preprocessor,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def prepare_raw_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize user-provided rows into a DataFrame compatible with the preprocessor."""
    df = pd.DataFrame(rows)

    # Derive common engineered categoricals if absent.
    if "age_group" not in df.columns and "age" in df.columns:
        df["age_group"] = df["age"].apply(derive_age_group)
    if "bmi_category" not in df.columns and "bmi" in df.columns:
        df["bmi_category"] = df["bmi"].apply(derive_bmi_category)

    # Ensure expected columns exist (imputer handles NaNs).
    for col in [
        "age",
        "sex",
        "race_ethnicity",
        "height_cm",
        "weight_kg",
        "bmi",
        "pack_years",
        "fev1_ml",
        "fvc_ml",
        "fev1_fvc_ratio",
        "fev1_pct_predicted",
        "smoking_status",
        "age_group",
        "bmi_category",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce numerics where appropriate.
    for col in [
        "age",
        "sex",
        "race_ethnicity",
        "height_cm",
        "weight_kg",
        "bmi",
        "pack_years",
        "fev1_ml",
        "fvc_ml",
        "fev1_fvc_ratio",
        "fev1_pct_predicted",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def transform_rows(
    rows: list[dict[str, Any]],
    artifacts: PreprocessingArtifacts,
) -> pd.DataFrame:
    """Transform raw rows to model feature-space using the saved preprocessor."""
    df_raw = prepare_raw_df(rows)
    arr = artifacts.preprocessor.transform(df_raw)

    # ColumnTransformer returns numpy array; get feature names.
    names = list(artifacts.preprocessor.get_feature_names_out())
    names = sanitize_feature_names(names)

    df_features = pd.DataFrame(arr, columns=names)
    return df_features
