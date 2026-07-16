"""Tests for the web application layer (FastAPI backend).

Three groups matter beyond ordinary endpoint coverage:

  * the serving contract — validate_preprocessor() refuses anything fitted on
    the columns that define the label;
  * agreement with eda_feature_engineering — the app's leakage boundary and
    derivations are asserted against that module directly, so if someone edits
    TARGET_CANDIDATES or engineer_features() and not features.py, this fails;
  * the cascade — severity staging isn't built yet, so screening must work
    without it.
"""

from __future__ import annotations

import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(REPO_ROOT, "app", "backend")
EDA_DIR = os.path.join(REPO_ROOT, "eda_feature_engineering")
for path in (BACKEND_DIR, EDA_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import config  # noqa: E402
import db  # noqa: E402
import inference  # noqa: E402
import main  # noqa: E402
from features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    EXPECTED_MODEL_COLUMNS,
    FEATURE_NAMES,
    RESPIRATORY_FLAG_COLUMNS,
    TARGET_COLUMN,
    TARGET_SIDE_COLUMNS,
    derive,
)
from inference import DIAGNOSIS_TARGET, GOLD_TARGET, ModelNotReady  # noqa: E402

DIAG_LABELS = ["no_copd", "copd"]
GOLD_LABELS = ["gold_1", "gold_2", "gold_3", "gold_4"]

VALID_INPUT = {
    "visit_age": 64.0,
    "gender": "1",
    "race": "1",
    "height_cm": 175.0,
    "weight_kg": 82.0,
    "visit_year": 2010,
    "smoking_status": "2",
    "smoke_start_age": 18,
    "cigs_per_day_avg": 20,
    "duration_smoking": 30,
    "blood_pressure_systolic": 130,
    "blood_pressure_diastolic": 80,
    "heart_rate": 72,
    "hours_on_oxygen": 0,
    "emphysema_percentage": 12.5,
    "gas_trapping_percentage": 22.1,
    "lung_volume_inspiratory": 6.1,
    "lung_volume_expiratory": 3.2,
    "mean_density_inspiratory": -860,
    "mean_density_expiratory": -750,
    "fvc": 3.9,
    "respiratory_conditions": ["asthma", "pneumonia"],
}


def _fit_preprocessor(columns: list[str]) -> ColumnTransformer:
    rng = np.random.default_rng(0)
    n = 60
    data = {}
    for col in columns:
        if col in CATEGORICAL_FEATURES:
            data[col] = rng.choice(["1", "2"], n)
        elif col == TARGET_COLUMN:
            data[col] = rng.choice([True, False], n)
        elif col.startswith("respiratory_"):
            data[col] = rng.choice([True, False], n)
        else:
            data[col] = rng.normal(size=n)
    df = pd.DataFrame(data)

    cat = [c for c in columns if c in CATEGORICAL_FEATURES]
    num = [c for c in columns if c not in cat]
    pre = ColumnTransformer(
        transformers=[
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pre.fit(df)
    return pre


class _StubModel:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)
        self.calls = 0
        self.last_n = None

    def predict(self, X):
        self.calls += 1
        self.last_n = len(X)
        return np.tile(self.probs, (len(X), 1))


def _champion(target, model_name, labels, run_id):
    return {
        "model_name": model_name,
        "mlflow_run_id": run_id,
        "artifact_uri": f"runs:/{run_id}/model",
        "target_name": target,
        "class_labels": labels,
        "metrics": {"f1_macro": 0.71},
        "created_at": "2026-07-16T18:00:00Z",
    }


def _wire(monkeypatch, diag_probs, gold_probs=(0.1, 0.6, 0.2, 0.1), with_gold=False):
    root = tempfile.mkdtemp(prefix="copd_preproc_")
    part = os.path.join(root, "2026-07-16")
    os.makedirs(part)
    joblib.dump(
        {"preprocessor": _fit_preprocessor(EXPECTED_MODEL_COLUMNS)},
        os.path.join(part, "serving_preprocessor.joblib"),
    )
    monkeypatch.setattr(config, "PREPROCESSED_ROOT", root)

    champs = {DIAGNOSIS_TARGET: _champion(DIAGNOSIS_TARGET, "xgb_gold_copd", DIAG_LABELS, "run-diag")}
    if with_gold:
        champs[GOLD_TARGET] = _champion(GOLD_TARGET, "xgb_stage", GOLD_LABELS, "run-gold")
    monkeypatch.setattr(db, "fetch_champions", lambda: champs)

    stubs = {
        "runs:/run-diag/model": _StubModel(diag_probs),
        "runs:/run-gold/model": _StubModel(gold_probs),
    }
    monkeypatch.setattr(inference, "load_model", lambda uri: stubs[uri])
    inference.reset_cache()
    return TestClient(main.app, raise_server_exceptions=False), stubs


@pytest.fixture
def positive_env(monkeypatch):
    return _wire(monkeypatch, diag_probs=(0.2, 0.8))


@pytest.fixture
def negative_env(monkeypatch):
    return _wire(monkeypatch, diag_probs=(0.9, 0.1))


# ---------------------------------------------------------------------------
# Agreement with eda_feature_engineering — the app must not drift from it
# ---------------------------------------------------------------------------

def test_target_matches_eda_module():
    import feature_engineering as fe

    assert TARGET_COLUMN == fe.RECOMMENDED_TARGET


def test_leakage_boundary_matches_eda_module():
    """TARGET_SIDE_COLUMNS must equal the module's excluded_features verbatim."""
    import feature_engineering as fe

    expected = fe.TARGET_CANDIDATES["gold_copd"]["excluded_features"]
    assert sorted(TARGET_SIDE_COLUMNS) == sorted(expected)


def test_fvc_is_a_feature_not_target_side():
    """Their documented decision: fvc alone can't reconstruct the ratio."""
    assert "fvc" not in TARGET_SIDE_COLUMNS
    assert "fvc" in EXPECTED_MODEL_COLUMNS


def test_pack_years_matches_eda_formula():
    """feature_engineering: (cigs_per_day_avg / 20) * duration_smoking."""
    row = derive(VALID_INPUT)
    assert row["pack_years"] == pytest.approx((20 / 20.0) * 30)


def test_pulse_pressure_matches_eda_formula():
    assert derive(VALID_INPUT)["pulse_pressure"] == pytest.approx(130 - 80)


def test_air_trapping_ratio_matches_eda_formula():
    assert derive(VALID_INPUT)["air_trapping_ratio"] == pytest.approx(3.2 / 6.1)


def test_respiratory_flags_match_selection():
    row = derive(VALID_INPUT)
    assert row["respiratory_is_asthma"] is True
    assert row["respiratory_is_pneumonia"] is True
    assert row["respiratory_is_emphysema"] is False
    assert row["respiratory_reported"] is True
    assert len(RESPIRATORY_FLAG_COLUMNS) == 8


def test_no_respiratory_history_reported_is_false():
    """25.6% of source rows are null here; empty selection must reproduce that."""
    row = derive({**VALID_INPUT, "respiratory_conditions": []})
    assert row["respiratory_reported"] is False
    assert all(row[c] is False for c in RESPIRATORY_FLAG_COLUMNS)


def test_bmi_is_derived_not_collected():
    """bmi is a source column but equals weight/(height/100)^2 to <0.005."""
    assert "bmi" not in FEATURE_NAMES
    assert derive(VALID_INPUT)["bmi"] == pytest.approx(26.78, abs=0.01)


def test_derive_produces_exactly_the_model_columns():
    assert set(derive(VALID_INPUT)) == set(EXPECTED_MODEL_COLUMNS)


# ---------------------------------------------------------------------------
# The serving contract
# ---------------------------------------------------------------------------

def test_validate_preprocessor_rejects_leakage():
    leaky = _fit_preprocessor(EXPECTED_MODEL_COLUMNS + TARGET_SIDE_COLUMNS)
    with pytest.raises(ModelNotReady) as exc:
        inference.validate_preprocessor(leaky)
    assert "target-side" in str(exc.value)
    for col in TARGET_SIDE_COLUMNS:
        assert col in str(exc.value)


def test_validate_preprocessor_rejects_the_label_column():
    leaky = _fit_preprocessor(EXPECTED_MODEL_COLUMNS + [TARGET_COLUMN])
    with pytest.raises(ModelNotReady) as exc:
        inference.validate_preprocessor(leaky)
    assert TARGET_COLUMN in str(exc.value)


def test_validate_preprocessor_accepts_serving_columns():
    inference.validate_preprocessor(_fit_preprocessor(EXPECTED_MODEL_COLUMNS))


def test_validate_preprocessor_reports_column_mismatch():
    pre = _fit_preprocessor(EXPECTED_MODEL_COLUMNS + ["some_new_column"])
    with pytest.raises(ModelNotReady) as exc:
        inference.validate_preprocessor(pre)
    assert "some_new_column" in str(exc.value)


def test_form_schema_excludes_target_and_label_columns():
    names = [f["name"] for f in TestClient(main.app).get("/model/schema").json()["features"]]
    for col in TARGET_SIDE_COLUMNS + [TARGET_COLUMN]:
        assert col not in names


def test_spirometry_leak_input_is_rejected(positive_env):
    """A caller must not be able to smuggle the label source in as a feature."""
    client, _ = positive_env
    r = client.post("/predict", json={"features": {**VALID_INPUT, "fev1_fvc_ratio": 0.62}})
    assert r.status_code == 422
    assert "fev1_fvc_ratio" in r.json()["detail"]["field_errors"]


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

def test_screens_without_a_staging_model(positive_env):
    """Severity staging isn't built yet; screening must stand alone."""
    client, _ = positive_env
    r = client.post("/predict", json={"features": VALID_INPUT})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["diagnosis"]["prediction"] == "copd"
    assert body["diagnosis"]["probabilities"]["copd"] == pytest.approx(0.8)
    assert body["gold_stage"] is None
    assert body["staging_skipped_reason"] == "no_gold_stage_champion"


def test_screen_negative_skips_staging(monkeypatch):
    client, stubs = _wire(monkeypatch, diag_probs=(0.9, 0.1), with_gold=True)
    body = client.post("/predict", json={"features": VALID_INPUT}).json()
    assert body["diagnosis"]["prediction"] == "no_copd"
    assert body["gold_stage"] is None
    assert body["staging_skipped_reason"] == "screen_negative"
    assert stubs["runs:/run-gold/model"].calls == 0


def test_screen_positive_runs_staging_when_available(monkeypatch):
    client, stubs = _wire(monkeypatch, diag_probs=(0.2, 0.8), with_gold=True)
    body = client.post("/predict", json={"features": VALID_INPUT}).json()
    assert body["gold_stage"]["prediction"] == "gold_2"
    assert stubs["runs:/run-gold/model"].calls == 1


def test_boolean_class_labels_still_stage(monkeypatch):
    """gold_copd is boolean: ["false","true"] must count as a positive screen."""
    client, stubs = _wire(monkeypatch, diag_probs=(0.2, 0.8), with_gold=True)
    champs = db.fetch_champions()
    champs[DIAGNOSIS_TARGET]["class_labels"] = ["false", "true"]
    monkeypatch.setattr(db, "fetch_champions", lambda: champs)
    body = client.post("/predict", json={"features": VALID_INPUT}).json()
    assert body["diagnosis"]["prediction"] == "true"
    assert body["gold_stage"] is not None


def test_missing_diagnosis_champion_is_503(monkeypatch):
    client, _ = _wire(monkeypatch, diag_probs=(0.2, 0.8))
    monkeypatch.setattr(db, "fetch_champions", lambda: {})
    inference.reset_cache()
    r = client.post("/predict", json={"features": VALID_INPUT})
    assert r.status_code == 503
    assert DIAGNOSIS_TARGET in r.json()["detail"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def test_schema_is_available_without_a_champion():
    assert TestClient(main.app).get("/model/schema").status_code == 200


def test_champions_endpoint(positive_env):
    client, _ = positive_env
    body = client.get("/model/champions").json()
    assert DIAGNOSIS_TARGET in body["targets"]
    assert body["diagnosis_target"] == DIAGNOSIS_TARGET


def test_missing_required_field_is_422(positive_env):
    client, _ = positive_env
    payload = {k: v for k, v in VALID_INPUT.items() if k != "visit_age"}
    r = client.post("/predict", json={"features": payload})
    assert r.status_code == 422
    assert "visit_age" in r.json()["detail"]["field_errors"]


def test_out_of_range_value_is_422(positive_env):
    client, _ = positive_env
    r = client.post("/predict", json={"features": {**VALID_INPUT, "visit_age": 900}})
    assert r.status_code == 422
    assert "visit_age" in r.json()["detail"]["field_errors"]


def test_uncoded_categorical_is_422(positive_env):
    """gender is coded 1/2 in this dataset, not 'male'/'female'."""
    client, _ = positive_env
    r = client.post("/predict", json={"features": {**VALID_INPUT, "gender": "male"}})
    assert r.status_code == 422
    assert "gender" in r.json()["detail"]["field_errors"]


def test_unknown_respiratory_condition_is_422(positive_env):
    client, _ = positive_env
    r = client.post(
        "/predict",
        json={"features": {**VALID_INPUT, "respiratory_conditions": ["gout"]}},
    )
    assert r.status_code == 422
    assert "respiratory_conditions" in r.json()["detail"]["field_errors"]


def test_batch_accepts_pipe_delimited_respiratory(positive_env):
    """CSV rows carry the source's own "asthma|pneumonia" format verbatim."""
    client, _ = positive_env
    cols = [k for k in VALID_INPUT if k != "respiratory_conditions"] + ["respiratory_conditions"]
    header = ",".join(cols)
    vals = {**VALID_INPUT, "respiratory_conditions": "asthma|pneumonia"}
    row = ",".join(str(vals[c]) for c in cols)
    csv_bytes = f"{header}\n{row}\n".encode()

    r = client.post("/predict/batch", files={"file": ("in.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["n_failed"] == 0


def test_batch_reports_per_row_errors_without_failing_the_run(positive_env):
    client, _ = positive_env
    cols = [k for k in VALID_INPUT if k != "respiratory_conditions"]
    header = ",".join(cols)
    good = ",".join(str(VALID_INPUT[c]) for c in cols)
    bad = ",".join("" if c == "visit_age" else str(VALID_INPUT[c]) for c in cols)
    csv_bytes = f"{header}\n{good}\n{bad}\n{good}\n".encode()

    r = client.post("/predict/batch", files={"file": ("in.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_rows"] == 3
    assert body["n_failed"] == 1
    assert body["errors"][0]["row"] == 1


def test_model_not_ready_is_503_not_500(monkeypatch):
    monkeypatch.setattr(config, "PREPROCESSED_ROOT", "/nonexistent")
    monkeypatch.setattr(db, "fetch_champions", lambda: {
        DIAGNOSIS_TARGET: _champion(DIAGNOSIS_TARGET, "m", DIAG_LABELS, "run-diag"),
    })
    inference.reset_cache()
    r = TestClient(main.app, raise_server_exceptions=False).post(
        "/predict", json={"features": VALID_INPUT}
    )
    assert r.status_code == 503
    assert "serving_preprocessor.joblib" in r.json()["detail"]
