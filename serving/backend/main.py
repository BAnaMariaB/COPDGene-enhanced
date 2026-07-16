from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from serving.common.champion_registry import connect as db_connect
from serving.common.champion_registry import fetch_active_champion
from serving.common.preprocessing import load_preprocessing_artifacts, transform_rows
from serving.common.system_bundle import load_system_bundle


MODEL_NAME = os.environ.get("CHAMPION_MODEL_NAME", "copd_double_target_system")
MODEL_TARGET = os.environ.get("CHAMPION_MODEL_TARGET", "system")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/copd_champion_cache")


class PatientInput(BaseModel):
    # Numeric
    age: float | None = None
    sex: float | None = None
    race_ethnicity: float | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    bmi: float | None = None
    pack_years: float | None = None

    # Optional spirometry inputs (may be missing)
    fev1_ml: float | None = None
    fvc_ml: float | None = None
    fev1_fvc_ratio: float | None = None
    fev1_pct_predicted: float | None = None

    # Categoricals
    smoking_status: str | None = None
    age_group: str | None = None
    bmi_category: str | None = None


@dataclass
class LoadedChampion:
    champion_id: int
    mlflow_run_id: str
    artifact_uri: str | None
    system_dir: str
    preprocessing_joblib: str
    system: Any
    preproc: Any
    metadata: dict[str, Any]


_lock = threading.Lock()
_cached: LoadedChampion | None = None


def _download_artifact(run_id: str, artifact_path: str) -> str:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    return mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=os.path.join(MODEL_CACHE_DIR, run_id),
    )


def _load_champion() -> LoadedChampion:
    global _cached

    with db_connect() as conn:
        row = fetch_active_champion(conn, model_name=MODEL_NAME, target=MODEL_TARGET)

    if row is None:
        raise HTTPException(status_code=503, detail="No active champion found in registry")

    if _cached is not None and _cached.champion_id == row.id:
        return _cached

    mlflow.set_tracking_uri(row.mlflow_tracking_uri)
    mlflow.set_experiment(row.mlflow_experiment_name)

    if not row.model_artifact_path or not row.preprocessing_artifact_path:
        raise HTTPException(
            status_code=500,
            detail="Champion row is missing model_artifact_path or preprocessing_artifact_path",
        )

    system_dir = _download_artifact(row.mlflow_run_id, row.model_artifact_path)
    preprocessing_joblib = _download_artifact(row.mlflow_run_id, row.preprocessing_artifact_path)

    system = load_system_bundle(system_dir)
    preproc = load_preprocessing_artifacts(preprocessing_joblib)

    metadata = {
        "champion_id": row.id,
        "model_name": row.model_name,
        "target": row.target,
        "mlflow_tracking_uri": row.mlflow_tracking_uri,
        "mlflow_experiment_name": row.mlflow_experiment_name,
        "mlflow_run_id": row.mlflow_run_id,
        "artifact_uri": row.artifact_uri,
        "model_artifact_path": row.model_artifact_path,
        "preprocessing_artifact_path": row.preprocessing_artifact_path,
        "metric_name": row.metric_name,
        "metric_value": row.metric_value,
        "registered_at": row.registered_at,
        "params": row.params_json or {},
        "diagnosis_threshold": getattr(system, "diagnosis_threshold", None),
    }

    loaded = LoadedChampion(
        champion_id=row.id,
        mlflow_run_id=row.mlflow_run_id,
        artifact_uri=row.artifact_uri,
        system_dir=system_dir,
        preprocessing_joblib=preprocessing_joblib,
        system=system,
        preproc=preproc,
        metadata=metadata,
    )

    _cached = loaded
    return loaded


app = FastAPI(title="COPD Champion Model API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model")
def model() -> dict[str, Any]:
    with _lock:
        ch = _load_champion()
    return ch.metadata


@app.post("/predict")
def predict(payload: PatientInput) -> dict[str, Any]:
    with _lock:
        ch = _load_champion()

    rows = [payload.model_dump()]
    X = transform_rows(rows, ch.preproc)

    diag_proba_full = ch.system.predict_diagnosis_proba(X)
    copd_idx = int(getattr(ch.system, "copd_label_value"))
    copd_proba = float(diag_proba_full[0, copd_idx])
    copd_pred = int(ch.system.predict_diagnosis(X)[0] == copd_idx)

    gold_pred = int(ch.system.predict_gold(X)[0])
    gold_proba = ch.system.predict_gold_proba(X)[0].tolist()

    return {
        "copd_proba": copd_proba,
        "copd_pred": copd_pred,
        "gold_stage_pred": gold_pred,
        "gold_stage_proba": gold_proba,
        "champion": ch.metadata,
    }


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    with _lock:
        ch = _load_champion()

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    rows = df.to_dict(orient="records")
    X = transform_rows(rows, ch.preproc)

    diag_proba_full = ch.system.predict_diagnosis_proba(X)
    copd_idx = int(getattr(ch.system, "copd_label_value"))
    copd_proba = diag_proba_full[:, copd_idx]
    copd_pred = (ch.system.predict_diagnosis(X) == copd_idx).astype(int)

    gold_pred = ch.system.predict_gold(X)
    gold_proba = ch.system.predict_gold_proba(X)

    out = df.copy()
    out["copd_proba"] = copd_proba
    out["copd_pred"] = copd_pred
    out["gold_stage_pred"] = gold_pred
    for k in range(5):
        out[f"gold_stage_proba_{k}"] = gold_proba[:, k]

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename or 'preds'}.csv"},
    )
