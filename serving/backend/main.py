from __future__ import annotations

import io
import os
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import requests
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
INGESTION_DAG_ID = os.environ.get("INGESTION_DAG_ID", "copd_ingestion")
TRAINING_DAG_ID = os.environ.get("TRAINING_DAG_ID", "copd_train_validate_test")
AIRFLOW_API_URL = os.environ.get("AIRFLOW_API_URL", "").strip().rstrip("/")
AIRFLOW_API_USERNAME = os.environ.get("AIRFLOW_API_USERNAME", "").strip()
AIRFLOW_API_PASSWORD = os.environ.get("AIRFLOW_API_PASSWORD", "").strip()
AIRFLOW_CLI_BIN = os.environ.get("AIRFLOW_CLI_BIN", "airflow").strip()
AIRFLOW_HOME_CFG = os.environ.get("AIRFLOW_HOME", "/workspace/airflow_home_compose").strip()
AIRFLOW_DAGS_FOLDER_CFG = os.environ.get("AIRFLOW__CORE__DAGS_FOLDER", "/workspace/airflow/dags").strip()
AIRFLOW_DAG_SUBDIR_CFG = os.environ.get("AIRFLOW_DAGS_SUBDIR", AIRFLOW_DAGS_FOLDER_CFG).strip()


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

class TriggerInput(BaseModel):
    ds: str | None = None
    conf: dict[str, Any] | None = None


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


def _trigger_dag_via_api(dag_id: str, ds: str | None, conf: dict[str, Any] | None) -> dict[str, Any]:
    if not AIRFLOW_API_URL:
        raise RuntimeError("AIRFLOW_API_URL is not set")
    url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns"
    auth = (AIRFLOW_API_USERNAME, AIRFLOW_API_PASSWORD) if AIRFLOW_API_USERNAME and AIRFLOW_API_PASSWORD else None
    last_error = None

    for logical_date in _logical_date_candidates(ds):
        dag_run_id = f"manual__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}__{uuid.uuid4().hex[:8]}"
        payload: dict[str, Any] = {"dag_run_id": dag_run_id, "conf": conf or {}}
        if logical_date:
            payload["logical_date"] = logical_date
        resp = requests.post(url, json=payload, auth=auth, timeout=30)
        if resp.status_code < 300:
            out = resp.json()
            return {
                "mechanism": "airflow_api",
                "dag_id": dag_id,
                "dag_run_id": out.get("dag_run_id", dag_run_id),
                "state": out.get("state"),
                "logical_date": out.get("logical_date"),
            }
        body = resp.text
        last_error = f"Airflow API trigger failed ({resp.status_code}): {body}"
        if not _is_duplicate_logical_date_error(body):
            break

    raise RuntimeError(last_error or "Airflow API trigger failed")


def _trigger_dag_via_cli(dag_id: str, ds: str | None, conf: dict[str, Any] | None) -> dict[str, Any]:
    env = os.environ.copy()
    env["AIRFLOW_HOME"] = AIRFLOW_HOME_CFG
    env["AIRFLOW__CORE__DAGS_FOLDER"] = AIRFLOW_DAGS_FOLDER_CFG
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = "/workspace" if not current_pythonpath else f"/workspace:{current_pythonpath}"
    last_proc = None
    for logical_date in _logical_date_candidates(ds):
        trigger_args = [AIRFLOW_CLI_BIN, "dags", "trigger", dag_id]
        if conf:
            import json as _json
            trigger_args += ["--conf", _json.dumps(conf)]
        if logical_date:
            trigger_args += ["--logical-date", logical_date]

        candidate_cmds = []
        if AIRFLOW_DAG_SUBDIR_CFG:
            candidate_cmds.append(trigger_args + ["--subdir", AIRFLOW_DAG_SUBDIR_CFG])
        candidate_cmds.append(trigger_args)

        duplicate_error = False
        for cmd in candidate_cmds:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            stderr_lower = (proc.stderr or "").lower()
            combined_output = f"{proc.stderr}\n{proc.stdout}"

            if proc.returncode != 0 and "initialize the database" in stderr_lower:
                migrate = subprocess.run(
                    [AIRFLOW_CLI_BIN, "db", "migrate"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                if migrate.returncode == 0:
                    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                    stderr_lower = (proc.stderr or "").lower()
                    combined_output = f"{proc.stderr}\n{proc.stdout}"

            if proc.returncode != 0 and "dagnotfound" in stderr_lower and AIRFLOW_DAG_SUBDIR_CFG:
                reserialize = subprocess.run(
                    [AIRFLOW_CLI_BIN, "dags", "reserialize", "--subdir", AIRFLOW_DAG_SUBDIR_CFG],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                if reserialize.returncode == 0:
                    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                    stderr_lower = (proc.stderr or "").lower()
                    combined_output = f"{proc.stderr}\n{proc.stdout}"

            if proc.returncode == 0:
                return {"mechanism": "airflow_cli", "dag_id": dag_id, "stdout": proc.stdout.strip()}

            if "unrecognized arguments: --subdir" in stderr_lower:
                last_proc = proc
                continue

            if _is_duplicate_logical_date_error(combined_output):
                duplicate_error = True
                last_proc = proc
                break

            last_proc = proc
            duplicate_error = False
            break

        if duplicate_error:
            continue

    if last_proc is None:
        raise RuntimeError("Airflow CLI trigger failed before a command was executed")
    raise RuntimeError(f"Airflow CLI trigger failed: {last_proc.stderr.strip() or last_proc.stdout.strip()}")


def _trigger_dag(dag_id: str, ds: str | None, conf: dict[str, Any] | None) -> dict[str, Any]:
    if AIRFLOW_API_URL:
        return _trigger_dag_via_api(dag_id, ds, conf)
    return _trigger_dag_via_cli(dag_id, ds, conf)


def _download_artifact(run_id: str, artifact_path: str) -> str:
    # Local-path fast path (useful for local docker-compose smoke tests).
    if os.path.isabs(artifact_path) and os.path.exists(artifact_path):
        return artifact_path
    if os.path.exists(artifact_path):
        return artifact_path
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    return mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=os.path.join(MODEL_CACHE_DIR, run_id),
    )


def _logical_date_candidates(ds: str | None) -> list[str | None]:
    if not ds:
        return [None]
    base = datetime.fromisoformat(f"{ds}T00:00:00+00:00")
    return [(base + timedelta(seconds=offset)).isoformat() for offset in range(0, 5)]


def _is_duplicate_logical_date_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "unique constraint failed: dag_run.dag_id, dag_run.logical_date" in lowered
        or "dagrunalreadyexists" in lowered
        or "already exists for this logical date" in lowered
    )


def _resolve_local_artifact_path(artifact_uri: str | None, artifact_path: str | None) -> str | None:
    if not artifact_path:
        return None
    if os.path.isabs(artifact_path) and os.path.exists(artifact_path):
        return artifact_path
    if os.path.exists(artifact_path):
        return artifact_path
    if not artifact_uri:
        return None
    local_root = artifact_uri
    if local_root.startswith("file://"):
        local_root = local_root[len("file://") :]
    if not os.path.isabs(local_root):
        return None
    if not os.path.exists(local_root):
        for marker in ("/mlruns/", "/data/", "/examples/"):
            if marker in local_root:
                suffix = local_root.split(marker, 1)[1]
                translated = os.path.join("/workspace", marker.strip("/"), suffix)
                if os.path.exists(translated):
                    local_root = translated
                    break
    candidate = os.path.join(local_root, artifact_path)
    return candidate if os.path.exists(candidate) else None


def _load_champion() -> LoadedChampion:
    global _cached

    with db_connect() as conn:
        row = fetch_active_champion(conn, model_name=MODEL_NAME, target=MODEL_TARGET)

    if row is None:
        raise HTTPException(status_code=503, detail="No active champion found in registry")

    if _cached is not None and _cached.champion_id == row.id:
        return _cached

    if not row.model_artifact_path or not row.preprocessing_artifact_path:
        raise HTTPException(
            status_code=500,
            detail="Champion row is missing model_artifact_path or preprocessing_artifact_path",
        )

    local_system_dir = _resolve_local_artifact_path(row.artifact_uri, row.model_artifact_path)
    local_preproc = _resolve_local_artifact_path(row.artifact_uri, row.preprocessing_artifact_path)

    if local_system_dir and local_preproc:
        system_dir = local_system_dir
        preprocessing_joblib = local_preproc
    else:
        mlflow.set_tracking_uri(row.mlflow_tracking_uri)
        mlflow.set_experiment(row.mlflow_experiment_name)
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

@app.post("/trigger_ingestion")
def trigger_ingestion(payload: TriggerInput) -> dict[str, Any]:
    try:
        return _trigger_dag(INGESTION_DAG_ID, payload.ds, payload.conf)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trigger_training")
def trigger_training(payload: TriggerInput) -> dict[str, Any]:
    try:
        return _trigger_dag(TRAINING_DAG_ID, payload.ds, payload.conf)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(payload: PatientInput) -> dict[str, Any]:
    with _lock:
        ch = _load_champion()

    rows = [payload.model_dump()]
    X = transform_rows(rows, ch.preproc)

    diag_proba_full = ch.system.predict_diagnosis_proba(X)
    copd_label = int(getattr(ch.system, "copd_label_value"))
    copd_col = int(getattr(ch.system, "copd_class_index", copd_label))
    copd_proba = float(diag_proba_full[0, copd_col])
    copd_pred = int(ch.system.predict_diagnosis(X)[0] == copd_label)

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
    copd_label = int(getattr(ch.system, "copd_label_value"))
    copd_col = int(getattr(ch.system, "copd_class_index", copd_label))
    copd_proba = diag_proba_full[:, copd_col]
    copd_pred = (ch.system.predict_diagnosis(X) == copd_label).astype(int)

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
