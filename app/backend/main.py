"""FastAPI backend for the COPD screening app.

Serves the champion(s) registered in PostgreSQL, loading artifacts from MLflow:

    gold_copd   binary      fev1_fvc_ratio < 0.70 (the repo's decided target)
    gold_stage  multiclass  optional second stage; not built in the repo yet

Endpoints:
    GET  /health            liveness + dependency status
    GET  /model/champions   champion metadata per target
    GET  /model/schema      form field definitions (drives the frontend form)
    POST /predict           single subject -> diagnosis, then staging if positive
    POST /predict/batch     CSV upload, one cascade per row
"""

from __future__ import annotations

import csv
import io
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import db
import inference
from features import FEATURE_SCHEMA, group_order
from inference import DIAGNOSIS_TARGET, GOLD_TARGET, ModelNotReady

app = FastAPI(
    title="COPD Screening API",
    description="Serves the champion classifier for COPD screening (gold_copd).",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Raw, human-readable field values.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _champions_or_503() -> dict[str, dict[str, Any]]:
    try:
        champions = db.fetch_champions()
    except Exception as exc:
        raise HTTPException(503, f"Model registry unavailable: {exc}") from exc
    try:
        inference.ensure_loaded(champions)
    except ModelNotReady as exc:
        raise HTTPException(503, str(exc)) from exc
    return champions


def _coerce(features: dict[str, Any]) -> dict[str, Any]:
    """Cast form values to declared types; collect field-level errors."""
    by_name = {f["name"]: f for f in FEATURE_SCHEMA}
    cleaned: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for name, spec in by_name.items():
        value = features.get(name)
        if value in (None, "", []):
            if spec["required"]:
                errors[name] = "This field is required."
            cleaned[name] = [] if spec["type"] == "multi_categorical" else None
            continue

        if spec["type"] == "number":
            try:
                num = float(value)
            except (TypeError, ValueError):
                errors[name] = "Enter a number."
                continue
            if "min" in spec and num < spec["min"]:
                errors[name] = f"Must be at least {spec['min']}."
                continue
            if "max" in spec and num > spec["max"]:
                errors[name] = f"Must be at most {spec['max']}."
                continue
            cleaned[name] = num

        elif spec["type"] == "multi_categorical":
            # Accepts a JSON list, or a pipe-delimited string so a CSV column can
            # carry the source's own "asthma|pneumonia" format verbatim.
            if isinstance(value, str):
                picked = [v.strip() for v in value.split("|") if v.strip()]
            elif isinstance(value, (list, tuple)):
                picked = [str(v).strip() for v in value if str(v).strip()]
            else:
                errors[name] = "Expected a list of values."
                continue
            picked = [v.lower() for v in picked]
            unknown = [v for v in picked if v not in spec["options"]]
            if unknown:
                errors[name] = f"Unrecognised: {', '.join(unknown)}."
                continue
            cleaned[name] = picked

        else:
            text = str(value).strip().lower()
            if text not in spec["options"]:
                errors[name] = f"Choose one of: {', '.join(spec['options'])}."
                continue
            cleaned[name] = text

    for name in set(features) - set(by_name):
        errors[name] = "Unknown field."

    if errors:
        raise HTTPException(422, detail={"field_errors": errors})
    return cleaned


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    db_ok = db.ping()
    return {
        "status": "ok" if db_ok else "degraded",
        "db": "ok" if db_ok else "unreachable",
        "mlflow_tracking_uri": config.MLFLOW_TRACKING_URI,
        **inference.status(),
    }


@app.get("/model/champions")
def champions() -> dict[str, Any]:
    champs = _champions_or_503()
    return {
        "targets": {
            target: {
                "model_name": c["model_name"],
                "mlflow_run_id": c["mlflow_run_id"],
                "class_labels": c["class_labels"],
                "metrics": c["metrics"],
                "created_at": c["created_at"],
            }
            for target, c in champs.items()
        },
        "diagnosis_target": DIAGNOSIS_TARGET,
        "gold_target": GOLD_TARGET,
    }


@app.get("/model/schema")
def schema() -> dict[str, Any]:
    """Field definitions for the input form.

    Deliberately does not require a loaded model: the form must render (and show
    a clear 'no champion yet' state) before the modeling DAG has ever run.
    """
    return {"groups": group_order(), "features": FEATURE_SCHEMA}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    champs = _champions_or_503()
    cleaned = _coerce(req.features)
    try:
        return inference.predict([cleaned], champs)[0]
    except ModelNotReady as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}") from exc


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)) -> dict[str, Any]:
    champs = _champions_or_503()

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file.")

    payload = (await file.read()).decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(payload))
    if reader.fieldnames is None:
        raise HTTPException(400, "The CSV has no header row.")

    valid_rows: list[dict[str, Any]] = []
    valid_idx: list[int] = []
    errors: list[dict[str, Any]] = []

    for i, row in enumerate(reader):
        try:
            valid_rows.append(_coerce({k: v for k, v in row.items() if k}))
            valid_idx.append(i)
        except HTTPException as exc:
            detail = exc.detail
            msg = detail.get("field_errors") if isinstance(detail, dict) else str(detail)
            errors.append({"row": i, "message": msg})

    results: list[dict[str, Any]] = []
    if valid_rows:
        try:
            preds = inference.predict(valid_rows, champs)
        except Exception as exc:
            raise HTTPException(500, f"Batch prediction failed: {exc}") from exc
        results = [{"row": idx, **p} for idx, p in zip(valid_idx, preds)]

    return {
        "n_rows": len(valid_rows) + len(errors),
        "n_failed": len(errors),
        "results": results,
        "errors": errors,
    }
