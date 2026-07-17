from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
PORT = int(os.environ.get("PORT", "7860"))
DIST_DIR = Path(os.environ.get("UI_DIST_DIR", Path(__file__).resolve().parents[1] / "dist"))
INDEX_HTML = DIST_DIR / "index.html"


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _get_json(path: str) -> dict[str, Any]:
    r = requests.get(f"{BACKEND_URL}{path}", timeout=60)
    r.raise_for_status()
    return r.json()


app = FastAPI(title="COPD Predictor UI")

if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
def root() -> Any:
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return RedirectResponse("/welcome", status_code=307)


@app.get("/welcome", response_class=HTMLResponse)
def welcome() -> Any:
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return HTMLResponse("<html><body><h1>UI build missing</h1></body></html>")


@app.get("/demo", response_class=HTMLResponse)
def demo() -> Any:
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return HTMLResponse("<html><body><h1>UI build missing</h1></body></html>")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model")
def model() -> dict[str, Any]:
    return _get_json("/model")


@app.post("/trigger_ingestion")
def trigger_ingestion(payload: dict[str, Any]) -> dict[str, Any]:
    return _post_json("/trigger_ingestion", payload)


@app.post("/trigger_training")
def trigger_training(payload: dict[str, Any]) -> dict[str, Any]:
    return _post_json("/trigger_training", payload)


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    return _post_json("/predict", payload)


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    raw = await file.read()
    r = requests.post(
        f"{BACKEND_URL}/predict_csv",
        files={"file": (file.filename or "upload.csv", io.BytesIO(raw), "text/csv")},
        timeout=180,
    )
    if r.status_code >= 300:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return StreamingResponse(
        iter([r.content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename or 'preds.csv'}"},
    )


@app.get("/{path:path}")
def spa_fallback(path: str) -> Any:
    # Let the React router own client-side navigation.
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    raise HTTPException(status_code=404, detail="UI bundle not built")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
