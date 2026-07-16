from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _normalize_pg_url(url: str) -> str:
    # Allow SQLAlchemy-style URLs in env vars.
    # psycopg2 expects postgresql://...
    if url.startswith("postgresql+psycopg2://"):
        return "postgresql://" + url[len("postgresql+psycopg2://") :]
    return url


def get_registry_db_url() -> str | None:
    url = os.environ.get("CHAMPION_REGISTRY_DATABASE_URL")
    return _normalize_pg_url(url) if url else None


@dataclass(frozen=True)
class ChampionRow:
    id: int
    model_name: str
    target: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_run_id: str
    artifact_uri: str | None
    model_artifact_path: str | None
    preprocessing_artifact_path: str | None
    metric_name: str | None
    metric_value: float | None
    params_json: dict[str, Any] | None
    registered_at: str


DDL = """
CREATE TABLE IF NOT EXISTS champion_models (
  id BIGSERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  target TEXT NOT NULL,
  mlflow_tracking_uri TEXT NOT NULL,
  mlflow_experiment_name TEXT NOT NULL,
  mlflow_run_id TEXT NOT NULL,
  artifact_uri TEXT,
  model_artifact_path TEXT,
  preprocessing_artifact_path TEXT,
  metric_name TEXT,
  metric_value DOUBLE PRECISION,
  params_json JSONB,
  registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  is_active BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS champion_models_active_idx
  ON champion_models(model_name, target)
  WHERE is_active;
"""


def connect():
    import psycopg2

    url = get_registry_db_url()
    if not url:
        raise ValueError("CHAMPION_REGISTRY_DATABASE_URL is not set")
    return psycopg2.connect(url)


def ensure_schema(conn) -> None:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(DDL)


def upsert_active_champion(
    conn,
    *,
    model_name: str,
    target: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_run_id: str,
    artifact_uri: str | None,
    model_artifact_path: str | None,
    preprocessing_artifact_path: str | None,
    metric_name: str | None,
    metric_value: float | None,
    params_json: dict[str, Any] | None,
) -> int:
    """Deactivate previous active row and insert the new champion as active."""
    ensure_schema(conn)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE champion_models SET is_active=FALSE WHERE model_name=%s AND target=%s AND is_active=TRUE",
            (model_name, target),
        )
        cur.execute(
            """
            INSERT INTO champion_models (
              model_name, target,
              mlflow_tracking_uri, mlflow_experiment_name,
              mlflow_run_id, artifact_uri,
              model_artifact_path, preprocessing_artifact_path,
              metric_name, metric_value, params_json,
              is_active
            ) VALUES (
              %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE
            ) RETURNING id
            """,
            (
                model_name,
                target,
                mlflow_tracking_uri,
                mlflow_experiment_name,
                mlflow_run_id,
                artifact_uri,
                model_artifact_path,
                preprocessing_artifact_path,
                metric_name,
                metric_value,
                params_json,
            ),
        )
        new_id = int(cur.fetchone()[0])
    conn.commit()
    return new_id


def fetch_active_champion(conn, *, model_name: str, target: str) -> ChampionRow | None:
    ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, model_name, target, mlflow_tracking_uri, mlflow_experiment_name,
                   mlflow_run_id, artifact_uri, model_artifact_path, preprocessing_artifact_path,
                   metric_name, metric_value, params_json, registered_at
            FROM champion_models
            WHERE model_name=%s AND target=%s AND is_active=TRUE
            ORDER BY registered_at DESC
            LIMIT 1
            """,
            (model_name, target),
        )
        row = cur.fetchone()
        if not row:
            return None
    return ChampionRow(
        id=int(row[0]),
        model_name=str(row[1]),
        target=str(row[2]),
        mlflow_tracking_uri=str(row[3]),
        mlflow_experiment_name=str(row[4]),
        mlflow_run_id=str(row[5]),
        artifact_uri=row[6],
        model_artifact_path=row[7],
        preprocessing_artifact_path=row[8],
        metric_name=row[9],
        metric_value=float(row[10]) if row[10] is not None else None,
        params_json=row[11],
        registered_at=str(row[12]),
    )
