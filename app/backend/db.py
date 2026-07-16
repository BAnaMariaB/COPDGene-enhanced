"""PostgreSQL access for the model registry.

This service is a READER. The modeling DAG is the only writer.

Two targets now means two champions in service at once (`copd_diagnosis` and
`gold_stage`), so "the champion" is no longer a single row — it is one row per
target_name.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import DATABASE_URL

_engine: Engine | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5)
    return _engine


def ping() -> bool:
    try:
        with engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


CHAMPIONS_SQL = text(
    """
    SELECT model_name, mlflow_run_id, artifact_uri, target_name,
           class_labels, metrics, created_at
    FROM model_registry
    WHERE is_champion IS TRUE
    """
)


def _row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "model_name": row["model_name"],
        "mlflow_run_id": row["mlflow_run_id"],
        "artifact_uri": row["artifact_uri"],
        "target_name": row["target_name"],
        "class_labels": row["class_labels"],
        "metrics": row["metrics"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


def fetch_champions() -> dict[str, dict[str, Any]]:
    """Return {target_name: champion_row}. Empty dict if nothing is registered."""
    with engine().connect() as conn:
        rows = conn.execute(CHAMPIONS_SQL).mappings().all()
    return {r["target_name"]: _row_to_dict(r) for r in rows}
