#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_HOME_DIR="${AIRFLOW_HOME:-$HOME/airflow}"
AIRFLOW_BIN_DIR="${AIRFLOW_BIN:-$HOME/airflow/airflow_venv/bin}"

export AIRFLOW_HOME="$AIRFLOW_HOME_DIR"
export AIRFLOW__CORE__DAGS_FOLDER="$ROOT_DIR/airflow/dags"
export COPD_RAW_ROOT="$ROOT_DIR/data/raw"
export COPD_PREPROCESSED_ROOT="$ROOT_DIR/data/preprocessed"
export COPD_ARTIFACT_ROOT="$ROOT_DIR/data/artifacts"
export PATH="$AIRFLOW_BIN_DIR:$PATH"

mkdir -p "$AIRFLOW_HOME"
mkdir -p "$COPD_RAW_ROOT" "$COPD_PREPROCESSED_ROOT" "$COPD_ARTIFACT_ROOT"

if ! command -v airflow >/dev/null 2>&1; then
  echo "airflow is not available in $AIRFLOW_BIN_DIR" >&2
  exit 1
fi

airflow db migrate
exec airflow standalone
