#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_HOME_DIR="${AIRFLOW_HOME:-$HOME/airflow}"

echo "Resetting Airflow state in: $AIRFLOW_HOME_DIR"
rm -rf "$AIRFLOW_HOME_DIR"

rm -rf "$ROOT_DIR/data/raw" "$ROOT_DIR/data/preprocessed" "$ROOT_DIR/data/artifacts"
mkdir -p "$ROOT_DIR/data/raw" "$ROOT_DIR/data/preprocessed" "$ROOT_DIR/data/artifacts"

echo "Airflow state and project output folders have been reset."
