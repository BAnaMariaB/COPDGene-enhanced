"""Fetch the three raw COPD sources into eda_feature_engineering/raw/.

Independent of the Airflow ingestion DAG so EDA can be iterated on without a
running Airflow instance. Mirrors the source URLs in
airflow/dags/copd_ingestion.py — if that dict changes, update this one too.
"""

from __future__ import annotations

import requests

from paths import RAW_DIR

SOURCES = {
    "demographics": (
        "https://raw.githubusercontent.com/khasenst/datasets_teaching/refs/heads/main/copd_data_demographics.csv",
        "csv",
    ),
    "imaging": (
        "https://raw.githubusercontent.com/khasenst/datasets_teaching/refs/heads/main/copd_data_imaging.json",
        "json",
    ),
    "spirometry": (
        "https://raw.githubusercontent.com/khasenst/datasets_teaching/refs/heads/main/copd_data_spirometry.html",
        "html",
    ),
}


def fetch_all(force: bool = False) -> dict[str, str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, (url, ext) in SOURCES.items():
        dest = RAW_DIR / f"{name}.{ext}"
        if dest.exists() and not force:
            print(f"[fetch] {name}: already have {dest}, skipping (force=True to re-download)")
        else:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"[fetch] {name}: {len(resp.content)} bytes -> {dest}")
        paths[name] = str(dest)
    return paths


if __name__ == "__main__":
    fetch_all()
