"""Load the three raw sources and merge them on `sid`.

Uses the same outer-join strategy as the `preprocessing` task in
airflow/dags/copd_ingestion.py, so the row set here matches what that task
sees (demographics: 2620 rows, imaging/spirometry: 2610 rows each — outer join
keeps the 10 demographics-only rows with NaN imaging/spirometry columns).
"""

from __future__ import annotations

import pandas as pd

from fetch_raw_data import fetch_all


def load_merged(force_refetch: bool = False) -> pd.DataFrame:
    paths = fetch_all(force=force_refetch)

    demographics = pd.read_csv(paths["demographics"])
    imaging = pd.read_json(paths["imaging"])
    spirometry = pd.read_html(paths["spirometry"])[0]

    merged = demographics.merge(imaging, on="sid", how="outer", suffixes=("_demographics", "_imaging"))
    merged = merged.merge(spirometry, on="sid", how="outer", suffixes=("", "_spirometry"))
    return merged


if __name__ == "__main__":
    df = load_merged()
    print(df.shape)
    print(df.head())
