"""End-to-end driver for the EDA / feature-engineering / synthetic-data work.

Staging area only -- output lands in eda_feature_engineering/output/, NOT in
data/preprocessed/ (that's where the ingestion DAG's own `preprocessing` task
writes today). Keeping these separate until the team confirms whether this
logic gets merged into that task, added as its own DAG step, or kept
standalone. See README.md for the open questions.
"""

from __future__ import annotations

import pandas as pd

import eda
import feature_engineering as fe
import synthetic_data as sd
from paths import OUTPUT_DIR

N_SYNTHETIC_ROWS = 500


def main() -> None:
    real_df = eda.run()  # loads, merges, prints EDA, saves distribution/correlation plots

    model_ready, manifest = fe.build_model_ready(real_df)
    fe.write_manifest(manifest)

    numeric_cols = [c for c in model_ready.select_dtypes(include="number").columns if c != "sid"]
    categorical_cols = list(fe.CATEGORICAL_COLUMNS)

    synthetic_df = sd.generate_synthetic_rows(
        model_ready, numeric_cols, categorical_cols, n_rows=N_SYNTHETIC_ROWS
    )
    sd.plot_real_vs_synthetic(model_ready, synthetic_df, numeric_cols)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_only = model_ready.assign(**{sd.IS_SYNTHETIC_COL: False})
    real_only_path = OUTPUT_DIR / "centralized_dataset_real.csv"
    real_only.to_csv(real_only_path, index=False)
    print(f"[pipeline] wrote real-only centralized CSV -> {real_only_path}")

    with_synthetic = pd.concat([real_only, synthetic_df], ignore_index=True)
    with_synthetic_path = OUTPUT_DIR / "centralized_dataset_with_synthetic.csv"
    with_synthetic.to_csv(with_synthetic_path, index=False)
    print(f"[pipeline] wrote real+synthetic centralized CSV -> {with_synthetic_path}")

    print(
        "\n[pipeline] NOTE: which of these two CSVs is the real 'handoff to Modeling' "
        "deliverable is still an open question -- see README.md."
    )


if __name__ == "__main__":
    main()
