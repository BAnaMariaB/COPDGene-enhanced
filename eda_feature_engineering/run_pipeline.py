"""End-to-end driver for the EDA / feature-engineering / synthetic-data work.

Staging area only -- output lands in eda_feature_engineering/output/, NOT in
data/preprocessed/ (that's where the ingestion DAG's own `preprocessing` task
writes today). Keeping these separate until the team confirms whether this
logic gets merged into that task, added as its own DAG step, or kept
standalone. See README.md for the open questions.

Team decision (2026-07-16): target is gold_copd (binary COPD classification
via the GOLD criterion) -- regression on fev1_phase2 hit an RMSE ceiling and
wasn't production-useful. This still emits target-ready CSVs for fev1 and
fev1_phase2 as rejected/reference candidates (they're the evidence for the
pivot), but centralized_dataset_target_gold_copd.csv is the one to model on.
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

    engineered, manifest = fe.engineer_features(real_df)
    fe.write_manifest(manifest)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full engineered dataset: every column, nothing target-specific dropped.
    # Useful for inspection and as the base for synthetic-data generation.
    full_real_path = OUTPUT_DIR / "centralized_dataset_full_real.csv"
    engineered.to_csv(full_real_path, index=False)
    print(f"[pipeline] wrote full engineered dataset -> {full_real_path}")

    numeric_cols = [c for c in engineered.select_dtypes(include="number").columns if c != "sid"]
    categorical_cols = list(fe.CATEGORICAL_COLUMNS)

    synthetic_df = sd.generate_synthetic_rows(
        engineered, numeric_cols, categorical_cols, n_rows=N_SYNTHETIC_ROWS
    )
    sd.plot_real_vs_synthetic(engineered, synthetic_df, numeric_cols)

    full_with_synthetic = pd.concat(
        [engineered.assign(**{sd.IS_SYNTHETIC_COL: False}), synthetic_df],
        ignore_index=True,
    )
    full_synthetic_path = OUTPUT_DIR / "centralized_dataset_full_with_synthetic.csv"
    full_with_synthetic.to_csv(full_synthetic_path, index=False)
    print(f"[pipeline] wrote full engineered + synthetic dataset -> {full_synthetic_path}")

    # Target-ready datasets: one per live candidate, each with the
    # leakage-appropriate columns dropped for THAT target specifically.
    target_manifest = {}
    for target_name in fe.TARGET_CANDIDATES:
        target_df, target_info = fe.build_dataset_for_target(engineered, target_name)
        target_path = OUTPUT_DIR / f"centralized_dataset_target_{target_name}.csv"
        target_df.to_csv(target_path, index=False)
        target_manifest[target_name] = {**target_info, "path": str(target_path)}
        print(f"[pipeline] wrote target-ready dataset for '{target_name}' -> {target_path}")

    fe.write_manifest(target_manifest, path=OUTPUT_DIR / "target_candidates_manifest.json")

    recommended_path = OUTPUT_DIR / f"centralized_dataset_target_{fe.RECOMMENDED_TARGET}.csv"
    print(
        f"\n[pipeline] RECOMMENDED TARGET: '{fe.RECOMMENDED_TARGET}' -> {recommended_path}\n"
        "  Train on this one. fev1 / fev1_phase2 CSVs are kept only as "
        "rejected/reference candidates -- see feature_engineering.py for why."
    )
    print(f"[pipeline] NOTE (not built yet): {fe.GOLD_STAGE_CASCADE_NOTE}")


if __name__ == "__main__":
    main()
