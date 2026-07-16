"""Exploratory data analysis for the merged COPD dataset.

Deliberately avoids boxplots (per team decision — histograms/KDE are easier to
read). Produces:
  - a missingness report
  - a histogram + KDE per numeric column
  - a correlation heatmap
  - a printed flag on columns that leak information about the FEV1 target
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from load_and_merge import load_merged
from paths import PLOTS_DIR

# Target is still undecided by the team -- see feature_engineering.py's
# TARGET_CANDIDATES for the three live options (fev1, fev1_phase2, gold_copd)
# and exactly which columns each one needs dropped from the feature set.
CANDIDATE_TARGETS = {
    "fev1": "baseline FEV1 (current lung function)",
    "fev1_phase2": "FEV1 five years after baseline (per the data dictionary -- a longitudinal follow-up value, NOT a repeat of the baseline test)",
    "fev1_fvc_ratio": "ratio -- thresholded at 0.70 for the GOLD classification candidate",
}
GOLD_THRESHOLD = 0.70


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "sid"]


def missingness_report(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().sort_values(ascending=False)


def plot_distributions(df: pd.DataFrame, columns: list[str], out_dir=None) -> None:
    out_dir = out_dir or (PLOTS_DIR / "distributions")
    out_dir.mkdir(parents=True, exist_ok=True)
    for col in columns:
        values = df[col].dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=30, density=True, alpha=0.6, color="#4C72B0", edgecolor="white")
        try:
            values.plot(kind="kde", ax=ax, color="#C44E52", linewidth=2)
        except Exception:
            pass  # KDE can fail on near-constant columns; the histogram alone is still useful
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}.png", dpi=120)
        plt.close(fig)
    print(f"[eda] saved {len(columns)} distribution plots -> {out_dir}")


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], out_path=None) -> None:
    out_path = out_path or (PLOTS_DIR / "correlation_heatmap.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 0.5), max(5, len(columns) * 0.5)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix (numeric columns)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[eda] saved correlation heatmap -> {out_path}")


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_merged()

    print("=== shape ===")
    print(df.shape)

    print("\n=== missingness (fraction NaN per column) ===")
    print(missingness_report(df))

    num_cols = numeric_columns(df)
    print(f"\n=== numeric columns ({len(num_cols)}) ===")
    print(num_cols)

    print("\n=== candidate targets (not yet decided by the team) ===")
    for col, desc in CANDIDATE_TARGETS.items():
        series = df[col]
        print(
            f"{col} ({desc}): mean={series.mean():.3f}, std={series.std():.3f}, "
            f"min={series.min():.3f}, max={series.max():.3f}"
        )
    gold_share = (df["fev1_fvc_ratio"] < GOLD_THRESHOLD).mean()
    print(f"Share of rows below GOLD threshold (fev1_fvc_ratio < {GOLD_THRESHOLD}): {gold_share:.1%}")

    print("\n=== leakage note ===")
    print(
        "fev1, fev1_phase2, and fev1_fvc_ratio are three candidate targets, not "
        "three independent features. See feature_engineering.TARGET_CANDIDATES "
        "for exactly which columns must be dropped from the feature set for "
        "each choice -- the correct exclusion set is different per target."
    )

    plot_distributions(df, num_cols)
    plot_correlation_heatmap(df, num_cols)

    return df


if __name__ == "__main__":
    run()
