"""Generate synthetic rows that mimic the real dataset's per-column distributions.

IMPORTANT limitation: each column is sampled independently (marginal-only
synthesis). Real relationships between columns -- e.g. taller people tend to
weigh more, FEV1 tends to fall with age -- are NOT preserved in the synthetic
rows. That's fine for padding out an otherwise-small dataset or stress-testing
plots, but synthetic rows should never end up in a validation/test split used
to judge a real model, since they don't represent real physiology.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from paths import PLOTS_DIR

IS_SYNTHETIC_COL = "is_synthetic"


def _sample_numeric_column(values: pd.Series, n: int, rng: np.random.Generator) -> np.ndarray:
    clean = values.dropna().to_numpy()
    if len(clean) < 2 or np.allclose(clean, clean[0]):
        # Not enough spread to fit a KDE -- fall back to resampling real values.
        return rng.choice(clean if len(clean) else np.array([np.nan]), size=n)
    kde = gaussian_kde(clean)
    seed = int(rng.integers(0, 2**31 - 1))
    return kde.resample(n, seed=seed).flatten()


def _sample_categorical_column(values: pd.Series, n: int, rng: np.random.Generator) -> np.ndarray:
    counts = values.dropna().value_counts(normalize=True)
    if counts.empty:
        return np.array([np.nan] * n, dtype=object)
    return rng.choice(counts.index.to_numpy(), size=n, p=counts.to_numpy())


def generate_synthetic_rows(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    n_rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    synthetic = pd.DataFrame(index=range(n_rows))

    for col in numeric_columns:
        synthetic[col] = _sample_numeric_column(df[col], n_rows, rng)
    for col in categorical_columns:
        synthetic[col] = _sample_categorical_column(df[col], n_rows, rng)

    synthetic["sid"] = [f"SYN{i:05d}" for i in range(n_rows)]
    synthetic[IS_SYNTHETIC_COL] = True
    return synthetic


def plot_real_vs_synthetic(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    columns: list[str],
    out_dir=None,
) -> None:
    out_dir = out_dir or (PLOTS_DIR / "synthetic_vs_real")
    out_dir.mkdir(parents=True, exist_ok=True)
    for col in columns:
        real_values = real_df[col].dropna()
        synth_values = synthetic_df[col].dropna()
        if real_values.empty or synth_values.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(real_values, bins=30, density=True, alpha=0.5, label="real", color="#4C72B0")
        ax.hist(synth_values, bins=30, density=True, alpha=0.5, label="synthetic", color="#DD8452")
        ax.set_title(f"Real vs synthetic: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}.png", dpi=120)
        plt.close(fig)
    print(f"[synthetic] saved real-vs-synthetic overlays -> {out_dir}")
