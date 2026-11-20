"""PSI and later-period monitoring starter."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[1]


def psi(expected, actual, bins=10):
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)
    # Quantile bins based on baseline; collapse duplicate edges if needed.
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    e_counts = np.histogram(expected, bins=edges)[0]
    a_counts = np.histogram(actual, bins=edges)[0]
    e_pct = np.maximum(e_counts / max(e_counts.sum(), 1), 1e-6)
    a_pct = np.maximum(a_counts / max(a_counts.sum(), 1), 1e-6)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def main():
    train = pd.read_csv(ROOT / "data/train.csv")
    later = pd.read_csv(ROOT / "data/later.csv")
    # TODO: load cleaned versions and the v1 model; implement per-outlet MAE,
    # temporal split (first 70%/last 30%), retraining and promotion decision.
    for col in NUMERIC_FEATURES:
        print(col, psi(train[col], later[col], bins=10))

if __name__ == "__main__":
    main()
