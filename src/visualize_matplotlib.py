from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_matplotlib_visualization(
    processed_file: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    processing_method: str,
    max_points: int = 5000,
) -> dict[str, Any]:
    """
    Builds a static PNG time-series chart from processed Kepler data.
    """
    processed_file = Path(processed_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_file)

    if df.empty:
        raise ValueError(f"Processed file is empty: {processed_file}")

    object_id = df["object_id"].value_counts().idxmax()
    subset = df[df["object_id"] == object_id].sort_values("time_bin").copy()

    if len(subset) > max_points:
        indices = np.linspace(0, len(subset) - 1, max_points).astype(int)
        subset = subset.iloc[indices].copy()

    output_file = output_dir / f"{dataset_name}_{processing_method}_matplotlib.png"

    fig, ax = plt.subplots(figsize=(12, 5))

    # Draw the main signal clearly. The trend is dashed to avoid hiding it.
    ax.plot(
        subset["time_bin"],
        subset["flux_norm"],
        linewidth=1.4,
        alpha=0.85,
        label="Normalized flux",
        zorder=3,
    )

    ax.plot(
        subset["time_bin"],
        subset["rolling_flux"],
        linewidth=2.0,
        linestyle="--",
        alpha=0.75,
        label="Rolling trend",
        zorder=2,
    )

    anomalies = subset[subset["is_anomaly"] == True]  # noqa: E712
    if len(anomalies) > 0:
        ax.scatter(
            anomalies["time_bin"],
            anomalies["flux_norm"],
            s=22,
            marker="o",
            edgecolors="black",
            linewidths=0.4,
            label="Anomaly",
            zorder=4,
        )

    ax.set_title(f"Kepler light curve: {object_id} | {dataset_name} | {processing_method} + Matplotlib")
    ax.set_xlabel("Time, days")
    ax.set_ylabel("Normalized flux")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return {
        "visualization_method": "matplotlib",
        "visualization_type": "static_png",
        "output_file": str(output_file),
        "object_id": str(object_id),
        "points_plotted": int(len(subset)),
        "anomaly_points_plotted": int(len(anomalies)),
    }
