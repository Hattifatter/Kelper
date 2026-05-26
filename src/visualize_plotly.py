from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def build_plotly_visualization(
    processed_file: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    processing_method: str,
    max_points: int = 10000,
) -> dict[str, Any]:
    """
    Builds an interactive HTML time-series chart from processed Kepler data.
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

    anomalies = subset[subset["is_anomaly"] == True]  # noqa: E712

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=subset["time_bin"],
            y=subset["flux_norm"],
            mode="lines",
            name="Normalized flux",
            hovertemplate="Time: %{x}<br>Flux: %{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=subset["time_bin"],
            y=subset["rolling_flux"],
            mode="lines",
            name="Rolling trend",
            hovertemplate="Time: %{x}<br>Trend: %{y}<extra></extra>",
        )
    )

    if len(anomalies) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomalies["time_bin"],
                y=anomalies["flux_norm"],
                mode="markers",
                name="Anomaly",
                hovertemplate="Time: %{x}<br>Anomaly flux: %{y}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Kepler light curve: {object_id} | {dataset_name} | {processing_method} + Plotly",
        xaxis_title="Time, days",
        yaxis_title="Normalized flux",
        hovermode="x unified",
        template="plotly_white",
    )

    output_file = output_dir / f"{dataset_name}_{processing_method}_plotly.html"
    fig.write_html(output_file, include_plotlyjs="cdn")

    return {
        "visualization_method": "plotly",
        "visualization_type": "interactive_html",
        "output_file": str(output_file),
        "object_id": str(object_id),
        "points_plotted": int(len(subset)),
        "anomaly_points_plotted": int(len(anomalies)),
    }
