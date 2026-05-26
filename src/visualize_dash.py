from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "dataset",
    "object_id",
    "quarter",
    "time_bin",
    "flux_norm",
    "rolling_flux",
    "flux_err",
    "points_count",
    "anomaly_count",
    "is_anomaly",
    "processing_method",
]


def _downsample_by_object(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Keeps the dashboard responsive while preserving points from each object."""
    if len(df) <= max_points:
        return df

    object_ids = sorted(df["object_id"].astype(str).unique().tolist())
    per_object_limit = max(1000, max_points // max(1, len(object_ids)))

    parts: list[pd.DataFrame] = []
    for object_id in object_ids:
        part = df[df["object_id"].astype(str) == object_id].sort_values("time_bin")
        if len(part) > per_object_limit:
            indices = np.linspace(0, len(part) - 1, per_object_limit).astype(int)
            part = part.iloc[indices].copy()
        parts.append(part)

    return pd.concat(parts, ignore_index=True) if parts else df.head(0).copy()


def build_dash_visualization(
    processed_file: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    processing_method: str,
    max_points: int = 50000,
) -> dict[str, Any]:
    """
    Prepares a dashboard-ready dataset for Dash.

    In the pair experiment this function represents the Dash visualization stage:
    it validates, reduces when needed, and saves data that the web interface will load.
    The interactive server itself is started separately by src/app_dash.py.
    """
    processed_file = Path(processed_file)
    output_dir = Path(output_dir)
    dash_data_dir = output_dir.parent / "dash_data"
    dash_data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_file)

    if df.empty:
        raise ValueError(f"Processed file is empty: {processed_file}")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Dash visualization: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["object_id"] = df["object_id"].astype(str)
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").fillna(-1).astype(int)
    df["time_bin"] = pd.to_numeric(df["time_bin"], errors="coerce")
    df["flux_norm"] = pd.to_numeric(df["flux_norm"], errors="coerce")
    df["rolling_flux"] = pd.to_numeric(df["rolling_flux"], errors="coerce")
    df["flux_err"] = pd.to_numeric(df["flux_err"], errors="coerce")
    df["points_count"] = pd.to_numeric(df["points_count"], errors="coerce").fillna(0).astype(int)
    df["anomaly_count"] = pd.to_numeric(df["anomaly_count"], errors="coerce").fillna(0).astype(int)
    df["is_anomaly"] = df["is_anomaly"].fillna(False).astype(bool)

    df = df.dropna(subset=["time_bin", "flux_norm"]).sort_values(["object_id", "quarter", "time_bin"])
    rows_before_downsample = len(df)
    df = _downsample_by_object(df, max_points=max_points)

    output_file = dash_data_dir / f"{dataset_name}_{processing_method}_dash_data.parquet"
    metadata_file = dash_data_dir / f"{dataset_name}_{processing_method}_dash_metadata.json"

    df.to_parquet(output_file, index=False)

    metadata = {
        "dataset": dataset_name,
        "processing_method": processing_method,
        "visualization_method": "dash",
        "visualization_type": "interactive_web_dashboard",
        "source_processed_file": str(processed_file),
        "output_file": str(output_file),
        "rows_before_downsample": int(rows_before_downsample),
        "rows_saved_for_dashboard": int(len(df)),
        "objects": sorted(df["object_id"].unique().tolist()),
        "quarters": sorted([int(value) for value in df["quarter"].dropna().unique().tolist()]),
        "anomaly_rows": int(df["is_anomaly"].sum()),
        "min_time": float(df["time_bin"].min()),
        "max_time": float(df["time_bin"].max()),
        "launch_command": "python src\\app_dash.py",
        "url": "http://127.0.0.1:8050",
    }

    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "visualization_method": "dash",
        "visualization_type": "interactive_web_dashboard",
        "output_file": str(output_file),
        "metadata_file": str(metadata_file),
        "object_id": ", ".join(metadata["objects"][:5]),
        "points_plotted": int(len(df)),
        "anomaly_points_plotted": int(df["is_anomaly"].sum()),
        "rows_before_downsample": int(rows_before_downsample),
    }
