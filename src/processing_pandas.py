from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_zscore(series: pd.Series) -> pd.Series:
    mean_value = series.mean()
    std_value = series.std()

    if pd.isna(std_value) or std_value == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - mean_value) / std_value


def process_pandas_dataset(
    input_file: str | Path,
    output_dir: str | Path,
    dataset_name: str,
) -> dict[str, Any]:
    """
    Pandas implementation of the common processing pipeline.

    Pipeline:
    1. Read Parquet.
    2. Convert numeric columns.
    3. Remove invalid rows.
    4. Prefer good quality records where possible.
    5. Normalize flux by object_id.
    6. Compute rolling trend.
    7. Detect anomaly points.
    8. Aggregate by time bins.
    9. Save processed result to Parquet.
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_file)
    rows_input = len(df)

    required_columns = ["object_id", "time", "flux", "flux_err", "quality"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    df = df.copy()

    for column in ["time", "flux", "flux_err", "quality"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["object_id", "time", "flux"])

    if "quarter" not in df.columns:
        df["quarter"] = -1

    if "source_file" not in df.columns:
        df["source_file"] = ""

    df["quality"] = df["quality"].fillna(0)

    # Kepler quality flag: 0 usually means no known quality issue.
    # If filtering removes all rows, keep the cleaned unfiltered dataset.
    good_quality = df[df["quality"].astype("int64") == 0].copy()
    if len(good_quality) > 0:
        df = good_quality

    df["flux_err"] = df["flux_err"].fillna(df["flux_err"].median())
    df = df.sort_values(["object_id", "quarter", "time"]).reset_index(drop=True)

    df["flux_norm"] = df.groupby("object_id", group_keys=False)["flux"].transform(_safe_zscore)

    # Rolling trend for local smoothing. Window 25 is suitable for Kepler long cadence
    # in this educational benchmark: it keeps the signal shape but suppresses noise.
    df["rolling_flux"] = (
        df.groupby("object_id", group_keys=False)["flux_norm"]
        .transform(lambda s: s.rolling(window=25, min_periods=3, center=True).median())
    )
    df["rolling_flux"] = df["rolling_flux"].fillna(df["flux_norm"])

    df["residual"] = df["flux_norm"] - df["rolling_flux"]
    residual_std = df.groupby("object_id")["residual"].transform("std").replace(0, np.nan)
    df["is_anomaly"] = (df["residual"].abs() > 3 * residual_std).fillna(False)

    # Kepler time is measured in days. 0.05 day is about 72 minutes.
    # This keeps enough detail and reduces the amount of points for visualization.
    df["time_bin"] = np.floor(df["time"] / 0.05) * 0.05

    aggregated = (
        df.groupby(["object_id", "quarter", "time_bin"], as_index=False)
        .agg(
            flux_norm=("flux_norm", "mean"),
            rolling_flux=("rolling_flux", "mean"),
            flux_err=("flux_err", "mean"),
            points_count=("flux", "size"),
            anomaly_count=("is_anomaly", "sum"),
            source_files=("source_file", "nunique"),
        )
    )

    aggregated["is_anomaly"] = aggregated["anomaly_count"] > 0
    aggregated["dataset"] = dataset_name
    aggregated["processing_method"] = "pandas"

    output_file = output_dir / f"{dataset_name}_pandas_processed.parquet"
    aggregated.to_parquet(output_file, index=False)

    return {
        "dataset": dataset_name,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "rows_input": int(rows_input),
        "rows_after_cleaning": int(len(df)),
        "rows_output": int(len(aggregated)),
        "objects": int(aggregated["object_id"].nunique()),
        "quarters": int(aggregated["quarter"].nunique()),
        "anomaly_rows": int(aggregated["is_anomaly"].sum()),
    }
