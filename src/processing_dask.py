from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import dask.dataframe as dd


TIME_BIN_DAYS = 0.05
ANOMALY_Z_THRESHOLD = 3.0
ROLLING_WINDOW = 25


def _to_numeric_dask(df: dd.DataFrame, column: str) -> dd.Series:
    """Safe numeric conversion for Dask dataframe columns."""
    return dd.to_numeric(df[column], errors="coerce")


def _compute_rolling_and_anomalies(aggregated: pd.DataFrame) -> pd.DataFrame:
    """
    Final visualization-oriented postprocessing.

    Dask performs the heavy tabular part: reading, cleaning, filtering,
    normalization and aggregation. Rolling smoothing is computed on the
    aggregated result because it is already reduced and small enough for
    deterministic plotting.
    """
    if aggregated.empty:
        return aggregated

    aggregated = aggregated.sort_values(["object_id", "quarter", "time_bin"]).reset_index(drop=True)

    aggregated["rolling_flux"] = (
        aggregated.groupby("object_id", group_keys=False)["flux_norm"]
        .transform(lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=3, center=True).median())
    )
    aggregated["rolling_flux"] = aggregated["rolling_flux"].fillna(aggregated["flux_norm"])

    aggregated["residual"] = aggregated["flux_norm"] - aggregated["rolling_flux"]
    residual_std = aggregated.groupby("object_id")["residual"].transform("std").replace(0, np.nan)

    anomaly_by_residual = (aggregated["residual"].abs() > ANOMALY_Z_THRESHOLD * residual_std).fillna(False)
    anomaly_from_raw_bins = aggregated["anomaly_count"].fillna(0) > 0
    aggregated["is_anomaly"] = anomaly_by_residual | anomaly_from_raw_bins

    return aggregated


def process_dask_dataset(
    input_file: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    npartitions: int | None = None,
) -> dict[str, Any]:
    """
    Dask implementation of the common Kepler processing pipeline.

    Pipeline:
    1. Read Parquet through Dask.
    2. Convert numeric columns.
    3. Remove invalid rows.
    4. Prefer records with quality == 0 when such records exist.
    5. Normalize flux by object_id using group statistics.
    6. Bin observations by time.
    7. Aggregate rows for visualization.
    8. Compute rolling trend and anomaly flags on the reduced result.
    9. Save processed result to Parquet.
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    ddf = dd.read_parquet(input_file)

    if npartitions is not None and npartitions > 0:
        ddf = ddf.repartition(npartitions=npartitions)

    required_columns = ["object_id", "time", "flux", "flux_err", "quality"]
    missing = [column for column in required_columns if column not in ddf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(ddf.columns)}")

    rows_input = int(ddf.shape[0].compute())

    for column in ["time", "flux", "flux_err", "quality"]:
        ddf[column] = _to_numeric_dask(ddf, column)

    if "quarter" not in ddf.columns:
        ddf["quarter"] = -1

    if "source_file" not in ddf.columns:
        ddf["source_file"] = ""

    ddf = ddf.dropna(subset=["object_id", "time", "flux"])
    ddf["quality"] = ddf["quality"].fillna(0)

    # Kepler quality flag: 0 usually means no known quality issue.
    # If quality filtering removes all rows, keep the cleaned unfiltered dataset.
    good_quality = ddf[ddf["quality"].astype("int64") == 0]
    good_quality_count = int(good_quality.shape[0].compute())
    if good_quality_count > 0:
        ddf = good_quality

    flux_err_mean = ddf["flux_err"].mean().compute()
    if pd.isna(flux_err_mean):
        flux_err_mean = 0.0
    ddf["flux_err"] = ddf["flux_err"].fillna(float(flux_err_mean))

    # Compute per-object normalization statistics and merge them back.
    stats = ddf.groupby("object_id")["flux"].agg(["mean", "std"]).compute().reset_index()
    stats = stats.rename(columns={"mean": "flux_mean", "std": "flux_std"})
    stats["flux_std"] = stats["flux_std"].replace(0, np.nan)
    stats_ddf = dd.from_pandas(stats, npartitions=1)

    ddf = ddf.merge(stats_ddf, on="object_id", how="left")
    ddf["flux_norm"] = ((ddf["flux"] - ddf["flux_mean"]) / ddf["flux_std"]).fillna(0)

    # Preliminary raw anomaly marker. Final anomaly is refined after aggregation.
    ddf["is_anomaly_prelim"] = ddf["flux_norm"].abs() > ANOMALY_Z_THRESHOLD

    # Kepler time is measured in days. 0.05 day is about 72 minutes.
    ddf["time_bin"] = (ddf["time"] // TIME_BIN_DAYS) * TIME_BIN_DAYS

    group_columns = ["object_id", "quarter", "time_bin"]
    aggregated_ddf = ddf.groupby(group_columns).agg(
        {
            "flux_norm": "mean",
            "flux_err": "mean",
            "flux": "count",
            "is_anomaly_prelim": "sum",
        }
    )

    aggregated = aggregated_ddf.compute().reset_index()
    aggregated = aggregated.rename(
        columns={
            "flux": "points_count",
            "is_anomaly_prelim": "anomaly_count",
        }
    )

    # Keep output schema compatible with existing visualizers.
    aggregated["points_count"] = pd.to_numeric(aggregated["points_count"], errors="coerce").fillna(0).astype(int)
    aggregated["anomaly_count"] = pd.to_numeric(aggregated["anomaly_count"], errors="coerce").fillna(0).astype(int)
    aggregated["source_files"] = 0
    aggregated["dataset"] = dataset_name
    aggregated["processing_method"] = "dask"

    aggregated = _compute_rolling_and_anomalies(aggregated)

    output_file = output_dir / f"{dataset_name}_dask_processed.parquet"
    aggregated.to_parquet(output_file, index=False)

    rows_after_cleaning = int(ddf.shape[0].compute())

    return {
        "dataset": dataset_name,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "rows_input": int(rows_input),
        "rows_after_cleaning": int(rows_after_cleaning),
        "rows_output": int(len(aggregated)),
        "objects": int(aggregated["object_id"].nunique()),
        "quarters": int(aggregated["quarter"].nunique()),
        "anomaly_rows": int(aggregated["is_anomaly"].sum()),
    }
