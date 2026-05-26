from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd


TIME_BIN_DAYS = 0.05
ANOMALY_Z_THRESHOLD = 3.0
ROLLING_WINDOW = 25


def _compute_rolling_and_anomalies(aggregated: pd.DataFrame) -> pd.DataFrame:
    """
    Final visualization-oriented postprocessing on the reduced table.

    Spark performs reading, cleaning, quality filtering, normalization and
    aggregation. Rolling smoothing is computed after aggregation because the
    result is compact and deterministic to visualize.
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


def _create_spark_session(app_name: str):
    """Creates a local Spark session suitable for the educational benchmark.

    Windows-specific note:
    Spark can fail when the computer hostname contains an underscore, for example
    ``Alina_Home``. In that case Spark builds an invalid internal RPC URL like
    ``spark://HeartbeatReceiver@Alina_Home:...``. The settings below force Spark
    to use localhost for the driver address.
    """
    # Ensure PySpark uses the active virtual environment Python.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # Force a valid local hostname/IP for Spark RPC on Windows.
    os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.local.hostname", "localhost")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _spark_rows_to_pandas(rows: list[Any]) -> pd.DataFrame:
    """Avoids pandas/Arrow version issues by converting collected rows explicitly."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([row.asDict(recursive=True) for row in rows])


def process_spark_dataset(
    input_file: str | Path,
    output_dir: str | Path,
    dataset_name: str,
) -> dict[str, Any]:
    """
    Spark implementation of the common Kepler processing pipeline.

    Pipeline:
    1. Start local Spark session.
    2. Read Parquet through Spark.
    3. Convert numeric columns and remove invalid rows.
    4. Prefer records with quality == 0 when possible.
    5. Normalize flux by object_id using Spark windowless group statistics.
    6. Bin observations by time.
    7. Aggregate rows for visualization.
    8. Compute rolling trend and final anomaly flags on the reduced result.
    9. Save processed result to Parquet.
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    spark = _create_spark_session(f"KeplerSparkProcessing_{dataset_name}")

    try:
        from pyspark.sql import functions as F

        sdf = spark.read.parquet(str(input_file))

        required_columns = ["object_id", "time", "flux", "flux_err", "quality"]
        missing = [column for column in required_columns if column not in sdf.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available columns: {sdf.columns}")

        rows_input = int(sdf.count())

        for column in ["time", "flux", "flux_err", "quality"]:
            sdf = sdf.withColumn(column, F.col(column).cast("double"))

        if "quarter" not in sdf.columns:
            sdf = sdf.withColumn("quarter", F.lit(-1).cast("int"))
        else:
            sdf = sdf.withColumn("quarter", F.col("quarter").cast("int"))

        if "source_file" not in sdf.columns:
            sdf = sdf.withColumn("source_file", F.lit(""))

        sdf = sdf.dropna(subset=["object_id", "time", "flux"])
        sdf = sdf.withColumn("quality", F.coalesce(F.col("quality"), F.lit(0.0)))

        good_quality = sdf.filter(F.col("quality").cast("long") == 0)
        good_quality_count = int(good_quality.count())
        if good_quality_count > 0:
            sdf = good_quality

        flux_err_mean_row = sdf.agg(F.mean("flux_err").alias("flux_err_mean")).first()
        flux_err_mean = flux_err_mean_row["flux_err_mean"] if flux_err_mean_row else None
        if flux_err_mean is None:
            flux_err_mean = 0.0
        sdf = sdf.withColumn("flux_err", F.coalesce(F.col("flux_err"), F.lit(float(flux_err_mean))))

        stats = sdf.groupBy("object_id").agg(
            F.mean("flux").alias("flux_mean"),
            F.stddev_samp("flux").alias("flux_std"),
        )

        sdf = sdf.join(stats, on="object_id", how="left")
        sdf = sdf.withColumn(
            "flux_norm",
            F.when((F.col("flux_std").isNull()) | (F.col("flux_std") == 0), F.lit(0.0))
            .otherwise((F.col("flux") - F.col("flux_mean")) / F.col("flux_std")),
        )

        sdf = sdf.withColumn("is_anomaly_prelim", F.abs(F.col("flux_norm")) > F.lit(ANOMALY_Z_THRESHOLD))
        sdf = sdf.withColumn("time_bin", F.floor(F.col("time") / F.lit(TIME_BIN_DAYS)) * F.lit(TIME_BIN_DAYS))

        aggregated_sdf = sdf.groupBy("object_id", "quarter", "time_bin").agg(
            F.mean("flux_norm").alias("flux_norm"),
            F.mean("flux_err").alias("flux_err"),
            F.count("flux").alias("points_count"),
            F.sum(F.col("is_anomaly_prelim").cast("int")).alias("anomaly_count"),
            F.countDistinct("source_file").alias("source_files"),
        )

        rows_after_cleaning = int(sdf.count())
        aggregated_rows = aggregated_sdf.collect()
        aggregated = _spark_rows_to_pandas(aggregated_rows)

        if aggregated.empty:
            raise RuntimeError("Spark aggregation returned empty result.")

        aggregated["object_id"] = aggregated["object_id"].astype(str)
        aggregated["quarter"] = pd.to_numeric(aggregated["quarter"], errors="coerce").fillna(-1).astype(int)
        aggregated["time_bin"] = pd.to_numeric(aggregated["time_bin"], errors="coerce")
        aggregated["flux_norm"] = pd.to_numeric(aggregated["flux_norm"], errors="coerce")
        aggregated["flux_err"] = pd.to_numeric(aggregated["flux_err"], errors="coerce")
        aggregated["points_count"] = pd.to_numeric(aggregated["points_count"], errors="coerce").fillna(0).astype(int)
        aggregated["anomaly_count"] = pd.to_numeric(aggregated["anomaly_count"], errors="coerce").fillna(0).astype(int)
        aggregated["source_files"] = pd.to_numeric(aggregated["source_files"], errors="coerce").fillna(0).astype(int)

        aggregated["dataset"] = dataset_name
        aggregated["processing_method"] = "spark"
        aggregated = _compute_rolling_and_anomalies(aggregated)

        output_file = output_dir / f"{dataset_name}_spark_processed.parquet"
        aggregated.to_parquet(output_file, index=False)

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

    finally:
        spark.stop()
