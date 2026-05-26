from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from metrics import measure_step
from processing_spark import process_spark_dataset
from visualize_matplotlib import build_matplotlib_visualization
from visualize_plotly import build_plotly_visualization
from visualize_dash import build_dash_visualization


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
PROCESSED_RESULTS_DIR = PROJECT_ROOT / "results" / "processed"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

DATASETS = {
    "small": SAMPLES_DIR / "kepler_small.parquet",
    "medium": SAMPLES_DIR / "kepler_medium.parquet",
    "large": SAMPLES_DIR / "kepler_large.parquet",
}

VISUALIZERS = {
    "matplotlib": build_matplotlib_visualization,
    "plotly": build_plotly_visualization,
    "dash": build_dash_visualization,
}


def _load_metrics_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to read metrics file {path}: {exc}")
        return None


def _save_combined_metrics(spark_metrics: pd.DataFrame) -> Path:
    """Combines existing Pandas/Dask metrics with new Spark metrics."""
    all_metrics: list[pd.DataFrame] = []

    for filename in [
        "pandas_pair_metrics_with_dash.csv",
        "dask_pair_metrics_with_dash.csv",
    ]:
        loaded = _load_metrics_if_exists(METRICS_DIR / filename)
        if loaded is not None:
            all_metrics.append(loaded)

    all_metrics.append(spark_metrics)

    combined = pd.concat(all_metrics, ignore_index=True, sort=False)
    combined_file = METRICS_DIR / "all_pair_metrics_latest.csv"
    combined.to_csv(combined_file, index=False, encoding="utf-8-sig")

    snapshot_file = METRICS_DIR / "all_pair_metrics_with_spark.csv"
    combined.to_csv(snapshot_file, index=False, encoding="utf-8-sig")

    return combined_file


def run() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    print("=== SPARK PAIR EXPERIMENTS WITH DASH ===")
    print("NOTE: local Spark startup time is included in processing_time_sec.")

    for dataset_name, input_file in DATASETS.items():
        print(f"\n--- Dataset: {dataset_name} ---")

        if not input_file.exists():
            print(f"SKIP: file not found: {input_file}")
            rows.append({
                "dataset": dataset_name,
                "processing_method": "spark",
                "visualization_method": "",
                "pair": "spark +",
                "status": "error",
                "error": f"Input file not found: {input_file}",
            })
            continue

        processed_meta, processing_metrics = measure_step(
            process_spark_dataset,
            input_file=input_file,
            output_dir=PROCESSED_RESULTS_DIR,
            dataset_name=dataset_name,
            sample_interval_sec=0.10,
        )

        if processing_metrics.status != "success":
            print(f"Processing error: {processing_metrics.error}")
            rows.append({
                "dataset": dataset_name,
                "processing_method": "spark",
                "visualization_method": "",
                "pair": "spark +",
                "processing_time_sec": processing_metrics.elapsed_sec,
                "processing_memory_peak_mb": processing_metrics.memory_peak_mb,
                "visualization_time_sec": None,
                "visualization_memory_peak_mb": None,
                "total_time_sec": processing_metrics.elapsed_sec,
                "status": "error",
                "error": processing_metrics.error,
            })
            continue

        processed_file = Path(processed_meta["output_file"])
        print(
            f"Spark processed: rows_in={processed_meta['rows_input']}, "
            f"rows_out={processed_meta['rows_output']}, "
            f"time={processing_metrics.elapsed_sec}s"
        )

        for visualization_method, visualizer in VISUALIZERS.items():
            print(f"Building visualization: spark + {visualization_method}")

            viz_meta, viz_metrics = measure_step(
                visualizer,
                processed_file=processed_file,
                output_dir=PLOTS_DIR,
                dataset_name=dataset_name,
                processing_method="spark",
            )

            total_time = round(processing_metrics.elapsed_sec + viz_metrics.elapsed_sec, 6)

            row = {
                "dataset": dataset_name,
                "processing_method": "spark",
                "visualization_method": visualization_method,
                "pair": f"spark + {visualization_method}",
                "processing_time_sec": processing_metrics.elapsed_sec,
                "processing_memory_peak_mb": processing_metrics.memory_peak_mb,
                "visualization_time_sec": viz_metrics.elapsed_sec,
                "visualization_memory_peak_mb": viz_metrics.memory_peak_mb,
                "total_time_sec": total_time,
                "status": "success" if viz_metrics.status == "success" else "error",
                "error": viz_metrics.error,
                **processed_meta,
            }

            if viz_meta:
                row.update({
                    "visualization_file": viz_meta.get("output_file"),
                    "visualization_metadata_file": viz_meta.get("metadata_file"),
                    "visualization_type": viz_meta.get("visualization_type"),
                    "points_plotted": viz_meta.get("points_plotted"),
                    "anomaly_points_plotted": viz_meta.get("anomaly_points_plotted"),
                    "rows_before_downsample": viz_meta.get("rows_before_downsample"),
                })

            rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics_file = METRICS_DIR / "spark_pair_metrics_with_dash.csv"
    metrics.to_csv(metrics_file, index=False, encoding="utf-8-sig")

    latest_metrics_file = METRICS_DIR / "spark_pair_metrics_latest.csv"
    metrics.to_csv(latest_metrics_file, index=False, encoding="utf-8-sig")

    combined_file = _save_combined_metrics(metrics)

    print("\n=== PAIR METRICS ===")
    display_columns = [
        "dataset",
        "pair",
        "processing_time_sec",
        "visualization_time_sec",
        "total_time_sec",
        "processing_memory_peak_mb",
        "status",
    ]
    existing_display_columns = [column for column in display_columns if column in metrics.columns]
    print(metrics[existing_display_columns].to_string(index=False))

    print(f"\nMetrics saved: {metrics_file}")
    print(f"Latest Spark metrics: {latest_metrics_file}")
    print(f"Combined Pandas + Dask + Spark metrics: {combined_file}")
    print(f"Plots saved:    {PLOTS_DIR}")
    print(f"Dash data saved:{PROJECT_ROOT / 'results' / 'dash_data'}")
    print("\nTo open the dashboard, run:")
    print("python src\\app_dash.py")
    print("Then open: http://127.0.0.1:8050")


if __name__ == "__main__":
    run()
