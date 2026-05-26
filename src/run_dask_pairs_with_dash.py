from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from metrics import measure_step
from processing_dask import process_dask_dataset
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


def _save_combined_metrics(dask_metrics: pd.DataFrame) -> Path:
    """Combines existing Pandas metrics with new Dask metrics for chapter 3 tables."""
    all_metrics: list[pd.DataFrame] = []

    pandas_metrics_file = METRICS_DIR / "pandas_pair_metrics_with_dash.csv"
    if pandas_metrics_file.exists():
        all_metrics.append(pd.read_csv(pandas_metrics_file))

    all_metrics.append(dask_metrics)

    combined = pd.concat(all_metrics, ignore_index=True, sort=False)
    combined_file = METRICS_DIR / "all_pair_metrics_latest.csv"
    combined.to_csv(combined_file, index=False, encoding="utf-8-sig")
    return combined_file


def run() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    print("=== DASK PAIR EXPERIMENTS WITH DASH ===")

    for dataset_name, input_file in DATASETS.items():
        print(f"\n--- Dataset: {dataset_name} ---")

        if not input_file.exists():
            print(f"SKIP: file not found: {input_file}")
            rows.append({
                "dataset": dataset_name,
                "processing_method": "dask",
                "visualization_method": "",
                "pair": "dask +",
                "status": "error",
                "error": f"Input file not found: {input_file}",
            })
            continue

        processed_meta, processing_metrics = measure_step(
            process_dask_dataset,
            input_file=input_file,
            output_dir=PROCESSED_RESULTS_DIR,
            dataset_name=dataset_name,
        )

        if processing_metrics.status != "success":
            print(f"Processing error: {processing_metrics.error}")
            rows.append({
                "dataset": dataset_name,
                "processing_method": "dask",
                "visualization_method": "",
                "pair": "dask +",
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
            f"Dask processed: rows_in={processed_meta['rows_input']}, "
            f"rows_out={processed_meta['rows_output']}, "
            f"time={processing_metrics.elapsed_sec}s"
        )

        for visualization_method, visualizer in VISUALIZERS.items():
            print(f"Building visualization: dask + {visualization_method}")

            viz_meta, viz_metrics = measure_step(
                visualizer,
                processed_file=processed_file,
                output_dir=PLOTS_DIR,
                dataset_name=dataset_name,
                processing_method="dask",
            )

            total_time = round(processing_metrics.elapsed_sec + viz_metrics.elapsed_sec, 6)

            row = {
                "dataset": dataset_name,
                "processing_method": "dask",
                "visualization_method": visualization_method,
                "pair": f"dask + {visualization_method}",
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
    metrics_file = METRICS_DIR / "dask_pair_metrics_with_dash.csv"
    metrics.to_csv(metrics_file, index=False, encoding="utf-8-sig")

    latest_metrics_file = METRICS_DIR / "dask_pair_metrics_latest.csv"
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
    print(f"Latest Dask metrics: {latest_metrics_file}")
    print(f"Combined Pandas + Dask metrics: {combined_file}")
    print(f"Plots saved:    {PLOTS_DIR}")
    print(f"Dash data saved:{PROJECT_ROOT / 'results' / 'dash_data'}")
    print("\nTo open the dashboard, run:")
    print("python src\\app_dash.py")
    print("Then open: http://127.0.0.1:8050")


if __name__ == "__main__":
    run()
