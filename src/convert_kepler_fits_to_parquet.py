from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from config import (
    DATASET_DIRS,
    SAMPLES_DIR,
    PROCESSED_DIR,
    ANOMALY_Z_THRESHOLD,
    ensure_directories,
)


def _first_existing_column(data, candidates: list[str]) -> Optional[str]:
    """Возвращает первое существующее имя колонки FITS-таблицы."""
    names = [name.upper() for name in data.names]
    for candidate in candidates:
        if candidate.upper() in names:
            return candidate.upper()
    return None


def _safe_header_value(header, key: str, default=None):
    try:
        value = header.get(key, default)
        if value is None:
            return default
        return value
    except Exception:
        return default


def read_kepler_fits(path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Читает один Kepler light curve FITS-файл и приводит его к единой структуре:
    dataset, source_file, object_id, quarter, time, flux, flux_err, quality.

    Для flux приоритет отдаётся PDCSAP_FLUX, потому что это предварительно
    откорректированный поток. Если его нет, используется SAP_FLUX.
    """
    with fits.open(path, memmap=True) as hdul:
        primary_header = hdul[0].header
        table_header = hdul[1].header
        data = hdul[1].data

        time_col = _first_existing_column(data, ["TIME"])
        flux_col = _first_existing_column(data, ["PDCSAP_FLUX", "SAP_FLUX"])
        flux_err_col = _first_existing_column(data, ["PDCSAP_FLUX_ERR", "SAP_FLUX_ERR"])
        quality_col = _first_existing_column(data, ["SAP_QUALITY", "QUALITY"])

        if time_col is None or flux_col is None:
            raise ValueError(
                f"File {path.name}: required columns TIME and FLUX are not found. "
                f"Available columns: {data.names}"
            )

        object_id = (
            _safe_header_value(primary_header, "KEPLERID")
            or _safe_header_value(table_header, "KEPLERID")
            or path.stem.split("-")[0].replace("kplr", "")
        )

        quarter = (
            _safe_header_value(primary_header, "QUARTER")
            or _safe_header_value(table_header, "QUARTER")
            or -1
        )

        df = pd.DataFrame({
            "dataset": dataset_name,
            "source_file": path.name,
            "object_id": str(object_id),
            "quarter": int(quarter) if str(quarter).lstrip("-").isdigit() else -1,
            "time": np.asarray(data[time_col], dtype="float64"),
            "flux": np.asarray(data[flux_col], dtype="float64"),
            "flux_err": (
                np.asarray(data[flux_err_col], dtype="float64")
                if flux_err_col is not None
                else np.full(len(data), np.nan, dtype="float64")
            ),
            "quality": (
                np.asarray(data[quality_col], dtype="float64")
                if quality_col is not None
                else np.zeros(len(data), dtype="float64")
            ),
        })

    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет первичную подготовку:
    - удаляет строки без времени и потока;
    - приводит числовые поля;
    - заполняет ошибки измерений;
    - нормализует поток внутри каждого объекта;
    - добавляет признак аномалии.
    """
    df = df.copy()

    numeric_columns = ["time", "flux", "flux_err", "quality"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["time", "flux"])

    if df.empty:
        return df

    if df["flux_err"].notna().any():
        df["flux_err"] = df["flux_err"].fillna(df["flux_err"].median())
    else:
        df["flux_err"] = df["flux_err"].fillna(0)

    df["quality"] = df["quality"].fillna(0)

    def normalize_flux(series: pd.Series) -> pd.Series:
        mean_value = series.mean()
        std_value = series.std()
        if pd.isna(std_value) or std_value == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - mean_value) / std_value

    df["flux_norm"] = df.groupby("object_id", group_keys=False)["flux"].transform(normalize_flux)
    df["is_anomaly"] = df["flux_norm"].abs() > ANOMALY_Z_THRESHOLD

    df = df.sort_values(["object_id", "quarter", "time"]).reset_index(drop=True)

    return df


def convert_dataset(dataset_name: str, raw_dir: Path) -> dict:
    fits_files = sorted(list(raw_dir.glob("*.fits")) + list(raw_dir.glob("*.FITS")))

    if not fits_files:
        print(f"[WARN] Dataset '{dataset_name}': no FITS files in {raw_dir}")
        return {
            "dataset": dataset_name,
            "fits_files": 0,
            "rows": 0,
            "objects": 0,
            "quarters": 0,
            "anomaly_rows": 0,
            "output_file": "",
            "size_mb": 0.0,
            "status": "no_files",
        }

    parts = []
    errors = []

    print(f"\n=== CONVERT DATASET: {dataset_name.upper()} ===")
    print(f"Input directory: {raw_dir}")
    print(f"FITS files: {len(fits_files)}")

    for path in tqdm(fits_files, desc=f"{dataset_name} FITS"):
        try:
            part = read_kepler_fits(path, dataset_name)
            if not part.empty:
                parts.append(part)
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})
            print(f"[ERROR] {path.name}: {exc}")

    if not parts:
        print(f"[ERROR] Dataset '{dataset_name}': all files failed.")
        return {
            "dataset": dataset_name,
            "fits_files": len(fits_files),
            "rows": 0,
            "objects": 0,
            "quarters": 0,
            "anomaly_rows": 0,
            "output_file": "",
            "size_mb": 0.0,
            "status": "failed",
        }

    df = pd.concat(parts, ignore_index=True)
    df = prepare_dataframe(df)

    output_path = SAMPLES_DIR / f"kepler_{dataset_name}.parquet"
    preview_path = SAMPLES_DIR / f"kepler_{dataset_name}_preview.csv"

    df.to_parquet(output_path, index=False)
    df.head(5000).to_csv(preview_path, index=False, encoding="utf-8-sig")

    size_mb = output_path.stat().st_size / 1024 / 1024

    summary = {
        "dataset": dataset_name,
        "fits_files": len(fits_files),
        "rows": int(len(df)),
        "objects": int(df["object_id"].nunique()) if not df.empty else 0,
        "quarters": int(df["quarter"].nunique()) if not df.empty else 0,
        "anomaly_rows": int(df["is_anomaly"].sum()) if "is_anomaly" in df else 0,
        "output_file": str(output_path),
        "size_mb": round(size_mb, 3),
        "status": "success_with_errors" if errors else "success",
    }

    print(f"Saved parquet: {output_path}")
    print(f"Saved preview: {preview_path}")
    print(f"Rows: {summary['rows']:,}")
    print(f"Objects: {summary['objects']}")
    print(f"Quarters: {summary['quarters']}")
    print(f"Anomaly rows: {summary['anomaly_rows']:,}")
    print(f"Size: {summary['size_mb']} MB")

    if errors:
        error_path = PROCESSED_DIR / f"kepler_{dataset_name}_conversion_errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False, encoding="utf-8-sig")
        print(f"Conversion errors saved: {error_path}")

    return summary


def main() -> None:
    ensure_directories()

    summaries = []
    for dataset_name, raw_dir in DATASET_DIRS.items():
        summaries.append(convert_dataset(dataset_name, raw_dir))

    summary_df = pd.DataFrame(summaries)
    summary_path = PROCESSED_DIR / "kepler_conversion_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== FINAL SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
