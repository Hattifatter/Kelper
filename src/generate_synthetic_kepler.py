import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def gb_to_bytes(value):
    return int(value * 1024 ** 3)


def generate_chunk(
    start_row,
    n_rows,
    n_objects,
    cadence_days,
    rng,
    base_flux,
    periods,
    transit_depths,
    transit_durations,
    dataset_name,
):
    global_idx = np.arange(start_row, start_row + n_rows, dtype=np.int64)

    object_idx = global_idx % n_objects
    object_id = object_idx.astype(np.int32)

    observation_idx = global_idx // n_objects
    time = 130.0 + observation_idx.astype(np.float32) * cadence_days

    quarter = ((time // 90) % 18 + 1).astype(np.int16)

    base = base_flux[object_idx]

    stellar_variability = 0.002 * np.sin(2 * np.pi * time / periods[object_idx])

    phase = (time % periods[object_idx]) / periods[object_idx]
    transit_mask = phase < transit_durations[object_idx]
    transit_signal = np.where(transit_mask, -transit_depths[object_idx], 0.0)

    noise = rng.normal(0, 0.0015, n_rows).astype(np.float32)

    flux = base * (1.0 + stellar_variability + transit_signal + noise)
    flux = flux.astype(np.float32)

    flux_err = (base * rng.uniform(0.0005, 0.0025, n_rows)).astype(np.float32)

    quality = np.zeros(n_rows, dtype=np.int16)

    bad_quality_mask = rng.random(n_rows) < 0.025
    quality_values = np.array([1, 2, 4, 8, 16, 32], dtype=np.int16)
    quality[bad_quality_mask] = rng.choice(
        quality_values,
        bad_quality_mask.sum()
    )

    nan_mask = rng.random(n_rows) < 0.01
    flux[nan_mask] = np.nan
    flux_err[nan_mask] = np.nan
    quality[nan_mask] = 1

    spike_mask = rng.random(n_rows) < 0.003
    flux[spike_mask] *= rng.uniform(
        0.97,
        1.03,
        spike_mask.sum()
    ).astype(np.float32)

    dataset_column = np.array([dataset_name] * n_rows)
    source_file_column = np.array([f"synthetic_chunk_{start_row}"] * n_rows)

    return pa.table(
        {
            "dataset": pa.array(dataset_column),
            "source_file": pa.array(source_file_column),
            "object_id": pa.array(object_id),
            "quarter": pa.array(quarter),
            "time": pa.array(time.astype(np.float32)),
            "flux": pa.array(flux),
            "flux_err": pa.array(flux_err),
            "quality": pa.array(quality),
        }
    )


def generate_dataset(
    output_path,
    target_gb,
    n_objects=100,
    chunk_rows=1_000_000,
    seed=42,
    compression="snappy",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    target_bytes = gb_to_bytes(target_gb)
    rng = np.random.default_rng(seed)

    dataset_name = output_path.stem

    base_flux = rng.uniform(80_000, 160_000, n_objects).astype(np.float32)
    periods = rng.uniform(1.5, 25.0, n_objects).astype(np.float32)
    transit_depths = rng.uniform(0.001, 0.02, n_objects).astype(np.float32)
    transit_durations = rng.uniform(0.01, 0.06, n_objects).astype(np.float32)

    # Kepler long cadence: примерно 29.4 минуты
    cadence_days = 29.4 / 60 / 24

    writer = None
    total_rows = 0
    chunk_number = 0

    print(f"Создание файла: {output_path}")
    print(f"Целевой размер: {target_gb:.2f} ГБ")
    print(f"Количество объектов: {n_objects}")
    print(f"Строк в одном блоке: {chunk_rows:,}")
    print("-" * 60)

    try:
        while True:
            table = generate_chunk(
                start_row=total_rows,
                n_rows=chunk_rows,
                n_objects=n_objects,
                cadence_days=cadence_days,
                rng=rng,
                base_flux=base_flux,
                periods=periods,
                transit_depths=transit_depths,
                transit_durations=transit_durations,
                dataset_name=dataset_name,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression=compression,
                    use_dictionary=True,
                )

            writer.write_table(table)

            total_rows += chunk_rows
            chunk_number += 1

            current_size = output_path.stat().st_size
            current_gb = current_size / 1024 ** 3

            print(
                f"Блок {chunk_number}: "
                f"строк {total_rows:,}; "
                f"размер {current_gb:.3f} ГБ"
            )

            if current_size >= target_bytes:
                break

    finally:
        if writer is not None:
            writer.close()

    final_size = output_path.stat().st_size
    final_gb = final_size / 1024 ** 3

    metadata = {
        "output_file": str(output_path),
        "target_gb": target_gb,
        "actual_gb": final_gb,
        "rows": total_rows,
        "objects": n_objects,
        "chunk_rows": chunk_rows,
        "compression": compression,
        "columns": [
            "dataset",
            "source_file",
            "object_id",
            "quarter",
            "time",
            "flux",
            "flux_err",
            "quality",
        ],
        "description": (
            "Synthetic Kepler-like time series dataset for stress testing "
            "Pandas, Dask and Spark processing pipelines."
        ),
    }

    metadata_path = output_path.with_suffix(".metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("-" * 60)
    print(f"Готово: {output_path}")
    print(f"Итоговый размер: {final_gb:.3f} ГБ")
    print(f"Количество строк: {total_rows:,}")
    print(f"Метаданные: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Генерация синтетического Kepler-подобного Parquet-датасета."
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Путь к выходному Parquet-файлу.",
    )

    parser.add_argument(
        "--target-gb",
        type=float,
        required=True,
        help="Целевой размер файла в ГБ.",
    )

    parser.add_argument(
        "--objects",
        type=int,
        default=100,
        help="Количество искусственных объектов наблюдения.",
    )

    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=1_000_000,
        help="Количество строк, генерируемых за один блок.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Зерно генератора случайных чисел.",
    )

    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "brotli", "zstd", "none"],
        help="Сжатие Parquet-файла.",
    )

    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression

    generate_dataset(
        output_path=args.output,
        target_gb=args.target_gb,
        n_objects=args.objects,
        chunk_rows=args.chunk_rows,
        seed=args.seed,
        compression=compression,
    )


if __name__ == "__main__":
    main()