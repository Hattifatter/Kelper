from pathlib import Path

# Корень проекта определяется автоматически: C:\Kelper, если файл лежит в C:\Kelper\src
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
KEPLER_REAL_DIR = RAW_DIR / "kepler_real"

SMALL_RAW_DIR = KEPLER_REAL_DIR / "small"
MEDIUM_RAW_DIR = KEPLER_REAL_DIR / "medium"
LARGE_RAW_DIR = KEPLER_REAL_DIR / "large"

SAMPLES_DIR = DATA_DIR / "samples"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"
SCREENSHOTS_DIR = RESULTS_DIR / "screenshots"

DATASET_DIRS = {
    "small": SMALL_RAW_DIR,
    "medium": MEDIUM_RAW_DIR,
    "large": LARGE_RAW_DIR,
}

# Основные рабочие колонки, которые будут использовать все методы обработки.
BASE_COLUMNS = [
    "dataset",
    "source_file",
    "object_id",
    "quarter",
    "time",
    "flux",
    "flux_err",
    "quality",
]

# Колонки после первичной подготовки.
PREPARED_COLUMNS = BASE_COLUMNS + [
    "flux_norm",
    "is_anomaly",
]

# Порог для простого выявления аномалий по нормализованному потоку.
ANOMALY_Z_THRESHOLD = 3.0


def ensure_directories() -> None:
    """Создаёт все рабочие директории проекта."""
    for path in [
        DATA_DIR,
        RAW_DIR,
        KEPLER_REAL_DIR,
        SMALL_RAW_DIR,
        MEDIUM_RAW_DIR,
        LARGE_RAW_DIR,
        SAMPLES_DIR,
        PROCESSED_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        LOGS_DIR,
        SCREENSHOTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"Project root: {PROJECT_ROOT}")
    print("Directories are ready.")
