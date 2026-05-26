import importlib
import platform
import sys
from pathlib import Path

from config import ensure_directories, PROJECT_ROOT, DATASET_DIRS


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "pyarrow",
    "astropy",
    "tqdm",
    "psutil",
    "matplotlib",
    "plotly",
    "dash",
    "dask",
    "pyspark",
]


def check_module(module_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version unknown")
        print(f"OK   {module_name:<12} {version}")
        return True
    except Exception as exc:
        print(f"FAIL {module_name:<12} {exc}")
        return False


def count_fits_files(path: Path) -> int:
    return len(list(path.glob("*.fits"))) + len(list(path.glob("*.FITS")))


def main() -> None:
    print("=== ENVIRONMENT CHECK ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    ensure_directories()

    ok = True
    for module_name in REQUIRED_MODULES:
        if not check_module(module_name):
            ok = False

    print()
    print("=== DATA DIRECTORIES ===")
    for dataset_name, path in DATASET_DIRS.items():
        print(f"{dataset_name:<7} {path} | FITS files: {count_fits_files(path)}")

    print()
    if ok:
        print("Environment is ready.")
    else:
        print("Environment is NOT ready. Install missing packages:")
        print("pip install -r requirements.txt")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
