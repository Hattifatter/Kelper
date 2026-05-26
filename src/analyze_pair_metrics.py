from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_FILE = PROJECT_ROOT / "results" / "metrics" / "all_pair_metrics_latest.csv"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
PLOTS_DIR = PROJECT_ROOT / "results" / "analysis" / "plots"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _split_pair(pair: str) -> tuple[str, str]:
    if not isinstance(pair, str):
        return "", ""
    parts = [p.strip().lower() for p in pair.split("+")]
    if len(parts) == 2:
        return parts[0], parts[1]
    return pair.strip().lower(), ""


def _save_table(df: pd.DataFrame, filename: str) -> Path:
    path = ANALYSIS_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _format_seconds(value: float) -> str:
    if pd.isna(value):
        return "нет данных"
    return f"{value:.6f}"


def _format_memory(value: float) -> str:
    if pd.isna(value):
        return "нет данных"
    return f"{value:.3f}"


def load_metrics() -> pd.DataFrame:
    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_FILE}\n"
            "Run pandas, dask and spark pair experiments first."
        )

    df = pd.read_csv(METRICS_FILE)

    required = [
        "dataset",
        "pair",
        "processing_time_sec",
        "visualization_time_sec",
        "total_time_sec",
        "processing_memory_peak_mb",
        "status",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in metrics file: {missing}")

    for column in ["processing_time_sec", "visualization_time_sec", "total_time_sec", "processing_memory_peak_mb"]:
        df[column] = _safe_float(df[column])

    pair_parts = df["pair"].apply(_split_pair)
    df["processing_method"] = pair_parts.apply(lambda x: x[0])
    df["visualization_method"] = pair_parts.apply(lambda x: x[1])

    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["status"] = df["status"].astype(str).str.lower()

    clean = df[df["status"] == "success"].copy()
    clean = clean.dropna(subset=["total_time_sec"])

    dataset_order = {"small": 0, "medium": 1, "large": 2}
    method_order = {"pandas": 0, "dask": 1, "spark": 2}
    viz_order = {"matplotlib": 0, "plotly": 1, "dash": 2}

    clean["dataset_order"] = clean["dataset"].map(dataset_order).fillna(99)
    clean["method_order"] = clean["processing_method"].map(method_order).fillna(99)
    clean["viz_order"] = clean["visualization_method"].map(viz_order).fillna(99)
    clean = clean.sort_values(["dataset_order", "method_order", "viz_order"]).reset_index(drop=True)

    return clean


def build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    tables["all_pair_metrics_clean"] = df.drop(
        columns=["dataset_order", "method_order", "viz_order"],
        errors="ignore",
    )

    tables["best_pair_by_dataset"] = (
        df.sort_values(["dataset", "total_time_sec"])
        .groupby("dataset", as_index=False)
        .first()[
            [
                "dataset",
                "pair",
                "processing_time_sec",
                "visualization_time_sec",
                "total_time_sec",
                "processing_memory_peak_mb",
            ]
        ]
    )

    tables["best_pair_by_processing_method"] = (
        df.sort_values(["processing_method", "dataset", "total_time_sec"])
        .groupby(["dataset", "processing_method"], as_index=False)
        .first()[
            [
                "dataset",
                "processing_method",
                "visualization_method",
                "pair",
                "processing_time_sec",
                "visualization_time_sec",
                "total_time_sec",
                "processing_memory_peak_mb",
            ]
        ]
    )

    tables["processing_method_summary"] = (
        df.groupby(["dataset", "processing_method"], as_index=False)
        .agg(
            mean_processing_time_sec=("processing_time_sec", "mean"),
            min_processing_time_sec=("processing_time_sec", "min"),
            mean_total_time_sec=("total_time_sec", "mean"),
            min_total_time_sec=("total_time_sec", "min"),
            mean_memory_peak_mb=("processing_memory_peak_mb", "mean"),
        )
        .sort_values(["dataset", "mean_total_time_sec"])
    )

    tables["visualization_method_summary"] = (
        df.groupby(["dataset", "visualization_method"], as_index=False)
        .agg(
            mean_visualization_time_sec=("visualization_time_sec", "mean"),
            min_visualization_time_sec=("visualization_time_sec", "min"),
            mean_total_time_sec=("total_time_sec", "mean"),
            min_total_time_sec=("total_time_sec", "min"),
        )
        .sort_values(["dataset", "mean_visualization_time_sec"])
    )

    tables["pair_ranking"] = (
        df.groupby(["pair", "processing_method", "visualization_method"], as_index=False)
        .agg(
            mean_total_time_sec=("total_time_sec", "mean"),
            max_total_time_sec=("total_time_sec", "max"),
            mean_processing_time_sec=("processing_time_sec", "mean"),
            mean_visualization_time_sec=("visualization_time_sec", "mean"),
            mean_memory_peak_mb=("processing_memory_peak_mb", "mean"),
        )
        .sort_values("mean_total_time_sec")
        .reset_index(drop=True)
    )
    tables["pair_ranking"]["rank"] = range(1, len(tables["pair_ranking"]) + 1)

    return tables


def save_summary_tables(tables: dict[str, pd.DataFrame]) -> list[Path]:
    output_paths: list[Path] = []
    for name, table in tables.items():
        output_paths.append(_save_table(table, f"{name}.csv"))
    return output_paths


def _plot_bar(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, output_path: Path, rotation: int = 30) -> None:
    plt.figure(figsize=(12, 6))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_plots(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> list[Path]:
    output_paths: list[Path] = []

    # Plot 1: total time for every pair grouped by dataset.
    plot_df = df.copy()
    plot_df["label"] = plot_df["dataset"] + " | " + plot_df["pair"]
    output = PLOTS_DIR / "total_time_all_pairs.png"
    _plot_bar(
        plot_df,
        x="label",
        y="total_time_sec",
        title="Total execution time for processing + visualization pairs",
        xlabel="Dataset and pair",
        ylabel="Total time, seconds",
        output_path=output,
        rotation=70,
    )
    output_paths.append(output)

    # Plot 2: best pair by dataset.
    output = PLOTS_DIR / "best_pair_by_dataset.png"
    best = tables["best_pair_by_dataset"].copy()
    best["label"] = best["dataset"] + " | " + best["pair"]
    _plot_bar(
        best,
        x="label",
        y="total_time_sec",
        title="Best pair by dataset",
        xlabel="Dataset and best pair",
        ylabel="Total time, seconds",
        output_path=output,
        rotation=30,
    )
    output_paths.append(output)

    # Plot 3: average total time by processing method.
    output = PLOTS_DIR / "avg_total_time_by_processing_method.png"
    avg_processing = (
        df.groupby("processing_method", as_index=False)
        .agg(mean_total_time_sec=("total_time_sec", "mean"))
        .sort_values("mean_total_time_sec")
    )
    _plot_bar(
        avg_processing,
        x="processing_method",
        y="mean_total_time_sec",
        title="Average total time by processing method",
        xlabel="Processing method",
        ylabel="Average total time, seconds",
        output_path=output,
        rotation=0,
    )
    output_paths.append(output)

    # Plot 4: average memory by processing method.
    output = PLOTS_DIR / "avg_memory_by_processing_method.png"
    avg_memory = (
        df.groupby("processing_method", as_index=False)
        .agg(mean_memory_peak_mb=("processing_memory_peak_mb", "mean"))
        .sort_values("mean_memory_peak_mb")
    )
    _plot_bar(
        avg_memory,
        x="processing_method",
        y="mean_memory_peak_mb",
        title="Average peak memory by processing method",
        xlabel="Processing method",
        ylabel="Peak memory, MB",
        output_path=output,
        rotation=0,
    )
    output_paths.append(output)

    # Plot 5: visualization time by visualization method.
    output = PLOTS_DIR / "avg_visualization_time_by_method.png"
    avg_viz = (
        df.groupby("visualization_method", as_index=False)
        .agg(mean_visualization_time_sec=("visualization_time_sec", "mean"))
        .sort_values("mean_visualization_time_sec")
    )
    _plot_bar(
        avg_viz,
        x="visualization_method",
        y="mean_visualization_time_sec",
        title="Average visualization preparation time by method",
        xlabel="Visualization method",
        ylabel="Average visualization time, seconds",
        output_path=output,
        rotation=0,
    )
    output_paths.append(output)

    return output_paths


def build_text_conclusions(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> str:
    best_by_dataset = tables["best_pair_by_dataset"]
    ranking = tables["pair_ranking"]
    proc_summary = tables["processing_method_summary"]

    overall_best = ranking.iloc[0]
    fastest_processing_rows = (
        proc_summary.sort_values(["dataset", "mean_processing_time_sec"])
        .groupby("dataset", as_index=False)
        .first()
    )

    lines: list[str] = []

    lines.append("ИТОГОВЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТАЛЬНОГО СТЕНДА")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1. Общие сведения")
    lines.append(
        "В ходе эксперимента были сопоставлены пары «метод обработки + метод визуализации». "
        "В качестве методов обработки использовались Pandas, Dask и Spark, а в качестве средств "
        "визуализации — Matplotlib, Plotly и Dash. Для каждой пары фиксировались время обработки, "
        "время подготовки визуализации, суммарное время выполнения и пиковое потребление памяти."
    )
    lines.append("")

    lines.append("2. Лучшие пары по каждому объёму данных")
    for _, row in best_by_dataset.iterrows():
        lines.append(
            f"- Набор {row['dataset']}: лучшая пара — {row['pair']}; "
            f"общее время — {_format_seconds(row['total_time_sec'])} с; "
            f"пиковая память — {_format_memory(row['processing_memory_peak_mb'])} MB."
        )
    lines.append("")

    lines.append("3. Общий рейтинг")
    lines.append(
        f"В среднем по всем наборам данных наименьшее суммарное время показала пара "
        f"{overall_best['pair']} со средним временем {_format_seconds(overall_best['mean_total_time_sec'])} с."
    )
    lines.append("")

    lines.append("4. Поведение методов обработки")
    for _, row in fastest_processing_rows.iterrows():
        lines.append(
            f"- Для набора {row['dataset']} минимальное среднее время обработки показал метод "
            f"{row['processing_method']} со средним временем обработки "
            f"{_format_seconds(row['mean_processing_time_sec'])} с."
        )
    lines.append(
        "На текущих объёмах данных локальная обработка Pandas, как правило, оказывается выгодной, "
        "поскольку файлы помещаются в оперативную память и не требуют организации распределённого выполнения. "
        "Dask и Spark имеют дополнительные накладные расходы на планирование вычислений и инициализацию "
        "исполнительной среды, поэтому их преимущество ожидаемо проявляется на существенно больших объёмах "
        "или при усложнении вычислительного конвейера."
    )
    lines.append("")

    lines.append("5. Поведение методов визуализации")
    viz_summary = tables["visualization_method_summary"]
    viz_global = (
        df.groupby("visualization_method", as_index=False)
        .agg(mean_visualization_time_sec=("visualization_time_sec", "mean"))
        .sort_values("mean_visualization_time_sec")
    )
    for _, row in viz_global.iterrows():
        lines.append(
            f"- {row['visualization_method']}: среднее время подготовки визуализации "
            f"{_format_seconds(row['mean_visualization_time_sec'])} с."
        )
    lines.append(
        "Matplotlib целесообразен для формирования статических отчётных графиков. Plotly обеспечивает "
        "интерактивный HTML-график без запуска отдельного веб-приложения. Dash требует подготовки данных "
        "для панели, но предоставляет наиболее развитый интерактивный интерфейс: выбор набора данных, объекта, "
        "временного диапазона и просмотр сводной статистики."
    )
    lines.append("")

    lines.append("6. Вывод для главы 3")
    lines.append(
        "Результаты подтверждают, что оценивать методы обработки и визуализации целесообразно именно парами, "
        "так как итоговая применимость решения определяется не только скоростью вычислительного этапа, "
        "но и способом представления результата пользователю. Для малых и умеренных фрагментов NASA Kepler "
        "наиболее рациональны простые связки на основе Pandas и интерактивной визуализации Plotly или Dash. "
        "Связки со Spark демонстрируют повышенные накладные расходы в локальной среде, однако сохраняют "
        "практическую значимость как масштабируемый вариант для более крупных наборов данных."
    )

    return "\n".join(lines)


def save_thesis_text(text: str) -> Path:
    path = ANALYSIS_DIR / "thesis_chapter3_conclusions.txt"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    print("=== FINAL PAIR METRICS ANALYSIS ===")
    print(f"Metrics file: {METRICS_FILE}")

    df = load_metrics()
    print(f"Loaded successful pair rows: {len(df)}")

    tables = build_summary_tables(df)
    table_paths = save_summary_tables(tables)
    plot_paths = make_plots(df, tables)

    text = build_text_conclusions(df, tables)
    text_path = save_thesis_text(text)

    print("\n=== GENERATED TABLES ===")
    for path in table_paths:
        print(path)

    print("\n=== GENERATED PLOTS ===")
    for path in plot_paths:
        print(path)

    print("\n=== GENERATED TEXT ===")
    print(text_path)

    print("\n=== BEST PAIRS BY DATASET ===")
    print(tables["best_pair_by_dataset"].to_string(index=False))

    print("\n=== OVERALL PAIR RANKING ===")
    print(tables["pair_ranking"][["rank", "pair", "mean_total_time_sec", "mean_memory_peak_mb"]].to_string(index=False))


if __name__ == "__main__":
    main()
