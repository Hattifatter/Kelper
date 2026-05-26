from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASH_DATA_DIR = PROJECT_ROOT / "results" / "dash_data"


def load_available_dash_data() -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}

    if not DASH_DATA_DIR.exists():
        return datasets

    for path in sorted(DASH_DATA_DIR.glob("*_dash_data.parquet")):
        # Examples: small_pandas_dash_data.parquet, large_dask_dash_data.parquet
        key = path.stem.replace("_dash_data", "")
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                df["object_id"] = df["object_id"].astype(str)
                datasets[key] = df
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load {path}: {exc}")

    return datasets


DATASETS = load_available_dash_data()

app = Dash(__name__)
app.title = "Kepler Data Dashboard"


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=message,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 18},
            }
        ],
    )
    return fig


def get_processing_method(df: pd.DataFrame) -> str:
    if "processing_method" not in df.columns or df.empty:
        return "unknown"
    values = df["processing_method"].dropna().astype(str)
    if values.empty:
        return "unknown"
    return values.mode().iloc[0]


def get_quarters_text(subset: pd.DataFrame) -> str:
    if "quarter" not in subset.columns:
        return "unknown"
    quarters = []
    for value in subset["quarter"].dropna().unique().tolist():
        try:
            quarter = int(value)
        except (TypeError, ValueError):
            continue
        if quarter >= 0:
            quarters.append(quarter)
    quarters = sorted(set(quarters))
    return ", ".join(map(str, quarters)) if quarters else "unknown"


app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "margin": "0 auto",
        "maxWidth": "1200px",
        "padding": "24px",
    },
    children=[
        html.H1("Kepler light curve dashboard"),
        html.P(
            "Интерактивная панель для анализа обработанных данных NASA Kepler. "
            "Панель загружает подготовленные данные для пар Pandas/Dask/Spark + Dash."
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "16px"},
            children=[
                html.Div(
                    children=[
                        html.Label("Датасет"),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            options=[{"label": key, "value": key} for key in DATASETS.keys()],
                            value=next(iter(DATASETS.keys()), None),
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Объект"),
                        dcc.Dropdown(id="object-dropdown", clearable=False),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Максимум точек на графике"),
                        dcc.Slider(
                            id="max-points-slider",
                            min=500,
                            max=20000,
                            step=500,
                            value=5000,
                            marks={500: "500", 5000: "5k", 10000: "10k", 20000: "20k"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            style={"marginTop": "24px"},
            children=[
                html.Label("Временной диапазон"),
                dcc.RangeSlider(
                    id="time-range-slider",
                    min=0,
                    max=1,
                    step=0.01,
                    value=[0, 1],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
        ),
        dcc.Graph(id="lightcurve-graph", style={"height": "620px", "marginTop": "24px"}),
        html.H2("Сводная статистика"),
        dash_table.DataTable(
            id="summary-table",
            columns=[
                {"name": "Показатель", "id": "metric"},
                {"name": "Значение", "id": "value"},
            ],
            data=[],
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"fontWeight": "bold"},
        ),
    ],
)


@app.callback(
    Output("object-dropdown", "options"),
    Output("object-dropdown", "value"),
    Input("dataset-dropdown", "value"),
)
def update_object_options(dataset_key: str | None):
    if not dataset_key or dataset_key not in DATASETS:
        return [], None

    df = DATASETS[dataset_key]
    objects = sorted(df["object_id"].astype(str).unique().tolist())
    options = [{"label": object_id, "value": object_id} for object_id in objects]
    value = objects[0] if objects else None
    return options, value


@app.callback(
    Output("time-range-slider", "min"),
    Output("time-range-slider", "max"),
    Output("time-range-slider", "value"),
    Output("time-range-slider", "step"),
    Input("dataset-dropdown", "value"),
    Input("object-dropdown", "value"),
)
def update_time_range(dataset_key: str | None, object_id: str | None):
    if not dataset_key or dataset_key not in DATASETS or not object_id:
        return 0, 1, [0, 1], 0.01

    df = DATASETS[dataset_key]
    subset = df[df["object_id"].astype(str) == str(object_id)]

    if subset.empty:
        return 0, 1, [0, 1], 0.01

    min_time = float(subset["time_bin"].min())
    max_time = float(subset["time_bin"].max())
    step = max((max_time - min_time) / 1000, 0.001)
    return min_time, max_time, [min_time, max_time], step


@app.callback(
    Output("lightcurve-graph", "figure"),
    Output("summary-table", "data"),
    Input("dataset-dropdown", "value"),
    Input("object-dropdown", "value"),
    Input("time-range-slider", "value"),
    Input("max-points-slider", "value"),
)
def update_graph(
    dataset_key: str | None,
    object_id: str | None,
    time_range: list[float] | None,
    max_points: int,
):
    if not DATASETS:
        return empty_figure("Нет данных для Dash. Сначала запустите парные эксперименты."), []

    if not dataset_key or dataset_key not in DATASETS or not object_id:
        return empty_figure("Выберите датасет и объект"), []

    df = DATASETS[dataset_key]
    processing_method = get_processing_method(df)
    subset = df[df["object_id"].astype(str) == str(object_id)].sort_values("time_bin").copy()

    if time_range and len(time_range) == 2:
        start_time, end_time = float(time_range[0]), float(time_range[1])
        subset = subset[(subset["time_bin"] >= start_time) & (subset["time_bin"] <= end_time)]

    rows_before_plot_limit = len(subset)
    if len(subset) > max_points:
        indices = pd.Series(range(len(subset))).sample(n=max_points, random_state=42).sort_values().to_numpy()
        subset = subset.iloc[indices].sort_values("time_bin")

    if subset.empty:
        return empty_figure("В выбранном диапазоне нет данных"), []

    anomalies = subset[subset["is_anomaly"] == True]  # noqa: E712

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=subset["time_bin"],
            y=subset["flux_norm"],
            mode="lines",
            name="Normalized flux",
            hovertemplate="Time: %{x}<br>Flux: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=subset["time_bin"],
            y=subset["rolling_flux"],
            mode="lines",
            name="Rolling trend",
            hovertemplate="Time: %{x}<br>Trend: %{y}<extra></extra>",
        )
    )

    if len(anomalies) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomalies["time_bin"],
                y=anomalies["flux_norm"],
                mode="markers",
                name="Anomaly",
                hovertemplate="Time: %{x}<br>Anomaly flux: %{y}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{dataset_key}: object {object_id} | {processing_method} + Dash",
        xaxis_title="Time, days",
        yaxis_title="Normalized flux",
        hovermode="x unified",
        template="plotly_white",
    )

    summary_data = [
        {"metric": "Dataset", "value": dataset_key},
        {"metric": "Object ID", "value": str(object_id)},
        {"metric": "Rows in selected range", "value": f"{rows_before_plot_limit:,}"},
        {"metric": "Rows plotted", "value": f"{len(subset):,}"},
        {"metric": "Quarters", "value": get_quarters_text(subset)},
        {"metric": "Anomaly rows plotted", "value": f"{len(anomalies):,}"},
        {"metric": "Mean normalized flux", "value": f"{subset['flux_norm'].mean():.6f}"},
        {"metric": "Std normalized flux", "value": f"{subset['flux_norm'].std():.6f}"},
        {"metric": "Processing method", "value": processing_method},
        {"metric": "Visualization method", "value": "dash"},
    ]

    return fig, summary_data


if __name__ == "__main__":
    if not DATASETS:
        print("No Dash data found.")
        print("Run first: python src\\run_pandas_pairs_with_dash.py or python src\\run_dask_pairs_with_dash.py")
    else:
        print("Loaded Dash datasets:")
        for key, df in DATASETS.items():
            print(f"- {key}: {len(df):,} rows, objects={df['object_id'].nunique()}")
        print("Open: http://127.0.0.1:8050")

    app.run(debug=True, host="127.0.0.1", port=8050)
