"""FastAPI application exposing cached WattCast analysis artifacts.

Run from the repository root with:
    python -m uvicorn backend.app.main:app --reload
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = Path(os.getenv("WATTCAST_PROCESSED_DIR", ROOT / "backend" / "data" / "processed"))
REPORTS_DIR = Path(os.getenv("WATTCAST_REPORTS_DIR", ROOT / "backend" / "reports"))

SeriesName = Literal["hourly", "daily"]
MODEL_COLUMNS = {
    "naive": ("baselines", "naive_forecast_kw"),
    "seasonal_naive": ("baselines", "seasonal_naive_forecast_kw"),
    "holt_winters": ("statistical_models", "holt_winters_forecast_kw"),
    "arima_1_1_1": ("statistical_models", "arima_1_1_1_forecast_kw"),
    "sarima": ("statistical_models", "sarima_forecast_kw"),
    "random_forest_lag_calendar": ("ml_forecasting", "random_forest_forecast_kw"),
}


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "WattCast API"
    version: str = "1.0.0"


class SeriesPoint(BaseModel):
    timestamp: str
    active_power_kw: float


class SeriesResponse(BaseModel):
    series: SeriesName
    count: int
    points: list[SeriesPoint]


class SummaryResponse(BaseModel):
    source_type: str | None = None
    source_label: str | None = None
    hourly: dict[str, Any] | None = None
    daily: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None


class ForecastPoint(BaseModel):
    timestamp: str
    actual_kw: float
    predicted_kw: float


class ForecastResponse(BaseModel):
    series: SeriesName
    model: str
    count: int
    points: list[ForecastPoint]
    note: str = "These are saved chronological holdout predictions, not a newly generated future forecast."


app = FastAPI(
    title="WattCast API",
    description="Read-only API for household electricity time-series analysis and saved forecast evaluation.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _json_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "artifact_not_found",
                "message": f"Required report is missing: {path.name}. Run the corresponding WattCast pipeline script first.",
                "path": str(path),
            },
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "artifact_unreadable", "message": f"Could not read {path.name}: {exc}"},
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=500,
            detail={"code": "invalid_artifact_schema", "message": f"{path.name} must contain a JSON object."},
        )
    return payload


def _csv_artifact(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "artifact_not_found",
                "message": f"Required data file is missing: {path.name}. Run the corresponding WattCast pipeline script first.",
                "path": str(path),
            },
        )
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "artifact_unreadable", "message": f"Could not read {path.name}: {exc}"},
        ) from exc
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "invalid_artifact_schema",
                "message": f"{path.name} is missing required columns: {missing}",
            },
        )
    return frame


def _valid_series(series: str) -> SeriesName:
    if series not in ("hourly", "daily"):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_series", "message": "series must be 'hourly' or 'daily'."},
        )
    return series  # type: ignore[return-value]


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Confirm the API process is running; this does not imply reports exist."""
    return HealthResponse()


@app.get("/api/summary", response_model=SummaryResponse, tags=["analysis"])
def summary() -> SummaryResponse:
    """Return compact EDA statistics and model-evaluation rankings."""
    eda = _json_artifact(REPORTS_DIR / "eda" / "eda_summary.json")
    evaluation = _json_artifact(REPORTS_DIR / "evaluation" / "evaluation_results.json")
    return SummaryResponse(
        source_type=eda.get("source_type"),
        source_label=eda.get("source_label"),
        hourly=eda.get("hourly"),
        daily=eda.get("daily"),
        evaluation=evaluation,
    )


@app.get("/api/series/{series}", response_model=SeriesResponse, tags=["data"])
def get_series(
    series: SeriesName,
    limit: int = Query(default=500, ge=1, le=2000, description="Maximum most-recent observations to return."),
) -> SeriesResponse:
    """Return recent cleaned hourly or daily observations."""
    series = _valid_series(series)
    frame = _csv_artifact(
        PROCESSED_DIR / f"{series}_energy.csv",
        {"timestamp", "active_power_kw"},
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["active_power_kw"] = pd.to_numeric(frame["active_power_kw"], errors="coerce")
    frame = frame.replace([float("inf"), float("-inf")], float("nan")).dropna(
        subset=["timestamp", "active_power_kw"]
    )
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").tail(limit)
    points = [
        SeriesPoint(timestamp=row.timestamp.isoformat(), active_power_kw=float(row.active_power_kw))
        for row in frame.itertuples(index=False)
    ]
    return SeriesResponse(series=series, count=len(points), points=points)


@app.get("/api/eda", tags=["analysis"])
def get_eda() -> dict[str, Any]:
    """Return the complete EDA summary and profile tables."""
    summary_data = _json_artifact(REPORTS_DIR / "eda" / "eda_summary.json")
    profiles = {}
    for filename in ("intraday_profile.csv", "weekday_profile.csv", "monthly_profile.csv"):
        path = REPORTS_DIR / "eda" / filename
        if path.is_file():
            profiles[filename.removesuffix(".csv")] = _csv_artifact(path, set()).to_dict(orient="records")
    return {"summary": summary_data, "profiles": profiles}


@app.get("/api/decomposition/{series}", tags=["analysis"])
def get_decomposition(series: SeriesName) -> dict[str, Any]:
    """Return saved observed/trend/seasonal/residual components."""
    series = _valid_series(series)
    frame = _csv_artifact(
        REPORTS_DIR / "time_series" / f"{series}_decomposition.csv",
        {"observed", "trend", "seasonal", "residual"},
    )
    if "timestamp" not in frame.columns:
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "timestamp"})
    frame = frame.replace([float("inf"), float("-inf")], float("nan"))
    frame = frame.astype(object).where(pd.notna(frame), None)
    return {"series": series, "count": len(frame), "points": frame.to_dict(orient="records")}


@app.get("/api/models/{series}", tags=["models"])
def get_model_comparison(series: SeriesName) -> dict[str, Any]:
    """Return model metrics and residual diagnostics for one frequency."""
    series = _valid_series(series)
    result = _json_artifact(REPORTS_DIR / "evaluation" / "evaluation_results.json")
    series_result = result.get("series", {}).get(series)
    if not isinstance(series_result, dict):
        raise HTTPException(
            status_code=404,
            detail={"code": "series_evaluation_not_found", "message": f"No evaluation result exists for {series}."},
        )
    return {"series": series, **series_result}


@app.get("/api/forecast/{series}", response_model=ForecastResponse, tags=["models"])
def get_saved_forecast(
    series: SeriesName,
    model: str = Query(default="holt_winters", description="Model identifier from the comparison endpoint."),
    limit: int = Query(default=500, ge=1, le=2000),
) -> ForecastResponse:
    """Return stored holdout predictions; does not fit a new model."""
    series = _valid_series(series)
    if model not in MODEL_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_model", "message": f"model must be one of: {', '.join(MODEL_COLUMNS)}."},
        )
    report_dir, prediction_column = MODEL_COLUMNS[model]
    path = REPORTS_DIR / report_dir / f"{series}_{'baseline_forecasts' if report_dir == 'baselines' else 'statistical_forecasts' if report_dir == 'statistical_models' else 'ml_forecasts'}.csv"
    frame = _csv_artifact(path, {"timestamp", "actual_kw", prediction_column})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in ("actual_kw", prediction_column):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([float("inf"), float("-inf")], float("nan")).dropna(
        subset=["timestamp", "actual_kw", prediction_column]
    )
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").tail(limit)
    points = [
        ForecastPoint(
            timestamp=row.timestamp.isoformat(),
            actual_kw=float(row.actual_kw),
            predicted_kw=float(getattr(row, prediction_column)),
        )
        for row in frame.itertuples(index=False)
    ]
    if not points:
        raise HTTPException(
            status_code=404,
            detail={"code": "forecast_not_found", "message": f"No valid saved predictions found for {series}/{model}."},
        )
    return ForecastResponse(series=series, model=model, count=len(points), points=points)

@app.get("/api/correlation/{series}", tags=["analysis"])
def get_correlation(series: str) -> dict[str, Any]:
    """Return saved Pearson and Spearman correlation analysis for a frequency."""
    series = _valid_series(series)
    result = _json_artifact(REPORTS_DIR / "correlation" / "correlation_results.json")
    series_result = result.get("series", {}).get(series)
    if not isinstance(series_result, dict):
        raise HTTPException(
            status_code=404,
            detail={"code": "correlation_not_found", "message": f"No correlation result exists for {series}. Run scripts/correlation_analysis.py first."},
        )
    return {
        "project": result.get("project", "WattCast"),
        "source_type": result.get("source_type", "unknown"),
        "source_label": result.get("source_label", "UNKNOWN SOURCE"),
        "variables_note": result.get("variables_note"),
        **series_result,
    }
