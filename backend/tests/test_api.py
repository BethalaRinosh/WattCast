"""Contract and error-path tests for the WattCast HTTP API."""
import json

import pandas as pd
from fastapi.testclient import TestClient

from backend.app import main as api

client = TestClient(api.app)


def test_health_does_not_require_report_artifacts():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "WattCast API"


def test_series_returns_sorted_recent_rows_and_enforces_limit(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({
        "timestamp": ["2024-01-01 02:00:00", "2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "active_power_kw": [3.0, 1.0, 2.0],
    }).to_csv(processed / "hourly_energy.csv", index=False)
    monkeypatch.setattr(api, "PROCESSED_DIR", processed)

    response = client.get("/api/series/hourly?limit=2")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [point["active_power_kw"] for point in body["points"]] == [2.0, 3.0]


def test_missing_processed_file_has_actionable_404(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PROCESSED_DIR", tmp_path)
    response = client.get("/api/series/daily")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "artifact_not_found"
    assert "Run the corresponding WattCast pipeline script first" in response.json()["detail"]["message"]


def test_summary_reads_eda_and_evaluation_artifacts(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    (reports / "eda").mkdir(parents=True)
    (reports / "evaluation").mkdir(parents=True)
    (reports / "eda" / "eda_summary.json").write_text(json.dumps({
        "source_type": "real",
        "source_label": "REAL UCI DATA",
        "hourly": {"mean_kw": 1.1},
        "daily": {"mean_kw": 1.0},
    }), encoding="utf-8")
    (reports / "evaluation" / "evaluation_results.json").write_text(json.dumps({
        "series": {"hourly": {"best_by_mae": "holt_winters"}}
    }), encoding="utf-8")
    monkeypatch.setattr(api, "REPORTS_DIR", reports)

    response = client.get("/api/summary")
    assert response.status_code == 200
    assert response.json()["source_type"] == "real"
    assert response.json()["evaluation"]["series"]["hourly"]["best_by_mae"] == "holt_winters"


def test_missing_report_returns_clean_error(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "REPORTS_DIR", tmp_path)
    response = client.get("/api/eda")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "artifact_not_found"


def test_forecast_rejects_unknown_model():
    response = client.get("/api/forecast/hourly?model=imaginary_model")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_model"


def test_forecast_returns_saved_holdout_predictions(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    folder = reports / "statistical_models"
    folder.mkdir(parents=True)
    pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "actual_kw": [1.0, 2.0],
        "holt_winters_forecast_kw": [1.1, 1.8],
    }).to_csv(folder / "hourly_statistical_forecasts.csv", index=False)
    monkeypatch.setattr(api, "REPORTS_DIR", reports)

    response = client.get("/api/forecast/hourly?model=holt_winters&limit=1")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["model"] == "holt_winters"
    assert body["points"][0]["actual_kw"] == 2.0
    assert body["points"][0]["predicted_kw"] == 1.8
    assert "not a newly generated future forecast" in body["note"]


def test_decomposition_uses_residual_column_name(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    folder = reports / "time_series"
    folder.mkdir(parents=True)
    pd.DataFrame({
        "timestamp": ["2024-01-01", "2024-01-02"],
        "observed": [1.0, 2.0],
        "trend": [None, 1.5],
        "seasonal": [0.1, -0.1],
        "residual": [None, 0.6],
    }).to_csv(folder / "daily_decomposition.csv", index=False)
    monkeypatch.setattr(api, "REPORTS_DIR", reports)

    response = client.get("/api/decomposition/daily")
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert response.json()["points"][1]["residual"] == 0.6

def test_correlation_endpoint_returns_saved_series_result(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    folder = reports / "correlation"
    folder.mkdir(parents=True)
    (folder / "correlation_results.json").write_text(json.dumps({
        "project": "WattCast",
        "source_type": "real",
        "source_label": "REAL UCI DATA",
        "variables_note": "Original UCI variables.",
        "series": {
            "hourly": {
                "series": "hourly",
                "available": True,
                "observations": 120,
                "variables": ["active_power_kw", "voltage_v"],
                "pearson": {"active_power_kw": {"active_power_kw": 1.0, "voltage_v": 0.2}},
                "spearman": {"active_power_kw": {"active_power_kw": 1.0, "voltage_v": 0.1}},
                "target_ranking": [],
            }
        },
    }), encoding="utf-8")
    monkeypatch.setattr(api, "REPORTS_DIR", reports)

    response = client.get("/api/correlation/hourly")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["source_type"] == "real"
    assert body["pearson"]["active_power_kw"]["voltage_v"] == 0.2


def test_correlation_rejects_invalid_frequency():
    response = client.get("/api/correlation/weekly")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_series"



def test_health_includes_api_version():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"



def test_summary_returns_404_when_eda_report_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "REPORTS_DIR", tmp_path)
    response = client.get("/api/summary")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "artifact_not_found"



def test_summary_returns_404_when_evaluation_report_is_missing(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    eda_dir = reports / "eda"
    eda_dir.mkdir(parents=True)
    (eda_dir / "eda_summary.json").write_text(json.dumps({"source_type": "real"}), encoding="utf-8")
    monkeypatch.setattr(api, "REPORTS_DIR", reports)
    response = client.get("/api/summary")
    assert response.status_code == 404
    assert "evaluation_results.json" in response.json()["detail"]["message"]



def test_summary_rejects_malformed_json_report(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    eda_dir = reports / "eda"
    eda_dir.mkdir(parents=True)
    (eda_dir / "eda_summary.json").write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(api, "REPORTS_DIR", reports)
    response = client.get("/api/summary")
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "artifact_unreadable"



def test_eda_returns_available_profile_rows(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    eda_dir = reports / "eda"
    eda_dir.mkdir(parents=True)
    (eda_dir / "eda_summary.json").write_text(json.dumps({"source_type": "real"}), encoding="utf-8")
    pd.DataFrame({"hour": [0, 1], "mean_kw": [0.8, 1.2]}).to_csv(eda_dir / "intraday_profile.csv", index=False)
    monkeypatch.setattr(api, "REPORTS_DIR", reports)
    response = client.get("/api/eda")
    assert response.status_code == 200
    assert response.json()["profiles"]["intraday_profile"][1]["mean_kw"] == 1.2



def test_eda_allows_optional_profiles_to_be_absent(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    eda_dir = reports / "eda"
    eda_dir.mkdir(parents=True)
    (eda_dir / "eda_summary.json").write_text(json.dumps({"source_type": "synthetic"}), encoding="utf-8")
    monkeypatch.setattr(api, "REPORTS_DIR", reports)
    response = client.get("/api/eda")
    assert response.status_code == 200
    assert response.json()["profiles"] == {}



def test_series_rejects_unsupported_frequency():
    response = client.get("/api/series/weekly")
    assert response.status_code == 422



def test_series_rejects_limit_above_api_maximum(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PROCESSED_DIR", tmp_path)
    response = client.get("/api/series/hourly?limit=2001")
    assert response.status_code == 422



def test_series_drops_invalid_timestamps_and_non_finite_values(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00", "not-a-date", "2024-01-01 02:00:00"],
        "active_power_kw": [1.0, 2.0, float("inf")],
    }).to_csv(processed / "hourly_energy.csv", index=False)
    monkeypatch.setattr(api, "PROCESSED_DIR", processed)
    response = client.get("/api/series/hourly")
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["points"][0]["active_power_kw"] == 1.0



def test_series_returns_empty_points_for_empty_valid_csv(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(columns=["timestamp", "active_power_kw"]).to_csv(processed / "daily_energy.csv", index=False)
    monkeypatch.setattr(api, "PROCESSED_DIR", processed)
    response = client.get("/api/series/daily")
    assert response.status_code == 200
    assert response.json()["count"] == 0
    assert response.json()["points"] == []



def test_series_reports_missing_required_columns(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({"timestamp": ["2024-01-01"], "power": [1.0]}).to_csv(processed / "hourly_energy.csv", index=False)
    monkeypatch.setattr(api, "PROCESSED_DIR", processed)
    response = client.get("/api/series/hourly")
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "invalid_artifact_schema"



def test_decomposition_rejects_unsupported_frequency():
    response = client.get("/api/decomposition/weekly")
    assert response.status_code == 422



def test_decomposition_reports_missing_component_columns(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    folder = reports / "time_series"
    folder.mkdir(parents=True)
    pd.DataFrame({"timestamp": ["2024-01-01"], "observed": [1.0]}).to_csv(folder / "daily_decomposition.csv", index=False)
    monkeypatch.setattr(api, "REPORTS_DIR", reports)
    response = client.get("/api/decomposition/daily")
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "invalid_artifact_schema"



def test_decomposition_serializes_missing_components_as_null(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    folder = reports / "time_series"
    folder.mkdir(parents=True)
    pd.DataFrame({
        "timestamp": ["2024-01-01"],
        "observed": [1.0],
        "trend": [None],
        "seasonal": [0.2],
        "residual": [None],
    }).to_csv(folder / "daily_decomposition.csv", index=False)
    monkeypatch.setattr(api, "REPORTS_DIR", reports)
    response = client.get("/api/decomposition/daily")
    assert response.status_code == 200
    assert response.json()["points"][0]["trend"] is None
    assert response.json()["points"][0]["residual"] is None



def test_models_returns_requested_frequency_metrics(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    folder = reports / "evaluation"
    folder.mkdir(parents=True)
    (folder / "evaluation_results.json").write_text(json.dumps({
        "series": {"daily": {"best_by_mae": "seasonal_naive", "evaluation_observations": 12}}
    }), encoding="utf-8")
    monkeypatch.setattr(api, "REPORTS_DIR", reports)
    response = client.get("/api/models/daily")
    assert response.status_code == 200
    assert response.json()["best_by_mae"] == "seasonal_naive"
    assert response.json()["evaluation_observations"] == 12



def test_models_returns_404_when_frequency_has_no_evaluation(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    folder = reports / "evaluation"
    folder.mkdir(parents=True)
    (folder / "evaluation_results.json").write_text(json.dumps({"series": {"hourly": {}}}), encoding="utf-8")
    monkeypatch.setattr(api, "REPORTS_DIR", reports)
    response = client.get("/api/models/daily")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "series_evaluation_not_found"



def test_forecast_rejects_unsupported_frequency():
    response = client.get("/api/forecast/weekly")
    assert response.status_code == 422



def test_forecast_rejects_limit_above_api_maximum():
    response = client.get("/api/forecast/hourly?limit=2001")
    assert response.status_code == 422



def test_forecast_returns_404_when_no_valid_prediction_rows_exist(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    folder = reports / "statistical_models"
    folder.mkdir(parents=True)
    pd.DataFrame({
        "timestamp": ["not-a-date"],
        "actual_kw": [1.0],
        "holt_winters_forecast_kw": [1.1],
    }).to_csv(folder / "hourly_statistical_forecasts.csv", index=False)
    monkeypatch.setattr(api, "REPORTS_DIR", reports)
    response = client.get("/api/forecast/hourly?model=holt_winters")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "forecast_not_found"



def test_models_returns_requested_frequency_metrics(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    folder = reports / "evaluation"
    folder.mkdir(parents=True)
    (folder / "evaluation_results.json").write_text(json.dumps({
        "series": {"daily": {"best_by_mae": "seasonal_naive", "evaluation_observations": 12}}
    }), encoding="utf-8")
    monkeypatch.setattr(api, "REPORTS_DIR", reports)
    response = client.get("/api/models/daily")
    assert response.status_code == 200
    assert response.json()["best_by_mae"] == "seasonal_naive"
    assert response.json()["evaluation_observations"] == 12



def test_models_returns_404_when_frequency_has_no_evaluation(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    folder = reports / "evaluation"
    folder.mkdir(parents=True)
    (folder / "evaluation_results.json").write_text(json.dumps({"series": {"hourly": {}}}), encoding="utf-8")
    monkeypatch.setattr(api, "REPORTS_DIR", reports)
    response = client.get("/api/models/daily")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "series_evaluation_not_found"



def test_forecast_rejects_unsupported_frequency():
    response = client.get("/api/forecast/weekly")
    assert response.status_code == 422



def test_forecast_rejects_limit_above_api_maximum():
    response = client.get("/api/forecast/hourly?limit=2001")
    assert response.status_code == 422
