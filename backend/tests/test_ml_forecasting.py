"""Tests for Step 8 lag/calendar-feature machine-learning forecasts."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pytest
from scripts.ml_forecasting import _build_training_matrix, _recursive_forecast, _fit_model, calendar_features, run_ml_forecasting

def test_calendar_features_are_deterministic_and_cyclical():
    f=calendar_features(pd.Timestamp("2024-01-01 06:00:00"),"h")
    assert f["hour"]==6 and f["day_of_week"]==0
    assert f["hour_sin"]==pytest.approx(1.0)
    assert f["hour_cos"]==pytest.approx(0.0,abs=1e-12)

def test_training_features_only_use_prior_values():
    idx=pd.date_range("2024-01-01",periods=24*10,freq="h")
    x,y=_build_training_matrix(pd.Series(np.arange(len(idx),dtype=float),index=idx),"h")
    assert x.index.equals(y.index)
    assert x.iloc[0]["lag_1"]==y.iloc[0]-1
    assert x.iloc[0]["lag_24"]==y.iloc[0]-24
    assert x.iloc[0]["rolling_mean_24"]==pytest.approx(y.iloc[0]-12.5)
    assert "target" not in x.columns

def test_training_matrix_requires_enough_history():
    train=pd.Series(np.arange(30,dtype=float),index=pd.date_range("2024-01-01",periods=30,freq="h"))
    with pytest.raises(ValueError,match="Not enough training data"): _build_training_matrix(train,"h")

def test_recursive_forecast_returns_finite_predictions_for_full_horizon():
    idx=pd.date_range("2024-01-01",periods=24*12,freq="h")
    train=pd.Series(1+np.sin(np.arange(len(idx))*2*np.pi/24),index=idx)
    model,_,_=_fit_model(train,"h")
    future=pd.date_range(idx[-1]+pd.Timedelta(hours=1),periods=6,freq="h")
    first=_recursive_forecast(model,train,future,"h"); second=_recursive_forecast(model,train,future,"h")
    assert np.isfinite(first).all() and np.array_equal(first,second) and len(first)==len(future)

def test_run_ml_forecasting_writes_artifacts_and_preserves_chronology(tmp_path):
    processed,output=tmp_path/"processed",tmp_path/"reports"; processed.mkdir()
    hourly=pd.date_range("2024-01-01",periods=24*35,freq="h")
    daily=pd.date_range("2024-01-01",periods=100,freq="D")
    pd.DataFrame({"timestamp":hourly,"active_power_kw":1.4+0.2*np.sin(np.arange(len(hourly))*2*np.pi/24)}).to_csv(processed/"hourly_energy.csv",index=False)
    pd.DataFrame({"timestamp":daily,"active_power_kw":1.4+0.2*np.sin(np.arange(len(daily))*2*np.pi/7)}).to_csv(processed/"daily_energy.csv",index=False)
    (processed/"preprocessing_report.json").write_text(json.dumps({"source_type":"synthetic"}),encoding="utf-8")
    result=run_ml_forecasting(processed,output,test_size=0.2)
    assert result["step"]==8 and result["source_type"]=="synthetic"
    assert "actual holdout values are never used" in result["feature_policy"]
    for label in ("hourly","daily"):
        d=result["series"][label]
        assert d["train_end"]<d["test_start"]
        assert d["train_observations"]+d["test_observations"]==d["observations"]
        assert d["metrics"]["test_observations"]==d["test_observations"]
        assert (output/f"{label}_ml_forecasts.csv").exists()
        assert (output/f"{label}_ml_forecasts.png").exists()
    assert (output/"ml_model_metrics.csv").exists() and (output/"ml_model_results.json").exists()

@pytest.mark.parametrize("test_size",[0,1,-0.1,1.1])
def test_run_rejects_invalid_test_size(tmp_path,test_size):
    with pytest.raises(ValueError,match="test_size"): run_ml_forecasting(tmp_path,tmp_path/"out",test_size=test_size)
