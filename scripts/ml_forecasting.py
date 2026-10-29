#!/usr/bin/env python3
"""Train and evaluate a lag/calendar-feature Random Forest forecaster for WattCast."""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
try:
    from .baseline_forecasting import DEFAULT_PROCESSED_DIR, chronological_split, evaluate_forecast, longest_regular_segment, read_series
except ImportError:
    from baseline_forecasting import DEFAULT_PROCESSED_DIR, chronological_split, evaluate_forecast, longest_regular_segment, read_series

LOGGER = logging.getLogger("wattcast.ml_forecasting")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "backend" / "reports" / "ml_forecasting"
MODEL_NAME = "random_forest_lag_calendar"

def feature_columns(frequency: str) -> list[str]:
    if frequency == "h":
        return ["lag_1","lag_2","lag_3","lag_24","lag_48","lag_168","rolling_mean_24","rolling_mean_168","hour","day_of_week","day_of_month","month","is_weekend","hour_sin","hour_cos","weekday_sin","weekday_cos","month_sin","month_cos"]
    if frequency == "D":
        return ["lag_1","lag_2","lag_3","lag_7","lag_14","lag_28","rolling_mean_7","rolling_mean_28","day_of_week","day_of_month","day_of_year","month","is_weekend","weekday_sin","weekday_cos","year_day_sin","year_day_cos","month_sin","month_cos"]
    raise ValueError("frequency must be 'h' (hourly) or 'D' (daily).")

def calendar_features(timestamp: pd.Timestamp, frequency: str) -> dict[str, float]:
    timestamp = pd.Timestamp(timestamp)
    weekday, month = float(timestamp.dayofweek), float(timestamp.month)
    out = {"day_of_week":weekday,"day_of_month":float(timestamp.day),"month":month,"is_weekend":float(timestamp.dayofweek >= 5),
           "weekday_sin":float(np.sin(2*np.pi*weekday/7)),"weekday_cos":float(np.cos(2*np.pi*weekday/7)),
           "month_sin":float(np.sin(2*np.pi*(month-1)/12)),"month_cos":float(np.cos(2*np.pi*(month-1)/12))}
    if frequency == "h":
        hour=float(timestamp.hour)
        out.update({"hour":hour,"hour_sin":float(np.sin(2*np.pi*hour/24)),"hour_cos":float(np.cos(2*np.pi*hour/24))})
    elif frequency == "D":
        day=float(timestamp.dayofyear)
        out.update({"day_of_year":day,"year_day_sin":float(np.sin(2*np.pi*(day-1)/365.25)),"year_day_cos":float(np.cos(2*np.pi*(day-1)/365.25))})
    else:
        raise ValueError("frequency must be 'h' (hourly) or 'D' (daily).")
    return out

def _lag_values(history: list[float], frequency: str) -> dict[str, float]:
    if frequency == "h": lags, windows = (1,2,3,24,48,168), (24,168)
    elif frequency == "D": lags, windows = (1,2,3,7,14,28), (7,28)
    else: raise ValueError("frequency must be 'h' (hourly) or 'D' (daily).")
    needed=max(max(lags),max(windows))
    if len(history)<needed: raise ValueError(f"Need at least {needed} historical observations for frequency={frequency}; got {len(history)}.")
    out={f"lag_{lag}":float(history[-lag]) for lag in lags}
    out.update({f"rolling_mean_{w}":float(np.mean(history[-w:])) for w in windows})
    return out

def _build_training_matrix(train: pd.Series, frequency: str) -> tuple[pd.DataFrame,pd.Series]:
    if frequency=="h": lags,windows=(1,2,3,24,48,168),(24,168)
    elif frequency=="D": lags,windows=(1,2,3,7,14,28),(7,28)
    else: raise ValueError("frequency must be 'h' (hourly) or 'D' (daily).")
    frame=pd.DataFrame({"target":train.astype(float)})
    for lag in lags: frame[f"lag_{lag}"]=frame["target"].shift(lag)
    for w in windows: frame[f"rolling_mean_{w}"]=frame["target"].shift(1).rolling(window=w).mean()
    cal=pd.DataFrame([calendar_features(pd.Timestamp(ts),frequency) for ts in frame.index],index=frame.index)
    frame=frame.join(cal).dropna()
    if frame.empty: raise ValueError(f"Not enough training data to build lag features for frequency={frequency}.")
    return frame[feature_columns(frequency)].astype(float),frame["target"].astype(float)

def _fit_model(train: pd.Series, frequency: str, random_state: int=42) -> tuple[RandomForestRegressor,list[str],int]:
    x,y=_build_training_matrix(train,frequency)
    model=RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=random_state,n_jobs=-1)
    model.fit(x,y)
    return model,list(x.columns),len(x)

def _recursive_forecast(model: RandomForestRegressor, train: pd.Series, test_index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    """Forecast the holdout recursively; actual holdout targets never become features."""
    history=train.astype(float).tolist()
    preds=[]
    for ts in test_index:
        features={**_lag_values(history,frequency),**calendar_features(pd.Timestamp(ts),frequency)}
        pred=float(model.predict(pd.DataFrame([features],columns=feature_columns(frequency),dtype=float))[0])
        if not np.isfinite(pred): raise ValueError(f"Model generated a non-finite prediction at {ts}.")
        preds.append(pred); history.append(pred)
    return np.asarray(preds,dtype=float)

def run_ml_forecasting(processed_dir: Path=DEFAULT_PROCESSED_DIR, output_dir: Path=DEFAULT_OUTPUT_DIR, test_size: float=0.2, random_state: int=42) -> dict[str,Any]:
    if not 0<test_size<1: raise ValueError("test_size must be greater than 0 and less than 1.")
    processed_dir,output_dir=Path(processed_dir),Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    provenance=processed_dir/"preprocessing_report.json"
    source_type=json.loads(provenance.read_text(encoding="utf-8")).get("source_type","unknown") if provenance.exists() else "unknown"
    summary={"project":"WattCast","step":8,"source_type":source_type,"target":"Mean active power (kW)","model":MODEL_NAME,
      "model_parameters":{"n_estimators":200,"min_samples_leaf":2,"random_state":random_state,"n_jobs":-1},
      "split":{"method":"chronological; no shuffle","test_size_fraction":test_size,"training_period_precedes_test_period":True},
      "feature_policy":"Lag and rolling features use past target values only. Forecasting is recursive: each prediction is appended to history, and actual holdout values are never used as later features.","series":{}}
    rows=[]
    for label,filename,freq in (("hourly","hourly_energy.csv","h"),("daily","daily_energy.csv","D")):
        series=read_series(processed_dir/filename)
        regular,reg=longest_regular_segment(series,freq)
        train,test=chronological_split(regular,test_size)
        required=168 if freq=="h" else 28
        if len(train)<=required: raise ValueError(f"{label} training data needs more than {required} observations for lag features; got {len(train)}.")
        model,cols,nrows=_fit_model(train,freq,random_state)
        preds=_recursive_forecast(model,train,test.index,freq)
        metrics=evaluate_forecast(test,preds)
        summary["series"][label]={"frequency":"hourly" if freq=="h" else "daily","observations":int(len(regular)),
          "train_observations":int(len(train)),"test_observations":int(len(test)),"training_feature_rows":int(nrows),
          "feature_columns":cols,"train_start":train.index.min().isoformat(),"train_end":train.index.max().isoformat(),
          "test_start":test.index.min().isoformat(),"test_end":test.index.max().isoformat(),"regularization":reg,
          "metrics":metrics,"model_status":"success"}
        rows.append({"series":label,"model":MODEL_NAME,"status":"success",**metrics})
        forecast=pd.DataFrame({"timestamp":test.index,"actual_kw":test.to_numpy(),"random_forest_forecast_kw":preds})
        forecast.to_csv(output_dir/f"{label}_ml_forecasts.csv",index=False)
        plot=forecast.iloc[:min(len(test),24*14 if freq=="h" else 90)]
        fig,ax=plt.subplots(figsize=(13,5),constrained_layout=True)
        ax.plot(plot["timestamp"],plot["actual_kw"],label="Actual",linewidth=1.2)
        ax.plot(plot["timestamp"],plot["random_forest_forecast_kw"],label="Random Forest",linewidth=0.9)
        ax.set_title(f"{label.title()} ML forecast | {source_type.upper()} DATA")
        ax.set_ylabel("Mean active power (kW)"); ax.set_xlabel("Timestamp"); ax.grid(True,alpha=0.25); ax.legend()
        fig.savefig(output_dir/f"{label}_ml_forecasts.png",dpi=140); plt.close(fig)
    pd.DataFrame(rows).to_csv(output_dir/"ml_model_metrics.csv",index=False)
    (output_dir/"ml_model_results.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    LOGGER.info("ML forecasting complete for source=%s; outputs saved to %s",source_type,output_dir)
    return summary

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir",type=Path,default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--test-size",type=float,default=0.2)
    p.add_argument("--random-state",type=int,default=42)
    return p.parse_args()

def main() -> int:
    logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
    args=parse_args(); result=run_ml_forecasting(args.processed_dir,args.output_dir,args.test_size,args.random_state)
    for label,details in result["series"].items():
        m=details["metrics"]
        LOGGER.info("%s split: %s train / %s test (%s to %s)",label,details["train_observations"],details["test_observations"],details["test_start"],details["test_end"])
        LOGGER.info("%s %s: MAE=%.5f RMSE=%.5f MAPE=%s",label,MODEL_NAME,m["mae"],m["rmse"],f"{m['mape_percent']:.3f}%" if m["mape_percent"] is not None else "n/a")
    return 0
if __name__=="__main__": raise SystemExit(main())
