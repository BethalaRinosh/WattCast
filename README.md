# ⚡ WattCast
### Energy Consumption Forecasting using Time Series Analysis
**Tagline:** Every unit tells a story.

WattCast is an academic micro project for the Data Science 3rd Internal Assessment. It analyses historical electricity consumption and compares time-series and machine-learning forecasting methods through a web dashboard.

> **Project status:** Steps 1–10 are verified locally. Step 9 was confirmed on real UCI data with 7 focused evaluation tests and 57 passing backend tests. Step 10's API tests passed 8/8 and the complete backend suite passed 65/65; Uvicorn started successfully and `/api/health`, `/docs`, and `/openapi.json` returned HTTP 200. Step 11's React/Vite frontend scaffold is committed; local npm build and browser verification are pending. One non-blocking Starlette deprecation warning remains for the current `httpx` test-client integration.

## Problem statement
Electricity consumption changes over time due to recurring daily and weekly routines, longer-term trends, and random variation. WattCast aims to explore these patterns and compare forecasting approaches using a chronological test set.

## Objectives and scope
- Inspect and clean a public household electricity-consumption dataset.
- Explore temporal patterns and time-series properties.
- Compare simple baselines, statistical forecasting methods, and a lag-feature machine-learning model.
- Evaluate forecasts using MAE, RMSE and MAPE, with residual diagnostics.
- Expose cached analysis through a FastAPI API and display it in a React dashboard.

## Dataset and provenance
**Primary source:** [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), DOI: [10.24432/C58K54](https://doi.org/10.24432/C58K54).

The UCI dataset contains 2,075,259 one-minute measurements from one household near Paris, France, collected between December 2006 and November 2010. Its original file is semicolon-delimited and includes missing measurement values. The dataset is licensed under CC BY 4.0; cite the source when using it.

**Citation:** Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54.

## Step 2: Download and synthetic fallback
- `scripts/download_data.py` downloads the official UCI ZIP, extracts `household_power_consumption.txt`, and writes provenance metadata.
- `scripts/generate_synthetic_data.py` creates reproducible hourly simulated readings with trend, daily, weekly and yearly patterns plus random noise.

Install dependencies:
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Download the real dataset, with explicit synthetic fallback if download fails:
```bash
python scripts/download_data.py
```

Generate synthetic data directly:
```bash
python scripts/generate_synthetic_data.py --years 4 --seed 42
```

To require real data and refuse fallback, use `python scripts/download_data.py --no-synthetic-fallback`. Synthetic output contains `data_source=synthetic` and metadata with `"source_type": "synthetic"`. Synthetic values are simulated, not observed household measurements.

Run the offline data-acquisition tests:
```bash
python -m pytest backend/tests/test_data_acquisition.py -q
```

## Step 3: Data loading, cleaning and resampling
Implemented in:
- `scripts/data_processing.py`: reusable functions to discover, load, inspect, clean and resample either source.
- `scripts/preprocess_data.py`: command-line entry point that saves hourly/daily series and a JSON report.
- `backend/tests/test_preprocessing.py`: focused tests for the real UCI text schema, synthetic CSV schema, missing values, inspection and resampling.

Run preprocessing from the repository root after downloading or generating data:
```bash
python scripts/preprocess_data.py
```

Optional explicit paths:
```bash
python scripts/preprocess_data.py --input backend/data/raw/synthetic_hourly_energy.csv --source-type synthetic
```

Outputs are written under `backend/data/processed/`:
- `hourly_energy.csv`: hourly mean active power in kW.
- `daily_energy.csv`: daily mean active power in kW.
- `preprocessing_report.json`: source type, input dimensions/types/date range/frequency/missing counts, output row counts, IQR outlier summary and cleaning policy.

### Cleaning decisions
1. **UCI parsing:** Read the semicolon-delimited text and treat `?` and empty measurement fields as missing. Combine `Date` and `Time` into a datetime index.
2. **Common schema:** Both sources become a single `active_power_kw` series indexed by timestamp.
3. **Datetime integrity:** Invalid timestamps are dropped, records are sorted chronologically, and duplicate timestamps are reduced to the first observation.
4. **Numeric integrity:** Non-numeric and infinite power values become missing.
5. **Resampling:** Use the mean power in each hour and day. The target is average active power in kW; for a one-hour interval, mean kW is numerically equivalent to approximate kWh consumed during that hour.
6. **Missing values:** Time interpolation fills only complete internal missing runs of at most three resampled bins. Longer gaps are left missing in full and dropped rather than partially inventing values.
7. **Outliers:** Report values outside the 1.5×IQR bounds. Do not automatically remove them, because a genuine high-demand period may be an important peak. The IQR rule is a screening tool, not proof of a data error.

Run the preprocessing tests:
```bash
python -m pytest backend/tests/test_preprocessing.py -q
```

**Verification note:** The user ran the real UCI download and preprocessing locally. Their console reported 2,075,259 input rows, 25,979 missing active-power values, 0 duplicate timestamps, 34,190 hourly rows and 1,441 daily rows. These results came from the user's terminal output; rerun the commands if regenerating the data. The full UCI source has over two million minute-level rows, so processing can take a little time and memory.


## Step 4: Exploratory data analysis (EDA)
Run this after Step 3 has generated the processed hourly and daily files:
```bash
python scripts/eda.py
```

Optional custom paths:
```bash
python scripts/eda.py --processed-dir backend/data/processed --output-dir backend/reports/eda
```

The EDA pipeline reads the processed CSVs, validates their schema, sorts timestamps and removes invalid target rows, then writes to `backend/reports/eda/`:
- `eda_summary.json`: source label, descriptive statistics, date range, average power by hour/day-of-week/month, and the hours/weekdays with highest and lowest mean power.
- `intraday_profile.csv`, `weekday_profile.csv`, `monthly_profile.csv`: reproducible profile values for later display or inspection.
- `hourly_series.png`: hourly mean active power over time.
- `daily_series.png`: daily mean active power with a 7-day rolling mean.
- `intraday_profile.png`: average profile across the 24 hours of a day.
- `weekday_profile.png`: average profile across weekdays.

All chart titles identify the source as **REAL UCI DATA** or **SYNTHETIC SIMULATED DATA** when the preprocessing report is available. Synthetic plots describe only simulated values and must not be presented as measured household behavior. The summaries are descriptive, not causal. EDA does not train models or report forecast accuracy. The user ran EDA locally on the real source: the script reported a 1.092 kW hourly mean and peak average hour 20 (8 PM). The user also reported 4 EDA tests and all 9 then-existing backend tests passing.

Run EDA tests:
```bash
python -m pytest backend/tests/test_eda.py -q
```


## Step 5: Decomposition, stationarity and ACF/PACF
Run after preprocessing:
```bash
python scripts/time_series_analysis.py
```

Optional custom paths:
```bash
python scripts/time_series_analysis.py --processed-dir backend/data/processed --output-dir backend/reports/time_series
```

The script is implemented in `scripts/time_series_analysis.py`. It creates `backend/reports/time_series/` containing:
- `time_series_analysis.json`: source label, decomposition metadata, ADF/KPSS results for daily levels and first differences, and ACF/PACF settings.
- `hourly_decomposition.csv` and `daily_decomposition.csv`: observed, trend, seasonal and residual components.
- `hourly_decomposition.png` and `daily_decomposition.png`: four-panel additive decomposition charts.
- `daily_acf_pacf.csv`: numerical ACF/PACF values through the selected lag.
- `daily_acf_pacf.png`: autocorrelation and partial-autocorrelation diagnostic plots.

### Method and interpretation
- **Regular intervals first:** Decomposition and autocorrelation assume equally spaced timestamps. The script reindexes each series to an hourly/daily grid, interpolates only up to three consecutive internal missing bins, and selects the longest remaining contiguous segment. Longer gaps are not bridged; the selected segment and interpolation counts are recorded in the JSON report.
- **Seasonal decomposition:** Additive decomposition is run on the regular hourly segment with period 24 (daily cycle) and the regular daily segment with period 7 (weekly cycle). It separates observed values into estimated trend, seasonal pattern and residual. It is a descriptive tool, not a forecasting model.
- **ADF test:** The null hypothesis is that the series has a unit root. A small p-value (commonly below 0.05) is evidence against that null.
- **KPSS test:** The null hypothesis is level stationarity. A small p-value is evidence against stationarity. ADF and KPSS have opposite null hypotheses, so interpret both rather than relying on one result.
- **Differencing:** The tests run on the daily level series and its first difference. Differencing can remove some trend, but it should be justified by diagnostics and model behavior.
- **ACF/PACF:** The longest regular daily segment is analysed up to 60 lags where the sample size permits. ACF shows correlation with lagged values; PACF estimates the direct association at each lag after accounting for shorter lags. Spikes can inform candidate AR/MA orders but do not prove forecast performance.
- **Source provenance:** Reports and plot titles distinguish real UCI data, synthetic simulated data and unknown source type. Synthetic results must never be presented as measured household behavior.

Run Step 5 tests:
```bash
python -m pytest backend/tests/test_time_series_analysis.py -q
python -m pytest backend/tests -q
```

**Step 5 viva checkpoint**

**Q: What does stationarity mean in time-series analysis?**  
A: A weakly stationary series has a stable mean and variance over time, and its covariance depends on lag rather than absolute time. Many classical time-series methods work best when the series is stationary or made stationary through an appropriate transformation.

**Q: Why do we run both ADF and KPSS?**  
A: Their null hypotheses are opposite. ADF tests for a unit root, while KPSS tests for level stationarity. Considering both gives a more informative diagnostic than treating one p-value as a definitive answer.

**Q: What is the difference between ACF and PACF?**  
A: ACF measures correlation between observations separated by a lag. PACF estimates the association at a lag after accounting for shorter lags. Both can help suggest candidate model orders, but final model selection requires chronological holdout evaluation.


## Step 6: Chronological split and baseline forecasts

Run after Step 3 has generated the hourly and daily processed series:
```bash
python scripts/baseline_forecasting.py
```

Optional arguments:
```bash
python scripts/baseline_forecasting.py --test-size 0.2
python scripts/baseline_forecasting.py --processed-dir backend/data/processed --output-dir backend/reports/baselines
```

Implemented in `scripts/baseline_forecasting.py`, with tests in `backend/tests/test_baseline_forecasting.py`. The script independently evaluates the hourly and daily series using a chronological 80/20 split by default. The script first reindexes each series to a regular hourly/daily grid, interpolates only complete internal missing runs of at most three bins, and selects the longest remaining contiguous segment. The earlier observations are training data and the final observations are the held-out test period. The split is not shuffled, and the baseline forecasts are formed only from training observations. The selected segment and interpolation counts are saved in the JSON report.

### Baselines

- **Naive:** Predict the last training observation for every timestamp in the test horizon. It is a simple reference for checking whether more advanced models add value.
- **Seasonal naive:** Repeat the final complete season in the training data, using 24 observations for hourly data (daily cycle) and 7 observations for daily data (weekly cycle). Test observations are never used to build the repeating pattern.

### Evaluation and outputs

Metrics are calculated separately for both baseline models and both frequencies:
- **MAE:** Average absolute prediction error, in kW.
- **RMSE:** Square root of the average squared error, in kW; larger errors receive more weight.
- **MAPE:** Mean absolute percentage error. Zero and near-zero actual values are excluded because percentage error is undefined or unstable there. The report states how many observations were excluded. MAE and RMSE still use every test observation.

Generated files are written to `backend/reports/baselines/`:
- `baseline_results.json`: source provenance, train/test date boundaries, observation counts, metrics and method notes.
- `baseline_metrics.csv`: one row per series/model combination.
- `hourly_baseline_forecasts.csv` and `daily_baseline_forecasts.csv`: timestamps, actual values and both baseline forecasts for the full holdout.
- `hourly_baseline_forecasts.png` and `daily_baseline_forecasts.png`: visual comparison on a readable subset of the holdout.

These are reference baselines, not advanced fitted models. Do not claim a baseline is “best” until the same held-out period is used to compare it with later candidates. The real-data metrics must come from running the script locally; none are hardcoded in this README.

Run Step 6 tests:
```bash
python -m pytest backend/tests/test_baseline_forecasting.py -q
python -m pytest backend/tests -q
```

**Step 6 viva checkpoint**

**Q: Why not randomly split time-series observations?**  
A: A random split can place later observations in training and earlier observations in testing. That leaks future information into the training set. A chronological split imitates the real task: use the past to predict the future.

**Q: What is a naive baseline?**  
A: It repeats the last observed training value. It is intentionally simple and gives us a reference score that more complex models should improve upon.

**Q: How is seasonal naive different from naive?**  
A: Naive repeats one final value for the whole forecast horizon. Seasonal naive repeats the last observed cycle, such as the last 24 hourly readings or the last 7 daily readings, to represent recurring patterns.

**Q: Why report MAE and RMSE as well as MAPE?**  
A: MAE is interpretable in the target's units, RMSE penalizes large errors more strongly, and MAPE expresses percentage error. MAPE is unreliable around zero, so WattCast reports its valid sample count and excludes zero or near-zero actuals from that metric only.


## Step 7: Statistical forecasting candidates

Run after Step 3 preprocessing has generated both processed series:
```bash
python scripts/statistical_forecasting.py
```

Optional arguments:
```bash
python scripts/statistical_forecasting.py --test-size 0.2
python scripts/statistical_forecasting.py --processed-dir backend/data/processed --output-dir backend/reports/statistical_models
```

Implemented in `scripts/statistical_forecasting.py`, with tests in `backend/tests/test_statistical_forecasting.py`. It reuses the Step 6 data-cleaning, regularization and chronological-split helpers so candidate metrics use the same policy and 80/20 holdout by default.

### Candidate models

- **Holt-Winters:** Exponential smoothing with additive trend and additive seasonality. The seasonal period is 24 for hourly readings and 7 for daily readings.
- **ARIMA(1,1,1):** An autoregressive integrated moving-average model with one autoregressive term, one difference and one moving-average term.
- **SARIMA:** `SARIMA(1,0,1) × (1,0,0,s)`, where `s=24` hourly or `s=7` daily. The seasonal autoregressive term represents dependence at the seasonal lag.

All three models are fitted on the training segment only. The held-out future observations are used only for evaluation. Candidate orders are initial, interpretable specifications, not hyperparameter-optimized winners. Each model fit is isolated: a failed fit is recorded in the report rather than hiding the other candidates' results. Warnings are also retained for inspection.

### Outputs

Generated files are written to `backend/reports/statistical_models/`:
- `statistical_model_results.json`: provenance, split boundaries, model specifications, fit status, warnings and metrics.
- `statistical_model_metrics.csv`: one row per series/model, including explicit failure status when a fit fails.
- `hourly_statistical_forecasts.csv` and `daily_statistical_forecasts.csv`: actual holdout values and forecasts from successful models.
- `hourly_statistical_forecasts.png` and `daily_statistical_forecasts.png`: actual values and successful candidate forecasts over a readable holdout window.

Run Step 7 tests:
```bash
python -m pytest backend/tests/test_statistical_forecasting.py -q
python -m pytest backend/tests -q
python scripts/statistical_forecasting.py
```

**Step 7 local verification:** The user confirmed the pipeline completed on real UCI data. The reported chronological splits were 14,869 training / 3,718 test observations hourly (test period 2009-01-09 03:00 through 2009-06-13 00:00) and 1,072 training / 269 test observations daily (test period 2009-11-22 through 2010-08-17). The focused statistical-model tests and the full backend suite both passed after the import fix; the full suite reported 41 passed.

**Step 7 viva checkpoint**

**Q: What is Holt-Winters exponential smoothing?**  
A: It forecasts using smoothed estimates of the level, trend and seasonal pattern. WattCast uses additive trend and additive seasonality as a first candidate because the series has daily or weekly recurring patterns.

**Q: What does ARIMA(1,1,1) mean?**  
A: The model has one autoregressive term, applies first differencing once, and includes one moving-average error term. These orders are candidate settings, not proof that the data's true process has exactly this structure.

**Q: How is SARIMA different from ARIMA?**  
A: SARIMA adds seasonal structure to the ARIMA framework. In WattCast, the seasonal autoregressive term is set at lag 24 for hourly data or lag 7 for daily data.

**Q: Why do all candidate models use the same chronological holdout?**  
A: It makes the comparison fair. Every model predicts the same future period, and no test observations are used to fit the models. Fit warnings and failures must be reported instead of silently omitted.

## Step 8: Lag-feature machine-learning forecasting

Run after preprocessing:
```bash
python scripts/ml_forecasting.py
```

Optional arguments:
```bash
python scripts/ml_forecasting.py --test-size 0.2
python scripts/ml_forecasting.py --processed-dir backend/data/processed --output-dir backend/reports/ml_forecasting --random-state 42
```

Implemented in `scripts/ml_forecasting.py` using scikit-learn's `RandomForestRegressor`.

### Features and leakage control

- **Hourly lags:** previous 1, 2 and 3 hours, plus 24, 48 and 168 hours.
- **Daily lags:** previous 1, 2 and 3 days, plus 7, 14 and 28 days.
- **Rolling means:** prior 24/168 hours for hourly data and prior 7/28 days for daily data. Windows are shifted to exclude the target being predicted.
- **Calendar features:** hour, weekday, month, weekend flag and cyclical sine/cosine encodings; daily data also uses day-of-year features.
- **Model:** deterministic Random Forest with 200 trees, minimum leaf size 2 and default seed 42.
- **Recursive holdout:** every prediction is appended to history; actual test values are never used as lag features for later test predictions. This avoids leakage but errors can accumulate over a long horizon.

### Outputs

Written to `backend/reports/ml_forecasting/`:
- `ml_model_results.json`: source provenance, split boundaries, model parameters, features and metrics.
- `ml_model_metrics.csv`: MAE, RMSE, MAPE and sample counts.
- `hourly_ml_forecasts.csv` and `daily_ml_forecasts.csv`: actuals and recursive predictions.
- `hourly_ml_forecasts.png` and `daily_ml_forecasts.png`: actual-versus-forecast plots.

Run the focused tests, full suite and real-data pipeline:
```bash
python -m pytest backend/tests/test_ml_forecasting.py -q
python -m pytest backend/tests -q
python scripts/ml_forecasting.py
```

**Step 8 viva checkpoint**

**Q: What is a lag feature?**  
A: An earlier target value, such as power one hour ago or the same hour on the previous day, used to predict the current target.

**Q: Why add calendar features?**  
A: Lags describe recent consumption, while calendar features identify when the target time occurs. Cyclical encodings represent the wraparound between the end and start of a cycle, such as hour 23 and hour 0.

**Q: How does WattCast avoid test leakage?**  
A: It trains only on the training segment and forecasts recursively. Actual holdout targets are used for scoring, never as later forecast inputs.

**Q: What is a Random Forest?**  
A: An ensemble of decision trees whose regression predictions are aggregated. It can learn nonlinear relationships, but still must be compared with simple baselines on the same test period.

**Step 8 real-data results (user-reported run):** Random Forest achieved hourly MAE 0.57555 kW, RMSE 0.79169 kW and MAPE 81.668%; daily MAE 0.19519 kW, RMSE 0.26134 kW and MAPE 23.026%. The same chronological holdout was used for the reported Step 6–8 runs. On hourly data, Holt-Winters remains ahead of Random Forest on MAE and RMSE. On daily data, Random Forest has lower MAE and RMSE than the Step 6 seasonal-naive and Step 7 statistical candidates. MAPE is not the only selection criterion and can be sensitive to low actual values. These results are reported from the user's terminal output and are not hardcoded into the application.

## Step 9: Model evaluation, residual diagnostics and caching

Run Steps 6, 7 and 8 first so their forecast CSVs exist, then run:
```bash
python scripts/model_evaluation.py
```

Optional arguments:
```bash
python scripts/model_evaluation.py --test-size 0.2
python scripts/model_evaluation.py --no-cache
python scripts/model_evaluation.py --baseline-dir backend/reports/baselines --statistical-dir backend/reports/statistical_models --ml-dir backend/reports/ml_forecasting --output-dir backend/reports/evaluation
```

Implemented in `scripts/model_evaluation.py`, tested in `backend/tests/test_model_evaluation.py`. It compares six candidate forecasts for each frequency: naive, seasonal naive, Holt-Winters, ARIMA(1,1,1), SARIMA and Random Forest. It aligns forecast outputs on shared timestamps and checks that models use consistent actual holdout values before ranking. Metrics are recomputed from the saved forecast CSVs, not copied from earlier terminal output.

### Residual diagnostics and cache

- Reports MAE, RMSE, zero-safe MAPE and valid sample counts for every model.
- Computes residual bias (actual minus predicted), median error, residual standard deviation, maximum absolute error, lag-1 residual autocorrelation and the fraction of residuals within two residual standard deviations.
- Produces residual-over-time and residual-versus-prediction plots for each model and frequency.
- Writes `model_evaluation_metrics.csv`, `evaluation_results.json`, `evaluation_cache.json` and PNG diagnostic plots under `backend/reports/evaluation/`.
- The cache fingerprint is based on the contents of all source forecast CSVs, the test-size setting and a cache-version value. If the inputs change, metrics are recomputed. Use `--no-cache` to force recomputation. Diagnostic plots are regenerated even on a cache hit so missing images can be restored.

Residuals are defined as actual minus predicted. Positive mean residual indicates average under-forecasting; negative mean residual indicates average over-forecasting. Residual autocorrelation can reveal time structure left unexplained, but it does not alone prove a model is invalid. The MAE/RMSE ranking applies to the shared evaluated timestamps and is not a guarantee of future performance.

Run Step 9 tests:
```bash
python -m pytest backend/tests/test_model_evaluation.py -q
python -m pytest backend/tests -q
python scripts/model_evaluation.py
```

**Step 9 viva checkpoint**

**Q: What is a residual?**  
A: The difference between the actual observation and the model's prediction. WattCast defines residual as actual minus predicted, so a positive residual means the model predicted too low.

**Q: Why inspect residuals after measuring MAE and RMSE?**  
A: MAE and RMSE summarize error size, but residual plots can reveal bias, changing error spread, outliers or patterns over time that a single score hides.

**Q: What does lag-1 residual autocorrelation suggest?**  
A: Neighboring residuals may still be related. That can indicate the model has not captured all temporal structure, though it should be interpreted with the plots and other diagnostics.

**Q: How does the cache know when to refresh?**  
A: WattCast hashes the forecast CSV contents along with the test-size setting and cache version. A changed input produces a different fingerprint and forces recalculation.

## Step 10: FastAPI backend and API contracts

The read-only API is implemented in `backend/app/main.py`. It serves saved artifacts produced by Steps 3–9; it does not rerun expensive model fitting for each browser request.

Start it from the repository root after running the relevant pipeline scripts:
```powershell
python -m uvicorn backend.app.main:app --reload
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Confirms the API process is running. It does not guarantee report files exist. |
| `GET /api/summary` | Returns EDA descriptive statistics and evaluation rankings. |
| `GET /api/series/hourly` or `/api/series/daily` | Returns the most recent cleaned observations; `limit` is between 1 and 2,000. |
| `GET /api/eda` | Returns the EDA summary and available intraday, weekday and monthly profile tables. |
| `GET /api/decomposition/hourly` or `/api/decomposition/daily` | Returns saved observed, trend, seasonal and residual components. |
| `GET /api/models/hourly` or `/api/models/daily` | Returns holdout model metrics and residual diagnostics. |
| `GET /api/forecast/hourly?model=holt_winters` | Returns stored chronological holdout predictions for a supported model. |

Supported model identifiers are `naive`, `seasonal_naive`, `holt_winters`, `arima_1_1_1`, `sarima`, and `random_forest_lag_calendar`. Forecast responses explicitly describe themselves as saved holdout predictions, not a newly generated future forecast.

### Validation and errors

- Pydantic response models define stable health, time-series and forecast response shapes.
- Request limits and model identifiers are validated; invalid requests return HTTP 422.
- Missing pipeline outputs return HTTP 404 with an actionable error code/message. Unreadable or malformed artifacts return a structured HTTP 500 error.
- CORS permits the local Vite development origins and only enables GET requests.
- Report/data paths default to `backend/data/processed` and `backend/reports`. Optional environment variables `WATTCAST_PROCESSED_DIR` and `WATTCAST_REPORTS_DIR` can override them.

### Tests

```powershell
python -m pytest backend/tests/test_api.py -q
python -m pytest backend/tests -q
```

The API contract tests cover health, sorted/limited series output, missing artifacts, summary composition, invalid models, saved holdout predictions and decomposition schema. Local verification: 8 API tests passed; the complete backend suite passed 65 tests in 24.49 seconds. Uvicorn started successfully, and the health endpoint, Swagger docs and OpenAPI schema returned HTTP 200. Pytest reported one non-blocking `StarletteDeprecationWarning` about the current `httpx` integration with `starlette.testclient`; it did not cause test failures.

**Step 10 viva checkpoint**

**Q: Why does the API read saved artifacts instead of training a model per request?**  
A: Training and statistical fitting can be expensive. Precomputed artifacts keep requests fast and reproducible; a future forecasting service could generate new forecasts through a separate controlled workflow.

**Q: What is the difference between HTTP 404 and HTTP 422 here?**  
A: A 404 means the requested saved artifact or result does not exist. A 422 means the request contains an invalid parameter, such as an unsupported model identifier.

**Q: Does `/api/health` mean the whole application is ready?**  
A: No. It confirms the API process is responding. Data and report endpoints can still return 404 if earlier pipeline steps have not generated their artifacts.

## Methodology
1. Repository scaffold and project documentation.
2. Dataset download and reproducible synthetic fallback.
3. Data loading, inspection, cleaning and hourly/daily resampling.
4. Exploratory data analysis (EDA) and chart artefacts.
5. Additive hourly/daily decomposition, ADF/KPSS stationarity diagnostics, first-difference checks and daily ACF/PACF.
6. Chronological train/test split and baseline forecasts.
7. Holt-Winters, ARIMA and SARIMA candidates.
8. Lag/calendar-feature machine-learning model.
9. Holdout evaluation, residual diagnostics and cached artefacts.
10. FastAPI endpoints, schemas, error handling and tests.
11. React/Vite/Tailwind frontend scaffold and API client.
12. Overview and data explorer dashboard.
13. Decomposition and model-comparison views.
14. Forecast page and controls.
15. UI polish, responsive behavior and integration checks.
16. Final audit, documentation, reproducibility notes and viva preparation.

All numerical results and plots will be generated by code from the selected dataset. No model scores or forecasts will be hardcoded in the UI.

## Models planned
- Naive baseline
- Seasonal-naive baseline
- Holt-Winters exponential smoothing
- ARIMA / SARIMA
- Random Forest using lag, rolling-window and calendar features (Step 8 candidate)

**Step 7 statistical-model results (real UCI data):** The user ran the pipeline locally on chronological holdout periods. MAE and RMSE are in kW; MAPE is a percentage. These figures are reported as observed and are not hardcoded into the application.

| Frequency | Model | MAE (kW) | RMSE (kW) | MAPE |
|---|---|---:|---:|---:|
| Hourly | Holt-Winters | 0.53257 | 0.75694 | 58.426% |
| Hourly | ARIMA(1,1,1) | 0.69846 | 0.86233 | 109.085% |
| Hourly | SARIMA | 0.68795 | 0.86905 | 93.899% |
| Daily | Holt-Winters | 0.31817 | 0.41312 | 43.983% |
| Daily | ARIMA(1,1,1) | 0.35632 | 0.44352 | 49.126% |
| Daily | SARIMA | 0.28683 | 0.36199 | 34.535% |

Among these candidates, Holt-Winters has the lowest hourly MAE and RMSE, while SARIMA has the lowest daily MAE and RMSE. MAPE can be unstable for near-zero actual readings, so interpret it alongside MAE and RMSE. These rankings apply only to this dataset and test split; final selection requires comparison with Step 6 baselines under the same evaluation policy.

## Installation and running
**Current status:** Steps 1–9 are implemented and locally verified. The Step 9 focused tests (7/7), full backend suite (57/57), and real-data evaluation pipeline completed successfully. The backend API and frontend remain future steps.

Planned local setup:
- Python 3.10 or newer
- Node.js LTS and npm
- Git

The final README will include one-command startup and separate manual backend/frontend commands after those scripts exist.

## API
Planned API base path: `/api`

| Endpoint | Purpose | Status |
|---|---|---|
| `GET /api/health` | Service health | Planned |
| `GET /api/summary` | Consumption KPIs and best model | Planned |
| `GET /api/series` | Hourly/daily time series | Planned |
| `GET /api/eda` | Exploratory summaries | Planned |
| `GET /api/decomposition` | Trend, seasonal and residual components | Planned |
| `GET /api/stationarity` | ADF/KPSS and ACF/PACF results | Planned |
| `GET /api/models` | Model comparison and diagnostics | Planned |
| `GET /api/forecast` | Forecast and uncertainty intervals | Planned |

## Project structure
```text
WattCast/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   ├── routers/
│   │   ├── schemas/
│   │   └── services/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── reports/
│   │   ├── baselines/           # Generated baseline forecasts (outputs ignored by Git)
│   │   ├── eda/                 # Generated EDA artefacts (outputs ignored by Git)
│   │   ├── statistical_models/  # Generated model forecasts (outputs ignored by Git)
│   │   ├── ml_forecasting/      # Generated ML forecasts (outputs ignored by Git)
│   │   ├── evaluation/          # Generated evaluation metrics/cache/residual plots (outputs ignored by Git)
│   │   └── time_series/         # Generated diagnostics (outputs ignored by Git)
│   ├── models/
│   ├── notebooks/
│   ├── tests/
│   │   ├── test_data_acquisition.py
│   │   ├── test_preprocessing.py
│   │   ├── test_eda.py
│   │   └── test_time_series_analysis.py
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── api/
│       ├── components/
│       ├── hooks/
│       └── pages/
├── scripts/
│   ├── ml_forecasting.py
│   ├── model_evaluation.py
│   ├── data_processing.py
│   ├── download_data.py
│   ├── eda.py
│   ├── generate_synthetic_data.py
│   ├── preprocess_data.py
│   ├── statistical_forecasting.py
│   └── time_series_analysis.py
├── reports/
│   ├── figures/                 # Reserved for selected report figures
│   └── results/                 # Reserved for final result summaries
├── README.md
└── .gitignore
```

## Screenshots
No application screenshots yet. They will be added after the UI is implemented and run successfully.

## Limitations and future work
- Forecast quality depends on data quality and the selected test period.
- The real dataset represents one household and is not a universal representation of every building or grid.
- Synthetic fallback values are simulated and must be labelled as such in charts, reports and results.
- Short-gap interpolation can smooth peaks; long gaps are dropped to avoid fabricating long sequences.
- IQR outlier flags can include legitimate demand peaks.
- Forecast intervals are estimates, not guarantees.
- MAPE can be unstable when actual values are zero or close to zero.
- Future extensions could include weather features, more robust interval calibration and additional datasets.

## Author
**Rick / BethalaRinosh**  
Academic project, B.Tech CSE (AI & ML).

## Progress log
- **Step 1/16 — Repository setup and scaffold:** Created the initial repository structure and documentation skeleton.
- **Step 2/16 — Dataset acquisition:** Added the official UCI downloader, safe extraction of the expected text file, a reproducible synthetic hourly generator, and JSON provenance metadata.
- **Step 3/16 — Loading, inspection, cleaning and resampling:** Added a shared loader for the UCI semicolon-delimited source and synthetic CSV, inspection summary, IQR outlier reporting, limited time interpolation, hourly/daily resampling, saved output/report paths and preprocessing tests. The user verified the real-data preprocessing locally.
- **Step 4/16 — Exploratory data analysis:** Added `scripts/eda.py` to compute descriptive statistics and intraday, weekday and monthly profiles, write JSON/CSV summaries and produce four labelled charts. The user verified the script on real data; 4 EDA tests and all 9 then-existing backend tests passed.
- **Step 5/16 — Time-series diagnostics:** Added additive hourly/daily decomposition, ADF and KPSS tests on daily levels and first differences, daily ACF/PACF tables and plots, source-labelled JSON outputs, and focused tests. The user confirmed the diagnostics ran on real UCI data and 13 tests passed. A subsequent repository audit added edge-case tests and hardening changes; the expanded suite must pass before Step 6.
- **Repository audit before Step 6:** Found and fixed partial interpolation across long missing runs, added explicit invalid-timestamp and duplicate counts to preprocessing inspection, hardened EDA input validation, made UCI extraction atomic so a failed refresh cannot truncate an existing dataset, corrected downloader provenance when refreshing an existing real dataset fails, made script imports package-safe, and ignored generated report files while retaining report directories. Added offline tests for acquisition, fallback provenance and data-quality edge cases. The user subsequently confirmed all 21 audit tests passed and the preprocessing, EDA and time-series diagnostic scripts completed on the real dataset.
- **Step 6/16 — Chronological split and baselines (verified locally):** Added chronological train/test splitting for hourly and daily data, naive and seasonal-naive forecasts, MAE/RMSE/zero-safe MAPE metrics, JSON/CSV reports, forecast plots and tests. The user confirmed 13 Step 6 tests, all 34 backend tests and the real-data baseline script passed; the working tree was clean.
- **Step 7/16 — Statistical forecasting candidates (verified locally):** Added Holt-Winters, ARIMA(1,1,1) and SARIMA candidates, reusing Step 6's regularization and chronological holdout. Fixed direct-script imports with a package/direct-script fallback. The user confirmed the pipeline completed on real UCI data and all 41 backend tests passed. Hourly Holt-Winters achieved MAE 0.53257 kW / RMSE 0.75694 kW; daily SARIMA achieved MAE 0.28683 kW / RMSE 0.36199 kW among the Step 7 candidates. Full metrics and the tested split periods are documented above.
- **Step 8/16 — Lag-feature machine-learning model (verified locally):** Added a Random Forest using past target lags, shifted rolling means and calendar features for hourly and daily data. The user confirmed the real-data pipeline completed and later confirmed all 9 focused tests and all 50 backend tests passed. Reported scores: hourly MAE 0.57555 kW / RMSE 0.79169 kW / MAPE 81.668%; daily MAE 0.19519 kW / RMSE 0.26134 kW / MAPE 23.026%. On hourly data Holt-Winters remains ahead on MAE/RMSE; on daily data Random Forest leads the Step 6-8 candidates on MAE/RMSE.
- **Step 9/16 — Model evaluation, residual diagnostics and caching (verified locally):** Added a shared evaluator for six forecast candidates per frequency, consistent-timestamp checks, residual statistics and plots, content-hash cache invalidation, and focused tests. The user confirmed 7 focused tests and all 57 backend tests passed, and the real-data evaluation pipeline completed on the shared holdouts (3,718 hourly and 269 daily observations). Holt-Winters ranked best on hourly MAE/RMSE; Random Forest ranked best on daily MAE/RMSE. This ranking is specific to the tested data and holdout.

## Viva checkpoint
**Q: Why do we resample the original minute-level data to hourly and daily data?**  
A: The original dataset contains over two million minute-level rows. Hourly and daily summaries reduce processing cost and make broad patterns easier to interpret. We use mean active power in kW as the forecasting target. For an hourly interval, the average power in kW is numerically equivalent to approximate energy in kWh over that hour.

**Q: Why are outliers reported but not automatically deleted?**  
A: A high reading might be a real demand peak rather than a measurement error. The IQR rule flags suspicious values for inspection, but deleting them automatically could remove the very peaks the forecasting model should learn.

**Q: Why is a chronological test split important?**  
A: Time-series forecasting predicts future values from past observations. A random split can leak future patterns into training data and make evaluation unrealistically optimistic. The model should be tested on observations that occur after the training period.


**EDA viva checkpoint**

**Q: What is the purpose of exploratory data analysis before modelling?**  
A: EDA helps us understand the target's distribution, date coverage and recurring time patterns before choosing models. It can reveal patterns or data-quality issues worth investigating, but it does not prove why those patterns occur.

**Q: Why calculate an average profile by hour and weekday?**  
A: Grouping observations by hour-of-day or day-of-week can reveal recurring daily and weekly patterns. These are descriptive summaries; the later evaluation step is needed to determine whether a forecasting model can predict future values accurately.


## Step 11: React frontend foundation

The frontend lives in `frontend/` and uses React, TypeScript, Vite, Tailwind CSS, Framer Motion, Lucide icons and Recharts. This step creates the app shell and API integration foundation; the data explorer, decomposition visualisations, model comparison and forecast experience are built in Steps 12–14.

### Start the frontend

From the repository root, pull the latest changes, then run:

```powershell
cd frontend
Copy-Item .env.example .env
npm install
npm run dev
```

Vite is configured to use `http://127.0.0.1:5173`. Keep the FastAPI process running in a second terminal from the repository root:

```powershell
python -m uvicorn backend.app.main:app --reload
```

Open the local URL printed by Vite. If your API is hosted elsewhere, edit `frontend/.env`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The value is the API origin, without an `/api` suffix. Vite reads environment variables at startup, so restart the dev server after changing the file.

### What Step 11 includes

- Responsive React single-page application shell with desktop sidebar and compact mobile navigation.
- Tailwind design tokens and a restrained light/dark theme toggle.
- Framer Motion page transitions and Lucide interface icons.
- Reusable typed API client in `frontend/src/api/client.ts`, with HTTP error parsing and a clear message when the API cannot be reached.
- Initial real connection check against `GET /api/health`; connection state is displayed as checking, connected or offline.
- Recharts is installed for later data visualisations. It is intentionally not used to display invented metrics in this scaffold.
- Strict TypeScript settings and a production build script: `npm run build`.

### API and data integrity

The frontend reads its API base URL from `VITE_API_BASE_URL`, falling back to `http://127.0.0.1:8000`. The current screen checks API health only. It does not yet load summary metrics or chart data. Steps 12–14 will connect those screens to the existing FastAPI endpoints and use the saved analysis artifacts.

### Step 11 local verification checklist

After `npm install` completes:

```powershell
npm run build
npm run dev
```

Check the following in the browser:
1. The app loads at `http://127.0.0.1:5173`.
2. With FastAPI running, the status changes to **API connected**.
3. With FastAPI stopped, the screen shows a useful connection error instead of crashing.
4. The theme toggle works, and the navigation remains usable at mobile widths.
5. The production build finishes without TypeScript or Vite errors.

**Verification status:** The scaffold is committed to GitHub. Do not treat the local build/browser checklist as passed until its output has been checked on the development machine.
