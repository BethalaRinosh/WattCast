# WattCast: Final Viva and Technical Preparation Guide

> Use this guide alongside the source code and the generated reports. If a result differs after rerunning a pipeline, use the newly generated report rather than memorising an old number.

## 1. Project in one minute

WattCast is an end-to-end electricity-consumption time-series analysis and forecasting application. It takes historical household electricity readings, cleans and aggregates the data into hourly and daily series, explores temporal structure, trains several forecasting candidates, evaluates them on a chronological holdout, and exposes the saved analysis through a FastAPI backend and a React dashboard.

**Core target:** mean active power, measured in kilowatts (kW).

**Dataset:** UCI Individual Household Electric Power Consumption, one household near Paris, December 2006 to November 2010, with 2,075,259 one-minute records in the original dataset.

**Models compared:** Naive, seasonal naive, Holt-Winters, ARIMA(1,1,1), SARIMA and Random Forest with lag/calendar features.

**Important boundary:** The Forecasts page visualises saved predictions on the chronological holdout. It does not generate a new forecast beyond the observed dataset.

## 2. Architecture and data flow

```text
UCI raw TXT / synthetic fallback
             |
             v
 scripts/download_data.py
             |
             v
 scripts/data_processing.py
             |
       +-----+------+
       |            |
       v            v
 hourly CSV     daily CSV
       |            |
       +------+-----+
              |
    +---------+-------------------------------+
    |         |          |          |         |
    v         v          v          v         v
   EDA   Time-series  Baselines  Statistical  Random
         diagnostics             models       Forest
    |         |          |          |         |
    +---------+----------+----------+---------+
                         |
                         v
              scripts/model_evaluation.py
                         |
                         v
              backend/reports/*.json/csv/png
                         |
                         v
                 FastAPI (GET /api/...)
                         |
                         v
              React + TypeScript + Vite
                         |
                         v
          Recharts dashboard in the browser
```

The analysis scripts do the computation and save artifacts. FastAPI reads those artifacts and returns structured responses. The frontend fetches the API and renders the results. The browser does not train models.

### Main folders

| Path | Responsibility |
|---|---|
| `scripts/` | Downloading, preprocessing, EDA, diagnostics, forecasting and evaluation pipelines |
| `backend/data/raw/` | Downloaded UCI input or generated synthetic input |
| `backend/data/processed/` | Clean hourly/daily active-power series and preprocessing report |
| `backend/reports/` | Saved analysis, forecast, evaluation and correlation artifacts |
| `backend/app/main.py` | FastAPI application, validation, artifact loading and API routes |
| `backend/tests/` | Automated backend tests |
| `frontend/src/api/client.ts` | Typed frontend API client |
| `frontend/src/pages/` | Dashboard pages |
| `frontend/src/components/` | Shared UI components |

## 3. Dataset and preprocessing

### What the data represents

The UCI dataset contains measurements from one household. It is not a representative sample of every household, building or electricity grid. The original file is semicolon-delimited. The forecasting pipeline focuses on `Global_active_power`, converted to the project field `active_power_kw`.

The correlation pipeline separately re-reads the original UCI file to retain all seven numeric measurements:
- Active power
- Reactive power
- Voltage
- Global intensity
- Sub-metering 1, 2 and 3

### Cleaning decisions

1. Parse the UCI date and time columns into a timestamp index.
2. Treat `?`, empty, non-numeric and infinite measurements as missing/invalid.
3. Drop invalid timestamps, sort chronologically and remove duplicate timestamps.
4. Resample the target using interval means to hourly and daily frequency.
5. Interpolate only complete internal missing runs of at most three resampled bins.
6. Leave longer gaps unfilled and drop remaining missing bins from the modelling series.
7. Detect IQR outliers for reporting, but do not automatically delete them.

**Why keep outliers?** A high consumption value may be a genuine demand peak. Removing it automatically could erase important behaviour.

**Why hourly and daily means?** They reduce the volume of minute-level data and expose daily/weekly patterns at two useful time scales. Mean kW over a one-hour interval is numerically equivalent to approximate kWh used during that hour.

### Real versus synthetic data

Synthetic data is a fallback for development and testing. Its source must be labelled as simulated. The current synthetic generator contains only active power, so cross-variable correlation is reported as unavailable rather than inventing extra features.

## 4. Exploratory data analysis

The EDA stage calculates descriptive statistics and average profiles by hour of day, weekday and month. It produces summary JSON, profile CSVs and plots.

EDA answers questions such as:
- What is the overall distribution and date coverage?
- Which hours have higher average active power?
- Do average profiles differ across weekdays?
- Are there broad patterns worth investigating?

EDA is descriptive. It does not establish why a pattern occurs and does not prove forecast accuracy.

## 5. Time-series diagnostics

### Additive decomposition

The project uses additive seasonal decomposition:

`Observed = Trend + Seasonal + Residual`

- **Observed:** original regularised series.
- **Trend:** smoothed long-term movement.
- **Seasonal:** repeating component for the selected period.
- **Residual:** what remains after trend and seasonal components are removed.

The hourly seasonal period is 24 (a daily cycle); the daily period is 7 (a weekly cycle). Decomposition is descriptive, not itself a forecasting model. It assumes additive seasonal effects.

### Regular intervals and gaps

Seasonality and autocorrelation calculations assume equally spaced observations. The diagnostics regularise the series, interpolate only short internal gaps up to three bins, and select the longest remaining contiguous segment. Longer gaps are not bridged.

### Stationarity: ADF and KPSS

A stationary series has statistical properties such as mean and variance that are stable over time (under the weak-stationarity interpretation).

- **ADF null hypothesis:** the series has a unit root (non-stationary). A small p-value is evidence against this null.
- **KPSS null hypothesis:** the series is level-stationary. A small p-value is evidence against this null.

Their null hypotheses are opposite, so read both tests together. WattCast runs these tests on daily levels and first differences. A p-value is evidence under a test's assumptions, not an absolute proof.

### ACF and PACF

- **ACF:** correlation between a series and its lagged values.
- **PACF:** association at a lag after accounting for shorter lags.

WattCast calculates daily ACF/PACF values up to the maximum lag allowed by the sample, capped at 60 days. Spikes can suggest candidate model orders, but model quality must be judged on held-out data.

## 6. Forecasting methods

All model candidates are evaluated on a chronological holdout. The split is not shuffled.

| Model | What it does | Role in WattCast |
|---|---|---|
| Naive | Repeats the last training observation across the test horizon | Simple reference baseline |
| Seasonal naive | Repeats the last complete training season (24 hourly or 7 daily observations) | Seasonal reference baseline |
| Holt-Winters | Exponential smoothing of level, trend and seasonal components | Statistical candidate with additive trend and seasonality |
| ARIMA(1,1,1) | One autoregressive term, first differencing and one moving-average term | Non-seasonal statistical candidate |
| SARIMA | Seasonal ARIMA structure, configured as (1,0,1) × (1,0,0,s) | Seasonal statistical candidate, s=24 hourly or s=7 daily |
| Random Forest lag/calendar | Tree ensemble using past target values, shifted rolling statistics and calendar features | Non-linear machine-learning candidate |

These are initial interpretable specifications, not an exhaustive hyperparameter search. Do not claim that a model is universally best. The best candidate can differ between hourly and daily data and between evaluation periods.

### Avoiding leakage

- Training observations occur before test observations.
- Model fitting uses training data only.
- Test observations are used for evaluation, not fitting.
- Rolling features for the Random Forest must be shifted so they use past values rather than the current target.
- The evaluator aligns model predictions by timestamp and calculates metrics on shared valid timestamps.

## 7. Evaluation and residual diagnostics

### Metrics

Let (y_t) be actual active power and (hat y_t) be predicted active power.

**MAE**

[
MAE = rac{1}{n}sum_{t=1}^{n}|y_t-hat y_t|
]

Average absolute error, expressed in kW. Lower is better.

**RMSE**

[
RMSE = sqrt{rac{1}{n}sum_{t=1}^{n}(y_t-hat y_t)^2}
]

Penalises larger errors more strongly than MAE. Lower is better.

**MAPE**

[
MAPE = rac{100}{n}sum_{t=1}^{n}left|rac{y_t-hat y_t}{y_t}ight|
]

Expresses error as a percentage. It is unstable or undefined when actual values are zero or close to zero. WattCast excludes zero/near-zero actuals from MAPE and reports the metric's valid sample count; MAE and RMSE still use their valid evaluation observations.

### Residuals

WattCast defines residual/error as:

`Residual = Actual - Predicted`

- Positive mean error means under-forecasting on average.
- Negative mean error means over-forecasting on average.
- Large residual spread indicates less consistent errors.
- Lag-1 residual autocorrelation can indicate remaining time structure.
- Residual-versus-predicted plots can reveal patterns such as changing error spread.

Residual diagnostics help explain model errors. They do not replace holdout metrics.

### Evaluation cache

The evaluation pipeline fingerprints the forecast artifact contents and test-size setting using SHA-256. If the inputs are unchanged and the cache is valid, it reuses the saved evaluation result. Changed input artifacts invalidate the fingerprint and trigger recalculation.

## 8. Correlation analysis

The correlation pipeline re-reads the original UCI source and retains all seven numeric measurements. It resamples them to hourly and daily means and calculates:

- **Pearson correlation:** linear association.
- **Spearman correlation:** monotonic association based on ranks.

The pipeline uses pairwise complete observations and does not impute missing feature values for this calculation. It ranks other variables against active power by absolute Pearson correlation and generates hourly/daily heatmaps.

Correlation does not imply causation. A high correlation does not prove that one measurement causes another. Aggregation frequency can also change the observed coefficient.

## 9. Backend and frontend

### FastAPI

The API is read-only and serves saved artifacts. It validates frequency, query limits and model identifiers; returns structured JSON; and produces clear errors when required artifacts are missing or unreadable.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Confirms the API process is running; does not guarantee reports exist |
| `GET /api/summary` | Compact EDA and evaluation summary |
| `GET /api/series/hourly` | Recent hourly observations |
| `GET /api/series/daily` | Recent daily observations |
| `GET /api/eda` | EDA summary and profile tables |
| `GET /api/decomposition/hourly` | Saved hourly observed/trend/seasonal/residual components |
| `GET /api/decomposition/daily` | Saved daily decomposition components |
| `GET /api/models/hourly` | Hourly model metrics and diagnostics |
| `GET /api/models/daily` | Daily model metrics and diagnostics |
| `GET /api/forecast/{series}?model=...&limit=...` | Saved holdout predictions for a selected model |
| `GET /api/correlation/hourly` | Hourly Pearson/Spearman results |
| `GET /api/correlation/daily` | Daily Pearson/Spearman results |

Series/forecast limits are bounded from 1 to 2,000. Forecast endpoint responses explicitly state that the returned values are saved chronological holdout predictions, not newly generated future forecasts.

### React frontend

The frontend uses React, TypeScript, Vite, Tailwind CSS, Framer Motion, Lucide icons and Recharts. Its typed API client communicates with FastAPI. Dashboard pages include Overview, Data explorer, Correlation analysis, Decomposition, Model lab and Forecasts. Charts visualise server-provided analysis artifacts.

## 10. Run the project locally

Run commands from the repository root unless the command explicitly changes directory.

### Backend environment and data

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python scripts/download_data.py
python scripts/preprocess_data.py
python scripts/eda.py
python scripts/time_series_analysis.py
python scripts/baseline_forecasting.py
python scripts/statistical_forecasting.py
python scripts/ml_forecasting.py
python scripts/model_evaluation.py
python scripts/correlation_analysis.py
```

If dataset download is unavailable, use the documented synthetic fallback. Clearly label all synthetic results.

### Start backend

From repository root, in Terminal 1:

```powershell
python -m uvicorn backend.app.main:app --reload
```

API docs are available at `http://127.0.0.1:8000/docs`.

### Start frontend

In Terminal 2:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open the Vite URL, normally `http://127.0.0.1:5173`. Keep the backend running. The frontend API origin is configured through `VITE_API_BASE_URL`.

### Automated checks

From repository root:

```powershell
python -m pytest backend/tests -q
cd frontend
npm run build
```

A successful frontend build runs TypeScript project checks and then creates the Vite production bundle.

## 11. Demo sequence for an academic presentation

1. Start the backend and frontend. Show the dashboard and API connection state.
2. Open **Overview** and explain the target, source provenance and summary indicators.
3. Open **Data explorer** and show the hourly/daily historical series.
4. Open **Decomposition**. Explain observed, trend, seasonal and residual components and the periods 24/7.
5. Open **Correlation analysis**. Switch hourly/daily and Pearson/Spearman. Emphasise association, not causation.
6. Open **Model lab**. Compare MAE, RMSE and MAPE, explain that lower error is better and discuss the shared holdout.
7. Open **Forecasts**. Select a frequency and model; explain that the chart compares actual and saved predicted values on the holdout, not future dates.
8. If asked about backend implementation, open `http://127.0.0.1:8000/docs` and show the read-only API endpoints.

Do not invent or memorise one static “best model” result. Read the current Model Lab results for the dataset and evaluation artifacts actually loaded in the demo.

## 12. Viva questions and answers

### Project fundamentals

**Q1. What problem does WattCast solve?**  
It provides a reproducible workflow to inspect historical household electricity consumption, analyse temporal patterns, compare forecasting methods and view results in a dashboard.

**Q2. Why is this a time-series problem?**  
Each observation is associated with a timestamp, and the order and temporal spacing matter. Past consumption can contain trend, seasonality and autocorrelation that are relevant to later observations.

**Q3. What is the target variable?**  
Mean active power in kilowatts, represented as `active_power_kw`.

**Q4. Is kW the same as kWh?**  
No. kW is a rate of energy use (power); kWh is energy. For a one-hour interval, average kW multiplied by one hour gives approximate kWh consumed during that interval. The project forecasts mean active power in kW.

**Q5. Is the dataset representative of all households?**  
No. It represents one household near Paris over the dataset period. Results should not be generalised to all homes or the electrical grid.

### Data and preprocessing

**Q6. Why did you resample minute-level data?**  
Hourly and daily aggregation reduces data volume and makes daily and weekly patterns easier to inspect and model.

**Q7. Why use the mean during resampling?**  
The target is average active power over each interval. The mean preserves the interpretation of average power for that interval.

**Q8. Why not fill every missing value?**  
Long-gap interpolation would fabricate extended sequences and could hide real changes. The pipeline fills only short, complete internal gaps of at most three bins.

**Q9. Why not remove IQR outliers?**  
Some high readings may be real consumption peaks. The pipeline flags them for inspection but retains them.

**Q10. What happens if the real dataset cannot be downloaded?**  
A reproducible synthetic fallback is available. Its provenance is labelled synthetic and it must not be described as real measured data.

### Time-series concepts

**Q11. What is stationarity?**  
A stationary series has stable statistical properties over time. In weak stationarity, the mean and variance are stable and covariance depends on lag rather than absolute time.

**Q12. Why use both ADF and KPSS?**  
ADF tests a unit-root null; KPSS tests a stationarity null. Their opposite null hypotheses provide complementary evidence.

**Q13. What is differencing?**  
Differencing replaces a value with its change from the previous observation. First differencing is (x_t-x_{t-1}) and can reduce trend or unit-root behaviour.

**Q14. What is decomposition?**  
It separates a series into estimated trend, seasonal and residual components. WattCast uses additive decomposition as a descriptive diagnostic.

**Q15. Why period 24 for hourly and 7 for daily?**  
They represent a 24-hour daily cycle in hourly data and a 7-day weekly cycle in daily data.

**Q16. What is the difference between ACF and PACF?**  
ACF measures correlation with lagged values. PACF estimates the association at a lag after accounting for shorter lags.

### Forecasting and evaluation

**Q17. Why use a chronological split instead of random train/test splitting?**  
A random split can place future observations in training and past observations in testing. A chronological split evaluates the realistic task of predicting later observations from earlier ones.

**Q18. What is a baseline model?**  
A simple reference method used to judge whether more complex models add predictive value. WattCast includes naive and seasonal-naive baselines.

**Q19. Explain Holt-Winters.**  
It is an exponential-smoothing method that models level, trend and seasonality. WattCast uses additive trend and additive seasonality as a candidate.

**Q20. What does ARIMA(1,1,1) mean?**  
It has one autoregressive term, one order of differencing and one moving-average term.

**Q21. How is SARIMA different from ARIMA?**  
SARIMA adds seasonal autoregressive, differencing and moving-average structure to model repeating seasonal dependencies.

**Q22. What is Random Forest doing in this project?**  
It learns a non-linear mapping from engineered past-value and calendar features to the target. Lag and rolling features provide historical context; calendar features encode time position.

**Q23. What is data leakage?**  
Information unavailable at prediction time enters model fitting or feature creation. WattCast uses chronological splitting and past-shifted lag/rolling features to reduce leakage.

**Q24. What is MAE?**  
The average absolute difference between actual and predicted values. It is measured in kW and is easy to interpret.

**Q25. Why use RMSE as well as MAE?**  
RMSE squares errors before averaging, so large errors have more influence. Comparing it with MAE helps reveal whether large misses are disproportionately costly.

**Q26. What is the weakness of MAPE?**  
It becomes undefined at zero and unstable near zero actual values. WattCast excludes zero/near-zero actual values from MAPE and reports the valid count.

**Q27. What are residuals?**  
The differences between actual and predicted values. In WattCast, residual equals actual minus predicted.

**Q28. What does positive mean residual indicate?**  
The model under-forecast on average, because actual minus predicted is positive.

**Q29. Why align timestamps before comparing models?**  
Metrics are only fair when models are evaluated on the same actual timestamps. The evaluation pipeline uses shared valid timestamps.

**Q30. Does the Forecasts page predict future dates beyond the dataset?**  
No. It displays stored predictions for the chronological test/holdout period. It does not fit a new model or forecast beyond the saved holdout.

### Correlation and system design

**Q31. What is Pearson correlation?**  
A measure of linear association between two numeric variables.

**Q32. What is Spearman correlation?**  
A measure of monotonic association based on ranked values. It can capture monotonic relationships that are not strictly linear.

**Q33. Does correlation prove causation?**  
No. Correlation alone cannot show that one variable causes another. Confounding factors and aggregation effects may influence the relationship.

**Q34. Why does correlation use a separate raw-data pipeline?**  
The main forecasting pipeline intentionally reduces the source to one target column. Correlation needs the other UCI measurements, so it re-reads the original file without changing the single-target forecasting contract.

**Q35. Why FastAPI?**  
It provides typed, documented HTTP endpoints and validation, allowing the frontend to retrieve analysis independently of the Python pipeline scripts.

**Q36. Does the API train models?**  
No. It reads saved JSON/CSV artifacts and returns them. The analysis and forecasting scripts generate those artifacts.

**Q37. Why React and TypeScript?**  
React organises the dashboard into reusable components and pages. TypeScript adds compile-time checks to the API response and UI code.

**Q38. What happens if a report artifact is missing?**  
The API returns a structured error indicating that the artifact is missing and that the relevant pipeline must be run. The health endpoint only confirms the API process is running; it does not guarantee all reports exist.

**Q39. What is the evaluation cache for?**  
It avoids recomputing metrics when the forecast inputs have not changed. A SHA-256 fingerprint of input artifacts and test-size settings determines whether the cached result can be reused.

### Limitations and future work

**Q40. What are the main limitations?**  
The source covers one household; external predictors such as weather are not included; model specifications are initial candidates rather than exhaustive tuning; synthetic fallback is not real measurement; MAPE has near-zero limitations; and the dashboard displays saved holdout predictions rather than live future forecasts.

**Q41. What would you improve next?**  
Evaluate on additional households/buildings, add exogenous variables such as weather when available, conduct systematic time-series cross-validation and hyperparameter tuning, investigate prediction intervals, and add a carefully designed future-forecast workflow with explicit retraining and validation.

**Q42. What is the strongest technical decision in the project?**  
Keeping preprocessing, diagnostics, model fitting, evaluation, API delivery and UI rendering as distinct stages makes the workflow reproducible and testable. It also prevents the frontend from inventing or silently recomputing results.

## 13. Honest claims checklist

Use these statements in your presentation:

- “We compare six forecasting candidates on a shared chronological holdout.”
- “The best-performing candidate depends on frequency and the evaluated holdout.”
- “The Forecasts page shows saved holdout predictions, not a live future forecast.”
- “The correlation page reports associations, not causal relationships.”
- “The primary UCI dataset represents one household.”
- “Synthetic data is a development fallback and is labelled separately.”
- “The API serves precomputed artifacts; model fitting happens in scripts.”

Avoid these unsupported claims:

- “Our model predicts electricity demand for any household.”
- “The system forecasts the next week live” (unless a separate future forecasting workflow has actually been implemented and validated).
- “Correlation proves that voltage/intensity causes power consumption.”
- “The model is 99% accurate” without a defined metric, dataset, holdout and reproducible report.
- “The API trains the AI model.”

## 14. Source files worth opening during viva

| Topic | Source |
|---|---|
| Dataset loading and cleaning | `scripts/data_processing.py` |
| EDA | `scripts/eda.py` |
| Decomposition, ADF/KPSS, ACF/PACF | `scripts/time_series_analysis.py` |
| Baseline forecasts and metrics | `scripts/baseline_forecasting.py` |
| Statistical models | `scripts/statistical_forecasting.py` |
| Random Forest features/model | `scripts/ml_forecasting.py` |
| Shared model comparison/cache/residuals | `scripts/model_evaluation.py` |
| Seven-variable correlation | `scripts/correlation_analysis.py` |
| API routes and artifact validation | `backend/app/main.py` |
| Typed API calls | `frontend/src/api/client.ts` |
| Model Lab UI | `frontend/src/pages/ModelLabPage.tsx` |
| Forecast UI | `frontend/src/pages/ForecastsPage.tsx` |
| Decomposition UI | `frontend/src/pages/DecompositionPage.tsx` |

## 15. Final preparation checklist

- [ ] Run preprocessing and analysis pipelines using the dataset you intend to demo.
- [ ] Confirm provenance is real UCI or synthetic and label it correctly.
- [ ] Run the full backend test suite.
- [ ] Run the frontend production build.
- [ ] Start backend and frontend before the presentation.
- [ ] Open the Model Lab and read the current ranking rather than memorising an old score.
- [ ] Be ready to explain chronological splitting, leakage, MAE/RMSE/MAPE and residuals.
- [ ] Be explicit that saved predictions are holdout predictions, not future forecasts.
- [ ] Keep the API docs available as a backup demonstration.
