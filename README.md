# ⚡ WattCast
### Energy Consumption Forecasting using Time Series Analysis
**Tagline:** Every unit tells a story.

WattCast is an academic micro project for the Data Science 3rd Internal Assessment. It analyses historical electricity consumption and compares time-series and machine-learning forecasting methods through a web dashboard.

> **Project status:** Step 3 of 16. Dataset download/generation scripts and the preprocessing pipeline are committed. The pipeline has unit tests in the repository, but execution against the full dataset has not yet been verified in this environment. No forecasting metrics have been generated.

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
6. **Missing values:** Time interpolation bridges at most three consecutive missing resampled bins and only fills internal gaps. Longer gaps are dropped rather than inventing long runs of data.
7. **Outliers:** Report values outside the 1.5×IQR bounds. Do not automatically remove them, because a genuine high-demand period may be an important peak. The IQR rule is a screening tool, not proof of a data error.

Run the preprocessing tests:
```bash
python -m pytest backend/tests/test_preprocessing.py -q
```

**Verification note:** Source files and README are committed on GitHub. Full local execution is not claimed because repository cloning/runtime execution was unavailable in the previous environment. Run the commands above locally to verify the dataset download, preprocessing outputs and tests. The full UCI source has over two million minute-level rows, so processing can take a little time and memory.

## Methodology
1. Problem definition and scope.
2. Data loading and inspection.
3. Cleaning, missing-value handling, outlier checks and resampling.
4. Exploratory data analysis (EDA).
5. Trend/seasonality decomposition, ADF/KPSS stationarity tests and ACF/PACF.
6. Chronological train/test split and naive baselines.
7. Holt-Winters, ARIMA and SARIMA candidates.
8. Lag/calendar-feature ML model.
9. Holdout evaluation, residual analysis and cached artefacts.
10. FastAPI endpoints and tests.
11. React/Vite/Tailwind application and API client.
12. Interactive visualisations, forecast controls and export.
13. End-to-end verification and documentation.

All numerical results and plots will be generated by code from the selected dataset. No model scores or forecasts will be hardcoded in the UI.

## Models planned
- Naive baseline
- Seasonal-naive baseline
- Holt-Winters exponential smoothing
- ARIMA / SARIMA
- Random Forest or Gradient Boosting using lag and calendar features

**Results table:** Not available yet. Metrics will be added after the evaluation pipeline has run successfully on the held-out test period.

| Model | MAE | RMSE | MAPE | Status |
|---|---:|---:|---:|---|
| Baselines and candidate models | — | — | — | Not trained |

## Installation and running
**Current status:** Dataset acquisition and preprocessing scripts are implemented. The application startup command, backend API and frontend will be added in later steps.

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
│   ├── models/
│   ├── notebooks/
│   ├── tests/
│   │   └── test_preprocessing.py
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── api/
│       ├── components/
│       ├── hooks/
│       └── pages/
├── scripts/
│   ├── data_processing.py
│   ├── download_data.py
│   ├── generate_synthetic_data.py
│   └── preprocess_data.py
├── reports/
├── README.md
├── VIVA_PREP.md
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
- **Step 3/16 — Loading, inspection, cleaning and resampling:** Added a shared loader for the UCI semicolon-delimited source and synthetic CSV, inspection summary, IQR outlier reporting, limited time interpolation, hourly/daily resampling, saved output/report paths and preprocessing tests. Code is committed; runtime test results remain unverified in this environment.
- **Next:** Step 4, exploratory data analysis. Do not treat Step 4 as started until approved.

## Viva checkpoint
**Q: Why do we resample the original minute-level data to hourly and daily data?**  
A: The original dataset contains over two million minute-level rows. Hourly and daily summaries reduce processing cost and make broad patterns easier to interpret. We use mean active power in kW as the forecasting target. For an hourly interval, the average power in kW is numerically equivalent to approximate energy in kWh over that hour.

**Q: Why are outliers reported but not automatically deleted?**  
A: A high reading might be a real demand peak rather than a measurement error. The IQR rule flags suspicious values for inspection, but deleting them automatically could remove the very peaks the forecasting model should learn.

**Q: Why is a chronological test split important?**  
A: Time-series forecasting predicts future values from past observations. A random split can leak future patterns into training data and make evaluation unrealistically optimistic. The model should be tested on observations that occur after the training period.
