# WattCast ⚡

**Electricity consumption analysis and time-series forecasting, from raw readings to an interactive dashboard.**

WattCast is an end-to-end data science project that analyses historical household electricity consumption, compares forecasting methods, and presents saved results in a React dashboard backed by FastAPI.

> **Important:** The Forecasts page displays predictions saved for a chronological test/holdout period. It does not generate forecasts beyond the end of the dataset.

## Features

- Download and preprocess the UCI household electricity dataset.
- Generate clearly labelled synthetic data when the real dataset is unavailable.
- Explore hourly and daily consumption patterns.
- Analyse trend, seasonality, stationarity, ACF and PACF.
- Compare six forecasting approaches: Naive, Seasonal Naive, Holt-Winters, ARIMA, SARIMA and Random Forest.
- Evaluate models using MAE, RMSE and MAPE, with residual diagnostics.
- Inspect Pearson and Spearman correlations across the UCI dataset's numeric measurements.
- Explore results through a responsive dashboard.

## Tech stack

| Layer | Technologies |
|---|---|
| Data processing | Python, Pandas, NumPy |
| Time-series analysis | Statsmodels, Scikit-learn |
| Visualisation | Matplotlib, Recharts |
| Backend | FastAPI, Uvicorn |
| Frontend | React, TypeScript, Vite |
| UI | Tailwind CSS, Framer Motion, Lucide |

## Dataset

The primary source is the [UCI Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (DOI: [10.24432/C58K54](https://doi.org/10.24432/C58K54)).

It contains 2,075,259 one-minute measurements from one household near Paris, France, collected between December 2006 and November 2010. The original data is semicolon-delimited and contains missing values.

**Citation:** Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54.

The dataset is licensed under CC BY 4.0. Retain attribution when using it.

### Real data vs. synthetic data

- The downloader attempts to obtain the real UCI dataset.
- If downloading fails, it can use a reproducible synthetic fallback.
- Synthetic readings are simulated, not real household measurements. Their provenance is recorded by the pipeline.
- The synthetic generator currently contains active-power data only, so cross-variable correlation is unavailable for synthetic data.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer and npm
- Git
- Internet access for downloading the dataset and installing dependencies

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/BethalaRinosh/WattCast.git
cd WattCast
```

### 2. Create and activate a Python environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again. Alternatively, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install backend dependencies

Run from the repository root:

```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 4. Download the dataset

```bash
python scripts/download_data.py
```

The script downloads and extracts the UCI dataset. If downloading fails, it can generate a synthetic fallback.

To require real data and prevent fallback:

```bash
python scripts/download_data.py --no-synthetic-fallback
```

To generate synthetic data directly:

```bash
python scripts/generate_synthetic_data.py --years 4 --seed 42
```

### 5. Run the analysis pipeline

Run these commands **from the repository root**, in order:

```bash
python scripts/preprocess_data.py
python scripts/eda.py
python scripts/time_series_analysis.py
python scripts/baseline_forecasting.py
python scripts/statistical_forecasting.py
python scripts/ml_forecasting.py
python scripts/model_evaluation.py
python scripts/correlation_analysis.py
```

The scripts create processed datasets and reports under `backend/data/` and `backend/reports/`. The UCI dataset contains more than two million rows, so processing may take a few minutes.

### 6. Start the backend

Keep the Python virtual environment activated. From the repository root:

```bash
python -m uvicorn backend.app.main:app --reload
```

- API base: http://127.0.0.1:8000
- Interactive Swagger docs: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Health check: http://127.0.0.1:8000/api/health

Leave this terminal running.

### 7. Start the frontend

Open a **second terminal** in the repository root:

```bash
cd frontend
```

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
npm install
npm run dev
```

**macOS / Linux**

```bash
cp .env.example .env
npm install
npm run dev
```

Open the local URL printed by Vite, normally http://127.0.0.1:5173.

The frontend reads its backend URL from `frontend/.env`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Keep this as the API origin. Do not append `/api`. Restart Vite after changing the environment file.

## Running it after the first setup

You do not need to reinstall dependencies every time.

1. Open a terminal in the repository root and activate `.venv`.
2. Start the backend: `python -m uvicorn backend.app.main:app --reload`.
3. Open another terminal, go to `frontend/`, and run `npm run dev`.
4. Open the Vite URL in your browser.

Generated data and reports are excluded from Git. On a fresh clone, run the data and analysis pipeline before expecting every dashboard page to have results.

## Dashboard sections

| Page | Description |
|---|---|
| Overview | Project summary and key dataset/analysis information |
| Data explorer | Historical hourly and daily consumption |
| Correlation analysis | Pearson and Spearman relationships |
| Decomposition | Observed series, trend, seasonal component and residual |
| Model lab | Forecast-model metrics and rankings |
| Forecasts | Actual values against saved holdout predictions |

## API endpoints

The FastAPI service exposes read-only endpoints backed by generated artifacts.

| Endpoint | Description |
|---|---|
| `GET /api/health` | API process health |
| `GET /api/summary` | EDA and evaluation summary |
| `GET /api/series/hourly` | Recent hourly observations |
| `GET /api/series/daily` | Recent daily observations |
| `GET /api/eda` | EDA summary and profile data |
| `GET /api/decomposition/hourly` | Hourly decomposition |
| `GET /api/decomposition/daily` | Daily decomposition |
| `GET /api/models/hourly` | Hourly model metrics |
| `GET /api/models/daily` | Daily model metrics |
| `GET /api/forecast/{series}` | Saved predictions for a selected model |
| `GET /api/correlation/hourly` | Hourly correlation results |
| `GET /api/correlation/daily` | Daily correlation results |

For forecast requests, `series` is `hourly` or `daily`. The `model` query parameter selects a saved model prediction, and `limit` controls the number of returned observations. See `/docs` for parameters and response schemas.

## Tests and production build

Run backend tests from the repository root:

```bash
python -m pytest backend/tests -q
```

Build the frontend:

```bash
cd frontend
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

## Project structure

```text
WattCast/
├── backend/
│   ├── app/                 # FastAPI application and routes
│   ├── data/
│   │   ├── raw/             # Downloaded or generated input data
│   │   └── processed/       # Clean hourly and daily datasets
│   ├── reports/             # Generated analysis/model artifacts
│   ├── models/              # Saved model artifacts, if generated
│   ├── tests/               # Backend tests
│   └── requirements.txt
├── docs/
│   └── VIVA_PREP.md         # Technical and viva guide
├── frontend/
│   ├── src/
│   │   ├── api/             # Typed API client
│   │   ├── components/      # Shared UI components
│   │   └── pages/           # Dashboard pages
│   ├── .env.example
│   └── package.json
├── scripts/                 # Data, analysis and forecasting pipelines
├── reports/                 # Reserved report output directories
├── .gitignore
└── README.md
```

## Methodology notes

- The forecasting target is mean active power in kW.
- Data is sorted chronologically and evaluated using a chronological train/test split, not a random shuffle.
- Only short internal missing runs are interpolated; long gaps are not indiscriminately filled.
- IQR outliers are reported, not automatically removed, because high readings may be genuine consumption peaks.
- Models are compared on a shared holdout using MAE, RMSE and MAPE. MAPE excludes zero or near-zero actual values.
- Pearson and Spearman correlation measure association, not causation.
- The dataset represents one household and is not representative of every home or the electricity grid.
- The dashboard reads saved analysis artifacts. The API does not train models when a page is opened.

## Troubleshooting

**The dashboard says the API is offline**
- Confirm the backend terminal is still running.
- Open http://127.0.0.1:8000/api/health.
- Check that `frontend/.env` contains `VITE_API_BASE_URL=http://127.0.0.1:8000`.
- Restart Vite after editing the environment file.

**A dashboard page reports that an artifact is missing**
- Run the analysis pipeline from the repository root.
- Check that the required files exist under `backend/data/processed/` or `backend/reports/`.
- The health endpoint only confirms that FastAPI is running; it does not confirm that reports exist.

**Python dependencies fail to install**
- Confirm that the virtual environment is activated.
- Check `python --version`, upgrade pip and retry the requirements installation.

**Frontend dependencies or build fail**
- Check `node --version` and `npm --version`.
- Run `npm install` from `frontend/`, then `npm run build`.

**Dataset download fails**
- Retry with an internet connection.
- Use the synthetic generator for development and label its results as simulated.
- Use `--no-synthetic-fallback` when real data is required.

## Documentation

See [docs/VIVA_PREP.md](docs/VIVA_PREP.md) for the architecture walkthrough, data-cleaning decisions, time-series concepts, model explanations, API map, demo sequence and viva questions.

## Author

**Bethala Rinosh**  
B.Tech CSE (AI & ML)

---

If you use or extend this project, retain the UCI dataset attribution and clearly distinguish real measurements from synthetic data.
