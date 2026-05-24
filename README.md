# Canadian Telecom Churn & Retention ROI Analyzer

![CI](https://github.com/Priyank0704/canadian-telecom-churn/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)

End-to-end customer churn analytics for the Canadian telecom market, with an
interactive retention ROI calculator that quantifies the business case for
keeping each customer. Built to mirror the analytical workflow used by retention
teams at Rogers, Bell, Telus, and the customer-analytics teams at Canada's Big
Five banks.

## Live demo

### Option A — Docker (recommended)

```cmd
docker compose up --build
```

Open http://localhost:8501.

### Option B — Local Python

```cmd
streamlit run src\dashboard\app.py
```

Three tabs:
1. **Portfolio Overview** — customer base composition, churn distribution, segment breakdown
2. **Retention Targeting Simulator** — adjust business assumptions, see optimal targeting depth + net savings
3. **Individual Customer Lookup** — filter and inspect specific customers with predicted churn probability

## Results

| Metric                   | Value                       |
|--------------------------|-----------------------------|
| Winning model            | LightGBM (vs XGBoost)       |
| Test ROC-AUC             | **0.845**                   |
| Test AUPRC               | **0.653** (baseline 0.265)  |
| Test F1 @ tuned threshold | **0.541**                   |
| Customers segmented      | 7,043                       |
| RFM segments identified  | 3 (incl. "New & light" — 58% churn) |

### Top churn drivers (by mean |SHAP|)

1. **tenure** — the honeymoon-cliff effect
2. **Contract_Two year** — locked-in customers don't leave
3. **PaymentMethod_Electronic check** — friction signal, 3× churn lift
4. **InternetService_Fiber optic** — pricing-pain signal
5. **Contract_One year**, **PaperlessBilling**, **SeniorCitizen**

## What makes this Canadian

The base behavioural data uses the IBM Telco Churn benchmark (7,043 customers).
On top of it, three Canadian context layers:

1. **CRTC / CCTS provider complaint rates** — complaints per 10k customers
2. **StatCan provincial telecom spend** — affordability normalizer
3. **Market-share-weighted provider assignment** — Big-3 vs flanker distribution

> **Transparency note:** customer-level Canadian churn data isn't publicly
> available (PIPEDA). The IBM benchmark is layered with realistic Canadian
> context so the analysis mirrors what an in-house analyst would produce.

## Tech stack

| Layer              | Tools                                                   |
|--------------------|---------------------------------------------------------|
| Data               | pandas, pyarrow                                         |
| ML                 | scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE) |
| Hyperparameters    | Optuna (Bayesian, 50 trials)                            |
| Threshold tuning   | Custom business-cost minimization                       |
| Explainability     | SHAP (TreeExplainer)                                    |
| Survival analysis  | lifelines (Kaplan-Meier, log-rank)                      |
| Dashboard          | Streamlit + Plotly                                      |
| Container          | Docker (multi-stage, non-root, healthchecked)           |
| CI                 | GitHub Actions (ruff + pytest + Docker build)           |

## Repo structure

```
canadian-telecom-churn/
├── .github/workflows/ci.yml         # CI: lint + test + Docker build
├── data/                            # gitignored
│   ├── raw/{telco,crtc,statcan}/
│   └── processed/
├── src/
│   ├── collection/                  # M1 — data acquisition + merging
│   ├── features/                    # M2/M3 — RFM segmentation + modelling matrix
│   ├── models/                      # M3 — train.py + explain.py
│   ├── visualization/               # M2/M4 — EDA + survival curves + flagship chart
│   ├── dashboard/                   # M4 — Streamlit app
│   └── config.py
├── tests/                           # pytest smoke tests
├── reports/
│   ├── figures/                     # 15 charts
│   └── shap/                        # SHAP plots + feature importance CSV
├── models/                          # serialized model + threshold + card
├── Dockerfile                       # multi-stage container build
├── docker-compose.yml               # one-command local launch
├── .dockerignore
├── pyproject.toml                   # ruff + pytest config
├── requirements.txt
└── README.md
```

## Reproduce from scratch

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Milestone 1 — Data
python src\collection\download_telco.py
python src\collection\build_crtc.py
python src\collection\build_statcan.py
python src\collection\build_master.py

# Milestone 2 — EDA + segmentation
python src\visualization\run_eda.py
python src\visualization\flagship_chart.py
python src\features\rfm_segmentation.py

# Milestone 3 — Modelling + explainability
python src\models\train.py            # ~10-15 min, 50 Optuna trials
python src\models\explain.py

# Milestone 4 — Survival + dashboard
python src\visualization\survival_curves.py
streamlit run src\dashboard\app.py    # opens in browser
```

## Run with Docker

```cmd
# Build and launch
docker compose up --build

# In another terminal, check it's healthy
docker compose ps

# Stop and remove
docker compose down
```

The compose file mounts your local `models/`, `data/processed/`, and `reports/`
directories read-only, so the container always serves your latest trained model
without needing a rebuild.

## CI / CD

Every push and pull request triggers `.github/workflows/ci.yml`, which:

1. Installs dependencies on Python 3.11
2. Lints with `ruff` (pycodestyle + pyflakes + isort rules)
3. Builds the data pipeline using a synthetic Telco CSV (no network deps)
4. Runs `pytest` against the full test suite
5. Builds the Docker image (cached via GitHub Actions cache)
6. Smoke-tests the image starts cleanly

If any step fails, the PR is blocked from merging.

## Milestones

- [x] **M1** — Project setup, data collection, master dataset (7,043 × 35)
- [x] **M2** — EDA + RFM segmentation (3 segments, K chosen by silhouette)
- [x] **M3** — XGBoost vs LightGBM benchmark, SMOTE-balanced CV, Optuna tuning, business-cost threshold, SHAP
- [x] **M4** — Kaplan-Meier cohort survival + Streamlit retention ROI calculator
- [x] **Production polish** — Docker + GitHub Actions CI

## Why this project

Canadian telecom has the highest customer acquisition costs in the OECD and
switching is rare, which makes retention modelling extremely valuable here.
Rogers, Bell, and Telus all have dedicated retention analytics teams.
The Big Five banks (RBC, TD, BMO, Scotia, CIBC) use nearly identical churn
methodology for their retail banking customers, so this project doubles as
banking-relevant.
