# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# Multi-stage build for the Canadian Telecom Churn Streamlit app.
#
# Stage 1 (builder): install Python dependencies into a virtual environment.
# Stage 2 (runtime): copy only the venv + source code, run as non-root.
#
# Why multi-stage: keeps the runtime image small (~600 MB instead of ~1.4 GB)
# and excludes build toolchains that aren't needed at runtime.
# ---------------------------------------------------------------------------

# ============================== STAGE 1: BUILDER ===========================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for lightgbm / xgboost / lifelines (need gcc + libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install into a clean venv so we can copy it cleanly to the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# ============================== STAGE 2: RUNTIME ===========================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgomp1 is required at runtime by lightgbm/xgboost; everything else can go
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — running containers as root is a CVE waiting to happen
RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app/canadian-telecom-churn

# Copy the prebuilt venv from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy source (model + data are mounted at runtime via volumes — see compose file)
COPY --chown=app:app src/ ./src/
COPY --chown=app:app README.md ./

# Pre-create the dirs the app expects to find (mounted later)
RUN mkdir -p models data/processed reports/figures reports/shap && \
    chown -R app:app /home/app/canadian-telecom-churn

USER app

EXPOSE 8501

# Healthcheck: Streamlit exposes /_stcore/health which returns 200 OK
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
