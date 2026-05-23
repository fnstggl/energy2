# ── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools only in builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY aurelius/pyproject.toml ./pyproject.toml
COPY aurelius/ ./aurelius/

# Install dependencies into a prefix so we can copy them cleanly
RUN pip install --no-cache-dir --prefix=/install \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    pydantic>=2.0.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.3.0 \
    lightgbm>=4.0.0 \
    pulp>=2.7.0 \
    jinja2>=3.1.0 \
    joblib>=1.3.0 \
    requests>=2.31.0 \
    pyyaml>=6.0 \
    strictyaml>=1.6.0

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps: libgomp1 required by LightGBM on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Non-root user for security
RUN groupadd --gid 1001 aurelius \
    && useradd --uid 1001 --gid 1001 --no-create-home aurelius

# Copy application code
COPY aurelius/ ./aurelius/
COPY scripts/ ./scripts/

# Create data directories with correct ownership
RUN mkdir -p /app/data /app/ml_artifacts /app/logs \
    && chown -R aurelius:aurelius /app

USER aurelius

# Health-check: verify the API starts
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "aurelius.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
