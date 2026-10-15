# MukhaSpectra API (research prototype — not for clinical use)
# Build from the repository root:  docker build -f backend/Dockerfile .
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# WeasyPrint needs Pango/HarfBuzz; Noto covers Devanagari and Kannada for patient reports.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libffi8 libjpeg62-turbo \
        libglib2.0-0 shared-mime-info fonts-dejavu-core fonts-noto-core fonts-inter \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pipeline/ pipeline/
COPY backend/ backend/
# [postgres] installs psycopg so MUKHA_DATABASE_URL=postgresql+psycopg://... works (compose profile).
RUN pip install './pipeline[raw,onnx]' './backend[postgres]'

# Build identifier recorded in every result: docker build --build-arg MUKHA_BUILD_ID=$(git rev-parse --short HEAD)
ARG MUKHA_BUILD_ID=""
ENV MUKHA_BUILD_ID=${MUKHA_BUILD_ID}

RUN useradd --create-home --uid 10001 mukha && mkdir -p /data && chown mukha /data
USER mukha

ENV MUKHA_DATABASE_URL=sqlite:////data/mukha.db \
    MUKHA_STORAGE_DIR=/data/storage

WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
