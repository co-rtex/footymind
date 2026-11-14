# FootyMind: API + web lab in one image.
#
# Metrica sample data is NOT baked into the image (the provider asks for responsible use
# and attribution; images can be pushed and redistributed). On first start the
# entrypoint downloads the checksum-pinned files into the /data volume and ingests them.
#
#   docker build -t footymind .
#   docker run --rm -p 8000:8000 -v footymind-data:/data footymind
#   open http://localhost:8000

# Base images are overridable, e.g. to use a registry mirror when Docker Hub rate-limits:
#   docker build --build-arg NODE_IMAGE=mirror.gcr.io/library/node:22-slim \
#                --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.12-slim -t footymind .
# Behind a TLS-intercepting proxy, pass the proxy as build args and its CA as a secret:
#   docker build --network=host --build-arg HTTPS_PROXY=... \
#                --secret id=ca_bundle,src=/path/to/ca.crt -t footymind .
# Without the secret these steps use the image's default trust store.
ARG NODE_IMAGE=node:22-slim
ARG PYTHON_IMAGE=python:3.12-slim

# ---------- web build ----------
FROM ${NODE_IMAGE} AS web
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -f /run/secrets/ca_bundle ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/ca_bundle; fi; \
    npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------- runtime ----------
FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    FOOTYMIND_DATA_DIR=/data \
    FOOTYMIND_WEB_DIST=/app/web/dist \
    FOOTYMIND_EXPERIMENTS_DIR=/app/experiments \
    PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -f /run/secrets/ca_bundle ]; then \
      export PIP_CERT=/run/secrets/ca_bundle SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    pip install --no-cache-dir uv==0.11.32 && uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -f /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    uv sync --frozen --no-dev --no-editable
COPY --from=web /app/web/dist ./web/dist
# committed results.json files: validation summaries served by the API (out-of-fold parquet files
# are excluded by .dockerignore)
COPY experiments/results ./experiments/results
COPY docker/entrypoint.sh /usr/local/bin/footymind-entrypoint
RUN chmod +x /usr/local/bin/footymind-entrypoint && useradd --create-home --uid 10001 app \
    && mkdir -p /data && chown app /data
USER app
VOLUME ["/data"]
EXPOSE 8000
# The server starts after the entrypoint has built any missing similar-moments index (about 40 s
# for Metrica and 3 minutes for all 20 SkillCorner matches on 4 vCPU), hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
ENTRYPOINT ["footymind-entrypoint"]
CMD ["footymind", "serve", "--host", "0.0.0.0", "--port", "8000"]
