# AMMRAG server
FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y \
    gcc build-essential libgl1 libglib2.0-0 \
    cron curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install uv

# Install dependencies (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Daily import
RUN echo "0 0 * * * root /usr/local/bin/trigger-import.sh" > /etc/cron.d/import-job \
    && chmod 0644 /etc/cron.d/import-job

EXPOSE 8000 8001

CMD bash -c '\
    printenv > /etc/environment && \
    printf "#!/bin/sh\n. /etc/environment\ncurl -s \"http://localhost:8000/import/project?name=\${MCP_COLLECTION_NAME}\" > /proc/1/fd/1 2>&1\n" \
        > /usr/local/bin/trigger-import.sh && \
    chmod +x /usr/local/bin/trigger-import.sh && \
    mkdir -p /app/config && \
    printf "config:\n  projects:\n    - name: %s\n      path: %s\n" \
        "${MCP_COLLECTION_NAME}" "${MCP_IMPORT_PATH}" > /app/config/config.yml && \
    cron && \
    exec uv run python start_services.py'
