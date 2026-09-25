#!/bin/sh
set -e

# `docker compose run --rm api <command>` passes <command> here as "$@" -
# always respect it exactly (this is how `alembic upgrade head`,
# `python -m app.cli ingest`, and `pytest` are invoked in the README).
if [ "$#" -gt 0 ]; then
  exec "$@"
fi

# No command given (a bare `docker compose up`, or `docker compose up api`):
# migrate, then serve the API. Works as of Day 2 (app/api/main.py exists,
# fastapi/uvicorn are installed).
echo "Running database migrations..."
alembic upgrade head

echo "Starting API server..."
exec uvicorn app.api.main:app --host 0.0.0.0 --port 8000
