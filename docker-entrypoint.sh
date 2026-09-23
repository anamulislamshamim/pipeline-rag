#!/bin/sh
set -e

# `docker compose run --rm api <command>` passes <command> here as "$@" -
# always respect it exactly (this is how `alembic upgrade head`,
# `python -m app.cli ingest`, and `pytest` are invoked in the README).
if [ "$#" -gt 0 ]; then
  exec "$@"
fi

# No command given - this happens if something starts the "api" service
# without specifying one, e.g. a bare `docker compose up` instead of
# `docker compose up db`. There is nothing to serve yet: app/api/main.py
# doesn't exist until Day 2, and fastapi/uvicorn aren't even installed yet
# (see requirements.txt). Exit cleanly with guidance instead of crashing.
cat <<'MSG'
[pipeline-rag] No command given to the "api" container, and there is no
default server to start yet (app/api/main.py lands on Day 2).

Use one of, instead:
  docker compose run --rm api alembic upgrade head
  docker compose run --rm api python -m app.cli ingest --categories cs.AI --max-papers 20
  docker compose run --rm api pytest tests/ -v

Exiting cleanly (exit code 0) - this is not an error.
MSG
exit 0