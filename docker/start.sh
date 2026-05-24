#!/usr/bin/env sh
# Container entrypoint for the Aurelius API on Railway / Docker.
#
# Runs schema migrations against DATABASE_URL (idempotent; no-op when unset),
# then execs the API server. Using a script rather than a chained startCommand
# avoids any ambiguity about whether the platform wraps the start command in a
# shell — the migrate step and the server launch are unambiguous here.
set -e

echo "entrypoint: running database migrations..."
python -m aurelius.database.migrate || echo "entrypoint: migrate failed (non-fatal); starting API anyway"

echo "entrypoint: starting API on port ${PORT:-8000}..."
exec python -m uvicorn aurelius.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
