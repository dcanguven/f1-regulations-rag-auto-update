#!/bin/bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

if ! curl -fs -m 3 "$OLLAMA_URL/api/tags" >/dev/null; then
  echo "Ollama is not running at $OLLAMA_URL. Start it with: ollama serve" >&2
  exit 1
fi

mkdir -p logs
uvicorn app.main:app --port "$API_PORT" > logs/api.log 2>&1 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 60); do
  if curl -fs -m 2 "http://localhost:$API_PORT/health" >/dev/null; then break; fi
  if ! kill -0 $API_PID 2>/dev/null; then echo "API failed to start; see logs/api.log" >&2; exit 1; fi
  sleep 1
done

echo "API:  http://localhost:$API_PORT   UI:  http://localhost:$UI_PORT   (Ctrl-C stops both)"
F1_API_URL="http://localhost:$API_PORT" streamlit run frontend/app.py --server.port "$UI_PORT"
