#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
OLLAMA_URL="http://localhost:11434"
EMBEDDING_MODEL="Qwen3-Embedding:4b"
LLM_MODEL="qwen3.5:4b"

for command in ollama curl docker; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command not found: $command" >&2
    exit 1
  fi
done

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose is required." >&2
  exit 1
fi

if ! curl --fail --silent "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "Starting Ollama on $OLLAMA_URL..."
  ollama serve >"${TMPDIR:-/tmp}/legrag-ollama.log" 2>&1 &

  for _ in {1..30}; do
    if curl --fail --silent "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! curl --fail --silent "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "Ollama did not become ready. Check ${TMPDIR:-/tmp}/legrag-ollama.log." >&2
  exit 1
fi

echo "Downloading Ollama models..."
ollama pull "$EMBEDDING_MODEL"
ollama pull "$LLM_MODEL"

warm_model() {
  local label="$1"
  local endpoint="$2"
  local payload="$3"
  local max_attempts=30
  local retry_delay=10

  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    echo "Warming $label (attempt $attempt/$max_attempts)..."
    if curl --fail --silent --show-error \
      --connect-timeout 10 \
      --max-time 300 \
      "$OLLAMA_URL$endpoint" \
      -H "Content-Type: application/json" \
      -d "$payload" \
      >/dev/null; then
      echo "$label is ready."
      return 0
    fi

    if ((attempt < max_attempts)); then
      echo "$label is still loading; retrying in ${retry_delay}s..."
      sleep "$retry_delay"
    fi
  done

  echo "Failed to warm $label after $max_attempts attempts." >&2
  return 1
}

warm_model \
  "embedding model $EMBEDDING_MODEL" \
  "/api/embed" \
  "{\"model\":\"$EMBEDDING_MODEL\",\"input\":\"warmup\",\"keep_alive\":-1}"

warm_model \
  "LLM $LLM_MODEL" \
  "/api/chat" \
  "{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Respond with OK\"}],\"stream\":false,\"keep_alive\":-1}"

loaded_models="$(ollama ps)"
if ! grep -Eiq "^${EMBEDDING_MODEL}[[:space:]]" <<<"$loaded_models"; then
  echo "Embedding model was downloaded but is not loaded: $EMBEDDING_MODEL" >&2
  echo "$loaded_models" >&2
  exit 1
fi
if ! grep -Eiq "^${LLM_MODEL}[[:space:]]" <<<"$loaded_models"; then
  echo "LLM was downloaded but is not loaded: $LLM_MODEL" >&2
  echo "$loaded_models" >&2
  exit 1
fi

echo "Both Ollama models are loaded:"
echo "$loaded_models"

tmp_env="$(mktemp)"
if [[ -f "$ENV_FILE" ]]; then
  grep -Ev '^(LEGRAG_EMBEDDING_BASE_URL|LEGRAG_EMBEDDING_MODEL|LEGRAG_EMBEDDING_API_KEY|LEGRAG_LLM_BASE_URL|LEGRAG_LLM_MODEL|LEGRAG_LLM_API_KEY)=' "$ENV_FILE" >"$tmp_env" || true
fi

cat >>"$tmp_env" <<EOF
LEGRAG_EMBEDDING_BASE_URL=http://host.docker.internal:11434/api/embed
LEGRAG_EMBEDDING_MODEL=$EMBEDDING_MODEL
LEGRAG_EMBEDDING_API_KEY=
LEGRAG_LLM_BASE_URL=http://host.docker.internal:11434/v1
LEGRAG_LLM_MODEL=$LLM_MODEL
LEGRAG_LLM_API_KEY=ollama
EOF
mv "$tmp_env" "$ENV_FILE"

echo "Wrote Ollama configuration to $ENV_FILE"
cd "$ROOT_DIR"
docker compose up
