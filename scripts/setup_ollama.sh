#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
OLLAMA_URL="http://localhost:11434"
EMBEDDING_MODEL="Qwen3-Embedding:4b"
LLM_BASE_MODEL="qwen3.5:4b"
# Ollama defaults to a 4096-token context window. The RAG prompt (27 chunks) already
# uses ~3500 tokens, so the structured answer gets truncated mid-JSON and /ask fails
# with "Model token limit ... exceeded before any response was generated". Bake a
# larger window into a derived model so it survives Ollama restarts and does not
# depend on how the server was started.
LLM_CONTEXT=32768
LLM_MODEL="$LLM_BASE_MODEL-32k"
# Bounded keep-alive: pinning both models forever makes Ollama thrash them in and
# out of VRAM on GPUs that cannot hold both at once.
KEEP_ALIVE="5m"

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
  # Run from / so deleting or renaming the repo cannot break model runners later
  # ("cannot get current path").
  ( cd / && exec ollama serve ) >"${TMPDIR:-/tmp}/legrag-ollama.log" 2>&1 &

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
ollama pull "$LLM_BASE_MODEL"

modelfile="$(mktemp)"
cat >"$modelfile" <<EOF
FROM $LLM_BASE_MODEL
PARAMETER num_ctx $LLM_CONTEXT
EOF
echo "Creating $LLM_MODEL with a $LLM_CONTEXT-token context window..."
ollama create "$LLM_MODEL" -f "$modelfile"
rm -f "$modelfile"

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
  "{\"model\":\"$EMBEDDING_MODEL\",\"input\":\"warmup\",\"keep_alive\":\"$KEEP_ALIVE\"}"

warm_model \
  "LLM $LLM_MODEL" \
  "/api/chat" \
  "{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Respond with OK\"}],\"stream\":false,\"keep_alive\":\"$KEEP_ALIVE\"}"

# Verify that both models are *installed*, not that they are resident at the same
# time. Ollama evicts loaded models when GPU memory is tight, so requiring both in
# `ollama ps` fails on machines that cannot hold both models at once.
installed_models="$(ollama list)"
for model in "$EMBEDDING_MODEL" "$LLM_MODEL"; do
  if ! grep -Eiq "^${model}[[:space:]]" <<<"$installed_models"; then
    echo "Ollama model is not installed: $model" >&2
    echo "$installed_models" >&2
    exit 1
  fi
done

echo "Both Ollama models are installed:"
echo "$installed_models"

cd "$ROOT_DIR"
docker compose up
