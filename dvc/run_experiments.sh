#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  # "max_tokens overlap top_k embedding_model"
  "256 64 27 qwen3-embedding:4b"
)
AGENT_MODELS=(deepseek-v4-pro)
JUDGE_MODELS=(deepseek-v4-flash)
JUDGE_MODELS=(openai/gpt-4o-mini)

TIMESTAMP=$(date "+%m%d%H%M%Y")

for config in "${CONFIGS[@]}"; do
  read -r chunk_size overlap top_k embedding_model<<< "${config}"
      for agent_model in "${AGENT_MODELS[@]}"; do
        for judge_model in "${JUDGE_MODELS[@]}"; do
          echo "Running: chunk=${chunk_size}, overlap=${overlap}, top_k=${top_k}, embedding=${embedding_model}, agent=${agent_model}, judge=${judge_model}"
          collection_name="civil_code_final"
          uv run dvc --cd dvc exp run \
            -S embed.embedding_model="${embedding_model}" \
            -S chunk.max_tokens_per_chunk="${chunk_size}" \
            -S chunk.chunk_overlap_tokens="${overlap}" \
            -S embed.collection_name="${collection_name}" \
            -S eval.agent_model="${agent_model}" \
            -S eval.judge_model="${judge_model}" \
            -S eval.top_k="${top_k}" \
            -S eval.distance_threshold=null \
            chunk embed eval
        done
      done
done

# uv run dvc --cd dvc exp show
