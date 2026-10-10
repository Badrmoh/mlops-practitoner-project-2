#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
    "256 64 5 5"
    "256 64 5 7"
    "256 64 5 9"
    "256 64 5 11"
    "256 64 5 13"
    "256 64 5 15"
    # "256 64 5 5 1"
    # "256 64 5 10 1"
    # "256 64 5 15 1"
    # "256 64 5 5 1.2"
    # "256 64 5 10 1.2"
    # "256 64 5 15 1.2"
    "224 32 5 5"
    "320 96 5 9"
)
CHUNK_SIZES=()
OVERLAPS=()
APPROX_CHARS=(5)
AGENT_MODELS=(deepseek-chat)
JUDGE_MODELS=(deepseek-chat)

TIMESTAMP=$(date "+%m%d%H%M%Y")

for config in "${CONFIGS[@]}"; do
  read -r chunk_size overlap approx_chars top_k<<< "${config}"
      for agent_model in "${AGENT_MODELS[@]}"; do
        for judge_model in "${JUDGE_MODELS[@]}"; do
          echo "Running: chunk=${chunk_size}, overlap=${overlap}, approx=${approx_chars}, agent=${agent_model}, judge=${judge_model}"
          collection_name="civil_code_cs${chunk_size}_ov${overlap}_ac${approx_chars}_${TIMESTAMP}"
          uv run dvc --cd dvc exp run \
            -S chunk.max_tokens_per_chunk="${chunk_size}" \
            -S chunk.chunk_overlap_tokens="${overlap}" \
            -S chunk.approx_chars_per_token="${approx_chars}" \
            -S embed.collection_name="${collection_name}" \
            -S eval.agent_model="${agent_model}" \
            -S eval.judge_model="${judge_model}" \
            -S eval.top_k="${top_k}" \
            #-S eval.distance_threshold="${distance_threshold}" \
            chunk embed eval
        done
      done
done

# uv run dvc --cd dvc exp show
