# Best MLflow Run

Best run selected from recent completed MLflow runs by sorting on RAGAS faithfulness.

- Experiment: `legrag-rag-eval`
- Run ID: `5fb164a2a93e42fcae018c61ce09398e`
- Run URL: <https://mlflow.badribrahim.com/#/experiments/1/runs/5fb164a2a93e42fcae018c61ce09398e/evaluations>
- RAGAS faithfulness: `0.8346984126984127`

Pinned configuration:

- Chunk size: `256`
- Chunk overlap: `64`
- Approx chars/token: `5`
- Embedding model: `qwen3-embedding:4b`
- Retrieval `top_k`: `27`
- Agent model: `deepseek-v4-pro`
- Judge model: `openai/gpt-4o-mini`

The final local rebuild uses the same selected configuration and writes a clean Chroma collection named `egyptian_civil_code` under `data/chroma`.
