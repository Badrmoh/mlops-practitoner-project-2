# Legal RAG · Egyptian Civil Code Q&A

RAG service for answering questions against the Egyptian Civil Code. The app serves a FastAPI `/ask` endpoint backed by a Chroma vector store built from article-level JSON extracted from the source PDF.

Answers cite Civil Code article numbers, not chunk IDs or page numbers.

## Run Q&A

Prerequisite: run Ollama, and make sure these local models are available:

- Embedding model: `qwen3-embedding:4b`
- Answer model: `qwen3.5:4b`

Three commands:

```bash
docker compose up
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"ما هي شروط العقد؟"}'
```

Dev compose, with source and reports mounted from the repo:

```bash
docker compose -f dev.compose.yaml up
```

The compose files work without `.env`. Override models with `OLLAMA_EMBEDDING_MODEL` or `OLLAMA_LLM_MODEL` if needed.

## API

`POST /ask`

Request:

```json
{"question":"ما هي شروط العقد؟"}
```

Response:

```json
{
  "answer": "...",
  "sources": ["Egyptian Civil Code, Article 89"]
}
```

`sources` are article citations only.

`GET /health`

```json
{"status":"healthy","documents_indexed":1234}
```

`GET /reports/` serves project evidence files, including MLflow screenshots and architecture diagrams.

## Rebuild Index

Download the Egyptian Civil Code PDF first and place it at `data/raw/law.pdf`:

```bash
mkdir -p data/raw
curl -L "https://sadanykhalifa.com/uploads/Laws/1576751803.pdf" -o data/raw/law.pdf
```

Then run DVC from the repo root with `--cd dvc`:

```bash
uv run dvc --cd dvc repro preprocess chunk embed
uv run dvc --cd dvc repro eval
```

Current best config is pinned in `dvc/params.yaml`:

- `chunk.max_tokens_per_chunk: 256`
- `chunk.chunk_overlap_tokens: 64`
- `embed.embedding_model: qwen3-embedding:4b`
- `embed.collection_name: egyptian_civil_code`
- `embed.chroma_persist_path: data/chroma`
- `eval.top_k: 27`
- `eval.agent_model: deepseek-v4-pro`
- `eval.judge_model: openai/gpt-4o-mini`

`approx_chars_per_token` is fixed in code via `Settings.approx_chars_per_token = 5.0`, not tuned through DVC params.

## Best Result

Best run was selected in the MLflow UI by filtering to recent completed runs and sorting by RAGAS faithfulness.

- MLflow run: `5fb164a2a93e42fcae018c61ce09398e`
- URL: <https://mlflow.badribrahim.com/#/experiments/1/runs/5fb164a2a93e42fcae018c61ce09398e/evaluations>
- RAGAS faithfulness: `0.8346984126984127`
- Chunk size: `256`
- Chunk overlap: `64`
- Approx chars/token: `5`
- Embedding model: `qwen3-embedding:4b`
- Retrieval `top_k`: `27`
- Agent model: `deepseek-v4-pro`
- Judge model: `openai/gpt-4o-mini`

The final DVC params rebuild the same selected configuration into a clean local collection named `egyptian_civil_code` under `data/chroma`.

## Tracking

- Source PDF is tracked through DVC at `data/raw/law.pdf.dvc`.
- Derived artifacts are rebuilt by `dvc/dvc.yaml` stages.
- RAGAS faithfulness metrics are written to `data/output/ragas_metrics.json` and logged to MLflow.
- Local/reviewer checks should fail if `ragas_faithfulness < 0.75`.
- Best run selection is done in the MLflow UI by filtering/sorting experiment runs; the selected config is pinned in `dvc/params.yaml`.

MLflow experiment: `legrag-rag-eval`.

## Reports

Current report artifacts:

- `reports/mlflow_experiments.png`
- `reports/mlflow_best_run.md`
- `reports/architecture.svg`
- `reports/architecture.md`
- `reports/architecture.mmd`

Open them through:

```bash
open http://localhost:8000/reports/
```
