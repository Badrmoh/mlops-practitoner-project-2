# Definition of Done Status

Legend:

- ✅ Done
- ⚠️ Partial / implemented differently
- ❌ Not implemented yet

## Code, Packaging, Tracking, Versioning

| Status | Requirement | Evidence / Notes |
| --- | --- | --- |
| ✅ | `src/` layout with `pyproject.toml`; editable/package install works | Package is defined as `legrag` in `pyproject.toml` with `[tool.uv] package = true`. CI installs with `uv sync --all-extras --dev`. |
| ✅ | Egyptian Civil Code PDF converted to structured JSON, one record per article, committed to the repo | Conversion exists through `dvc/preprocess.py`; structured output is `data/output/articles.json`. It is committed to Git and used as the `chunk` stage dependency. |
| ✅ | Extraction validated: article numbers normalized to integers, Arabic text spot-checked/validated, repealed articles flagged | `tests/test_ingestion_validation.py` checks contiguous unique article numbers, non-empty Arabic text except documented exception, sane text lengths, and repealed article flags. |
| ✅ | Ingestion pipeline: JSON → chunk by article → embed → vector store | DVC stages in `dvc/dvc.yaml`: `preprocess`, `chunk`, `embed`. Chroma output is `data/chroma/chroma.sqlite3`. |
| ✅ | FastAPI `/ask`: `{question: str}` → `{answer: str, sources: list[str]}`; sources are article citations, not chunk IDs | Implemented in `src/legrag/api.py`; tested in `tests/test_api.py`. Sources are formatted as `Egyptian Civil Code, Article N`. |
| ✅ | Pydantic rejects an empty question; `422` returned and tested | `AskRequest` validation in `src/legrag/api.py`; covered by `test_ask_rejects_empty_question`. |
| ✅ | `/health` returns `{status: healthy, documents_indexed: N}` | Implemented in `src/legrag/api.py`; covered by `test_health_returns_document_count`. |
| ✅ | Dockerfile includes vector store and embedded documents; `docker compose up` starts on port `8000` | `Dockerfile` copies `data/chroma` into `/app/data/chroma`; `compose.yaml` maps `8000:8000`. |
| ✅ | README has 3 commands to run Q&A on any machine | `README.md` includes `docker compose up`, `/health`, and `/ask` curl commands. |
| ✅ | MLflow logs each config: chunk size, overlap, embedding model, faithfulness | `dvc/eval.py` logs params and metrics when `MLFLOW_TRACKING_URI` is set. |
| ✅ | At least 5 runs compared; MLflow screenshot in `reports/` | Evidence file: `reports/0-mlflow_experiments.png`. The best-run summary is embedded in `README.md`. |
| ✅ | Best chunking config registered in the MLflow Registry | Registered as `legrag-best-chunking-config` with alias `best`; evidence file: `reports/2-registered-model.png`. Registration helper: `scripts/register_best_config.py`. |
| ⚠️ | Source PDF and derived JSON both tracked with DVC; `dvc pull` fetches them; `dvc repro` rebuilds JSON from PDF | Source PDF is downloaded in CI from `LEGRAG_RAW_PDF_URL`, not tracked with a `.dvc` file. `data/output/articles.json` is committed to Git to satisfy the structured-JSON deliverable, so it is intentionally not declared as a DVC output; `chunk` tracks it as a dependency. |
| ✅ | `dvc repro` re-indexes the corpus reproducibly from tracked/configured documents | `dvc --cd dvc repro preprocess chunk embed` rebuilds preprocess/chunk/embed stages from `data/raw/law.pdf` and params. CI downloads the PDF before running DVC. |
| ✅ | GitHub Actions: lint → test → rebuild index → push Docker image | `.github/workflows/ci.yml` runs Pyright, pytest, raw PDF download, DVC rebuild/eval, and Docker build/push on `main` pushes. |
| ✅ | CI fails if RAGAS faithfulness `< 0.75` on the golden-question test set | CI gates on `data/output/ragas_metrics.json` with `ragas_faithfulness >= 0.75`. Golden questions are in `data/eval/golden_questions.json`. |

## Serving, Monitoring, Optimization

| Status | Requirement | Evidence / Notes |
| --- | --- | --- |
| ❌ | BentoML wraps the RAG pipeline with an async `/ask` endpoint | Not implemented. Current serving layer is FastAPI directly. |
| ❌ | vLLM serves the generative LLM; model name in README | Not implemented. Current setup uses Ollama locally or OpenAI-compatible providers through config. |
| ❌ | Streaming: tokens appear progressively in curl output | Not implemented. `/ask` returns a complete JSON response. |
| ❌ | Locust report at 50 concurrent users in `reports/` | No Locust config/report found. |
| ❌ | Batch re-indexing script tested with at least one new document | No dedicated batch re-indexing script/test found beyond the DVC rebuild path. |
| ❌ | Canary rollout config documented in README | No canary rollout config found. |
| ⚠️ | RAGAS on at least 50 questions; all 4 metrics logged | Current eval uses `data/eval/golden_questions.json` and logs RAGAS faithfulness plus custom retrieval/answer metrics. It does not log all four standard RAGAS metrics. |
| ✅ | RAGAS results stored in MLflow; trend visible across sessions | `dvc/eval.py` logs metrics to MLflow when `MLFLOW_TRACKING_URI` is configured. Evidence: `reports/mlflow_experiments.png`. |
| ❌ | Alert: faithfulness `< 0.80` triggers notification | No alerting/notification config found. CI gate is `< 0.75`, not `< 0.80`. |
| ✅ | Langfuse self-hosted; every `/ask` creates a trace with spans | `/ask` is wrapped with Langfuse `@observe` in `src/legrag/api.py`; `.env.example` documents `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`. Verified manually against the self-hosted Langfuse instance. |
| ❌ | RAGAS faithfulness score attached to each Langfuse trace | Not implemented. |
| ❌ | Cosine similarity on query embeddings vs baseline; drift logged | Not implemented. |
| ❌ | Token usage in Prometheus; Grafana shows cost/hour | Not implemented. |
| ⚠️ | Guardrails / PII detection active on all `/ask` responses | Instruction-level legal answer guardrails exist in `src/legrag/agent/capabilities.py`. PII detection is not implemented. |
| ❌ | RAGAS scores: original vs quantized; faithfulness drop `<= 0.03` | Not implemented. |
| ✅ | Final architecture diagram covering project components | The Mermaid architecture is embedded directly in `README.md`. |

## Current Known Gaps

1. Source PDF is no longer DVC-tracked directly; CI downloads it with `curl` from `LEGRAG_RAW_PDF_URL`.
2. Chroma tracking is intentionally left imperfect for now because Chroma mutates its storage files on access.
3. MLflow Registry registration is complete for the best chunking config, with evidence in `reports/registered-model.png`.
4. Serving/monitoring optimization items such as BentoML, vLLM, streaming, Locust, Langfuse metrics/scoring, Prometheus/Grafana, and quantization are not implemented yet.
5. CI can fail from provider rate limits because eval calls external embedding/LLM APIs.
