# Legal RAG Architecture

```mermaid
flowchart LR
    user[User / curl / browser]
    api[FastAPI /ask and /health]
    agent[Pydantic AI legal RAG agent]
    retriever[LegalRetriever]
    embed[Embedding endpoint<br/>qwen3-embedding:4b]
    chroma[(Chroma vector store<br/>embedded in Docker image)]
    llm[Generative LLM<br/>Ollama or OpenAI-compatible endpoint]
    dvc[DVC pipeline<br/>preprocess -> chunk -> embed -> eval]
    pdf[Source PDF<br/>DVC-tracked]
    articles[Structured article JSON]
    chunks[Article chunks]
    mlflow[MLflow tracking<br/>params + RAGAS faithfulness]
    reports[/reports artifacts<br/>MLflow screenshot + architecture]
    ci[GitHub Actions<br/>lint -> test -> rebuild index -> eval gate -> Docker push]

    user --> api
    api --> agent
    agent --> retriever
    retriever --> embed
    retriever --> chroma
    agent --> llm
    agent --> api
    api --> reports

    pdf --> dvc
    dvc --> articles
    articles --> chunks
    chunks --> embed
    embed --> chroma
    dvc --> mlflow
    mlflow --> reports
    ci --> dvc
    ci --> mlflow
```
