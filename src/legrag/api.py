from pathlib import Path
import re
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv

from legrag.agent import LegalRagAgent
from legrag.agent.retrieve import LegalRetriever
from legrag.config import REPO_ROOT, Settings
from legrag.store import ChromaVectorStore

load_dotenv(REPO_ROOT / ".env")

from langfuse import get_client, observe


class AskRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Legal RAG</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 840px; margin: 40px auto; padding: 0 16px; }
    textarea { width: 100%; min-height: 110px; font: inherit; padding: 12px; }
    button { margin-top: 12px; padding: 10px 16px; font: inherit; cursor: pointer; }
    pre { white-space: pre-wrap; background: #f6f8fa; padding: 16px; border-radius: 8px; }
    .sources { margin-top: 16px; }
  </style>
</head>
<body>
  <h1>Legal RAG</h1>
  <form id="ask-form">
    <label for="question">Question</label>
    <textarea id="question" required placeholder="ما هي شروط العقد؟"></textarea>
    <button type="submit">Ask</button>
  </form>
  <h2>Answer</h2>
  <pre id="answer">No question yet.</pre>
  <div class="sources">
    <h2>Sources</h2>
    <ul id="sources"></ul>
  </div>
  <script>
    const form = document.getElementById('ask-form');
    const answer = document.getElementById('answer');
    const sources = document.getElementById('sources');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      answer.textContent = 'Loading...';
      sources.innerHTML = '';
      const question = document.getElementById('question').value;
      const response = await fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question}),
      });
      const data = await response.json();
      if (!response.ok) {
        answer.textContent = JSON.stringify(data, null, 2);
        return;
      }
      answer.textContent = data.answer;
      for (const source of data.sources) {
        const li = document.createElement('li');
        li.textContent = source;
        sources.appendChild(li);
      }
    });
  </script>
</body>
</html>
"""

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
REPORT_EXTENSIONS = IMAGE_EXTENSIONS | {".html", ".md", ".mmd"}
ARTICLE_CITATION_RE = re.compile(r"Egyptian Civil Code, Article \d+")


def format_article_citation(article_number: int) -> str:
    return f"Egyptian Civil Code, Article {article_number}"


def article_sources(result: Any) -> list[str]:
    cited_article_numbers = getattr(result, "cited_article_numbers", None)
    if cited_article_numbers is not None:
        return [
            format_article_citation(number)
            for number in dict.fromkeys(cited_article_numbers)
            if isinstance(number, int)
        ]

    return [
        citation
        for citation in getattr(result, "citations", [])
        if isinstance(citation, str) and ARTICLE_CITATION_RE.fullmatch(citation)
    ]


def create_default_agent(settings: Settings) -> LegalRagAgent:
    retriever = LegalRetriever(
        collection_name=settings.collection_name,
        persist_path=str(settings.chroma_persist_path),
        embedding_url=settings.embedding_base_url,
        embedding_model=settings.embedding_model,
        embedding_api_key=settings.embedding_api_key,
    )
    provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    model = OpenAIChatModel(settings.llm_model, provider=provider)
    return LegalRagAgent(model=model, retriever=retriever, top_k=settings.retrieval_top_k)


def get_documents_indexed(settings: Settings) -> int:
    store = ChromaVectorStore(
        collection_name=settings.collection_name,
        persist_path=str(settings.chroma_persist_path),
    )
    return store.count()


def create_app(
    agent: Any | None = None,
    documents_indexed: Callable[[], int] | None = None,
) -> FastAPI:
    settings = Settings()
    app = FastAPI(title="Legal RAG API")
    app.state.agent = agent
    app.state.create_agent = lambda: create_default_agent(settings)
    app.state.documents_indexed = documents_indexed or (lambda: get_documents_indexed(settings))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @observe(name="api.ask", as_type="span")
    def ask_observed(request: AskRequest) -> AskResponse:
        if app.state.agent is None:
            app.state.agent = app.state.create_agent()
        result = app.state.agent.answer(request.question)
        return AskResponse(answer=result.answer, sources=article_sources(result))

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        response = ask_observed(request)
        get_client().flush()
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="healthy", documents_indexed=app.state.documents_indexed())

    @app.get("/reports/", response_class=HTMLResponse)
    def reports() -> str:
        reports_dir = settings.reports_dir
        report_files = sorted(
            path
            for path in reports_dir.iterdir()
            if path.is_file() and path.suffix.lower() in REPORT_EXTENSIONS
        ) if reports_dir.exists() else []
        items = []
        for path in report_files:
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append(
                    f'<figure><img src="/reports/{path.name}" style="max-width:100%;height:auto"><figcaption>{path.name}</figcaption></figure>'
                )
            else:
                items.append(f'<p><a href="/reports/{path.name}">{path.name}</a></p>')
        return f"""
        <!doctype html>
        <html><head><title>Reports</title></head>
        <body style="font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 16px;">
        <h1>Reports</h1>
        {"\n".join(items) or "<p>No report files found.</p>"}
        </body></html>
        """

    @app.get("/reports/{filename}")
    def report_file(filename: str):
        reports_dir = settings.reports_dir.resolve()
        path = (reports_dir / filename).resolve()
        if reports_dir not in path.parents or not path.is_file() or path.suffix.lower() not in REPORT_EXTENSIONS:
            raise HTTPException(status_code=404, detail="report file not found")
        if path.suffix.lower() in {".md", ".mmd"}:
            return PlainTextResponse(path.read_text(encoding="utf-8"))
        return FileResponse(path)

    return app


app = create_app()
