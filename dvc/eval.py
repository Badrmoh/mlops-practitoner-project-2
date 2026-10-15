import json
from os import getenv
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.run_config import RunConfig

from legrag.agent import LegalRagAgent
from legrag.agent.retrieve import LegalRetriever
from legrag.config import REPO_ROOT, Settings


def load_params() -> dict[str, Any]:
    with (REPO_ROOT / "dvc/params.yaml").open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def ollama_embedding_model_name(model: str) -> str:
    return model.removeprefix("ollama:")


def build_agent(settings: Settings, embed_params: dict[str, Any], eval_params: dict[str, Any]) -> LegalRagAgent:
    retriever = LegalRetriever(
        collection_name=embed_params["collection_name"],
        persist_path=str(resolve_repo_path(embed_params["chroma_persist_path"])),
        embedding_url=settings.embedding_base_url,
        embedding_model=ollama_embedding_model_name(embed_params["embedding_model"]),
        embedding_api_key=settings.embedding_api_key,
    )
    llm_provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    agent_model = OpenAIChatModel(eval_params["agent_model"], provider=llm_provider)
    return LegalRagAgent(
        model=agent_model,
        retriever=retriever,
        top_k=eval_params["top_k"],
        distance_threshold=eval_params.get("distance_threshold"),
    )


def run_exp(settings: Settings, params: dict[str, Any]) -> dict[str, float | int]:
    embed_params = params["embed"]
    eval_params = params["eval"]
    agent = build_agent(settings, embed_params, eval_params)

    questions_path = resolve_repo_path(eval_params["golden_questions_path"])
    metrics_path = resolve_repo_path(eval_params["metrics_json_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    eval_rows = []
    detailed_results = []

    for item in questions:
        answer = agent.answer(item["question"])
        contexts = [chunk.text for chunk in answer.contexts]
        eval_rows.append(
            {
                "user_input": item["question"],
                "response": answer.answer,
                "retrieved_contexts": contexts,
            }
        )
        detailed_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "golden_answer": item["answer"],
                "expected_article_number": item["article_number"],
                "answer": answer.answer,
                "answered": answer.answered,
                "citations": answer.citations,
                "cited_article_numbers": answer.cited_article_numbers,
                "retrieved_article_numbers": [chunk.article_number for chunk in answer.contexts],
                "retrieved_languages": [chunk.language for chunk in answer.contexts],
                "retrieved_chunk_ids": [chunk.chunk_id for chunk in answer.contexts],
                "retrieved_distances": [chunk.distance for chunk in answer.contexts],
                "retrieved_contexts": contexts,
            }
        )

    dataset = Dataset.from_list(eval_rows)
    judge_llm = ChatOpenAI(
        model=eval_params["judge_model"],
        base_url=settings.llm_base_url,
        api_key=SecretStr(settings.llm_api_key),
        temperature=0,
    )
    run_config = RunConfig(timeout=300, max_workers=1, max_retries=1)
    ragas_result = evaluate(dataset, metrics=[faithfulness], llm=judge_llm, run_config=run_config)
    scores = ragas_result.to_pandas()
    faithfulness_scores = [
        float(score)
        for score in scores["faithfulness"].tolist()
        if score == score
    ]
    faithfulness_mean = sum(faithfulness_scores) / len(faithfulness_scores)

    num_questions = len(questions)
    answered_count = sum(1 for item in detailed_results if item["answered"])
    retrieval_hits = sum(
        1
        for item in detailed_results
        if item["expected_article_number"] in item["retrieved_article_numbers"]
    )
    no_context_count = sum(1 for item in detailed_results if not item["retrieved_contexts"])
    avg_retrieved_contexts = sum(len(item["retrieved_contexts"]) for item in detailed_results) / num_questions
    avg_cited_articles = sum(len(item["cited_article_numbers"]) for item in detailed_results) / num_questions
    expected_ranks = []
    reciprocal_ranks = []
    retrieved_language_counts: dict[str, int] = {}
    retrieved_context_chars = []
    answered_when_hit = 0
    hit_count = 0
    cited_expected_count = 0

    for item in detailed_results:
        expected_article_number = item["expected_article_number"]
        retrieved_article_numbers = item["retrieved_article_numbers"]
        retrieved_context_chars.extend(len(context) for context in item["retrieved_contexts"])
        for language in item["retrieved_languages"]:
            key = language or "unknown"
            retrieved_language_counts[key] = retrieved_language_counts.get(key, 0) + 1

        if expected_article_number in retrieved_article_numbers:
            rank = retrieved_article_numbers.index(expected_article_number) + 1
            expected_ranks.append(rank)
            reciprocal_ranks.append(1 / rank)
            hit_count += 1
            if item["answered"]:
                answered_when_hit += 1

        if expected_article_number in item["cited_article_numbers"]:
            cited_expected_count += 1

    total_retrieved_contexts = sum(retrieved_language_counts.values()) or 1

    metrics = {
        "ragas_faithfulness": faithfulness_mean,
        "num_questions": num_questions,
        "answered_rate": answered_count / num_questions,
        "retrieval_hit_rate": retrieval_hits / num_questions,
        "no_context_count": no_context_count,
        "avg_retrieved_contexts": avg_retrieved_contexts,
        "avg_cited_articles": avg_cited_articles,
        "expected_article_at_1_rate": sum(1 for rank in expected_ranks if rank == 1) / num_questions,
        "expected_article_mrr": sum(reciprocal_ranks) / num_questions,
        "avg_expected_article_rank": (sum(expected_ranks) / len(expected_ranks)) if expected_ranks else 0,
        "answered_when_expected_retrieved_rate": (answered_when_hit / hit_count) if hit_count else 0,
        "cited_expected_article_rate": cited_expected_count / num_questions,
        "retrieved_language_en_rate": retrieved_language_counts.get("en", 0) / total_retrieved_contexts,
        "retrieved_language_ar_rate": retrieved_language_counts.get("ar", 0) / total_retrieved_contexts,
        "avg_retrieved_context_chars": (
            sum(retrieved_context_chars) / len(retrieved_context_chars)
        ) if retrieved_context_chars else 0,
        "max_retrieved_context_chars": max(retrieved_context_chars) if retrieved_context_chars else 0,
    }

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return metrics


def main() -> None:
    settings = Settings()
    params = load_params()

    if mlflow_tracking_uri := getenv("MLFLOW_TRACKING_URI"):
        import mlflow

        embed_params = params["embed"]
        eval_params = params["eval"]

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("legrag-rag-eval")
        with mlflow.start_run(run_name="rag-eval-1"):
            mlflow.log_param("chunk.max_tokens_per_chunk", params["chunk"]["max_tokens_per_chunk"])
            mlflow.log_param("chunk.chunk_overlap_tokens", params["chunk"]["chunk_overlap_tokens"])
            mlflow.log_param("chunk.approx_chars_per_token", settings.approx_chars_per_token)
            mlflow.log_param("embedding.model", embed_params["embedding_model"])
            mlflow.log_param("embedding.collection_name", embed_params["collection_name"])
            mlflow.log_param("eval.top_k", eval_params["top_k"])
            mlflow.log_param("eval.agent_model", eval_params["agent_model"])
            mlflow.log_param("eval.judge_model", eval_params["judge_model"])
            mlflow.log_param("eval.distance_threshold", eval_params.get("distance_threshold"))

            metrics = run_exp(settings, params)
            mlflow.log_metrics(metrics)
    else:
        _ = run_exp(settings, params)


if __name__ == "__main__":
    main()
