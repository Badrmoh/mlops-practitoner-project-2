from pydantic_ai import Capability, RunContext, Tool

from legrag.agent.schemas import RagDeps, RetrievedChunk


def retrieve_legal_context(ctx: RunContext[RagDeps], query: str) -> list[RetrievedChunk]:
    """Retrieve relevant Egyptian Civil Code chunks with article_number, citation, language, text, and distance metadata."""
    chunks = ctx.deps.retriever.retrieve(
        query,
        top_k=ctx.deps.top_k,
        distance_threshold=ctx.deps.distance_threshold,
    )
    ctx.deps.retrieved_contexts = chunks
    return chunks


legal_retrieval_capability = Capability(
    id="legal-retrieval",
    description="Retrieve Egyptian Civil Code article chunks with citation metadata.",
    instructions=(
        "Use retrieve_legal_context to obtain relevant Egyptian Civil Code chunks before answering. "
        "Use the returned article_number values for cited_article_numbers. "
        "Use the returned text as the only legal source."
    ),
    tools=[Tool(retrieve_legal_context, takes_ctx=True)],  # type: ignore[arg-type]
)


legal_answer_guardrails = Capability(
    id="legal-answer-guardrails",
    description="Instruction-level grounding, citation, and answer-discipline rules for legal RAG answers.",
    instructions=(
        "Think privately about which supplied article most directly answers the question, but do not reveal reasoning. "
        "Answer only from the supplied chunks. "
        "Answer only the exact question asked. "
        "Use the minimum number of articles needed, preferring the single most directly relevant article. "
        "Do not add related rules, exceptions, contrasts, or neighboring article summaries unless they are necessary to answer the question. "
        "Do not mention retrieved context, provided context, chunks, or excerpts. "
        "If at least one supplied article directly answers the question, answer using that article. "
        "Set answered=false only when no supplied article addresses the question. "
        "Do not speculate or infer legal procedures from related but non-direct articles. "
        "Cite only article numbers present in the supplied chunks. "
        "If no supplied article addresses the question, set answered=false and say no direct answer is available in the same language as the question. "
        "Do not synthesize from related rules. "
        "Do not explain what the context is about. "
        "Every legal claim must be supported by an article number."
    ),
)
