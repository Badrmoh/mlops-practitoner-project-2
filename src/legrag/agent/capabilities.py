from pydantic_ai import Capability, RunContext, Tool

from legrag.agent.schemas import RagDeps, RetrievedChunk


def retrieve_legal_context(ctx: RunContext, query: str) -> list[RetrievedChunk]:
    """Retrieve relevant Egyptian Civil Code chunks for a legal question."""
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
        "Use retrieved legal context to answer. Cite article numbers explicitly. "
        "If the retrieved context is insufficient, say that the answer is not available from the provided law excerpts."
    ),
    tools=[Tool(retrieve_legal_context, takes_ctx=True)],  # type: ignore[arg-type]
)


legal_answer_guardrails = Capability(
    id="legal-answer-guardrails",
    description="Grounding, citation, and answer-discipline rules for legal RAG answers.",
    instructions=(
        "Think privately about whether the supplied chunks directly answer the question, but do not reveal reasoning. "
        "Answer only from the supplied chunks. "
        "Do not mention retrieved context, provided context, chunks, or excerpts. "
        "Do not speculate or infer legal procedures from related but non-direct articles. "
        "Cite only article numbers present in the supplied chunks. "
        "If no retrieved article directly states the answer, set answered=false and say no direct answer is available in the same language as the question. "
        "Do not synthesize from related rules. "
        "Do not explain what the context is about. "
        "Every legal claim must be supported by an article number."
    ),
)
