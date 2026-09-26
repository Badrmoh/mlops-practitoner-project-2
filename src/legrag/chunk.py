import json
import re
import logging
from os import getenv
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


load_dotenv()

CHUNK_SIZE_CHARS = int(getenv("CHUNK_SIZE_CHARS", 1200))
CHUNK_OVERLAP_CHARS = int(getenv("CHUNK_OVERLAP_CHARS", 150))
LEGAL_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
    separators=[
        r"(?<=[.!؟:؛])\s+(?=\d+\b)",  # numbered paragraphs
        r"(?<=[.!؟])\s+",             # sentences
        "\n\n",
        "\n",
        " ",
        "",
    ],
    is_separator_regex=True,
)

model = getenv("EMBEDDING_MODEL", "nomic-embed-text-v2-moe")


def load_article(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def split_article_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE_CHARS:
        return [text]
    return LEGAL_TEXT_SPLITTER.split_text(text)


def create_documents(articles: list[dict]) -> list[Document]:
    documents: list[Document] = []
    metadata: dict = {}
    for article in articles:
        chunks_en: str = ""
        chunks_ar: str = ""
        metadata={
            "article_number": article["article_number"],
            "book": article["book"],
            "chapter": article["chapter"],
            "section": article["section"],
            "topic": article["topic"],
            "source_page": article["source_page"],
            "citation": article["citation"],
            "is_repealed": article["is_repealed"],
            "referenced_articles": article["references"],
        }

        chunks_en = split_article_text(article["text_en"])
        for i, chunk in enumerate(chunks_en):
            eng_doc = Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "language": "en",
                    "chunk_index": i,
                    "chunk_count": len(chunks_en),
                    "chunk_id": (
                        f"article-"
                        f"{article["article_number"]}-"
                        f"en-{i}"
                    )
                }
            )
            documents.append(eng_doc)

        chunks_ar = split_article_text(article["text_ar"])
        for i, chunk in enumerate(chunks_ar):
            ar_doc = Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "language": "ar",
                    "chunk_count": len(chunks_ar),
                    "chunk_id": (
                        f"article-"
                        f"{article["article_number"]}-"
                        f"ar-{i}"
                    )
                }
            )
            documents.append(ar_doc)

    return documents


def create_vector_store():
    embeddings = OllamaEmbeddings(
        model=model,
        base_url=getenv("OLLAMA_BASE_URL"),
        dimensions=768
    )

    vector_store = PGVector(
        embeddings=embeddings,
        embedding_length=768,
        collection_name=getenv(
            "COLLECTION_NAME",
            "egyptian_civil_code_v3"
        ),
        connection=getenv("POSTGRES_CONNECTION"),
        use_jsonb=True
    )
    return vector_store, embeddings


def add_documents_in_batches(vector_store,
                             documents,
                             ids: list[str],
                             batch_size: int = 128
                             ):
    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        vector_store.add_documents(batch,
            ids=ids[start:start + batch_size]
        )
        _log.info(f"added {start + len(batch)}/{len(documents)}")


def embed():
    articles = load_article("law.articles.json")
    documents = create_documents(articles)
    ids = [
        f"{getenv('COLLECTION_NAME')}-{doc.metadata['chunk_id']}"
        for doc in documents
    ]
    vector_store, _ = create_vector_store()
    add_documents_in_batches(vector_store, documents, ids)
    return vector_store

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger(__name__)
    _log.info("Parameters")
    _log.info(f"chunk_size: {CHUNK_SIZE_CHARS}")
    _log.info(f"chunk_overlap: {CHUNK_OVERLAP_CHARS}")
    _log.info(f"collection_name: {getenv('COLLECTION_NAME')}")
    vector_store = embed()
    _log.info("Vector store created successfully.")
    question = "ما هي شروط العقد؟"
    #question = "what are the terms of the contract?"
    #vector_store, _ = create_vector_store()
    documents_with_scores = vector_store.similarity_search_with_score(question, k=5)
    _log.info(documents_with_scores)
    for doc, score in documents_with_scores:
        _log.info("Score:", str(score))
        _log.info(doc.page_content)
        _log.info(doc.metadata)
