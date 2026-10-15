from legrag.api import create_app


class FakeAnswer:
    answer = "Answer text"
    cited_article_numbers = [1]
    citations = ["Egyptian Civil Code, Article 1"]


class FakeAgent:
    def answer(self, question: str) -> FakeAnswer:
        return FakeAnswer()


def test_ask_rejects_empty_question():
    from fastapi.testclient import TestClient

    client = TestClient(create_app(agent=FakeAgent(), documents_indexed=lambda: 3))
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_health_returns_document_count():
    from fastapi.testclient import TestClient

    client = TestClient(create_app(agent=FakeAgent(), documents_indexed=lambda: 3))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "documents_indexed": 3}


def test_ask_returns_answer_and_article_citations():
    from fastapi.testclient import TestClient

    client = TestClient(create_app(agent=FakeAgent(), documents_indexed=lambda: 3))
    response = client.post("/ask", json={"question": "What is Article 1?"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer text",
        "sources": ["Egyptian Civil Code, Article 1"],
    }


def test_ask_sources_prefer_article_numbers_over_agent_citation_strings():
    from fastapi.testclient import TestClient

    class ChunkCitationAnswer:
        answer = "Answer text"
        cited_article_numbers = [12, 12, 13]
        citations = ["article-12-en-0", "page 4"]

    class ChunkCitationAgent:
        def answer(self, question: str) -> ChunkCitationAnswer:
            return ChunkCitationAnswer()

    client = TestClient(create_app(agent=ChunkCitationAgent(), documents_indexed=lambda: 3))
    response = client.post("/ask", json={"question": "What is Article 12?"})
    assert response.status_code == 200
    assert response.json()["sources"] == [
        "Egyptian Civil Code, Article 12",
        "Egyptian Civil Code, Article 13",
    ]


def test_ask_sources_filter_non_article_citation_fallbacks():
    from fastapi.testclient import TestClient

    class LegacyAnswer:
        answer = "Answer text"
        citations = ["Egyptian Civil Code, Article 20", "article-20-en-0", "page 7"]

    class LegacyAgent:
        def answer(self, question: str) -> LegacyAnswer:
            return LegacyAnswer()

    client = TestClient(create_app(agent=LegacyAgent(), documents_indexed=lambda: 3))
    response = client.post("/ask", json={"question": "What is Article 20?"})
    assert response.status_code == 200
    assert response.json()["sources"] == ["Egyptian Civil Code, Article 20"]
