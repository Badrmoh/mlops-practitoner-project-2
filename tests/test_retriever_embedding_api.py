from legrag.agent.retrieve import embedding_request_url, parse_embedding_response


def test_embedding_request_url_appends_embeddings_for_openai_base_url():
    assert (
        embedding_request_url("https://openrouter.ai/api/v1")
        == "https://openrouter.ai/api/v1/embeddings"
    )


def test_embedding_request_url_preserves_openai_embeddings_endpoint():
    assert (
        embedding_request_url("https://openrouter.ai/api/v1/embeddings")
        == "https://openrouter.ai/api/v1/embeddings"
    )


def test_embedding_request_url_preserves_ollama_embed_endpoint():
    assert embedding_request_url("http://localhost:11434/api/embed") == "http://localhost:11434/api/embed"


def test_parse_embedding_response_supports_ollama_shape():
    assert parse_embedding_response({"embeddings": [[1, "2.5"]]}) == [1.0, 2.5]


def test_parse_embedding_response_supports_openai_shape():
    assert parse_embedding_response({"data": [{"embedding": [1, "2.5"]}]}) == [1.0, 2.5]
