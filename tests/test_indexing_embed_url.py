from legrag.indexing.embed import openai_embedding_base_url


def test_openai_embedding_base_url_preserves_base_url():
    assert openai_embedding_base_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/v1"


def test_openai_embedding_base_url_strips_embeddings_endpoint():
    assert (
        openai_embedding_base_url("https://openrouter.ai/api/v1/embeddings")
        == "https://openrouter.ai/api/v1"
    )


def test_openai_embedding_base_url_strips_trailing_slash():
    assert openai_embedding_base_url("https://openrouter.ai/api/v1/") == "https://openrouter.ai/api/v1"
