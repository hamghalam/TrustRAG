
import pytest

from unittest.mock import Mock

from backend.app.services.llm_service import (
    OpenAICompatibleProvider,
    LLMProviderError,
    generate_grounded_answer,
)


from backend.app.services.llm_service import (
    OllamaProvider,
    get_llm_provider,
)


def test_ollama_provider_returns_answer(monkeypatch):
    mock_response = Mock()
    mock_response.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "RAG retrieves relevant information."
        }
    }

    monkeypatch.setattr(
        "httpx.post",
        lambda *args, **kwargs: mock_response,
    )

    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="qwen3.5:0.8b",
    )

    answer = provider.generate(
        system_prompt="System",
        user_prompt="Explain RAG.",
    )

    assert answer == "RAG retrieves relevant information."


def test_ollama_provider_uses_native_api(monkeypatch):
    mock_response = Mock()
    mock_response.json.return_value = {
        "message": {"content": "Test answer"}
    }

    captured = {}

    def mock_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return mock_response

    monkeypatch.setattr("httpx.post", mock_post)

    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="qwen3.5:0.8b",
    )

    provider.generate("System", "Question")

    assert captured["url"] == (
        "http://localhost:11434/api/chat"
    )

    assert captured["json"]["think"] is False
    assert captured["json"]["stream"] is False
    assert captured["json"]["model"] == "qwen3.5:0.8b"


def test_get_llm_provider_returns_ollama(monkeypatch):
    monkeypatch.setenv("TRUSTRAG_LLM_PROVIDER", "ollama")
    monkeypatch.setenv(
        "TRUSTRAG_LLM_BASE_URL",
        "http://localhost:11434",
    )
    monkeypatch.setenv(
        "TRUSTRAG_LLM_MODEL",
        "qwen3.5:0.8b",
    )

    provider = get_llm_provider()

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen3.5:0.8b"


def test_get_llm_provider_returns_openai_compatible(monkeypatch):
    monkeypatch.setenv(
        "TRUSTRAG_LLM_PROVIDER",
        "openai_compatible",
    )
    monkeypatch.setenv(
        "TRUSTRAG_LLM_BASE_URL",
        "https://example.com/v1",
    )
    monkeypatch.setenv(
        "TRUSTRAG_LLM_MODEL",
        "test-model",
    )

    provider = get_llm_provider()

    assert isinstance(provider, OpenAICompatibleProvider)


def test_get_llm_provider_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv(
        "TRUSTRAG_LLM_PROVIDER",
        "unknown",
    )

    with pytest.raises(ValueError):
        get_llm_provider()

def test_provider_returns_answer(monkeypatch):
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "The tumor decreased in size [1]."
                }
            }
        ]
    }

    monkeypatch.setattr(
        "httpx.post",
        lambda *args, **kwargs: mock_response,
    )

    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        model="qwen3:8b",
    )

    answer = provider.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="What happened?",
    )

    assert answer == "The tumor decreased in size [1]."


def test_provider_sends_correct_request(monkeypatch):
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Answer [1]."
                }
            }
        ]
    }

    captured = {}

    def mock_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return mock_response

    monkeypatch.setattr("httpx.post", mock_post)

    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1/",
        model="test-model",
        api_key="test-key",
    )

    provider.generate(
        system_prompt="System",
        user_prompt="Question",
    )

    assert captured["url"] == (
        "http://localhost:11434/v1/chat/completions"
    )

    assert captured["headers"]["Authorization"] == (
        "Bearer test-key"
    )

    assert captured["json"]["model"] == "test-model"


def test_provider_rejects_empty_model():
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1",
            model="",
        )


def test_provider_rejects_invalid_timeout():
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1",
            model="test-model",
            timeout=0,
        )


def test_empty_context_does_not_call_llm(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("LLM should not be called.")

    monkeypatch.setattr(
        "backend.app.services.llm_service.get_llm_provider",
        unexpected_call,
    )

    result = generate_grounded_answer({
        "query": "What happened?",
        "context": "",
        "sources": [],
    })

    assert result["sources"] == []
    assert result["model"] is None
    assert "could not find" in result["answer"].lower()


def test_grounded_answer_preserves_sources(monkeypatch):
    mock_provider = Mock()
    mock_provider.model = "test-model"
    mock_provider.generate.return_value = "The tumor decreased [1]."

    monkeypatch.setattr(
        "backend.app.services.llm_service.get_llm_provider",
        lambda: mock_provider,
    )

    sources = [
        {
            "citation": "[1]",
            "filename": "report.pdf",
            "page_number": 2,
        }
    ]

    result = generate_grounded_answer({
        "query": "What happened to the tumor?",
        "context": "[1]\nThe tumor decreased in size.",
        "sources": sources,
    })

    assert result["answer"] == "The tumor decreased [1]."
    assert result["sources"] == sources
    assert result["model"] == "test-model"


def test_provider_raises_error_on_http_failure(monkeypatch):
    import httpx

    def mock_post(*args, **kwargs):
        request = httpx.Request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
        )

        response = httpx.Response(
            500,
            request=request,
        )

        raise httpx.HTTPStatusError(
            "Server error",
            request=request,
            response=response,
        )

    monkeypatch.setattr("httpx.post", mock_post)

    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        model="test-model",
    )

    with pytest.raises(LLMProviderError):
        provider.generate(
            system_prompt="System",
            user_prompt="Question",
        )


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        generate_grounded_answer({
            "query": " ",
            "context": "Some context",
            "sources": [],
        })
