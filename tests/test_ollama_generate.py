"""
Unit tests for OllamaService.generate()

Covers:
- Normal single response
- Timeout retries up to configured limit
- ConnectError raises 503 immediately
- ResponseError raises immediately with upstream status
"""
import asyncio
import pytest
import httpx
import ollama
from unittest.mock import MagicMock, patch

from src.services.ollama_service import OllamaService
from fastapi import HTTPException


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_response(response="", done=True, thinking=None, done_reason=None, eval_count=None, model="gpt-oss:latest"):
    """Build a mock ollama.GenerateResponse."""
    resp = MagicMock(spec=ollama.GenerateResponse)
    resp.response = response
    resp.done = done
    resp.thinking = thinking
    resp.done_reason = done_reason
    resp.eval_count = eval_count
    resp.model = model
    resp.model_dump = lambda: {
        "model": model, "response": response, "done": done,
        "thinking": thinking, "done_reason": done_reason, "eval_count": eval_count,
    }
    return resp


def _mock_client(response=None, raises=None):
    """
    Return a factory producing a mock ollama.AsyncClient.
    generate() returns `response` or raises `raises`.
    """
    async def _generate(**_):
        if raises is not None:
            raise raises
        return response or _make_response()

    mock_client_instance = MagicMock()
    mock_client_instance.generate = _generate
    return MagicMock(return_value=mock_client_instance)


@pytest.fixture
def service() -> OllamaService:
    svc = OllamaService(base_url="http://localhost:11434", model="gpt-oss:latest", timeout=30)
    svc.retries = 0
    return svc


# ── generate() tests ──────────────────────────────────────────────────────────

def test_generate_normal_response(service):
    resp = _make_response(response="Paris is the capital of France.", eval_count=10)

    with patch.object(service, "_client", _mock_client(response=resp)):
        result = _run(service.generate(prompt="What is the capital of France?"))

    assert result.response == "Paris is the capital of France."
    assert result.done is True


def test_generate_retries_on_timeout_then_succeeds(service):
    service.retries = 2
    call_count = 0
    success_resp = _make_response(response="Success after retry.")

    async def _generate(**_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.TimeoutException("timeout")
        return success_resp

    mock_client_instance = MagicMock()
    mock_client_instance.generate = _generate

    with patch.object(service, "_client", MagicMock(return_value=mock_client_instance)):
        result = _run(service.generate(prompt="Hello"))

    assert result.response == "Success after retry."
    assert call_count == 2


def test_generate_raises_504_after_all_retries_exhausted(service):
    service.retries = 2

    with patch.object(service, "_client", _mock_client(raises=httpx.TimeoutException("timeout"))):
        with pytest.raises(HTTPException) as exc_info:
            _run(service.generate(prompt="Hello"))

    assert exc_info.value.status_code == 504


def test_generate_raises_503_on_connect_error_without_retry(service):
    service.retries = 3
    call_count = 0

    async def _generate(**_):
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("refused")

    mock_client_instance = MagicMock()
    mock_client_instance.generate = _generate

    with patch.object(service, "_client", MagicMock(return_value=mock_client_instance)):
        with pytest.raises(HTTPException) as exc_info:
            _run(service.generate(prompt="Hello"))

    assert exc_info.value.status_code == 503
    assert call_count == 1  # no retries on connect error


def test_generate_raises_503_on_connection_error_without_retry(service):
    """ollama library wraps httpx.ConnectError as ConnectionError in non-streaming path."""
    service.retries = 3

    with patch.object(service, "_client", _mock_client(raises=ConnectionError("refused"))):
        with pytest.raises(HTTPException) as exc_info:
            _run(service.generate(prompt="Hello"))

    assert exc_info.value.status_code == 503


def test_generate_raises_on_response_error(service):
    err = ollama.ResponseError("Internal Server Error", status_code=500)

    with patch.object(service, "_client", _mock_client(raises=err)):
        with pytest.raises(HTTPException) as exc_info:
            _run(service.generate(prompt="Hello"))

    assert exc_info.value.status_code == 500


def test_generate_max_tokens_applied_by_default(service):
    """num_predict must always be set so thinking models have a token budget."""
    captured_options = {}

    async def _generate(**kwargs):
        captured_options["options"] = kwargs.get("options")
        return _make_response(response="Hi.")

    mock_client_instance = MagicMock()
    mock_client_instance.generate = _generate

    with patch.object(service, "_client", MagicMock(return_value=mock_client_instance)):
        _run(service.generate(prompt="Hello"))

    opts = captured_options.get("options")
    assert opts is not None and opts.num_predict is not None, (
        "num_predict must be set so thinking models have a token budget for their response"
    )


def test_generate_done_reason_length_returns_empty_response(service):
    """When done_reason='length' the response may be empty — returned as-is without error."""
    resp = _make_response(response="", done=True, done_reason="length", eval_count=150)

    with patch.object(service, "_client", _mock_client(response=resp)):
        result = _run(service.generate(prompt="Hello"))

    assert result.done is True
    assert result.response == ""
