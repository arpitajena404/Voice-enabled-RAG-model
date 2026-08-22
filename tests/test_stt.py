"""
Tests for the STT wrapper (app/stt.py).

Covers:
  - Happy path: mock httpx returns 200 + transcript → TranscriptionResult
  - Retry on transient: mock httpx fails twice with 503, then succeeds → retried
  - Fail fast on 4xx: mock httpx returns 401 → no retry, immediate error
  - Fallback mode: no API key → simulated transcript with used_fallback=True
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

# We need to patch config before importing stt
import app.config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_retry_config(monkeypatch):
    """Ensure fast retries for tests (no real backoff delay)."""
    monkeypatch.setattr(app.config.config, "RETRY_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(app.config.config, "RETRY_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(app.config.config, "RETRY_BACKOFF_MAX", 0.02)


def _make_response(status_code: int, json_body: dict | None = None, text: str = "") -> httpx.Response:
    """Create a fake httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://api.sarvam.ai/speech-to-text"),
    )
    if json_body is not None:
        resp._content = __import__("json").dumps(json_body).encode()
    else:
        resp._content = text.encode()
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTranscribeAudioHappyPath:
    """Mock the API to return a successful transcript."""

    @pytest.mark.asyncio
    async def test_returns_transcription_result(self, monkeypatch):
        monkeypatch.setattr(app.config.config, "SARVAM_API_KEY", "test-key-123")

        mock_response = _make_response(200, {"transcript": "hello world"})
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.stt.httpx.AsyncClient", return_value=mock_client):
            from app.stt import transcribe_audio
            result = await transcribe_audio(b"fake-audio", "test.wav", "en")

        assert result.text == "hello world"
        assert result.language_code == "en"
        assert result.used_fallback is False


class TestTranscribeAudioRetryOnTransient:
    """Verify that 5xx errors trigger retries and eventually succeed."""

    @pytest.mark.asyncio
    async def test_retries_on_503_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(app.config.config, "SARVAM_API_KEY", "test-key-123")

        fail_resp = _make_response(503, text="Service Unavailable")
        ok_resp = _make_response(200, {"transcript": "retried successfully"})

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return fail_resp
            return ok_resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = mock_post

        with patch("app.stt.httpx.AsyncClient", return_value=mock_client):
            from app.stt import transcribe_audio
            result = await transcribe_audio(b"fake-audio", "test.webm", "hi")

        assert result.text == "retried successfully"
        assert call_count == 3  # 2 failures + 1 success


class TestTranscribeAudioFailFastOn4xx:
    """Verify that 4xx errors do NOT trigger retries."""

    @pytest.mark.asyncio
    async def test_no_retry_on_401(self, monkeypatch):
        monkeypatch.setattr(app.config.config, "SARVAM_API_KEY", "bad-key")

        fail_resp = _make_response(401, text="Unauthorized")

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fail_resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = mock_post

        with patch("app.stt.httpx.AsyncClient", return_value=mock_client):
            from app.stt import transcribe_audio
            with pytest.raises(RuntimeError, match="client error"):
                await transcribe_audio(b"fake-audio", "test.wav", "en")

        # Should have been called exactly once — no retries
        assert call_count == 1


class TestTranscribeAudioFallback:
    """When SARVAM_API_KEY is empty, use simulated fallback."""

    @pytest.mark.asyncio
    async def test_fallback_returns_simulated(self, monkeypatch):
        monkeypatch.setattr(app.config.config, "SARVAM_API_KEY", "")

        from app.stt import transcribe_audio
        result = await transcribe_audio(b"fake-audio", "test.wav", "en")

        assert result.used_fallback is True
        assert result.text == "What is the procedure to get accurate blood pressure reading?"
        assert result.language_code == "en"

