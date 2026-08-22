import logging
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
from app.config import config
from app.schemas import TranscriptionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry predicate: only retry on *transient* failures, NOT on 4xx / auth
# ---------------------------------------------------------------------------
class SarvamTransientError(RuntimeError):
    """Raised when the Sarvam API returns a retryable (5xx / timeout) error."""


def _is_transient(exc: BaseException) -> bool:
    """Return True for errors that should trigger a retry."""
    if isinstance(exc, SarvamTransientError):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    return False


async def _call_sarvam(file_bytes: bytes, filename: str, language_code: str) -> str:
    """
    Low-level Sarvam API call wrapped with tenacity retry.

    Retries on transient failures (5xx, timeouts, connection errors).
    Fails fast on 4xx / auth errors (no retry).
    """
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": config.SARVAM_API_KEY
    }

    # Standard configuration parameters for Sarvam Speech-to-Text
    data = {
        "model": "saaras:v3",
        "mode": "codemix"  # handles mixed languages (e.g. Hindi + English)
    }

    # Audio content types based on filename extensions
    content_type = "audio/webm"
    if filename.endswith(".wav"):
        content_type = "audio/wav"
    elif filename.endswith(".mp3"):
        content_type = "audio/mp3"

    files = {
        "file": (filename, file_bytes, content_type)
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, headers=headers, files=files, data=data)

        # 4xx → fail fast (auth error, bad request, etc.)
        if 400 <= response.status_code < 500:
            logger.error(f"Sarvam API client error: Status {response.status_code}, Body: {response.text}")
            raise RuntimeError(f"Sarvam STT client error (HTTP {response.status_code}) — not retryable")

        # 5xx → raise a transient error so tenacity retries
        if response.status_code >= 500:
            logger.warning(f"Sarvam API server error: Status {response.status_code}, will retry...")
            raise SarvamTransientError(f"Sarvam STT server error (HTTP {response.status_code})")

        if response.status_code != 200:
            logger.error(f"Sarvam API unexpected status: {response.status_code}, Body: {response.text}")
            raise RuntimeError(f"Sarvam STT failed with status {response.status_code}")

        res_json = response.json()
        # Sarvam STT typically returns: {"transcript": "..."}
        transcript = res_json.get("transcript", res_json.get("text", ""))
        return transcript.strip()


def _build_retry_decorator():
    """Build a tenacity retry decorator using current config values."""
    return retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=config.RETRY_BACKOFF_BASE,
            max=config.RETRY_BACKOFF_MAX,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


async def transcribe_audio(
    file_bytes: bytes,
    filename: str = "audio.webm",
    language_code: str = "hi",
) -> TranscriptionResult:
    """
    Transcribes audio bytes using Sarvam AI Speech-to-Text API.
    If no SARVAM_API_KEY is provided, runs in simulated fallback mode.

    Returns a structured TranscriptionResult.
    """
    if not config.SARVAM_API_KEY:
        logger.warning("SARVAM_API_KEY not set. Running in simulated STT mode.")
        # Simulating transcription based on typical validation queries
        simulated_transcripts = {
            "hi": "रक्तचाप मापने के लिए सटीक रीडिंग प्राप्त करने की प्रक्रिया क्या है?",
            "bn": "রক্তচাপ মাপার জন্য সঠিক রিডিং নিশ্চিত করার উপায় কী?",
            "en": "What is the procedure to get accurate blood pressure reading?",
            "ta": "இரத்த அழுத்தத்தை துல்லியமாக அளவிடுவதற்கான வழிமுறை என்ன?",
            "te": "రక్తపోటును ఖచ్చితంగా కొలవడానికి విధానం ఏమిటి?",
            "mr": "रक्तदाब अचूक मोजण्याची पद्धत काय आहे?",
            "gu": "બ્લડ પ્રેશરનું ચોક્કસ રીડિંગ મેળવવાની પ્રક્રિયા શું છે?",
            "kn": "ರಕ್ತದೊತ್ತಡವನ್ನು ನಿಖರವಾಗಿ ಅಳೆಯುವ ವಿಧಾನ ಯಾವುದು?",
            "ml": "രക്തസമ്മർദ്ദം കൃത്യമായി അളക്കുന്നതിനുള്ള നടപടിക്രമം എന്താണ്?",
            "pa": "ਬਲੱਡ ਪ੍ਰੈਸ਼ਰ ਦੀ ਸਹੀ ਰੀਡਿੰਗ ਲੈਣ ਦੀ ਪ੍ਰਕਿਰਿਆ ਕੀ ਹੈ?",
            "od": "ରକ୍ତଚାପର ସଠିକ୍ ରିଡିଂ ପାଇବା ପାଇଁ ପ୍ରକ୍ରିୟା କ’ଣ?"
        }

        return TranscriptionResult(
            text=simulated_transcripts.get(language_code, simulated_transcripts["hi"]),
            language_code=language_code,
            latency_ms=0.0,  # caller fills in real latency via timed_stage
            used_fallback=True,
        )

    try:
        # Apply retry decorator at call time so it picks up current config
        retrying_call = _build_retry_decorator()(_call_sarvam)
        transcript = await retrying_call(file_bytes, filename, language_code)
        return TranscriptionResult(
            text=transcript,
            language_code=language_code,
            latency_ms=0.0,  # caller fills in real latency via timed_stage
            used_fallback=False,
        )

    except Exception as e:
        logger.exception("Error calling Sarvam Speech-to-Text API (all retries exhausted)")
        raise RuntimeError(f"Sarvam STT API call failed: {str(e)}")
