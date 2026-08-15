import logging
import httpx
from app.config import config

logger = logging.getLogger(__name__)

async def transcribe_audio(file_bytes: bytes, filename: str = "audio.webm", language_code: str = "hi") -> str:
    """
    Transcribes audio bytes using Sarvam AI Speech-to-Text API.
    If no SARVAM_API_KEY is provided, runs in simulated fallback mode.
    """
    if not config.SARVAM_API_KEY:
        logger.warning("SARVAM_API_KEY not set. Running in simulated STT mode.")
        # Simulating transcription based on typical validation queries
        simulated_transcripts = {
            "hi": "रक्तचाप मापने के लिए सटीक रीडिंग प्राप्त करने की प्रक्रिया क्या है?",
            "bn": "রক্তচাপ মাপার জন্য সঠিক রিডিং নিশ্চিত করার উপায় কী?",
            "en": "how to get accurate blood pressure reading"
        }
        return simulated_transcripts.get(language_code, "रक्तचाप मापने के लिए सटीक रीडिंग प्राप्त करने की प्रक्रिया क्या है?")

    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": config.SARVAM_API_KEY
    }
    
    # Map input language to appropriate Sarvam mode/model if needed
    # Standard configuration parameters for Sarvam Speech-to-Text
    data = {
        "model": "saaras:v3",
        "mode": "codemix"  # "codemix" handles mixed languages (e.g. Hindi + English) exceptionally well
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

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, files=files, data=data)
            
            if response.status_code != 200:
                logger.error(f"Sarvam API error: Status {response.status_code}, Body: {response.text}")
                raise RuntimeError(f"Sarvam STT failed with status {response.status_code}")
                
            res_json = response.json()
            # Sarvam STT typically returns: {"transcript": "..."}
            transcript = res_json.get("transcript", res_json.get("text", ""))
            return transcript.strip()
            
    except Exception as e:
        logger.exception("Error calling Sarvam Speech-to-Text API")
        # In case of API failures, fall back to a reasonable message or raise
        raise RuntimeError(f"Sarvam STT API call failed: {str(e)}")
