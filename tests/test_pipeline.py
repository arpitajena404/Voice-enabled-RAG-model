import pytest
from app.pipeline import _has_script_for_language, _detect_query_language, pipeline

def test_has_script_for_language_odia():
    # English hallucination should return False for Odia
    assert _has_script_for_language("I'm not sorry to go get the bunny door car", "od") is False
    # Native Odia script should return True for Odia
    assert _has_script_for_language("ରକ୍ତଚାପର ସଠିକ୍ ରିଡିଂ ପାଇବା ପାଇଁ ପ୍ରକ୍ରିୟା କ’ଣ?", "od") is True

def test_has_script_for_language_hindi():
    assert _has_script_for_language("how to check blood pressure", "hi") is False
    assert _has_script_for_language("रक्तचाप की जांच कैसे करें?", "hi") is True

@pytest.mark.asyncio
async def test_pipeline_odia_audio_fallback():
    # Simulate receiving English hallucination from browser STT along with audio bytes for Odia language
    res = await pipeline.run(
        query_text="I'm not sorry to go get the bunny door car",
        audio_bytes=b"dummy_audio_content",
        language="od",
        strategy="semantic"
    )
    # The pipeline should reject the English hallucination and use Odia audio transcription
    assert res.query != "I'm not sorry to go get the bunny door car"
    assert "ରକ୍ତଚାପ" in res.query or "ସଠିକ୍" in res.query
    assert res.refusal is False
