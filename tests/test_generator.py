import pytest
import asyncio
from app.generator import generate_answer, RAGResponse

@pytest.mark.asyncio
async def test_generate_answer_simulated_success():
    passages = [
        {"text": "रक्तचाप मापने से 30 मिनट पहले कैफीन का सेवन न करें।", "url": "https://example.org/bp"}
    ]
    res = await generate_answer("रक्तचाप की जांच कैसे करें?", passages, language="hi")
    assert isinstance(res, RAGResponse)
    assert res.grounded is True
    assert res.refusal is False
    assert len(res.citations) > 0

@pytest.mark.asyncio
async def test_generate_answer_simulated_no_context():
    res = await generate_answer("unrelated query without matching topic", [], language="hi")
    assert isinstance(res, RAGResponse)
    assert res.refusal is True
    assert res.grounded is False
