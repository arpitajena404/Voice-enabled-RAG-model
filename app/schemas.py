"""
Structured request/response models for the Voice-Enabled RAG pipeline.

Each pipeline stage produces a typed result object. PipelineResponse
composes them and provides `.to_api_dict()` for backward-compatible
JSON serialization matching what static/app.js expects.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    """Output of the STT stage."""
    text: str = Field(description="Transcribed text from the audio input.")
    language_code: str = Field(description="Language code used for transcription (e.g. 'hi', 'bn', 'en').")
    latency_ms: float = Field(default=0.0, description="Wall-clock time for the STT call in milliseconds.")
    used_fallback: bool = Field(default=False, description="True if the simulated/fallback STT path was used instead of the real API.")


class RetrievalResult(BaseModel):
    """Output of the retrieval stage."""
    passages: list[dict] = Field(default_factory=list, description="Ranked list of retrieved passages with scores and metadata.")
    strategy: str = Field(default="semantic", description="Chunking/retrieval strategy used (naive, semantic, parent_child).")
    latency_ms: float = Field(default=0.0, description="Wall-clock time for retrieval in milliseconds.")


class GenerationResult(BaseModel):
    """Output of the LLM generation + grounding guardrail stages."""
    answer: str = Field(default="", description="Generated answer text.")
    grounded: bool = Field(default=False, description="Whether the answer is grounded in retrieved passages.")
    confidence: float = Field(default=0.0, description="Confidence score between 0.0 and 1.0.")
    citations: list[str] = Field(default_factory=list, description="Source URLs/labels supporting the answer.")
    refusal: bool = Field(default=False, description="Whether the query was refused.")
    refusal_reason: str = Field(default="", description="Reason for refusal (empty if not refused).")
    latency_ms: float = Field(default=0.0, description="Wall-clock time for generation in milliseconds.")


class PipelineResponse(BaseModel):
    """
    Complete response from a single RAG pipeline run.

    Composes stage-level results and provides `.to_api_dict()` to
    serialize into the exact JSON shape that server.py / static/app.js
    already consume.
    """
    # Stage results (None when the stage was skipped)
    transcription: Optional[TranscriptionResult] = None
    retrieval: Optional[RetrievalResult] = None
    generation: Optional[GenerationResult] = None

    # Top-level fields (populated from stage results for convenience)
    query: str = ""
    answer: str = ""
    grounded: bool = False
    confidence: float = 0.0
    refusal: bool = False
    refusal_reason: str = ""
    citations: list[str] = Field(default_factory=list)
    passages: list[dict] = Field(default_factory=list)

    # Latency breakdown (mirrors the dict shape the frontend reads)
    latencies: dict = Field(default_factory=lambda: {
        "stt": 0.0,
        "retrieval": 0.0,
        "generation": 0.0,
        "guardrails": 0.0,
        "total": 0.0,
    })
    total_latency_ms: float = 0.0

    # Error field (non-empty only when something went wrong)
    error: Optional[str] = None

    def to_api_dict(self) -> dict:
        """
        Serialize to the JSON shape expected by the frontend (static/app.js).

        Frontend reads:
            data.query, data.answer, data.grounded, data.confidence,
            data.refusal, data.refusal_reason, data.citations,
            data.passages[].{text, score, dense_score, sparse_score, url},
            data.latencies.{stt, retrieval, generation, guardrails, total}
        """
        d: dict = {
            "query": self.query,
            "answer": self.answer,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "refusal": self.refusal,
            "refusal_reason": self.refusal_reason,
            "citations": self.citations,
            "passages": self.passages,
            "latencies": self.latencies,
        }
        if self.error is not None:
            d["error"] = self.error
        return d
