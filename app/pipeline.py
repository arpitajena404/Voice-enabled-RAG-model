import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from app.stt import transcribe_audio
from app.retriever import retrieve_passages
from app.generator import generate_answer, RAGResponse, REFUSAL_MESSAGES

from app.guardrails import check_input_safety, check_off_topic, check_grounding
from app.config import config
from app.schemas import (
    TranscriptionResult,
    RetrievalResult,
    GenerationResult,
    PipelineResponse,
)
from app.timing import timed_stage, timed_stage_sync

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry wrapper for the generation call (Gemini / Groq are external APIs)
# ---------------------------------------------------------------------------
def _generation_retry():
    """Build a tenacity retry decorator for generation calls."""
    return retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=config.RETRY_BACKOFF_BASE,
            max=config.RETRY_BACKOFF_MAX,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def _detect_query_language(query_text: str, fallback_lang: str) -> str:
    """Auto-detects language based on character script (Hindi, Bengali, Tamil, Telugu, English)."""
    cleaned = "".join(c for c in query_text if c.isalnum())
    if not cleaned:
        return fallback_lang
    if any('\u0900' <= c <= '\u097F' for c in cleaned):
        return "hi"
    if any('\u0980' <= c <= '\u09FF' for c in cleaned):
        return "bn"
    if any('\u0B80' <= c <= '\u0BFF' for c in cleaned):
        return "ta"
    if any('\u0C00' <= c <= '\u0C7F' for c in cleaned):
        return "te"
    if all(ord(c) < 128 for c in cleaned):
        return "en"
    return fallback_lang


class RAGPipeline:
    async def run(
        self,
        query_text: str = None,
        audio_bytes: bytes = None,
        audio_filename: str = "audio.webm",
        language: str = "hi",
        strategy: str = "semantic",
        provider: str = "gemini",
        top_k: int = 3
    ) -> PipelineResponse:
        """
        Executes the end-to-end RAG pipeline, measuring latency for each stage.
        Returns a structured PipelineResponse.
        """
        latencies = {
            "stt": 0.0,
            "retrieval": 0.0,
            "generation": 0.0,
            "guardrails": 0.0,
            "total": 0.0,
        }

        transcription_result: TranscriptionResult | None = None
        retrieval_result: RetrievalResult | None = None
        generation_result: GenerationResult | None = None

        # ── 1. Speech-to-Text ────────────────────────────────────────
        query = query_text
        async with timed_stage("total") as t_total:

            if audio_bytes:
                async with timed_stage("stt") as t_stt:
                    try:
                        stt_result = await transcribe_audio(
                            audio_bytes, audio_filename, language_code=language
                        )
                        query = stt_result.text
                        transcription_result = stt_result
                    except Exception as e:
                        logger.exception("STT Transcription failed")
                        latencies["stt"] = t_stt.latency_ms

                        return PipelineResponse(
                            error=f"Speech transcription failed: {str(e)}",
                            query="",
                            answer="Speech transcription failed. Please try again.",
                            grounded=False,
                            refusal=True,
                            refusal_reason="STT transcription error",
                            latencies=latencies,
                        )
                latencies["stt"] = t_stt.latency_ms
                if transcription_result:
                    transcription_result.latency_ms = t_stt.latency_ms
                logger.info(f"STT Completed in {latencies['stt']:.2f}ms. Transcribed: '{query}'")

            if not query or not query.strip():
                return PipelineResponse(
                    error="Query is empty",
                    query="",
                    answer="Please ask a question.",
                    grounded=False,
                    refusal=True,
                    refusal_reason="Empty query",
                    latencies=latencies,
                )

            # Auto-detect language script
            effective_lang = _detect_query_language(query, language)

            # ── 2. Input Guardrails ──────────────────────────────────
            with timed_stage_sync("guardrails_input") as t_gi:
                is_safe, safety_reason = check_input_safety(query)
                if is_safe:
                    is_off_topic, off_topic_reason = check_off_topic(query)
                    if is_off_topic:
                        is_safe = False
                        safety_reason = off_topic_reason

            if not is_safe:
                latencies["guardrails"] = t_gi.latency_ms
                latencies["total"] = t_total.latency_ms  # will be filled on exit but set early for the return
                refusal_msg = REFUSAL_MESSAGES.get(effective_lang, REFUSAL_MESSAGES["hi"])
                return PipelineResponse(
                    query=query,
                    answer=refusal_msg,
                    grounded=False,
                    confidence=0.0,
                    refusal=True,
                    refusal_reason=f"Input safety block: {safety_reason}",
                    citations=[],
                    passages=[],
                    latencies=latencies,
                )

            guardrails_ms = t_gi.latency_ms

            # ── 3. Retrieval ─────────────────────────────────────────
            with timed_stage_sync("retrieval") as t_ret:
                try:
                    passages = retrieve_passages(query, strategy=strategy, top_k=top_k)
                except Exception as e:
                    logger.exception("Retrieval failed")
                    latencies["retrieval"] = t_ret.latency_ms
                    latencies["guardrails"] = guardrails_ms
                    return PipelineResponse(
                        error=f"Retrieval failed: {str(e)}",
                        query=query,
                        answer=REFUSAL_MESSAGES.get(effective_lang, REFUSAL_MESSAGES["hi"]),
                        grounded=False,
                        refusal=True,
                        refusal_reason="Retrieval system error",
                        latencies=latencies,
                    )
            latencies["retrieval"] = t_ret.latency_ms
            retrieval_result = RetrievalResult(
                passages=passages,
                strategy=strategy,
                latency_ms=t_ret.latency_ms,
            )
            logger.info(f"Retrieval completed in {latencies['retrieval']:.2f}ms. Retrieved {len(passages)} passages.")

            # ── 4. LLM Answer Generation (with retry) ────────────────
            async with timed_stage("generation") as t_gen:
                try:
                    retrying_generate = _generation_retry()(generate_answer)
                    llm_response: RAGResponse = await retrying_generate(
                        query=query,
                        retrieved_passages=passages,
                        language=effective_lang,
                        provider=provider,
                    )
                except Exception as e:
                    logger.exception("LLM generation failed")
                    latencies["generation"] = t_gen.latency_ms
                    latencies["guardrails"] = guardrails_ms
                    return PipelineResponse(
                        error=f"LLM generation failed: {str(e)}",
                        query=query,
                        answer=REFUSAL_MESSAGES.get(effective_lang, REFUSAL_MESSAGES["hi"]),
                        grounded=False,
                        refusal=True,
                        refusal_reason="LLM generation error",
                        latencies=latencies,
                    )
            latencies["generation"] = t_gen.latency_ms
            logger.info(f"LLM Generation completed in {latencies['generation']:.2f}ms.")


            # ── 5. Output Grounding Guardrail ─────────────────────────
            with timed_stage_sync("guardrails_output") as t_go:
                is_grounded = True
                overlap_score = 1.0
                if not llm_response.refusal and passages:
                    is_grounded, overlap_score = check_grounding(llm_response.answer, passages)

            guardrails_ms += t_go.latency_ms
            latencies["guardrails"] = guardrails_ms

            # Compose final answer with grounding override logic
            grounded_final = llm_response.grounded and is_grounded
            refusal_final = llm_response.refusal or (
                not grounded_final and not llm_response.refusal and overlap_score < 0.15
            )

            refusal_reason_final = llm_response.refusal_reason
            answer_final = llm_response.answer
            if refusal_final and not llm_response.refusal:
                refusal_reason_final = "Output grounding guardrail failure"
                answer_final = REFUSAL_MESSAGES.get(language, REFUSAL_MESSAGES["hi"])


        # t_total context exited → latency is filled
        latencies["total"] = t_total.latency_ms

        generation_result = GenerationResult(
            answer=answer_final,
            grounded=grounded_final,
            confidence=min(float(llm_response.confidence), float(overlap_score)),
            citations=llm_response.citations,
            refusal=refusal_final,
            refusal_reason=refusal_reason_final,
            latency_ms=t_gen.latency_ms,
        )

        return PipelineResponse(
            transcription=transcription_result,
            retrieval=retrieval_result,
            generation=generation_result,
            query=query,
            answer=answer_final,
            grounded=grounded_final,
            confidence=min(float(llm_response.confidence), float(overlap_score)),
            refusal=refusal_final,
            refusal_reason=refusal_reason_final,
            citations=llm_response.citations,
            passages=passages,
            latencies=latencies,
            total_latency_ms=t_total.latency_ms,
        )


# Instantiate global pipeline
pipeline = RAGPipeline()
