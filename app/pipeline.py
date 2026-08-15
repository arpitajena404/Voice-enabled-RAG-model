import time
import logging
from app.stt import transcribe_audio
from app.retriever import retrieve_passages
from app.generator import generate_answer, RAGResponse
from app.guardrails import check_input_safety, check_grounding

logger = logging.getLogger(__name__)

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
    ) -> dict:
        """
        Executes the end-to-end RAG pipeline, measuring latency for each stage.
        """
        latencies = {
            "stt": 0.0,
            "retrieval": 0.0,
            "generation": 0.0,
            "guardrails": 0.0,
            "total": 0.0
        }
        
        t_start = time.time()
        
        # 1. Speech-to-Text (STT) if audio is provided
        query = query_text
        if audio_bytes:
            t0 = time.time()
            try:
                query = await transcribe_audio(audio_bytes, audio_filename, language_code=language)
                latencies["stt"] = float((time.time() - t0) * 1000)
                logger.info(f"STT Completed in {latencies['stt']:.2f}ms. Transcribed: '{query}'")
            except Exception as e:
                logger.exception("STT Transcription failed")
                return {
                    "error": f"Speech transcription failed: {str(e)}",
                    "query": "",
                    "answer": "मुझे खेद है, मैं आपकी आवाज़ को समझ नहीं पाया। कृपया पुनः प्रयास करें।",
                    "grounded": False,
                    "refusal": True,
                    "refusal_reason": "STT transcription error",
                    "latencies": latencies
                }
                
        if not query or not query.strip():
            return {
                "error": "Query is empty",
                "query": "",
                "answer": "कृपया एक प्रश्न पूछें।",
                "grounded": False,
                "refusal": True,
                "refusal_reason": "Empty query",
                "latencies": latencies
            }

        # 2. Input Guardrails
        t_g0 = time.time()
        is_safe, safety_reason = check_input_safety(query)
        if not is_safe:
            latencies["guardrails"] += float((time.time() - t_g0) * 1000)
            latencies["total"] = float((time.time() - t_start) * 1000)
            return {
                "query": query,
                "answer": "मुझे खेद है, लेकिन मैं इस प्रकार के प्रश्नों का उत्तर नहीं दे सकता।",
                "grounded": False,
                "confidence": 0.0,
                "refusal": True,
                "refusal_reason": f"Input safety block: {safety_reason}",
                "citations": [],
                "passages": [],
                "latencies": latencies
            }

        # 3. Retrieval
        t_r0 = time.time()
        try:
            passages = retrieve_passages(query, strategy=strategy, top_k=top_k)
            latencies["retrieval"] = float((time.time() - t_r0) * 1000)
            logger.info(f"Retrieval completed in {latencies['retrieval']:.2f}ms. Retrieved {len(passages)} passages.")
        except Exception as e:
            logger.exception("Retrieval failed")
            latencies["guardrails"] += float((time.time() - t_g0) * 1000)
            return {
                "error": f"Retrieval failed: {str(e)}",
                "query": query,
                "answer": "रिफ्रेश करते समय कुछ गलत हुआ। कृपया थोड़ी देर बाद पुनः प्रयास करें।",
                "grounded": False,
                "refusal": True,
                "refusal_reason": "Retrieval system error",
                "latencies": latencies
            }

        # 4. LLM Answer Generation
        t_gen0 = time.time()
        try:
            # Generate the response
            llm_response = await generate_answer(
                query=query,
                retrieved_passages=passages,
                language=language,
                provider=provider
            )
            latencies["generation"] = float((time.time() - t_gen0) * 1000)
            logger.info(f"LLM Generation completed in {latencies['generation']:.2f}ms.")
        except Exception as e:
            logger.exception("LLM generation failed")
            latencies["guardrails"] += float((time.time() - t_g0) * 1000)
            return {
                "error": f"LLM generation failed: {str(e)}",
                "query": query,
                "answer": "उत्तर उत्पन्न करने में समस्या हुई। कृपया पुनः प्रयास करें।",
                "grounded": False,
                "refusal": True,
                "refusal_reason": "LLM generation error",
                "latencies": latencies
            }

        # 5. Output Grounding Guardrail
        t_g1 = time.time()
        is_grounded = True
        overlap_score = 1.0
        
        # Only verify grounding if not already refused by the LLM itself
        if not llm_response.refusal and passages:
            is_grounded, overlap_score = check_grounding(llm_response.answer, passages)
            
        latencies["guardrails"] += float((time.time() - t_g0 + time.time() - t_g1) * 1000)
        latencies["total"] = float((time.time() - t_start) * 1000)
        
        # Override grounded flag if guardrail fails
        grounded_final = llm_response.grounded and is_grounded
        refusal_final = llm_response.refusal or (not grounded_final and not llm_response.refusal and overlap_score < 0.15)
        
        refusal_reason_final = llm_response.refusal_reason
        answer_final = llm_response.answer
        if refusal_final and not llm_response.refusal:
            refusal_reason_final = "Output grounding guardrail failure"
            answer_final = "मुझे खेद है, लेकिन प्रदान किए गए संदर्भ में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।"

        return {
            "query": query,
            "answer": answer_final,
            "grounded": grounded_final,
            "confidence": min(float(llm_response.confidence), float(overlap_score)),
            "refusal": refusal_final,
            "refusal_reason": refusal_reason_final,
            "citations": llm_response.citations,
            "passages": passages,
            "latencies": latencies
        }

# Instantiate global pipeline
pipeline = RAGPipeline()
