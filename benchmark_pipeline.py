"""
Voice-Enabled RAG Latency Benchmark
====================================

Runs a batch of test queries through the FULL pipeline (including real STT
when SARVAM_API_KEY is set) and reports per-stage + end-to-end P50/P70/P100
latency analytics with a pass/fail flag against the 200ms target.

Usage:
    python benchmark_pipeline.py [strategy] [provider] [limit] [--mode full|text-only]

    strategy   : naive | semantic | parent_child  (default: semantic)
    provider   : gemini | groq                     (default: gemini)
    limit      : number of queries to benchmark    (default: 30)
    --mode     : full      → real STT via audio files/synthetic WAV
                 text-only → skip STT, text-RAG only (legacy)

Examples:
    python benchmark_pipeline.py semantic gemini 30 --mode full
    python benchmark_pipeline.py naive groq 50 --mode text-only
"""

import os
import sys
import io
import json
import wave
import math
import struct
import time
import pickle
import logging
import asyncio
import numpy as np
from tabulate import tabulate
from app.config import config
from app.pipeline import pipeline

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("benchmark")

LATENCY_TARGET_MS = 200.0


# ---------------------------------------------------------------------------
# Synthetic WAV generator (avoids needing real audio files in the repo)
# ---------------------------------------------------------------------------

def generate_synthetic_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """
    Generate a short in-memory WAV file containing a 440 Hz sine tone.
    Used to exercise the real STT API path in benchmark mode=full.
    """
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            sample = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    strategy: str = "semantic",
    provider: str = "gemini",
    limit: int = 30,
    mode: str = "full",
):
    """
    Runs an offline benchmark over seeded queries and calculates P50, P70,
    and P100 latencies per stage and end-to-end, plus pass/fail against the
    200ms target.
    """
    raw_path = config.DATA_DIR / "raw_examples.pkl"
    if not raw_path.exists():
        logger.error(f"Seeded data not found at {raw_path}. Run seed_database.py first!")
        sys.exit(1)

    with open(raw_path, "rb") as f:
        examples = pickle.load(f)

    # Pick a subset of queries for benchmarking
    test_examples = examples[:limit]
    logger.info(
        f"Loaded {len(test_examples)} queries for benchmark "
        f"(Strategy: {strategy}, Provider: {provider}, Mode: {mode})"
    )

    # In full mode, generate a single synthetic WAV to feed through STT
    use_audio = mode == "full"
    synth_wav: bytes | None = None
    if use_audio:
        if not config.SARVAM_API_KEY:
            logger.warning(
                "SARVAM_API_KEY not set — full mode will use simulated STT. "
                "Set the key for real API latency measurement."
            )
        synth_wav = generate_synthetic_wav(duration_s=1.0)
        logger.info(f"Generated synthetic WAV ({len(synth_wav)} bytes) for STT benchmarking.")

    stt_times: list[float] = []
    retrieval_times: list[float] = []
    generation_times: list[float] = []
    guardrails_times: list[float] = []
    total_times: list[float] = []

    grounded_count = 0
    refusal_count = 0
    success_count = 0

    results = []

    for idx, ex in enumerate(test_examples):
        query = ex.get("query", "")
        lang = ex.get("language", "hi")

        if not query:
            continue

        logger.info(f"[{idx+1}/{len(test_examples)}] Benchmarking: '{query[:40]}...' ({lang})")

        try:
            if use_audio and synth_wav is not None:
                # Full pipeline: audio → STT → retrieval → generation
                res = await pipeline.run(
                    audio_bytes=synth_wav,
                    audio_filename="benchmark_synth.wav",
                    language=lang,
                    strategy=strategy,
                    provider=provider,
                    top_k=3,
                )
            else:
                # Text-only: skip STT stage
                res = await pipeline.run(
                    query_text=query,
                    language=lang,
                    strategy=strategy,
                    provider=provider,
                    top_k=3,
                )

            # PipelineResponse → extract latencies via .to_api_dict()
            res_dict = res.to_api_dict()
            lats = res_dict.get("latencies", {})

            stt_times.append(lats.get("stt", 0.0))
            retrieval_times.append(lats.get("retrieval", 0.0))
            generation_times.append(lats.get("generation", 0.0))
            guardrails_times.append(lats.get("guardrails", 0.0))
            total_times.append(lats.get("total", 0.0))

            if res_dict.get("grounded"):
                grounded_count += 1
            if res_dict.get("refusal"):
                refusal_count += 1
            success_count += 1

            results.append({
                "query": query,
                "language": lang,
                "answer": res_dict.get("answer"),
                "grounded": res_dict.get("grounded"),
                "refusal": res_dict.get("refusal"),
                "latencies": lats,
            })

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")

        # Small delay to respect rate limits if using online LLMs
        if config.GEMINI_API_KEY or config.GROQ_API_KEY:
            await asyncio.sleep(0.5)

    if success_count == 0:
        logger.error("No queries completed successfully.")
        return

    # ------------------------------------------------------------------
    # Calculate Latency Percentiles (P50, P70, P100) per stage + total
    # ------------------------------------------------------------------
    total_label = "Total (End-to-End)" if use_audio else "Total (Text RAG)"

    metrics: dict[str, list[float]] = {}
    if use_audio:
        metrics["STT"] = stt_times
    metrics["Retrieval"] = retrieval_times
    metrics["Generation"] = generation_times
    metrics["Guardrails"] = guardrails_times
    metrics[total_label] = total_times

    report_rows = []
    stage_p70: dict[str, float] = {}
    for name, times in metrics.items():
        if not times:
            continue
        p50 = float(np.percentile(times, 50))
        p70 = float(np.percentile(times, 70))
        p100 = float(np.percentile(times, 100))
        mean_val = float(np.mean(times))
        report_rows.append([name, f"{p50:.2f}ms", f"{p70:.2f}ms", f"{p100:.2f}ms", f"{mean_val:.2f}ms"])
        stage_p70[name] = p70

    # Build display table
    headers = ["Component", "P50 Latency", "P70 Latency", "P100 (Max) Latency", "Average Latency"]
    table_str = tabulate(report_rows, headers=headers, tablefmt="grid")

    # Calculate ratios
    grounded_ratio = (grounded_count / success_count) * 100
    refusal_ratio = (refusal_count / success_count) * 100

    # Pass/fail against 200ms target
    p70_total = stage_p70.get(total_label, 0.0)
    passes_target = p70_total <= LATENCY_TARGET_MS

    # Identify bottleneck stage when failing
    bottleneck = ""
    if not passes_target:
        # Find the stage contributing the most to P70
        stage_only = {k: v for k, v in stage_p70.items() if k != total_label}
        if stage_only:
            bottleneck = max(stage_only, key=stage_only.get)

    logger.info("\n" + "=" * 60 + "\nBENCHMARK REPORT\n" + "=" * 60)
    logger.info(f"\nMode: {mode.upper()}")
    logger.info(f"Completed Queries: {success_count}/{len(test_examples)}")
    logger.info(f"Groundedness Rate: {grounded_ratio:.2f}%")
    logger.info(f"Refusal Rate (Safety/Off-topic): {refusal_ratio:.2f}%")
    logger.info("\nLatency Statistics:\n" + table_str)

    # Pass/fail banner
    if passes_target:
        logger.info(f"\n✅ PASS — P70 end-to-end ({p70_total:.2f}ms) ≤ {LATENCY_TARGET_MS}ms target")
    else:
        logger.warning(
            f"\n❌ FAIL — P70 end-to-end ({p70_total:.2f}ms) > {LATENCY_TARGET_MS}ms target\n"
            f"   Bottleneck stage: {bottleneck} (P70 = {stage_p70.get(bottleneck, 0):.2f}ms)"
        )

    # ------------------------------------------------------------------
    # Save report (backward-compatible superset of existing format)
    # ------------------------------------------------------------------
    report_data = {
        "timestamp": time.time(),
        "strategy": strategy,
        "provider": provider,
        "mode": mode,
        "summary": {
            "total_queries": len(test_examples),
            "successful_queries": success_count,
            "groundedness_rate": grounded_ratio,
            "refusal_rate": refusal_ratio,
        },
        "latency_target_ms": LATENCY_TARGET_MS,
        "pass_200ms": passes_target,
        "bottleneck": bottleneck,
        "latencies": {
            name: {
                "p50": float(np.percentile(times, 50)),
                "p70": float(np.percentile(times, 70)),
                "p100": float(np.percentile(times, 100)),
                "mean": float(np.mean(times)),
            }
            for name, times in metrics.items()
            if times
        },
        "queries": results,
    }

    report_file = config.DATA_DIR / f"latency_report_{strategy}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nReport written to {report_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    strategy = "semantic"
    provider = "gemini"
    limit = 30
    mode = "full"

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if len(args) > 0:
        strategy = args[0]
    if len(args) > 1:
        provider = args[1]
    if len(args) > 2:
        try:
            limit = int(args[2])
        except ValueError:
            pass

    for flag in flags:
        if flag.startswith("--mode"):
            # Support --mode=full or --mode full
            if "=" in flag:
                mode = flag.split("=", 1)[1]
            else:
                idx = sys.argv.index(flag)
                if idx + 1 < len(sys.argv):
                    mode = sys.argv[idx + 1]

    asyncio.run(run_benchmark(strategy, provider, limit, mode))
