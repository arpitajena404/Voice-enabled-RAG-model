import os
import sys
import json
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

async def run_benchmark(strategy: str = "semantic", provider: str = "gemini", limit: int = 30):
    """
    Runs an offline benchmark over seeded queries and calculates P50, P70, and P100 latencies,
    along with groundedness and refusal ratios.
    """
    raw_path = config.DATA_DIR / "raw_examples.pkl"
    if not raw_path.exists():
        logger.error(f"Seeded data not found at {raw_path}. Run seed_database.py first!")
        sys.exit(1)
        
    with open(raw_path, "rb") as f:
        examples = pickle.load(f)
        
    # Pick a subset of queries for benchmarking
    test_examples = examples[:limit]
    logger.info(f"Loaded {len(test_examples)} queries for benchmark (Strategy: {strategy}, Provider: {provider})")
    
    stt_times = []
    retrieval_times = []
    generation_times = []
    guardrails_times = []
    total_times = []
    
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
        
        # We simulate the speech-to-text latency (e.g., standard API roundtrip ~120ms if offline, or actual call if online)
        # To avoid making actual STT calls for all queries (which would consume a lot of API credits and rate limits),
        # we run RAG on text, but simulate a realistic STT latency of 100-150ms if we want to show full voice pipeline stats,
        # or we just report Text-RAG stats. Let's record the actual RAG pipeline latencies and add a constant/random STT factor
        # if audio_bytes were passed. We will run it as standard text query here.
        
        t0 = time.time()
        try:
            res = await pipeline.run(
                query_text=query,
                language=lang,
                strategy=strategy,
                provider=provider
            )
            
            # Record execution times
            lats = res.get("latencies", {})
            stt_times.append(lats.get("stt", 0.0))
            retrieval_times.append(lats.get("retrieval", 0.0))
            generation_times.append(lats.get("generation", 0.0))
            guardrails_times.append(lats.get("guardrails", 0.0))
            total_times.append(lats.get("total", 0.0))
            
            if res.get("grounded"):
                grounded_count += 1
            if res.get("refusal"):
                refusal_count += 1
            success_count += 1
            
            results.append({
                "query": query,
                "language": lang,
                "answer": res.get("answer"),
                "grounded": res.get("grounded"),
                "refusal": res.get("refusal"),
                "latencies": lats
            })
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            
        # Small delay to respect rate limits if using online LLMs
        if config.GEMINI_API_KEY or config.GROQ_API_KEY:
            await asyncio.sleep(0.5)

    if success_count == 0:
        logger.error("No queries completed successfully.")
        return
        
    # Calculate Latency Percentiles (P50, P70, P100)
    metrics = {
        "Retrieval": retrieval_times,
        "Generation": generation_times,
        "Guardrails": guardrails_times,
        "Total (Text RAG)": total_times
    }
    
    report_rows = []
    for name, times in metrics.items():
        if not times:
            continue
        p50 = np.percentile(times, 50)
        p70 = np.percentile(times, 70)
        p100 = np.percentile(times, 100)
        mean_val = np.mean(times)
        report_rows.append([name, f"{p50:.2f}ms", f"{p70:.2f}ms", f"{p100:.2f}ms", f"{mean_val:.2f}ms"])
        
    # Build tables
    headers = ["Component", "P50 Latency", "P70 Latency", "P100 (Max) Latency", "Average Latency"]
    table_str = tabulate(report_rows, headers=headers, tablefmt="grid")
    
    # Calculate ratios
    grounded_ratio = (grounded_count / success_count) * 100
    refusal_ratio = (refusal_count / success_count) * 100
    
    logger.info("\n" + "="*50 + "\nBENCHMARK REPORT\n" + "="*50)
    logger.info(f"\nCompleted Queries: {success_count}/{len(test_examples)}")
    logger.info(f"Groundedness Rate: {grounded_ratio:.2f}%")
    logger.info(f"Refusal Rate (Safety/Off-topic): {refusal_ratio:.2f}%")
    logger.info("\nLatency Statistics:\n" + table_str)
    
    # Save report
    report_data = {
        "timestamp": time.time(),
        "strategy": strategy,
        "provider": provider,
        "summary": {
            "total_queries": len(test_examples),
            "successful_queries": success_count,
            "groundedness_rate": grounded_ratio,
            "refusal_rate": refusal_ratio
        },
        "latencies": {
            name: {
                "p50": float(np.percentile(times, 50)),
                "p70": float(np.percentile(times, 70)),
                "p100": float(np.percentile(times, 100)),
                "mean": float(np.mean(times))
            } for name, times in metrics.items() if times
        },
        "queries": results
    }
    
    report_file = config.DATA_DIR / f"latency_report_{strategy}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"\nReport written to {report_file}")

if __name__ == "__main__":
    strategy = "semantic"
    provider = "gemini"
    limit = 30
    
    if len(sys.argv) > 1:
        strategy = sys.argv[1]
    if len(sys.argv) > 2:
        provider = sys.argv[2]
    if len(sys.argv) > 3:
        try:
            limit = int(sys.argv[3])
        except ValueError:
            pass
            
    asyncio.run(run_benchmark(strategy, provider, limit))
