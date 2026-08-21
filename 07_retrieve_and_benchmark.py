"""
07_retrieve_and_benchmark.py
STEP 7 of the chunking & retrieval track.

WHY THIS SCRIPT EXISTS:
Two jobs, both required by the brief:

1. retrieve() is the actual function Member 3 (generation) calls, and the
   shape it returns MUST match the interface contract you agreed with the
   team on Day 1: query string in, ranked chunks + metadata out. Keeping
   that contract stable here is what lets integration on Day 3 be a plug-in
   instead of a rewrite.

2. benchmark() produces the P50/P70/P100 latency numbers the brief
   explicitly requires, measured across a REASONABLE NUMBER of queries —
   not one best-case run. It also benchmarks all three chunking strategies
   side by side, which is the hard evidence backing whichever strategy you
   claim is "best" in your writeup.
"""

import time
import statistics
from utils import load_jsonl, load_json, save_json

STRATEGIES = ["fixed_size", "sentence_boundary", "metadata_aware"]


def retrieve(query, strategy, model, index, id_map, top_k=5):
    """
    Given a query string, return the top_k ranked chunks for one strategy.

    Return shape matches the team's Day 1 interface contract:
      {
        "chunks": [{"text": ..., "source": ..., "score": ...}, ...],
        "latency_ms": <float>
      }
    Member 3's generation code should be built against exactly this shape.
    """
    import numpy as np

    t0 = time.time()
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, indices = index.search(q_emb, top_k)
    elapsed_ms = (time.time() - t0) * 1000

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(id_map):
            continue
        chunk = id_map[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["doc_id"],
            "score": float(score),
        })

    return {"chunks": results, "latency_ms": round(elapsed_ms, 2)}


def percentile(values, p):
    values = sorted(values)
    idx = min(int(len(values) * p / 100), len(values) - 1)
    return values[idx]


def benchmark(queries, strategy, model, index, id_map, top_k=5):
    """
    Run every query in `queries` through retrieve() for one strategy and
    report P50/P70/P100 latency — the exact numbers the brief asks for.

    WHY MULTIPLE QUERIES, NOT ONE: a single query can hit a lucky cache-warm
    run. P50/P70/P100 across a real batch is what the brief explicitly
    asks for, and it's also just honest — one fast run doesn't tell you
    what a real user will experience.
    """
    latencies = []
    for q in queries:
        result = retrieve(q, strategy, model, index, id_map, top_k=top_k)
        latencies.append(result["latency_ms"])

    return {
        "strategy": strategy,
        "n_queries": len(queries),
        "p50_ms": percentile(latencies, 50),
        "p70_ms": percentile(latencies, 70),
        "p100_ms": percentile(latencies, 100),
        "mean_ms": round(statistics.mean(latencies), 2),
    }


def main():
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    examples = load_json("sample_data.json")
    test_queries = [ex["query"] for ex in examples][:30]  # a real batch, not 1-2

    all_results = []
    for strategy in STRATEGIES:
        index = faiss.read_index(f"index_{strategy}.faiss")
        id_map = load_json(f"idmap_{strategy}.json")

        report = benchmark(test_queries, strategy, model, index, id_map)
        all_results.append(report)
        print(f"[{strategy}] P50={report['p50_ms']}ms  "
              f"P70={report['p70_ms']}ms  P100={report['p100_ms']}ms  "
              f"(n={report['n_queries']})")

    save_json(all_results, "latency_benchmark.json")
    print("\nSaved latency_benchmark.json — this is your evidence for both "
          "the required latency analytics AND the chunking-strategy "
          "comparison writeup.")


if __name__ == "__main__":
    main()