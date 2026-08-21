"""
03_baseline_chunker.py
STEP 3 of the chunking & retrieval track.

WHY THIS SCRIPT EXISTS:
This is deliberately the SIMPLEST possible chunker: fixed word count, fixed
overlap, no awareness of sentence or meaning boundaries. It's not the
differentiator the brief asks for — it's the scaffolding. Building this
first gets a chunk -> embed -> retrieve path working end-to-end today, so
tomorrow's smarter strategies (04, 05) have a working pipeline to slot
into instead of you debugging chunking AND retrieval AND indexing all at
once on integration day.

It's also your control group: when you write up "why we chose strategy X,"
you need a naive baseline to compare against, or the comparison has
nothing to be better than.
"""

from utils import load_json, flatten_passages, save_jsonl

CHUNK_SIZE_WORDS = 40   # tune after looking at profile_report.json from step 2
OVERLAP_WORDS = 8       # ~20% overlap — keeps context across chunk boundaries


def fixed_size_chunk(text, chunk_size=CHUNK_SIZE_WORDS, overlap=OVERLAP_WORDS):
    """
    Split text into fixed-size word chunks with overlap.

    WHY OVERLAP AT ALL: without it, a sentence that straddles a chunk
    boundary gets cut in half in BOTH resulting chunks, and neither chunk
    contains the full thought. Overlap trades a bit of duplicated storage
    for fewer broken-context retrievals — a standard, defensible tradeoff
    to state in your writeup.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def main():
    examples = load_json("sample_data.json")
    passages = flatten_passages(examples)

    chunk_records = []
    for p in passages:
        pieces = fixed_size_chunk(p["text"])
        for ci, piece in enumerate(pieces):
            chunk_records.append({
                "chunk_id": f"{p['doc_id']}_fixed{ci}",
                "doc_id": p["doc_id"],
                "query_id": p["query_id"],
                "lang": p["lang"],
                "text": piece,
                "strategy": "fixed_size",
            })

    save_jsonl(chunk_records, "chunks_fixed.jsonl")
    print(f"Baseline fixed-size chunking: {len(passages)} passages "
          f"-> {len(chunk_records)} chunks "
          f"(chunk_size={CHUNK_SIZE_WORDS} words, overlap={OVERLAP_WORDS} words)")


if __name__ == "__main__":
    main()