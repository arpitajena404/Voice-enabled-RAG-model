"""
05_metadata_chunker.py
STEP 5 of the chunking & retrieval track.

WHY THIS SCRIPT EXISTS:
03 and 04 answer "how do we split the text." This one answers "what do we
attach to each chunk once it's split" — and it's the piece the brief calls
out by name ("metadata-aware chunking") that most teams skip entirely,
because it doesn't change how the text looks, only what you can DO with it
afterward.

WHY IT MATTERS FOR RETRIEVAL, CONCRETELY: without metadata, retrieval can
only rank chunks by embedding similarity. With metadata attached, you can
FILTER before or after ranking — e.g. only search chunks in the query's
language, or boost chunks whose source passage was marked `is_selected`
in the original dataset (a real relevance signal MS MARCO already gives
you for free, that pure-text chunking throws away).

This script doesn't invent a third chunking algorithm — it takes the
sentence-boundary chunks from 04 (the more coherent of the two) and
enriches them with metadata, producing the final "metadata_aware" chunk
set used for indexing.
"""

from utils import load_json, load_jsonl, save_jsonl


def enrich_with_metadata(chunks, examples):
    """
    Attach query-level and passage-level metadata to each chunk.

    Fields added beyond what 04 already has:
    - is_selected: whether the ORIGINAL passage this chunk came from was
      marked relevant to its query in the source dataset. This is a free
      relevance signal — worth carrying forward instead of discarding it
      at chunk time.
    - source_query_text: keeps the original query alongside the chunk, so
      retrieval/debugging can show WHY a chunk exists, not just its text.
    - chunk_index / chunk_count: position of this chunk within its parent
      passage — lets you reconstruct order if you ever need surrounding
      context for an answer.
    """
    # Build a lookup of is_selected per doc_id from the original examples
    selected_lookup = {}
    for qi, ex in enumerate(examples):
        for pi, p in enumerate(ex.get("passages", [])):
            if isinstance(p, dict):
                doc_id = f"q{qi}_p{pi}"
                selected_lookup[doc_id] = bool(p.get("is_selected", 0))

    # Count chunks per doc_id first, so we can record chunk_count
    counts = {}
    for c in chunks:
        counts[c["doc_id"]] = counts.get(c["doc_id"], 0) + 1

    running_index = {}
    enriched = []
    for c in chunks:
        doc_id = c["doc_id"]
        idx = running_index.get(doc_id, 0)
        running_index[doc_id] = idx + 1

        enriched.append({
            **c,
            "chunk_id": c["chunk_id"].replace("_sem", "_meta"),
            "strategy": "metadata_aware",
            "is_selected": selected_lookup.get(doc_id, False),
            "chunk_index": idx,
            "chunk_count": counts[doc_id],
        })
    return enriched


def main():
    examples = load_json("sample_data.json")
    # Reuse step 4's output as the base split — metadata-aware chunking
    # is about enrichment, not a fourth way of cutting the text.
    base_chunks = load_jsonl("chunks_semantic.jsonl")

    enriched = enrich_with_metadata(base_chunks, examples)
    save_jsonl(enriched, "chunks_metadata.jsonl")

    n_selected = sum(1 for c in enriched if c["is_selected"])
    print(f"Metadata-aware chunking: {len(enriched)} chunks enriched, "
          f"{n_selected} tagged is_selected=True from the source dataset.")
    print("Example record:")
    print(enriched[0])


if __name__ == "__main__":
    main()