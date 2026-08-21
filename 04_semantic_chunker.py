"""
04_semantic_chunker.py
STEP 4 of the chunking & retrieval track.

WHY THIS SCRIPT EXISTS:
The baseline (03) cuts text at arbitrary word-count boundaries, which can
split a sentence in half mid-thought. This script never breaks a sentence
apart — it groups whole sentences together up to a size budget, so every
chunk is a coherent unit of meaning. This is the strategy most teams stop
at when they say "semantic chunking," so it's a solid middle strategy —
but see the second function below for something stronger.

TWO FUNCTIONS ARE PROVIDED:

1. sentence_boundary_chunk() — groups sentences up to a word budget. Fast,
   no ML model needed, works today with zero dependencies.

2. similarity_based_chunk() — this is the "real" semantic chunking judges
   are likely expecting when the brief says "semantic vs fixed-size":
   it embeds each sentence and starts a NEW chunk when consecutive
   sentences become topically dissimilar (cosine similarity drops below a
   threshold), instead of using a fixed size at all. This is the strategy
   worth demoing on camera — showing a topic shift triggering a chunk
   break is a much stronger visual than "we split every 40 words."

   It needs sentence-transformers, which is heavier — run it after 03/04's
   simpler version is already working, not as your first attempt today.
"""

import re
from utils import load_json, flatten_passages, save_jsonl

MAX_CHUNK_WORDS = 40


def split_sentences(text):
    """Lightweight sentence splitter — no spacy/nltk dependency needed."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def sentence_boundary_chunk(text, max_words=MAX_CHUNK_WORDS):
    """
    Group whole sentences into chunks, never splitting a sentence in half.

    WHY THIS OVER FIXED-SIZE: a fixed-size chunk boundary is blind to
    meaning — it can land in the middle of a sentence, producing a chunk
    that ends on half a thought. Grouping by sentence guarantees every
    chunk is self-contained, which matters directly for retrieval quality:
    a chunk that's a complete thought is more likely to actually answer
    the query it's retrieved for.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [text]

    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_len = len(sent.split())
        if current and current_len + sent_len > max_words:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def similarity_based_chunk(text, model=None, threshold=0.55):
    """
    Break into a new chunk whenever consecutive sentences stop being
    topically related, instead of using any fixed size at all.

    WHY THIS IS THE STRONGER STRATEGY: fixed size and sentence-boundary
    chunking both ask "how much text fits in a chunk?" This asks "where
    does the TOPIC actually change?" — which is the question retrieval
    quality actually depends on. A chunk boundary placed at a genuine topic
    shift means each chunk is coherent AND self-contained, not just
    shorter than some word limit.

    Requires a sentence-transformers model to be passed in (loaded once in
    06_embed_index.py and reused here, so we're not reloading the model
    per passage).
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1 or model is None:
        return sentences or [text]

    import numpy as np
    embeddings = model.encode(sentences)

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]) /
                     (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-8))
        if sim < threshold:
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks


def main():
    examples = load_json("sample_data.json")
    passages = flatten_passages(examples)

    chunk_records = []
    for p in passages:
        pieces = sentence_boundary_chunk(p["text"])
        for ci, piece in enumerate(pieces):
            chunk_records.append({
                "chunk_id": f"{p['doc_id']}_sem{ci}",
                "doc_id": p["doc_id"],
                "query_id": p["query_id"],
                "lang": p["lang"],
                "text": piece,
                "strategy": "sentence_boundary",
            })

    save_jsonl(chunk_records, "chunks_semantic.jsonl")
    print(f"Sentence-boundary chunking: {len(passages)} passages "
          f"-> {len(chunk_records)} chunks")
    print("\nNote: similarity_based_chunk() is available in this file for "
          "the embedding-similarity variant — call it from 06_embed_index.py "
          "once your embedding model is loaded, since it needs the model "
          "passed in. That's the version worth showing on camera as the "
          "'real' semantic strategy.")


if __name__ == "__main__":
    main()