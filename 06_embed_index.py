"""
06_embed_index.py
STEP 6 of the chunking & retrieval track.

WHY THIS SCRIPT EXISTS:
This is where the three chunk sets (fixed, semantic, metadata) actually
become searchable. Each strategy gets its OWN FAISS index, kept separate
rather than merged — that's deliberate: it's the only way to later query
"strategy A vs strategy B vs strategy C" independently and produce the
comparison your judging deliverable needs. Merging them now would make
that comparison impossible to reconstruct later.

WHY THIS EMBEDDING MODEL: a multilingual sentence-transformers model
(paraphrase-multilingual-MiniLM-L12-v2) is used because MSMARCO-XI is
Indic-language data — an English-only embedding model would perform badly
on Hindi/Tamil/etc. text. It's also small (~120MB) and fast enough to
embed a few hundred chunks in seconds, which matters on a Day 2 deadline.

Run this AFTER 03, 04, and 05 have produced their chunk .jsonl files.
"""

import time
from utils import load_jsonl, save_json

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

STRATEGY_FILES = {
    "fixed_size": "chunks_fixed.jsonl",
    "sentence_boundary": "chunks_semantic.jsonl",
    "metadata_aware": "chunks_metadata.jsonl",
}


def build_index(chunks, model):
    """
    Embed every chunk's text and build a FAISS index over the vectors.

    WHY FAISS: it's a local, dependency-light vector index — no server to
    stand up, which matters when you have days not weeks. IndexFlatIP
    (inner product on normalized vectors = cosine similarity) is the
    simplest correct choice for a dataset this size; swap in an
    approximate index (IVF/HNSW) only if you see real latency problems
    once you're testing against the full dataset, not before — premature
    optimization here just costs debugging time you don't have.
    """
    import faiss
    import numpy as np

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    from sentence_transformers import SentenceTransformer
    import faiss

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    for strategy, path in STRATEGY_FILES.items():
        chunks = load_jsonl(path)
        if not chunks:
            print(f"Skipping {strategy}: no chunks found in {path}")
            continue

        t0 = time.time()
        index = build_index(chunks, model)
        elapsed_ms = (time.time() - t0) * 1000

        index_path = f"index_{strategy}.faiss"
        faiss.write_index(index, index_path)

        # Save the id mapping so 07 can go from FAISS row -> chunk metadata
        id_map = [c for c in chunks]
        save_json(id_map, f"idmap_{strategy}.json")

        print(f"[{strategy}] indexed {len(chunks)} chunks in "
              f"{elapsed_ms:.1f}ms -> {index_path}")


if __name__ == "__main__":
    main()