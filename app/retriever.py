import logging
import numpy as np
from app.indexer import indexer

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(dense_scores: np.ndarray, sparse_scores: np.ndarray, k_rrf: int = 60) -> np.ndarray:
    """
    Applies Reciprocal Rank Fusion (RRF) to fuse rankings from dense and sparse search.
    RRF score = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
    """
    n = len(dense_scores)
    
    # argsort returns indices that would sort the array.
    # To get ranks, we do argsort twice.
    # For descending order, we sort negative scores.
    dense_ranks = np.argsort(np.argsort(-dense_scores))
    sparse_ranks = np.argsort(np.argsort(-sparse_scores))
    
    # Calculate RRF scores
    rrf_scores = 1.0 / (k_rrf + dense_ranks) + 1.0 / (k_rrf + sparse_ranks)
    return rrf_scores

def retrieve_passages(query: str, strategy: str = "semantic", top_k: int = 3, hybrid_weight: float = 0.7) -> list[dict]:
    """
    Performs hybrid retrieval (Dense + Sparse) on the selected strategy index.
    Fuses rankings with RRF and returns the top_k retrieved items.
    Supports parent lookup for parent-child retrieval.
    """
    logger.info(f"Retrieving passages for query: '{query}' using strategy '{strategy}'")
    
    # Load index data
    index_data = indexer.get_index(strategy)
    dense_matrix = index_data["dense"]
    chunks = index_data["chunks"]
    metadata = index_data["metadata"]
    tfidf = index_data["tfidf"]
    tfidf_matrix = index_data["tfidf_matrix"]
    
    if len(chunks) == 0:
        return []

    # 1. Dense Retrieval Score (Cosine Similarity)
    #
    # Always embed the query with the same real model used to embed the
    # documents in indexer.py. A random-vector fallback was used here
    # previously to skip model overhead when no LLM API key was set — but
    # the embedding model is a local sentence-transformers model with no
    # API key requirement at all, so that condition never had anything to
    # do with whether real embeddings were available. The practical effect
    # was that dense retrieval returned meaningless (random) similarity
    # scores by default, leaving only the sparse/TF-IDF half doing real
    # work. The model is lazy-loaded and cached (see get_embedding_model),
    # so encoding one short query string here costs milliseconds once
    # warm — not a meaningful hit to the 200ms budget.
    model = indexer.get_embedding_model()
    query_emb = model.encode(query, show_progress_bar=False)
    query_emb = np.array(query_emb, dtype=np.float32)
    
    # Norms for division
    q_norm = np.linalg.norm(query_emb)
    doc_norms = np.linalg.norm(dense_matrix, axis=1)
    doc_norms[doc_norms == 0] = 1e-10
    
    if q_norm > 0:
        dense_scores = np.dot(dense_matrix, query_emb) / (doc_norms * q_norm)
    else:
        dense_scores = np.zeros(len(chunks))
        
    # 2. Sparse Retrieval Score (TF-IDF Similarity)
    query_sparse = tfidf.transform([query])
    sparse_scores = (tfidf_matrix * query_sparse.T).toarray().ravel()
    
    # 3. Rank Fusion (RRF)
    fused_scores = reciprocal_rank_fusion(dense_scores, sparse_scores)
    
    # Get top items based on fused score
    top_indices = np.argsort(-fused_scores)
    
    results = []
    seen_parents = set()
    
    for idx in top_indices:
        if len(results) >= top_k:
            break
            
        chunk_text = chunks[idx]
        meta = metadata[idx]
        
        # Check if parent-child strategy: perform parent lookup
        if strategy == "parent_child" and "parent_text" in meta:
            parent_text = meta["parent_text"]
            parent_id = meta.get("query_id", "")
            
            # Avoid duplicate parents in final context
            if parent_text in seen_parents:
                continue
                
            seen_parents.add(parent_text)
            results.append({
                "text": parent_text,
                "score": float(fused_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "sparse_score": float(sparse_scores[idx]),
                "url": meta.get("url", ""),
                "language": meta.get("language", "hi"),
                "chunk_type": "parent"
            })
        else:
            results.append({
                "text": chunk_text,
                "score": float(fused_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "sparse_score": float(sparse_scores[idx]),
                "url": meta.get("url", ""),
                "language": meta.get("language", "hi"),
                "chunk_type": meta.get("chunk_type", "standard")
            })
            
    logger.info(f"Retrieved {len(results)} items successfully.")
    return results