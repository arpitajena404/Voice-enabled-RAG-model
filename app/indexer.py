import pickle
import logging
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from app.config import config
from app.chunker import chunk_naive, chunk_semantic, chunk_parent_child

logger = logging.getLogger(__name__)

class IndicIndexer:
    def __init__(self):
        self.embedding_model = None
        self.indices = {}

    def get_embedding_model(self):
        """Lazy load the sentence transformer model."""
        if self.embedding_model is None:
            logger.info(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        return self.embedding_model

    def build_and_save_index(self, dataset_examples: list[dict], strategy: str = "semantic") -> dict:
        """
        Processes dataset examples, chunks them according to strategy, 
        generates dense/sparse embeddings, and saves the index to config.INDEX_DIR.
        """
        logger.info(f"Building {strategy} index for {len(dataset_examples)} examples...")
        model = self.get_embedding_model()
        
        chunks = []
        metadata = []
        
        for idx, item in enumerate(dataset_examples):
            lang = item.get("language", "hi")
            query_id = item.get("query_id", "")
            
            passages_data = item.get("passages", {})
            passage_texts = passages_data.get("passage_text", [])
            urls = passages_data.get("url", [])
            is_selected_list = passages_data.get("is_selected", [])
            
            # Fallback to original passages if translation is missing
            if not passage_texts:
                passages_data = item.get("original_passages", {})
                passage_texts = passages_data.get("passage_text", [])
                urls = passages_data.get("url", [])
                is_selected_list = passages_data.get("is_selected", [])

            # Chunk each passage independently rather than joining them into
            # one blob first. Joining before chunking lets a single chunk
            # span content from two unrelated passages, which muddies both
            # retrieval precision and the parent-child hierarchy. Chunking
            # per-passage also lets us carry `is_selected` (the dataset's
            # own relevance label) through to each chunk instead of losing
            # it at the join step.
            for p_idx, passage_text in enumerate(passage_texts):
                if not passage_text.strip():
                    continue

                url = urls[p_idx] if p_idx < len(urls) else ""
                is_selected = bool(is_selected_list[p_idx]) if p_idx < len(is_selected_list) else False

                if strategy == "naive":
                    strategy_chunks = chunk_naive(passage_text)
                    for chunk in strategy_chunks:
                        chunks.append(chunk)
                        metadata.append({
                            "language": lang,
                            "query_id": query_id,
                            "url": url,
                            "chunk_type": "naive",
                            "is_selected": is_selected,
                        })
                elif strategy == "semantic":
                    strategy_chunks = chunk_semantic(passage_text, model)
                    for chunk in strategy_chunks:
                        chunks.append(chunk)
                        metadata.append({
                            "language": lang,
                            "query_id": query_id,
                            "url": url,
                            "chunk_type": "semantic",
                            "is_selected": is_selected,
                        })
                elif strategy == "parent_child":
                    # Parent is this single passage, not the whole
                    # concatenated set — keeps parent context focused
                    # instead of pulling in unrelated passages.
                    strategy_chunks = chunk_parent_child(passage_text, f"{idx}_{p_idx}")
                    for child in strategy_chunks:
                        chunks.append(child["child_text"])
                        metadata.append({
                            "language": lang,
                            "query_id": query_id,
                            "url": url,
                            "parent_text": child["parent_text"],
                            "chunk_type": "child",
                            "is_selected": is_selected,
                        })
                else:
                    raise ValueError(f"Unknown chunking strategy: {strategy}")

        if not chunks:
            logger.warning(f"No chunks created for strategy {strategy}")
            return {"num_chunks": 0}

        # 1. Compute Dense Embeddings
        logger.info(f"Encoding {len(chunks)} chunks using SentenceTransformer...")
        dense_embeddings = model.encode(chunks, show_progress_bar=True)
        dense_embeddings = np.array(dense_embeddings, dtype=np.float32)
        
        # 2. Fit Sparse TF-IDF Vectorizer
        logger.info(f"Fitting sparse TF-IDF vectorizer on {len(chunks)} chunks...")
        # We use char_wb analyzer to handle multilingual/morphological matching cleanly
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        sparse_matrix = tfidf.fit_transform(chunks)
        
        # 3. Save Files
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        dense_path = config.INDEX_DIR / f"{strategy}_dense.npy"
        meta_path = config.INDEX_DIR / f"{strategy}_meta.pkl"
        
        np.save(str(dense_path), dense_embeddings)
        
        index_meta = {
            "chunks": chunks,
            "metadata": metadata,
            "tfidf_vectorizer": tfidf,
            "tfidf_matrix": sparse_matrix
        }
        
        with open(meta_path, "wb") as f:
            pickle.dump(index_meta, f)
            
        logger.info(f"Saved {strategy} index. Dense path: {dense_path}, Meta path: {meta_path}")
        
        # Cache in memory
        self.indices[strategy] = {
            "dense": dense_embeddings,
            "chunks": chunks,
            "metadata": metadata,
            "tfidf": tfidf,
            "tfidf_matrix": sparse_matrix
        }
        
        return {"num_chunks": len(chunks)}

    def load_index(self, strategy: str = "semantic") -> bool:
        """
        Loads the specified strategy index from disk.
        Returns True if successful, False otherwise.
        """
        dense_path = config.INDEX_DIR / f"{strategy}_dense.npy"
        meta_path = config.INDEX_DIR / f"{strategy}_meta.pkl"
        
        if not dense_path.exists() or not meta_path.exists():
            logger.warning(f"Index files for {strategy} strategy not found.")
            return False
            
        try:
            dense_embeddings = np.load(str(dense_path))
            with open(meta_path, "rb") as f:
                index_meta = pickle.load(f)
                
            self.indices[strategy] = {
                "dense": dense_embeddings,
                "chunks": index_meta["chunks"],
                "metadata": index_meta["metadata"],
                "tfidf": index_meta["tfidf_vectorizer"],
                "tfidf_matrix": index_meta["tfidf_matrix"]
            }
            logger.info(f"Loaded {strategy} index with {len(index_meta['chunks'])} chunks.")
            return True
        except Exception as e:
            logger.exception(f"Error loading {strategy} index")
            return False

    def get_index(self, strategy: str) -> dict:
        """
        Returns the index dictionary for the strategy. Loads it if not already in memory.
        """
        if strategy not in self.indices:
            success = self.load_index(strategy)
            if not success:
                raise RuntimeError(f"Index for strategy '{strategy}' is not loaded and could not be found on disk. Run seed_database.py first.")
        return self.indices[strategy]
        
# Instantiate global indexer
indexer = IndicIndexer()