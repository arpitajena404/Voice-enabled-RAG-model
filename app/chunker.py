import re
import numpy as np

def split_sentences(text: str) -> list[str]:
    """
    Splits text into sentences, supporting both English punctuation (. ! ?)
    and Indic punctuation (e.g. the Devanagari danda '।').
    """
    if not text:
        return []
    # Split on . ! ? or Devanagari danda (\u0964) followed by whitespace
    sentences = re.split(r'(?<=[.!?\u0964])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_naive(text: str, chunk_size_words: int = 80, overlap_words: int = 15) -> list[str]:
    """
    Naive word-count based chunking with overlap.
    """
    words = text.split()
    if len(words) <= chunk_size_words:
        return [text]
        
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += (chunk_size_words - overlap_words)
        
    return chunks

def chunk_semantic(text: str, embedding_model, threshold: float = 0.65) -> list[str]:
    """
    Splits text on sentence boundaries when the similarity of consecutive sentences 
    drops below a specific threshold.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences
        
    # Get embeddings for all sentences
    embeddings = embedding_model.encode(sentences)
    
    chunks = []
    current_chunk_sentences = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Compute cosine similarity between current sentence and previous sentence
        vec1 = embeddings[i-1]
        vec2 = embeddings[i]
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        else:
            similarity = 0.0
            
        # If similarity is below threshold, start a new chunk
        if similarity < threshold:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
        else:
            current_chunk_sentences.append(sentences[i])
            
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return chunks

def chunk_parent_child(text: str, parent_id: str, sentences_per_child: int = 2, overlap_sentences: int = 0) -> list[dict]:
    """
    Splits a parent text into smaller child text chunks.
    Returns a list of dicts mapping each child chunk back to its parent.
    """
    sentences = split_sentences(text)
    if len(sentences) <= sentences_per_child:
        return [{
            "child_text": text,
            "parent_id": parent_id,
            "parent_text": text
        }]
        
    child_chunks = []
    start = 0
    while start < len(sentences):
        end = start + sentences_per_child
        child_text = " ".join(sentences[start:end])
        child_chunks.append({
            "child_text": child_text,
            "parent_id": parent_id,
            "parent_text": text
        })
        start += (sentences_per_child - overlap_sentences)
        
    return child_chunks
