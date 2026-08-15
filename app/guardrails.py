import re
import logging

logger = logging.getLogger(__name__)

# Basic unsafe/blacklisted keyword regex patterns
UNSAFE_PATTERNS = [
    r"\b(hack|jailbreak|bypass|ignore previous instructions)\b",
    r"\b(abuse|kill|suicide|exploit|bomb|weapons)\b"
]

def check_input_safety(query: str) -> tuple[bool, str]:
    """
    Checks if the input query is safe and appropriate.
    Returns (is_safe, reason).
    """
    cleaned_query = query.lower().strip()
    
    # 1. Check for prompt injection or unsafe keywords
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, cleaned_query):
            logger.warning(f"Input safety guardrail triggered for query: '{query}'")
            return False, "Unsafe keyword or injection attempt detected."
            
    # 2. Check for empty or extremely short queries
    if len(cleaned_query) < 2:
        return False, "Query is too short."
        
    return True, ""

def check_grounding(answer: str, retrieved_passages: list[dict], threshold: float = 0.2) -> tuple[bool, float]:
    """
    A fast deterministic output guardrail checking if the generated answer is grounded in retrieved context.
    Computes overlap of key terms (nouns, numbers, specific terms) from the answer against the context.
    Returns (is_grounded, overlap_ratio).
    """
    if not answer or not retrieved_passages:
        return False, 0.0
        
    # Combine context texts
    context = " ".join([p["text"] for p in retrieved_passages]).lower()
    
    # Tokenize answer and context into alphanumeric tokens (handles Indic char sets properly)
    answer_tokens = re.findall(r"\w+", answer.lower())
    
    if not answer_tokens:
        return True, 1.0  # Safe fallback if no word tokens are parsed
        
    # Filter out stopwords (indic and english) to focus on content words
    stopwords = {
        "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "है", "हैं", "का", "की", "के", "को", "ने", "से", "में", "पर", "और", "या", "भी", "ही",
        "হয়", "এবং", "ও", "এর", "কে", "তে", "থেকে", "দ্বারা", "দিয়ে", "ওপর"
    }
    
    content_tokens = [t for t in answer_tokens if t not in stopwords and len(t) > 1]
    
    if not content_tokens:
        # Fallback to checking all tokens if all were filtered
        content_tokens = answer_tokens
        
    # Check how many content words from answer exist in the context
    matched = 0
    for token in content_tokens:
        if token in context:
            matched += 1
            
    overlap_ratio = matched / len(content_tokens)
    
    is_grounded = overlap_ratio >= threshold
    logger.info(f"Grounding check: matched {matched}/{len(content_tokens)} terms (ratio: {overlap_ratio:.2f}). Grounded: {is_grounded}")
    
    return is_grounded, overlap_ratio
