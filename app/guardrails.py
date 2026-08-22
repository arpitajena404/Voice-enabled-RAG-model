import re
import logging

logger = logging.getLogger(__name__)

# Enhanced unsafe / prompt injection regex patterns
UNSAFE_PATTERNS = [
    r"\b(hack|jailbreak|bypass|ignore\s+previous\s+instructions|system\s+prompt|overwrite\s+instructions)\b",
    r"\b(abuse|kill|suicide|exploit|bomb|weapon|weapons|terrorist|attack)\b",
    r"\b(malware|virus|trojan|ransomware|phishing|exploit)\b",
    r"(ignore\s+all\s+rules|you\s+are\s+now\s+DAN|do\s+anything\s+now)"
]

# Off-topic pattern triggers (code generation, creative writing, out-of-domain requests)
OFF_TOPIC_PATTERNS = [
    r"\b(write\s+a\s+(python|java|javascript|cpp|c\+\+|code|script|program|poem|song|essay|story))\b",
    r"\b(generate\s+(code|script|program|poem|song))\b",
    r"\b(code\s+for|how\s+to\s+code)\b",
    r"\b(tell\s+me\s+a\s+(joke|story))\b"
]

INDIC_ENGLISH_STOPWORDS = {
    "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "of", "from", "as", "be", "was", "were", "been", "that", "this", "it", "are",
    "है", "हैं", "का", "की", "के", "को", "ने", "से", "में", "पर", "और", "या", "भी", "ही",
    "था", "थी", "थे", "होता", "होती", "होते", "किया", "गया", "अपने", "इस", "उस",
    "হয়", "এবং", "ও", "এর", "কে", "তে", "থেকে", "দ্বারা", "দিয়ে", "ওপর", "করা", "হলে"
}

def check_input_safety(query: str) -> tuple[bool, str]:
    """
    Checks if the input query is safe and appropriate.
    Returns (is_safe, reason).
    """
    cleaned_query = query.lower().strip()
    
    # 1. Check for prompt injection or unsafe keywords
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            logger.warning(f"Input safety guardrail triggered for query: '{query}'")
            return False, "Unsafe keyword or injection attempt detected."
            
    # 2. Check for empty or extremely short queries
    if len(cleaned_query) < 2:
        return False, "Query is too short."
        
    return True, ""

def check_off_topic(query: str) -> tuple[bool, str]:
    """
    Checks if the query is explicitly off-topic (e.g., requesting code generation,
    creative writing, or out-of-domain actions).
    Returns (is_off_topic, reason).
    """
    cleaned_query = query.lower().strip()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            logger.info(f"Off-topic guardrail triggered for query: '{query}'")
            return True, "Query requests off-topic action (code, creative writing, or non-domain response)."
    return False, ""

def check_grounding(answer: str, retrieved_passages: list[dict], threshold: float = 0.2) -> tuple[bool, float]:
    """
    A fast deterministic output guardrail checking if the generated answer is grounded in retrieved context.
    Computes overlap of key content terms (nouns, numbers, specific terms) from the answer against the context.
    Returns (is_grounded, overlap_ratio).
    """
    if not answer or not retrieved_passages:
        return False, 0.0
        
    # Combine context texts
    context = " ".join([p.get("text", "") for p in retrieved_passages]).lower()
    
    # Tokenize answer into word tokens (handles Indic char sets properly)
    answer_tokens = re.findall(r"\w+", answer.lower())
    
    if not answer_tokens:
        return True, 1.0  # Safe fallback if no word tokens are parsed
        
    content_tokens = [t for t in answer_tokens if t not in INDIC_ENGLISH_STOPWORDS and len(t) > 1]
    
    if not content_tokens:
        content_tokens = answer_tokens
        
    matched = 0
    for token in content_tokens:
        if token in context:
            matched += 1
            
    overlap_ratio = matched / len(content_tokens)
    
    is_grounded = overlap_ratio >= threshold
    logger.info(f"Grounding check: matched {matched}/{len(content_tokens)} terms (ratio: {overlap_ratio:.2f}). Grounded: {is_grounded}")
    
    return is_grounded, overlap_ratio

