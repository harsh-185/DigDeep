"""
Lightweight relevance scorer — no LLM calls, uses token overlap + keyword matching.

Scores how relevant a finding is to the original question.
Used to filter junk before it wastes episodic buffer slots.
"""

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _extract_key_terms(text: str) -> set[str]:
    """Extract meaningful terms (skip very common words)."""
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "that", "this", "these", "those", "it", "its", "they", "them",
        "what", "which", "who", "whom", "how", "when", "where", "why",
        "if", "then", "than", "more", "most", "some", "any", "all", "each",
        "about", "between", "such", "also", "very", "just", "only",
    }
    tokens = _tokenize(text)
    return {t for t in tokens if t not in stop and len(t) > 2}


def relevance_score(question: str, finding_content: str) -> float:
    """
    Score from 0.0 to 1.0 indicating how relevant the finding is to the question.

    Uses:
    - Keyword overlap between question terms and finding
    - Bigram overlap for multi-word concept matching
    - Length penalty for very short content (likely junk)
    """
    if not finding_content or len(finding_content.strip()) < 30:
        return 0.0

    q_terms = _extract_key_terms(question)
    f_terms = _extract_key_terms(finding_content)

    if not q_terms:
        return 0.5  # can't score, assume neutral

    # Unigram overlap: what fraction of question terms appear in finding
    overlap = q_terms & f_terms
    unigram_score = len(overlap) / len(q_terms) if q_terms else 0.0

    # Bigram overlap for multi-word concepts
    q_tokens = _tokenize(question)
    f_tokens = _tokenize(finding_content)
    q_bigrams = {(q_tokens[i], q_tokens[i + 1]) for i in range(len(q_tokens) - 1)}
    f_bigrams = {(f_tokens[i], f_tokens[i + 1]) for i in range(len(f_tokens) - 1)}
    bigram_score = len(q_bigrams & f_bigrams) / max(len(q_bigrams), 1)

    # Combined score (unigrams weighted more)
    score = 0.7 * unigram_score + 0.3 * bigram_score

    return min(score, 1.0)
