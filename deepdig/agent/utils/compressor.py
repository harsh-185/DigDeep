"""
Lightweight extractive compressor — no external deps beyond tiktoken.

Scores sentences by information density (numbers, proper nouns, technical
terms, URLs) and drops low-value filler to hit a token budget.
"""

import re

from agent.utils.token_counter import count_tokens, truncate_to_tokens

# Patterns that signal high-information content
_NUMBER = re.compile(r"\d[\d,.%$€£]+")
_PROPER_NOUN = re.compile(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*")
_URL = re.compile(r"https?://\S+")
_TECHNICAL = re.compile(
    r"\b(?:API|GPU|CPU|LLM|AI|ML|NLP|RAG|RLHF|SaaS|IoT|SDK|"
    r"GDP|CAGR|YoY|ROI|EBITDA|IPO|VC|"
    r"billion|million|trillion|percent|"
    r"patent|regulation|compliance|framework|architecture|"
    r"benchmark|dataset|parameter|token|latency|throughput)\b",
    re.IGNORECASE,
)

# Low-value filler patterns
_FILLER = re.compile(
    r"^(in conclusion|overall|additionally|furthermore|moreover|"
    r"it is worth noting|it should be noted|as mentioned|"
    r"in this article|in this section|click here|subscribe|"
    r"related articles|share this|read more)\b",
    re.IGNORECASE,
)

# Sentence splitter — handles ., !, ? followed by space or end
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _score_sentence(sentence: str) -> float:
    """Score a sentence by information density. Higher = more valuable."""
    s = sentence.strip()
    if not s or len(s) < 15:
        return 0.0

    # Penalise filler
    if _FILLER.search(s):
        return 0.1

    score = 1.0
    score += len(_NUMBER.findall(s)) * 1.5        # numbers are high-value
    score += len(_PROPER_NOUN.findall(s)) * 0.8    # named entities
    score += len(_URL.findall(s)) * 0.3             # sources (modest value)
    score += len(_TECHNICAL.findall(s)) * 1.0       # domain terms

    # Bonus for concise, fact-dense sentences
    words = s.split()
    if 8 <= len(words) <= 35:
        score += 0.5

    return score


def compress_text(text: str, max_tokens: int) -> str:
    """
    Extract the most information-dense sentences that fit within max_tokens.

    If the text already fits, returns it unchanged.
    """
    if count_tokens(text) <= max_tokens:
        return text

    sentences = _SENT_SPLIT.split(text)
    if not sentences:
        return truncate_to_tokens(text, max_tokens)

    # Score each sentence
    scored = [(s, _score_sentence(s)) for s in sentences]

    # Sort by score descending, pick best until budget filled
    scored.sort(key=lambda x: x[1], reverse=True)

    picked: list[tuple[str, int]] = []  # (sentence, original_index)
    tokens_used = 0

    # Keep a map of original order for reassembly
    original_order = {s: i for i, s in enumerate(sentences)}

    for sent, score in scored:
        if score <= 0.1:
            continue
        t = count_tokens(sent)
        if tokens_used + t > max_tokens:
            continue
        picked.append((sent, original_order.get(sent, 0)))
        tokens_used += t

    if not picked:
        return truncate_to_tokens(text, max_tokens)

    # Reassemble in original order for coherence
    picked.sort(key=lambda x: x[1])
    return " ".join(s for s, _ in picked)
