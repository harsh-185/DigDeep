from agent.utils.compressor import compress_text, _score_sentence
from agent.utils.token_counter import count_tokens


def test_short_text_unchanged():
    """Text under budget passes through unchanged."""
    text = "GDP grew 3.2% in Q1 2025."
    result = compress_text(text, 100)
    assert result == text


def test_compression_respects_budget():
    """Output fits within the token budget."""
    long_text = (
        "The EU AI Act was enacted in 2024. "
        "It imposed compliance costs of $2.1 million on average for startups. "
        "European VC funding dropped 18% year-over-year. "
        "American AI startups raised $47 billion in 2025. "
        "Additionally, it is worth noting that many companies adapted quickly. "
        "Furthermore, the regulatory landscape continues to evolve. "
        "Moreover, several industry groups have formed to lobby for changes. "
        "In conclusion, the market dynamics are shifting rapidly."
    )
    budget = 40
    result = compress_text(long_text, budget)
    assert count_tokens(result) <= budget


def test_filler_sentences_deprioritized():
    """Filler sentences score lower than fact-dense ones."""
    filler = "In conclusion, this is a general overview of the topic at hand."
    factual = "European VC funding dropped 18% to $12.3 billion in Q1 2025."
    assert _score_sentence(factual) > _score_sentence(filler)


def test_numbers_boost_score():
    """Sentences with numbers score higher."""
    with_numbers = "Revenue reached $4.7 billion, up 23% year-over-year."
    without_numbers = "Revenue increased significantly compared to last year."
    assert _score_sentence(with_numbers) > _score_sentence(without_numbers)


def test_preserves_original_order():
    """Compressed output maintains sentence order from the original."""
    text = (
        "First came regulation in 2024. "
        "Additionally, nothing much happened. "
        "Then compliance costs hit $2.1 million. "
        "Furthermore, this is filler text only. "
        "Finally, VC funding reached $47 billion."
    )
    result = compress_text(text, 30)
    # The fact-dense sentences should appear in their original order
    if "regulation" in result and "$47 billion" in result:
        assert result.index("regulation") < result.index("$47 billion")
