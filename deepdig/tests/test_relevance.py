from agent.utils.relevance import relevance_score


def test_relevant_content_scores_high():
    """Content matching the question should score well."""
    question = "How has the EU AI Act affected AI startups in Europe?"
    content = (
        "The EU AI Act has imposed significant compliance costs on European AI startups. "
        "Companies report spending $200K-$500K on initial regulatory compliance."
    )
    score = relevance_score(question, content)
    assert score > 0.3


def test_irrelevant_content_scores_low():
    """Completely unrelated content should score low."""
    question = "How has the EU AI Act affected AI startups in Europe?"
    content = (
        "The best chocolate cake recipe uses dark chocolate, butter, and eggs. "
        "Bake at 350 degrees for 25 minutes until a toothpick comes out clean."
    )
    score = relevance_score(question, content)
    assert score < 0.15


def test_empty_content_scores_zero():
    """Empty or very short content should score zero."""
    assert relevance_score("any question", "") == 0.0
    assert relevance_score("any question", "too short") == 0.0


def test_partial_overlap_scores_moderate():
    """Content with some keyword overlap should score moderately."""
    question = "Compare GPT-4 and Claude memory architectures"
    content = (
        "GPT-4 uses a transformer architecture with attention mechanisms. "
        "The model has shown impressive performance on various benchmarks."
    )
    score = relevance_score(question, content)
    assert 0.1 < score < 0.7


def test_bigram_matching_boosts_score():
    """Content with matching bigrams should score higher."""
    question = "EU AI Act compliance costs"
    with_bigrams = "The EU AI Act requires compliance costs of $300K for startups."
    without_bigrams = "European regulations affect artificial intelligence companies costs."
    score_with = relevance_score(question, with_bigrams)
    score_without = relevance_score(question, without_bigrams)
    assert score_with > score_without
