from agent.utils.token_counter import count_tokens, truncate_to_tokens


def test_count_tokens_basic():
    text = "Hello, world!"
    tokens = count_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_truncate_within_limit():
    text = "Short text"
    result = truncate_to_tokens(text, 100)
    assert result == text


def test_truncate_exceeds_limit():
    text = "This is a much longer piece of text that should be truncated " * 20
    result = truncate_to_tokens(text, 10)
    result_tokens = count_tokens(result)
    assert result_tokens <= 10


def test_truncate_preserves_content():
    text = "Hello world foo bar"
    result = truncate_to_tokens(text, 2)
    assert len(result) < len(text)
