from unittest.mock import patch

from agent.memory.cost_tracker import CostTracker


def test_decompose_query_parses_json():
    """Test that the decomposer correctly parses a JSON response."""
    mock_response = {
        "content": '[{"id": 1, "query": "sub q1", "purpose": "test", "depends_on": []}]',
        "tool_calls": [],
        "budget_exceeded": False,
        "tokens_used": 100,
    }
    with patch("agent.decomposer.decomposer.call_llm", return_value=mock_response):
        from agent.decomposer.decomposer import decompose_query

        tracker = CostTracker()
        result = decompose_query("What is AI?", tracker)
        assert len(result) == 1
        assert result[0]["query"] == "sub q1"


def test_decompose_query_handles_markdown_json():
    """Test that the decomposer handles JSON wrapped in markdown code blocks."""
    mock_response = {
        "content": '```json\n[{"id": 1, "query": "test", "purpose": "p", "depends_on": []}]\n```',
        "tool_calls": [],
        "budget_exceeded": False,
        "tokens_used": 100,
    }
    with patch("agent.decomposer.decomposer.call_llm", return_value=mock_response):
        from agent.decomposer.decomposer import decompose_query

        tracker = CostTracker()
        result = decompose_query("complex question", tracker)
        assert len(result) == 1


def test_decompose_query_fallback_on_bad_json():
    """Test that invalid JSON falls back to the original question."""
    mock_response = {
        "content": "This is not valid JSON at all",
        "tool_calls": [],
        "budget_exceeded": False,
        "tokens_used": 100,
    }
    with patch("agent.decomposer.decomposer.call_llm", return_value=mock_response):
        from agent.decomposer.decomposer import decompose_query

        tracker = CostTracker()
        result = decompose_query("fallback question", tracker)
        assert len(result) == 1
        assert result[0]["query"] == "fallback question"


def test_decompose_query_enforces_max_sub_queries():
    """Test that the max sub-queries constraint is enforced."""
    many_queries = [
        {"id": i, "query": f"q{i}", "purpose": f"p{i}", "depends_on": []}
        for i in range(1, 20)
    ]
    import json

    mock_response = {
        "content": json.dumps(many_queries),
        "tool_calls": [],
        "budget_exceeded": False,
        "tokens_used": 100,
    }
    with patch("agent.decomposer.decomposer.call_llm", return_value=mock_response):
        from agent.decomposer.decomposer import decompose_query

        tracker = CostTracker()
        result = decompose_query("big question", tracker)
        assert len(result) <= 8  # MAX_SUB_QUERIES default
