"""
End-to-end tests using FastAPI test client.
These mock external APIs (Claude, Tavily) to test the full pipeline.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.main import app

client = TestClient(app)


def mock_call_llm(system_prompt, user_message, cost_tracker, purpose, **kwargs):
    """Mock LLM that returns canned responses based on purpose."""
    if purpose == "query_decomposition":
        return {
            "content": '[{"id": 1, "query": "test query", "purpose": "testing", "depends_on": []}]',
            "tool_calls": [
                {
                    "name": "submit_sub_queries",
                    "input": {
                        "sub_queries": [
                            {"id": 1, "query": "test query", "purpose": "testing", "depends_on": []}
                        ]
                    },
                }
            ],
            "budget_exceeded": False,
            "tokens_used": 50,
        }
    elif purpose == "synthesis":
        return {
            "content": "This is a synthesized answer based on the research findings.",
            "tool_calls": [],
            "budget_exceeded": False,
            "tokens_used": 50,
        }
    elif purpose.startswith("summarization"):
        return {
            "content": "Compressed summary of the finding.",
            "tool_calls": [],
            "budget_exceeded": False,
            "tokens_used": 30,
        }
    return {
        "content": "default response",
        "tool_calls": [],
        "budget_exceeded": False,
        "tokens_used": 20,
    }


_mock_call_count = 0


async def mock_parallel_research(sub_queries, seen_urls=None):
    """Mock parallel research that returns canned findings with unique URLs."""
    global _mock_call_count
    results = []
    for sq in sub_queries:
        _mock_call_count += 1
        q = sq["query"]
        results.append({
            "sub_query": sq,
            "findings": [
                {
                    "content": f"Research on {q}: quantum computing applications in healthcare and AI impact analysis with detailed data.",
                    "source": f"https://example.com/{_mock_call_count}/a",
                    "token_count": 20,
                },
                {
                    "content": f"Study of {q}: European AI startups and computing technology applications overview with statistics.",
                    "source": f"https://example.com/{_mock_call_count}/b",
                    "token_count": 18,
                },
            ],
        })
    return {"results": results, "dedup_count": 0}


@patch("agent.decomposer.decomposer.call_llm", side_effect=mock_call_llm)
@patch("agent.synthesizer.synthesizer.call_llm", side_effect=mock_call_llm)
@patch("agent.main.research_sub_queries_parallel", side_effect=mock_parallel_research)
def test_full_research_pipeline(mock_research, mock_synth, mock_decomp):
    """Test the complete research pipeline end-to-end."""
    response = client.post(
        "/research",
        json={"question": "What is the impact of AI on healthcare?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sub_queries" in data
    assert "session_id" in data
    assert "session_report" in data
    assert len(data["answer"]) > 0


@patch("agent.decomposer.decomposer.call_llm", side_effect=mock_call_llm)
@patch("agent.synthesizer.synthesizer.call_llm", side_effect=mock_call_llm)
@patch("agent.main.research_sub_queries_parallel", side_effect=mock_parallel_research)
def test_session_continuity(mock_research, mock_synth, mock_decomp):
    """Test that follow-up queries reuse the same session memory."""
    # First request
    r1 = client.post(
        "/research",
        json={"question": "What is quantum computing?"},
    )
    assert r1.status_code == 200
    session_id = r1.json()["session_id"]

    # Second request reusing session
    r2 = client.post(
        "/research",
        json={"question": "What are the applications?", "session_id": session_id},
    )
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id

    # The second request should show more accumulated data
    report = r2.json()["session_report"]
    assert report["episodic_entries"] > 0


@patch("agent.decomposer.decomposer.call_llm", side_effect=mock_call_llm)
@patch("agent.synthesizer.synthesizer.call_llm", side_effect=mock_call_llm)
@patch("agent.main.research_sub_queries_parallel", side_effect=mock_parallel_research)
def test_session_report_endpoint(mock_research, mock_synth, mock_decomp):
    """Test the session report endpoint."""
    # Create a session
    r = client.post(
        "/research",
        json={"question": "Test question"},
    )
    session_id = r.json()["session_id"]

    # Get report
    report_resp = client.get(f"/session/{session_id}/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert "session_id" in report
    assert "cost_report" in report


def test_session_not_found():
    """Test 404 for non-existent session."""
    resp = client.get("/session/nonexistent/report")
    assert resp.status_code == 404


def test_health_endpoint():
    """Test the health check endpoint."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
