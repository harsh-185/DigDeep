import json

from agent.config import settings
from agent.memory.cost_tracker import CostTracker
from agent.utils.llm_client import call_llm

DECOMPOSE_SYSTEM = (
    "Break the question into max {max_sub} independent sub-queries for web search. "
    "Order by dependency. Simple questions get 1 sub-query. "
    "Use the submit_sub_queries tool to return your answer."
)

# Claude tool_use schema — forces structured output, no JSON parsing needed
DECOMPOSE_TOOL = {
    "name": "submit_sub_queries",
    "description": "Submit the decomposed sub-queries for the research question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sub_queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "Unique ID starting from 1"},
                        "query": {"type": "string", "description": "Self-contained web search query"},
                        "purpose": {"type": "string", "description": "What this sub-query answers"},
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "IDs of sub-queries this depends on (empty if independent)",
                        },
                    },
                    "required": ["id", "query", "purpose", "depends_on"],
                },
            }
        },
        "required": ["sub_queries"],
    },
}


def decompose_query(question: str, cost_tracker: CostTracker) -> list[dict]:
    """Break a complex question into ranked, independent sub-queries using tool_use."""
    result = call_llm(
        system_prompt=DECOMPOSE_SYSTEM.format(max_sub=settings.MAX_SUB_QUERIES),
        user_message=question,
        cost_tracker=cost_tracker,
        purpose="query_decomposition",
        tools=[DECOMPOSE_TOOL],
        max_tokens=1024,
    )

    # Try tool_use output first (structured, reliable)
    for tc in result.get("tool_calls", []):
        if tc["name"] == "submit_sub_queries":
            sub_queries = tc["input"].get("sub_queries", [])
            return sub_queries[: settings.MAX_SUB_QUERIES]

    # Fallback: parse text output (in case model doesn't use tool)
    try:
        content = result["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        sub_queries = json.loads(content)
        return sub_queries[: settings.MAX_SUB_QUERIES]
    except (json.JSONDecodeError, IndexError, ValueError):
        return [
            {"id": 1, "query": question, "purpose": "direct search", "depends_on": []}
        ]
