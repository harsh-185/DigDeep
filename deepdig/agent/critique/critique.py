"""
Self-critique: given an initial answer, identify gaps and generate
targeted follow-up search queries to fill them.
"""

from agent.memory.cost_tracker import CostTracker
from agent.utils.llm_client import call_llm

CRITIQUE_SYSTEM = (
    "You are a research quality reviewer. Given a question and a draft answer, "
    "identify 2-4 specific gaps, unsupported claims, or missing perspectives. "
    "For each gap, generate a targeted web search query to fill it. "
    "Use the report_gaps tool."
)

CRITIQUE_TOOL = {
    "name": "report_gaps",
    "description": "Report identified gaps and follow-up search queries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "gap": {
                            "type": "string",
                            "description": "What is missing or weakly supported",
                        },
                        "search_query": {
                            "type": "string",
                            "description": "Targeted web search query to fill this gap",
                        },
                    },
                    "required": ["gap", "search_query"],
                },
                "minItems": 1,
                "maxItems": 4,
            }
        },
        "required": ["gaps"],
    },
}


def critique_answer(
    question: str,
    answer: str,
    cost_tracker: CostTracker,
) -> list[dict]:
    """
    Review an answer and return targeted follow-up queries for gaps.
    Returns list of {"gap": "...", "search_query": "..."}.
    Returns empty list if budget is too tight.
    """
    if not cost_tracker.can_afford():
        return []

    user_msg = (
        f"Question: {question}\n\n"
        f"Draft answer:\n{answer[:2000]}\n\n"
        f"Identify specific gaps and generate follow-up searches."
    )

    result = call_llm(
        system_prompt=CRITIQUE_SYSTEM,
        user_message=user_msg,
        cost_tracker=cost_tracker,
        purpose="self_critique",
        tools=[CRITIQUE_TOOL],
        max_tokens=512,
    )

    # Extract from tool_use
    for tc in result.get("tool_calls", []):
        if tc["name"] == "report_gaps":
            return tc["input"].get("gaps", [])

    return []
