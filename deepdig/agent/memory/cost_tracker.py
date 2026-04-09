from agent.config import settings
from agent.utils.token_counter import count_tokens

# Pricing lookup by model name
MODEL_PRICING = {
    settings.LLM_MODEL: {
        "input": settings.SONNET_INPUT_COST_PER_1K,
        "output": settings.SONNET_OUTPUT_COST_PER_1K,
    },
    settings.LLM_MODEL_LIGHT: {
        "input": settings.HAIKU_INPUT_COST_PER_1K,
        "output": settings.HAIKU_OUTPUT_COST_PER_1K,
    },
}

# Default pricing (Sonnet) for backward compat
_DEFAULT_INPUT = settings.SONNET_INPUT_COST_PER_1K
_DEFAULT_OUTPUT = settings.SONNET_OUTPUT_COST_PER_1K


class CostTracker:
    """Tracks token usage and cost across all LLM calls in a session."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_log = []  # every LLM call logged

    def record_call(self, input_text: str, output_text: str, purpose: str, model: str = None) -> dict:
        """Record an LLM call and its cost using model-specific pricing."""
        input_tokens = count_tokens(input_text)
        output_tokens = count_tokens(output_text)

        pricing = MODEL_PRICING.get(model, {})
        input_rate = pricing.get("input", _DEFAULT_INPUT)
        output_rate = pricing.get("output", _DEFAULT_OUTPUT)

        cost = (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.call_log.append(
            {
                "purpose": purpose,
                "model": model or settings.LLM_MODEL,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "cumulative_cost": self.total_cost,
            }
        )
        return {"cost": cost, "cumulative": self.total_cost}

    def can_afford(self, estimated_input_tokens: int = 2000) -> bool:
        """Check if we can afford another LLM call within budget."""
        # Use cheapest model pricing for optimistic estimate
        estimated_cost = (estimated_input_tokens / 1000) * settings.SONNET_INPUT_COST_PER_1K
        return (self.total_cost + estimated_cost) <= settings.MAX_COST_PER_SESSION

    def get_report(self) -> dict:
        """Return a full cost report for this session."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": round(self.total_cost, 6),
            "budget_remaining": round(
                settings.MAX_COST_PER_SESSION - self.total_cost, 6
            ),
            "budget_used_pct": round(
                (self.total_cost / settings.MAX_COST_PER_SESSION) * 100, 1
            ),
            "call_log": self.call_log,
        }
