import anthropic

from agent.config import settings
from agent.memory.cost_tracker import CostTracker
from agent.utils.token_counter import count_tokens, truncate_to_tokens

client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def call_llm(
    system_prompt: str,
    user_message: str,
    cost_tracker: CostTracker,
    purpose: str,
    tools: list | None = None,
    max_tokens: int = 1024,
    model: str | None = None,
) -> dict:
    """
    Central LLM call with constraint enforcement and model routing.
    Every call goes through here — nothing calls Claude directly.

    model: which model to use. Defaults to settings.LLM_MODEL (Sonnet).
           Pass settings.LLM_MODEL_LIGHT for cheaper tasks.
    """
    resolved_model = model or settings.LLM_MODEL

    # CONSTRAINT: Enforce max context per call
    total_input = system_prompt + user_message
    input_tokens = count_tokens(total_input)

    if input_tokens > settings.MAX_CONTEXT_TOKENS_PER_CALL:
        # Truncate user message to fit (preserve system prompt)
        system_tokens = count_tokens(system_prompt)
        available = settings.MAX_CONTEXT_TOKENS_PER_CALL - system_tokens
        user_message = truncate_to_tokens(user_message, max(available, 500))

    # CONSTRAINT: Check cost budget before calling
    if not cost_tracker.can_afford(count_tokens(system_prompt + user_message)):
        return {
            "content": "[BUDGET EXCEEDED] Cannot make this call — session cost limit reached.",
            "tool_calls": [],
            "budget_exceeded": True,
            "tokens_used": 0,
        }

    kwargs = {
        "model": resolved_model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    if tools:
        kwargs["tools"] = tools

    response = client.messages.create(**kwargs)

    # Extract text content
    output_text = ""
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            output_text += block.text
        elif block.type == "tool_use":
            tool_calls.append({"name": block.name, "input": block.input})

    # Record cost with model-specific pricing
    cost_tracker.record_call(system_prompt + user_message, output_text, purpose, model=resolved_model)

    return {
        "content": output_text,
        "tool_calls": tool_calls,
        "budget_exceeded": False,
        "tokens_used": count_tokens(total_input) + count_tokens(output_text),
    }
