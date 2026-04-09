from agent.memory.manager import MemoryManager
from agent.utils.llm_client import call_llm

SYNTHESIZE_SYSTEM = """Answer using ONLY the provided context. Cite sources as [Source: URL]. Note conflicts. State gaps. Be thorough but concise."""


def synthesize_answer(
    original_question: str,
    memory_manager: MemoryManager,
) -> dict:
    """Build final answer from accumulated research context."""

    # Retrieve relevant context within token budget
    context = memory_manager.retrieve_context(
        query=original_question,
        max_tokens=2500,  # leave room for system prompt + answer
    )

    user_msg = (
        f"Q: {original_question}\n\n"
        f"Context:\n{context}\n\n"
        f"Synthesize answer from context above."
    )

    result = call_llm(
        system_prompt=SYNTHESIZE_SYSTEM,
        user_message=user_msg,
        cost_tracker=memory_manager.cost_tracker,
        purpose="synthesis",
        max_tokens=1500,
    )

    return {
        "answer": result["content"],
        "budget_exceeded": result.get("budget_exceeded", False),
        "session_report": memory_manager.get_session_report(),
    }
