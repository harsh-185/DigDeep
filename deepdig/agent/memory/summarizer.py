from agent.memory.cost_tracker import CostTracker
from agent.utils.llm_client import call_llm

SUMMARIZE_PROMPT = (
    "Compress to ~30% length. Keep facts, numbers, names, conclusions. "
    "Remove filler. Output ONLY the summary."
)


class SummarizationCascade:
    """
    When episodic buffer evicts old entries, they come here.
    Level 1: Individual finding -> compressed summary
    Level 2: Multiple Level 1 summaries -> mega-summary (when vector store gets large)

    This is the key architectural trick: old context doesn't disappear,
    it gets progressively compressed and stored in the vector store.
    """

    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker

    def compress(self, text: str) -> str:
        """Level 1: Compress a single finding."""
        result = call_llm(
            system_prompt=SUMMARIZE_PROMPT,
            user_message=text,
            cost_tracker=self.cost_tracker,
            purpose="summarization_L1",
            max_tokens=512,
                    )
        return result["content"]

    def mega_compress(self, texts: list[str]) -> str:
        """Level 2: Compress multiple summaries into one."""
        combined = "\n---\n".join(texts)
        result = call_llm(
            system_prompt=SUMMARIZE_PROMPT + " Merge multiple summaries into one.",
            user_message=combined,
            cost_tracker=self.cost_tracker,
            purpose="summarization_L2",
            max_tokens=512,
                    )
        return result["content"]
