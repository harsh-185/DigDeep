import uuid

from agent.config import settings
from agent.memory.cost_tracker import CostTracker
from agent.memory.episodic_buffer import EpisodicBuffer
from agent.memory.summarizer import SummarizationCascade
from agent.memory.vector_store import VectorStore
from agent.utils.token_counter import count_tokens, truncate_to_tokens


class MemoryManager:
    """
    Unified interface for all memory operations.
    This is what the rest of the system talks to — it handles
    the routing between episodic buffer, vector store, and summarizer.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.cost_tracker = CostTracker()
        self.episodic = EpisodicBuffer()
        self.vector = VectorStore(self.session_id)
        self.summarizer = SummarizationCascade(self.cost_tracker)
        self._eviction_batch: list[dict] = []  # batch evictions for compression

    def resize_buffer(self, new_size: int):
        """Dynamically resize the episodic buffer (orchestrator-driven)."""
        from collections import deque
        old = list(self.episodic.buffer)
        self.episodic.buffer = deque(old[-new_size:], maxlen=new_size)

    def store_finding(self, content: str, source: str, sub_query: str):
        """Store a new research finding. Batches evictions for compression."""
        finding = {
            "content": content,
            "source": source,
            "sub_query": sub_query,
            "token_count": count_tokens(content),
        }

        evicted = self.episodic.add(finding)

        if evicted:
            self._eviction_batch.append(evicted)

    def flush_evictions(self):
        """
        Compress all batched evictions in a SINGLE LLM call (or zero if budget is tight).
        Call this after finishing all sub-queries for a research cycle, not per-finding.
        This is the key optimization: 1 LLM call instead of N.
        """
        if not self._eviction_batch:
            return

        # Combine all evicted findings into one text block
        combined_parts = []
        for entry in self._eviction_batch:
            combined_parts.append(
                f"{entry['sub_query']}|{entry['content']}"
            )
        combined_text = "\n".join(combined_parts)

        if self.cost_tracker.can_afford():
            # One LLM call to compress all evicted entries together
            compressed = self.summarizer.compress(combined_text)
        else:
            # Budget tight — just truncate, no LLM call
            compressed = truncate_to_tokens(combined_text, 400)

        # Store the single compressed summary in vector store
        sources = list({e["source"] for e in self._eviction_batch})
        sub_queries = list({e["sub_query"] for e in self._eviction_batch})
        self.vector.store(
            compressed,
            {
                "source": ", ".join(sources[:3]),
                "sub_query": ", ".join(sub_queries[:3]),
                "type": "batch_compressed",
                "num_findings": str(len(self._eviction_batch)),
            },
        )

        self._eviction_batch.clear()

    def retrieve_context(self, query: str, max_tokens: int | None = None) -> str:
        """
        Build context for an LLM call within token budget.
        Priority: episodic (recent) first, then vector (semantic).
        """
        budget = max_tokens or (settings.MAX_CONTEXT_TOKENS_PER_CALL // 2)
        context_parts = []
        tokens_used = 0

        # 1. Recent findings from episodic buffer (high priority)
        recent = self.episodic.get_recent(5)
        for entry in recent:
            entry_tokens = count_tokens(entry["content"])
            if tokens_used + entry_tokens > budget:
                remaining = budget - tokens_used
                if remaining > 100:
                    context_parts.append(
                        f"R({entry['sub_query']}): "
                        + truncate_to_tokens(entry["content"], remaining)
                    )
                break
            context_parts.append(
                f"R({entry['sub_query']}): {entry['content']}"
            )
            tokens_used += entry_tokens

        # 2. Semantic retrieval from vector store (fill remaining budget)
        if tokens_used < budget and self.vector.count() > 0:
            retrieved = self.vector.retrieve(query)
            for item in retrieved:
                item_tokens = count_tokens(item["content"])
                if tokens_used + item_tokens > budget:
                    remaining = budget - tokens_used
                    if remaining > 100:
                        context_parts.append(
                            f"A: {truncate_to_tokens(item['content'], remaining)}"
                        )
                    break
                context_parts.append(f"A: {item['content']}")
                tokens_used += item_tokens

        return "\n\n".join(context_parts) if context_parts else "[No prior context]"

    def get_session_report(self) -> dict:
        """Return a summary of this session's memory and cost state."""
        return {
            "session_id": self.session_id,
            "episodic_entries": len(self.episodic.get_all()),
            "vector_entries": self.vector.count(),
            "cost_report": self.cost_tracker.get_report(),
        }
