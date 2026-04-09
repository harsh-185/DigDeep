from collections import deque
from datetime import datetime, timezone

from agent.config import settings


class EpisodicBuffer:
    """
    Short-term memory: holds the most recent N findings in full detail.
    When full, oldest entries get sent to summarization cascade.
    Think of this as "working memory."
    """

    def __init__(self):
        self.buffer: deque[dict] = deque(maxlen=settings.EPISODIC_BUFFER_SIZE)

    def add(self, finding: dict) -> dict | None:
        """
        Add a research finding. Auto-evicts oldest if full.
        Returns the evicted entry (if any) so the caller can compress it.
        """
        entry = {
            "content": finding["content"],
            "source": finding.get("source", "unknown"),
            "sub_query": finding.get("sub_query", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "token_count": finding.get("token_count", 0),
        }

        evicted = None
        if len(self.buffer) == self.buffer.maxlen:
            evicted = self.buffer[0]  # will be auto-removed by deque

        self.buffer.append(entry)
        return evicted

    def get_recent(self, n: int = 5) -> list[dict]:
        """Get the N most recent entries."""
        return list(self.buffer)[-n:]

    def get_all(self) -> list[dict]:
        """Get all entries in the buffer."""
        return list(self.buffer)

    def get_total_tokens(self) -> int:
        """Total token count across all buffered entries."""
        return sum(e["token_count"] for e in self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
