import pytest

from agent.config import settings
from agent.memory.episodic_buffer import EpisodicBuffer
from agent.memory.vector_store import VectorStore


class TestEpisodicBuffer:
    def test_add_and_retrieve(self):
        buf = EpisodicBuffer()
        evicted = buf.add({"content": "Finding 1", "source": "url1", "sub_query": "q1", "token_count": 10})
        assert evicted is None
        assert len(buf.get_all()) == 1

    def test_get_recent(self):
        buf = EpisodicBuffer()
        for i in range(5):
            buf.add({"content": f"Finding {i}", "source": f"url{i}", "sub_query": f"q{i}", "token_count": 10})
        recent = buf.get_recent(3)
        assert len(recent) == 3
        assert recent[-1]["content"] == "Finding 4"

    def test_eviction_when_full(self):
        buf = EpisodicBuffer()
        capacity = settings.EPISODIC_BUFFER_SIZE
        # Fill buffer to capacity
        for i in range(capacity):
            buf.add({"content": f"Finding {i}", "source": f"url{i}", "sub_query": f"q{i}", "token_count": 10})

        # Next entry should evict the oldest
        evicted = buf.add({"content": f"Finding {capacity}", "source": f"url{capacity}", "sub_query": f"q{capacity}", "token_count": 10})
        assert evicted is not None
        assert evicted["content"] == "Finding 0"
        assert len(buf.get_all()) == capacity

    def test_get_total_tokens(self):
        buf = EpisodicBuffer()
        buf.add({"content": "A", "source": "s", "sub_query": "q", "token_count": 50})
        buf.add({"content": "B", "source": "s", "sub_query": "q", "token_count": 30})
        assert buf.get_total_tokens() == 80

    def test_clear(self):
        buf = EpisodicBuffer()
        buf.add({"content": "A", "source": "s", "sub_query": "q", "token_count": 10})
        buf.clear()
        assert len(buf.get_all()) == 0


class TestVectorStore:
    def test_store_and_retrieve(self):
        vs = VectorStore("test_session_1")
        vs.store("Machine learning is a subset of AI", {"source": "wiki"})
        vs.store("Deep learning uses neural networks", {"source": "paper"})

        results = vs.retrieve("What is machine learning?", top_k=2)
        assert len(results) > 0
        assert "content" in results[0]

    def test_empty_store_returns_empty(self):
        vs = VectorStore("test_session_2")
        results = vs.retrieve("anything")
        assert results == []

    def test_count(self):
        vs = VectorStore("test_session_3")
        assert vs.count() == 0
        vs.store("Document 1")
        vs.store("Document 2")
        assert vs.count() == 2
