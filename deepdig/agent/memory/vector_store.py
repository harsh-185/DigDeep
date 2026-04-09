import chromadb
from chromadb.config import Settings as ChromaSettings

from agent.config import settings


class VectorStore:
    """
    Long-term memory: stores compressed/summarized findings.
    Used for semantic retrieval when episodic buffer doesn't have
    relevant context for the current sub-query.
    """

    def __init__(self, session_id: str):
        self.client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=f"session_{session_id}",
            metadata={"hnsw:space": "cosine"},
        )
        self._doc_counter = 0

    def store(self, text: str, metadata: dict | None = None):
        """Store a document in the vector store."""
        self._doc_counter += 1
        self.collection.add(
            documents=[text],
            ids=[f"doc_{self._doc_counter}"],
            metadatas=[metadata] if metadata else None,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrieve the most semantically similar documents."""
        k = top_k or settings.VECTOR_TOP_K
        n_docs = self.collection.count()
        if n_docs == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, n_docs),
        )
        if not results["documents"][0]:
            return []

        return [
            {"content": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def count(self) -> int:
        """Number of documents in the store."""
        return self.collection.count()
