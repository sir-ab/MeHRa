"""
Memory storage backends: Chroma DB for vector/metadata storage.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError(
        "chromadb is required. Install with: pip install chromadb"
    )

from .memory import MemoryEntry, MemoryType, MemoryBackend

logger = logging.getLogger(__name__)


class ChromaBackend(MemoryBackend):
    """
    Chroma DB backend for memory storage and vector search.
    
    Features:
    - Persistent SQLite storage with automatic embeddings.
    - Supports filtering by type, user, and metadata.
    - Hybrid search combining semantic similarity with metadata filters.
    """

    def __init__(self, persist_dir: str = "./memory_storage", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize Chroma backend.
        
        Args:
            persist_dir: Directory for persistent storage (default: ./memory_storage).
            embedding_model: Embedding model name (Chroma's default is all-MiniLM-L6-v2, 
                           a small, fast model suitable for local use).
        """
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model

        # Initialize Chroma client with persistence
        settings = Settings(
            is_persistent=True,
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)

        # Create or get collection for memories
        # Note: Chroma uses built-in embeddings via HuggingFace
        self.collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(f"Initialized Chroma backend with persist_dir={persist_dir}")

    def index(self, entry: MemoryEntry) -> None:
        """
        Index a memory entry in Chroma.
        
        Chroma handles embeddings automatically if embedding is None.
        Otherwise, use the provided embedding.
        """
        # Prepare metadata (exclude embedding and text)
        metadata = {
            "type": entry.type.value,
            "source": entry.source,
            "importance": float(entry.importance_score),
            "timestamp": entry.timestamp.isoformat(),
            "user_id": entry.user_id or "",
            "tags": ",".join(entry.tags),
            **entry.metadata,
        }

        # If entry already exists, delete it first (Chroma upsert doesn't exist)
        try:
            self.collection.delete(ids=[entry.id])
        except Exception:
            pass  # Memory may not exist yet

        # Add to collection
        # If entry.embedding is None, Chroma will compute it automatically
        if entry.embedding:
            self.collection.add(
                ids=[entry.id],
                documents=[entry.text],
                metadatas=[metadata],
                embeddings=[entry.embedding],
            )
        else:
            self.collection.add(
                ids=[entry.id],
                documents=[entry.text],
                metadatas=[metadata],
            )

        logger.debug(f"Indexed memory {entry.id} in Chroma")

    def search(
        self,
        query_embedding: Optional[List[float]],
        top_k: int,
        mem_types: Optional[List[MemoryType]] = None,
    ) -> List[MemoryEntry]:
        """
        Search for memories using semantic similarity.
        
        Args:
            query_embedding: Embedding vector (ignored if None; Chroma will embed the query text).
            top_k: Number of results to return.
            mem_types: Optional filter by memory types.
        
        Returns:
            List of MemoryEntry objects.
        """
        # Build where filter if needed
        where_filter = None
        if mem_types:
            type_values = [t.value for t in mem_types]
            if len(type_values) == 1:
                where_filter = {"type": {"$eq": type_values[0]}}
            else:
                where_filter = {"type": {"$in": type_values}}

        # Query Chroma
        # If query_embedding provided, use it; otherwise Chroma auto-embeds query text
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding] if query_embedding else None,
                n_results=top_k,
                where=where_filter,
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []

        # Convert results to MemoryEntry objects
        memories = []
        if results and results.get("ids"):
            for i, mem_id in enumerate(results["ids"][0]):
                try:
                    metadata = results["metadatas"][0][i]
                    text = results["documents"][0][i]
                    embedding = (
                        results["embeddings"][0][i]
                        if results.get("embeddings") and results["embeddings"][0]
                        else None
                    )

                    from datetime import datetime
                    entry = MemoryEntry(
                        id=mem_id,
                        text=text,
                        type=MemoryType(metadata.get("type", "episodic")),
                        timestamp=datetime.fromisoformat(metadata["timestamp"]),
                        embedding=embedding,
                        importance_score=float(metadata.get("importance", 0.5)),
                        source=metadata.get("source", "unknown"),
                        tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                        user_id=metadata.get("user_id") or None,
                        metadata={k: v for k, v in metadata.items() 
                                 if k not in ["type", "source", "importance", "timestamp", "user_id", "tags"]},
                    )
                    memories.append(entry)
                except Exception as e:
                    logger.error(f"Failed to parse memory {mem_id}: {e}")

        return memories

    def delete(self, mem_id: str) -> None:
        """Delete a memory from Chroma."""
        try:
            self.collection.delete(ids=[mem_id])
            logger.debug(f"Deleted memory {mem_id} from Chroma")
        except Exception as e:
            logger.error(f"Failed to delete memory {mem_id}: {e}")

    def similarity(
        self, emb1: Optional[List[float]], emb2: Optional[List[float]]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.
        
        Returns:
            Similarity score (0-1); 0 = dissimilar, 1 = identical.
        """
        if emb1 is None or emb2 is None:
            return 0.5  # Default if embeddings missing

        emb1 = np.array(emb1, dtype=np.float32)
        emb2 = np.array(emb2, dtype=np.float32)

        # Cosine similarity: (A · B) / (||A|| * ||B||)
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        # Map from [-1, 1] to [0, 1]
        return (similarity + 1.0) / 2.0

    def clear(self) -> None:
        """Delete all memories from the collection."""
        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data["ids"]:
                self.collection.delete(ids=all_data["ids"])
            logger.info("Cleared all memories from Chroma")
        except Exception as e:
            logger.error(f"Failed to clear Chroma collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {"memory_count": count, "persist_dir": self.persist_dir}
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
