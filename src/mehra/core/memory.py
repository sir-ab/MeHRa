"""
Memory system for LLM conversation context.
Supports short-term, episodic, semantic, and long-term memory with retrieval and ranking.
"""

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory in the system."""
    SHORT_TERM = "short_term"  # Current conversation window
    EPISODIC = "episodic"  # Time-stamped events/utterances
    SEMANTIC = "semantic"  # Facts, persistent preferences
    WORKING = "working"  # Intermediate summaries/derived state
    LONG_TERM = "long_term"  # Compressed historical summaries


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    text: str
    type: MemoryType
    timestamp: datetime
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None
    importance_score: float = 0.5  # 0-1 scale
    source: str = "conversation"  # Where the memory came from
    tags: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl_seconds: Optional[int] = None  # Time-to-live; None = indefinite

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['type'] = self.type.value
        d['timestamp'] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Restore from dictionary."""
        data = data.copy()
        data['type'] = MemoryType(data['type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def is_expired(self) -> bool:
        """Check if memory has exceeded its TTL."""
        if self.ttl_seconds is None:
            return False
        age = datetime.now() - self.timestamp
        return age.total_seconds() > self.ttl_seconds

    def recency_weight(self, decay_hours: float = 24.0) -> float:
        """Compute recency weight: closer to now = higher weight (0-1)."""
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600.0
        # Exponential decay: weight = exp(-age / decay_hours)
        import math
        return math.exp(-age_hours / decay_hours)


class MemoryManager:
    """
    Main interface for memory management.
    Supports adding, retrieving, and ranking memories with pluggable backend.
    """

    def __init__(self, backend: "MemoryBackend", embedding_func=None):
        """
        Initialize MemoryManager.
        
        Args:
            backend: Storage and indexing backend (e.g., FaissBackend).
            embedding_func: Optional callable(text) -> List[float] for embeddings.
                           If None, memories won't be embedded.
        """
        self.backend = backend
        self.embedding_func = embedding_func
        self.memories: Dict[str, MemoryEntry] = {}

    def add(
        self,
        text: str,
        mem_type: MemoryType = MemoryType.EPISODIC,
        importance_score: float = 0.5,
        source: str = "conversation",
        tags: List[str] = None,
        user_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Add a new memory entry.
        
        Args:
            text: Memory content.
            mem_type: Type of memory.
            importance_score: Importance (0-1); higher = more important.
            source: Where the memory originated.
            tags: Optional tags for filtering.
            user_id: Optional user ID for multi-user contexts.
            ttl_seconds: Optional time-to-live in seconds.
            metadata: Optional metadata dict.
        
        Returns:
            Memory ID.
        """
        mem_id = str(uuid.uuid4())
        now = datetime.now()

        # Compute embedding if available
        embedding = None
        if self.embedding_func:
            try:
                embedding = self.embedding_func(text)
            except Exception as e:
                logger.warning(f"Embedding failed for memory {mem_id}: {e}")

        entry = MemoryEntry(
            id=mem_id,
            text=text,
            type=mem_type,
            timestamp=now,
            embedding=embedding,
            importance_score=importance_score,
            source=source,
            tags=tags or [],
            user_id=user_id,
            metadata=metadata or {},
            ttl_seconds=ttl_seconds,
        )

        self.memories[mem_id] = entry
        self.backend.index(entry)
        logger.debug(f"Added memory {mem_id} (type={mem_type})")
        return mem_id

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        mem_types: Optional[List[MemoryType]] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> List[MemoryEntry]:
        """
        Retrieve memories most relevant to a query.
        
        Uses hybrid ranking: semantic similarity + recency + importance.
        
        Args:
            query: Query text.
            top_k: Number of results to return.
            mem_types: Optional filter to specific memory types.
            user_id: Optional filter by user.
            tags: Optional filter by tags (any match).
            time_range: Optional (start, end) datetime range.
        
        Returns:
            List of MemoryEntry sorted by relevance.
        """
        # Get query embedding if available
        query_embedding = None
        if self.embedding_func:
            try:
                query_embedding = self.embedding_func(query)
            except Exception as e:
                logger.warning(f"Query embedding failed: {e}")

        # Search backend
        candidates = self.backend.search(
            query_embedding=query_embedding,
            top_k=top_k * 3,  # Over-fetch to apply filters
            mem_types=mem_types,
        )

        # Apply metadata filters and expiry checks
        filtered = [m for m in candidates if not m.is_expired()]

        if user_id:
            filtered = [m for m in filtered if m.user_id == user_id]

        if tags:
            filtered = [m for m in filtered if any(t in m.tags for t in tags)]

        if time_range:
            start, end = time_range
            filtered = [m for m in filtered if start <= m.timestamp <= end]

        # Re-rank with hybrid scoring
        def score(m: MemoryEntry) -> float:
            """Hybrid score: semantic + recency + importance."""
            # Semantic score (0-1)
            sem_score = 0.5 if query_embedding is None else self.backend.similarity(
                query_embedding, m.embedding
            )
            # Recency score (0-1)
            rec_score = m.recency_weight(decay_hours=24.0)
            # Importance (already 0-1)
            imp_score = m.importance_score

            # Weighted combination (tunable)
            alpha, beta, gamma = 0.5, 0.3, 0.2  # Weights
            return alpha * sem_score + beta * rec_score + gamma * imp_score

        filtered.sort(key=score, reverse=True)
        return filtered[:top_k]

    def update(
        self,
        mem_id: str,
        text: Optional[str] = None,
        importance_score: Optional[float] = None,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing memory entry.
        
        Returns:
            True if successful, False if memory not found.
        """
        if mem_id not in self.memories:
            logger.warning(f"Memory {mem_id} not found for update")
            return False

        entry = self.memories[mem_id]
        
        if text is not None:
            entry.text = text
            if self.embedding_func:
                try:
                    entry.embedding = self.embedding_func(text)
                except Exception as e:
                    logger.warning(f"Re-embedding failed for {mem_id}: {e}")

        if importance_score is not None:
            entry.importance_score = max(0.0, min(1.0, importance_score))

        if summary is not None:
            entry.summary = summary

        if tags is not None:
            entry.tags = tags

        if metadata is not None:
            entry.metadata.update(metadata)

        self.backend.index(entry)  # Re-index
        logger.debug(f"Updated memory {mem_id}")
        return True

    def delete(self, mem_id: str) -> bool:
        """
        Delete a memory entry.
        
        Returns:
            True if successful, False if not found.
        """
        if mem_id not in self.memories:
            return False
        del self.memories[mem_id]
        self.backend.delete(mem_id)
        logger.debug(f"Deleted memory {mem_id}")
        return True

    def summarize(
        self,
        mem_types: Optional[List[MemoryType]] = None,
        user_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        summarizer_func=None,
    ) -> Optional[MemoryEntry]:
        """
        Create a summary memory from a set of memories.
        
        Args:
            mem_types: Types to summarize.
            user_id: Optional user filter.
            time_range: Optional time range.
            summarizer_func: Callable(texts: List[str]) -> str for summarization.
                            If None, concatenates texts.
        
        Returns:
            New MemoryEntry of type LONG_TERM, or None if no memories to summarize.
        """
        candidates = [m for m in self.memories.values() if not m.is_expired()]

        if mem_types:
            candidates = [m for m in candidates if m.type in mem_types]

        if user_id:
            candidates = [m for m in candidates if m.user_id == user_id]

        if time_range:
            start, end = time_range
            candidates = [m for m in candidates if start <= m.timestamp <= end]

        if not candidates:
            logger.debug("No memories to summarize")
            return None

        texts = [m.text for m in candidates]

        if summarizer_func:
            try:
                summary_text = summarizer_func(texts)
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
                summary_text = " ".join(texts[:5])  # Fallback: first 5 texts
        else:
            # Default: concatenate with newlines
            summary_text = "\n".join(texts[:10])

        # Create summary memory
        summary_mem = self.add(
            text=summary_text,
            mem_type=MemoryType.LONG_TERM,
            importance_score=0.8,
            source="summarizer",
            tags=["summary"] + list(set(t for m in candidates for t in m.tags)),
            user_id=user_id,
            ttl_seconds=None,  # Summaries don't expire
            metadata={"summarized_count": len(candidates)},
        )

        logger.info(f"Created summary memory {summary_mem} from {len(candidates)} entries")
        return self.memories[summary_mem]

    def persist(self, filepath: str) -> None:
        """Save all memories to a JSON file."""
        data = {
            "memories": [m.to_dict() for m in self.memories.values()],
            "timestamp": datetime.now().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Persisted {len(self.memories)} memories to {filepath}")

    def load(self, filepath: str) -> None:
        """Load memories from a JSON file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            self.memories.clear()
            for mem_dict in data.get("memories", []):
                entry = MemoryEntry.from_dict(mem_dict)
                self.memories[entry.id] = entry
                self.backend.index(entry)
            logger.info(f"Loaded {len(self.memories)} memories from {filepath}")
        except FileNotFoundError:
            logger.debug(f"Memory file {filepath} not found; starting fresh")
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")

    def cleanup(self, max_age_hours: float = 72.0) -> int:
        """
        Remove expired memories and very old low-importance entries.
        
        Returns:
            Number of memories removed.
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_delete = [
            mem_id
            for mem_id, mem in self.memories.items()
            if mem.is_expired() or (mem.timestamp < cutoff and mem.importance_score < 0.3)
        ]
        for mem_id in to_delete:
            self.delete(mem_id)
        logger.info(f"Cleaned up {len(to_delete)} memories")
        return len(to_delete)

    def get_context_block(self, top_k: int = 5, query: Optional[str] = None) -> str:
        """
        Assemble a context block for prompt injection.
        
        Args:
            top_k: Number of top memories to include.
            query: Optional query to retrieve relevant memories.
        
        Returns:
            Formatted string suitable for injection into LLM prompt.
        """
        if query:
            hits = self.retrieve(query, top_k=top_k)
        else:
            # Return most recent non-expired memories
            hits = sorted(
                [m for m in self.memories.values() if not m.is_expired()],
                key=lambda m: m.timestamp,
                reverse=True,
            )[:top_k]

        if not hits:
            return ""

        context_lines = ["## Recent Context\n"]
        for i, mem in enumerate(hits, 1):
            context_lines.append(
                f"{i}. [{mem.type.value}] {mem.text[:100]}"
                f" (importance={mem.importance_score:.2f}, "
                f"tags={','.join(mem.tags) if mem.tags else 'none'})"
            )

        return "\n".join(context_lines)


class MemoryBackend:
    """Abstract base for memory storage and indexing backends."""

    def index(self, entry: MemoryEntry) -> None:
        """Index a memory entry."""
        raise NotImplementedError

    def search(
        self,
        query_embedding: Optional[List[float]],
        top_k: int,
        mem_types: Optional[List[MemoryType]] = None,
    ) -> List[MemoryEntry]:
        """
        Search for memories. Returns top-k candidates (not yet ranked).
        """
        raise NotImplementedError

    def delete(self, mem_id: str) -> None:
        """Delete a memory from index."""
        raise NotImplementedError

    def similarity(self, emb1: Optional[List[float]], emb2: Optional[List[float]]) -> float:
        """Compute similarity between two embeddings (0-1)."""
        raise NotImplementedError
