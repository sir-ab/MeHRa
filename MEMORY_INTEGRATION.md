# Memory System Integration Guide

## Overview
The MeHRa memory system enables LLM contexts to persist and retrieve relevant information across conversations. Uses **Chroma DB** for local, persistent vector storage with automatic embeddings.

## Components

### 1. **core/memory.py** — Core Memory Management
- `MemoryEntry`: Represents a single memory with metadata, embeddings, timestamps, and TTL.
- `MemoryManager`: Main interface for adding, retrieving, updating, and summarizing memories.
- `MemoryType`: Enum for memory categories (SHORT_TERM, EPISODIC, SEMANTIC, WORKING, LONG_TERM).
- `MemoryBackend`: Abstract base for pluggable storage backends.

### 2. **core/memory_backends.py** — Chroma Backend
- `ChromaBackend`: Production-ready implementation using Chroma DB.
  - Automatic persistence to disk (`./memory_storage/` by default).
  - Built-in embeddings via HuggingFace (`all-MiniLM-L6-v2` by default).
  - Metadata filtering and hybrid similarity search.

### 3. **workshop/memory_demo.py** — Demo & Examples
- Run with `python workshop/memory_demo.py` to see all features in action.
- Examples: basic CRUD, filtering, summarization, context assembly.

### 4. **core/test_memory.py** — Unit Tests
- Run with `python -m pytest core/test_memory.py` (or `python -m unittest core.test_memory`).
- Tests for memory lifecycle, retrieval, filtering, and backend operations.

---

## Quick Start

### Installation
```bash
pip install chromadb numpy
```

### Basic Usage
```python
from core.memory import MemoryManager, MemoryType
from core.memory_backends import ChromaBackend

# Initialize backend and manager
backend = ChromaBackend(persist_dir="./memory_storage")
memory = MemoryManager(backend=backend)

# Add a memory
mem_id = memory.add(
    text="User likes Python and machine learning",
    mem_type=MemoryType.SEMANTIC,
    importance_score=0.8,
    tags=["user_preference", "tech"],
)

# Retrieve relevant memories
hits = memory.retrieve(
    query="What does the user prefer?",
    top_k=5,
)

# Build context for LLM prompt
context = memory.get_context_block(top_k=3, query="coding")
```

---

## Integration with Conversation.py

### Pattern 1: Auto-Store User Messages
Modify `Conversation.add_message()` to also store in memory:

```python
from core.memory import MemoryManager, MemoryType
from core.memory_backends import ChromaBackend

class Conversation:
    def __init__(self):
        self.messages = []
        # Initialize memory system
        self.memory = MemoryManager(
            backend=ChromaBackend(persist_dir="./memory_storage")
        )
    
    def add_message(self, role: str, content: str) -> None:
        """Add message and store in memory if user/assistant."""
        if self.messages and self.messages[-1].role == role:
            self.messages[-1].content += " " + content
        else:
            self.messages.append(Message(role=role, content=content))
        
        # Store user/important content in memory
        if role == "user":
            self.memory.add(
                text=content,
                mem_type=MemoryType.EPISODIC,
                importance_score=0.6,
                source="user_message",
            )
        elif role == "assistant":
            # Optionally store assistant summaries
            if len(content) > 100:  # Only store substantial responses
                self.memory.add(
                    text=content[:200],  # Truncate for efficiency
                    mem_type=MemoryType.WORKING,
                    importance_score=0.4,
                    source="assistant_response",
                )
```

### Pattern 2: Inject Memories into LLM Context
Before calling the LLM, retrieve relevant memories:

```python
def get_history(self, include_memory=True) -> List[Dict[str, str]]:
    """Get conversation history, optionally with memory context."""
    history = [{"role": msg.role, "content": msg.content} for msg in self.messages]
    
    if include_memory and self.messages:
        # Get last user message as query
        last_user_msg = next(
            (msg.content for msg in reversed(self.messages) if msg.role == "user"),
            None
        )
        if last_user_msg:
            # Retrieve relevant memories
            hits = self.memory.retrieve(query=last_user_msg, top_k=3)
            if hits:
                # Inject as system message
                context_block = self.memory.get_context_block(top_k=3, query=last_user_msg)
                history.insert(0, {
                    "role": "system",
                    "content": context_block
                })
    
    return history
```

### Pattern 3: Periodic Memory Summarization
Clean up old memories and summarize them:

```python
def periodic_cleanup(self, interval_messages: int = 50) -> None:
    """Run memory cleanup and summarization every N messages."""
    if len(self.messages) % interval_messages == 0:
        # Clean up expired memories
        removed = self.memory.cleanup(max_age_hours=72.0)
        
        # Summarize episodic memories older than 1 week
        from datetime import datetime, timedelta
        week_ago = datetime.now() - timedelta(days=7)
        self.memory.summarize(
            mem_types=[MemoryType.EPISODIC],
            time_range=(None, week_ago),  # Will need adjustment for filtering
        )
        
        print(f"Memory cleanup: removed {removed} expired memories")
```

---

## Memory Types & Use Cases

| Type | Purpose | TTL | Example |
|------|---------|-----|---------|
| **SHORT_TERM** | Current conversation window | 1 hour | Immediate context |
| **EPISODIC** | Timestamped events/utterances | 30 days | "User asked about X" |
| **SEMANTIC** | Facts & persistent preferences | Indefinite | "User likes Python" |
| **WORKING** | Intermediate reasoning state | 1 day | "Currently discussing Y" |
| **LONG_TERM** | Compressed summaries | Indefinite | "Summary: Topics covered Z" |

---

## Retrieval & Ranking

**Hybrid Score** combines:
- **Semantic Similarity** (50%): Vector similarity to query.
- **Recency** (30%): How recent the memory is (exponential decay over 24h).
- **Importance** (20%): User-assigned importance score (0-1).

Tune weights in `MemoryManager.retrieve()`:
```python
alpha, beta, gamma = 0.6, 0.25, 0.15  # Your custom weights
```

---

## Configuration & Performance

### Storage Location
Set custom persist directory:
```python
backend = ChromaBackend(persist_dir="/custom/path")
```

### Embedding Model
Chroma uses `all-MiniLM-L6-v2` by default (384-dim, fast). For other models:
```python
# Currently Chroma auto-selects; custom embedding requires extending ChromaBackend
```

### Performance Tips
1. **Limit context window**: Retrieve only top-K (default 5-10).
2. **Tag memories**: Use tags for fast filtering.
3. **Batch operations**: Summarize in bulk rather than per-message.
4. **TTL settings**: Short TTL for transient memories, long for semantic facts.

---

## Testing
```bash
# Run all tests
python -m pytest core/test_memory.py -v

# Run specific test
python -m pytest core/test_memory.py::TestMemoryManager::test_retrieve_basic -v
```

---

## Example Full Integration
```python
from core.conversation import Conversation
from core.memory import MemoryManager, MemoryType
from core.memory_backends import ChromaBackend

# Initialize conversation with memory
conv = Conversation()
conv.memory = MemoryManager(backend=ChromaBackend())

# Add user message (auto-stored in memory)
conv.add_message("user", "I love building AI projects with Python")

# Later, get enriched history for LLM
context_block = conv.memory.get_context_block(top_k=3, query="Python")
enriched_history = conv.get_history(include_memory=True)

# LLM sees:
# - System: "## Recent Context\n1. [episodic] I love building AI projects..."
# - User/Assistant messages
```

---

## Privacy & Data Handling

- **Local-first**: All memories stored locally in `./memory_storage/`.
- **Encryption**: Add SQLite encryption by configuring Chroma settings if needed.
- **Retention**: Configure TTL per memory type for automatic cleanup.
- **Redaction**: Before storing sensitive data, pass through redaction filter.

---

## Future Enhancements

1. **LLM Summarization**: Use actual LLM for better summaries.
2. **Semantic Clustering**: Group similar memories and deduplicate.
3. **User Multi-Tenancy**: Extend metadata to support multiple users with isolation.
4. **Feedback Loop**: Learn importance scores from user reactions.
5. **Multi-Modal**: Store embeddings for images, audio, etc.

---

## Troubleshooting

**Q: Memory not found after restart?**  
A: Ensure `ChromaBackend(persist_dir=...)` points to same directory.

**Q: Chroma embedding errors?**  
A: Verify internet connection (HuggingFace downloads model on first run). Alternatively, pre-download the model.

**Q: Context block is empty?**  
A: Check that memories exist and haven't expired. Use `backend.get_stats()` to verify.

**Q: Slow retrieval?**  
A: Reduce `top_k`, add more specific filters (tags, user_id), or increase TTL to shrink collection.

---

## References
- [Chroma Docs](https://docs.trychroma.com/)
- [Memory System Design](../core/memory.py)
- [Demo Script](../workshop/memory_demo.py)
