"""
Memory system demo and integration example.
Shows how to use MemoryManager with Chroma backend in a conversation context.
"""

import logging
from datetime import datetime, timedelta

from core.memory import MemoryManager, MemoryType
from core.memory_backends import ChromaBackend

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def demo_basic_usage():
    """Demonstrate basic memory operations."""
    print("\n=== Basic Memory Operations ===\n")

    # Initialize backend and manager
    backend = ChromaBackend(persist_dir="./memory_storage")
    memory = MemoryManager(backend=backend)

    # Add some episodic memories (events/utterances)
    mem1 = memory.add(
        text="User said they like python and machine learning",
        mem_type=MemoryType.EPISODIC,
        importance_score=0.8,
        tags=["user_preference", "tech"],
    )
    print(f"Added memory 1: {mem1}")

    mem2 = memory.add(
        text="User solved a bug in the conversation module last week",
        mem_type=MemoryType.EPISODIC,
        importance_score=0.6,
        tags=["user_activity", "coding"],
    )
    print(f"Added memory 2: {mem2}")

    mem3 = memory.add(
        text="The LLM should remember context across sessions for better continuity",
        mem_type=MemoryType.SEMANTIC,
        importance_score=0.9,
        tags=["system_knowledge", "design"],
    )
    print(f"Added memory 3: {mem3}")

    # Retrieve relevant memories
    print("\n--- Retrieval: Query 'python and coding' ---")
    hits = memory.retrieve(
        query="python and coding projects",
        top_k=3,
    )
    for i, hit in enumerate(hits, 1):
        print(
            f"{i}. [{hit.type.value}] {hit.text}\n"
            f"   Importance: {hit.importance_score:.2f}, Tags: {hit.tags}\n"
        )

    # Get context block for prompt injection
    print("\n--- Context Block for LLM Prompt ---")
    context = memory.get_context_block(top_k=2, query="coding")
    print(context)

    # Update a memory
    print("\n--- Updating Memory ---")
    memory.update(mem1, importance_score=0.95)
    print(f"Updated memory {mem1} importance to 0.95")

    # Show backend stats
    print("\n--- Backend Stats ---")
    stats = backend.get_stats()
    print(f"Memories stored: {stats.get('memory_count', 0)}")

    return memory, backend


def demo_filtering():
    """Demonstrate filtering by type, user, tags."""
    print("\n=== Filtering Examples ===\n")

    backend = ChromaBackend(persist_dir="./memory_storage_demo2")
    memory = MemoryManager(backend=backend)

    # Add memories with different types and users
    memory.add(
        text="User A likes coffee",
        mem_type=MemoryType.SEMANTIC,
        user_id="user_a",
        tags=["preference"],
    )
    memory.add(
        text="User B prefers tea",
        mem_type=MemoryType.SEMANTIC,
        user_id="user_b",
        tags=["preference"],
    )
    memory.add(
        text="Conversation turn: User A discussed machine learning",
        mem_type=MemoryType.EPISODIC,
        user_id="user_a",
        tags=["conversation"],
    )

    # Filter by user
    print("--- Memories for user_a ---")
    hits = memory.retrieve("", top_k=10, user_id="user_a")
    for hit in hits:
        print(f"  {hit.text}")

    # Filter by type
    print("\n--- Only SEMANTIC memories ---")
    hits = memory.retrieve("preference", top_k=10, mem_types=[MemoryType.SEMANTIC])
    for hit in hits:
        print(f"  {hit.text}")

    backend.clear()


def demo_summarization():
    """Demonstrate memory summarization."""
    print("\n=== Summarization Example ===\n")

    backend = ChromaBackend(persist_dir="./memory_storage_demo3")
    memory = MemoryManager(backend=backend)

    # Add a series of episodic memories
    for i in range(5):
        memory.add(
            text=f"Conversation turn {i+1}: User discussed topic {i}",
            mem_type=MemoryType.EPISODIC,
            importance_score=0.5,
        )

    print(f"Created 5 episodic memories")

    # Simple summarizer: just concatenate
    def simple_summarizer(texts):
        return f"Summary of conversation: {len(texts)} turns discussed topics {0}-{len(texts)-1}"

    summary = memory.summarize(
        mem_types=[MemoryType.EPISODIC],
        summarizer_func=simple_summarizer,
    )

    if summary:
        print(f"\nSummary created:")
        print(f"  ID: {summary.id}")
        print(f"  Text: {summary.text}")
        print(f"  Type: {summary.type.value}")

    backend.clear()


def demo_context_assembly():
    """Demonstrate assembling context for LLM prompts."""
    print("\n=== Context Assembly for Prompts ===\n")

    backend = ChromaBackend(persist_dir="./memory_storage_demo4")
    memory = MemoryManager(backend=backend)

    # Simulate a multi-turn conversation
    memory.add(
        text="User asked about Python best practices",
        mem_type=MemoryType.EPISODIC,
        importance_score=0.7,
        tags=["python", "bestpractices"],
    )
    memory.add(
        text="Discussed async/await patterns for concurrent code",
        mem_type=MemoryType.EPISODIC,
        importance_score=0.6,
        tags=["python", "async"],
    )
    memory.add(
        text="User prefers type-hinted code",
        mem_type=MemoryType.SEMANTIC,
        importance_score=0.8,
        tags=["user_preference"],
    )

    # New user query
    user_query = "How should I structure my async functions?"

    print(f"User query: {user_query}\n")

    # Get context block
    context = memory.get_context_block(top_k=3, query=user_query)
    print("Injected context:")
    print(context)

    # Show how to use in a full prompt
    print("\n--- Full Prompt Template ---")
    full_prompt = f"""<system>
You are a helpful coding assistant. Use the following context about the user to provide better responses.

{context}
</system>

<user>
{user_query}
</user>

<assistant>
"""
    print(full_prompt)

    backend.clear()


if __name__ == "__main__":
    demo_basic_usage()
    demo_filtering()
    demo_summarization()
    demo_context_assembly()

    print("\n✓ All demos completed successfully!")
