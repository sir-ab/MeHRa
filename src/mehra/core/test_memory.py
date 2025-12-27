"""
Unit tests for the memory system.
Tests MemoryManager and ChromaBackend functionality.
"""

import unittest
from datetime import datetime, timedelta
from .memory import MemoryManager, MemoryType, MemoryEntry
from .memory_backends import ChromaBackend
import tempfile
import shutil


class TestMemoryEntry(unittest.TestCase):
    """Test MemoryEntry functionality."""

    def test_memory_entry_creation(self):
        """Test creating and serializing a memory entry."""
        entry = MemoryEntry(
            id="test_1",
            text="Test memory",
            type=MemoryType.EPISODIC,
            timestamp=datetime.now(),
            importance_score=0.8,
        )
        self.assertEqual(entry.id, "test_1")
        self.assertEqual(entry.type, MemoryType.EPISODIC)
        self.assertEqual(entry.importance_score, 0.8)

    def test_memory_expiry(self):
        """Test TTL expiration logic."""
        now = datetime.now()
        # Entry that expired
        old_entry = MemoryEntry(
            id="old",
            text="Old memory",
            type=MemoryType.EPISODIC,
            timestamp=now - timedelta(seconds=200),
            ttl_seconds=100,
        )
        self.assertTrue(old_entry.is_expired())

        # Entry that hasn't expired
        new_entry = MemoryEntry(
            id="new",
            text="New memory",
            type=MemoryType.EPISODIC,
            timestamp=now,
            ttl_seconds=100,
        )
        self.assertFalse(new_entry.is_expired())

    def test_recency_weight(self):
        """Test recency weight computation."""
        now = datetime.now()
        entry = MemoryEntry(
            id="test",
            text="Test",
            type=MemoryType.EPISODIC,
            timestamp=now,
        )
        # Fresh entry should have high weight
        weight = entry.recency_weight(decay_hours=24.0)
        self.assertGreater(weight, 0.99)

        # Old entry should have low weight
        old_entry = MemoryEntry(
            id="old",
            text="Old",
            type=MemoryType.EPISODIC,
            timestamp=now - timedelta(days=10),
        )
        old_weight = old_entry.recency_weight(decay_hours=24.0)
        self.assertLess(old_weight, 0.01)

    def test_memory_serialization(self):
        """Test serializing and deserializing memories."""
        entry = MemoryEntry(
            id="test",
            text="Test memory",
            type=MemoryType.SEMANTIC,
            timestamp=datetime.now(),
            tags=["test", "demo"],
            importance_score=0.75,
        )
        # Serialize
        data = entry.to_dict()
        self.assertIsInstance(data["timestamp"], str)
        self.assertEqual(data["type"], "semantic")

        # Deserialize
        restored = MemoryEntry.from_dict(data)
        self.assertEqual(restored.id, entry.id)
        self.assertEqual(restored.text, entry.text)
        self.assertEqual(restored.type, entry.type)


class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager functionality."""

    def setUp(self):
        """Create a temporary directory for test storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = ChromaBackend(persist_dir=self.temp_dir)
        self.manager = MemoryManager(backend=self.backend)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_memory(self):
        """Test adding memories."""
        mem_id = self.manager.add(
            text="Test memory",
            mem_type=MemoryType.EPISODIC,
            importance_score=0.7,
        )
        self.assertIsNotNone(mem_id)
        self.assertIn(mem_id, self.manager.memories)

    def test_add_multiple_types(self):
        """Test adding different memory types."""
        types = [
            MemoryType.SHORT_TERM,
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.WORKING,
            MemoryType.LONG_TERM,
        ]
        for mem_type in types:
            mem_id = self.manager.add(
                text=f"Test {mem_type.value}",
                mem_type=mem_type,
            )
            self.assertEqual(self.manager.memories[mem_id].type, mem_type)

    def test_retrieve_basic(self):
        """Test basic retrieval."""
        self.manager.add("Python is great", mem_type=MemoryType.EPISODIC)
        self.manager.add("Java is verbose", mem_type=MemoryType.EPISODIC)

        # Retrieve all
        results = self.manager.retrieve("Python", top_k=10)
        self.assertGreater(len(results), 0)

    def test_retrieve_with_filters(self):
        """Test retrieval with type and tag filters."""
        id1 = self.manager.add(
            "User likes Python",
            mem_type=MemoryType.SEMANTIC,
            tags=["preference"],
        )
        id2 = self.manager.add(
            "User was coding",
            mem_type=MemoryType.EPISODIC,
            tags=["activity"],
        )

        # Filter by type
        semantic_only = self.manager.retrieve(
            "Python", top_k=10, mem_types=[MemoryType.SEMANTIC]
        )
        types = [m.type for m in semantic_only]
        self.assertTrue(all(t == MemoryType.SEMANTIC for t in types))

        # Filter by tag
        tagged = self.manager.retrieve(
            "User", top_k=10, tags=["preference"]
        )
        self.assertGreater(len(tagged), 0)

    def test_update_memory(self):
        """Test updating a memory."""
        mem_id = self.manager.add("Original text", importance_score=0.5)
        self.assertTrue(
            self.manager.update(mem_id, importance_score=0.9)
        )
        self.assertEqual(self.manager.memories[mem_id].importance_score, 0.9)

    def test_delete_memory(self):
        """Test deleting a memory."""
        mem_id = self.manager.add("To be deleted")
        self.assertIn(mem_id, self.manager.memories)
        self.assertTrue(self.manager.delete(mem_id))
        self.assertNotIn(mem_id, self.manager.memories)

    def test_summarize(self):
        """Test memory summarization."""
        self.manager.add("Fact 1", mem_type=MemoryType.EPISODIC)
        self.manager.add("Fact 2", mem_type=MemoryType.EPISODIC)
        self.manager.add("Fact 3", mem_type=MemoryType.EPISODIC)

        def simple_summarizer(texts):
            return f"Summary of {len(texts)} items"

        summary = self.manager.summarize(
            mem_types=[MemoryType.EPISODIC],
            summarizer_func=simple_summarizer,
        )
        self.assertIsNotNone(summary)
        self.assertEqual(summary.type, MemoryType.LONG_TERM)

    def test_cleanup(self):
        """Test cleanup of expired memories."""
        # Add a memory with short TTL
        self.manager.add("Expired", ttl_seconds=1)
        self.assertEqual(len(self.manager.memories), 1)

        # Wait and cleanup
        import time
        time.sleep(1.5)
        removed = self.manager.cleanup(max_age_hours=0.0001)
        self.assertGreater(removed, 0)

    def test_context_block(self):
        """Test context block generation."""
        self.manager.add("Memory 1", mem_type=MemoryType.EPISODIC)
        self.manager.add("Memory 2", mem_type=MemoryType.SEMANTIC)

        context = self.manager.get_context_block(top_k=2)
        self.assertIsInstance(context, str)
        self.assertIn("Recent Context", context)

    def test_persistence(self):
        """Test save/load functionality."""
        mem_id = self.manager.add("Persistent memory", importance_score=0.8)

        persist_file = f"{self.temp_dir}/memories.json"
        self.manager.persist(persist_file)

        # Create new manager and load
        new_manager = MemoryManager(backend=self.backend)
        new_manager.load(persist_file)

        self.assertIn(mem_id, new_manager.memories)
        self.assertEqual(
            new_manager.memories[mem_id].importance_score, 0.8
        )


class TestChromaBackend(unittest.TestCase):
    """Test ChromaBackend functionality."""

    def setUp(self):
        """Create temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = ChromaBackend(persist_dir=self.temp_dir)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_index_and_search(self):
        """Test indexing and searching."""
        entry = MemoryEntry(
            id="test_1",
            text="Test memory content",
            type=MemoryType.EPISODIC,
            timestamp=datetime.now(),
        )
        self.backend.index(entry)

        results = self.backend.search(query_embedding=None, top_k=10)
        self.assertGreater(len(results), 0)

    def test_delete(self):
        """Test deletion."""
        entry = MemoryEntry(
            id="test_delete",
            text="To be deleted",
            type=MemoryType.EPISODIC,
            timestamp=datetime.now(),
        )
        self.backend.index(entry)
        self.backend.delete("test_delete")

        results = self.backend.search(query_embedding=None, top_k=10)
        ids = [r.id for r in results]
        self.assertNotIn("test_delete", ids)

    def test_similarity(self):
        """Test similarity computation."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        sim = self.backend.similarity(emb1, emb2)
        self.assertAlmostEqual(sim, 1.0, places=5)

        emb3 = [0.0, 1.0, 0.0]
        sim2 = self.backend.similarity(emb1, emb3)
        self.assertAlmostEqual(sim2, 0.5, places=5)

    def test_stats(self):
        """Test getting backend stats."""
        stats = self.backend.get_stats()
        self.assertIn("memory_count", stats)
        self.assertIn("persist_dir", stats)


if __name__ == "__main__":
    unittest.main()
