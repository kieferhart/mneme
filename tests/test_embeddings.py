"""Unit tests for Mneme embeddings (sentence-transformers + hybrid retrieval)."""

import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))

import pytest


class TestEmbedFunction:
    """Tests for the embeddings.embed helper."""

    def test_embed_returns_384_floats(self):
        from mneme.embeddings import embed, EMBEDDING_DIM
        vec = embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in vec)

    def test_embed_normalized(self):
        """all-MiniLM-L6-v2 vectors are L2-normalized → norm ≈ 1.0."""
        import math
        from mneme.embeddings import embed
        vec = embed("anything")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-3

    def test_embed_empty_string(self):
        from mneme.embeddings import embed, EMBEDDING_DIM
        vec = embed("")
        assert len(vec) == EMBEDDING_DIM


class TestCosine:
    """Tests for embeddings.cosine."""

    def test_identical_vectors(self):
        from mneme.embeddings import cosine
        # already normalized inputs
        v = [1.0, 0.0, 0.0]
        assert cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from mneme.embeddings import cosine
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_empty_vectors(self):
        from mneme.embeddings import cosine
        assert cosine([], [1.0, 2.0]) == 0.0

    def test_length_mismatch(self):
        from mneme.embeddings import cosine
        assert cosine([1.0, 0.0], [1.0]) == 0.0


class TestAddMemoryEmbedsEagerly:
    """add_memory should populate the embedding column on insert."""

    def test_embedding_stored_on_add(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        from mneme.utils import row_as_dict
        add_memory(conn, "Python Basics", "Introduction to Python programming")
        result = conn.execute(
            "MATCH (m:Memory {title: 'Python Basics'}) RETURN m.embedding AS e"
        )
        d = row_as_dict(result.get_next(), result.get_column_names())
        assert d["e"] is not None
        assert len(d["e"]) == 384


class TestBackfillEmbeddings:
    """Backfill is for legacy data without embeddings."""

    def test_backfill_is_noop_for_new_inserts(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory, backfill_embeddings
        add_memory(conn, "Python Basics", "Introduction to Python")
        add_memory(conn, "Cake Recipe", "How to bake a cake")
        # add_memory already embeds — backfill should find nothing to do.
        result = backfill_embeddings(conn)
        assert result["total"] == 0
        assert result["updated"] == 0

    def test_backfill_empty_db(self, conn):
        conn, db, path = conn
        from mneme.memory import backfill_embeddings
        result = backfill_embeddings(conn)
        assert result["total"] == 0
        assert result["updated"] == 0


class TestFindCandidateNodesWithEmbeddings:
    """Hybrid retrieval should beat pure keyword on synonym/paraphrase."""

    def test_finds_relevant_nodes(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        from mneme.traverse import find_candidate_nodes
        add_memory(conn, "Python Basics", "Introduction to Python programming language")
        add_memory(conn, "Cake Recipe", "How to bake a chocolate cake")
        add_memory(conn, "JavaScript Intro", "JavaScript for beginners tutorial")
        candidates = find_candidate_nodes(conn, "python programming", top_n=5)
        assert len(candidates) >= 1
        assert candidates[0]["title"] == "Python Basics"
        assert candidates[0]["overlap_score"] > 0

    def test_synonym_match_via_embeddings(self, conn):
        """Query with no shared keywords should still find the right doc."""
        conn, db, path = conn
        from mneme.memory import add_memory
        from mneme.traverse import find_candidate_nodes
        add_memory(conn, "Auto Repair", "How to fix a car engine")
        add_memory(conn, "Cake Recipe", "How to bake a chocolate cake")
        # No literal overlap with the doc title or body.
        candidates = find_candidate_nodes(conn, "vehicle motor maintenance", top_n=5)
        # Auto Repair should be the top hit even though no keywords overlap.
        assert candidates[0]["title"] == "Auto Repair"

    def test_hybrid_scoring_returns_all_scores(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        from mneme.traverse import find_candidate_nodes
        add_memory(conn, "Python Basics", "Introduction to Python")
        add_memory(conn, "Cake Recipe", "Bake a cake")
        candidates = find_candidate_nodes(conn, "python", top_n=5)
        assert len(candidates) >= 1
        assert "cosine_score" in candidates[0]
        assert "keyword_score" in candidates[0]
        assert "overlap_score" in candidates[0]

    def test_sorted_by_score(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        from mneme.traverse import find_candidate_nodes
        add_memory(conn, "Python", "python programming python")
        add_memory(conn, "Topic", "about topic")
        add_memory(conn, "Other", "completely unrelated stuff")
        candidates = find_candidate_nodes(conn, "python", top_n=5)
        assert len(candidates) >= 1
        assert candidates[0]["title"] == "Python"
        for i in range(len(candidates) - 1):
            assert candidates[i]["overlap_score"] >= candidates[i + 1]["overlap_score"]

    def test_limited_to_top_n(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        from mneme.traverse import find_candidate_nodes
        for i in range(10):
            add_memory(conn, f"Item {i}", f"This is about topic {i}")
        candidates = find_candidate_nodes(conn, "topic", top_n=3)
        assert len(candidates) <= 3
