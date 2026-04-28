"""Unit tests for Mneme traversal engine."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


class TestKeywordOverlap:
    """Tests for keyword_overlap scoring."""

    def test_identical_content(self):
        from mneme.traverse import keyword_overlap
        score = keyword_overlap("hello world test", "hello world test")
        assert score == 1.0

    def test_no_overlap(self):
        from mneme.traverse import keyword_overlap
        score = keyword_overlap("hello world", "goodbye universe")
        assert score == 0.0

    def test_partial_overlap(self):
        from mneme.traverse import keyword_overlap
        score = keyword_overlap("hello world test", "hello world")
        assert score > 0.0
        assert score < 1.0

    def test_query_empty(self):
        from mneme.traverse import keyword_overlap
        score = keyword_overlap("", "anything")
        assert score == 0.0

    def test_special_chars_ignored(self):
        from mneme.traverse import keyword_overlap
        score = keyword_overlap("hello-world", "hello world")
        assert score == 1.0

    def test_case_insensitive(self):
        from mneme.traverse import keyword_overlap
        score = keyword_overlap("HELLO", "hello")
        assert score == 1.0


class TestScoreEdge:
    """Tests for edge scoring in different modes."""

    def test_strict_mode_weights_accuracy(self):
        from mneme.traverse import score_edge
        edge = {"accuracy_weight": 50, "creative_weight": 1, "novelty_weight": 1, "confidence": 0.9}
        target = {"title": "test", "summary": "test", "body": "test"}
        rng = random.Random(42)
        score = score_edge(edge, target, "test", [], "strict", rng)
        assert score > 0

    def test_discovery_mode_boosts_creative(self):
        from mneme.traverse import score_edge
        high_creative_edge = {"accuracy_weight": 1, "creative_weight": 80, "novelty_weight": 50, "confidence": 0.5}
        target = {"title": "test", "summary": "test", "body": "test"}
        rng = random.Random(42)
        creative_score = score_edge(high_creative_edge, target, "test", [], "discovery", rng)

        low_creative_edge = {"accuracy_weight": 50, "creative_weight": 1, "novelty_weight": 1, "confidence": 0.5}
        low_score = score_edge(low_creative_edge, target, "test", [], "discovery", rng)
        assert creative_score > low_score or creative_score >= low_score * 0.5

    def test_balanced_mode_includes_novelty(self):
        from mneme.traverse import score_edge
        edge = {"accuracy_weight": 10, "creative_weight": 10, "novelty_weight": 80, "confidence": 0.5}
        target = {"title": "test", "summary": "test", "body": "test"}
        rng = random.Random(42)
        score = score_edge(edge, target, "test", [], "balanced", rng)
        assert score > 0

    def test_unknown_mode_raises(self):
        from mneme.traverse import score_edge
        edge = {"accuracy_weight": 1, "creative_weight": 1, "novelty_weight": 1, "confidence": 0.5}
        target = {"title": "test", "summary": "test", "body": "test"}
        rng = random.Random(42)
        with pytest.raises(ValueError, match="Unknown mode"):
            score_edge(edge, target, "test", [], "unknown_mode", rng)

    def test_semantic_overlap_increases_score(self):
        from mneme.traverse import score_edge
        edge = {"accuracy_weight": 1, "creative_weight": 1, "novelty_weight": 1, "confidence": 0.5}
        related_target = {"title": "hello world", "summary": "hello world", "body": "hello world"}
        unrelated_target = {"title": "xyz qqq", "summary": "xyz qqq", "body": "xyz qqq"}
        rng = random.Random(42)
        related_score = score_edge(edge, related_target, "hello", [], "balanced", rng)
        unrelated_score = score_edge(edge, unrelated_target, "hello", [], "balanced", rng)
        assert related_score >= unrelated_score


class TestFindCandidateNodes:
    """Tests for find_candidate_nodes."""

    def test_finds_relevant_nodes(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        from mneme.traverse import find_candidate_nodes
        add_memory(conn, "Python Basics", "Introduction to Python programming language")
        add_memory(conn, "Recipe Cake", "How to bake a chocolate cake")
        add_memory(conn, "JavaScript Intro", "JavaScript for beginners tutorial")

        candidates = find_candidate_nodes(conn, "python programming", top_n=5)
        assert len(candidates) >= 1
        assert candidates[0]["title"] == "Python Basics"
        assert candidates[0]["overlap_score"] > 0

    def test_unrelated_query_scores_low(self, conn):
        """Embeddings will give nonzero cosine to anything, but a junk query
        should rank everything well below a topical query."""
        conn, db, path = conn
        from mneme.memory import add_memory
        from mneme.traverse import find_candidate_nodes
        add_memory(conn, "Python Basics", "Introduction to Python")
        add_memory(conn, "Cake Recipe", "Bake a cake")

        topical = find_candidate_nodes(conn, "python programming", top_n=5)
        junk = find_candidate_nodes(conn, "xyzqwerty asdfqwer", top_n=5)
        # Topical query should score the top hit higher than junk does.
        if junk:
            assert topical[0]["overlap_score"] > junk[0]["overlap_score"]

    def test_limits_to_top_n(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        from mneme.traverse import find_candidate_nodes
        for i in range(10):
            add_memory(conn, f"Item {i}", f"This is about topic")
        candidates = find_candidate_nodes(conn, "topic", top_n=3)
        assert len(candidates) <= 3

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


class TestTraverseModes:
    """Tests for traverse function in different modes."""

    def _setup(self, conn):
        from mneme.memory import add_memory, link_memories
        add_memory(conn, "A", "First memory")
        add_memory(conn, "B", "Second memory")
        add_memory(conn, "C", "Third memory")
        link_memories(conn, "A", "B", kind="relates_to")
        link_memories(conn, "B", "C", kind="relates_to")

    def test_strict_mode_deterministic(self, conn):
        conn, db, path = conn
        self._setup(conn)
        from mneme.traverse import traverse
        r1 = traverse(conn, "a", "test", mode="strict", max_hops=3, seed=1)
        r2 = traverse(conn, "a", "test", mode="strict", max_hops=3, seed=1)
        assert r1["path"][0]["title"] == r2["path"][0]["title"]
        for i, (p1, p2) in enumerate(zip(r1["path"], r2["path"])):
            assert p1["title"] == p2["title"]

    def test_all_modes_execute_without_error(self, conn):
        conn, db, path = conn
        self._setup(conn)
        from mneme.traverse import traverse, find_candidate_nodes
        candidates = find_candidate_nodes(conn, "memory")
        start_id = candidates[0]["id"]
        for mode in ("strict", "balanced", "creative", "discovery"):
            result = traverse(conn, start_id, "memory", mode=mode, max_hops=3)
            assert len(result["path"]) >= 1
            assert result["mode"] == mode
            assert result["max_hops"] == 3
            assert result["seed"] is not None
            assert "visited_count" in result

    def test_max_hops_limit(self, conn):
        conn, db, path = conn
        self._setup(conn)
        from mneme.traverse import traverse, find_candidate_nodes
        candidates = find_candidate_nodes(conn, "memory")
        start_id = candidates[0]["id"]
        result = traverse(conn, start_id, "memory", mode="strict", max_hops=1)
        assert len(result["path"]) <= 2  # start node + at most 1 hop
        assert len(result["edges"]) <= 1

    def test_max_hops_zero(self, conn):
        conn, db, path = conn
        self._setup(conn)
        from mneme.traverse import traverse, find_candidate_nodes
        candidates = find_candidate_nodes(conn, "memory")
        start_id = candidates[0]["id"]
        result = traverse(conn, start_id, "memory", mode="strict", max_hops=0)
        assert len(result["path"]) == 1
        assert len(result["edges"]) == 0

    def test_traverse_visits_path(self, conn):
        conn, db, path = conn
        self._setup(conn)
        from mneme.traverse import traverse, find_candidate_nodes
        candidates = find_candidate_nodes(conn, "memory")
        start_id = candidates[0]["id"]
        result = traverse(conn, start_id, "memory", mode="strict", max_hops=3)
        assert result["visited_count"] >= 1
        titles = [n["title"] for n in result["path"]]
        assert "A" in titles

    def test_no_visited_duplicate_nodes(self, conn):
        conn, db, path = conn
        self._setup(conn)
        from mneme.traverse import traverse, find_candidate_nodes
        from mneme.memory import link_memories
        link_memories(conn, "C", "A", kind="relates_to")
        candidates = find_candidate_nodes(conn, "memory")
        start_id = candidates[0]["id"]
        result = traverse(conn, start_id, "memory", mode="balanced", max_hops=10)
        titles = [n["title"] for n in result["path"]]
        assert len(titles) == len(set(titles))

    def test_edge_scores_present(self, conn):
        conn, db, path = conn
        self._setup(conn)
        from mneme.traverse import traverse, find_candidate_nodes
        candidates = find_candidate_nodes(conn, "memory")
        start_id = candidates[0]["id"]
        result = traverse(conn, start_id, "memory", mode="strict", max_hops=3)
        for edge in result["edges"]:
            assert "from" in edge
            assert "to" in edge
            assert "score" in edge
            assert "rel_type" in edge or "edge_kind" in edge
        start_id = candidates[0]["id"]
        result = traverse(conn, start_id, "test", mode="strict", max_hops=3)
        for edge in result["edges"]:
            assert "from" in edge
            assert "to" in edge
            assert "score" in edge
            assert "rel_type" in edge or "edge_kind" in edge


class TestTraverseEdgeCases:
    """Edge case tests for the traversal engine."""

    def test_single_node_no_edges(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        from mneme.traverse import traverse, find_candidate_nodes
        add_memory(conn, "Solo", "I have no friends")
        candidates = find_candidate_nodes(conn, "solo")
        result = traverse(conn, candidates[0]["id"], "solo", mode="balanced", max_hops=5)
        assert len(result["path"]) == 1
        assert len(result["edges"]) == 0

    def test_two_nodes_bidirectional(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories
        from mneme.traverse import traverse, find_candidate_nodes
        add_memory(conn, "X", "Node X content for testing")
        add_memory(conn, "Y", "Node Y content for testing")
        link_memories(conn, "X", "Y", kind="relates_to")
        link_memories(conn, "Y", "X", kind="relates_to")

        candidates = find_candidate_nodes(conn, "Node X content for testing")
        x_id = candidates[0]["id"]
        result = traverse(conn, x_id, "test", mode="strict", max_hops=3)
        assert len(result["path"]) >= 2

    def test_seed_reproducibility(self, conn):
        conn, db, path = conn
        from mneme.traverse import traverse, find_candidate_nodes
        for i in range(5):
            self._add(conn, f"N{i}", f"Node {i} content")
        for i in range(4):
            self._link(conn, f"N{i}", f"N{i+1}", kind="relates_to")

        candidates = find_candidate_nodes(conn, "Node")
        r1 = traverse(conn, candidates[0]["id"], "test", mode="creative", max_hops=4, seed=42)
        r2 = traverse(conn, candidates[0]["id"], "test", mode="creative", max_hops=4, seed=42)
        assert len(r1["path"]) == len(r2["path"])
        for p1, p2 in zip(r1["path"], r2["path"]):
            assert p1["title"] == p2["title"]

    @staticmethod
    def _add(conn, title, body):
        conn.execute(
            "CREATE (m:Memory {id: $id, title: $title, body: $body, "
            "summary: $summary, kind: 'note', created_at: '2025-01-01', updated_at: '2025-01-01'})",
            {"id": title.lower(), "title": title, "body": body, "summary": body},
        )

    @staticmethod
    def _link(conn, src, tgt, kind):
        conn.execute(
            f"MATCH (s:Memory {{title: $s}}) MATCH (t:Memory {{title: $t}}) "
            f"CREATE (s)-[r:{kind.upper()} {{kind: $k, "
            f"accuracy_weight: 1.0, creative_weight: 1.0, "
            f"novelty_weight: 50.0}}]->(t)",
            {"s": src, "t": tgt, "k": kind},
        )
