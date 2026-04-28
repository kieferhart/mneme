"""Unit tests for Mneme core modules."""

import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))

import pytest


def _setup_link_memories(conn):
    """Setup for TestLinkMemories tests."""
    from mneme.memory import add_memory, link_memories, get_neighbors, show_memory
    add_memory(conn, "Source", "Source body")
    add_memory(conn, "Target", "Target body")
    return add_memory, link_memories, get_neighbors, show_memory


def _setup_reward_edge(conn):
    """Setup for TestRewardEdge tests."""
    from mneme.memory import add_memory, link_memories, get_neighbors
    from mneme.learn import reward_edge
    add_memory(conn, "Source", "Source body")
    add_memory(conn, "Target1", "Target 1 body")
    add_memory(conn, "Target2", "Target 2 body")
    link_memories(conn, "Source", "Target1", kind="relates_to")
    link_memories(conn, "Source", "Target2", kind="relates_to")
    return add_memory, link_memories, get_neighbors, reward_edge


class TestAddMemory:
    def test_add_single_memory(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        mem = add_memory(conn, "Test Topic", "Test body text here.", kind="note")
        assert mem["id"] == "test-topic"
        assert mem["title"] == "Test Topic"
        assert mem["body"] == "Test body text here."
        assert mem["kind"] == "note"
        assert mem["created_at"] is not None
        assert mem["updated_at"] is not None

    def test_add_multiple_memories(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        m1 = add_memory(conn, "Alpha", "Body A")
        m2 = add_memory(conn, "Beta", "Body B")
        assert m1["id"] == "alpha"
        assert m2["id"] == "beta"

    def test_add_memory_default_kind(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        mem = add_memory(conn, "Default Kind Test", "Body")
        assert mem["kind"] == "note"


class TestLinkMemories:
    def test_link_relates_to(self, conn):
        conn, db, path = conn
        add_memory, link_memories, get_neighbors, show_memory = _setup_link_memories(conn)
        link_memories(conn, "Source", "Target", kind="relates_to")
        neighbors = get_neighbors(conn, "Source")
        assert len(neighbors) == 1
        assert neighbors[0]["target_title"] == "Target"
        assert neighbors[0]["kind"] == "relates_to"

    def test_link_analogous_to(self, conn):
        conn, db, path = conn
        add_memory, link_memories, get_neighbors, show_memory = _setup_link_memories(conn)
        link_memories(conn, "Source", "Target", kind="analogous_to", reason="They are similar")
        neighbors = get_neighbors(conn, "Source")
        assert len(neighbors) == 1
        assert neighbors[0]["target_title"] == "Target"
        assert neighbors[0]["kind"] == "analogous_to"
        assert neighbors[0]["reason"] == "They are similar"

    def test_link_with_default_weights(self, conn):
        conn, db, path = conn
        add_memory, link_memories, get_neighbors, show_memory = _setup_link_memories(conn)
        link_memories(conn, "Source", "Target", kind="relates_to")
        neighbors = get_neighbors(conn, "Source")
        assert neighbors[0]["accuracy_weight"] == pytest.approx(1.0)
        assert neighbors[0]["creative_weight"] == pytest.approx(1.0)

    def test_get_neighbors(self, conn):
        conn, db, path = conn
        add_memory, link_memories, get_neighbors, show_memory = _setup_link_memories(conn)
        link_memories(conn, "Source", "Target", kind="relates_to")
        neighbors = get_neighbors(conn, "Source")
        assert len(neighbors) == 1
        assert neighbors[0]["target_title"] == "Target"

    def test_get_neighbors_no_links(self, conn):
        conn, db, path = conn
        add_memory, link_memories, get_neighbors, show_memory = _setup_link_memories(conn)
        neighbors = get_neighbors(conn, "Source")
        assert len(neighbors) == 0

    def test_show_memory(self, conn):
        conn, db, path = conn
        add_memory, link_memories, get_neighbors, show_memory = _setup_link_memories(conn)
        link_memories(conn, "Source", "Target", kind="relates_to")
        info = show_memory(conn, "Source")
        assert info["id"] == "source"
        assert info["title"] == "Source"
        assert len(info["outgoing_edges"]) == 1
        assert info["incoming_edges"] == []
        edge = info["outgoing_edges"][0]
        assert edge["kind"] == "relates_to"
        assert edge["target_title"] == "Target"


class TestRewardEdge:
    def test_reward_increases_weight(self, conn):
        conn, db, path = conn
        add_memory, link_memories, get_neighbors, reward_edge = _setup_reward_edge(conn)
        result = reward_edge(conn, "Source", "Target1", "RELATES_TO", weight_field="accuracy_weight", amount=5)
        assert result["before_accuracy_weight"] == pytest.approx(1.0)
        assert result["after_accuracy_weight"] == pytest.approx(6.0)

    def test_reward_normalizes_budget(self, conn):
        conn, db, path = conn
        add_memory, link_memories, get_neighbors, reward_edge = _setup_reward_edge(conn)
        reward_edge(conn, "Source", "Target1", "RELATES_TO", weight_field="accuracy_weight", amount=5)
        neighbors = get_neighbors(conn, "Source")
        total = sum(n["accuracy_weight"] for n in neighbors)
        assert abs(total - 100.0) < 0.01

    def test_reward_creative_weight(self, conn):
        conn, db, path = conn
        add_memory, link_memories, get_neighbors, reward_edge = _setup_reward_edge(conn)
        result = reward_edge(conn, "Source", "Target1", "RELATES_TO", weight_field="creative_weight", amount=3)
        assert result["after_creative_weight"] == pytest.approx(4.0)

    def test_reward_increases_counters(self, conn):
        conn, db, path = conn
        add_memory, link_memories, get_neighbors, reward_edge = _setup_reward_edge(conn)
        reward_edge(conn, "Source", "Target1", "RELATES_TO", weight_field="accuracy_weight", amount=5)
        neighbors = get_neighbors(conn, "Source")
        target1 = [n for n in neighbors if n["target_title"] == "Target1"][0]
        assert target1["use_count"] >= 1
        assert target1["useful_count"] >= 1

    def test_reward_unknown_relationship(self, conn):
        conn, db, path = conn
        add_memory, link_memories, get_neighbors, reward_edge = _setup_reward_edge(conn)
        with pytest.raises(ValueError, match="Relationship not found"):
            reward_edge(conn, "Source", "Nonexistent", "RELATES_TO", amount=5)


class TestSession:
    def test_create_session(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        add_memory(conn, "Memory A", "Body A")
        add_memory(conn, "Memory B", "Body B")
        from mneme.session import create_session
        session = create_session(conn, "Test query", "creative")
        assert session["user_query"] == "Test query"
        assert session["mode"] == "creative"
        assert session["created_at"] is not None

    def test_list_sessions(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        add_memory(conn, "Memory A", "Body A")
        add_memory(conn, "Memory B", "Body B")
        from mneme.session import create_session, list_sessions
        create_session(conn, "Query 1", "creative")
        create_session(conn, "Query 2", "accuracy")
        sessions = list_sessions(conn)
        assert len(sessions) == 2

    def test_session_memory_ids(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory
        add_memory(conn, "Memory A", "Body A")
        add_memory(conn, "Memory B", "Body B")
        from mneme.session import create_session, list_sessions
        session = create_session(conn, "Test", "creative")
        sessions = list_sessions(conn)
        assert session["id"] in [s["id"] for s in sessions]


class TestTraverse:
    def test_traverse(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories
        add_memory(conn, "Start", "Start body")
        add_memory(conn, "Middle", "Middle body")
        add_memory(conn, "End", "End body")
        link_memories(conn, "Start", "Middle", kind="relates_to")
        link_memories(conn, "Middle", "End", kind="relates_to")
        from mneme.traverse import traverse, find_candidate_nodes
        candidates = find_candidate_nodes(conn, "Start body test traversal")
        assert len(candidates) >= 1
        result = traverse(conn, candidates[0]["id"], "test traversal", mode="balanced", max_hops=3)
        assert len(result["path"]) >= 1
        assert len(result["edges"]) >= 1

    def test_traverse_empty(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories
        add_memory(conn, "Start", "Start body")
        add_memory(conn, "Middle", "Middle body")
        add_memory(conn, "End", "End body")
        link_memories(conn, "Start", "Middle", kind="relates_to")
        link_memories(conn, "Middle", "End", kind="relates_to")
        add_memory(conn, "Isolated", "No links here")
        from mneme.traverse import traverse
        result = traverse(conn, "isolated", "test", mode="balanced", max_hops=2)
        assert len(result["path"]) == 1
        assert len(result["edges"]) == 0


class TestSchema:
    def test_all_tables_exist(self, conn):
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories
        add_memory(conn, "Memory A", "Body A")
        add_memory(conn, "Memory B", "Body B")
        link_memories(conn, "Memory A", "Memory B", kind="relates_to")
        result = conn.execute("MATCH (m:Memory) RETURN m.id")
        rows = result.get_all()
        assert len(rows) == 2
        result = conn.execute("MATCH ()-[r:RELATES_TO]->() RETURN r.kind")
        edges = result.get_all()
        assert len(edges) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
