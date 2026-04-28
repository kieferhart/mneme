"""Unit tests for Mneme learning/reward module."""

import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))

import pytest


def _make_test_data(conn):
    """Helper: add Source, Target1-3, and link them."""
    from mneme.memory import add_memory, link_memories
    add_memory(conn, "Source", "Source body")
    add_memory(conn, "Target1", "Target 1 body")
    add_memory(conn, "Target2", "Target 2 body")
    add_memory(conn, "Target3", "Target 3 body")
    link_memories(conn, "Source", "Target1", kind="relates_to")
    link_memories(conn, "Source", "Target2", kind="relates_to")
    link_memories(conn, "Source", "Target3", kind="relates_to")


def _make_mixed_rel_data(conn):
    """Helper: Source with two targets of different rel types."""
    from mneme.memory import add_memory, link_memories
    add_memory(conn, "Source", "Source body")
    add_memory(conn, "TargetA", "Target A body")
    add_memory(conn, "TargetB", "Target B body")
    link_memories(conn, "Source", "TargetA", kind="relates_to")
    link_memories(conn, "Source", "TargetB", kind="analogous_to")


def _make_single_edge_data(conn):
    """Helper: single source with one edge."""
    from mneme.memory import add_memory, link_memories
    add_memory(conn, "SoloSource", "Solo source")
    add_memory(conn, "OnlyTarget", "Only target")
    link_memories(conn, "SoloSource", "OnlyTarget", kind="relates_to")


def _make_reward_error_data(conn):
    """Helper: source+target with one edge (for error tests)."""
    from mneme.memory import add_memory, link_memories
    add_memory(conn, "Source", "Source body")
    add_memory(conn, "Target", "Target body")
    link_memories(conn, "Source", "Target", kind="relates_to")


class TestRewardEdgeBudgetNormalization:
    """Tests for the fixed-budget reward mechanism."""

    def test_three_edges_budget_normalization(self, conn):
        """With 3 edges, rewarding one should keep total budget at 100."""
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories, get_neighbors
        from mneme.learn import reward_edge
        _make_test_data(conn)
        reward_edge(conn, "Source", "Target1", "RELATES_TO", amount=10)
        neighbors = get_neighbors(conn, "Source")
        total = sum(n["accuracy_weight"] for n in neighbors)
        assert abs(total - 100.0) < 0.01

    def test_rewarded_edge_becomes_largest(self, conn):
        """The rewarded edge should end up with the highest weight."""
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories, get_neighbors
        from mneme.learn import reward_edge
        _make_test_data(conn)
        reward_edge(conn, "Source", "Target1", "RELATES_TO", amount=60)
        neighbors = get_neighbors(conn, "Source")
        target1 = [n for n in neighbors if n["target_title"] == "Target1"][0]
        others = [n for n in neighbors if n["target_title"] != "Target1"]
        for n in others:
            assert target1["accuracy_weight"] > n["accuracy_weight"]

    def test_rewarded_edge_weight_increases(self, conn):
        """The rewarded edge should gain the reward amount."""
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories, get_neighbors
        from mneme.learn import reward_edge
        _make_test_data(conn)
        neighbors_before = get_neighbors(conn, "Source")
        t1_before = [n for n in neighbors_before if n["target_title"] == "Target1"][0]
        reward_edge(conn, "Source", "Target1", "RELATES_TO", amount=10)
        neighbors_after = get_neighbors(conn, "Source")
        t1_after = [n for n in neighbors_after if n["target_title"] == "Target1"][0]
        assert t1_after["accuracy_weight"] > t1_before["accuracy_weight"]

    def test_multiple_rewards_accumulate(self, conn):
        """Multiple rewards should keep budget normalized."""
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories, get_neighbors
        from mneme.learn import reward_edge
        _make_test_data(conn)
        for _ in range(3):
            reward_edge(conn, "Source", "Target1", "RELATES_TO", amount=5)
        neighbors = get_neighbors(conn, "Source")
        total = sum(n["accuracy_weight"] for n in neighbors)
        assert abs(total - 100.0) < 0.01

    def test_reward_updates_counters(self, conn):
        """Rewarding an edge increments use_count and useful_count."""
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories, get_neighbors
        from mneme.learn import reward_edge
        _make_test_data(conn)
        reward_edge(conn, "Source", "Target1", "RELATES_TO", amount=5)
        neighbors = get_neighbors(conn, "Source")
        target1 = [n for n in neighbors if n["target_title"] == "Target1"][0]
        assert target1["use_count"] >= 1
        assert target1["useful_count"] >= 1


class TestRewardMixedRelTypes:
    """Budget normalization should consider ALL outgoing edges, regardless of type."""

    def test_budget_across_rel_types(self, conn):
        """Rewarding RELATES_TO should normalize against ANALOGOUS_TO too."""
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories, get_neighbors
        from mneme.learn import reward_edge
        _make_mixed_rel_data(conn)
        reward_edge(conn, "Source", "TargetA", "RELATES_TO", amount=20)
        neighbors = get_neighbors(conn, "Source")
        total = sum(n["accuracy_weight"] for n in neighbors)
        assert abs(total - 100.0) < 0.01


class TestRewardSingleEdge:
    """When source has only one outgoing edge, budget should set it to 100."""

    def test_single_edge_set_to_100(self, conn):
        """With only one edge, rewarding should set its weight to 100."""
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories, get_neighbors
        from mneme.learn import reward_edge
        _make_single_edge_data(conn)
        reward_edge(conn, "SoloSource", "OnlyTarget", "RELATES_TO", amount=5)
        neighbors = get_neighbors(conn, "SoloSource")
        assert len(neighbors) == 1
        assert neighbors[0]["accuracy_weight"] == pytest.approx(100.0)


class TestRewardErrors:
    """Tests for error handling in reward_edge."""

    def test_reward_nonexistent_target(self, conn):
        conn, db, path = conn
        from mneme.learn import reward_edge
        _make_reward_error_data(conn)
        with pytest.raises(ValueError, match="Relationship not found"):
            reward_edge(conn, "Source", "Nonexistent", "RELATES_TO", amount=5)

    def test_reward_nonexistent_source(self, conn):
        conn, db, path = conn
        from mneme.learn import reward_edge
        _make_reward_error_data(conn)
        with pytest.raises(ValueError, match="No relationships from"):
            reward_edge(conn, "Nonexistent", "Target", "RELATES_TO", amount=5)


class TestRewardReturnValues:
    """Tests for reward_edge return value structure."""

    def test_return_has_required_fields(self, conn):
        conn, db, path = conn
        from mneme.learn import reward_edge
        _make_reward_error_data(conn)
        result = reward_edge(conn, "Source", "Target", "RELATES_TO", amount=5)
        required = ["source_title", "target_title", "rel_type", "weight_field",
                     "amount", "before_accuracy_weight", "after_accuracy_weight",
                     "before_creative_weight", "after_creative_weight"]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_default_amount_is_3(self, conn):
        """Default amount parameter is 3.0."""
        conn, db, path = conn
        from mneme.learn import reward_edge
        _make_reward_error_data(conn)
        result = reward_edge(conn, "Source", "Target", "RELATES_TO")
        assert result["amount"] == 3.0
        assert result["after_accuracy_weight"] > result["before_accuracy_weight"]


class TestBudgetInvariant:
    """Budget-invariant test: total per-source accuracy_weight should stay ≈ 100."""

    def test_budget_invariant_mixed_rel_types(self, conn):
        """Create a source with 3+ edges of mixed rel types, perform multiple
        rewards on the same weight_field, and assert total ≈ 100."""
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories, get_neighbors
        from mneme.learn import reward_edge

        # Source with 3 edges of different types
        add_memory(conn, "BudgetSource", "Budget test source")
        add_memory(conn, "TargetA", "Target A")
        add_memory(conn, "TargetB", "Target B")
        add_memory(conn, "TargetC", "Target C")
        link_memories(conn, "BudgetSource", "TargetA", kind="relates_to")
        link_memories(conn, "BudgetSource", "TargetB", kind="analogous_to")
        link_memories(conn, "BudgetSource", "TargetC", kind="relates_to")

        # Initial total is ~3 (each edge starts at 1.0) — normalization kicks in on reward
        neighbors = get_neighbors(conn, "BudgetSource")
        total = sum(n["accuracy_weight"] for n in neighbors)
        assert abs(total - 3.0) < 0.01

        # Perform multiple rewards — all on accuracy_weight
        # (With single-budget normalization, only the rewarded budget normalizes)
        reward_edge(conn, "BudgetSource", "TargetA", "RELATES_TO",
                    weight_field="accuracy_weight", amount=10)
        reward_edge(conn, "BudgetSource", "TargetB", "ANALOGOUS_TO",
                    weight_field="accuracy_weight", amount=20)
        reward_edge(conn, "BudgetSource", "TargetA", "RELATES_TO",
                    weight_field="accuracy_weight", amount=5)

        # Total should now be ≈ 100 for accuracy_weight (budget normalization applied)
        neighbors = get_neighbors(conn, "BudgetSource")
        total = sum(n["accuracy_weight"] for n in neighbors)
        assert abs(total - 100.0) < 0.01, f"Total weight {total} != 100"

        # Each individual edge should still be between 0 and 100
        for n in neighbors:
            assert 0 < n["accuracy_weight"] <= 100, \
                f"Edge to {n['target_title']} weight {n['accuracy_weight']} out of bounds"

    def test_budget_invariant_creative_weight(self, conn):
        """Budget invariant holds for creative_weight too."""
        conn, db, path = conn
        from mneme.memory import add_memory, link_memories, get_neighbors
        from mneme.learn import reward_edge

        add_memory(conn, "Src", "Body")
        add_memory(conn, "T1", "Body")
        add_memory(conn, "T2", "Body")
        link_memories(conn, "Src", "T1", kind="relates_to")
        link_memories(conn, "Src", "T2", kind="relates_to")

        reward_edge(conn, "Src", "T1", "RELATES_TO",
                    weight_field="creative_weight", amount=15)
        reward_edge(conn, "Src", "T2", "RELATES_TO",
                    weight_field="creative_weight", amount=10)

        neighbors = get_neighbors(conn, "Src")
        total = sum(n["creative_weight"] for n in neighbors)
        assert abs(total - 100.0) < 0.01, f"Creative total {total} != 100"
