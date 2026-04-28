"""Unit tests for Mneme creativity module."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestComputeCrossSimilarity:
    """Tests for compute_cross_similarity."""

    def test_identical_memories(self):
        from mneme.creativity import compute_cross_similarity
        mem = {"title": "test", "summary": "test", "body": "test"}
        score = compute_cross_similarity(mem, mem)
        assert score == 1.0

    def test_no_overlap(self):
        from mneme.creativity import compute_cross_similarity
        a = {"title": "hello", "summary": "hello", "body": "hello"}
        b = {"title": "goodbye", "summary": "goodbye", "body": "goodbye"}
        score = compute_cross_similarity(a, b)
        assert score == 0.0

    def test_partial_overlap(self):
        from mneme.creativity import compute_cross_similarity
        a = {"title": "hello world python", "summary": "", "body": ""}
        b = {"title": "hello world java", "summary": "", "body": ""}
        score = compute_cross_similarity(a, b)
        assert score > 0.0
        assert score < 1.0

    def test_empty_memories(self):
        from mneme.creativity import compute_cross_similarity
        a = {"title": "", "summary": "", "body": ""}
        b = {"title": "test", "summary": "test", "body": "test"}
        score = compute_cross_similarity(a, b)
        assert score == 0.0

    def test_both_empty(self):
        from mneme.creativity import compute_cross_similarity
        a = {"title": "", "summary": "", "body": ""}
        b = {"title": "", "summary": "", "body": ""}
        score = compute_cross_similarity(a, b)
        assert score == 0.0

    def test_returns_float_0_to_1(self):
        from mneme.creativity import compute_cross_similarity
        a = {"title": "python programming", "summary": "python", "body": "python programming language"}
        b = {"title": "python snake", "summary": "python", "body": "a snake"}
        score = compute_cross_similarity(a, b)
        assert 0.0 <= score <= 1.0

    def test_case_insensitive(self):
        from mneme.creativity import compute_cross_similarity
        a = {"title": "Hello", "summary": "", "body": ""}
        b = {"title": "hello", "summary": "", "body": ""}
        score = compute_cross_similarity(a, b)
        assert score > 0.0


class TestSuggestNewLinks:
    """Tests for suggest_new_links."""

    def test_no_suggestions_when_all_linked(self):
        from mneme.creativity import suggest_new_links
        path = [
            {"title": "A", "summary": "a", "body": "a"},
            {"title": "B", "summary": "b", "body": "b"},
        ]
        edges = [{"source_title": "A", "target_title": "B"}]
        suggestions = suggest_new_links(path, edges, threshold=0.0)
        assert suggestions == []

    def test_suggests_unlinked_similar_nodes(self):
        from mneme.creativity import suggest_new_links
        path = [
            {"title": "Python", "summary": "python programming", "body": "python is a language"},
            {"title": "JavaScript", "summary": "javascript programming", "body": "js is a language"},
        ]
        edges = []
        suggestions = suggest_new_links(path, edges, threshold=0.0)
        assert len(suggestions) == 1
        assert suggestions[0]["source_title"] == "Python"
        assert suggestions[0]["target_title"] == "JavaScript"
        assert suggestions[0]["similarity"] > 0
        assert suggestions[0]["confidence"] > 0

    def test_threshold_filters(self):
        from mneme.creativity import suggest_new_links
        path = [
            {"title": "Python", "summary": "python", "body": "python"},
            {"title": "Cake", "summary": "cake", "body": "cake"},
        ]
        edges = []
        # High threshold should filter out low similarity
        suggestions = suggest_new_links(path, edges, threshold=0.5)
        assert suggestions == []

    def test_suggestion_has_required_fields(self):
        from mneme.creativity import suggest_new_links
        path = [
            {"title": "A", "summary": "a overlap", "body": "a overlap"},
            {"title": "B", "summary": "b overlap", "body": "b overlap"},
        ]
        suggestions = suggest_new_links(path, [], threshold=0.0)
        for s in suggestions:
            assert "source_title" in s
            assert "target_title" in s
            assert "similarity" in s
            assert "initial_creative_weight" in s
            assert "confidence" in s
            assert "reason" in s

    def test_no_duplicate_suggestions(self):
        from mneme.creativity import suggest_new_links
        path = [
            {"title": "A", "summary": "abc", "body": "abc"},
            {"title": "B", "summary": "abc", "body": "abc"},
            {"title": "C", "summary": "abc", "body": "abc"},
        ]
        edges = []
        suggestions = suggest_new_links(path, edges, threshold=0.0)
        pairs = [(s["source_title"], s["target_title"]) for s in suggestions]
        assert len(pairs) == len(set(pairs))

    def test_suggestions_count_n_choose_2_minus_existing(self):
        from mneme.creativity import suggest_new_links
        path = [
            {"title": "A", "summary": "overlap", "body": "overlap"},
            {"title": "B", "summary": "overlap", "body": "overlap"},
            {"title": "C", "summary": "overlap", "body": "overlap"},
        ]
        edges = [{"source_title": "A", "target_title": "B"}]
        suggestions = suggest_new_links(path, edges, threshold=0.0)
        assert len(suggestions) == 2  # A-C and B-C

    def test_directional_pairs(self):
        from mneme.creativity import suggest_new_links
        path = [
            {"title": "A", "summary": "overlap", "body": "overlap"},
            {"title": "B", "summary": "overlap", "body": "overlap"},
        ]
        edges = [{"target_title": "A", "source_title": "B"}]  # reverse direction
        suggestions = suggest_new_links(path, edges, threshold=0.0)
        assert suggestions == []  # Undirected check should cover reverse

    def test_initial_creative_weight_scales_with_similarity(self):
        from mneme.creativity import suggest_new_links
        # Very similar
        path1 = [
            {"title": "A", "summary": "python programming language", "body": "python"},
            {"title": "B", "summary": "python programming language", "body": "python"},
        ]
        s1 = suggest_new_links(path1, [], threshold=0.0)

        # Less similar
        path2 = [
            {"title": "A", "summary": "python", "body": "python"},
            {"title": "B", "summary": "python snake", "body": "snake"},
        ]
        s2 = suggest_new_links(path2, [], threshold=0.0)

        assert s1[0]["similarity"] >= s2[0]["similarity"]
        assert s1[0]["initial_creative_weight"] >= s2[0]["initial_creative_weight"]



