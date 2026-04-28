"""Unit tests for citation-grounded reward stubs."""

import importlib
import json
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def _set_llm_env(url: str, key: str) -> None:
    """Helper to set LLM env vars and reload the config module."""
    os.environ["MNEME_LLM_API_URL"] = url
    os.environ["MNEME_LLM_API_KEY"] = key
    # Reload config to pick up new env vars
    from mneme import config
    importlib.reload(config)


def _clear_llm_env() -> None:
    """Helper to clear LLM env vars and reload the config module.

    Sets the vars to empty strings rather than unsetting them — config's
    auto-loader honors anything already set, so empty strings prevent the
    user's ``~/.config/mneme.env`` (if present) from leaking in.
    Empty strings make ``is_llm_configured()`` return False.
    """
    os.environ["MNEME_LLM_API_URL"] = ""
    os.environ["MNEME_LLM_API_KEY"] = ""
    from mneme import config
    importlib.reload(config)


class TestCitationRewards:
    """Tests for apply_citation_rewards."""

    def test_no_api_key_skips(self, conn):
        """When no LLM key, returns skipped status."""
        _clear_llm_env()

        from mneme.learn import apply_citation_rewards

        real_conn = conn[0]

        path = [
            {"id": "1", "title": "A", "body": "body A"},
            {"id": "2", "title": "B", "body": "body B"},
        ]
        edges = [
            {"rel_type": "RELATES_TO", "from": "A", "to": "B"},
        ]
        result = apply_citation_rewards(real_conn, path, edges, "test query")
        assert result["status"] == "skipped"

    def test_parse_citations_filters_hallucinations(self, conn):
        """Citations not in path are dropped."""
        _set_llm_env("http://fake:1234/v1/chat/completions", "fake-key")

        from mneme.learn import apply_citation_rewards

        real_conn = conn[0]

        path = [
            {"id": "node1", "title": "Alpha", "body": "alpha body"},
            {"id": "node2", "title": "Beta", "body": "beta body"},
        ]
        edges = [
            {"rel_type": "RELATES_TO", "from": "Alpha", "to": "Beta"},
        ]

        mock_response = {
            "answer": "test",
            "cited": [
                {"id": "node1", "used_for": "context"},
                {"id": "nonexistent", "used_for": "hallucination"},
                {"id": "node2", "used_for": "answer"},
            ],
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": json.dumps(mock_response)}}]
            }).encode()
            mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_resp)
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

            result = apply_citation_rewards(real_conn, path, edges, "test query")
            assert result["status"] == "done"
            assert "nonexistent" not in result["citations"]
            assert "node1" in result["citations"]
            assert "node2" in result["citations"]

        _clear_llm_env()

    def test_reward_cited_edges(self, conn):
        """Cited edges get rewarded with correct amount."""
        _set_llm_env("http://fake:1234/v1/chat/completions", "fake-key")

        from mneme.learn import apply_citation_rewards
        from mneme.memory import add_memory, link_memories

        real_conn = conn[0]

        # Add test data
        add_memory(real_conn, "MemoryA", "Body A")
        add_memory(real_conn, "MemoryB", "Body B")
        add_memory(real_conn, "MemoryC", "Body C")
        link_memories(real_conn, "MemoryA", "MemoryB", kind="relates_to")
        link_memories(real_conn, "MemoryB", "MemoryC", kind="relates_to")

        path = [
            {"id": "ma", "title": "MemoryA", "body": "body A"},
            {"id": "mb", "title": "MemoryB", "body": "body B"},
            {"id": "mc", "title": "MemoryC", "body": "body C"},
        ]
        edges = [
            {"rel_type": "RELATES_TO", "from": "MemoryA", "to": "MemoryB"},
            {"rel_type": "RELATES_TO", "from": "MemoryB", "to": "MemoryC"},
        ]

        mock_response = {
            "answer": "test",
            "cited": [{"id": "mb", "used_for": "answer"}],
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": json.dumps(mock_response)}}]
            }).encode()
            mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_resp)
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

            result = apply_citation_rewards(real_conn, path, edges, "test query")
            assert result["status"] == "done"
            assert result["edges_rewarded"] == 1

        _clear_llm_env()
