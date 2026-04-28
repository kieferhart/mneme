"""Unit tests for async pairwise comparison stubs."""

import importlib
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def _set_llm_env(url: str, key: str) -> None:
    """Helper to set LLM env vars and reload the config module."""
    os.environ["MNEME_LLM_API_URL"] = url
    os.environ["MNEME_LLM_API_KEY"] = key
    from mneme import config
    importlib.reload(config)


def _clear_llm_env() -> None:
    """Helper to clear LLM env vars and reload the config module."""
    os.environ.pop("MNEME_LLM_API_URL", None)
    os.environ.pop("MNEME_LLM_API_KEY", None)
    from mneme import config
    importlib.reload(config)


class TestPairwiseStubs:
    """Tests for run_pairwise_async and should_run_pairwise."""

    def test_no_api_key_skips(self, conn):
        """When no LLM key, run_pairwise_async returns silently (no error)."""
        _clear_llm_env()

        from mneme.pairwise import run_pairwise_async

        path_result = {
            "start_id": "1",
            "mode": "balanced",
            "max_hops": 3,
            "seed": 123,
            "path": [{"id": "1", "title": "A", "body": "body"}],
            "edges": [],
        }
        # Should not raise
        run_pairwise_async(conn=None, query="query", path_result=path_result)

    def test_should_run_pairwise_rate(self):
        """Verify ~10% rate over many trials with default sample rate."""
        from mneme.pairwise import should_run_pairwise
        from mneme import config

        old_rate = config.PAIRWISE_SAMPLE_RATE

        try:
            config.PAIRWISE_SAMPLE_RATE = 0.10
            n_trials = 2000
            hits = 0
            for _ in range(n_trials):
                if should_run_pairwise():
                    hits += 1
            rate = hits / n_trials
            # Allow a 5% absolute tolerance (i.e. 5% to 15%)
            assert 0.05 <= rate <= 0.15, f"Observed rate {rate:.3f} outside [0.05, 0.15]"
        finally:
            config.PAIRWISE_SAMPLE_RATE = old_rate

    def test_pairwise_stubs_run_without_api(self):
        """ThreadPoolExecutor doesn't crash when no API key."""
        _clear_llm_env()

        from mneme.pairwise import run_pairwise_async

        path_result = {
            "start_id": "1",
            "mode": "balanced",
            "max_hops": 3,
            "seed": 456,
            "path": [
                {"id": "1", "title": "Node A", "body": "body A"},
                {"id": "2", "title": "Node B", "body": "body B"},
            ],
            "edges": [
                {"rel_type": "RELATES_TO", "from": "Node A", "to": "Node B"},
            ],
        }
        run_pairwise_async(conn=None, query="test query", path_result=path_result)
        # If we got here without exception, the stub is working
