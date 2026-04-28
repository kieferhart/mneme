"""Configuration for Mneme — LLM and reward settings.

Auto-loads ``~/.config/mneme.env`` if present so CLI users and embedded
callers (e.g. the Hermes plugin) get the same env without manual sourcing.
Existing ``os.environ`` values always win — the file only fills gaps.
"""

import os
from pathlib import Path

_ENV_FILE = Path.home() / ".config" / "mneme.env"


def _load_env_file(path: Path) -> None:
    """Set unset os.environ entries from a KEY=VALUE file.

    Lines starting with ``#`` and blank lines are ignored. Surrounding quotes
    on values are stripped. Existing env vars are never overwritten.
    """
    if not path.is_file():
        return
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_env_file(_ENV_FILE)


LLM_API_URL = os.environ.get("MNEME_LLM_API_URL")
LLM_API_KEY = os.environ.get("MNEME_LLM_API_KEY")
LLM_MODEL = os.environ.get("MNEME_LLM_MODEL", "gpt-4o-mini")  # default for when key is set

PAIRWISE_SAMPLE_RATE = float(os.environ.get("MNEME_PAIRWISE_SAMPLE_RATE", "0.10"))
CITATION_REWARD_AMOUNT = float(os.environ.get("MNEME_CITATION_REWARD_AMOUNT", "2.0"))
PAIRWISE_REWARD_AMOUNT = float(os.environ.get("MNEME_PAIRWISE_REWARD_AMOUNT", "5.0"))


def is_llm_configured() -> bool:
    """Return True only when both URL and key are set."""
    return bool(LLM_API_URL and LLM_API_KEY)
