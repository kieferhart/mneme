"""Utility functions for Mneme."""

import re
import uuid
from datetime import datetime, timezone


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug. Falls back to UUID if empty."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or str(uuid.uuid4())


def summarize_simple(body: str, max_len: int = 240) -> str:
    """Truncate body to max_len characters, breaking at word boundary."""
    body = body.strip().replace("\n", " ")
    if max_len <= 0 or len(body) == 0:
        return ""
    if len(body) <= max_len:
        return body
    return body[:max_len].rsplit(" ", 1)[0] + "..."


def get_col(row, col_name: str, col_names: list[str] | None = None):
    """Get a column value from a Kuzu row (supports both dict and list).

    For dict rows, look up by key.
    For list rows, use col_names to convert via row_as_dict, then look up by key.
    """
    if isinstance(row, dict):
        return row.get(col_name)
    if col_names is not None:
        return row_as_dict(row, col_names).get(col_name)
    return row[0]  # legacy fallback


def row_as_dict(row, col_names: list[str]) -> dict:
    """Convert a Kuzu row (list) to a dict by column name."""
    return dict(zip(col_names, row))


def _edge_to_dict(edge_dict: dict) -> dict:
    """Extract edge properties from a Kuzu edge dict."""
    return {
        "kind": edge_dict.get("kind"),
        "accuracy_weight": edge_dict.get("accuracy_weight"),
        "creative_weight": edge_dict.get("creative_weight"),
        "novelty_weight": edge_dict.get("novelty_weight"),
        "confidence": edge_dict.get("confidence"),
        "use_count": edge_dict.get("use_count"),
        "useful_count": edge_dict.get("useful_count"),
        "failed_count": edge_dict.get("failed_count"),
        "last_used_at": edge_dict.get("last_used_at"),
        "reason": edge_dict.get("reason"),
    }
