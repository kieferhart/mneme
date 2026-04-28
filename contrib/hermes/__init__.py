"""Mneme memory provider for Hermes.

Wraps Mneme as a Hermes ``MemoryProvider`` plugin. Memories are stored in a
Kuzu graph with 384-dim sentence-transformer embeddings; edges carry weights
and confidence; useful paths gain weight through citation-grounded
reinforcement.

Configuration in ``$HERMES_HOME/config.yaml``::

    memory:
      provider: mneme

    plugins:
      mneme:
        db_path: $HERMES_HOME/mneme.kuzu     # default; or any absolute path
        prefetch_top_n: 5
        prefetch_mode: balanced
        traverse_on_prefetch: false          # hop traversal is heavier
        lock_retry_attempts: 5
        lock_retry_initial_delay: 0.2

Activation requires ``mneme`` to be importable in hermes's Python environment::

    /path/to/hermes/venv/bin/pip install mneme

Concurrency note: Kuzu uses an exclusive file lock per database. The plugin
opens and closes a connection per operation (with bounded retry on lock
collision) so multiple hermes processes — gateway, TUI, cron — can share the
same database without permanently locking each other out.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error, tool_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

MNEME_SCHEMA: Dict[str, Any] = {
    "name": "mneme",
    "description": (
        "Mneme — local memory graph with semantic embeddings + weighted "
        "edges that get reinforced by use.\n\n"
        "ACTIONS:\n"
        "• add — Store a memory worth recalling later. Use for facts, decisions, "
        "preferences, project context, anything you'd want to retrieve.\n"
        "• link — Connect two existing memories with a typed edge "
        "(relates_to | analogous_to). Builds the graph.\n"
        "• ask — Traverse the graph: find a starting node from the query, walk "
        "weighted edges, return the path. Use this to answer questions where "
        "context has likely been stored before.\n"
        "• show — Display a memory + all its incoming/outgoing edges.\n"
        "• neighbors — List outgoing edges from a memory with weights.\n"
        "• reward — Manually reinforce an edge (the citation/pairwise loop "
        "usually handles this automatically).\n"
        "• suggest_links — From a list of related memory titles, surface "
        "high-similarity unconnected pairs worth linking.\n\n"
        "MODES (for ask): strict | balanced | creative | discovery. "
        "Discovery targets the analogical mid-band (cosine 0.4–0.6) — useful "
        "for cross-domain insights."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "link", "ask", "show", "neighbors",
                         "reward", "suggest_links"],
            },
            "title": {"type": "string", "description": "Memory title (add/show/neighbors)."},
            "body": {"type": "string", "description": "Memory body (add)."},
            "kind": {"type": "string", "description": "Memory kind tag, or edge kind for link (relates_to|analogous_to)."},
            "source": {"type": "string", "description": "Source memory title (link/reward)."},
            "target": {"type": "string", "description": "Target memory title (link/reward)."},
            "reason": {"type": "string", "description": "Reason annotation for an analogous_to link."},
            "query": {"type": "string", "description": "Query for ask."},
            "mode": {
                "type": "string",
                "enum": ["strict", "balanced", "creative", "discovery"],
                "description": "Traversal mode for ask (default: balanced).",
            },
            "max_hops": {"type": "integer", "description": "Max hops for ask (default: 3)."},
            "weight_field": {
                "type": "string",
                "enum": ["accuracy_weight", "creative_weight"],
                "description": "Which budget to reward (default: accuracy_weight).",
            },
            "amount": {"type": "number", "description": "Reward amount (default: 3.0)."},
            "titles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Memory titles to consider for suggest_links.",
            },
            "threshold": {
                "type": "number",
                "description": "Similarity threshold for suggest_links (default: 0.05).",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
    except Exception:
        return {}
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("plugins", {}).get("mneme", {}) or {}
    except Exception:
        return {}


def _expand(value: str, hermes_home: str) -> str:
    if not isinstance(value, str):
        return value
    value = value.replace("$HERMES_HOME", hermes_home)
    value = value.replace("${HERMES_HOME}", hermes_home)
    return os.path.expanduser(value)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class MnemeProvider(MemoryProvider):
    """Hermes memory provider backed by Mneme's Kuzu graph.

    Connections are opened and closed per operation so multiple hermes
    processes (gateway, TUI, cron) can share the same DB without permanently
    locking each other out. Lock collisions are retried with exponential
    backoff (see :meth:`_open_db`).
    """

    # Default retry budget. ~37s max wait, generous enough to cover an LLM
    # call holding the lock in another hermes process (apply_citation_rewards
    # waits on a remote completion that can take 5–10s).
    _LOCK_MAX_DELAY = 2.0
    _LOCK_DEFAULT_ATTEMPTS = 20
    _LOCK_DEFAULT_INITIAL_DELAY = 0.2

    def __init__(self, config: Optional[dict] = None):
        self._config = config or _load_plugin_config()
        self._db_path: Optional[str] = None
        self._session_id: Optional[str] = None
        self._available: Optional[bool] = None

    def _retry_budget(self) -> tuple[int, float]:
        """Read retry params from current config (re-read each open so config
        edits take effect without a hermes restart)."""
        attempts = int(self._config.get("lock_retry_attempts",
                                        self._LOCK_DEFAULT_ATTEMPTS))
        initial = float(self._config.get("lock_retry_initial_delay",
                                         self._LOCK_DEFAULT_INITIAL_DELAY))
        return attempts, initial

    @property
    def name(self) -> str:
        return "mneme"

    # -- lifecycle ----------------------------------------------------------

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import mneme  # noqa: F401
            from mneme.embeddings import embed  # noqa: F401
            import kuzu  # noqa: F401
            self._available = True
        except ImportError as exc:
            logger.warning(
                "mneme plugin: required dependency not importable (%s). "
                "Install Mneme into hermes's venv: "
                "<hermes-venv>/bin/pip install mneme",
                exc,
            )
            self._available = False
        return self._available

    def initialize(self, session_id: str, **kwargs) -> None:
        if not self.is_available():
            return

        try:
            from hermes_constants import get_hermes_home
            hermes_home = str(get_hermes_home())
        except Exception:
            hermes_home = str(Path.home() / ".hermes")

        default_db = str(Path(hermes_home) / "mneme.kuzu")
        db_path = _expand(self._config.get("db_path", default_db), hermes_home)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._session_id = session_id

        # Eagerly run the schema once; subsequent connections inherit it.
        try:
            with self._open_db() as (_, conn):
                from mneme.schema import init_schema
                init_schema(conn)
        except Exception as exc:
            logger.warning("mneme: initial schema check failed: %s", exc)

    def shutdown(self) -> None:
        # Nothing to release — connections are scoped to each call.
        return None

    # -- connection helper --------------------------------------------------

    @contextlib.contextmanager
    def _open_db(self):
        """Open a Kuzu DB + connection, retrying on lock collisions.

        Releases the lock as soon as the ``with`` block exits — both objects
        are dropped and ``gc.collect()`` runs to force Kuzu's destructor.
        Backoff doubles each attempt up to ``_LOCK_MAX_DELAY``.
        """
        import kuzu

        attempts, delay = self._retry_budget()
        last_err: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                db = kuzu.Database(self._db_path)
                conn = kuzu.Connection(db)
                try:
                    yield db, conn
                finally:
                    del conn
                    del db
                    gc.collect()
                return
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "lock" in msg or "exclusive" in msg:
                    last_err = exc
                    if attempt < attempts - 1:
                        time.sleep(min(delay, self._LOCK_MAX_DELAY))
                        delay = min(delay * 2, self._LOCK_MAX_DELAY)
                    continue
                raise
        raise RuntimeError(
            f"mneme: could not acquire database lock at {self._db_path} after "
            f"{attempts} attempts; another process is holding it. "
            f"Last error: {last_err}"
        )

    # -- prompt + recall ----------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._db_path:
            return ""
        try:
            with self._open_db() as (_, conn):
                res = conn.execute("MATCH (m:Memory) RETURN COUNT(*) AS n")
                n = res.get_next()[0] if res.has_next() else 0
        except Exception as exc:
            logger.debug("mneme system_prompt_block failed: %s", exc)
            return ""
        if n == 0:
            return (
                "# Mneme\n"
                "Active. Empty memory graph — proactively store memories worth "
                "recalling later via mneme(action='add'). "
                "Use mneme(action='ask', query=...) to traverse stored "
                "context, and mneme(action='link', ...) to connect related "
                "memories."
            )
        return (
            f"# Mneme\n"
            f"Active. {n} memories stored in a weighted graph with semantic "
            f"embeddings.\n"
            f"- ask: traversal-based recall (modes: strict/balanced/creative/discovery).\n"
            f"- add/link: extend the graph.\n"
            f"- show/neighbors: inspect a memory and its edges."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._db_path or not query:
            return ""
        top_n = int(self._config.get("prefetch_top_n", 5))
        traverse_on_prefetch = bool(self._config.get("traverse_on_prefetch", False))
        mode = str(self._config.get("prefetch_mode", "balanced"))
        try:
            from mneme.traverse import find_candidate_nodes, traverse
            with self._open_db() as (_, conn):
                candidates = find_candidate_nodes(conn, query, top_n=top_n)
                lines = ["## Mneme recall"]
                for c in candidates:
                    score = c.get("overlap_score", 0)
                    title = c.get("title", "?")
                    summary = (c.get("summary") or c.get("body") or "")[:200]
                    lines.append(f"- [{score:.2f}] **{title}** — {summary}")

                if traverse_on_prefetch and candidates:
                    start = candidates[0]
                    path_res = traverse(
                        conn, start["id"], query,
                        mode=mode, max_hops=int(self._config.get("max_hops", 3)),
                    )
                    path_titles = [n.get("title", "?") for n in path_res.get("path", [])]
                    if len(path_titles) > 1:
                        lines.append("")
                        lines.append("Path: " + " → ".join(path_titles))
        except Exception as exc:
            logger.debug("mneme prefetch failed: %s", exc)
            return ""
        if not candidates:
            return ""
        return "\n".join(lines)

    # -- tools --------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [MNEME_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "mneme":
            return tool_error(f"Unknown tool: {tool_name}")
        if not self._db_path:
            return tool_error("mneme provider not initialized")

        action = args.get("action")
        try:
            if action == "add":
                return self._do_add(args)
            if action == "link":
                return self._do_link(args)
            if action == "ask":
                return self._do_ask(args)
            if action == "show":
                return self._do_show(args)
            if action == "neighbors":
                return self._do_neighbors(args)
            if action == "reward":
                return self._do_reward(args)
            if action == "suggest_links":
                return self._do_suggest_links(args)
            return tool_error(f"Unknown action: {action}")
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            logger.exception("mneme tool call failed")
            return tool_error(str(exc))

    # -- mirror built-in memory writes --------------------------------------

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        if action != "add" or not self._db_path or not content:
            return
        try:
            from mneme.memory import add_memory
            kind = "user_pref" if target == "user" else "memory_note"
            title = (content.splitlines()[0] or content)[:80].strip() or "untitled"
            with self._open_db() as (_, conn):
                add_memory(conn, title=title, body=content, kind=kind)
        except Exception as exc:
            logger.debug("mneme on_memory_write failed: %s", exc)

    # -- setup wizard -------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "db_path", "description": "Kuzu database path",
             "default": "$HERMES_HOME/mneme.kuzu"},
            {"key": "prefetch_top_n", "description": "Memories surfaced per turn",
             "default": "5"},
            {"key": "prefetch_mode", "description": "Traversal mode for prefetch",
             "default": "balanced",
             "choices": ["strict", "balanced", "creative", "discovery"]},
            {"key": "traverse_on_prefetch",
             "description": "Walk the graph during prefetch (heavier)",
             "default": "false", "choices": ["true", "false"]},
            {"key": "lock_retry_attempts",
             "description": "Retries when another process holds the DB",
             "default": "5"},
            {"key": "lock_retry_initial_delay",
             "description": "Initial backoff seconds (doubles each retry)",
             "default": "0.2"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing: dict = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["mneme"] = values
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as exc:
            logger.warning("mneme save_config failed: %s", exc)

    # -- action handlers ----------------------------------------------------

    def _do_add(self, args: dict) -> str:
        from mneme.memory import add_memory
        title = args["title"]
        body = args.get("body", "")
        kind = args.get("kind", "note")
        with self._open_db() as (_, conn):
            result = add_memory(conn, title=title, body=body, kind=kind)
        return tool_result({
            "id": result["id"], "title": result["title"], "kind": result["kind"],
            "summary": result.get("summary"),
        })

    def _do_link(self, args: dict) -> str:
        from mneme.memory import link_memories
        kind = args.get("kind", "relates_to")
        kind = "relates_to" if kind == "relates_to" else "analogous_to"
        with self._open_db() as (_, conn):
            result = link_memories(
                conn, args["source"], args["target"],
                kind=kind, reason=args.get("reason", ""),
            )
        return tool_result(result)

    def _do_ask(self, args: dict) -> str:
        from mneme.traverse import find_candidate_nodes, traverse
        from mneme.session import create_session, log_session_edges
        from mneme.learn import apply_citation_rewards
        query = args["query"]
        mode = args.get("mode", "balanced")
        max_hops = int(args.get("max_hops", 3))

        with self._open_db() as (_, conn):
            candidates = find_candidate_nodes(conn, query, top_n=5)
            if not candidates:
                return tool_result({"path": [], "candidates": [],
                                    "note": "no memories matched the query"})

            session = create_session(conn, user_query=query, mode=mode)
            path_res = traverse(conn, candidates[0]["id"], query,
                                mode=mode, max_hops=max_hops)
            try:
                log_session_edges(conn, session["id"], path_res)
            except Exception as exc:
                logger.debug("log_session_edges failed: %s", exc)

            # Citation rewards run synchronously inside the same lock window.
            # Pairwise rewards (async) are intentionally skipped from the plugin
            # path: they would otherwise outlive the lock window and race the
            # next caller. Run pairwise via the mneme CLI if you want it.
            citation = apply_citation_rewards(conn, path_res["path"],
                                              path_res["edges"], query)

        return tool_result({
            "session_id": session["id"],
            "starting_candidates": [
                {"title": c["title"],
                 "score": c.get("overlap_score"),
                 "cosine": c.get("cosine_score"),
                 "keyword": c.get("keyword_score")}
                for c in candidates
            ],
            "path": [
                {"title": n.get("title"), "summary": n.get("summary")}
                for n in path_res.get("path", [])
            ],
            "edges": [
                {"from": e.get("from"), "to": e.get("to"),
                 "rel_type": e.get("rel_type"), "score": e.get("score")}
                for e in path_res.get("edges", [])
            ],
            "citation_status": citation.get("status"),
            "edges_rewarded": citation.get("edges_rewarded", 0),
        })

    def _do_show(self, args: dict) -> str:
        from mneme.memory import show_memory
        with self._open_db() as (_, conn):
            result = show_memory(conn, args["title"])
        if not result:
            return tool_error(f"Memory not found: {args['title']}")
        return tool_result({
            "id": result["id"], "title": result["title"],
            "summary": result.get("summary"), "kind": result.get("kind"),
            "outgoing_edges": [
                {"target": e.get("target_title"),
                 "kind": e.get("kind"),
                 "accuracy_weight": e.get("accuracy_weight"),
                 "creative_weight": e.get("creative_weight"),
                 "use_count": e.get("use_count")}
                for e in result.get("outgoing_edges", [])
            ],
            "incoming_edges": [
                {"source": e.get("source_title"),
                 "kind": e.get("kind"),
                 "accuracy_weight": e.get("accuracy_weight")}
                for e in result.get("incoming_edges", [])
            ],
        })

    def _do_neighbors(self, args: dict) -> str:
        from mneme.memory import get_neighbors
        with self._open_db() as (_, conn):
            edges = get_neighbors(conn, args["title"])
        return tool_result({
            "title": args["title"],
            "edges": [
                {"target": e.get("target_title"),
                 "kind": e.get("kind"),
                 "accuracy_weight": e.get("accuracy_weight"),
                 "creative_weight": e.get("creative_weight"),
                 "novelty_weight": e.get("novelty_weight"),
                 "confidence": e.get("confidence"),
                 "use_count": e.get("use_count")}
                for e in edges
            ],
        })

    def _do_reward(self, args: dict) -> str:
        from mneme.learn import reward_edge
        from mneme.memory import get_neighbors
        source = args["source"]
        target = args["target"]
        with self._open_db() as (_, conn):
            rel_table = "RELATES_TO"
            for n in get_neighbors(conn, source):
                if n.get("target_title") == target:
                    kind = str(n.get("kind", "")).lower()
                    rel_table = "ANALOGOUS_TO" if "analog" in kind else "RELATES_TO"
                    break
            result = reward_edge(
                conn, source, target, rel_table,
                weight_field=args.get("weight_field", "accuracy_weight"),
                amount=float(args.get("amount", 3.0)),
            )
        return tool_result(result)

    def _do_suggest_links(self, args: dict) -> str:
        from mneme.memory import find_memory_by_title, get_neighbors
        from mneme.creativity import suggest_new_links
        titles = args.get("titles", [])
        if not titles:
            return tool_error("suggest_links requires 'titles' list")
        threshold = float(args.get("threshold", 0.05))
        with self._open_db() as (_, conn):
            path = []
            existing: list[dict] = []
            for t in titles:
                mem = find_memory_by_title(conn, t)
                if not mem:
                    continue
                path.append(mem)
                for e in get_neighbors(conn, t):
                    existing.append({"source_title": t,
                                     "target_title": e.get("target_title")})
            suggestions = suggest_new_links(path, existing, threshold=threshold,
                                            conn=conn)
        return tool_result({"suggestions": suggestions, "count": len(suggestions)})


def register(ctx) -> None:
    """Register the Mneme provider with the plugin system."""
    config = _load_plugin_config()
    provider = MnemeProvider(config=config)
    ctx.register_memory_provider(provider)
