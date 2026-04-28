"""Session management for Mneme."""

import uuid
from .utils import now_iso


def create_session(conn, user_query: str, mode: str,
                   session_id: str = None) -> dict:
    """Create a new session node."""
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    ts = now_iso()
    result = conn.execute(
        "CREATE (s:Session {id: $id, user_query: $query, "
        "mode: $mode, created_at: $ts}) "
        "RETURN s.id AS id, s.user_query AS user_query, "
        "s.mode AS mode, s.created_at AS created_at",
        {"id": session_id, "query": user_query,
         "mode": mode, "ts": ts},
    )
    row = result.get_next()
    d = dict(zip(result.get_column_names(), row))
    return {
        "id": d.get("id"),
        "user_query": d.get("user_query"),
        "mode": d.get("mode"),
        "created_at": d.get("created_at"),
    }


def log_session_edges(conn, session_id: str,
                      traversal_result: dict) -> None:
    """Create USED_IN_SESSION relationships for visited nodes.

    Note: Kuzu 0.11.x does not support parameterized properties on
    relationship CREATE, so we use safe string interpolation.
    The Session node is already created by create_session(), so we
    MATCH it rather than CREATE it again.
    """
    for idx, node in enumerate(traversal_result.get("path", [])):
        node_id = node.get("id")
        if not node_id:
            continue
        role = "starting" if idx == 0 else "visited"
        # Escape single quotes to prevent syntax errors in Cypher strings
        safe_role = str(role).replace("'", "''")
        conn.execute(
            f"MATCH (s:Session {{id: '{session_id}'}}), "
            f"(m:Memory {{id: '{node_id}'}}) CREATE "
            f"(s)-[r:USED_IN_SESSION {{role: '{safe_role}', "
            f"usefulness: 1.0, step_order: {idx}}}]->(m)",
        )


def list_sessions(conn) -> list[dict]:
    """List all sessions, newest first."""
    result = conn.execute(
        "MATCH (s:Session) "
        "RETURN s.id AS id, s.user_query AS query, "
        "s.mode AS mode, s.created_at AS created_at "
        "ORDER BY s.created_at DESC",
    )
    sessions = []
    while result.has_next():
        row = result.get_next()
        d = dict(zip(result.get_column_names(), row))
        sessions.append({
            "id": d.get("id"),
            "query": d.get("query"),
            "mode": d.get("mode"),
            "created_at": d.get("created_at"),
        })
    return sessions
