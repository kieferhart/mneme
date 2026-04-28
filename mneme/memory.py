"""Memory operations: add, link, query, and delete memories."""

from .utils import slugify, now_iso, summarize_simple, row_as_dict, _edge_to_dict
from .embeddings import embed, embed_batch


def add_memory(conn, title: str, body: str, kind: str = "note") -> dict:
    """Add a new memory node to the database."""
    memory_id = slugify(title)
    summary = summarize_simple(body)
    ts = now_iso()
    emb = embed(f"{title} {summary} {body}")

    check = conn.execute(
        "MATCH (m:Memory {title: $title}) RETURN m.id AS id",
        {"title": title},
    )
    existing = check.get_next() if check.has_next() else None

    if existing:
        result = conn.execute(
            "MATCH (m:Memory {title: $title}) "
            "SET m.body = $body, m.summary = $summary, "
            "    m.kind = $kind, m.updated_at = $ts, "
            "    m.embedding = $emb "
            "RETURN m",
            {"title": title, "body": body, "summary": summary,
             "kind": kind, "ts": ts, "emb": emb},
        )
    else:
        result = conn.execute(
            "CREATE (m:Memory {id: $id, title: $title, "
            "  body: $body, summary: $summary, kind: $kind, "
            "  created_at: $ts, updated_at: $ts, embedding: $emb}) "
            "RETURN m",
            {"id": memory_id, "title": title, "body": body,
             "summary": summary, "kind": kind, "ts": ts, "emb": emb},
        )

    row = result.get_next()
    d = row_as_dict(row, result.get_column_names())
    mem = d["m"]
    return {
        "id": mem["id"], "title": mem["title"],
        "body": mem["body"], "summary": mem["summary"],
        "kind": mem["kind"], "created_at": mem["created_at"],
        "updated_at": mem["updated_at"],
    }


def find_memory_by_title(conn, title: str) -> dict | None:
    """Find a memory node by exact title match."""
    result = conn.execute(
        "MATCH (m:Memory {title: $title}) RETURN m",
        {"title": title},
    )
    if not result.has_next():
        return None
    row = result.get_next()
    d = row_as_dict(row, result.get_column_names())
    mem = d["m"]
    return {
        "id": mem["id"], "title": mem["title"],
        "body": mem["body"], "summary": mem["summary"],
        "kind": mem["kind"], "created_at": mem["created_at"],
        "updated_at": mem["updated_at"],
    }


def link_memories(conn, source_title: str, target_title: str,
                  kind: str = "relates_to", reason: str = "") -> dict:
    """Create a relationship between two memories."""
    source = find_memory_by_title(conn, source_title)
    target = find_memory_by_title(conn, target_title)

    if not source:
        raise ValueError(f"Source memory not found: {source_title}")
    if not target:
        raise ValueError(f"Target memory not found: {target_title}")

    rel_table = "RELATES_TO" if kind == "relates_to" else "ANALOGOUS_TO"

    check = conn.execute(
        f"MATCH (s:Memory {{title: $s}})-[r:{rel_table}]->"
        f"(t:Memory {{title: $t}}) RETURN r",
        {"s": source_title, "t": target_title},
    )
    if check.has_next():
        if rel_table == "ANALOGOUS_TO" and reason:
            conn.execute(
                f"MATCH (s:Memory {{title: $s}})-[r:{rel_table}]->"
                f"(t:Memory {{title: $t}}) "
                f"SET r.reason = $reason",
                {"s": source_title, "t": target_title, "reason": reason},
            )
    else:
        params = {
            "s": source_title, "t": target_title, "kind": kind,
            "aw": 1.0, "cw": 1.0, "nw": 50.0, "conf": 0.5,
            "uc": 0, "usc": 0, "fc": 0,
        }
        if rel_table == "ANALOGOUS_TO":
            params["reason"] = reason
            conn.execute(
                f"MATCH (s:Memory {{title: $s}}) "
                f"MATCH (t:Memory {{title: $t}}) "
                f"CREATE (s)-[r:{rel_table} {{"
                f"  kind: $kind, reason: $reason,"
                f"  accuracy_weight: $aw, creative_weight: $cw,"
                f"  novelty_weight: $nw, confidence: $conf,"
                f"  use_count: $uc, useful_count: $usc, "
                f"  failed_count: $fc, last_used_at: NULL}}]->(t)",
                params,
            )
        else:
            conn.execute(
                f"MATCH (s:Memory {{title: $s}}) "
                f"MATCH (t:Memory {{title: $t}}) "
                f"CREATE (s)-[r:{rel_table} {{"
                f"  kind: $kind,"
                f"  accuracy_weight: $aw, creative_weight: $cw,"
                f"  novelty_weight: $nw, confidence: $conf,"
                f"  use_count: $uc, useful_count: $usc, "
                f"  failed_count: $fc, last_used_at: NULL}}]->(t)",
                params,
            )

    return {
        "source_id": source["id"],
        "source_title": source_title,
        "target_id": target["id"],
        "target_title": target_title,
        "kind": kind,
        "rel_table": rel_table,
    }


def get_neighbors(conn, title: str) -> list[dict]:
    """Get all outgoing relationships from a memory node."""
    memory = find_memory_by_title(conn, title)
    if not memory:
        return []

    result = conn.execute(
        "MATCH (m:Memory {title: $title})-[r]->(t:Memory) "
        "RETURN r, t.title AS target_title",
        {"title": title},
    )
    edges = []
    while result.has_next():
        row = result.get_next()
        d = row_as_dict(row, result.get_column_names())
        edge = _edge_to_dict(d["r"])
        edge["target_title"] = d.get("target_title")
        edges.append(edge)
    return edges


def show_memory(conn, title: str) -> dict | None:
    """Show a single memory with all its details and edges."""
    memory = find_memory_by_title(conn, title)
    if not memory:
        return None

    out_result = conn.execute(
        "MATCH (m:Memory {title: $title})-[r]->(t:Memory) "
        "RETURN r, t.title AS target_title",
        {"title": title},
    )
    outgoing = []
    while out_result.has_next():
        row = out_result.get_next()
        d = row_as_dict(row, out_result.get_column_names())
        edge = _edge_to_dict(d["r"])
        edge["target_title"] = d.get("target_title")
        outgoing.append(edge)

    in_result = conn.execute(
        "MATCH (s:Memory)-[r]->(t:Memory {title: $title}) "
        "RETURN r, s.title AS source_title",
        {"title": title},
    )
    incoming = []
    while in_result.has_next():
        row = in_result.get_next()
        d = row_as_dict(row, in_result.get_column_names())
        edge = _edge_to_dict(d["r"])
        edge["source_title"] = d.get("source_title")
        incoming.append(edge)

    return {**memory, "outgoing_edges": outgoing, "incoming_edges": incoming}


def backfill_embeddings(conn) -> dict:
    """Backfill sentence-transformer embeddings for memories missing them.

    Returns a dict with counts: {"total": N, "updated": M}.
    """
    result = conn.execute(
        "MATCH (m:Memory) "
        "RETURN m.id AS id, m.title AS title, m.body AS body, "
        "m.summary AS summary, m.embedding AS embedding",
    )
    memories = []
    while result.has_next():
        row = result.get_next()
        d = row_as_dict(row, result.get_column_names())
        if d.get("embedding") is not None:
            continue
        title = str(d.get("title", ""))
        body = str(d.get("body", ""))
        summary = str(d.get("summary", ""))
        memories.append({
            "id": d.get("id"),
            "text": f"{title} {summary} {body}",
        })

    if not memories:
        return {"total": 0, "updated": 0}

    vectors = embed_batch([m["text"] for m in memories])
    for mem, vec in zip(memories, vectors):
        conn.execute(
            "MATCH (m:Memory {id: $id}) SET m.embedding = $emb",
            {"id": mem["id"], "emb": vec},
        )

    return {"total": len(memories), "updated": len(memories)}
