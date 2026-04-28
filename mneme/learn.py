"""Learning rules for Mneme.

Implements the fixed-budget reward mechanism:
- Each source node has a fixed outgoing attention budget (total = 100)
- When an edge is rewarded, sibling edges lose weight proportionally
- Separate budgets for accuracy_weight and creative_weight

Budget normalization considers ALL outgoing edges from the source,
regardless of relationship type.
"""

import json
from .utils import now_iso
from . import config


def reward_edge(conn, source_title: str, target_title: str,
                rel_table: str,
                weight_field: str = "accuracy_weight",
                amount: float = 3.0) -> dict:
    """Reward a specific edge using the local fixed-budget mechanism.

    Budget normalization considers ALL outgoing edges from the source
    across all relationship types.
    """
    # Fetch ALL outgoing edges from source (across all rel types)
    fetch = conn.execute(
        "MATCH (s:Memory {title: $src})-[r]->(t:Memory) "
        "RETURN t.title AS tgt, r, "
        "COALESCE(r.accuracy_weight, 0) AS aw, "
        "COALESCE(r.creative_weight, 0) AS cw",
        {"src": source_title},
    )
    all_rows = fetch.get_all()

    # Build edge list with rel_type so we know which table to update
    edges = []
    for row in all_rows:
        tgt = row[0]
        edge = row[1] if isinstance(row, list) else row.get("r")
        rel_label = edge.get("_label", "") if isinstance(edge, dict) else ""
        edges.append({
            "target": tgt,
            "edge": edge,
            "rel_table": rel_label,
            "aw": float(row[2]),
            "cw": float(row[3]),
        })

    if not edges:
        raise ValueError(
            f"No relationships from '{source_title}' of type {rel_table}"
        )

    target_edge = next((e for e in edges if e["target"] == target_title), None)
    if target_edge is None:
        raise ValueError(
            f"Relationship not found from '{source_title}' to "
            f"'{target_title}' of type {rel_table}"
        )

    before_aw = float(target_edge["edge"].get("accuracy_weight") or 0)
    before_cw = float(target_edge["edge"].get("creative_weight") or 0)

    # Map weight field names to short keys used in the edges list
    _field_map = {"accuracy_weight": "aw", "creative_weight": "cw"}

    # Determine which budget to boost; only that one changes
    if weight_field == "accuracy_weight":
        chosen_wfield = "accuracy_weight"
        chosen_before = before_aw
        other_field = "creative_weight"
    else:
        chosen_wfield = "creative_weight"
        chosen_before = before_cw
        other_field = "accuracy_weight"

    short_key = _field_map[chosen_wfield]

    # Normalize ONLY the chosen budget across ALL outgoing edges
    siblings = [e for e in edges if e["target"] != target_title]

    if not siblings:
        # Single edge — set the rewarded budget to the full budget (100)
        chosen_after = 100.0
        conn.execute(
            f"MATCH (s:Memory {{title: $src}})-[r:{rel_table}]->"
            f"(t:Memory {{title: $tgt}}) "
            f"SET r.{chosen_wfield} = $val",
            {"src": source_title, "tgt": target_title, "val": 100.0},
        )
    else:
        chosen_with_reward = chosen_before + amount
        chosen_after = chosen_before + amount
        remaining_budget = 100.0 - chosen_with_reward
        sibling_total = sum(e[short_key] for e in siblings)
        if sibling_total <= 0:
            per = remaining_budget / len(siblings)
            for s in siblings:
                conn.execute(
                    f"MATCH (s:Memory {{title: $src}})-[r:{s['rel_table']}]->"
                    f"(t:Memory {{title: $tgt}}) "
                    f"SET r.{chosen_wfield} = $val",
                    {"src": source_title, "tgt": s["target"], "val": per},
                )
        else:
            for s in siblings:
                new_w = (s[short_key] / sibling_total) * remaining_budget
                conn.execute(
                    f"MATCH (s:Memory {{title: $src}})-[r:{s['rel_table']}]->"
                    f"(t:Memory {{title: $tgt}}) "
                    f"SET r.{chosen_wfield} = $val",
                    {"src": source_title, "tgt": s["target"],
                     "val": round(new_w, 6)},
                )

    # Update chosen edge with new weight values + counters
    ts = now_iso()
    conn.execute(
        f"MATCH (s:Memory {{title: $s}})-[r:{rel_table}]->"
        f"(t:Memory {{title: $t}}) "
        f"SET r.accuracy_weight = $aw, "
        f"r.creative_weight = $cw, "
        f"r.use_count = COALESCE(r.use_count, 0) + 1, "
        f"r.useful_count = COALESCE(r.useful_count, 0) + 1, "
        f"r.last_used_at = $ts",
        {"s": source_title, "t": target_title,
         "aw": chosen_after if weight_field == "accuracy_weight" else before_aw,
         "cw": chosen_after if weight_field == "creative_weight" else before_cw,
         "ts": ts},
    )

    return {
        "source_title": source_title,
        "target_title": target_title,
        "rel_type": rel_table,
        "weight_field": weight_field,
        "amount": amount,
        "before_accuracy_weight": before_aw,
        "after_accuracy_weight": chosen_after if weight_field == "accuracy_weight" else before_aw,
        "before_creative_weight": before_cw,
        "after_creative_weight": chosen_after if weight_field == "creative_weight" else before_cw,
    }


def _format_path_for_llm(path: list[dict]) -> str:
    """Format a traversal path as numbered source lines for the LLM.

    Each node becomes ``[ID] Title: body[:200]``.
    """
    lines = []
    for i, node in enumerate(path):
        nid = node.get("id", str(i))
        title = node.get("title", "Untitled")
        body = (node.get("body", "") or "")[:200]
        lines.append(f"[{nid}] {title}: {body}")
    return "\n".join(lines)


def apply_citation_rewards(conn, path: list[dict], edges: list[dict],
                           query: str) -> dict:
    """Citation-grounded reward: ask an LLM which sources were used, reward them.

    Steps:
      1. If no LLM key configured → return skipped status.
      2. Format the path for the LLM.
      3. Call the OpenAI-compatible API with a structured-output prompt.
      4. Parse response; drop citations whose IDs are not in the path.
      5. For each valid cited memory Mk (k > 0), reward edge M(k-1) → Mk.
      6. Return status dict.

    Args:
        conn: Kuzu connection (passed through to ``reward_edge``).
        path: List of visited memory dicts from a traversal.
        edges: List of edge dicts from the traversal.
        query: The original user query.

    Returns:
        Dict with keys ``status``, ``citations``, ``edges_rewarded``.
    """
    if not config.is_llm_configured():
        return {"status": "skipped", "reason": "no LLM API key"}

    try:
        import urllib.request, urllib.error
    except ImportError:
        return {"status": "error", "reason": "urllib not available"}

    formatted = _format_path_for_llm(path)

    # Build set of valid path IDs for filtering
    valid_ids = {node.get("id") for node in path}

    system_prompt = (
        "You are an answer generator with sources. "
        "Return JSON with 'answer' and 'cited' fields."
    )
    user_prompt = (
        f"Query: {query}\n\n"
        f"Sources:\n{formatted}\n\n"
        "Answer the query using the sources above. "
        "Cite which source IDs you actually used."
    )

    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "citation_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "cited": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "used_for": {"type": "string"},
                                },
                                "required": ["id", "used_for"],
                            },
                        },
                    },
                    "required": ["answer", "cited"],
                },
            },
        },
        "temperature": 0,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        config.LLM_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.LLM_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
        # OpenAI-compatible response
        content = body["choices"][0]["message"]["content"]
        result = json.loads(content)
    except Exception as exc:
        return {"status": "error", "reason": f"LLM call failed: {exc}"}

    # Filter citations to only those actually in the path
    raw_citations = result.get("cited", [])
    citations = [
        c for c in raw_citations
        if str(c.get("id")) in valid_ids
    ]

    # Reward edges M(k-1) → Mk for each cited memory
    edges_rewarded = 0
    id_to_index = {node.get("id"): i for i, node in enumerate(path)}
    cited_ids = set()
    for c in citations:
        cited_id = str(c.get("id"))
        if cited_id in id_to_index and cited_id not in cited_ids:
            idx = id_to_index[cited_id]
            if idx > 0:
                prev_title = path[idx - 1].get("title", "")
                curr_title = path[idx].get("title", "")
                rel_type = edges[idx - 1].get("rel_type", "RELATES_TO")
                if prev_title and curr_title:
                    try:
                        reward_edge(
                            conn,
                            prev_title,
                            curr_title,
                            rel_type.upper(),
                            amount=config.CITATION_REWARD_AMOUNT,
                        )
                        edges_rewarded += 1
                    except Exception:
                        pass
            cited_ids.add(cited_id)

    return {
        "status": "done",
        "citations": [c.get("id") for c in citations],
        "edges_rewarded": edges_rewarded,
    }


def apply_pairwise_rewards(conn, path_result: dict, query: str,
                           mode: str = "balanced") -> None:
    """Stubs for pairwise reward application (Direction B).

    Delegates to :func:`mneme.pairwise.run_pairwise_async`,
    which runs in a background thread pool.
    """
    from .pairwise import run_pairwise_async
    run_pairwise_async(conn, query, path_result, mode=mode)
