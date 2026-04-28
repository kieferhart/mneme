"""Traversal engine for Mneme.

Implements graph traversal across weighted relationships with 4 modes:
- strict: factual, deterministic, accuracy-focused (high cosine threshold)
- balanced: default, small novelty bonus (middle mix)
- creative: brainstorming, samples from top candidates (relaxed cosine)
- discovery: finds unusual connections via the analogical mid-band
  (cosine 0.4–0.6 — related but not obviously)

Hybrid retrieval combines sentence-transformer cosine similarity with
keyword overlap (default 0.7 / 0.3 mix).
"""

import random
import re

from .embeddings import embed, cosine
from .utils import row_as_dict, _edge_to_dict


COSINE_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3
DISCOVERY_BAND = (0.4, 0.6)


def keyword_overlap(query: str, text: str) -> float:
    """Calculate keyword overlap between query and text."""
    query_words = set(re.findall(r"[a-z]+", query.lower()))
    text_words = set(re.findall(r"[a-z]+", text.lower()))
    if not query_words:
        return 0.0
    overlap = query_words & text_words
    return len(overlap) / len(query_words)


def hybrid_similarity(query: str, query_vec: list[float],
                      target: dict) -> float:
    """Combine cosine similarity (target.embedding vs query_vec) with
    keyword overlap on title+summary+body.

    Falls back to keyword-only when the target has no stored embedding.
    """
    content = (
        f"{target.get('title', '')} "
        f"{target.get('summary', '')} "
        f"{target.get('body', '')}"
    )
    lex = keyword_overlap(query, content)
    emb = target.get("embedding")
    if emb and query_vec:
        cos = cosine(query_vec, list(emb))
        return COSINE_WEIGHT * cos + LEXICAL_WEIGHT * lex
    return lex


def _band_proximity(cos_score: float) -> float:
    """Return how close ``cos_score`` is to the discovery mid-band 0.4–0.6.

    1.0 inside the band; falls off linearly outside.
    """
    lo, hi = DISCOVERY_BAND
    if lo <= cos_score <= hi:
        return 1.0
    if cos_score < lo:
        return max(0.0, 1.0 - (lo - cos_score) / lo)
    return max(0.0, 1.0 - (cos_score - hi) / max(1e-6, 1.0 - hi))


def score_edge(edge: dict, target_memory: dict, query: str,
               query_vec: list[float], mode: str,
               rng: random.Random) -> float:
    """Score a candidate edge for traversal in a given mode."""
    aw_norm = edge.get("accuracy_weight", 1.0) / 100.0
    cw_norm = edge.get("creative_weight", 1.0) / 100.0
    novelty_norm = edge.get("novelty_weight", 50.0) / 100.0
    confidence = edge.get("confidence", 0.5)

    emb = target_memory.get("embedding")
    cos_score = cosine(query_vec, list(emb)) if (emb and query_vec) else 0.0

    if mode == "strict":
        # High cosine + heavy lexical anchor — factual, exact-match favored.
        sim = 0.5 * cos_score + 0.5 * keyword_overlap(
            query,
            f"{target_memory.get('title', '')} "
            f"{target_memory.get('summary', '')} "
            f"{target_memory.get('body', '')}",
        )
        return (0.70 * sim + 0.25 * aw_norm
                + 0.03 * confidence + 0.02 * rng.random())

    sim = hybrid_similarity(query, query_vec, target_memory)

    if mode == "balanced":
        return (0.55 * sim + 0.25 * aw_norm
                + 0.10 * confidence + 0.07 * novelty_norm
                + 0.03 * rng.random())
    if mode == "creative":
        return (0.40 * sim + 0.20 * aw_norm
                + 0.20 * cw_norm + 0.15 * novelty_norm
                + 0.05 * rng.random())
    if mode == "discovery":
        # Reward proximity to the analogical mid-band rather than raw similarity.
        band = _band_proximity(cos_score)
        return (0.45 * band + 0.10 * sim + 0.10 * aw_norm
                + 0.20 * cw_norm + 0.10 * novelty_norm
                + 0.05 * rng.random())
    raise ValueError(f"Unknown mode: {mode}")


def find_candidate_nodes(conn, query: str, top_n: int = 5) -> list[dict]:
    """Find starting nodes using hybrid retrieval (cosine + keyword overlap)."""
    query_vec = embed(query)

    result = conn.execute(
        "MATCH (m:Memory) RETURN m.id AS id, m.title AS title, "
        "m.body AS body, m.summary AS summary, m.kind AS kind, "
        "m.embedding AS embedding",
    )
    scored = []
    while result.has_next():
        row = result.get_next()
        d = row_as_dict(row, result.get_column_names())
        emb = d.get("embedding")
        mem = {
            "id": d.get("id"), "title": d.get("title"),
            "body": d.get("body"), "summary": d.get("summary"),
            "kind": d.get("kind"),
            "embedding": emb,
        }

        content = f"{mem.get('title', '')} {mem.get('summary', '')} {mem.get('body', '')}"
        kw_score = keyword_overlap(query, content)
        cos_score = cosine(query_vec, list(emb)) if emb else 0.0

        if emb:
            combined = COSINE_WEIGHT * cos_score + LEXICAL_WEIGHT * kw_score
        else:
            combined = kw_score

        if combined > 0:
            scored.append({
                **mem,
                "overlap_score": round(combined, 6),
                "cosine_score": round(cos_score, 6),
                "keyword_score": round(kw_score, 6),
            })

    scored.sort(key=lambda x: x["overlap_score"], reverse=True)
    return scored[:top_n]


def traverse(conn, start_id: str, query: str,
             mode: str = "balanced", max_hops: int = 3,
             seed: int = None) -> dict:
    """Traverse the graph from a starting node."""
    rng = random.Random(seed) if seed is not None else random.Random()
    query_vec = embed(query)

    path = []
    edge_scores = []
    visited = set()

    start_result = conn.execute(
        "MATCH (m:Memory {id: $id}) "
        "RETURN m.title AS title, m.body AS body, "
        "m.summary AS summary, m.embedding AS embedding",
        {"id": start_id},
    )
    if not start_result.has_next():
        return {
            "start_id": start_id, "mode": mode, "max_hops": max_hops,
            "seed": rng.random(), "path": [], "edges": [],
            "visited_count": 0,
        }

    start_row = start_result.get_next()
    sd = row_as_dict(start_row, start_result.get_column_names())
    current_info = {
        "id": start_id,
        "title": sd.get("title"),
        "body": sd.get("body", ""),
        "summary": sd.get("summary", ""),
        "embedding": sd.get("embedding"),
    }
    path.append(current_info)
    visited.add(start_id)

    for hop in range(max_hops):
        current_id = path[-1]["id"]

        edges_result = conn.execute(
            "MATCH (m {id: $id})-[r]->(t:Memory) "
            "RETURN r, t.id AS target_id, t.title AS target_title, "
            "t.body AS target_body, t.summary AS target_summary, "
            "t.embedding AS target_embedding",
            {"id": current_id},
        )
        candidates = []
        while edges_result.has_next():
            row = edges_result.get_next()
            d = row_as_dict(row, edges_result.get_column_names())
            target_id = d.get("target_id")
            if target_id in visited:
                continue
            edge_info = _edge_to_dict(d["r"])
            target_info = {
                "id": target_id,
                "title": d.get("target_title"),
                "body": d.get("target_body", ""),
                "summary": d.get("target_summary", ""),
                "embedding": d.get("target_embedding"),
            }
            score = score_edge(edge_info, target_info, query, query_vec, mode, rng)
            candidates.append({
                "edge": edge_info,
                "target": target_info,
                "score": score,
            })

        if not candidates:
            break

        candidates.sort(key=lambda x: x["score"], reverse=True)

        if mode == "strict":
            chosen = candidates[0]
        elif mode == "balanced":
            if rng.random() < 0.15 and len(candidates) > 1:
                chosen = rng.choice(candidates[:min(3, len(candidates))])
            else:
                chosen = candidates[0]
        elif mode == "creative":
            top_k = min(7, len(candidates))
            top_cands = candidates[:top_k]
            scores = [c["score"] for c in top_cands]
            total = sum(scores)
            if total > 0:
                weights = [s / total for s in scores]
                chosen_idx = rng.choices(
                    range(len(top_cands)), weights=weights, k=1)[0]
                chosen = top_cands[chosen_idx]
            else:
                chosen = top_cands[0]
        elif mode == "discovery":
            top_k = min(7, len(candidates))
            top_cands = candidates[:top_k]
            for c in top_cands:
                if c["edge"].get("kind") == "ANALOGOUS_TO":
                    c["score"] *= 1.2
            top_cands.sort(key=lambda x: x["score"], reverse=True)
            scores = [c["score"] for c in top_cands]
            total = sum(scores)
            if total > 0:
                weights = [s / total for s in scores]
                chosen_idx = rng.choices(
                    range(len(top_cands)), weights=weights, k=1)[0]
                chosen = top_cands[chosen_idx]
            else:
                chosen = top_cands[0]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        edge_scores.append({
            "from": path[-1]["title"] if path else None,
            "to": chosen["target"]["title"],
            "rel_type": chosen["edge"].get("kind"),
            "edge_kind": chosen["edge"].get("kind"),
            "score": round(chosen["score"], 4),
            "reason": chosen["edge"].get("reason"),
        })

        visited.add(chosen["target"]["id"])
        path.append(chosen["target"])

    return {
        "start_id": start_id,
        "mode": mode,
        "max_hops": max_hops,
        "seed": rng.random(),
        "path": path,
        "edges": edge_scores,
        "visited_count": len(visited),
    }
