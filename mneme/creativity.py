"""Creativity module for Mneme.

Provides creativity scoring for relationship suggestions and
cross-domain link discovery.

Key concept: when two memories appear in the same useful path,
suggest a new relationship between them. Initial confidence is low;
rewarding the suggested link increases confidence over time.
"""

import re

from .embeddings import cosine


def _jaccard_similarity(mem_a: dict, mem_b: dict) -> float:
    """Compute keyword overlap (Jaccard) similarity between two memories."""
    content_a = f"{mem_a.get('title', '')} {mem_a.get('summary', '')} {mem_a.get('body', '')}"
    content_b = f"{mem_b.get('title', '')} {mem_b.get('summary', '')} {mem_b.get('body', '')}"

    words_a = set(re.findall(r'[a-z]+', content_a.lower()))
    words_b = set(re.findall(r'[a-z]+', content_b.lower()))

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def compute_cross_similarity(mem_a: dict, mem_b: dict,
                             conn=None) -> float:
    """Compute a similarity score between two memories.

    When *conn* is provided and both memories have stored embeddings,
    uses cosine similarity on embeddings with a lexical (Jaccard) boost.
    Otherwise falls back to pure keyword overlap.

    Args:
        mem_a: First memory dict (title, body, summary).
        mem_b: Second memory dict.
        conn: Optional Kuzu connection to fetch embeddings from the DB.

    Returns:
        Float between 0.0 and 1.0.
    """
    keyword = _jaccard_similarity(mem_a, mem_b)

    emb_a = mem_a.get("embedding")
    emb_b = mem_b.get("embedding")
    if emb_a and emb_b:
        cos = cosine(list(emb_a), list(emb_b))
        if cos > 0:
            return 0.7 * cos + 0.3 * keyword
    return keyword


def suggest_new_links(path: list[dict], existing_edges: list[dict],
                      threshold: float = 0.05,
                      conn=None) -> list[dict]:
    """Suggest new links between unconnected memories in a path.

    For each pair of visited memories, check if an edge already exists.
    If not, compute similarity and suggest a link if above threshold.

    Args:
        path: List of visited memory dicts from a traversal.
        existing_edges: List of existing edge dicts (source, target, type).
        threshold: Minimum similarity to suggest a link.
        conn: Optional Kuzu connection to pass to compute_cross_similarity.

    Returns:
        List of suggested edge dicts with similarity score and reason.
    """
    suggestions = []
    existing_pairs = set()
    for edge in existing_edges:
        src = edge.get("source_title", "")
        tgt = edge.get("target_title", "")
        existing_pairs.add((src.lower(), tgt.lower()))
        existing_pairs.add((tgt.lower(), src.lower()))  # Undirected check

    for i, node_a in enumerate(path):
        for j, node_b in enumerate(path):
            if i >= j:
                continue
            pair = (node_a.get("title", "").lower(),
                    node_b.get("title", "").lower())
            if pair in existing_pairs:
                continue

            similarity = compute_cross_similarity(node_a, node_b, conn=conn)
            if similarity >= threshold:
                suggestions.append({
                    "source_title": node_a.get("title"),
                    "target_title": node_b.get("title"),
                    "similarity": round(similarity, 4),
                    "initial_creative_weight": max(2, round(similarity * 20, 2)),
                    "confidence": round(min(0.8, 0.3 + similarity * 0.5), 2),
                    "reason": f"Co-occur in same session with {similarity:.0%} content similarity",
                })

    return suggestions
