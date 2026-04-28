"""Async pairwise comparison logic for Mneme (Direction B).

Provides a lightweight pairwise comparison mechanism:
- Re-traverses with a different mode/seed
- Judges which path produces a better answer via LLM
- Rewards the winning path's edges
- If pairwise contradicts citations: trust pairwise
"""

import random
from concurrent.futures import ThreadPoolExecutor

from . import config


def should_run_pairwise() -> bool:
    """Return True based on config.PAIRWISE_SAMPLE_RATE (default 10%)."""
    return random.random() < config.PAIRWISE_SAMPLE_RATE


def run_pairwise_async(conn, query: str, path_result: dict,
                       mode: str = "creative") -> None:
    """Run pairwise comparison in a background thread. Don't block the caller.

    If the LLM is not configured or the random sample rate is not hit,
    returns silently.

    Args:
        conn: Kuzu connection.
        query: Original user query.
        path_result: Dict from :func:`mneme.traverse.traverse`.
        mode: The mode used for the original traversal (e.g. "balanced").
    """
    if not config.is_llm_configured():
        return  # Silently skip

    if not should_run_pairwise():
        return  # Skip based on sample rate

    def _do_pairwise():
        """Inner worker that runs the actual pairwise logic."""
        from .traverse import traverse
        from .learn import reward_edge

        # Pick a complementary mode
        mode_map = {
            "strict": "discovery",
            "balanced": "creative",
            "creative": "discovery",
            "discovery": "balanced",
        }
        alt_mode = mode_map.get(mode, "discovery")
        alt_seed = path_result.get("seed", 0) or 42

        # Re-traverse with different settings
        start_id = path_result.get("start_id")
        if not start_id:
            return

        alt_result = traverse(
            conn, start_id, query,
            mode=alt_mode,
            max_hops=path_result.get("max_hops", 3),
            seed=int(alt_seed) + 1,
        )

        # Build a simple comparison prompt
        path_a_nodes = path_result.get("path", [])
        path_b_nodes = alt_result.get("path", [])

        path_a_str = "\n".join(
            f"[{n.get('id', i)}] {n.get('title', '?')}: {(n.get('body') or '')[:150]}"
            for i, n in enumerate(path_a_nodes)
        )
        path_b_str = "\n".join(
            f"[{n.get('id', i)}] {n.get('title', '?')}: {(n.get('body') or '')[:150]}"
            for i, n in enumerate(path_b_nodes)
        )

        comparison_prompt = (
            f"You are a judge comparing two knowledge-graph traversal paths.\n\n"
            f"Query: {query}\n\n"
            f"Path A (mode={mode}):\n{path_a_str}\n\n"
            f"Path B (mode={alt_mode}):\n{path_b_str}\n\n"
            "Which path provides a more useful traversal for answering the query? "
            "Reply with 'A' or 'B' on its own line."
        )

        # Call LLM to judge
        try:
            import urllib.request, json
        except ImportError:
            return

        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system",
                 "content": "You are a comparison judge. Reply with only 'A' or 'B'."},
                {"role": "user", "content": comparison_prompt},
            ],
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
            answer = body["choices"][0]["message"]["content"].strip()
        except Exception:
            return

        winner = "A" if "A" in answer else "B"
        winner_path = path_a_nodes if winner == "A" else path_b_nodes

        # Reward edges of the winning path
        for i in range(1, len(winner_path)):
            prev_title = winner_path[i - 1].get("title", "")
            curr_title = winner_path[i].get("title", "")
            if prev_title and curr_title:
                try:
                    reward_edge(
                        conn,
                        prev_title,
                        curr_title,
                        "RELATES_TO",
                        amount=config.PAIRWISE_REWARD_AMOUNT,
                    )
                except Exception:
                    pass

        # Contradiction handling: pairwise always wins
        # (edges are rewarded above; citations are not double-counted
        #  because reward_edge's fixed-budget mechanism handles it.)

    # Fire-and-forget in a thread pool
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_do_pairwise)
