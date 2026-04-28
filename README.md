# Mneme — Local AI Memory Graph

Mneme is a local-first memory system: a weighted, typed property graph
that gets smarter the more you use it. Memories are nodes, relationships are
edges, and edges that walk a useful path gain weight at the expense of their
siblings. The total outgoing attention budget for any node is fixed at 100,
so the graph is always making trade-offs about where to send attention next.

Built on [Kuzu](https://kuzudb.com/) (embedded graph DB) with sentence-transformer
embeddings for semantic retrieval and an OpenAI-compatible LLM call for
citation-grounded reinforcement.

---

## The core idea

```
Memory is stored as nodes and typed relationships.
Useful paths through memory gain weight.
Other paths from the same source lose relative weight.
The total outgoing attention budget stays fixed at 100.
```

Three things make this more than a fancy graph:

1. **Hybrid retrieval.** Starting nodes are picked by `0.7 * cosine(query, memory)
   + 0.3 * keyword_overlap`. Cosine uses 384-dim vectors from
   `all-MiniLM-L6-v2`; the lexical signal is preserved because personal memory
   graphs are full of proper nouns and identifiers where exact match matters.

2. **Citation-grounded rewards.** After a traversal, the path is handed to an
   LLM that answers the query and emits a structured list of which memories it
   actually used. Edges leading to cited memories get rewarded; everything else
   passively decays. The reinforcement loop closes itself — no manual `reward`
   call needed.

3. **Pairwise judging on a sample.** ~10% of `ask` calls trigger a background
   re-traversal under a different mode and an LLM judge picks the winner. The
   winner's edges get a stronger reward (5.0 vs 2.0). Sparse but high-signal,
   asynchronous so the user never waits.

---

## Install

```bash
# from PyPI (once published)
pip install mneme

# or from source
git clone https://github.com/<owner>/mneme && cd mneme
pip install -e .
```

> ⚠️ **First-run download.** The first `mneme add` or `mneme ask` will
> download the embedding model (`all-MiniLM-L6-v2`, ~80MB) from HuggingFace.
> You need network access on first use. Set `HF_HOME` to control where the
> model is cached. The full install also pulls `torch` (~2GB) as a
> transitive dep of `sentence-transformers`.

## Quick start

```bash
mneme init
mneme add "My First Memory" "This is the body of my first memory."
mneme add "Another Memory" "This is another memory."
mneme link "My First Memory" "Another Memory" --kind relates_to
mneme ask "Explore my memories" --mode balanced
```

The default database lives at the platform's user-data directory:
- Linux: `~/.local/share/mneme/db.kuzu`
- macOS: `~/Library/Application Support/mneme/db.kuzu`
- Windows: `%LOCALAPPDATA%\mneme\db.kuzu`

Override with the `MNEME_DB` environment variable.

---

## Commands

| Command | Purpose |
|---|---|
| `mneme init` | Create the database and schema |
| `mneme add TITLE BODY [--kind K]` | Add a memory (auto-embeds on insert) |
| `mneme embed` | Backfill embeddings for any memory missing one (legacy) |
| `mneme link SRC TGT [--kind K] [--reason R]` | Create an edge between two memories |
| `mneme neighbors TITLE` | Show outgoing edges and weights |
| `mneme ask QUERY [--mode M] [--max-hops N]` | Traverse the graph; closes the reinforcement loop |
| `mneme reward --session S --source X --target Y [--weight-field F] [--amount A]` | Manually reward an edge |
| `mneme show TITLE` | Show a memory + all its incoming/outgoing edges |
| `mneme sessions` | List past traversal sessions |

---

## Traversal modes

Modes aren't just randomness knobs — they target different similarity regimes.

| Mode | Selection | Similarity profile |
|------|-----------|--------------------|
| `strict` | Top-1, deterministic | Heavy lexical anchor + high cosine; for factual recall |
| `balanced` | Top-1 with 15% top-3 sample | Standard 70/30 hybrid mix |
| `creative` | Weighted-random over top-7 | Hybrid mix; novelty/creative weights count more |
| `discovery` | Weighted-random over top-7, ANALOGOUS_TO ×1.2 | Targets the **0.4–0.6 cosine band** — the analogical sweet spot, where memories are related to the query but not obviously |

Discovery is implemented via `_band_proximity` (traverse.py): score peaks
inside the band and falls off linearly outside, instead of just rewarding raw
similarity.

---

## End-to-end flow of `mneme ask`

```
                 query
                   │
                   ▼
   ┌─────────────────────────────────┐
   │ find_candidate_nodes()          │   embed query → cosine vs every
   │                                 │   memory's stored embedding,
   │   0.7·cosine + 0.3·keyword      │   hybrid-rank top N
   └─────────────────────────────────┘
                   │
                   ▼
   ┌─────────────────────────────────┐
   │ traverse(start, query, mode)    │   per hop: score_edge() over
   │                                 │   outgoing edges using mode-
   │   hop 0 → hop 1 → … → hop N     │   specific weights, then sample
   └─────────────────────────────────┘
                   │
                   ▼
   ┌─────────────────────────────────┐
   │ create_session() +              │   USED_IN_SESSION rels logged
   │ log_session_edges()             │   for traceability
   └─────────────────────────────────┘
                   │
                   ▼
   ┌─────────────────────────────────┐   (synchronous, in-process)
   │ apply_citation_rewards()        │   LLM answers query w/ structured
   │                                 │   citation list → reward each cited
   │   uses learn.reward_edge()      │   edge by CITATION_REWARD_AMOUNT (2.0)
   └─────────────────────────────────┘
                   │
                   ▼
   ┌─────────────────────────────────┐   (async, ThreadPoolExecutor,
   │ apply_pairwise_rewards()        │    fires on PAIRWISE_SAMPLE_RATE
   │                                 │    of calls — default 10%)
   │   re-traverse w/ alt mode +     │   reward winner's edges by
   │   LLM judge picks winner        │   PAIRWISE_REWARD_AMOUNT (5.0)
   └─────────────────────────────────┘
                   │
                   ▼
            results to user
```

### What happens in `reward_edge`

The fixed-budget mechanic. For a chosen `(source, target, weight_field)`:

1. Fetch all outgoing edges from `source` (across all rel tables).
2. The chosen edge's `weight_field` gains `amount`.
3. Siblings are renormalized so the chosen budget across all of `source`'s
   outgoing edges sums to 100.
4. The other budget (`accuracy_weight` if you rewarded `creative_weight`, etc.)
   is left untouched. This was the behavior fix for the original code review —
   `weight_field` actually means something now.
5. `use_count`, `useful_count`, and `last_used_at` get bumped.

The invariant is asserted in `tests/test_learn.py::TestBudgetInvariant`.

---

## Dependencies

Runtime dependencies (`pyproject.toml`):

| Package | Why |
|---|---|
| `kuzu` | Embedded property-graph database |
| `typer` | CLI framework |
| `rich` | Pretty terminal output |
| `pydantic` | Data validation (light use) |
| `numpy` | Numeric operations |
| `sentence-transformers` | Embedding model (`all-MiniLM-L6-v2`, 384-dim, ~80MB). Pulls in `torch` transitively, which is the heavy part of the install. |
| `platformdirs` | Cross-platform user-data directory resolution for the default DB path. |

LLM features (citations, pairwise) are optional and gated by environment
variables (see Configuration). Without them the system runs but the
reinforcement loop stays passive — edges only move when you call `mneme reward`
manually.

---

## Configuration

All via environment variables (see `mneme/config.py`).

| Variable | Default | Purpose |
|---|---|---|
| `MNEME_DB` | `<user-data-dir>/mneme/db.kuzu` | Path to the Kuzu database file |
| `MNEME_LLM_API_URL` | (unset) | OpenAI-compatible chat completions endpoint |
| `MNEME_LLM_API_KEY` | (unset) | API key — both URL and key required to enable LLM features |
| `MNEME_LLM_MODEL` | `gpt-4o-mini` | Model name to send |
| `MNEME_PAIRWISE_SAMPLE_RATE` | `0.10` | Fraction of `ask` calls that trigger pairwise judging |
| `MNEME_CITATION_REWARD_AMOUNT` | `2.0` | Weight gain per cited edge |
| `MNEME_PAIRWISE_REWARD_AMOUNT` | `5.0` | Weight gain per edge on the pairwise winner's path |
| `HF_HOME` / `TRANSFORMERS_CACHE` | (HuggingFace defaults) | Where the embedding model is cached on disk |

Mneme also auto-loads `~/.config/mneme.env` at import time so you don't need
to source it before running the CLI. Existing `os.environ` values always win;
the file only fills gaps.

If LLM env vars are unset, `apply_citation_rewards` and `apply_pairwise_rewards`
silently no-op — the rest of the system still works.

---

## Hermes integration

A memory-provider plugin for the [Hermes Agent](https://github.com/) lives at
[`contrib/hermes/`](contrib/hermes/). It exposes a single action-dispatched
`mneme` tool with prefetch + memory-write mirroring. See its README for setup.

---

## Schema

Two node tables and three relationship tables (defined in `schema.py`):

```
Memory(id, title, body, summary, kind, created_at, updated_at, embedding FLOAT[384])
Session(id, user_query, mode, created_at)

(Memory) -[RELATES_TO]-> (Memory)
   kind, accuracy_weight, creative_weight, novelty_weight,
   confidence, use_count, useful_count, failed_count, last_used_at

(Memory) -[ANALOGOUS_TO]-> (Memory)
   reason, kind, accuracy_weight, creative_weight, novelty_weight,
   confidence, use_count, useful_count, failed_count, last_used_at

(Session) -[USED_IN_SESSION]-> (Memory)
   role, usefulness, step_order
```

The `embedding` column is fixed-size `FLOAT[384]` — Kuzu round-trips it as a
Python list, no string parsing needed.

---

## Module map

```
mneme/
  mneme/
    __init__.py
    cli.py          # Typer commands; the only place stdout output lives
    config.py       # Env-var-backed settings + is_llm_configured()
    db.py           # Kuzu connection helper (path resolution)
    schema.py       # CREATE NODE/REL TABLE IF NOT EXISTS DDL
    memory.py       # add/find/link memories, get_neighbors, show, backfill
    embeddings.py   # all-MiniLM-L6-v2 loader, embed/embed_batch, cosine
    traverse.py     # find_candidate_nodes, score_edge, traverse, hybrid_similarity
    creativity.py   # compute_cross_similarity (Jaccard + cosine), suggest_new_links
    learn.py        # reward_edge (fixed-budget), apply_citation_rewards,
                    # apply_pairwise_rewards (delegates to pairwise.py)
    pairwise.py     # async re-traverse + LLM-judged reward
    session.py      # create_session, log_session_edges, list_sessions
    utils.py        # slugify, now_iso, summarize_simple, row_as_dict, _edge_to_dict
  contrib/
    hermes/         # Hermes Agent memory-provider plugin
  tests/
    conftest.py     # `conn` fixture: fresh tmp Kuzu DB per test
    test_*.py       # 115 tests
  .github/
    workflows/test.yml   # CI: pytest on Python 3.11 + 3.12
  CHANGELOG.md
  CONTRIBUTING.md
  LICENSE
  pyproject.toml
```

---

## Tests

```bash
pytest tests/
```

Coverage:

- **test_learn.py** — fixed-budget invariant, mixed-rel-type normalization,
  single-edge cap, error paths, return shape.
- **test_memory.py** — add/link/find/show roundtrips.
- **test_traverse.py** — keyword overlap, score_edge per mode, find_candidate_nodes.
- **test_embeddings.py** — embed shape + L2-norm, hybrid retrieval,
  synonym matching ("vehicle motor" → "Auto Repair") proves semantic embeddings
  are doing their job.
- **test_creativity.py** — cross-similarity, link suggestion.
- **test_citation.py** / **test_pairwise.py** — citation parsing, hallucination
  drop, pairwise sample-rate gating, alt-mode mapping.
- **test_utils.py** — small utilities.

The `conn` fixture creates a fresh tmpfile-backed Kuzu DB per test and
deletes it on teardown.

---

## Operational notes

- **First-run cost.** The embedding model is ~80MB. Downloading it requires
  network on first use; subsequent runs read from the HF cache. Document this
  if you ship the project.
- **Per-`add` cost.** Embedding a single memory is 5–50ms on CPU. A bulk import
  of thousands should use `embed_batch` (already used by `backfill_embeddings`).
- **Heavy install.** `sentence-transformers` pulls in `torch`. The wheel is
  large; expect a slow first `pip install -e .`.
- **Kuzu schema is `IF NOT EXISTS`.** Existing databases keep their old schema.
  If you've upgraded from a pre-FLOAT[384] version, delete the database file
  and re-init — there's no migration script.
- **LLM calls are best-effort.** Citation parsing and pairwise judging both
  swallow exceptions and continue. Failures don't block traversal.
- **Pairwise is fire-and-forget.** It uses a one-shot `ThreadPoolExecutor` per
  call. Errors in the background don't surface to the user; check process
  stderr if you suspect something's wrong.
