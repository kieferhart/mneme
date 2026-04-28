# Mneme — Hermes memory provider

Wraps Mneme as a Hermes `MemoryProvider` plugin. Memories live in a Kuzu graph
with sentence-transformer embeddings; edges carry weights and confidence;
useful paths gain weight via citation-grounded reinforcement.

## Install

The plugin lives in `$HERMES_HOME/plugins/mneme/` and is auto-discovered by
hermes alongside the bundled providers.

```bash
# 1. Install Mneme into hermes's Python environment
/path/to/hermes/venv/bin/pip install mneme

# 2. Drop the plugin into hermes's plugins dir
mkdir -p ~/.hermes/plugins/mneme
cp __init__.py plugin.yaml README.md ~/.hermes/plugins/mneme/
```

If you cloned the Mneme repo, those three files are the contents of
`contrib/hermes/`.

## Activate

In `~/.hermes/config.yaml`:

```yaml
memory:
  provider: mneme

plugins:
  mneme:
    db_path: $HERMES_HOME/mneme.kuzu     # default; or any absolute path
    prefetch_top_n: 5
    prefetch_mode: balanced
    traverse_on_prefetch: false
```

Verify with:

```bash
hermes memory status
# should show: "Provider: mneme" + "mneme  (local) ← active"
```

## What you get

- **Per-turn prefetch** of the top-N most relevant memories injected into
  context. Hybrid retrieval (cosine on embeddings + keyword overlap), so it
  surfaces semantically related memories even when keywords don't match.
- **One tool, action-dispatched** — `mneme(action="add"|"link"|"ask"|...)`.
- **Auto-mirroring** of built-in memory writes (`MEMORY.md` / `USER.md` adds
  flow into the graph).

## Tool reference

```jsonc
{"action": "add", "title": "Auto Repair", "body": "How to fix a car engine"}
{"action": "link", "source": "Auto Repair", "target": "Cake Recipe", "kind": "relates_to"}
{"action": "ask", "query": "vehicle motor maintenance", "mode": "balanced"}
{"action": "show", "title": "Auto Repair"}
{"action": "neighbors", "title": "Auto Repair"}
{"action": "reward", "source": "Auto Repair", "target": "Cake Recipe", "amount": 3.0}
{"action": "suggest_links", "titles": ["Auto Repair", "Cake Recipe"]}
```

### `ask` modes

| Mode | Best for |
|---|---|
| `strict` | Factual recall — high lexical anchor, top-1 pick, deterministic |
| `balanced` | Default — 70/30 hybrid, occasional sampling |
| `creative` | Brainstorming — weighted random over top-7 |
| `discovery` | Cross-domain insight — targets the cosine 0.4–0.6 mid-band |

## Config reference

| Key | Default | Purpose |
|---|---|---|
| `db_path` | `$HERMES_HOME/mneme.kuzu` | Where the Kuzu DB lives |
| `prefetch_top_n` | `5` | Memories surfaced per turn |
| `prefetch_mode` | `balanced` | Traversal mode if `traverse_on_prefetch` is true |
| `traverse_on_prefetch` | `false` | Walk the graph during prefetch (heavier) |
| `max_hops` | `3` | Hop limit for prefetch traversal |

The plugin also picks up Mneme's own LLM env vars when present:

| Var | Effect |
|---|---|
| `MNEME_LLM_API_URL` + `MNEME_LLM_API_KEY` | Enables citation rewards and pairwise judging on `ask` |
| `MNEME_LLM_MODEL` | Defaults to `gpt-4o-mini` |
| `HF_HOME` | Where `all-MiniLM-L6-v2` is cached on disk |

If LLM vars are unset, `ask` still works — it just doesn't reinforce edges
automatically. Mneme will also auto-load `~/.config/mneme.env` at import time
if present, so you can put the LLM config there.

## Caveats

- **First load downloads the embedding model** (~80MB).
- **One external memory provider at a time.** Switching from another provider
  doesn't migrate data; both DBs persist, only the active one gets writes.
- **Schema is `IF NOT EXISTS`.** Old Mneme databases (pre-FLOAT[384]) won't
  auto-upgrade. Delete the DB file and re-init if you have an old one.
