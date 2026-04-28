# Changelog

All notable changes to Mneme will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Hermes plugin: multi-process Kuzu lock contention.** The plugin used to
  hold a long-lived Kuzu connection, which locked out any other hermes
  process (gateway vs. TUI vs. cron) trying to use the same DB. It now
  opens and closes the connection per operation, with bounded
  exponential-backoff retry on lock collisions (defaults: 20 attempts,
  ~37s max wait). Pairwise rewards are skipped from the plugin path
  because they outlive the lock window; run them via the `mneme` CLI.

## [0.1.0] — 2026-04-27

Initial release.

### Added
- Kuzu-backed weighted memory graph with `Memory` nodes and typed
  `RELATES_TO` / `ANALOGOUS_TO` edges.
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`, 384-dim) with hybrid
  retrieval (cosine + keyword overlap, 70/30).
- Four traversal modes: `strict`, `balanced`, `creative`, `discovery`.
  Discovery mode targets the analogical mid-band (cosine 0.4–0.6).
- Fixed-budget reward mechanic: per-source attention budget stays at 100
  across all outgoing edges; rewarded edges gain at the expense of siblings.
- Citation-grounded reinforcement: an LLM call (OpenAI-compatible endpoint)
  answers the query with a structured citation list, which drives edge
  rewards on the traversed path.
- Pairwise comparison on a configurable sample (~10% by default), running
  asynchronously in a background thread; the LLM judge's winning path gets
  a stronger reward (5.0 vs the citation default of 2.0).
- Auto-loading of `~/.config/mneme.env` at import time so CLI users and
  embedded callers get the same environment without manual sourcing.
- Hermes memory-provider plugin under `contrib/hermes/` exposing a single
  action-dispatched `mneme` tool with prefetch + memory-write mirroring.
- 115 tests covering reward invariants, hybrid retrieval, citation parsing,
  pairwise sampling, and traversal modes.
