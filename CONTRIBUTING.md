# Contributing to Mneme

Thanks for your interest. This is a small project, so the workflow is light.

## Setup

```bash
git clone https://github.com/<owner>/mneme
cd mneme
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~80MB) from
HuggingFace. Set `HF_HOME` if you want to control where it's cached.

## Running tests

```bash
pytest tests/
```

All 115 tests should pass on a clean install. If you see failures unrelated to
your changes, double-check that `sentence-transformers` and `kuzu` installed
correctly.

## Coding standards

- No comments unless they explain *why* something non-obvious is happening.
- No trailing summaries of what changed — that's what the diff is for.
- Default to editing existing files; only create new modules when there's no
  good home for the code.
- Keep test coverage tight: parametrize where it makes sense, and add a test
  whenever fixing a bug.

## What I'm looking for

- Bug fixes with a regression test.
- Performance improvements with before/after numbers.
- New traversal modes or scoring functions, with rationale.
- Memory-provider integrations (e.g. for other agent frameworks).

## What I'm wary of

- New heavy dependencies — the install is already non-trivial because of
  `torch`. If your change pulls in another big native library, justify it.
- Breaking schema changes — Kuzu's `IF NOT EXISTS` means existing DBs don't
  auto-migrate. Provide a migration path.
- Refactors without behavioral changes that touch many files. Open an issue
  first to align on direction.

## Pull requests

- One concern per PR. If your change accidentally also fixes a different bug,
  split it.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Mention the issue it closes (if any).

That's it. Open an issue if anything's unclear.
