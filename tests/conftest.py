"""Pytest fixtures for Mneme tests.

Provides fresh Kuzu database + connection fixtures with automatic setup/teardown.
"""

import gc
import os
import tempfile

import pytest


@pytest.fixture()
def conn():
    """Create a fresh Kuzu database + connection with schema initialized.

    Returns (conn, db, path).  Cleanup is done automatically by the context.
    """
    import kuzu
    from mneme.schema import init_schema

    path = tempfile.mktemp(suffix=".db")
    db = kuzu.Database(path, auto_checkpoint=False, max_db_size=64 * 1024 * 1024)
    connection = kuzu.Connection(db)
    init_schema(connection)

    yield connection, db, path

    # Teardown
    del connection
    del db
    gc.collect()
    if os.path.exists(path):
        os.unlink(path)
