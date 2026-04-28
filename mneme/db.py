"""Database connection helper for Mneme."""

import os
from pathlib import Path

import kuzu
from platformdirs import user_data_dir


def _default_db_path() -> str:
    """Return the user-scoped default DB path.

    Uses platformdirs so pip-installed users land in a sensible location:
      - Linux:   ~/.local/share/mneme/db.kuzu
      - macOS:   ~/Library/Application Support/mneme/db.kuzu
      - Windows: %LOCALAPPDATA%\\mneme\\db.kuzu

    Override with the ``MNEME_DB`` environment variable.
    """
    return os.path.join(user_data_dir("mneme"), "db.kuzu")


DEFAULT_DB_PATH = _default_db_path()


def get_connection(db_path: str = None) -> kuzu.Connection:
    """Get a Kuzu database connection.

    Args:
        db_path: Path to the Kuzu database file. Defaults to a user-scoped
                 directory (see :func:`_default_db_path`); overridden by
                 the ``MNEME_DB`` environment variable.
    """
    if db_path is None:
        db_path = os.environ.get("MNEME_DB", DEFAULT_DB_PATH)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = kuzu.Database(db_path)
    return kuzu.Connection(db)
