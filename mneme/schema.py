"""Schema initialization for Mneme.

Creates all node tables and relationship tables using Kuzu's DDL.
These are idempotent (IF NOT EXISTS).
"""

SCHEMA_QUERIES = [
    """
    CREATE NODE TABLE IF NOT EXISTS Memory(
        id STRING,
        title STRING,
        body STRING,
        summary STRING,
        kind STRING,
        created_at STRING,
        updated_at STRING,
        embedding FLOAT[384],
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Session(
        id STRING,
        user_query STRING,
        mode STRING,
        created_at STRING,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS RELATES_TO(
        FROM Memory TO Memory,
        kind STRING,
        accuracy_weight DOUBLE,
        creative_weight DOUBLE,
        novelty_weight DOUBLE,
        confidence DOUBLE,
        use_count INT64,
        useful_count INT64,
        failed_count INT64,
        last_used_at STRING
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS ANALOGOUS_TO(
        FROM Memory TO Memory,
        reason STRING,
        kind STRING,
        accuracy_weight DOUBLE,
        creative_weight DOUBLE,
        novelty_weight DOUBLE,
        confidence DOUBLE,
        use_count INT64,
        useful_count INT64,
        failed_count INT64,
        last_used_at STRING
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS USED_IN_SESSION(
        FROM Session TO Memory,
        role STRING,
        usefulness DOUBLE,
        step_order INT64
    )
    """,
]


def init_schema(conn) -> None:
    """Execute all schema initialization queries on the connection.

    This is safe to call multiple times — all queries use IF NOT EXISTS.
    """
    for query in SCHEMA_QUERIES:
        conn.execute(query)
