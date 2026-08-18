import os

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "newspaper_archive")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))

_CONNINFO = (
    f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
    f"user={DB_USER} password={DB_PASSWORD}"
)

_pool = ConnectionPool(
    conninfo=_CONNINFO,
    min_size=DB_POOL_MIN_SIZE,
    max_size=DB_POOL_MAX_SIZE,
    kwargs={"row_factory": dict_row},
    open=False,
)


def get_connection():
    """
    Borrow a PostgreSQL connection for the newspaper archive from the
    shared pool. Use as a context manager:

        with get_connection() as conn:
            ...

    On exit, the transaction is committed (or rolled back on
    exception) and the connection is returned to the pool rather
    than closed.
    """

    if _pool.closed:
        _pool.open()

    return _pool.connection()


def close_pool():
    """
    Close the pool's connections. Call on application shutdown.
    """

    if not _pool.closed:
        _pool.close()
