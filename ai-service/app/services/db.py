"""PostgreSQL connection pool (psycopg 3) with pgvector type support.

All queries in this project use parameters (%s / %(name)s) - values are never pasted into
SQL strings, which is what prevents SQL injection.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

_pool: ConnectionPool | None = None


def _configure(conn) -> None:
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(get_settings().database_url, min_size=1, max_size=8, configure=_configure,
                               kwargs={"row_factory": dict_row}, open=True, timeout=10)
        atexit.register(close_pool)  # scripts exit cleanly
    return _pool


@contextmanager
def connection():
    with get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
