"""SQLite client (WAL mode) + logging handler for the soros trading bot.

Provides:
  - Database: thin connection wrapper that applies schema and WAL pragma
  - SQLiteHandler: logging.Handler that persists records to event_log
  - get_connection(): convenience factory used by all components
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import config

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """Manages a single SQLite connection with WAL mode and schema bootstrap."""

    def __init__(self, db_path: str = config.DB_PATH) -> None:
        self._path = db_path
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        self._apply_schema(conn)
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> sqlite3.Connection:
        return self.connect()

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_schema(self, conn: sqlite3.Connection) -> None:
        sql = _SCHEMA_PATH.read_text()
        conn.executescript(sql)
        self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns introduced after the initial schema creation."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(signals)")}
        if "ignition_score" not in existing:
            conn.execute("ALTER TABLE signals ADD COLUMN ignition_score REAL")
            conn.commit()


# Module-level singleton; components call get_connection() instead of
# instantiating Database directly.
_db = Database()


def get_connection() -> sqlite3.Connection:
    """Return the shared WAL connection, opening it on first call."""
    return _db.connect()


# ---------------------------------------------------------------------------
# Logging handler — persists to event_log table
# ---------------------------------------------------------------------------

class SQLiteHandler(logging.Handler):
    """Logging handler that writes records to the event_log SQLite table."""

    # Map Python log level names to the CHECK constraint in schema
    _LEVEL_MAP = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = self._LEVEL_MAP.get(record.levelname, "INFO")
            component = record.name
            message = self.format(record)

            # Collect extra fields added by the caller (exclude stdlib attrs)
            _stdlib = logging.LogRecord.__dict__.keys() | {
                "message", "asctime", "exc_text", "stack_info",
            }
            extra: dict[str, Any] = {
                k: v for k, v in record.__dict__.items()
                if k not in _stdlib and not k.startswith("_")
            }
            extra_json = json.dumps(extra, default=str) if extra else None

            conn = get_connection()
            conn.execute(
                """
                INSERT INTO event_log (level, component, message, extra_json)
                VALUES (?, ?, ?, ?)
                """,
                (level, component, message, extra_json),
            )
            conn.commit()
        except Exception:  # noqa: BLE001
            self.handleError(record)


def get_logger(name: str, level: str = config.LOG_LEVEL) -> logging.Logger:
    """Return a logger wired to both stderr and the SQLite event_log table."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    db_handler = SQLiteHandler()
    db_handler.setFormatter(fmt)
    logger.addHandler(db_handler)

    return logger
