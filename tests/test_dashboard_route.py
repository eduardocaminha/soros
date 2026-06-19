"""Tests for dashboard/app/api/data/route.ts WAL-mode SQLite behaviour."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest


def _create_wal_db(db_path: Path) -> None:
    """Create a minimal WAL-mode DB that mirrors the schema queried by the route."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE equity_curve (
            ts TEXT,
            equity REAL,
            peak_equity REAL,
            drawdown_pct REAL,
            is_paper INTEGER
        )"""
    )
    conn.execute(
        "INSERT INTO equity_curve VALUES ('2024-01-01T00:00:00', 10000.0, 10000.0, 0.0, 1)"
    )
    conn.execute(
        """CREATE TABLE positions (
            symbol TEXT, asset_class TEXT, side TEXT, quantity REAL,
            entry_price REAL, current_price REAL, unrealized_pnl REAL,
            realized_pnl REAL, is_paper INTEGER, status TEXT, opened_at TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE orders (
            symbol TEXT, asset_class TEXT, side TEXT, quantity REAL,
            price REAL, order_type TEXT, status TEXT, is_paper INTEGER,
            created_at TEXT, filled_at TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE signals (
            symbol TEXT, asset_class TEXT, momentum_score REAL,
            volatility_score REAL, funding_score REAL, composite_score REAL,
            action TEXT, ts TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE sentiment_signals (
            symbol TEXT, asset_class TEXT, score REAL, label TEXT,
            confidence REAL, debate_used INTEGER, ts TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE event_log (
            ts TEXT, level TEXT, component TEXT, message TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE screener_runs (
            run_ts INTEGER, symbol TEXT, asset_class TEXT, is_pinned INTEGER,
            volume_usd_24h REAL, composite_score REAL, sentiment_score REAL,
            conviction REAL, selected INTEGER, reason TEXT
        )"""
    )
    conn.execute(
        """INSERT INTO screener_runs VALUES
            (1700000000, 'BTC/USDT', 'crypto', 1, 2000000.0, 0.7, 0.5, 0.7, 1, 'pinned')"""
    )
    conn.commit()
    conn.close()


@pytest.mark.skipif(
    shutil.which("bun") is None,
    reason="bun runtime not available",
)
def test_wal_db_readable_without_writer(tmp_path: Path) -> None:
    """bun:sqlite readwrite open must succeed on a checkpointed WAL DB (no -wal/-shm).

    Reproduces the bot-stopped scenario: when the bot (writer) exits cleanly,
    SQLite checkpoints the WAL and removes -wal/-shm, leaving only the main .db
    file.  The dashboard route previously opened with ``readonly:true``, which in
    some bun versions throws "unable to open database file" in this state, causing
    the UI to show empty data.  The fix switches to ``readwrite:true`` so bun can
    create a fresh -shm mapping without needing a pre-existing one.
    """
    db_path = tmp_path / "soros.db"
    _create_wal_db(db_path)

    # After connection close Python's sqlite3 checkpoints the WAL — these files
    # must be absent, matching the state that triggered the original bug.
    assert not (tmp_path / "soros.db-wal").exists(), "-wal must be absent after checkpoint"
    assert not (tmp_path / "soros.db-shm").exists(), "-shm must be absent after checkpoint"

    bun_script = f"""
import {{ Database }} from "bun:sqlite";
const db = new Database({json.dumps(str(db_path))}, {{ readwrite: true, create: false }});
const rows = db.prepare(
  "SELECT ts, equity, drawdown_pct FROM equity_curve ORDER BY ts DESC LIMIT 1"
).all();
db.close();
process.stdout.write(JSON.stringify(rows));
"""
    result = subprocess.run(
        ["bun", "-e", bun_script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"bun:sqlite readwrite open failed on checkpointed WAL DB.\n"
        f"stderr: {result.stderr}"
    )
    data = json.loads(result.stdout)
    assert len(data) == 1, "Expected one equity_curve row"
    assert data[0]["equity"] == pytest.approx(10000.0)


@pytest.mark.skipif(
    shutil.which("bun") is None,
    reason="bun runtime not available",
)
def test_wal_db_readable_with_active_writer(tmp_path: Path) -> None:
    """bun:sqlite readwrite open also works when -wal/-shm are present (bot running)."""
    db_path = tmp_path / "soros.db"
    _create_wal_db(db_path)

    # Simulate active-writer state: leave -wal and -shm files present.
    # Python's WAL checkpoint removes them on close, so we create them manually.
    (tmp_path / "soros.db-wal").write_bytes(b"")
    (tmp_path / "soros.db-shm").write_bytes(b"\x00" * 32768)

    bun_script = f"""
import {{ Database }} from "bun:sqlite";
try {{
  const db = new Database({json.dumps(str(db_path))}, {{ readwrite: true, create: false }});
  const rows = db.prepare(
    "SELECT ts, equity FROM equity_curve ORDER BY ts DESC LIMIT 1"
  ).all();
  db.close();
  process.stdout.write(JSON.stringify({{ ok: true, rows }}));
}} catch (e) {{
  process.stdout.write(JSON.stringify({{ ok: false, error: e.message }}));
}}
"""
    result = subprocess.run(
        ["bun", "-e", bun_script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"bun process crashed: {result.stderr}"
    out = json.loads(result.stdout)
    assert out["ok"], f"unexpected error with -wal/-shm present: {out.get('error')}"
    assert len(out["rows"]) == 1


@pytest.mark.skipif(
    shutil.which("bun") is None,
    reason="bun runtime not available",
)
def test_screener_rows_returned(tmp_path: Path) -> None:
    """Route returns screener_runs entries for the latest run_ts."""
    db_path = tmp_path / "soros.db"
    _create_wal_db(db_path)

    bun_script = f"""
import {{ Database }} from "bun:sqlite";
const db = new Database({json.dumps(str(db_path))}, {{ readwrite: true, create: false }});
const rows = db.prepare(
  `SELECT symbol, asset_class, is_pinned, selected, reason
   FROM screener_runs
   WHERE run_ts = (SELECT MAX(run_ts) FROM screener_runs)`
).all();
db.close();
process.stdout.write(JSON.stringify(rows));
"""
    result = subprocess.run(
        ["bun", "-e", bun_script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"bun process crashed: {result.stderr}"
    rows = json.loads(result.stdout)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTC/USDT"
    assert rows[0]["is_pinned"] == 1
    assert rows[0]["selected"] == 1
    assert rows[0]["reason"] == "pinned"
