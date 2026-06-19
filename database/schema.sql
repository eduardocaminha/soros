-- Soros trading bot — SQLite schema
-- Enable WAL at connection time: PRAGMA journal_mode=WAL;

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ─────────────────────────────────────────────
-- OHLCV + funding rate snapshots
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prices (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    asset_class  TEXT    NOT NULL CHECK (asset_class IN ('crypto', 'stocks')),
    timeframe    TEXT    NOT NULL DEFAULT '1h',
    ts           INTEGER NOT NULL,  -- unix seconds UTC
    open         REAL    NOT NULL,
    high         REAL    NOT NULL,
    low          REAL    NOT NULL,
    close        REAL    NOT NULL,
    volume       REAL    NOT NULL,
    funding_rate REAL,              -- NULL for stocks
    inserted_at  INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_prices_symbol_tf_ts
    ON prices (symbol, timeframe, ts);
CREATE INDEX IF NOT EXISTS ix_prices_symbol_ts
    ON prices (symbol, ts DESC);

-- ─────────────────────────────────────────────
-- Deterministic signals per symbol per cycle
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT    NOT NULL,
    asset_class         TEXT    NOT NULL CHECK (asset_class IN ('crypto', 'stocks')),
    ts                  INTEGER NOT NULL,  -- cycle timestamp (unix seconds)
    momentum_score      REAL    NOT NULL,  -- normalised [-1, 1]
    volatility_score    REAL    NOT NULL,  -- normalised [-1, 1]
    funding_score       REAL,              -- NULL for stocks
    ignition_score      REAL,              -- NULL for stocks; volume z-score + ROC
    composite_score     REAL    NOT NULL,  -- weighted aggregate
    action              TEXT    NOT NULL CHECK (action IN ('buy', 'sell', 'hold')),
    inserted_at         INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS ix_signals_symbol_ts
    ON signals (symbol, ts DESC);

-- ─────────────────────────────────────────────
-- Sentiment signals (written by sentiment runner,
-- read by bot as 4th signal)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sentiment_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    asset_class     TEXT    NOT NULL CHECK (asset_class IN ('crypto', 'stocks')),
    ts              INTEGER NOT NULL,
    score           REAL    NOT NULL CHECK (score BETWEEN -1.0 AND 1.0),
    label           TEXT    NOT NULL CHECK (label IN ('bullish', 'bearish', 'neutral')),
    confidence      REAL    NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    debate_used     INTEGER NOT NULL DEFAULT 0 CHECK (debate_used IN (0, 1)),
    raw_json        TEXT,   -- full LLM response for audit
    inserted_at     INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS ix_sentiment_symbol_ts
    ON sentiment_signals (symbol, ts DESC);

-- ─────────────────────────────────────────────
-- Open and closed positions
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    asset_class     TEXT    NOT NULL CHECK (asset_class IN ('crypto', 'stocks')),
    side            TEXT    NOT NULL CHECK (side IN ('long', 'short')),
    quantity        REAL    NOT NULL,
    entry_price     REAL    NOT NULL,
    current_price   REAL    NOT NULL,
    unrealized_pnl  REAL    NOT NULL DEFAULT 0.0,
    realized_pnl    REAL    NOT NULL DEFAULT 0.0,
    is_paper        INTEGER NOT NULL DEFAULT 1 CHECK (is_paper IN (0, 1)),
    status          TEXT    NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    opened_at       INTEGER NOT NULL DEFAULT (unixepoch()),
    closed_at       INTEGER
);

CREATE INDEX IF NOT EXISTS ix_positions_symbol_status
    ON positions (symbol, status);
CREATE INDEX IF NOT EXISTS ix_positions_status
    ON positions (status);

-- ─────────────────────────────────────────────
-- Order log (paper + live)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    asset_class     TEXT    NOT NULL CHECK (asset_class IN ('crypto', 'stocks')),
    side            TEXT    NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity        REAL    NOT NULL,
    price           REAL    NOT NULL,
    order_type      TEXT    NOT NULL DEFAULT 'market' CHECK (order_type IN ('market', 'limit')),
    status          TEXT    NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'filled', 'cancelled', 'rejected')),
    exchange_id     TEXT,   -- broker/exchange order ID; NULL for paper
    is_paper        INTEGER NOT NULL DEFAULT 1 CHECK (is_paper IN (0, 1)),
    position_id     INTEGER REFERENCES positions (id),
    signal_ts       INTEGER,  -- which cycle triggered this order
    created_at      INTEGER NOT NULL DEFAULT (unixepoch()),
    filled_at       INTEGER
);

CREATE INDEX IF NOT EXISTS ix_orders_symbol_created
    ON orders (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_orders_status
    ON orders (status);

-- ─────────────────────────────────────────────
-- Equity curve — for drawdown calculation
-- Risk manager enforces 15 % drawdown stop
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS equity_curve (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           INTEGER NOT NULL DEFAULT (unixepoch()),
    equity       REAL    NOT NULL,
    peak_equity  REAL    NOT NULL,
    drawdown_pct REAL    NOT NULL DEFAULT 0.0,
    is_paper     INTEGER NOT NULL DEFAULT 1 CHECK (is_paper IN (0, 1))
);

CREATE INDEX IF NOT EXISTS ix_equity_ts
    ON equity_curve (ts DESC);

-- ─────────────────────────────────────────────
-- Event / audit log (errors, lifecycle events)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS event_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          INTEGER NOT NULL DEFAULT (unixepoch()),
    level       TEXT    NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    component   TEXT    NOT NULL,
    message     TEXT    NOT NULL,
    extra_json  TEXT
);

CREATE INDEX IF NOT EXISTS ix_event_log_ts
    ON event_log (ts DESC);
CREATE INDEX IF NOT EXISTS ix_event_log_level
    ON event_log (level);

-- ─────────────────────────────────────────────
-- Screener snapshots — one row per symbol per run
-- run_ts groups all entries from a single screen() call
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS screener_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts          INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    asset_class     TEXT    NOT NULL CHECK (asset_class IN ('crypto', 'stocks')),
    is_pinned       INTEGER NOT NULL CHECK (is_pinned IN (0, 1)),
    volume_usd_24h  REAL    NOT NULL DEFAULT 0.0,
    composite_score REAL    NOT NULL DEFAULT 0.0,
    sentiment_score REAL    NOT NULL DEFAULT 0.0,
    conviction      REAL    NOT NULL DEFAULT 0.0,
    selected        INTEGER NOT NULL CHECK (selected IN (0, 1)),
    reason          TEXT    NOT NULL,
    inserted_at     INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS ix_screener_run_ts
    ON screener_runs (run_ts DESC);
