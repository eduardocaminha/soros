// Force dynamic so Next.js never pre-renders this route at build time.
// bun:sqlite is a Bun runtime built-in — lazy-imported at request time.
export const dynamic = "force-dynamic";

import path from "path";
import { existsSync } from "fs";
import { NextResponse } from "next/server";
import type { Database as BunDatabase } from "bun:sqlite";

const DB_PATH =
  process.env.DB_PATH ??
  path.resolve(process.cwd(), "..", "data", "soros.db");

type Param = string | number | bigint | boolean | null | Uint8Array;

function rows<T = Record<string, unknown>>(
  db: BunDatabase,
  sql: string,
  params: Param[] = []
): T[] {
  // bun:sqlite's Statement.all() accepts spread params
  return (db.prepare(sql).all as (...args: Param[]) => T[])(...params);
}

export async function GET() {
  if (!existsSync(DB_PATH)) {
    return NextResponse.json({ empty: true, ts: Math.floor(Date.now() / 1000) });
  }

  let db: BunDatabase | null = null;
  try {
    const { Database } = (await import("bun:sqlite")) as typeof import("bun:sqlite");
    // Open read-write (not readonly) so bun:sqlite handles WAL mode correctly when
    // -wal/-shm are absent (checkpointed state with bot stopped). We only issue
    // SELECT queries so nothing is ever written.
    db = new Database(DB_PATH, { readwrite: true, create: false });

    const equity = db
      .prepare(
        `SELECT equity, peak_equity, drawdown_pct, is_paper, ts
         FROM equity_curve ORDER BY ts DESC LIMIT 1`
      )
      .get() as Record<string, unknown> | null;

    const equityCurve = rows(
      db,
      `SELECT ts, equity, drawdown_pct FROM equity_curve ORDER BY ts DESC LIMIT 48`
    ).reverse();

    const positions = rows(
      db,
      `SELECT symbol, asset_class, side, quantity, entry_price,
              current_price, unrealized_pnl, realized_pnl, is_paper, status, opened_at
       FROM positions WHERE status = 'open' ORDER BY opened_at DESC`
    );

    const realizedRow = db
      .prepare(
        `SELECT COALESCE(SUM(realized_pnl), 0) AS total FROM positions WHERE status = 'closed'`
      )
      .get() as { total: number };

    const orders = rows(
      db,
      `SELECT symbol, asset_class, side, quantity, price, order_type,
              status, is_paper, created_at, filled_at
       FROM orders ORDER BY created_at DESC LIMIT 20`
    );

    const signals = rows(
      db,
      `SELECT s.symbol, s.asset_class, s.momentum_score, s.volatility_score,
              s.funding_score, s.composite_score, s.action, s.ts
       FROM signals s
       INNER JOIN (
         SELECT symbol, MAX(ts) AS max_ts FROM signals GROUP BY symbol
       ) latest ON s.symbol = latest.symbol AND s.ts = latest.max_ts
       ORDER BY s.asset_class, s.symbol`
    );

    const sentiment = rows(
      db,
      `SELECT ss.symbol, ss.asset_class, ss.score, ss.label,
              ss.confidence, ss.debate_used, ss.ts
       FROM sentiment_signals ss
       INNER JOIN (
         SELECT symbol, MAX(ts) AS max_ts FROM sentiment_signals GROUP BY symbol
       ) latest ON ss.symbol = latest.symbol AND ss.ts = latest.max_ts
       ORDER BY ss.asset_class, ss.symbol`
    );

    const events = rows(
      db,
      `SELECT ts, level, component, message
       FROM event_log
       WHERE level IN ('WARNING', 'ERROR', 'CRITICAL')
       ORDER BY ts DESC LIMIT 30`
    );

    let screener: Record<string, unknown>[] = [];
    try {
      screener = rows(
        db,
        `SELECT symbol, asset_class, is_pinned, volume_usd_24h, composite_score,
                sentiment_score, conviction, selected, reason, origin, run_ts
         FROM screener_runs
         WHERE run_ts = (SELECT MAX(run_ts) FROM screener_runs)
         ORDER BY asset_class, is_pinned DESC, selected DESC, conviction DESC`
      );
    } catch {
      // screener_runs absent in databases created before this feature
    }

    return NextResponse.json({
      ts: Math.floor(Date.now() / 1000),
      equity,
      equityCurve,
      positions,
      realizedPnl: realizedRow.total,
      orders,
      signals,
      sentiment,
      events,
      screener,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (message.includes("no such file") || message.includes("ENOENT")) {
      return NextResponse.json({ empty: true, ts: Math.floor(Date.now() / 1000) });
    }
    return NextResponse.json({ error: message }, { status: 500 });
  } finally {
    db?.close();
  }
}
