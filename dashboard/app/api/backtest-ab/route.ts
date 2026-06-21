// Force dynamic so Next.js never pre-renders this route at build time.
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
  return (db.prepare(sql).all as (...args: Param[]) => T[])(...params);
}

type ABRow = {
  run_id: string;
  run_ts: number;
  start_ts: number;
  end_ts: number;
  symbols_json: string;
  fng_coverage_pct: number;
  off_total_return: number;
  off_cagr: number;
  off_sharpe: number;
  off_max_dd: number;
  off_win_rate: number;
  off_n_trades: number;
  on_total_return: number;
  on_cagr: number;
  on_sharpe: number;
  on_max_dd: number;
  on_win_rate: number;
  on_n_trades: number;
  off_equity_json: string;
  on_equity_json: string;
};

export async function GET() {
  if (!existsSync(DB_PATH)) {
    return NextResponse.json({ empty: true, ts: Math.floor(Date.now() / 1000) });
  }

  let db: BunDatabase | null = null;
  try {
    const { Database } = (await import("bun:sqlite")) as typeof import("bun:sqlite");
    db = new Database(DB_PATH, { readwrite: true, create: false });

    const tableExists = db
      .prepare(
        `SELECT 1 FROM sqlite_master WHERE type='table' AND name='backtest_ab_results'`
      )
      .get();

    if (!tableExists) {
      return NextResponse.json({ empty: true, ts: Math.floor(Date.now() / 1000) });
    }

    const result = rows<ABRow>(
      db,
      `SELECT run_id, run_ts, start_ts, end_ts, symbols_json, fng_coverage_pct,
              off_total_return, off_cagr, off_sharpe, off_max_dd, off_win_rate, off_n_trades,
              on_total_return,  on_cagr,  on_sharpe,  on_max_dd,  on_win_rate,  on_n_trades,
              off_equity_json, on_equity_json
       FROM backtest_ab_results
       ORDER BY run_ts DESC
       LIMIT 1`
    );

    if (result.length === 0) {
      return NextResponse.json({ empty: true, ts: Math.floor(Date.now() / 1000) });
    }

    const r = result[0];

    // Parse JSON blobs — stored as [[ts, equity], ...]; transform to {ts, equity} arrays.
    const offCurveRaw: [number, number][] = JSON.parse(r.off_equity_json);
    const onCurveRaw: [number, number][] = JSON.parse(r.on_equity_json);

    // Align timestamps: take the union intersection (they share the same timestamps).
    // Use the OFF curve timestamps as the reference since both runs share the same price data.
    const onMap = new Map<number, number>(onCurveRaw);
    const aligned = offCurveRaw
      .filter(([ts]) => onMap.has(ts))
      .map(([ts, offEq]) => ({ ts, offEq, onEq: onMap.get(ts)! }));

    const timestamps = aligned.map((p) => p.ts);
    const offEquity = aligned.map((p) => p.offEq);
    const onEquity = aligned.map((p) => p.onEq);

    return NextResponse.json({
      ts: Math.floor(Date.now() / 1000),
      run_id: r.run_id,
      run_ts: r.run_ts,
      start_ts: r.start_ts,
      end_ts: r.end_ts,
      symbols: (JSON.parse(r.symbols_json) as [string, string][]).map(
        ([sym, cls]) => `${sym}:${cls}`
      ),
      fng_coverage_pct: r.fng_coverage_pct,
      off: {
        total_return: r.off_total_return,
        cagr: r.off_cagr,
        sharpe: r.off_sharpe,
        max_dd: r.off_max_dd,
        win_rate: r.off_win_rate,
        n_trades: r.off_n_trades,
      },
      on: {
        total_return: r.on_total_return,
        cagr: r.on_cagr,
        sharpe: r.on_sharpe,
        max_dd: r.on_max_dd,
        win_rate: r.on_win_rate,
        n_trades: r.on_n_trades,
      },
      series: {
        timestamps,
        offEquity,
        onEquity,
        initialCapital: offEquity[0] ?? 0,
        windowStart: timestamps[0] ?? r.start_ts,
        windowEnd: timestamps[timestamps.length - 1] ?? r.end_ts,
        nPoints: timestamps.length,
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (
      message.includes("no such table") ||
      message.includes("no such file") ||
      message.includes("ENOENT")
    ) {
      return NextResponse.json({ empty: true, ts: Math.floor(Date.now() / 1000) });
    }
    return NextResponse.json({ error: message }, { status: 500 });
  } finally {
    db?.close();
  }
}
