// Force dynamic so Next.js never pre-renders this route at build time.
export const dynamic = "force-dynamic";

import path from "path";
import { existsSync } from "fs";
import { NextRequest, NextResponse } from "next/server";
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

type SweepRow = {
  signal_threshold: number;
  total_return: number;
  cagr: number;
  sharpe: number;
  max_dd: number;
  win_rate: number;
  n_trades: number;
};

type SweepMeta = {
  sweep_id: string;
  run_ts: number;
};

function resolveCurrentThreshold(): number {
  const raw = process.env.SIGNAL_THRESHOLD;
  return raw != null ? parseFloat(raw) : 0.25;
}

export async function GET(req: NextRequest) {
  if (!existsSync(DB_PATH)) {
    return NextResponse.json({ empty: true, ts: Math.floor(Date.now() / 1000) });
  }

  const { searchParams } = req.nextUrl;
  const thresholdParam = searchParams.get("threshold");
  const filterThreshold = thresholdParam != null ? parseFloat(thresholdParam) : null;

  let db: BunDatabase | null = null;
  try {
    const { Database } = (await import("bun:sqlite")) as typeof import("bun:sqlite");
    db = new Database(DB_PATH, { readwrite: true, create: false });

    // Find the most recent sweep run.
    const meta = db
      .prepare(
        `SELECT sweep_id, run_ts FROM sweep_results ORDER BY run_ts DESC LIMIT 1`
      )
      .get() as SweepMeta | null;

    if (!meta) {
      return NextResponse.json({ empty: true, ts: Math.floor(Date.now() / 1000) });
    }

    // Fetch all rows for the latest sweep_id, optionally filtered by threshold.
    const sweepRows: SweepRow[] =
      filterThreshold != null
        ? rows<SweepRow>(
            db,
            `SELECT signal_threshold, total_return, cagr, sharpe, max_dd, win_rate, n_trades
             FROM sweep_results
             WHERE sweep_id = ? AND signal_threshold = ?
             ORDER BY signal_threshold`,
            [meta.sweep_id, filterThreshold]
          )
        : rows<SweepRow>(
            db,
            `SELECT signal_threshold, total_return, cagr, sharpe, max_dd, win_rate, n_trades
             FROM sweep_results
             WHERE sweep_id = ?
             ORDER BY signal_threshold`,
            [meta.sweep_id]
          );

    return NextResponse.json({
      ts: Math.floor(Date.now() / 1000),
      sweep_id: meta.sweep_id,
      run_ts: meta.run_ts,
      current_threshold: resolveCurrentThreshold(),
      rows: sweepRows,
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
