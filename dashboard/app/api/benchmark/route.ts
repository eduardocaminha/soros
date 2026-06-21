// Force dynamic so Next.js never pre-renders this route at build time.
export const dynamic = "force-dynamic";

import path from "path";
import { existsSync } from "fs";
import { NextResponse } from "next/server";
import type { Database as BunDatabase } from "bun:sqlite";

import { buildBtcBenchmark, computeMetrics } from "../../../lib/benchmark";

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

export async function GET() {
  if (!existsSync(DB_PATH)) {
    return NextResponse.json({ empty: true, ts: Math.floor(Date.now() / 1000) });
  }

  let db: BunDatabase | null = null;
  try {
    const { Database } = (await import("bun:sqlite")) as typeof import("bun:sqlite");
    db = new Database(DB_PATH, { readwrite: true, create: false });

    const equityRows = rows<{ ts: number; equity: number }>(
      db,
      "SELECT ts, equity FROM equity_curve ORDER BY ts ASC, id ASC"
    );
    const btcRows = rows<{ ts: number; close: number }>(
      db,
      "SELECT ts, close FROM prices WHERE symbol = 'BTC/USDT' AND timeframe = '1h' ORDER BY ts ASC"
    );

    if (equityRows.length === 0) {
      return NextResponse.json({
        empty: true,
        ts: Math.floor(Date.now() / 1000),
        reason: "no equity snapshots",
      });
    }

    const snapshots: [number, number][] = equityRows.map((r) => [
      Number(r.ts),
      Number(r.equity),
    ]);
    const btcCloses: [number, number][] = btcRows.map((r) => [
      Number(r.ts),
      Number(r.close),
    ]);

    let series;
    try {
      series = buildBtcBenchmark(snapshots, btcCloses);
    } catch (e) {
      return NextResponse.json({
        empty: true,
        ts: Math.floor(Date.now() / 1000),
        reason: e instanceof Error ? e.message : String(e),
      });
    }

    const metrics = computeMetrics(series);

    return NextResponse.json({
      ts: Math.floor(Date.now() / 1000),
      series: {
        timestamps: series.timestamps,
        sorosEquity: series.sorosEquity,
        btcEquity: series.btcEquity,
        initialCapital: series.initialCapital,
        btcStartPrice: series.btcStartPrice,
        windowStart: series.windowStart,
        windowEnd: series.windowEnd,
        nPoints: series.nPoints,
        nBtcGaps: series.nBtcGaps,
      },
      metrics,
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
