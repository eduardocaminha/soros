// Force dynamic so Next.js never pre-renders this route at build time.
export const dynamic = "force-dynamic";

import path from "path";
import { existsSync } from "fs";
import { NextResponse } from "next/server";
import type { Database as BunDatabase } from "bun:sqlite";

const DB_PATH =
  process.env.DB_PATH ??
  path.resolve(process.cwd(), "..", "data", "soros.db");

const MIN_SHARPE_N = 30;
const SECONDS_PER_YEAR = 365.25 * 24 * 3600;

type Param = string | number | bigint | boolean | null | Uint8Array;

function rows<T = Record<string, unknown>>(
  db: BunDatabase,
  sql: string,
  params: Param[] = []
): T[] {
  return (db.prepare(sql).all as (...args: Param[]) => T[])(...params);
}

type SnapRow = { ts: number; equity: number; drawdown_pct: number };

function totalReturn(equity: number[]): number {
  if (equity.length < 1) return 0;
  return (equity[equity.length - 1] - equity[0]) / equity[0];
}

function periodReturns(equity: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < equity.length; i++)
    out.push((equity[i] - equity[i - 1]) / equity[i - 1]);
  return out;
}

function mean(arr: number[]): number {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function stdev(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
}

function medianInterval(timestamps: number[]): number {
  if (timestamps.length < 2) return 3600;
  const diffs: number[] = [];
  for (let i = 1; i < timestamps.length; i++) diffs.push(timestamps[i] - timestamps[i - 1]);
  diffs.sort((a, b) => a - b);
  const mid = Math.floor(diffs.length / 2);
  return diffs.length % 2 === 0 ? (diffs[mid - 1] + diffs[mid]) / 2 : diffs[mid];
}

function computeSharpe(equity: number[], timestamps: number[]): number | null {
  if (equity.length < 2) return null;
  const rets = periodReturns(equity);
  if (rets.length < 1) return null;
  const std = stdev(rets);
  if (std === 0) return null;
  const medInt = medianInterval(timestamps);
  const annFactor = medInt > 0 ? SECONDS_PER_YEAR / medInt : SECONDS_PER_YEAR / 3600;
  return (mean(rets) / std) * Math.sqrt(annFactor);
}

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
        `SELECT 1 FROM sqlite_master WHERE type='table' AND name='forward_shadow_snapshots'`
      )
      .get();

    if (!tableExists) {
      return NextResponse.json({ empty: true, ts: Math.floor(Date.now() / 1000) });
    }

    const realSnaps = rows<SnapRow>(
      db,
      `SELECT ts, equity, drawdown_pct FROM forward_shadow_snapshots WHERE variant='real' ORDER BY ts ASC, id ASC`
    );
    const shadowSnaps = rows<SnapRow>(
      db,
      `SELECT ts, equity, drawdown_pct FROM forward_shadow_snapshots WHERE variant='shadow' ORDER BY ts ASC, id ASC`
    );

    // Pair by index: real and shadow are inserted in the same cycle, seconds apart.
    const n = Math.min(realSnaps.length, shadowSnaps.length);
    if (n === 0) {
      return NextResponse.json({ empty: true, ts: Math.floor(Date.now() / 1000) });
    }

    const timestamps = realSnaps.slice(0, n).map((r) => r.ts);
    const realEquity = realSnaps.slice(0, n).map((r) => r.equity);
    const shadowEquity = shadowSnaps.slice(0, n).map((r) => r.equity);
    const realMaxDd = Math.max(...realSnaps.slice(0, n).map((r) => r.drawdown_pct));
    const shadowMaxDd = Math.max(...shadowSnaps.slice(0, n).map((r) => r.drawdown_pct));

    const sharpeConclusive = n >= MIN_SHARPE_N;

    return NextResponse.json({
      ts: Math.floor(Date.now() / 1000),
      real: {
        total_return: totalReturn(realEquity),
        max_dd: realMaxDd,
        sharpe: computeSharpe(realEquity, timestamps),
      },
      shadow: {
        total_return: totalReturn(shadowEquity),
        max_dd: shadowMaxDd,
        sharpe: computeSharpe(shadowEquity, timestamps),
      },
      series: {
        timestamps,
        realEquity,
        shadowEquity,
        initialCapital: realEquity[0] ?? 0,
        windowStart: timestamps[0] ?? 0,
        windowEnd: timestamps[timestamps.length - 1] ?? 0,
        nPoints: n,
      },
      sharpeConclusive,
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
