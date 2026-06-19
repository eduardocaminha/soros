export const dynamic = "force-dynamic";

import path from "path";
import { existsSync } from "fs";
import { NextResponse } from "next/server";
import type { Database as BunDatabase } from "bun:sqlite";

const DB_PATH =
  process.env.DB_PATH ??
  path.resolve(process.cwd(), "..", "data", "soros.db");

type AllowlistEntry = {
  type: "int" | "float" | "bool";
  min: number | null;
  max: number | null;
};

const SETTINGS_ALLOWLIST: Record<string, AllowlistEntry> = {
  LOOP_INTERVAL_SECONDS:       { type: "int",   min: 60,    max: 86400 },
  SIGNAL_THRESHOLD:            { type: "float", min: 0.0,   max: 1.0 },
  DEBATE_DIVERGENCE_THRESHOLD: { type: "float", min: 0.0,   max: 1.0 },
  SCREENER_ENABLED:            { type: "bool",  min: null,  max: null },
  SCREENER_TOP_N:              { type: "int",   min: 1,     max: 20 },
  SCREENER_MIN_VOLUME_USD:     { type: "float", min: 0.0,   max: null },
  MARKETCAP_TOP_N:             { type: "int",   min: 1,     max: 200 },
  MARKETCAP_REFRESH_SECS:      { type: "int",   min: 60,    max: 86400 },
  DEX_BOOST_MULTIPLIER:        { type: "float", min: 1.0,   max: 10.0 },
  DEX_SCAN_CACHE_SECS:         { type: "int",   min: 60,    max: 3600 },
  GEM_VOLUME_SURGE_MULTIPLIER: { type: "float", min: 1.0,   max: 20.0 },
  GEM_ROC_MIN_PCT:             { type: "float", min: 0.0,   max: 50.0 },
  GEM_TOP_N:                   { type: "int",   min: 1,     max: 50 },
  GEM_MIN_VOLUME_USD:          { type: "float", min: 0.0,   max: null },
  IGNITION_WEIGHT:             { type: "float", min: 0.0,   max: 1.0 },
  GEM_POSITION_SIZE_PCT:       { type: "float", min: 0.01,  max: 0.5 },
  GEM_TRAILING_STOP_PCT:       { type: "float", min: 0.0,   max: 0.5 },
  POSITION_SIZE_PCT:           { type: "float", min: 0.01,  max: 0.5 },
  WATCHLIST_OHLCV_LIMIT:       { type: "int",   min: 26,    max: 500 },
  SENTIMENT_MAX_AGE_SECONDS:   { type: "int",   min: 300,   max: 86400 },
  INITIAL_CAPITAL:             { type: "float", min: 100.0, max: null },
  FEE_PCT:                     { type: "float", min: 0.0,   max: 0.1 },
  SLIPPAGE_PCT:                { type: "float", min: 0.0,   max: 0.1 },
};

const SETTINGS_LOCKED = new Set([
  "CRYPTO_LIVE",
  "STOCKS_LIVE",
  "SENTIMENT_ENABLED",
  "MAX_DRAWDOWN_PCT",
  "MAX_OPEN_POSITIONS",
]);

type RouteContext = { params: Promise<{ key: string }> };

function parseAndValidate(
  key: string,
  spec: AllowlistEntry,
  raw: string
): { value: number | boolean } | { error: string } {
  let value: number | boolean;

  if (spec.type === "bool") {
    if (raw.toLowerCase() === "true" || raw === "1") value = true;
    else if (raw.toLowerCase() === "false" || raw === "0") value = false;
    else return { error: `${key}: expected bool ('true'/'false'), got ${JSON.stringify(raw)}` };
  } else if (spec.type === "int") {
    const n = parseInt(raw, 10);
    if (!Number.isInteger(n) || String(n) !== raw.trim())
      return { error: `${key}: cannot convert ${JSON.stringify(raw)} to int` };
    value = n;
  } else {
    const n = parseFloat(raw);
    if (Number.isNaN(n))
      return { error: `${key}: cannot convert ${JSON.stringify(raw)} to float` };
    value = n;
  }

  if (spec.min !== null && (value as number) < spec.min)
    return { error: `${key}: ${value} is below minimum ${spec.min}` };
  if (spec.max !== null && (value as number) > spec.max)
    return { error: `${key}: ${value} exceeds maximum ${spec.max}` };

  return { value };
}

export async function PUT(request: Request, context: RouteContext) {
  const { key } = await context.params;

  if (SETTINGS_LOCKED.has(key)) {
    return NextResponse.json(
      { error: `${key} is locked and cannot be overridden at runtime` },
      { status: 403 }
    );
  }

  const spec = SETTINGS_ALLOWLIST[key];
  if (!spec) {
    return NextResponse.json(
      { error: `${key} is not in the settings allowlist` },
      { status: 403 }
    );
  }

  let body: { value?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (body.value == null) {
    return NextResponse.json({ error: "Missing required field: value" }, { status: 400 });
  }

  const raw = String(body.value);
  const result = parseAndValidate(key, spec, raw);
  if ("error" in result) {
    return NextResponse.json({ error: result.error }, { status: 422 });
  }

  if (!existsSync(DB_PATH)) {
    return NextResponse.json({ error: "Database not found" }, { status: 503 });
  }

  let db: BunDatabase | null = null;
  try {
    const { Database } = (await import("bun:sqlite")) as typeof import("bun:sqlite");
    db = new Database(DB_PATH, { readwrite: true, create: false });
    db.prepare(
      `INSERT INTO settings (key, value, updated_at)
       VALUES (?, ?, ?)
       ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`
    ).run(key, raw, Math.floor(Date.now() / 1000));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  } finally {
    db?.close();
  }

  return NextResponse.json({ key, value: raw, effectiveValue: result.value });
}

export async function DELETE(_request: Request, context: RouteContext) {
  const { key } = await context.params;

  if (SETTINGS_LOCKED.has(key)) {
    return NextResponse.json(
      { error: `${key} is locked and cannot be modified` },
      { status: 403 }
    );
  }

  if (!SETTINGS_ALLOWLIST[key]) {
    return NextResponse.json(
      { error: `${key} is not in the settings allowlist` },
      { status: 403 }
    );
  }

  if (!existsSync(DB_PATH)) {
    return NextResponse.json({ error: "Database not found" }, { status: 503 });
  }

  let db: BunDatabase | null = null;
  try {
    const { Database } = (await import("bun:sqlite")) as typeof import("bun:sqlite");
    db = new Database(DB_PATH, { readwrite: true, create: false });
    db.prepare("DELETE FROM settings WHERE key = ?").run(key);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  } finally {
    db?.close();
  }

  return NextResponse.json({ key, deleted: true });
}
