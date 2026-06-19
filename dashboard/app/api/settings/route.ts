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

const TUNABLE_DEFAULTS: Record<string, number | boolean> = {
  LOOP_INTERVAL_SECONDS:       3600,
  SIGNAL_THRESHOLD:            0.25,
  DEBATE_DIVERGENCE_THRESHOLD: 0.10,
  SCREENER_ENABLED:            false,
  SCREENER_TOP_N:              3,
  SCREENER_MIN_VOLUME_USD:     1_000_000.0,
  MARKETCAP_TOP_N:             20,
  MARKETCAP_REFRESH_SECS:      3600,
  DEX_BOOST_MULTIPLIER:        1.5,
  DEX_SCAN_CACHE_SECS:         300,
  GEM_VOLUME_SURGE_MULTIPLIER: 2.0,
  GEM_ROC_MIN_PCT:             3.0,
  GEM_TOP_N:                   5,
  GEM_MIN_VOLUME_USD:          500_000.0,
  IGNITION_WEIGHT:             0.15,
  GEM_POSITION_SIZE_PCT:       0.05,
  GEM_TRAILING_STOP_PCT:       0.05,
  POSITION_SIZE_PCT:           0.10,
  WATCHLIST_OHLCV_LIMIT:       50,
  SENTIMENT_MAX_AGE_SECONDS:   7200,
  INITIAL_CAPITAL:             10_000.0,
  FEE_PCT:                     0.001,
  SLIPPAGE_PCT:                0.0005,
};

type LockedEntry = {
  type: "bool" | "float" | "int";
  envDefault: string;
};

const SETTINGS_LOCKED: Record<string, LockedEntry> = {
  CRYPTO_LIVE:        { type: "bool",  envDefault: "false" },
  STOCKS_LIVE:        { type: "bool",  envDefault: "false" },
  SENTIMENT_ENABLED:  { type: "bool",  envDefault: "false" },
  MAX_DRAWDOWN_PCT:   { type: "float", envDefault: "0.15" },
  MAX_OPEN_POSITIONS: { type: "int",   envDefault: "3" },
};

function resolveEnv(key: string, spec: AllowlistEntry): number | boolean | null {
  const raw = process.env[key];
  if (raw == null) return null;
  if (spec.type === "bool") return raw.toLowerCase() === "true";
  if (spec.type === "int") return parseInt(raw, 10);
  return parseFloat(raw);
}

function resolveLockedValue(key: string, entry: LockedEntry): number | boolean {
  const raw = process.env[key] ?? entry.envDefault;
  if (entry.type === "bool") return raw.toLowerCase() === "true";
  if (entry.type === "int") return parseInt(raw, 10);
  return parseFloat(raw);
}

export async function GET() {
  // Gather DB overrides when available
  const overrides: Record<string, string> = {};

  if (existsSync(DB_PATH)) {
    let db: BunDatabase | null = null;
    try {
      const { Database } = (await import("bun:sqlite")) as typeof import("bun:sqlite");
      db = new Database(DB_PATH, { readwrite: true, create: false });
      const rows = db.prepare("SELECT key, value FROM settings").all() as {
        key: string;
        value: string;
      }[];
      for (const { key, value } of rows) overrides[key] = value;
    } catch {
      // settings table absent or DB unavailable — continue with env/defaults only
    } finally {
      db?.close();
    }
  }

  const editable = Object.entries(SETTINGS_ALLOWLIST).map(([key, spec]) => {
    const override = overrides[key] ?? null;
    const envVal = resolveEnv(key, spec);

    let effectiveValue: number | boolean;
    if (envVal !== null) {
      effectiveValue = envVal;
    } else if (override !== null) {
      if (spec.type === "bool") effectiveValue = override.toLowerCase() === "true";
      else if (spec.type === "int") effectiveValue = parseInt(override, 10);
      else effectiveValue = parseFloat(override);
    } else {
      effectiveValue = TUNABLE_DEFAULTS[key] as number | boolean;
    }

    return {
      key,
      editable: true,
      locked: false,
      type: spec.type,
      min: spec.min,
      max: spec.max,
      effectiveValue,
      override,
    };
  });

  const locked = Object.entries(SETTINGS_LOCKED).map(([key, entry]) => ({
    key,
    editable: false,
    locked: true,
    type: entry.type,
    min: null,
    max: null,
    effectiveValue: resolveLockedValue(key, entry),
    override: null,
  }));

  return NextResponse.json({ settings: [...editable, ...locked] });
}
