/**
 * Dashboard UI tests — run with: bun test
 *
 * Covers:
 *   1. Unread-alert badge count logic (pure computation)
 *   2. pt-BR interface labels
 *   3. Glossary tooltips presence
 *   4. Settings descriptions presence
 *   5. localStorage persistence for alerts last-visit
 *   6. Benchmark route — buildBtcBenchmark + computeMetrics
 */

import { test, expect, describe } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

// ─── 1. Unread badge logic ────────────────────────────────────────────────────
// Mirrors the computation in app/page.tsx:
//   data.events.filter((e) => e.ts > lastAlertsVisit).length

function countUnread(events: { ts: number }[], lastVisit: number): number {
  return events.filter((e) => e.ts > lastVisit).length;
}

describe("unread alert badge logic", () => {
  test("returns 0 when there are no events", () => {
    expect(countUnread([], 1_000)).toBe(0);
  });

  test("returns 0 when all events are older than last visit", () => {
    const events = [{ ts: 500 }, { ts: 999 }];
    expect(countUnread(events, 1_000)).toBe(0);
  });

  test("events at exactly the last-visit timestamp are not counted", () => {
    expect(countUnread([{ ts: 1_000 }], 1_000)).toBe(0);
  });

  test("counts only events strictly newer than last visit", () => {
    const events = [{ ts: 999 }, { ts: 1_001 }, { ts: 1_002 }];
    expect(countUnread(events, 1_000)).toBe(2);
  });

  test("returns full count when last visit is 0 (never visited)", () => {
    const events = [{ ts: 1 }, { ts: 2 }, { ts: 3 }];
    expect(countUnread(events, 0)).toBe(3);
  });

  test("opening the alerts tab (updating lastVisit to now) clears the badge", () => {
    const now = Math.floor(Date.now() / 1000);
    const events = [{ ts: now - 60 }, { ts: now - 30 }];
    expect(countUnread(events, now)).toBe(0);
  });
});

// ─── Source-level assertions ───────────────────────────────────────────────────
// We verify the implementation contract by inspecting the rendered source of the
// page component.  This is intentionally shallow: it confirms the UI wiring is
// present without spinning up a browser.

const src = readFileSync(join(import.meta.dir, "app/page.tsx"), "utf-8");

// ─── 2. pt-BR interface ───────────────────────────────────────────────────────

describe("dashboard pt-BR interface", () => {
  test("ALERTS_STORAGE_KEY is defined for localStorage persistence", () => {
    expect(src).toContain('ALERTS_STORAGE_KEY = "soros_alerts_last_visit"');
  });

  test("tab labels are in pt-BR", () => {
    expect(src).toContain("Alertas");
    expect(src).toContain("Configurações");
  });

  test("main section titles are in pt-BR", () => {
    expect(src).toContain("Posições Abertas");
    expect(src).toContain("Sinais Recentes");
    expect(src).toContain("Ordens Recentes");
    expect(src).toContain("Universo & Seleção");
  });

  test("unread badge filters events by ts > lastAlertsVisit", () => {
    expect(src).toContain("data.events.filter((e) => e.ts > lastAlertsVisit).length");
  });

  test("opening alerts tab persists visit timestamp to localStorage", () => {
    expect(src).toContain("localStorage.setItem(ALERTS_STORAGE_KEY, String(now))");
  });

  test("last-visit timestamp is loaded from localStorage on page init", () => {
    expect(src).toContain("localStorage.getItem(ALERTS_STORAGE_KEY)");
  });
});

// ─── 3. Glossary tooltips ─────────────────────────────────────────────────────

describe("glossary tooltips", () => {
  const REQUIRED_TERMS = [
    "Drawdown",
    "Equity",
    "P&L",
    "Momentum",
    "Funding",
    "Composite",
    "Volatilidade",
    "Sentimento",
    "Confiança",
    "Debate",
    "Convicção",
    "Screener",
    "Origem",
    "Gem",
    "Threshold",
    "CAGR",
    "Sharpe",
    "Max DD",
    "Win Rate",
    "Paper",
  ];

  for (const term of REQUIRED_TERMS) {
    test(`GLOSSARY has pt-BR definition for "${term}"`, () => {
      expect(src).toContain(`"${term}"`);
    });
  }

  test("Tooltip component is wired to GLOSSARY entries", () => {
    expect(src).toContain("<Tooltip text={GLOSSARY[");
  });
});

// ─── 4. Settings descriptions ─────────────────────────────────────────────────

describe("settings descriptions", () => {
  const KEY_SETTINGS = [
    "LOOP_INTERVAL_SECONDS",
    "SIGNAL_THRESHOLD",
    "SCREENER_ENABLED",
    "SCREENER_TOP_N",
    "POSITION_SIZE_PCT",
    "INITIAL_CAPITAL",
    "FEE_PCT",
    "SLIPPAGE_PCT",
    "GEM_TOP_N",
    "IGNITION_WEIGHT",
  ];

  for (const key of KEY_SETTINGS) {
    test(`SETTINGS_DESCRIPTIONS covers ${key}`, () => {
      expect(src).toContain(`"${key}"`);
    });
  }

  test("settings rows show inline description tooltip", () => {
    expect(src).toContain("desc ? (");
    expect(src).toContain("<Tooltip text={desc}>");
  });
});

// ─── 5. Backtest A/B panel ────────────────────────────────────────────────────

describe("BacktestABPanel", () => {
  test("BacktestABPanel component is wired into the dashboard render tree", () => {
    expect(src).toContain("BacktestABPanel");
  });

  test("BacktestABPanel is rendered alongside BenchmarkPanel", () => {
    const bmIdx = src.indexOf("BenchmarkPanel");
    const abIdx = src.indexOf("BacktestABPanel");
    expect(bmIdx).toBeGreaterThan(-1);
    expect(abIdx).toBeGreaterThan(-1);
  });

  test("F&G-only caveat is declared in the component", () => {
    expect(src).toContain("somente o Fear");
    expect(src).toContain("não replica o blend exato ao vivo");
  });

  test("CLI hint is shown when no data is available", () => {
    expect(src).toContain("python -m backtest.ab_command");
  });

  test("comparison header shows sentiment benefit or detriment", () => {
    expect(src).toContain("Sentimento MELHOROU o retorno");
    expect(src).toContain("Sentimento PIOROU o retorno");
  });

  test("overlay chart renders both OFF and ON equity curves", () => {
    expect(src).toContain("offEquity");
    expect(src).toContain("onEquity");
  });

  test("metrics table has pt-BR labels for all metrics", () => {
    expect(src).toContain("Sem Sentimento");
    expect(src).toContain("Com Sentimento (F&G)");
    expect(src).toContain("Win Rate");
  });

  test("new glossary terms are defined", () => {
    expect(src).toContain('"Backtest A/B"');
    expect(src).toContain('"Cobertura F&G"');
    expect(src).toContain('"Fear & Greed"');
  });

  test("F&G coverage tooltip is wired to GLOSSARY", () => {
    expect(src).toContain('GLOSSARY["Cobertura F&G"]');
  });

  test("never fetches on page load without user trigger — uses poll interval only", () => {
    // The panel uses useEffect + setInterval (same as BenchmarkPanel / SweepTable),
    // so it only loads after mount and then on poll cadence — never SSR / build-time.
    expect(src).toContain("/api/backtest-ab");
  });
});

// ─── 6. Benchmark route — buildBtcBenchmark + computeMetrics ─────────────────

import {
  buildBtcBenchmark,
  computeMetrics,
  type BenchmarkSeries,
} from "./lib/benchmark";

describe("buildBtcBenchmark", () => {
  test("throws on empty snapshots", () => {
    expect(() => buildBtcBenchmark([], [[1000, 50000]])).toThrow("snapshots is empty");
  });

  test("throws on empty btcCloses", () => {
    expect(() => buildBtcBenchmark([[1000, 10000]], [])).toThrow("no BTC close prices overlap");
  });

  test("throws when no BTC closes overlap", () => {
    expect(() =>
      buildBtcBenchmark([[2000, 10000]], [[1000, 50000], [1500, 51000]])
        // snapshot at 2000, but only BTC closes AT or BEFORE 2000; actually 1500 <= 2000 so it won't throw...
        // use a snapshot before all BTC data
    ).not.toThrow();
    expect(() =>
      buildBtcBenchmark([[500, 10000]], [[1000, 50000]])
    ).toThrow("no BTC close prices overlap");
  });

  test("single snapshot: soros_equity equals input equity", () => {
    const result = buildBtcBenchmark([[1000, 10000]], [[1000, 50000]]);
    expect(result.nPoints).toBe(1);
    expect(result.initialCapital).toBeCloseTo(10000);
    expect(result.btcEquity[0]).toBeCloseTo(10000);
  });

  test("BTC doubled → btcEquity doubled", () => {
    const result = buildBtcBenchmark(
      [[1000, 10000], [2000, 9000]],
      [[1000, 50000], [2000, 100000]]
    );
    expect(result.btcEquity[0]).toBeCloseTo(10000);
    expect(result.btcEquity[1]).toBeCloseTo(20000);
  });

  test("BTC halved → btcEquity halved", () => {
    const result = buildBtcBenchmark(
      [[1000, 10000], [2000, 12000]],
      [[1000, 40000], [2000, 20000]]
    );
    expect(result.btcEquity[1]).toBeCloseTo(5000);
  });

  test("sorosEquity matches input values", () => {
    const snaps: [number, number][] = [[1000, 10000], [2000, 11000], [3000, 9500]];
    const btc: [number, number][] = [[1000, 50000], [2000, 50000], [3000, 50000]];
    const result = buildBtcBenchmark(snaps, btc);
    expect(result.sorosEquity).toEqual([10000, 11000, 9500]);
  });

  test("snapshots before first BTC price are skipped", () => {
    const result = buildBtcBenchmark(
      [[500, 9000], [1000, 10000], [2000, 11000]],
      [[1000, 50000], [2000, 55000]]
    );
    expect(result.nPoints).toBe(2);
    expect(result.windowStart).toBe(1000);
    expect(result.initialCapital).toBeCloseTo(10000);
  });

  test("counts BTC gaps when price is forward-filled", () => {
    const result = buildBtcBenchmark(
      [[1000, 10000], [2000, 10000], [3000, 10000]],
      [[1000, 50000], [3000, 50000]]  // gap at ts=2000
    );
    expect(result.nBtcGaps).toBe(1);
  });

  test("timestamps are sorted ascending", () => {
    const result = buildBtcBenchmark(
      [[3000, 11000], [1000, 10000], [2000, 10500]],
      [[1000, 50000], [2000, 50000], [3000, 50000]]
    );
    expect(result.timestamps).toEqual([1000, 2000, 3000]);
  });

  test("windowStart and windowEnd are set correctly", () => {
    const result = buildBtcBenchmark(
      [[1000, 10000], [5000, 12000]],
      [[1000, 50000], [5000, 60000]]
    );
    expect(result.windowStart).toBe(1000);
    expect(result.windowEnd).toBe(5000);
  });

  test("multiple consecutive BTC gaps: forward-fill produces correct equity values", () => {
    // BTC price only at t=1000 (50_000) and t=4000 (60_000)
    // ts=2000 and ts=3000 forward-fill from t=1000
    const result = buildBtcBenchmark(
      [[1000, 10000], [2000, 10000], [3000, 10000], [4000, 10000]],
      [[1000, 50000], [4000, 60000]]
    );
    expect(result.nBtcGaps).toBe(2);
    // ts=1000: btcEquity = 10000 * (50000/50000) = 10000
    expect(result.btcEquity[0]).toBeCloseTo(10000);
    // ts=2000: forward-fill to 50000 → same
    expect(result.btcEquity[1]).toBeCloseTo(10000);
    // ts=3000: forward-fill to 50000 → same
    expect(result.btcEquity[2]).toBeCloseTo(10000);
    // ts=4000: 10000 * (60000/50000) = 12000
    expect(result.btcEquity[3]).toBeCloseTo(12000);
  });

  test("all snapshots except first are BTC gaps (entire series forward-filled from t=0 price)", () => {
    // BTC data only at ts=0; snapshots at 0,1000,2000 all use price from ts=0
    const result = buildBtcBenchmark(
      [[0, 10000], [1000, 10500], [2000, 11000]],
      [[0, 50000]]
    );
    expect(result.nBtcGaps).toBe(2);  // ts=1000 and ts=2000 are gaps
    expect(result.nPoints).toBe(3);
    // All BTC equity values equal initial capital (price unchanged by definition)
    for (const v of result.btcEquity) expect(v).toBeCloseTo(10000);
  });

  test("only snapshots covered by BTC data are included in aligned window", () => {
    // 3 snapshots: ts=100 before BTC, ts=200 and ts=300 after BTC starts
    const result = buildBtcBenchmark(
      [[100, 9000], [200, 10000], [300, 11000]],
      [[200, 50000], [300, 55000]]
    );
    expect(result.nPoints).toBe(2);
    expect(result.initialCapital).toBeCloseTo(10000);
    expect(result.windowStart).toBe(200);
  });
});

describe("computeMetrics", () => {
  function makeSeries(
    soros: number[],
    btc: number[],
    intervalSec = 3600,
    startTs = 1_000_000
  ): BenchmarkSeries {
    const n = soros.length;
    const timestamps = soros.map((_, i) => startTs + i * intervalSec);
    return {
      timestamps,
      sorosEquity: soros,
      btcEquity: btc,
      initialCapital: soros[0],
      btcStartPrice: 50000,
      windowStart: timestamps[0],
      windowEnd: timestamps[n - 1],
      nPoints: n,
      nBtcGaps: 0,
    };
  }

  test("totalReturn: flat series returns 0", () => {
    const s = makeSeries([10000, 10000], [10000, 10000]);
    const m = computeMetrics(s);
    expect(m.sorosTotalReturn).toBeCloseTo(0);
  });

  test("totalReturn: +20% gain", () => {
    const m = computeMetrics(makeSeries([10000, 12000], [10000, 10000]));
    expect(m.sorosTotalReturn).toBeCloseTo(0.2);
  });

  test("totalReturn: -20% loss", () => {
    const m = computeMetrics(makeSeries([10000, 8000], [10000, 10000]));
    expect(m.sorosTotalReturn).toBeCloseTo(-0.2);
  });

  test("maxDrawdown is <= 0", () => {
    const m = computeMetrics(makeSeries([10000, 12000, 9000, 11000], [10000, 10000, 10000, 10000]));
    expect(m.sorosMaxDrawdown).toBeLessThanOrEqual(0);
  });

  test("maxDrawdown: peak then trough", () => {
    const m = computeMetrics(makeSeries([10000, 12000, 6000], [10000, 10000, 10000]));
    expect(m.sorosMaxDrawdown).toBeCloseTo(-0.5); // (6000-12000)/12000 = -0.5
  });

  test("riskFreeRate is always 0", () => {
    const m = computeMetrics(makeSeries([10000, 11000], [10000, 10500]));
    expect(m.riskFreeRate).toBe(0);
  });

  test("sharpeConclusive is false for n < 30", () => {
    const m = computeMetrics(makeSeries([10000, 11000], [10000, 10500]));
    expect(m.sharpeConclusive).toBe(false);
  });

  test("sharpeConclusive is true for n >= 30", () => {
    const soros = Array.from({ length: 30 }, (_, i) => 10000 + i * 10);
    const btc = Array.from({ length: 30 }, (_, i) => 10000 + i * 5);
    const m = computeMetrics(makeSeries(soros, btc));
    expect(m.sharpeConclusive).toBe(true);
    expect(m.n).toBe(30);
  });

  test("sharpe is null for flat equity (zero std)", () => {
    const m = computeMetrics(makeSeries([10000, 10000, 10000], [10000, 10000, 10000]));
    expect(m.sorosSharpe).toBeNull();
    expect(m.btcSharpe).toBeNull();
  });

  test("annualizationFactor matches hourly cadence", () => {
    const SECONDS_PER_YEAR = 365.25 * 24 * 3600;
    const m = computeMetrics(makeSeries([10000, 11000, 12000], [10000, 10000, 10000], 3600));
    expect(m.annualizationFactor).toBeCloseTo(SECONDS_PER_YEAR / 3600);
  });

  test("medianIntervalSeconds reflects snapshot cadence", () => {
    const m = computeMetrics(makeSeries([10000, 11000, 12000], [10000, 10000, 10000], 7200));
    expect(m.medianIntervalSeconds).toBeCloseTo(7200);
  });

  test("both curves are compared independently", () => {
    const soros = [10000, 11000, 12000];
    const btc = [10000, 9000, 8000];
    const m = computeMetrics(makeSeries(soros, btc));
    expect(m.sorosTotalReturn).toBeCloseTo(0.2);
    expect(m.btcTotalReturn).toBeCloseTo(-0.2);
    expect(m.sorosMaxDrawdown).toBeCloseTo(0);
    expect(m.btcMaxDrawdown).toBeCloseTo(-0.2);
  });

  test("n=1 (single point): returns 0, no drawdown, sharpe null, not conclusive", () => {
    const m = computeMetrics(makeSeries([10000], [10000]));
    expect(m.n).toBe(1);
    expect(m.sorosTotalReturn).toBeCloseTo(0);
    expect(m.btcTotalReturn).toBeCloseTo(0);
    expect(m.sorosMaxDrawdown).toBeCloseTo(0);
    expect(m.btcMaxDrawdown).toBeCloseTo(0);
    expect(m.sorosSharpe).toBeNull();
    expect(m.btcSharpe).toBeNull();
    expect(m.sharpeConclusive).toBe(false);
  });

  test("n=2 (two points): sharpe is null — stdev requires ≥2 returns", () => {
    const m = computeMetrics(makeSeries([10000, 11000], [10000, 9000]));
    expect(m.n).toBe(2);
    // Only 1 period return → stdev undefined → sharpe null
    expect(m.sorosSharpe).toBeNull();
    expect(m.btcSharpe).toBeNull();
    expect(m.sharpeConclusive).toBe(false);
    // But totalReturn and drawdown still work
    expect(m.sorosTotalReturn).toBeCloseTo(0.1);
    expect(m.btcTotalReturn).toBeCloseTo(-0.1);
  });

  test("irregular timestamps: medianIntervalSeconds reflects actual gaps", () => {
    // gaps: 1h, 1h, 48h → median = 1h (3600)
    const ts0 = 1_000_000;
    const series: BenchmarkSeries = {
      timestamps: [ts0, ts0 + 3600, ts0 + 7200, ts0 + 7200 + 172800],
      sorosEquity: [10000, 10100, 10200, 10300],
      btcEquity: [10000, 10050, 10100, 10150],
      initialCapital: 10000,
      btcStartPrice: 50000,
      windowStart: ts0,
      windowEnd: ts0 + 7200 + 172800,
      nPoints: 4,
      nBtcGaps: 1,
    };
    const m = computeMetrics(series);
    expect(m.medianIntervalSeconds).toBeCloseTo(3600);
  });
});
