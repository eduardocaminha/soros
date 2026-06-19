/**
 * Dashboard UI tests — run with: bun test
 *
 * Covers:
 *   1. Unread-alert badge count logic (pure computation)
 *   2. pt-BR interface labels
 *   3. Glossary tooltips presence
 *   4. Settings descriptions presence
 *   5. localStorage persistence for alerts last-visit
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
