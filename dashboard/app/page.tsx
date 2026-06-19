"use client";

import { useEffect, useState, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface EquitySnap {
  equity: number;
  peak_equity: number;
  drawdown_pct: number;
  is_paper: number;
  ts: number;
}

interface EquityPoint {
  ts: number;
  equity: number;
  drawdown_pct: number;
}

interface Position {
  symbol: string;
  asset_class: string;
  side: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  is_paper: number;
  status: string;
  opened_at: number;
}

interface Order {
  symbol: string;
  asset_class: string;
  side: string;
  quantity: number;
  price: number;
  order_type: string;
  status: string;
  is_paper: number;
  created_at: number;
  filled_at: number | null;
}

interface Signal {
  symbol: string;
  asset_class: string;
  momentum_score: number;
  volatility_score: number;
  funding_score: number | null;
  composite_score: number;
  action: string;
  ts: number;
}

interface SentimentRow {
  symbol: string;
  asset_class: string;
  score: number;
  label: string;
  confidence: number;
  debate_used: number;
  ts: number;
}

interface Event {
  ts: number;
  level: string;
  component: string;
  message: string;
}

interface ScreenerEntry {
  symbol: string;
  asset_class: string;
  is_pinned: number;
  volume_usd_24h: number;
  composite_score: number;
  sentiment_score: number;
  conviction: number;
  selected: number;
  reason: string;
  origin: string;
  run_ts: number;
}

interface DashData {
  ts: number;
  equity: EquitySnap | null;
  equityCurve: EquityPoint[];
  positions: Position[];
  realizedPnl: number;
  orders: Order[];
  signals: Signal[];
  sentiment: SentimentRow[];
  events: Event[];
  screener: ScreenerEntry[];
  empty?: boolean;
  error?: string;
}

interface SettingRow {
  key: string;
  editable: boolean;
  locked: boolean;
  type: "int" | "float" | "bool";
  min: number | null;
  max: number | null;
  effectiveValue: number | boolean;
  override: string | null;
}

interface SweepApiRow {
  signal_threshold: number;
  total_return: number;
  cagr: number;
  sharpe: number;
  max_dd: number;
  win_rate: number;
  n_trades: number;
}

interface SweepData {
  ts: number;
  sweep_id: string;
  run_ts: number;
  current_threshold: number;
  rows: SweepApiRow[];
  empty?: boolean;
  error?: string;
}

type TabId = "dashboard" | "settings";

const SETTING_GROUPS: Array<{ label: string; keys: string[] }> = [
  { label: "Loop", keys: ["LOOP_INTERVAL_SECONDS"] },
  { label: "Signals", keys: ["SIGNAL_THRESHOLD", "DEBATE_DIVERGENCE_THRESHOLD"] },
  { label: "Screener", keys: ["SCREENER_ENABLED", "SCREENER_TOP_N", "SCREENER_MIN_VOLUME_USD"] },
  { label: "Market Cap / DEX", keys: ["MARKETCAP_TOP_N", "MARKETCAP_REFRESH_SECS", "DEX_BOOST_MULTIPLIER", "DEX_SCAN_CACHE_SECS"] },
  { label: "Gems", keys: ["GEM_VOLUME_SURGE_MULTIPLIER", "GEM_ROC_MIN_PCT", "GEM_TOP_N", "GEM_MIN_VOLUME_USD", "IGNITION_WEIGHT", "GEM_POSITION_SIZE_PCT", "GEM_TRAILING_STOP_PCT"] },
  { label: "Position & Fees", keys: ["POSITION_SIZE_PCT", "WATCHLIST_OHLCV_LIMIT", "SENTIMENT_MAX_AGE_SECONDS", "INITIAL_CAPITAL", "FEE_PCT", "SLIPPAGE_PCT"] },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

const POLL_MS = 15_000;

function fmt(n: number, decimals = 2) {
  return n.toFixed(decimals);
}

function fmtPnl(n: number) {
  const s = (n >= 0 ? "+" : "") + fmt(n);
  return <span className={n >= 0 ? "positive" : "negative"}>{s}</span>;
}

function fmtScore(n: number) {
  const s = (n >= 0 ? "+" : "") + fmt(n, 3);
  return <span className={n > 0.05 ? "positive" : n < -0.05 ? "negative" : "neutral"}>{s}</span>;
}

function ts2str(ts: number) {
  return new Date(ts * 1000).toLocaleString("pt-BR", { hour12: false });
}

function actionBadge(action: string) {
  return <span className={`badge badge-${action.toLowerCase()}`}>{action.toUpperCase()}</span>;
}

function sideBadge(side: string) {
  return <span className={`badge badge-${side.toLowerCase()}`}>{side.toUpperCase()}</span>;
}

// ─── Sparkline (simple SVG) ───────────────────────────────────────────────────

function Sparkline({ points }: { points: EquityPoint[] }) {
  if (points.length < 2) return <span className="neutral">—</span>;
  const vals = points.map((p) => p.equity);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const range = max - min || 1;
  const W = 160;
  const H = 36;
  const xs = points.map((_, i) => (i / (points.length - 1)) * W);
  const ys = vals.map((v) => H - ((v - min) / range) * H);
  const d = xs.map((x, i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(" ");
  const last = vals[vals.length - 1];
  const first = vals[0];
  const color = last >= first ? "var(--green)" : "var(--red)";
  return (
    <svg width={W} height={H} style={{ verticalAlign: "middle" }}>
      <path d={d} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

// ─── Sections ────────────────────────────────────────────────────────────────

function EquityCard({ data }: { data: DashData }) {
  const eq = data.equity;
  const totalUnrealized = data.positions.reduce((s, p) => s + p.unrealized_pnl, 0);

  return (
    <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginBottom: 24 }}>
      <Card label="Equity">
        {eq ? (
          <>
            <big>${fmt(eq.equity)}</big>
            {eq.is_paper ? <span className="badge badge-paper" style={{ marginLeft: 8 }}>PAPER</span> : <span className="badge badge-live" style={{ marginLeft: 8 }}>LIVE</span>}
          </>
        ) : <span className="neutral">—</span>}
      </Card>

      <Card label="Drawdown">
        {eq ? (
          <span className={eq.drawdown_pct > 10 ? "negative" : eq.drawdown_pct > 5 ? "" : "positive"}>
            {fmt(eq.drawdown_pct, 1)}%{" "}
            <span className="neutral" style={{ fontSize: 11 }}>/ 15% limit</span>
          </span>
        ) : <span className="neutral">—</span>}
      </Card>

      <Card label="Unrealized P&L">
        {data.positions.length ? fmtPnl(totalUnrealized) : <span className="neutral">—</span>}
      </Card>

      <Card label="Realized P&L">
        {fmtPnl(data.realizedPnl)}
      </Card>

      <Card label="Equity (48h)" wide>
        <Sparkline points={data.equityCurve} />
      </Card>
    </div>
  );
}

function Card({ label, children, wide }: { label: string; children: React.ReactNode; wide?: boolean }) {
  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: 8,
      padding: "12px 16px",
      minWidth: wide ? 200 : 140,
    }}>
      <div style={{ color: "var(--text-muted)", fontSize: 11, marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
      <div style={{ fontSize: wide ? 13 : 18, fontWeight: 600 }}>{children}</div>
    </div>
  );
}

function PositionsTable({ positions }: { positions: Position[] }) {
  return (
    <Section title="Open Positions" count={positions.length}>
      {positions.length === 0 ? (
        <p className="neutral" style={{ padding: "12px 10px" }}>No open positions.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Current</th>
              <th>Unrealized</th><th>Mode</th><th>Opened</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p, i) => (
              <tr key={i}>
                <td><b>{p.symbol}</b> <span className="neutral" style={{ fontSize: 11 }}>{p.asset_class}</span></td>
                <td>{sideBadge(p.side)}</td>
                <td>{fmt(p.quantity, 4)}</td>
                <td>${fmt(p.entry_price)}</td>
                <td>${fmt(p.current_price)}</td>
                <td>{fmtPnl(p.unrealized_pnl)}</td>
                <td><span className={`badge badge-${p.is_paper ? "paper" : "live"}`}>{p.is_paper ? "paper" : "live"}</span></td>
                <td className="neutral">{ts2str(p.opened_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Section>
  );
}

function SignalsTable({ signals }: { signals: Signal[] }) {
  return (
    <Section title="Latest Signals">
      {signals.length === 0 ? (
        <p className="neutral" style={{ padding: "12px 10px" }}>No signals yet.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Class</th><th>Momentum</th><th>Volatility</th>
              <th>Funding</th><th>Composite</th><th>Action</th><th>At</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((s, i) => (
              <tr key={i}>
                <td><b>{s.symbol}</b></td>
                <td className="neutral">{s.asset_class}</td>
                <td>{fmtScore(s.momentum_score)}</td>
                <td>{fmtScore(s.volatility_score)}</td>
                <td>{s.funding_score != null ? fmtScore(s.funding_score) : <span className="neutral">—</span>}</td>
                <td><b>{fmtScore(s.composite_score)}</b></td>
                <td>{actionBadge(s.action)}</td>
                <td className="neutral">{ts2str(s.ts)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Section>
  );
}

function SentimentTable({ sentiment }: { sentiment: SentimentRow[] }) {
  return (
    <Section title="Sentiment">
      {sentiment.length === 0 ? (
        <p className="neutral" style={{ padding: "12px 10px" }}>No sentiment data yet.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Class</th><th>Score</th><th>Label</th>
              <th>Confidence</th><th>Debate</th><th>At</th>
            </tr>
          </thead>
          <tbody>
            {sentiment.map((s, i) => (
              <tr key={i}>
                <td><b>{s.symbol}</b></td>
                <td className="neutral">{s.asset_class}</td>
                <td>{fmtScore(s.score)}</td>
                <td><span className={s.label}>{s.label}</span></td>
                <td>{fmt(s.confidence * 100, 0)}%</td>
                <td>{s.debate_used ? <span className="badge badge-warn">yes</span> : <span className="neutral">no</span>}</td>
                <td className="neutral">{ts2str(s.ts)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Section>
  );
}

const REASON_LABELS: Record<string, string> = {
  pinned: "pinned",
  screener: "selected",
  volume_floor: "low volume",
  sentiment_gate: "bearish sentiment",
  not_ranked: "not ranked",
};

const ORIGIN_LABELS: Record<string, string> = {
  base: "base",
  gem: "gem",
  dex_boosted: "dex",
};

function originBadge(origin: string) {
  if (!origin) return <span className="neutral">—</span>;
  if (origin === "base") return <span className="badge badge-open">{ORIGIN_LABELS[origin] ?? origin}</span>;
  if (origin === "dex_boosted") return <span className="badge badge-warn">{ORIGIN_LABELS[origin] ?? origin}</span>;
  return <span className="badge badge-filled">{ORIGIN_LABELS[origin] ?? origin}</span>;
}

function ScreenerTable({ screener }: { screener: ScreenerEntry[] }) {
  if (screener.length === 0) return null;
  const runTs = screener[0]?.run_ts;
  const selected = screener.filter((e) => e.selected);
  return (
    <Section title="Universe & Selection" count={selected.length}>
      <div style={{ padding: "8px 12px", color: "var(--text-muted)", fontSize: 11, borderBottom: "1px solid var(--border)" }}>
        Last screener run: {runTs ? ts2str(runTs) : "—"} · {selected.length} of {screener.length} selected
      </div>
      <table>
        <thead>
          <tr>
            <th>Symbol</th><th>Class</th><th>Origin</th><th>Type</th><th>Vol 24h (USD)</th>
            <th>Composite</th><th>Sentiment</th><th>Conviction</th><th>Status</th>
          </tr>
        </thead>
        <tbody>
          {screener.map((e, i) => (
            <tr key={i} style={{ opacity: e.selected ? 1 : 0.55 }}>
              <td><b>{e.symbol}</b></td>
              <td className="neutral">{e.asset_class}</td>
              <td>{originBadge(e.origin)}</td>
              <td>{e.is_pinned ? <span className="badge badge-open">pinned</span> : <span className="neutral">watch</span>}</td>
              <td className="neutral">{e.volume_usd_24h >= 1_000_000 ? `$${(e.volume_usd_24h / 1_000_000).toFixed(1)}M` : e.volume_usd_24h > 0 ? `$${(e.volume_usd_24h / 1_000).toFixed(0)}K` : "—"}</td>
              <td>{fmtScore(e.composite_score)}</td>
              <td>{fmtScore(e.sentiment_score)}</td>
              <td className="neutral">{e.conviction.toFixed(3)}</td>
              <td>
                {e.selected
                  ? <span className="badge badge-filled">✓ {REASON_LABELS[e.reason] ?? e.reason}</span>
                  : <span className="badge badge-closed">{REASON_LABELS[e.reason] ?? e.reason}</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}

function OrdersTable({ orders }: { orders: Order[] }) {
  return (
    <Section title="Recent Orders" count={orders.length}>
      {orders.length === 0 ? (
        <p className="neutral" style={{ padding: "12px 10px" }}>No orders yet.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th>
              <th>Type</th><th>Status</th><th>Mode</th><th>Created</th>
            </tr>
          </thead>
          <tbody>
            {orders.map((o, i) => (
              <tr key={i}>
                <td><b>{o.symbol}</b> <span className="neutral" style={{ fontSize: 11 }}>{o.asset_class}</span></td>
                <td>{sideBadge(o.side)}</td>
                <td>{fmt(o.quantity, 4)}</td>
                <td>${fmt(o.price)}</td>
                <td className="neutral">{o.order_type}</td>
                <td><span className={`badge badge-${o.status === "filled" ? "filled" : o.status === "pending" ? "open" : "closed"}`}>{o.status}</span></td>
                <td><span className={`badge badge-${o.is_paper ? "paper" : "live"}`}>{o.is_paper ? "paper" : "live"}</span></td>
                <td className="neutral">{ts2str(o.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Section>
  );
}

function EventsTable({ events }: { events: Event[] }) {
  if (events.length === 0) return null;
  return (
    <Section title="Alerts">
      <table>
        <thead>
          <tr><th>Time</th><th>Level</th><th>Component</th><th>Message</th></tr>
        </thead>
        <tbody>
          {events.map((e, i) => (
            <tr key={i}>
              <td className="neutral">{ts2str(e.ts)}</td>
              <td>
                <span className={`badge badge-${e.level === "ERROR" || e.level === "CRITICAL" ? "error" : "warn"}`}>
                  {e.level}
                </span>
              </td>
              <td className="neutral">{e.component}</td>
              <td style={{ maxWidth: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{e.message}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}

function SweepTable() {
  const [sweep, setSweep] = useState<SweepData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/sweep");
      const json = (await res.json()) as SweepData;
      if (json.error) setError(json.error);
      else { setSweep(json); setError(null); }
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  if (error) return null;
  if (!sweep || sweep.empty || !sweep.rows || sweep.rows.length === 0) return null;

  const cur = sweep.current_threshold;
  const bestSharpe = Math.max(...sweep.rows.map((r) => r.sharpe));

  return (
    <Section title="Threshold Robustness">
      <div style={{ padding: "8px 12px", color: "var(--text-muted)", fontSize: 11, borderBottom: "1px solid var(--border)" }}>
        Last sweep: {ts2str(sweep.run_ts)} · sweep id: <code style={{ fontSize: 10 }}>{sweep.sweep_id.slice(0, 8)}</code>
        {" · "}
        <span style={{ color: "var(--blue)" }}>highlighted row = current threshold ({cur})</span>
        {" · "}
        <span style={{ color: "var(--text-muted)" }}>goal: stable neighbourhood, not max return</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Threshold</th>
            <th>Return</th>
            <th>CAGR</th>
            <th>Sharpe</th>
            <th>Max DD</th>
            <th>Win Rate</th>
            <th>Trades</th>
          </tr>
        </thead>
        <tbody>
          {sweep.rows.map((r) => {
            const isCurrent = Math.abs(r.signal_threshold - cur) < 1e-9;
            const isBestSharpe = Math.abs(r.sharpe - bestSharpe) < 1e-9 && sweep.rows.length > 1;
            return (
              <tr
                key={r.signal_threshold}
                style={
                  isCurrent
                    ? { background: "rgba(88,166,255,0.08)", outline: "1px solid rgba(88,166,255,0.3)" }
                    : undefined
                }
              >
                <td>
                  <code style={{ fontSize: 12 }}>{r.signal_threshold.toFixed(2)}</code>
                  {isCurrent && (
                    <span className="badge badge-open" style={{ marginLeft: 6, fontSize: 10 }}>active</span>
                  )}
                </td>
                <td>
                  <span className={r.total_return >= 0 ? "positive" : "negative"}>
                    {r.total_return >= 0 ? "+" : ""}{(r.total_return * 100).toFixed(1)}%
                  </span>
                </td>
                <td>
                  <span className={r.cagr >= 0 ? "positive" : "negative"}>
                    {r.cagr >= 0 ? "+" : ""}{(r.cagr * 100).toFixed(1)}%
                  </span>
                </td>
                <td>
                  <span className={r.sharpe >= 1 ? "positive" : r.sharpe >= 0 ? "" : "negative"}>
                    {r.sharpe.toFixed(2)}
                    {isBestSharpe && <span style={{ marginLeft: 4, color: "var(--text-muted)", fontSize: 10 }}>▲</span>}
                  </span>
                </td>
                <td>
                  <span className={r.max_dd > 20 ? "negative" : r.max_dd > 10 ? "" : "positive"}>
                    {r.max_dd.toFixed(1)}%
                  </span>
                </td>
                <td>{(r.win_rate * 100).toFixed(1)}%</td>
                <td className="neutral">{r.n_trades}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Section>
  );
}

function Section({ title, count, children }: { title: string; count?: number; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <h2 style={{ marginBottom: 8 }}>
        {title}
        {count !== undefined && count > 0 && (
          <span style={{ marginLeft: 8, fontSize: 11, fontWeight: 400, color: "var(--blue)" }}>({count})</span>
        )}
      </h2>
      <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden" }}>
        {children}
      </div>
    </div>
  );
}

// ─── Settings ────────────────────────────────────────────────────────────────

function LockedRow({ setting }: { setting: SettingRow }) {
  return (
    <tr style={{ opacity: 0.65 }}>
      <td><code style={{ fontSize: 11 }}>{setting.key}</code></td>
      <td className="neutral" style={{ fontSize: 11 }}>{setting.type}</td>
      <td>
        {setting.type === "bool" ? (
          <input
            type="checkbox"
            checked={setting.effectiveValue as boolean}
            disabled
            style={{ cursor: "not-allowed" }}
          />
        ) : (
          <span style={{ fontSize: 12 }}>{String(setting.effectiveValue)}</span>
        )}
      </td>
      <td><span className="badge badge-error" style={{ fontSize: 10 }}>locked</span></td>
    </tr>
  );
}

function EditableRow({ setting, onRefresh }: { setting: SettingRow; onRefresh: () => void }) {
  const [draft, setDraft] = useState(String(setting.effectiveValue));
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [flash, setFlash] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    if (!dirty) setDraft(String(setting.effectiveValue));
  }, [setting.effectiveValue, dirty]);

  const showFlash = (ok: boolean, text: string) => {
    setFlash({ ok, text });
    setTimeout(() => setFlash(null), 2500);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const value =
        setting.type === "bool"
          ? draft === "true"
          : setting.type === "int"
          ? parseInt(draft, 10)
          : parseFloat(draft);
      const res = await fetch(`/api/settings/${setting.key}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value }),
      });
      if (res.ok) {
        setDirty(false);
        showFlash(true, "saved");
        onRefresh();
      } else {
        const json = (await res.json()) as { error?: string };
        showFlash(false, json.error ?? "error");
      }
    } catch (e) {
      showFlash(false, String(e));
    } finally {
      setSaving(false);
    }
  };

  const handleBoolToggle = (checked: boolean) => {
    const prev = draft;
    setDraft(checked ? "true" : "false");
    setSaving(true);
    (async () => {
      try {
        const res = await fetch(`/api/settings/${setting.key}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ value: checked }),
        });
        if (res.ok) {
          showFlash(true, "saved");
          onRefresh();
        } else {
          setDraft(prev);
          const json = (await res.json()) as { error?: string };
          showFlash(false, json.error ?? "error");
        }
      } catch (e) {
        setDraft(prev);
        showFlash(false, String(e));
      } finally {
        setSaving(false);
      }
    })();
  };

  const handleReset = async () => {
    setSaving(true);
    try {
      const res = await fetch(`/api/settings/${setting.key}`, { method: "DELETE" });
      if (res.ok) {
        setDirty(false);
        showFlash(true, "reset");
        onRefresh();
      } else {
        showFlash(false, "reset failed");
      }
    } catch (e) {
      showFlash(false, String(e));
    } finally {
      setSaving(false);
    }
  };

  const rangeStr =
    setting.min != null && setting.max != null
      ? `${setting.min}–${setting.max}`
      : setting.min != null
      ? `≥${setting.min}`
      : setting.max != null
      ? `≤${setting.max}`
      : "—";

  const btnBase: React.CSSProperties = {
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: 11,
    cursor: "pointer",
    fontFamily: "inherit",
    background: "transparent",
    marginLeft: 6,
  };

  return (
    <tr>
      <td>
        <code style={{ fontSize: 11 }}>{setting.key}</code>
        {setting.override !== null && (
          <span className="badge badge-warn" style={{ marginLeft: 6, fontSize: 10 }}>override</span>
        )}
      </td>
      <td className="neutral" style={{ fontSize: 11 }}>{setting.type}</td>
      <td className="neutral" style={{ fontSize: 11 }}>{rangeStr}</td>
      <td>
        {setting.type === "bool" ? (
          <input
            type="checkbox"
            checked={draft === "true"}
            disabled={saving}
            onChange={(e) => handleBoolToggle(e.target.checked)}
            style={{ cursor: saving ? "wait" : "pointer", accentColor: "var(--blue)" }}
          />
        ) : (
          <input
            type="number"
            value={draft}
            min={setting.min ?? undefined}
            max={setting.max ?? undefined}
            step={setting.type === "int" ? 1 : "any"}
            disabled={saving}
            onChange={(e) => { setDraft(e.target.value); setDirty(true); }}
            onKeyDown={(e) => { if (e.key === "Enter" && dirty) handleSave(); }}
            style={{
              background: "var(--bg)",
              border: `1px solid ${dirty ? "var(--blue)" : "var(--border)"}`,
              borderRadius: 4,
              color: "var(--text)",
              padding: "2px 6px",
              width: 130,
              fontFamily: "inherit",
              fontSize: 12,
            }}
          />
        )}
      </td>
      <td style={{ whiteSpace: "nowrap" }}>
        {dirty && (
          <button
            onClick={handleSave}
            disabled={saving}
            style={{ ...btnBase, color: "var(--blue)", border: "1px solid rgba(88,166,255,0.3)" }}
          >
            {saving ? "…" : "save"}
          </button>
        )}
        {setting.override !== null && !dirty && (
          <button
            onClick={handleReset}
            disabled={saving}
            style={{ ...btnBase, color: "var(--text-muted)", border: "1px solid var(--border)" }}
          >
            reset
          </button>
        )}
        {flash && (
          <span className={flash.ok ? "positive" : "negative"} style={{ marginLeft: 8, fontSize: 11 }}>
            {flash.text}
          </span>
        )}
      </td>
    </tr>
  );
}

function SettingsTab() {
  const [settings, setSettings] = useState<SettingRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/settings");
      const json = (await res.json()) as { settings: SettingRow[] };
      setSettings(json.settings);
      setLastUpdate(new Date());
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  if (error) {
    return (
      <div style={{ background: "rgba(248,81,73,0.1)", border: "1px solid var(--red)", borderRadius: 8, padding: "10px 14px", color: "var(--red)" }}>
        {error}
      </div>
    );
  }
  if (!settings) return <p className="neutral">Loading settings…</p>;

  const settingsMap = Object.fromEntries(settings.map((s) => [s.key, s]));
  const lockedSettings = settings.filter((s) => s.locked);

  return (
    <>
      <div style={{ color: "var(--text-muted)", fontSize: 11, marginBottom: 16 }}>
        {lastUpdate ? <>last refreshed: {lastUpdate.toLocaleTimeString("pt-BR")} · </> : null}
        polls every {POLL_MS / 1000}s · changes take effect on the next bot cycle
      </div>

      {SETTING_GROUPS.map((group) => {
        const rows = group.keys.map((k) => settingsMap[k]).filter(Boolean) as SettingRow[];
        if (rows.length === 0) return null;
        return (
          <Section key={group.label} title={group.label}>
            <table>
              <thead>
                <tr>
                  <th style={{ width: "42%" }}>Key</th>
                  <th>Type</th>
                  <th>Range</th>
                  <th>Value</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((s) => (
                  <EditableRow key={s.key} setting={s} onRefresh={load} />
                ))}
              </tbody>
            </table>
          </Section>
        );
      })}

      <Section title="Execution & Risk — locked">
        <div style={{ padding: "8px 12px", color: "var(--text-muted)", fontSize: 11, borderBottom: "1px solid var(--border)" }}>
          Cannot be changed from the UI. Edit .env and restart the bot to modify these.
        </div>
        <table>
          <thead>
            <tr>
              <th style={{ width: "42%" }}>Key</th>
              <th>Type</th>
              <th>Value</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {lockedSettings.map((s) => (
              <LockedRow key={s.key} setting={s} />
            ))}
          </tbody>
        </table>
      </Section>
    </>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [data, setData] = useState<DashData | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("dashboard");

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/data");
      const json = (await res.json()) as DashData;
      if (json.error) {
        setError(json.error);
      } else {
        setData(json);
        setLastUpdate(new Date());
        setError(null);
      }
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  const tabBtnStyle = (tab: TabId): React.CSSProperties => ({
    padding: "4px 14px",
    borderRadius: 6,
    fontSize: 12,
    cursor: "pointer",
    fontFamily: "inherit",
    background: activeTab === tab ? "var(--surface)" : "transparent",
    border: `1px solid ${activeTab === tab ? "var(--border)" : "transparent"}`,
    color: activeTab === tab ? "var(--text)" : "var(--text-muted)",
  });

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "24px 20px" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24 }}>
        <h1>Soros</h1>
        <div style={{ display: "flex", gap: 4 }}>
          <button style={tabBtnStyle("dashboard")} onClick={() => setActiveTab("dashboard")}>Dashboard</button>
          <button style={tabBtnStyle("settings")} onClick={() => setActiveTab("settings")}>Settings</button>
        </div>
        <div style={{ marginLeft: "auto", color: "var(--text-muted)", fontSize: 11 }}>
          {lastUpdate ? <>last update: {lastUpdate.toLocaleTimeString("pt-BR")}{" · "}</> : null}
          polls every {POLL_MS / 1000}s
        </div>
      </div>

      {error && (
        <div style={{ background: "rgba(248,81,73,0.1)", border: "1px solid var(--red)", borderRadius: 8, padding: "10px 14px", marginBottom: 20, color: "var(--red)" }}>
          {error}
        </div>
      )}

      {activeTab === "settings" ? (
        <SettingsTab />
      ) : !data ? (
        <p className="neutral">Loading…</p>
      ) : data.empty ? (
        <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "40px 20px", textAlign: "center" }}>
          <p className="neutral">Database not found yet.</p>
          <p className="neutral" style={{ marginTop: 8, fontSize: 11 }}>Start the bot first: <code>python main.py</code></p>
        </div>
      ) : (
        <>
          <EquityCard data={data} />
          <SweepTable />
          <PositionsTable positions={data.positions} />
          <ScreenerTable screener={data.screener ?? []} />
          <SignalsTable signals={data.signals} />
          <SentimentTable sentiment={data.sentiment} />
          <OrdersTable orders={data.orders} />
          <EventsTable events={data.events} />
        </>
      )}
    </div>
  );
}
