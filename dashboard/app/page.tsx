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

// ─── Helpers ─────────────────────────────────────────────────────────────────

const POLL_MS = 15_000; // 15 s

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
      {/* Equity */}
      <Card label="Equity">
        {eq ? (
          <>
            <big>${fmt(eq.equity)}</big>
            {eq.is_paper ? <span className="badge badge-paper" style={{ marginLeft: 8 }}>PAPER</span> : <span className="badge badge-live" style={{ marginLeft: 8 }}>LIVE</span>}
          </>
        ) : <span className="neutral">—</span>}
      </Card>

      {/* Drawdown */}
      <Card label="Drawdown">
        {eq ? (
          <span className={eq.drawdown_pct > 10 ? "negative" : eq.drawdown_pct > 5 ? "" : "positive"}>
            {fmt(eq.drawdown_pct, 1)}%{" "}
            <span className="neutral" style={{ fontSize: 11 }}>/ 15% limit</span>
          </span>
        ) : <span className="neutral">—</span>}
      </Card>

      {/* Unrealized P&L */}
      <Card label="Unrealized P&L">
        {data.positions.length ? fmtPnl(totalUnrealized) : <span className="neutral">—</span>}
      </Card>

      {/* Realized P&L */}
      <Card label="Realized P&L">
        {fmtPnl(data.realizedPnl)}
      </Card>

      {/* Sparkline */}
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

// ─── App ─────────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [data, setData] = useState<DashData | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "24px 20px" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <h1>Soros</h1>
        <div style={{ color: "var(--text-muted)", fontSize: 11 }}>
          {lastUpdate ? <>last update: {lastUpdate.toLocaleTimeString("pt-BR")}{" · "}</> : null}
          polls every {POLL_MS / 1000}s
        </div>
      </div>

      {error && (
        <div style={{ background: "rgba(248,81,73,0.1)", border: "1px solid var(--red)", borderRadius: 8, padding: "10px 14px", marginBottom: 20, color: "var(--red)" }}>
          {error}
        </div>
      )}

      {!data ? (
        <p className="neutral">Loading…</p>
      ) : data.empty ? (
        <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "40px 20px", textAlign: "center" }}>
          <p className="neutral">Database not found yet.</p>
          <p className="neutral" style={{ marginTop: 8, fontSize: 11 }}>Start the bot first: <code>python main.py</code></p>
        </div>
      ) : (
        <>
          <EquityCard data={data} />
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
