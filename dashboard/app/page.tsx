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

interface BenchmarkSeries {
  timestamps: number[];
  sorosEquity: number[];
  btcEquity: number[];
  initialCapital: number;
  btcStartPrice: number;
  windowStart: number;
  windowEnd: number;
  nPoints: number;
  nBtcGaps: number;
}

interface BenchmarkMetrics {
  sorosTotalReturn: number;
  sorosSharpe: number | null;
  sorosMaxDrawdown: number;
  btcTotalReturn: number;
  btcSharpe: number | null;
  btcMaxDrawdown: number;
  n: number;
  annualizationFactor: number;
  medianIntervalSeconds: number;
  sharpeConclusive: boolean;
  riskFreeRate: number;
}

interface BenchmarkData {
  ts: number;
  series?: BenchmarkSeries;
  metrics?: BenchmarkMetrics;
  empty?: boolean;
  reason?: string;
  error?: string;
}

interface ABMetrics {
  total_return: number;
  cagr: number;
  sharpe: number;
  max_dd: number;
  win_rate: number;
  n_trades: number;
}

interface ABSeries {
  timestamps: number[];
  offEquity: number[];
  onEquity: number[];
  initialCapital: number;
  windowStart: number;
  windowEnd: number;
  nPoints: number;
}

interface BacktestABData {
  ts: number;
  run_id?: string;
  run_ts?: number;
  start_ts?: number;
  end_ts?: number;
  symbols?: string[];
  fng_coverage_pct?: number;
  off?: ABMetrics;
  on?: ABMetrics;
  series?: ABSeries;
  empty?: boolean;
  error?: string;
}

type TabId = "dashboard" | "alerts" | "settings";

const ALERTS_STORAGE_KEY = "soros_alerts_last_visit";

const SETTING_GROUPS: Array<{ label: string; keys: string[] }> = [
  { label: "Loop", keys: ["LOOP_INTERVAL_SECONDS"] },
  { label: "Sinais", keys: ["SIGNAL_THRESHOLD", "DEBATE_DIVERGENCE_THRESHOLD"] },
  { label: "Screener", keys: ["SCREENER_ENABLED", "SCREENER_TOP_N", "SCREENER_MIN_VOLUME_USD"] },
  { label: "Market Cap / DEX", keys: ["MARKETCAP_TOP_N", "MARKETCAP_REFRESH_SECS", "DEX_BOOST_MULTIPLIER", "DEX_SCAN_CACHE_SECS"] },
  { label: "Gems", keys: ["GEM_VOLUME_SURGE_MULTIPLIER", "GEM_ROC_MIN_PCT", "GEM_TOP_N", "GEM_MIN_VOLUME_USD", "IGNITION_WEIGHT", "GEM_POSITION_SIZE_PCT", "GEM_TRAILING_STOP_PCT"] },
  { label: "Posição & Taxas", keys: ["POSITION_SIZE_PCT", "WATCHLIST_OHLCV_LIMIT", "SENTIMENT_MAX_AGE_SECONDS", "INITIAL_CAPITAL", "FEE_PCT", "SLIPPAGE_PCT"] },
];

// ─── Glossário ───────────────────────────────────────────────────────────────

const GLOSSARY: Record<string, string> = {
  "Drawdown": "Queda máxima do pico de equity até o vale mais profundo. O bot interrompe novas ordens ao atingir 15%.",
  "Equity": "Valor total da carteira: capital base + P&L realizado + P&L não realizado das posições abertas.",
  "P&L": "Profit & Loss — lucro ou prejuízo. 'Realizado': trades fechados. 'Não realizado': posições ainda abertas.",
  "Momentum": "Sinal baseado em tendência de preço (MACD, EMA, RSI). Positivo = tendência de alta; negativo = baixa.",
  "Funding": "Taxa periódica em contratos perpétuos de cripto. Positiva = longs pagam shorts (mercado aquecido); negativa = shorts pagam longs.",
  "Composite": "Score ponderado que combina momentum, volatilidade, funding e sentimento. Superar o threshold aciona compra/venda.",
  "Volatilidade": "Dispersão dos retornos do ativo. Alta volatilidade indica risco elevado e é usada como sinal de qualidade de entrada.",
  "Sentimento": "Score de sentimento gerado pelo Claude (IA). Analisa notícias e indicadores para estimar o viés otimista/pessimista.",
  "Confiança": "Grau de certeza do modelo de sentimento na classificação emitida (0–100%).",
  "Debate": "Debate bull × bear entre dois agentes IA. Ativado quando o sentimento contradiz o score determinístico, refinando o sinal.",
  "Convicção": "Score final do screener para o símbolo (0–1). Combina composite e sentimento ponderados para ranquear candidatos.",
  "Screener": "Módulo de seleção automática: ranqueia o universo por convicção e escolhe os top-N símbolos a cada ciclo.",
  "Origem": "Como o símbolo entrou no universo: base (market cap), gem (ignição de volume), dex (tendência em DEX).",
  "Gem": "Candidato de ignição: ativo com volume surging (≥ 2× média) e ROC elevado, detectado pelo gem scanner.",
  "Threshold": "Limiar mínimo do composite score para acionar uma ordem. Mais alto = mais seletivo; mais baixo = mais trades.",
  "CAGR": "Compound Annual Growth Rate — retorno anualizado da estratégia como se crescesse a taxa constante.",
  "Sharpe": "Relação retorno/risco ajustado. Quanto maior, melhor a qualidade do retorno. Acima de 1,0 é considerado bom.",
  "Max DD": "Drawdown máximo no período do backtest — maior queda pico-a-vale registrada.",
  "Win Rate": "Percentual de trades lucrativos. Ex: 60% = 6 de cada 10 trades fecharam no positivo.",
  "Paper": "Modo simulação: ordens são executadas virtualmente sem dinheiro real. Obrigatório 48 h+ antes de ativar live.",
  "Benchmark BTC": "Quanto valeria o mesmo capital inicial se simplesmente comprado e mantido em BTC desde o início do período (buy-and-hold). É o benchmark honesto para bots de cripto: mais de 80% dos bots de varejo ficam abaixo desta barra depois de custos.",
  "Retorno Total": "Variação percentual do capital desde o início da janela até o momento atual: (valor_atual − capital_inicial) / capital_inicial × 100.",
  "Backtest A/B": "Comparação entre duas variantes do backtest: 'Sem Sentimento' usa apenas sinais determinísticos (momentum, volatilidade, funding); 'Com Sentimento' injeta o Fear & Greed histórico como sinal de sentimento. Permite avaliar se o sentimento melhora ou piora os resultados.",
  "Cobertura F&G": "Fração das barras do backtest que tinham um valor do índice Fear & Greed disponível no alternative.me. Abaixo de 80% indica que parte do período é estimada por backward-fill.",
  "Fear & Greed": "Índice de sentimento de mercado do alternative.me (0 = medo extremo, 100 = ganância extrema). Valores < 40 geram score negativo (bearish); > 60 geram score positivo (bullish); 40–60 são neutros.",
};

const SETTINGS_DESCRIPTIONS: Record<string, string> = {
  "LOOP_INTERVAL_SECONDS": "Intervalo entre ciclos do bot (segundos). A cada ciclo, o bot reavalia o mercado e executa ordens.",
  "SIGNAL_THRESHOLD": "Score composto mínimo para acionar compra/venda. Mais alto = estratégia mais conservadora e seletiva.",
  "DEBATE_DIVERGENCE_THRESHOLD": "Quando o |score composto| cai abaixo deste valor, o debate bull/bear via IA é ativado para refinar o sinal.",
  "SCREENER_ENABLED": "Ativa o screener automático. Quando true, o bot ranqueia o universo e seleciona os top-N símbolos por ciclo.",
  "SCREENER_TOP_N": "Máximo de símbolos selecionados pelo screener por ciclo (excluindo pinned, que são sempre incluídos).",
  "SCREENER_MIN_VOLUME_USD": "Volume mínimo de 24 h (USD) que um símbolo precisa ter para ser selecionado pelo screener.",
  "MARKETCAP_TOP_N": "Top-N coins por market cap (CoinGecko) incluídas no universo base a cada refresh.",
  "MARKETCAP_REFRESH_SECS": "Intervalo de atualização do ranking de market cap via CoinGecko (segundos).",
  "DEX_BOOST_MULTIPLIER": "Multiplicador no gem_score quando o ativo também está em alta em DEX (DexScreener/GeckoTerminal). 1.0 = sem boost.",
  "DEX_SCAN_CACHE_SECS": "Tempo de cache dos resultados DEX antes de nova consulta (segundos).",
  "GEM_VOLUME_SURGE_MULTIPLIER": "Volume surge mínimo: o ativo precisa de ≥ N× o seu volume médio para qualificar como gem.",
  "GEM_ROC_MIN_PCT": "Rate-of-change mínimo (%) na janela curta para qualificar como candidato gem.",
  "GEM_TOP_N": "Máximo de gems surfaced pelo scanner por ciclo.",
  "GEM_MIN_VOLUME_USD": "Piso de liquidez (USD 24 h) para um candidato gem.",
  "IGNITION_WEIGHT": "Peso do sinal de ignição (gem) no composite score. 0.0 = sinal de ignição desabilitado.",
  "GEM_POSITION_SIZE_PCT": "Fração do equity por posição gem (deve ser ≤ POSITION_SIZE_PCT). Ex: 0.05 = 5%.",
  "GEM_TRAILING_STOP_PCT": "Distância do trailing stop para posições gem (fração). Ex: 0.05 = 5%. 0.0 = desabilitado.",
  "POSITION_SIZE_PCT": "Fração do equity alocada por posição padrão. Ex: 0.10 = 10% do capital por trade.",
  "WATCHLIST_OHLCV_LIMIT": "Candles históricos por símbolo da watchlist. Mínimo de 26 para que os indicadores funcionem.",
  "SENTIMENT_MAX_AGE_SECONDS": "Idade máxima (segundos) do sinal de sentimento antes de ser descartado como obsoleto.",
  "INITIAL_CAPITAL": "Capital inicial (USD) para cálculo de P&L e equity curve no modo paper.",
  "FEE_PCT": "Taxa de corretagem por lado de cada trade. Ex: 0.001 = 0,1% (padrão Binance maker).",
  "SLIPPAGE_PCT": "Deslizamento estimado de preço por lado de cada trade. Ex: 0.0005 = 0,05%.",
};

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

// ─── Tooltip ─────────────────────────────────────────────────────────────────

function Tooltip({ text, children }: { text: string; children: React.ReactNode }) {
  const [show, setShow] = useState(false);
  return (
    <span
      style={{ position: "relative", display: "inline-flex", alignItems: "center", gap: 3 }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      <span style={{
        fontSize: 9,
        color: "var(--text-muted)",
        border: "1px solid var(--border)",
        borderRadius: "50%",
        width: 13,
        height: 13,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
        cursor: "help",
        lineHeight: 1,
      }}>?</span>
      {show && (
        <span style={{
          position: "absolute",
          top: "calc(100% + 4px)",
          left: 0,
          zIndex: 1000,
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 6,
          padding: "7px 10px",
          fontSize: 11,
          color: "var(--text)",
          whiteSpace: "normal",
          minWidth: 200,
          maxWidth: 280,
          boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
          lineHeight: 1.45,
          pointerEvents: "none",
        }}>
          {text}
        </span>
      )}
    </span>
  );
}

// ─── Sections ────────────────────────────────────────────────────────────────

function EquityCard({ data }: { data: DashData }) {
  const eq = data.equity;
  const totalUnrealized = data.positions.reduce((s, p) => s + p.unrealized_pnl, 0);

  return (
    <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginBottom: 24 }}>
      <Card label={<Tooltip text={GLOSSARY["Equity"]}>Equity</Tooltip>}>
        {eq ? (
          <>
            <big>${fmt(eq.equity)}</big>
            {eq.is_paper ? <span className="badge badge-paper" style={{ marginLeft: 8 }}>PAPER</span> : <span className="badge badge-live" style={{ marginLeft: 8 }}>LIVE</span>}
          </>
        ) : <span className="neutral">—</span>}
      </Card>

      <Card label={<Tooltip text={GLOSSARY["Drawdown"]}>Drawdown</Tooltip>}>
        {eq ? (
          <span className={eq.drawdown_pct > 10 ? "negative" : eq.drawdown_pct > 5 ? "" : "positive"}>
            {fmt(eq.drawdown_pct, 1)}%{" "}
            <span className="neutral" style={{ fontSize: 11 }}>/ limite 15%</span>
          </span>
        ) : <span className="neutral">—</span>}
      </Card>

      <Card label={<Tooltip text={GLOSSARY["P&L"]}>P&L Não Realizado</Tooltip>}>
        {data.positions.length ? fmtPnl(totalUnrealized) : <span className="neutral">—</span>}
      </Card>

      <Card label={<Tooltip text={GLOSSARY["P&L"]}>P&L Realizado</Tooltip>}>
        {fmtPnl(data.realizedPnl)}
      </Card>

      <Card label="Equity (48h)" wide>
        <Sparkline points={data.equityCurve} />
      </Card>
    </div>
  );
}

function Card({ label, children, wide }: { label: React.ReactNode; children: React.ReactNode; wide?: boolean }) {
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
    <Section title="Posições Abertas" count={positions.length}>
      {positions.length === 0 ? (
        <p className="neutral" style={{ padding: "12px 10px" }}>Sem posições abertas.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Lado</th><th>Qtd</th><th>Entrada</th><th>Atual</th>
              <th>Não Real.</th><th>Modo</th><th>Abertura</th>
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
    <Section title="Sinais Recentes">
      {signals.length === 0 ? (
        <p className="neutral" style={{ padding: "12px 10px" }}>Sem sinais ainda.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Classe</th>
              <th><Tooltip text={GLOSSARY["Momentum"]}>Momentum</Tooltip></th>
              <th><Tooltip text={GLOSSARY["Volatilidade"]}>Volatilidade</Tooltip></th>
              <th><Tooltip text={GLOSSARY["Funding"]}>Funding</Tooltip></th>
              <th><Tooltip text={GLOSSARY["Composite"]}>Composite</Tooltip></th>
              <th>Ação</th><th>Em</th>
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
    <Section title="Sentimento">
      {sentiment.length === 0 ? (
        <p className="neutral" style={{ padding: "12px 10px" }}>Sem dados de sentimento ainda.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Classe</th>
              <th><Tooltip text={GLOSSARY["Composite"]}>Score</Tooltip></th>
              <th>Rótulo</th>
              <th><Tooltip text={GLOSSARY["Confiança"]}>Confiança</Tooltip></th>
              <th><Tooltip text={GLOSSARY["Debate"]}>Debate</Tooltip></th>
              <th>Em</th>
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
  pinned: "fixado",
  screener: "selecionado",
  volume_floor: "vol. baixo",
  sentiment_gate: "sentimento baixista",
  not_ranked: "sem ranking",
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
    <Section title="Universo & Seleção" count={selected.length}>
      <div style={{ padding: "8px 12px", color: "var(--text-muted)", fontSize: 11, borderBottom: "1px solid var(--border)" }}>
        Último screener: {runTs ? ts2str(runTs) : "—"} · {selected.length} de {screener.length} selecionados
      </div>
      <table>
        <thead>
          <tr>
            <th>Symbol</th><th>Classe</th>
            <th><Tooltip text={GLOSSARY["Origem"]}>Origem</Tooltip></th>
            <th>Tipo</th><th>Vol 24h (USD)</th>
            <th><Tooltip text={GLOSSARY["Composite"]}>Composite</Tooltip></th>
            <th><Tooltip text={GLOSSARY["Sentimento"]}>Sentimento</Tooltip></th>
            <th><Tooltip text={GLOSSARY["Convicção"]}>Convicção</Tooltip></th>
            <th>Estado</th>
          </tr>
        </thead>
        <tbody>
          {screener.map((e, i) => (
            <tr key={i} style={{ opacity: e.selected ? 1 : 0.55 }}>
              <td><b>{e.symbol}</b></td>
              <td className="neutral">{e.asset_class}</td>
              <td>{originBadge(e.origin)}</td>
              <td>{e.is_pinned ? <span className="badge badge-open">fixado</span> : <span className="neutral">monitorado</span>}</td>
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
    <Section title="Ordens Recentes" count={orders.length}>
      {orders.length === 0 ? (
        <p className="neutral" style={{ padding: "12px 10px" }}>Sem ordens ainda.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Lado</th><th>Qtd</th><th>Preço</th>
              <th>Tipo</th><th>Estado</th><th>Modo</th><th>Criada</th>
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

function EventsTable({ events, compact }: { events: Event[]; compact?: boolean }) {
  if (events.length === 0) {
    if (compact) return null;
    return (
      <Section title="Alertas">
        <p className="neutral" style={{ padding: "12px 10px" }}>Sem alertas.</p>
      </Section>
    );
  }
  const rows = compact ? events.slice(0, 5) : events;
  return (
    <Section title="Alertas" count={events.length}>
      <table>
        <thead>
          <tr><th>Hora</th><th>Nível</th><th>Componente</th><th>Mensagem</th></tr>
        </thead>
        <tbody>
          {rows.map((e, i) => (
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

function AlertsTab({ events }: { events: Event[] }) {
  return (
    <div>
      <div style={{ color: "var(--text-muted)", fontSize: 11, marginBottom: 16 }}>
        WARNINGs e ERRORs registrados pelo bot.
        {events.length > 0 && <> · {events.length} alerta{events.length !== 1 ? "s" : ""} no total.</>}
      </div>
      <EventsTable events={events} />
    </div>
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
    <Section title="Robustez do Threshold">
      <div style={{ padding: "8px 12px", color: "var(--text-muted)", fontSize: 11, borderBottom: "1px solid var(--border)" }}>
        Último sweep: {ts2str(sweep.run_ts)} · id do sweep: <code style={{ fontSize: 10 }}>{sweep.sweep_id.slice(0, 8)}</code>
        {" · "}
        <span style={{ color: "var(--blue)" }}>linha destacada = threshold atual ({cur})</span>
        {" · "}
        <span style={{ color: "var(--text-muted)" }}>objetivo: vizinhança estável, não retorno máximo</span>
      </div>
      <table>
        <thead>
          <tr>
            <th><Tooltip text={GLOSSARY["Threshold"]}>Threshold</Tooltip></th>
            <th>Retorno</th>
            <th><Tooltip text={GLOSSARY["CAGR"]}>CAGR</Tooltip></th>
            <th><Tooltip text={GLOSSARY["Sharpe"]}>Sharpe</Tooltip></th>
            <th><Tooltip text={GLOSSARY["Max DD"]}>Max DD</Tooltip></th>
            <th><Tooltip text={GLOSSARY["Win Rate"]}>Win Rate</Tooltip></th>
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
                    <span className="badge badge-open" style={{ marginLeft: 6, fontSize: 10 }}>ativo</span>
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

// ─── Benchmark Panel ──────────────────────────────────────────────────────────

function fmtEquityLabel(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `$${(v / 1_000).toFixed(1)}K`;
  return `$${Math.round(v)}`;
}

function BenchmarkChart({ series }: { series: BenchmarkSeries }) {
  const n = series.timestamps.length;
  if (n < 2) return <p className="neutral" style={{ padding: "12px 16px" }}>Dados insuficientes para o gráfico.</p>;

  const VW = 900;
  const VH = 180;
  const PAD = { top: 12, right: 16, bottom: 28, left: 64 };
  const chartW = VW - PAD.left - PAD.right;
  const chartH = VH - PAD.top - PAD.bottom;

  const all = [...series.sorosEquity, ...series.btcEquity];
  const minY = Math.min(...all);
  const maxY = Math.max(...all);
  const rangeY = maxY - minY || 1;

  const xOf = (i: number) => PAD.left + (i / (n - 1)) * chartW;
  const yOf = (v: number) => PAD.top + chartH - ((v - minY) / rangeY) * chartH;

  const sorosPath = series.sorosEquity
    .map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`)
    .join(" ");
  const btcPath = series.btcEquity
    .map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`)
    .join(" ");

  const yTicks = 4;
  const yTickVals = Array.from({ length: yTicks + 1 }, (_, k) => minY + (rangeY * k) / yTicks);

  const startDate = new Date(series.windowStart * 1000).toLocaleDateString("pt-BR");
  const endDate = new Date(series.windowEnd * 1000).toLocaleDateString("pt-BR");

  return (
    <svg
      viewBox={`0 0 ${VW} ${VH}`}
      style={{ width: "100%", height: VH, display: "block" }}
      aria-label="Gráfico de equity: Soros vs BTC buy-and-hold"
    >
      {yTickVals.map((v, k) => (
        <g key={k}>
          <line
            x1={PAD.left} y1={yOf(v)}
            x2={PAD.left + chartW} y2={yOf(v)}
            stroke="#30363d" strokeWidth="0.5" strokeDasharray="3,4"
          />
          <text x={PAD.left - 6} y={yOf(v) + 4} textAnchor="end" fontSize="10" fill="#8b949e">
            {fmtEquityLabel(v)}
          </text>
        </g>
      ))}

      <path d={btcPath} fill="none" stroke="#d29922" strokeWidth="2" strokeLinejoin="round" />
      <path d={sorosPath} fill="none" stroke="#58a6ff" strokeWidth="2" strokeLinejoin="round" />

      <text x={PAD.left} y={VH - 6} fontSize="10" fill="#8b949e">{startDate}</text>
      <text x={PAD.left + chartW} y={VH - 6} fontSize="10" fill="#8b949e" textAnchor="end">{endDate}</text>

      {/* Legend */}
      <rect x={PAD.left + chartW - 140} y={PAD.top} width={12} height={3} fill="#58a6ff" rx="1" />
      <text x={PAD.left + chartW - 124} y={PAD.top + 8} fontSize="11" fill="#e6edf3">Soros</text>
      <rect x={PAD.left + chartW - 70} y={PAD.top} width={12} height={3} fill="#d29922" rx="1" />
      <text x={PAD.left + chartW - 54} y={PAD.top + 8} fontSize="11" fill="#e6edf3">BTC B&amp;H</text>
    </svg>
  );
}

function BenchmarkPanel() {
  const [bm, setBm] = useState<BenchmarkData | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/benchmark");
      const json = (await res.json()) as BenchmarkData;
      setBm(json);
    } catch {
      // silent — will retry on next poll
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  if (!bm || bm.empty || bm.error || !bm.series || !bm.metrics) return null;

  const m = bm.metrics;
  const s = bm.series;
  const beating = m.sorosTotalReturn > m.btcTotalReturn;

  const fmtPct = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;

  const fmtSharpe = (v: number | null) => {
    if (v === null) return <span className="neutral">—</span>;
    return (
      <span className={v >= 1 ? "positive" : v >= 0 ? "" : "negative"}>
        {v.toFixed(2)}
        {!m.sharpeConclusive && <span className="neutral" style={{ fontSize: 10 }}> *</span>}
      </span>
    );
  };

  const fmtDd = (v: number) => {
    const pct = Math.abs(v) * 100;
    return <span className={pct > 20 ? "negative" : pct > 10 ? "" : "positive"}>{pct.toFixed(1)}%</span>;
  };

  return (
    <Section title="Benchmark vs BTC Buy-and-Hold">
      {/* Beating / losing indicator */}
      <div style={{
        padding: "10px 16px",
        borderBottom: "1px solid var(--border)",
        background: beating ? "rgba(63,185,80,0.06)" : "rgba(248,81,73,0.06)",
        display: "flex",
        alignItems: "center",
        gap: 16,
        flexWrap: "wrap",
      }}>
        <span style={{ fontSize: 15, fontWeight: 700, color: beating ? "var(--green)" : "var(--red)" }}>
          {beating ? "▲ Soros está BATENDO o benchmark" : "▼ Soros está PERDENDO para o benchmark"}
        </span>
        <span className="neutral" style={{ fontSize: 11 }}>
          Retorno Soros: <span className={m.sorosTotalReturn >= 0 ? "positive" : "negative"}>{fmtPct(m.sorosTotalReturn)}</span>
          {" · "}
          Retorno BTC B&amp;H: <span className={m.btcTotalReturn >= 0 ? "positive" : "negative"}>{fmtPct(m.btcTotalReturn)}</span>
        </span>
      </div>

      {/* Overlay chart */}
      <div style={{ padding: "12px 16px 0" }}>
        <BenchmarkChart series={s} />
      </div>

      {/* Side-by-side metrics table */}
      <table>
        <thead>
          <tr>
            <th style={{ width: "40%" }}>Métrica</th>
            <th>Soros</th>
            <th><Tooltip text={GLOSSARY["Benchmark BTC"]}>BTC Buy-and-Hold</Tooltip></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Tooltip text={GLOSSARY["Retorno Total"]}>Retorno Total</Tooltip></td>
            <td><span className={m.sorosTotalReturn >= 0 ? "positive" : "negative"}>{fmtPct(m.sorosTotalReturn)}</span></td>
            <td><span className={m.btcTotalReturn >= 0 ? "positive" : "negative"}>{fmtPct(m.btcTotalReturn)}</span></td>
          </tr>
          <tr>
            <td><Tooltip text={GLOSSARY["Sharpe"]}>Sharpe Anualizado</Tooltip></td>
            <td>{fmtSharpe(m.sorosSharpe)}</td>
            <td>{fmtSharpe(m.btcSharpe)}</td>
          </tr>
          <tr>
            <td><Tooltip text={GLOSSARY["Drawdown"]}>Max Drawdown</Tooltip></td>
            <td>{fmtDd(m.sorosMaxDrawdown)}</td>
            <td>{fmtDd(m.btcMaxDrawdown)}</td>
          </tr>
        </tbody>
      </table>

      {/* Footnotes */}
      <div style={{ padding: "8px 12px", color: "var(--text-muted)", fontSize: 10, borderTop: "1px solid var(--border)" }}>
        n={m.n} pontos · risk-free=0% · fator de anualização={m.annualizationFactor.toFixed(0)}×
        {!m.sharpeConclusive && ` · * Sharpe com n<30 pontos — não conclusivo`}
        {s.nBtcGaps > 0 && ` · ${s.nBtcGaps} gap(s) de preço BTC preenchidos por forward-fill`}
      </div>
    </Section>
  );
}

// ─── Backtest A/B Panel ───────────────────────────────────────────────────────

function BacktestABChart({ series }: { series: ABSeries }) {
  const n = series.timestamps.length;
  if (n < 2) return <p className="neutral" style={{ padding: "12px 16px" }}>Dados insuficientes para o gráfico.</p>;

  const VW = 900;
  const VH = 180;
  const PAD = { top: 12, right: 16, bottom: 28, left: 64 };
  const chartW = VW - PAD.left - PAD.right;
  const chartH = VH - PAD.top - PAD.bottom;

  const all = [...series.offEquity, ...series.onEquity];
  const minY = Math.min(...all);
  const maxY = Math.max(...all);
  const rangeY = maxY - minY || 1;

  const xOf = (i: number) => PAD.left + (i / (n - 1)) * chartW;
  const yOf = (v: number) => PAD.top + chartH - ((v - minY) / rangeY) * chartH;

  const offPath = series.offEquity
    .map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`)
    .join(" ");
  const onPath = series.onEquity
    .map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`)
    .join(" ");

  const yTicks = 4;
  const yTickVals = Array.from({ length: yTicks + 1 }, (_, k) => minY + (rangeY * k) / yTicks);

  const startDate = new Date(series.windowStart * 1000).toLocaleDateString("pt-BR");
  const endDate = new Date(series.windowEnd * 1000).toLocaleDateString("pt-BR");

  return (
    <svg
      viewBox={`0 0 ${VW} ${VH}`}
      style={{ width: "100%", height: VH, display: "block" }}
      aria-label="Backtest A/B: curvas de equity sem vs com sentimento"
    >
      {yTickVals.map((v, k) => (
        <g key={k}>
          <line
            x1={PAD.left} y1={yOf(v)}
            x2={PAD.left + chartW} y2={yOf(v)}
            stroke="#30363d" strokeWidth="0.5" strokeDasharray="3,4"
          />
          <text x={PAD.left - 6} y={yOf(v) + 4} textAnchor="end" fontSize="10" fill="#8b949e">
            {fmtEquityLabel(v)}
          </text>
        </g>
      ))}

      <path d={offPath} fill="none" stroke="#8b949e" strokeWidth="2" strokeLinejoin="round" strokeDasharray="6,3" />
      <path d={onPath} fill="none" stroke="#58a6ff" strokeWidth="2" strokeLinejoin="round" />

      <text x={PAD.left} y={VH - 6} fontSize="10" fill="#8b949e">{startDate}</text>
      <text x={PAD.left + chartW} y={VH - 6} fontSize="10" fill="#8b949e" textAnchor="end">{endDate}</text>

      {/* Legend */}
      <line x1={PAD.left + chartW - 160} y1={PAD.top + 4} x2={PAD.left + chartW - 148} y2={PAD.top + 4} stroke="#8b949e" strokeWidth="2" strokeDasharray="6,3" />
      <text x={PAD.left + chartW - 144} y={PAD.top + 8} fontSize="11" fill="#e6edf3">Sem sentimento</text>
      <rect x={PAD.left + chartW - 60} y={PAD.top + 1} width={12} height={3} fill="#58a6ff" rx="1" />
      <text x={PAD.left + chartW - 44} y={PAD.top + 8} fontSize="11" fill="#e6edf3">Com F&amp;G</text>
    </svg>
  );
}

function BacktestABPanel() {
  const [ab, setAb] = useState<BacktestABData | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/backtest-ab");
      const json = (await res.json()) as BacktestABData;
      setAb(json);
    } catch {
      // silent — will retry on next poll
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  if (!ab || ab.error || ab.empty || !ab.off || !ab.on || !ab.series) {
    if (!ab || ab.empty) {
      return (
        <Section title="Backtest A/B — Sentimento OFF vs ON">
          <div style={{ padding: "16px 20px", color: "var(--text-muted)", fontSize: 12 }}>
            Nenhum resultado ainda. Execute o comando para gerar o A/B:
            <pre style={{
              marginTop: 8,
              background: "var(--bg)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              padding: "8px 12px",
              fontSize: 11,
              color: "var(--text)",
              overflowX: "auto",
            }}>
              python -m backtest.ab_command --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31
            </pre>
          </div>
        </Section>
      );
    }
    return null;
  }

  const off = ab.off;
  const on = ab.on;
  const s = ab.series;

  const fmtPct = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;

  const fmtSharpe = (v: number) => {
    const conclusive = s.nPoints >= 30;
    return (
      <span className={v >= 1 ? "positive" : v >= 0 ? "" : "negative"}>
        {v.toFixed(2)}
        {!conclusive && <span className="neutral" style={{ fontSize: 10 }}> *</span>}
      </span>
    );
  };

  const fmtDd = (v: number) => {
    const pct = v * 100;
    return <span className={pct > 20 ? "negative" : pct > 10 ? "" : "positive"}>{pct.toFixed(1)}%</span>;
  };

  const onBetter = on.total_return > off.total_return;

  return (
    <Section title="Backtest A/B — Sentimento OFF vs ON">
      {/* F&G-only caveat — always visible */}
      <div style={{
        padding: "8px 16px",
        borderBottom: "1px solid var(--border)",
        background: "rgba(210,153,34,0.06)",
        fontSize: 11,
        color: "var(--text-muted)",
        display: "flex",
        gap: 8,
        alignItems: "flex-start",
      }}>
        <span style={{ color: "#d29922", fontWeight: 600, flexShrink: 0 }}>Aviso:</span>
        <span>
          O sentimento no backtest usa <b>somente o Fear &amp; Greed histórico</b> (alternative.me) —
          o índice de mercado inteiro, não por moeda. Votos CoinGecko e Claude não têm histórico,
          portanto este A/B <b>não replica o blend exato ao vivo</b>. Use como indicação, não conclusão.
          {typeof ab.fng_coverage_pct === "number" && (
            <> · <Tooltip text={GLOSSARY["Cobertura F&G"]}>Cobertura F&G: {(ab.fng_coverage_pct * 100).toFixed(1)}%</Tooltip></>
          )}
        </span>
      </div>

      {/* Overlay chart */}
      <div style={{ padding: "12px 16px 0" }}>
        <BacktestABChart series={s} />
      </div>

      {/* Side-by-side comparison header */}
      <div style={{
        padding: "8px 16px",
        borderTop: "1px solid var(--border)",
        fontSize: 13,
        fontWeight: 600,
        color: onBetter ? "var(--green)" : "var(--red)",
      }}>
        {onBetter
          ? "▲ Sentimento MELHOROU o retorno neste período"
          : "▼ Sentimento PIOROU o retorno neste período"}
      </div>

      {/* Side-by-side metrics */}
      <table>
        <thead>
          <tr>
            <th style={{ width: "38%" }}>Métrica</th>
            <th>Sem Sentimento</th>
            <th><Tooltip text={GLOSSARY["Fear & Greed"]}>Com Sentimento (F&G)</Tooltip></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Tooltip text={GLOSSARY["Retorno Total"]}>Retorno Total</Tooltip></td>
            <td><span className={off.total_return >= 0 ? "positive" : "negative"}>{fmtPct(off.total_return)}</span></td>
            <td><span className={on.total_return >= 0 ? "positive" : "negative"}>{fmtPct(on.total_return)}</span></td>
          </tr>
          <tr>
            <td><Tooltip text={GLOSSARY["CAGR"]}>CAGR</Tooltip></td>
            <td><span className={off.cagr >= 0 ? "positive" : "negative"}>{fmtPct(off.cagr)}</span></td>
            <td><span className={on.cagr >= 0 ? "positive" : "negative"}>{fmtPct(on.cagr)}</span></td>
          </tr>
          <tr>
            <td><Tooltip text={GLOSSARY["Sharpe"]}>Sharpe</Tooltip></td>
            <td>{fmtSharpe(off.sharpe)}</td>
            <td>{fmtSharpe(on.sharpe)}</td>
          </tr>
          <tr>
            <td><Tooltip text={GLOSSARY["Max DD"]}>Max Drawdown</Tooltip></td>
            <td>{fmtDd(off.max_dd)}</td>
            <td>{fmtDd(on.max_dd)}</td>
          </tr>
          <tr>
            <td><Tooltip text={GLOSSARY["Win Rate"]}>Win Rate</Tooltip></td>
            <td>{(off.win_rate * 100).toFixed(1)}%</td>
            <td>{(on.win_rate * 100).toFixed(1)}%</td>
          </tr>
          <tr>
            <td>Trades</td>
            <td className="neutral">{off.n_trades}</td>
            <td className="neutral">{on.n_trades}</td>
          </tr>
        </tbody>
      </table>

      {/* Footnote */}
      <div style={{ padding: "8px 12px", color: "var(--text-muted)", fontSize: 10, borderTop: "1px solid var(--border)" }}>
        n={s.nPoints} pontos
        {s.nPoints < 30 && " · * Sharpe com n<30 — não conclusivo"}
        {ab.run_ts && <> · executado em {ts2str(ab.run_ts)}</>}
        {ab.symbols && <> · {ab.symbols.join(", ")}</>}
      </div>
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
  const desc = SETTINGS_DESCRIPTIONS[setting.key];
  return (
    <tr style={{ opacity: 0.65 }}>
      <td>
        {desc ? (
          <Tooltip text={desc}><code style={{ fontSize: 11 }}>{setting.key}</code></Tooltip>
        ) : (
          <code style={{ fontSize: 11 }}>{setting.key}</code>
        )}
      </td>
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
      <td><span className="badge badge-error" style={{ fontSize: 10 }}>bloqueado</span></td>
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
        showFlash(true, "salvo");
        onRefresh();
      } else {
        const json = (await res.json()) as { error?: string };
        showFlash(false, json.error ?? "erro");
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
          showFlash(true, "salvo");
          onRefresh();
        } else {
          setDraft(prev);
          const json = (await res.json()) as { error?: string };
          showFlash(false, json.error ?? "erro");
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
        showFlash(true, "redefinido");
        onRefresh();
      } else {
        showFlash(false, "falha ao redefinir");
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

  const desc = SETTINGS_DESCRIPTIONS[setting.key];
  return (
    <tr>
      <td>
        {desc ? (
          <Tooltip text={desc}><code style={{ fontSize: 11 }}>{setting.key}</code></Tooltip>
        ) : (
          <code style={{ fontSize: 11 }}>{setting.key}</code>
        )}
        {setting.override !== null && (
          <span className="badge badge-warn" style={{ marginLeft: 6, fontSize: 10 }}>substituído</span>
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
            {saving ? "…" : "salvar"}
          </button>
        )}
        {setting.override !== null && !dirty && (
          <button
            onClick={handleReset}
            disabled={saving}
            style={{ ...btnBase, color: "var(--text-muted)", border: "1px solid var(--border)" }}
          >
            redefinir
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
  if (!settings) return <p className="neutral">Carregando configurações…</p>;

  const settingsMap = Object.fromEntries(settings.map((s) => [s.key, s]));
  const lockedSettings = settings.filter((s) => s.locked);

  return (
    <>
      <div style={{ color: "var(--text-muted)", fontSize: 11, marginBottom: 16 }}>
        {lastUpdate ? <>atualizado: {lastUpdate.toLocaleTimeString("pt-BR")} · </> : null}
        atualiza a cada {POLL_MS / 1000}s · alterações entram em vigor no próximo ciclo do bot
      </div>

      {SETTING_GROUPS.map((group) => {
        const rows = group.keys.map((k) => settingsMap[k]).filter(Boolean) as SettingRow[];
        if (rows.length === 0) return null;
        return (
          <Section key={group.label} title={group.label}>
            <table>
              <thead>
                <tr>
                  <th style={{ width: "42%" }}>Chave</th>
                  <th>Tipo</th>
                  <th>Intervalo</th>
                  <th>Valor</th>
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

      <Section title="Execução & Risco — bloqueado">
        <div style={{ padding: "8px 12px", color: "var(--text-muted)", fontSize: 11, borderBottom: "1px solid var(--border)" }}>
          Não pode ser alterado pela UI. Edite o .env e reinicie o bot para modificar.
        </div>
        <table>
          <thead>
            <tr>
              <th style={{ width: "42%" }}>Chave</th>
              <th>Tipo</th>
              <th>Valor</th>
              <th>Estado</th>
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
  const [lastAlertsVisit, setLastAlertsVisit] = useState<number>(() => {
    if (typeof window === "undefined") return 0;
    return parseInt(localStorage.getItem(ALERTS_STORAGE_KEY) ?? "0", 10);
  });

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

  const unreadAlerts = data
    ? data.events.filter((e) => e.ts > lastAlertsVisit).length
    : 0;

  const handleTabClick = (tab: TabId) => {
    setActiveTab(tab);
    if (tab === "alerts") {
      const now = Math.floor(Date.now() / 1000);
      localStorage.setItem(ALERTS_STORAGE_KEY, String(now));
      setLastAlertsVisit(now);
    }
  };

  const tabBtnStyle = (tab: TabId): React.CSSProperties => ({
    padding: "4px 14px",
    borderRadius: 6,
    fontSize: 12,
    cursor: "pointer",
    fontFamily: "inherit",
    background: activeTab === tab ? "var(--surface)" : "transparent",
    border: `1px solid ${activeTab === tab ? "var(--border)" : "transparent"}`,
    color: activeTab === tab ? "var(--text)" : "var(--text-muted)",
    position: "relative",
  });

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "24px 20px" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24 }}>
        <h1>Soros</h1>
        <div style={{ display: "flex", gap: 4 }}>
          <button style={tabBtnStyle("dashboard")} onClick={() => handleTabClick("dashboard")}>Dashboard</button>
          <button style={tabBtnStyle("alerts")} onClick={() => handleTabClick("alerts")}>
            Alertas
            {unreadAlerts > 0 && (
              <span style={{
                marginLeft: 6,
                background: "var(--red)",
                color: "#fff",
                borderRadius: 8,
                fontSize: 10,
                fontWeight: 700,
                padding: "1px 5px",
                lineHeight: 1.4,
                verticalAlign: "middle",
              }}>{unreadAlerts}</span>
            )}
          </button>
          <button style={tabBtnStyle("settings")} onClick={() => handleTabClick("settings")}>Configurações</button>
        </div>
        <div style={{ marginLeft: "auto", color: "var(--text-muted)", fontSize: 11 }}>
          {lastUpdate ? <>última atualização: {lastUpdate.toLocaleTimeString("pt-BR")}{" · "}</> : null}
          atualiza a cada {POLL_MS / 1000}s
        </div>
      </div>

      {error && (
        <div style={{ background: "rgba(248,81,73,0.1)", border: "1px solid var(--red)", borderRadius: 8, padding: "10px 14px", marginBottom: 20, color: "var(--red)" }}>
          {error}
        </div>
      )}

      {activeTab === "settings" ? (
        <SettingsTab />
      ) : activeTab === "alerts" ? (
        !data ? (
          <p className="neutral">Carregando…</p>
        ) : (
          <AlertsTab events={data.events} />
        )
      ) : !data ? (
        <p className="neutral">Carregando…</p>
      ) : data.empty ? (
        <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "40px 20px", textAlign: "center" }}>
          <p className="neutral">Banco de dados ainda não encontrado.</p>
          <p className="neutral" style={{ marginTop: 8, fontSize: 11 }}>Inicie o bot primeiro: <code>python main.py</code></p>
        </div>
      ) : (
        <>
          <EquityCard data={data} />
          <BenchmarkPanel />
          <BacktestABPanel />
          <SweepTable />
          <PositionsTable positions={data.positions} />
          <ScreenerTable screener={data.screener ?? []} />
          <SignalsTable signals={data.signals} />
          <SentimentTable sentiment={data.sentiment} />
          <OrdersTable orders={data.orders} />
        </>
      )}
    </div>
  );
}
