/**
 * Pure benchmark and metrics computation — no I/O, no Next.js dependencies.
 * Mirrors engine/benchmark.py and engine/metrics.py.
 */

// ---------------------------------------------------------------------------
// Benchmark construction
// ---------------------------------------------------------------------------

export interface BenchmarkSeries {
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

/**
 * Build a buy-and-hold BTC equity curve aligned to soros equity snapshots.
 *
 * For each snapshot timestamp, locates the most recent BTC/USDT close at or
 * before that timestamp (forward-fill for gaps).  Snapshots that fall before
 * the first available BTC price are excluded from the aligned window.
 */
export function buildBtcBenchmark(
  snapshots: [number, number][],
  btcCloses: [number, number][]
): BenchmarkSeries {
  if (snapshots.length === 0) throw new Error("snapshots is empty");

  const snap = [...snapshots].sort((a, b) => a[0] - b[0]);
  const sortedBtc = [...btcCloses].sort((a, b) => a[0] - b[0]);
  const btcTsArr = sortedBtc.map((x) => x[0]);
  const btcPriceArr = sortedBtc.map((x) => x[1]);

  const outTimestamps: number[] = [];
  const outSoros: number[] = [];
  const outBtcRaw: number[] = [];
  let nBtcGaps = 0;
  let btcStartPrice: number | null = null;

  for (const [snapTs, snapEq] of snap) {
    // bisect_right: rightmost idx where btcTsArr[idx] <= snapTs
    let lo = 0;
    let hi = btcTsArr.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (btcTsArr[mid] <= snapTs) lo = mid + 1;
      else hi = mid;
    }
    const idx = lo - 1;
    if (idx < 0) continue; // no BTC price at or before this snapshot

    const btcPrice = btcPriceArr[idx];
    if (btcTsArr[idx] !== snapTs) nBtcGaps++;

    if (btcStartPrice === null) btcStartPrice = btcPrice;

    outTimestamps.push(snapTs);
    outSoros.push(snapEq);
    outBtcRaw.push(btcPrice);
  }

  if (btcStartPrice === null)
    throw new Error("no BTC close prices overlap with the equity snapshot window");

  const initialCapital = outSoros[0];
  const btcEquity = outBtcRaw.map((p) => initialCapital * (p / btcStartPrice!));

  return {
    timestamps: outTimestamps,
    sorosEquity: outSoros,
    btcEquity,
    initialCapital,
    btcStartPrice,
    windowStart: outTimestamps[0],
    windowEnd: outTimestamps[outTimestamps.length - 1],
    nPoints: outTimestamps.length,
    nBtcGaps,
  };
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

export const MIN_SHARPE_N = 30;
const SECONDS_PER_YEAR = 365.25 * 24 * 3600;

export interface ComparisonMetrics {
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

function totalReturn(equity: number[]): number {
  if (equity.length < 1) return 0;
  return (equity[equity.length - 1] - equity[0]) / equity[0];
}

function maxDrawdown(equity: number[]): number {
  if (equity.length < 2) return 0;
  let peak = equity[0];
  let maxDd = 0;
  for (const v of equity) {
    if (v > peak) peak = v;
    const dd = (v - peak) / peak;
    if (dd < maxDd) maxDd = dd;
  }
  return maxDd;
}

function periodReturns(equity: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < equity.length; i++) out.push((equity[i] - equity[i - 1]) / equity[i - 1]);
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

function sharpe(equity: number[], annFactor: number): number | null {
  if (equity.length < 2) return null;
  const rets = periodReturns(equity);
  if (rets.length < 1) return null;
  const std = stdev(rets);
  if (std === 0) return null;
  return (mean(rets) / std) * Math.sqrt(annFactor);
}

function medianInterval(timestamps: number[]): number {
  if (timestamps.length < 2) return 3600;
  const diffs: number[] = [];
  for (let i = 1; i < timestamps.length; i++) diffs.push(timestamps[i] - timestamps[i - 1]);
  diffs.sort((a, b) => a - b);
  const mid = Math.floor(diffs.length / 2);
  return diffs.length % 2 === 0 ? (diffs[mid - 1] + diffs[mid]) / 2 : diffs[mid];
}

function annualizationFactor(medianIntervalSec: number): number {
  if (medianIntervalSec <= 0) return SECONDS_PER_YEAR / 3600;
  return SECONDS_PER_YEAR / medianIntervalSec;
}

export function computeMetrics(series: BenchmarkSeries): ComparisonMetrics {
  const medInterval = medianInterval(series.timestamps);
  const annFactor = annualizationFactor(medInterval);

  return {
    sorosTotalReturn: totalReturn(series.sorosEquity),
    sorosSharpe: sharpe(series.sorosEquity, annFactor),
    sorosMaxDrawdown: maxDrawdown(series.sorosEquity),
    btcTotalReturn: totalReturn(series.btcEquity),
    btcSharpe: sharpe(series.btcEquity, annFactor),
    btcMaxDrawdown: maxDrawdown(series.btcEquity),
    n: series.nPoints,
    annualizationFactor: annFactor,
    medianIntervalSeconds: medInterval,
    sharpeConclusive: series.nPoints >= MIN_SHARPE_N,
    riskFreeRate: 0.0,
  };
}
