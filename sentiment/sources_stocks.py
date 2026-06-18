"""Stock sentiment sources — no API key required.

Fetches three publicly available signals for a stock ticker:
  1. Yahoo Finance: 24 h / 5 d price change, volume ratio vs 5 d avg
  2. VIX (^VIX) via Yahoo Finance as market-wide fear proxy
  3. Yahoo Finance RSS: up to 5 recent news headlines

All network calls are best-effort.  Any individual failure populates
that field with None / [] rather than raising, so partial data is still
useful for the analyst.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from urllib.error import URLError
from urllib.request import Request, urlopen

_log = logging.getLogger(__name__)

_TIMEOUT = 8  # seconds per HTTP request


@dataclass
class StockSources:
    """All collected sentiment signals for one stock symbol."""

    symbol: str       # e.g. "AAPL" or "AAPL:NASDAQ"
    fetched_at: int   # unix seconds

    # Yahoo Finance market data
    price_change_24h_pct: float | None = None
    price_change_5d_pct: float | None = None
    volume_ratio: float | None = None  # last day volume / 5 d average

    # VIX as market-wide fear proxy (higher = more fear)
    vix_value: float | None = None

    # Yahoo Finance RSS headlines (max 5)
    news_headlines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ticker(symbol: str) -> str:
    """'AAPL:NASDAQ' → 'AAPL'; 'AAPL' → 'AAPL'."""
    return symbol.split(":")[0].upper()


def _get_json(url: str) -> dict | list | None:
    """GET *url* and decode JSON; return None on any network / parse error."""
    try:
        req = Request(url, headers={"User-Agent": "soros-bot/1.0"})
        with urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except (URLError, ValueError, TimeoutError, OSError) as exc:
        _log.debug("_get_json failed for %s: %s", url, exc)
        return None


def _fetch_yf_quote(tick: str) -> tuple[float | None, float | None, float | None]:
    """Return (price_change_24h_pct, price_change_5d_pct, volume_ratio).

    Uses Yahoo Finance chart endpoint (no API key required).
    volume_ratio is last-day volume / average of all preceding days in window.
    """
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{tick}"
        "?interval=1d&range=7d"
    )
    data = _get_json(url)
    if not isinstance(data, dict):
        return None, None, None
    try:
        result = data["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]

        closes = [c for c in quote.get("close", []) if c is not None]
        volumes = [v for v in quote.get("volume", []) if v is not None]

        if len(closes) < 2:
            return None, None, None

        prev_close = closes[-2]
        cur_close = closes[-1]
        change_24h = ((cur_close - prev_close) / prev_close * 100) if prev_close else None

        change_5d = None
        if len(closes) >= 6:
            old = closes[-6]
            change_5d = ((cur_close - old) / old * 100) if old else None

        vol_ratio = None
        if len(volumes) >= 2:
            avg_prev = sum(volumes[:-1]) / len(volumes[:-1])
            vol_ratio = (volumes[-1] / avg_prev) if avg_prev else None

        return change_24h, change_5d, vol_ratio
    except (KeyError, IndexError, TypeError, ZeroDivisionError):
        return None, None, None


def _fetch_vix() -> float | None:
    """Return current VIX close from Yahoo Finance."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=2d"
    data = _get_json(url)
    if not isinstance(data, dict):
        return None
    try:
        closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        valids = [c for c in closes if c is not None]
        return float(valids[-1]) if valids else None
    except (KeyError, IndexError, TypeError):
        return None


def _fetch_news(tick: str) -> list[str]:
    """Return up to 5 recent headlines from Yahoo Finance RSS."""
    url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={tick}&region=US&lang=en-US"
    )
    try:
        req = Request(url, headers={"User-Agent": "soros-bot/1.0"})
        with urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        headlines: list[str] = []
        for item in root.iter("item"):
            title_el = item.find("title")
            if title_el is not None and isinstance(title_el.text, str):
                headlines.append(title_el.text.strip())
            if len(headlines) >= 5:
                break
        return headlines
    except (URLError, ET.ParseError, TimeoutError, OSError) as exc:
        _log.debug("_fetch_news failed for %s: %s", tick, exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch(symbol: str) -> StockSources:
    """Fetch all sentiment sources for *symbol* (e.g. 'AAPL' or 'AAPL:NASDAQ').

    Each source is fetched independently; a partial failure yields a
    partially populated ``StockSources`` rather than raising.
    """
    tick = _ticker(symbol)
    sources = StockSources(symbol=symbol, fetched_at=int(time.time()))

    (
        sources.price_change_24h_pct,
        sources.price_change_5d_pct,
        sources.volume_ratio,
    ) = _fetch_yf_quote(tick)
    sources.vix_value = _fetch_vix()
    sources.news_headlines = _fetch_news(tick)

    return sources


def to_prompt_text(sources: StockSources) -> str:
    """Format *sources* as a concise text block for an LLM sentiment prompt."""
    lines: list[str] = [f"Stock sentiment context for {sources.symbol}:"]

    if sources.price_change_24h_pct is not None:
        sign = "+" if sources.price_change_24h_pct >= 0 else ""
        lines.append(f"- 24h price change: {sign}{sources.price_change_24h_pct:.2f}%")

    if sources.price_change_5d_pct is not None:
        sign = "+" if sources.price_change_5d_pct >= 0 else ""
        lines.append(f"- 5d price change: {sign}{sources.price_change_5d_pct:.2f}%")

    if sources.volume_ratio is not None:
        lines.append(f"- Volume vs 5d avg: {sources.volume_ratio:.2f}x")

    if sources.vix_value is not None:
        lines.append(f"- VIX (market fear): {sources.vix_value:.1f}")

    if sources.news_headlines:
        lines.append("- Recent headlines:")
        for headline in sources.news_headlines:
            lines.append(f"  * {headline}")

    return "\n".join(lines)
