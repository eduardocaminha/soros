"""Stocks sentiment sources — no API key required.

Fetches three publicly available signals for a stock ticker:
  1. CNN Fear & Greed Index  (production.dataviz.cnn.io)
  2. Yahoo Finance: 24 h / 7 d price change
  3. Yahoo Finance: up to 5 recent news headlines

All network calls are best-effort.  Any individual failure populates
that field with None / [] rather than raising, so partial data is still
useful for the analyst.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from urllib.error import URLError
from urllib.request import Request, urlopen

_log = logging.getLogger(__name__)

_TIMEOUT = 8  # seconds per HTTP request


@dataclass
class StocksSources:
    """All collected sentiment signals for one stock symbol."""

    symbol: str       # e.g. "AAPL"
    fetched_at: int   # unix seconds

    # CNN Fear & Greed Index (market-wide; 0 = extreme fear, 100 = extreme greed)
    fear_greed_value: int | None = None
    fear_greed_label: str | None = None

    # Yahoo Finance price data
    price_change_24h_pct: float | None = None
    price_change_7d_pct: float | None = None

    # Yahoo Finance news (max 5 headlines)
    news_headlines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_json(url: str) -> dict | list | None:
    """GET *url* and decode JSON; return None on any network / parse error."""
    try:
        req = Request(url, headers={"User-Agent": "soros-bot/1.0"})
        with urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except (URLError, ValueError, TimeoutError, OSError) as exc:
        _log.debug("_get_json failed for %s: %s", url, exc)
        return None


def _fetch_fear_greed() -> tuple[int | None, str | None]:
    """Fetch CNN Fear & Greed Index; return (value, label)."""
    data = _get_json("https://production.dataviz.cnn.io/index/fearandgreed/graphdata")
    if not isinstance(data, dict):
        return None, None
    try:
        fg = data["fear_and_greed"]
        value = int(round(float(fg["score"])))
        label = str(fg["rating"])
        return value, label
    except (KeyError, TypeError, ValueError):
        return None, None


def _fetch_yahoo_price(ticker: str) -> tuple[float | None, float | None]:
    """Return (price_change_24h_pct, price_change_7d_pct) from Yahoo Finance.

    Uses the unofficial chart endpoint with a 10-day daily range so we always
    have enough closing prices to compute both windows.
    """
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{ticker}?interval=1d&range=10d"
    )
    data = _get_json(url)
    if not isinstance(data, dict):
        return None, None
    try:
        result = data["chart"]["result"][0]
        closes: list[float] = result["indicators"]["quote"][0]["close"]
        # Drop any trailing None values that Yahoo sometimes returns for today
        closes = [c for c in closes if c is not None]
        if len(closes) < 2:
            return None, None

        last = closes[-1]
        prev = closes[-2]
        change_24h = (last - prev) / prev * 100.0 if prev else None

        change_7d: float | None = None
        if len(closes) >= 8:
            base = closes[-8]
            change_7d = (last - base) / base * 100.0 if base else None

        return change_24h, change_7d
    except (KeyError, IndexError, TypeError, ZeroDivisionError):
        return None, None


def _fetch_yahoo_news(ticker: str) -> list[str]:
    """Return up to 5 recent news headlines for *ticker* from Yahoo Finance."""
    url = (
        "https://query2.finance.yahoo.com/v1/finance/search"
        f"?q={ticker}&newsCount=5&quotesCount=0&enableFuzzyQuery=false"
    )
    data = _get_json(url)
    if not isinstance(data, dict):
        return []
    try:
        articles = data.get("news", [])
        return [
            a["title"]
            for a in articles[:5]
            if isinstance(a.get("title"), str)
        ]
    except (KeyError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch(symbol: str) -> StocksSources:
    """Fetch all sentiment sources for *symbol* (e.g. 'AAPL').

    Each source is fetched independently; a partial failure yields a
    partially populated ``StocksSources`` rather than raising.
    """
    sources = StocksSources(symbol=symbol, fetched_at=int(time.time()))

    sources.fear_greed_value, sources.fear_greed_label = _fetch_fear_greed()
    sources.price_change_24h_pct, sources.price_change_7d_pct = _fetch_yahoo_price(symbol)
    sources.news_headlines = _fetch_yahoo_news(symbol)

    return sources


def to_prompt_text(sources: StocksSources) -> str:
    """Format *sources* as a concise text block for an LLM sentiment prompt."""
    lines: list[str] = [f"Stock sentiment context for {sources.symbol}:"]

    if sources.fear_greed_value is not None:
        lines.append(
            f"- CNN Fear & Greed Index: {sources.fear_greed_value}/100"
            f" ({sources.fear_greed_label})"
        )

    if sources.price_change_24h_pct is not None:
        sign = "+" if sources.price_change_24h_pct >= 0 else ""
        lines.append(f"- 24h price change: {sign}{sources.price_change_24h_pct:.2f}%")

    if sources.price_change_7d_pct is not None:
        sign = "+" if sources.price_change_7d_pct >= 0 else ""
        lines.append(f"- 7d price change: {sign}{sources.price_change_7d_pct:.2f}%")

    if sources.news_headlines:
        lines.append("- Recent headlines:")
        for headline in sources.news_headlines:
            lines.append(f"  * {headline}")

    return "\n".join(lines)
