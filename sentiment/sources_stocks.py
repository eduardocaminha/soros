"""Stock sentiment sources — keyless always-on + optional keyed adapters.

Fetches sentiment signals for a stock symbol:
  1. CNN Fear & Greed Index  (production.dataviz.cnn.io) — keyless
  2. Yahoo Finance: 24 h price change, 5 d price change — keyless
  3. Yahoo Finance: up to 5 recent news headlines — keyless
  4. Finnhub: per-ticker news sentiment score — optional (FINNHUB_API_KEY)

All network calls are best-effort.  Any individual failure populates
that field with None / [] rather than raising, so partial data is still
useful.  The keyed Finnhub source degrades to None (treated as neutral)
when no key is present or the call fails.

``pre_score()`` aggregates available numeric signals into a single [-1, 1]
score without calling any LLM.
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
class StockSources:
    """All collected sentiment signals for one stock symbol."""

    symbol: str       # e.g. "AAPL"
    fetched_at: int   # unix seconds

    # CNN Fear & Greed Index (market-wide, keyless)
    fear_greed_value: int | None = None   # 0 (extreme fear) … 100 (extreme greed)
    fear_greed_label: str | None = None   # e.g. "Fear", "Extreme Greed"

    # Yahoo Finance price data (keyless)
    price_change_24h_pct: float | None = None
    price_change_5d_pct: float | None = None

    # Yahoo Finance news (keyless, max 5 headlines)
    news_headlines: list[str] = field(default_factory=list)

    # Finnhub per-ticker news sentiment (optional — None when key absent or call fails)
    # Range: -1.0 (very bearish) … +1.0 (very bullish)
    finnhub_sentiment_score: float | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, *, extra_headers: dict[str, str] | None = None) -> dict | list | None:
    """GET *url* and decode JSON; return None on any network / parse error."""
    headers = {"User-Agent": "soros-bot/1.0"}
    if extra_headers:
        headers.update(extra_headers)
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except (URLError, ValueError, TimeoutError, OSError) as exc:
        _log.debug("_get_json failed for %s: %s", url, exc)
        return None


def _fetch_cnn_fear_greed() -> tuple[int | None, str | None]:
    """Fetch CNN Fear & Greed Index (market-wide signal for stocks)."""
    data = _get_json("https://production.dataviz.cnn.io/index/fearandgreed/graphdata/")
    if not isinstance(data, dict):
        return None, None
    try:
        fg = data["fear_and_greed"]
        score = round(float(fg["score"]))
        label = str(fg["rating"]).title()  # "fear" → "Fear"
        return score, label
    except (KeyError, TypeError, ValueError):
        return None, None


def _fetch_yahoo_price(symbol: str) -> tuple[float | None, float | None]:
    """Return (price_change_24h_pct, price_change_5d_pct) from Yahoo Finance."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        "?range=5d&interval=1d"
    )
    data = _get_json(url)
    if not isinstance(data, dict):
        return None, None
    try:
        result = data["chart"]["result"][0]
        meta = result.get("meta", {})

        # 24 h change is provided directly in meta
        change_24h = meta.get("regularMarketChangePercent")

        # 5 d change: first vs last close in the 5d window
        closes: list[float | None] = result["indicators"]["quote"][0].get("close", [])
        valid = [c for c in closes if c is not None]
        if len(valid) >= 2:
            change_5d = (valid[-1] - valid[0]) / valid[0] * 100.0
        else:
            change_5d = None

        return (
            float(change_24h) if change_24h is not None else None,
            change_5d,
        )
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return None, None


def _fetch_yahoo_news(symbol: str) -> list[str]:
    """Return up to 5 recent news headlines for *symbol* (e.g. 'AAPL')."""
    url = (
        f"https://query2.finance.yahoo.com/v1/finance/search"
        f"?q={symbol}&newsCount=5&enableNavLinks=false"
    )
    data = _get_json(url)
    if not isinstance(data, dict):
        return []
    try:
        news = data.get("news", [])
        return [
            item["title"]
            for item in news[:5]
            if isinstance(item.get("title"), str)
        ]
    except (KeyError, TypeError):
        return []


def _fetch_finnhub_sentiment(symbol: str, api_key: str) -> float | None:
    """Fetch per-ticker news sentiment from Finnhub (keyed).

    Returns a score in [-1, 1] derived from ``bullishPercent`` minus
    ``bearishPercent`` from the Finnhub news-sentiment endpoint, or None
    when the call fails or yields no data.
    """
    url = (
        f"https://finnhub.io/api/v1/news-sentiment"
        f"?symbol={symbol}&token={api_key}"
    )
    data = _get_json(url)
    if not isinstance(data, dict):
        return None
    try:
        sentiment = data.get("sentiment") or {}
        bullish = float(sentiment.get("bullishPercent", 0))
        bearish = float(sentiment.get("bearishPercent", 0))
        total = bullish + bearish
        if total == 0:
            return None
        # Map to [-1, 1]: net bullish fraction
        return max(-1.0, min(1.0, (bullish - bearish) / total))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch(symbol: str, *, finnhub_api_key: str = "") -> StockSources:
    """Fetch all sentiment sources for *symbol* (e.g. 'AAPL').

    Each source is fetched independently; a partial failure yields a
    partially populated ``StockSources`` rather than raising.

    Parameters
    ----------
    symbol:
        Stock ticker, e.g. ``'AAPL'``.
    finnhub_api_key:
        When non-empty, enables the Finnhub keyed source.  Absent or
        empty key is a no-op — ``finnhub_sentiment_score`` stays None.
    """
    sources = StockSources(symbol=symbol, fetched_at=int(time.time()))

    sources.fear_greed_value, sources.fear_greed_label = _fetch_cnn_fear_greed()
    sources.price_change_24h_pct, sources.price_change_5d_pct = _fetch_yahoo_price(symbol)
    sources.news_headlines = _fetch_yahoo_news(symbol)

    if finnhub_api_key:
        sources.finnhub_sentiment_score = _fetch_finnhub_sentiment(symbol, finnhub_api_key)

    return sources


def pre_score(sources: StockSources) -> float:
    """Aggregate available numeric signals into a single score in [-1, 1].

    No LLM call is made.  Available sources are averaged; missing sources
    are treated as neutral (0.0) only when no numeric source is available at
    all (fallback).  When at least one numeric source exists, only present
    values are averaged so absent keyed sources do not dilute the signal.

    Scoring:
    - Fear & Greed: (value - 50) / 50  → [-1, 1]
    - Price 24 h: clamped to ±10 % → [-1, 1]
    - Finnhub sentiment: already [-1, 1]
    """
    scores: list[float] = []

    if sources.fear_greed_value is not None:
        scores.append((sources.fear_greed_value - 50) / 50.0)

    if sources.price_change_24h_pct is not None:
        scores.append(max(-1.0, min(1.0, sources.price_change_24h_pct / 10.0)))

    if sources.finnhub_sentiment_score is not None:
        scores.append(sources.finnhub_sentiment_score)

    if not scores:
        return 0.0
    return max(-1.0, min(1.0, sum(scores) / len(scores)))


def to_prompt_text(sources: StockSources) -> str:
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

    if sources.price_change_5d_pct is not None:
        sign = "+" if sources.price_change_5d_pct >= 0 else ""
        lines.append(f"- 5d price change: {sign}{sources.price_change_5d_pct:.2f}%")

    if sources.finnhub_sentiment_score is not None:
        lines.append(f"- Finnhub sentiment score: {sources.finnhub_sentiment_score:+.2f}")

    if sources.news_headlines:
        lines.append("- Recent headlines:")
        for headline in sources.news_headlines:
            lines.append(f"  * {headline}")

    return "\n".join(lines)
