"""Crypto sentiment sources — all keyless, always-on.

Fetches sentiment signals for a crypto symbol:
  1. Crypto Fear & Greed Index  (alternative.me/fng) — keyless
  2. CoinGecko: 24 h / 7 d price change, market-cap rank — keyless
  3. CoinGecko: community sentiment votes per coin — keyless
  4. CryptoCompare: up to 5 recent news headlines — keyless

All network calls are best-effort.  Any individual failure populates
that field with None / [] rather than raising, so partial data is still
useful.  Absent sources degrade to neutral (0.0) in ``pre_score()``.

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

# Base-currency → CoinGecko coin ID.
_COINGECKO_IDS: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "UNI": "uniswap",
    "LTC": "litecoin",
    "ATOM": "cosmos",
}


@dataclass
class CryptoSources:
    """All collected sentiment signals for one crypto symbol."""

    symbol: str       # e.g. "BTC/USDT"
    fetched_at: int   # unix seconds

    # alternative.me Fear & Greed Index (keyless)
    fear_greed_value: int | None = None   # 0 (extreme fear) … 100 (extreme greed)
    fear_greed_label: str | None = None   # e.g. "Fear", "Extreme Greed"

    # CoinGecko market data (keyless)
    price_change_24h_pct: float | None = None
    price_change_7d_pct: float | None = None
    market_cap_rank: int | None = None

    # CryptoCompare news headlines (keyless, max 5)
    news_headlines: list[str] = field(default_factory=list)

    # CoinGecko community sentiment votes per coin (keyless)
    # Derived from sentiment_votes_up_percentage via (up - 50) / 50 → [-1, 1]
    coingecko_sentiment_score: float | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _base(symbol: str) -> str:
    """'BTC/USDT' → 'BTC'."""
    return symbol.split("/")[0].upper()


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
    data = _get_json("https://api.alternative.me/fng/?limit=1")
    if not isinstance(data, dict):
        return None, None
    try:
        entry = data["data"][0]
        return int(entry["value"]), str(entry["value_classification"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None, None


def _fetch_coingecko(base: str) -> tuple[float | None, float | None, int | None]:
    """Return (price_change_24h_pct, price_change_7d_pct, market_cap_rank)."""
    coin_id = _COINGECKO_IDS.get(base, base.lower())
    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        f"?vs_currency=usd&ids={coin_id}&price_change_percentage=7d"
    )
    data = _get_json(url)
    if not isinstance(data, list) or not data:
        return None, None, None
    coin = data[0]
    return (
        coin.get("price_change_percentage_24h"),
        coin.get("price_change_percentage_7d_in_currency"),
        coin.get("market_cap_rank"),
    )


def _fetch_coingecko_sentiment(base: str) -> float | None:
    """Fetch per-coin community sentiment from CoinGecko (keyless).

    Hits ``/coins/{id}`` and reads ``sentiment_votes_up_percentage``.
    Converts to [-1, 1] via ``(up - 50) / 50``: a coin with 100 % up-votes
    scores +1.0, 50 % scores 0.0, 0 % scores -1.0.

    Returns None when the call fails or the field is absent.
    """
    coin_id = _COINGECKO_IDS.get(base, base.lower())
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        "?localization=false&tickers=false&market_data=false"
        "&community_data=false&developer_data=false&sparkline=false"
    )
    data = _get_json(url)
    if not isinstance(data, dict):
        return None
    try:
        up = float(data["sentiment_votes_up_percentage"])
        return max(-1.0, min(1.0, (up - 50.0) / 50.0))
    except (KeyError, TypeError, ValueError):
        return None


def _fetch_news(base: str) -> list[str]:
    """Return up to 5 recent news headlines for *base* (e.g. 'BTC')."""
    url = (
        "https://min-api.cryptocompare.com/data/v2/news/"
        f"?categories={base}&excludeCategories=Sponsored&lang=EN"
    )
    data = _get_json(url)
    if not isinstance(data, dict):
        return []
    try:
        articles = data.get("Data", [])
        return [a["title"] for a in articles[:5] if isinstance(a.get("title"), str)]
    except (KeyError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch(symbol: str) -> CryptoSources:
    """Fetch all sentiment sources for *symbol* (e.g. 'BTC/USDT').

    All sources are keyless and always attempted.  Each source is fetched
    independently; a partial failure yields a partially populated
    ``CryptoSources`` rather than raising.

    Parameters
    ----------
    symbol:
        Trading pair, e.g. ``'BTC/USDT'``.
    """
    base = _base(symbol)
    sources = CryptoSources(symbol=symbol, fetched_at=int(time.time()))

    sources.fear_greed_value, sources.fear_greed_label = _fetch_fear_greed()
    (
        sources.price_change_24h_pct,
        sources.price_change_7d_pct,
        sources.market_cap_rank,
    ) = _fetch_coingecko(base)
    sources.news_headlines = _fetch_news(base)
    sources.coingecko_sentiment_score = _fetch_coingecko_sentiment(base)

    return sources


def pre_score(sources: CryptoSources) -> float:
    """Aggregate available numeric signals into a single score in [-1, 1].

    No LLM call is made.  Available sources are averaged; missing sources
    are treated as neutral (0.0) only when no numeric source is available at
    all (fallback).  When at least one numeric source exists, only present
    values are averaged so absent sources do not dilute the signal.

    Scoring:
    - Fear & Greed: (value - 50) / 50  → [-1, 1]
    - Price 24 h: clamped to ±10 % → [-1, 1]
    - CoinGecko sentiment: already [-1, 1] via (up - 50) / 50
    """
    scores: list[float] = []

    if sources.fear_greed_value is not None:
        scores.append((sources.fear_greed_value - 50) / 50.0)

    if sources.price_change_24h_pct is not None:
        scores.append(max(-1.0, min(1.0, sources.price_change_24h_pct / 10.0)))

    if sources.coingecko_sentiment_score is not None:
        scores.append(sources.coingecko_sentiment_score)

    if not scores:
        return 0.0
    return max(-1.0, min(1.0, sum(scores) / len(scores)))


def to_prompt_text(sources: CryptoSources) -> str:
    """Format *sources* as a concise text block for an LLM sentiment prompt."""
    lines: list[str] = [f"Crypto sentiment context for {sources.symbol}:"]

    if sources.fear_greed_value is not None:
        lines.append(
            f"- Fear & Greed Index: {sources.fear_greed_value}/100"
            f" ({sources.fear_greed_label})"
        )

    if sources.price_change_24h_pct is not None:
        sign = "+" if sources.price_change_24h_pct >= 0 else ""
        lines.append(f"- 24h price change: {sign}{sources.price_change_24h_pct:.2f}%")

    if sources.price_change_7d_pct is not None:
        sign = "+" if sources.price_change_7d_pct >= 0 else ""
        lines.append(f"- 7d price change: {sign}{sources.price_change_7d_pct:.2f}%")

    if sources.market_cap_rank is not None:
        lines.append(f"- Market cap rank: #{sources.market_cap_rank}")

    if sources.coingecko_sentiment_score is not None:
        lines.append(
            f"- CoinGecko community sentiment score: {sources.coingecko_sentiment_score:+.2f}"
        )

    if sources.news_headlines:
        lines.append("- Recent headlines:")
        for headline in sources.news_headlines:
            lines.append(f"  * {headline}")

    return "\n".join(lines)
