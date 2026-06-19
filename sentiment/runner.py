"""Sentiment runner — orchestrates the full sentiment pipeline per cycle.

For each symbol:
  1. Fetch sentiment sources (crypto or stocks)
  2. Compute pre-scored aggregate (no LLM) → base score
  3. Run bull/bear debate only when quant and sentiment diverge → DebateResult
  4. Persist to sentiment_signals (SQLite)

Call ``run()`` once per main-loop cycle, passing the latest deterministic
composite scores so the debate module can detect sign conflicts.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import config
from database.db import get_connection
from sentiment import analyst, debate, sources_crypto, sources_stocks
from sentiment.claude_client import ClaudeClient
from sentiment.debate import DebateResult

_log = logging.getLogger(__name__)


@dataclass
class SentimentRecord:
    """Mirrors one row in the sentiment_signals table."""

    symbol: str
    asset_class: str   # 'crypto' | 'stocks'
    ts: int            # unix seconds
    score: float       # -1.0 … +1.0
    label: str         # 'bullish' | 'bearish' | 'neutral'
    confidence: float  # 0.0 … 1.0
    debate_used: bool
    raw_json: str | None


def _score_to_label(score: float) -> str:
    if score > 0.1:
        return "bullish"
    if score < -0.1:
        return "bearish"
    return "neutral"


def _analyse_symbol(
    symbol: str,
    asset_class: str,
    client: ClaudeClient,
    det_score: float,
) -> DebateResult:
    """Fetch sources, compute pre-score aggregate (no LLM), optionally debate."""
    if asset_class == "crypto":
        sources = sources_crypto.fetch(symbol)
        base_score = sources_crypto.pre_score(sources)
        sources_text = sources_crypto.to_prompt_text(sources)
    else:
        sources = sources_stocks.fetch(
            symbol, finnhub_api_key=config.FINNHUB_API_KEY
        )
        base_score = sources_stocks.pre_score(sources)
        sources_text = sources_stocks.to_prompt_text(sources)

    base_result = analyst.AnalystResult(
        symbol=symbol,
        score=base_score,
        rationale="pre-scored aggregate",
        analysed_at=int(time.time()),
        llm_used=False,
    )
    return debate.debate(sources_text, base_result, det_score, client)


def _persist(record: SentimentRecord) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO sentiment_signals
            (symbol, asset_class, ts, score, label, confidence, debate_used, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.symbol,
            record.asset_class,
            record.ts,
            record.score,
            record.label,
            record.confidence,
            1 if record.debate_used else 0,
            record.raw_json,
        ),
    )
    conn.commit()


def run(
    crypto_symbols: list[str] | None = None,
    stock_symbols: list[str] | None = None,
    *,
    deterministic_scores: dict[str, float] | None = None,
    client: ClaudeClient | None = None,
) -> list[SentimentRecord]:
    """Run the sentiment pipeline for all symbols and persist to SQLite.

    Parameters
    ----------
    crypto_symbols:
        Symbols to analyse as crypto (e.g. ``['BTC/USDT', 'ETH/USDT']``).
        Defaults to ``config.CRYPTO_SYMBOLS``.
    stock_symbols:
        Symbols to analyse as stocks (e.g. ``['AAPL', 'MSFT']``).
        Defaults to ``config.STOCK_SYMBOLS``.
    deterministic_scores:
        Latest composite quant scores per symbol in [-1, 1].  Passed to
        ``debate.debate()`` so it can detect sign conflicts with the analyst.
        Missing symbols default to 0.0 (only low-conviction debate triggered).
    client:
        ``ClaudeClient`` to reuse across symbols.  One is created if absent.

    Returns
    -------
    list[SentimentRecord]
        One record per symbol that was successfully analysed and persisted.
    """
    if crypto_symbols is None:
        crypto_symbols = list(config.CRYPTO_SYMBOLS)
    if stock_symbols is None:
        stock_symbols = list(config.STOCK_SYMBOLS)
    if deterministic_scores is None:
        deterministic_scores = {}
    if client is None:
        client = ClaudeClient()

    symbols: list[tuple[str, str]] = [
        (s, "crypto") for s in crypto_symbols
    ] + [
        (s, "stocks") for s in stock_symbols
    ]

    records: list[SentimentRecord] = []

    for symbol, asset_class in symbols:
        try:
            det_score = deterministic_scores.get(symbol, 0.0)
            result = _analyse_symbol(symbol, asset_class, client, det_score)

            raw = json.dumps({
                "score": result.score,
                "rationale": result.rationale,
                "debated": result.debated,
                "debated_at": result.debated_at,
            })
            record = SentimentRecord(
                symbol=symbol,
                asset_class=asset_class,
                ts=result.debated_at,
                score=result.score,
                label=_score_to_label(result.score),
                confidence=round(abs(result.score), 4),
                debate_used=result.debated,
                raw_json=raw,
            )
            _persist(record)
            records.append(record)
            _log.info(
                "sentiment: %s → score=%.2f label=%s debate=%s",
                symbol, record.score, record.label, record.debate_used,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("sentiment runner failed for %s: %s", symbol, exc)

    return records
